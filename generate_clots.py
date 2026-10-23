"""Minimal CUDA Clot Database Generator (M = 9, 6, 3 | 300 s | 30 s frames).

Generates complete multi-slab 2-D clots using the paper's non-overlapping seed
scheme (each slab has a unique seed, and each whole clot has a unique block of
M consecutive seeds). Each slab is saved as a single self-contained compressed
NPZ file containing the full 30 s trajectory (t = 0, 30, ..., 300 s), network
topology, cohesive links, and filopodia states.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
from program19_core.config import CAPSULE_SAMPLE1_VOLUME_UM3, Program19Config  # noqa: E402
from program19_core.cuda_mechanics import (  # noqa: E402
    _Buffer,
    _session,
    integrate_interval_cuda,
    runtime,
)
from program19_core.geometry import build_initial_state, junction_degree_targets  # noqa: E402
from program19_core.state import SimulationState  # noqa: E402

R_UM = 24.0
LZ_UM = 32.0
MEAN_CALIPER_UM = 2.2
PLATELET_NV_UM3 = 45.0 / CAPSULE_SAMPLE1_VOLUME_UM3

# Canonical non-overlapping seed blocks starting at 4000 for Clot 1 of M = 9, 6, 3
DEFAULT_SEED_BLOCKS: dict[tuple[int, int], int] = {
    (9, 1): 4000,  # seeds 4000..4008 (9 slabs)
    (6, 1): 4009,  # seeds 4009..4014 (6 slabs)
    (3, 1): 4015,  # seeds 4015..4017 (3 slabs)
}


def make_slab_config(
    M: int,
    *,
    seed: int,
    duration_s: float = 300.0,
    frame_interval_s: float = 30.0,
    dt_s: float = 0.001,
    f0_nN: float = 2.223,
) -> Program19Config:
    k_um = LZ_UM / float(M)
    area_um2 = math.pi * R_UM**2
    platelets = int(round((MEAN_CALIPER_UM + k_um) * PLATELET_NV_UM3 * area_um2))
    return Program19Config(
        duration_s=float(duration_s),
        dt_s=float(dt_s),
        output_interval_s=float(frame_interval_s),
        seed=int(seed),
        equation_profile="paper_consistent",
        mechanics_backend="cuda",
        save_trajectory=True,
        trajectory_detail="topology",
        domain_size_um=2.0 * R_UM,
        effective_thickness_um=k_um,
        platelet_count=platelets,
        filopodia_f0_nN=float(f0_nN),
        mechanical_layers=int(M),
        layer_spacing_um=k_um,
    )


def _filopodia_snapshot(state: SimulationState) -> np.ndarray:
    maximum = max((len(arms) for arms in state.filopodia), default=0)
    result = np.full((len(state.filopodia), maximum), -1, dtype=np.int32)
    for platelet_id, arms in enumerate(state.filopodia):
        for arm_id, node_id in enumerate(arms):
            if node_id is not None:
                result[platelet_id, arm_id] = int(node_id)
    return result


def _flatten_cohesion_frames(
    frames: list[np.ndarray],
    reference_frames: list[np.ndarray],
    birth_frames: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets = [0]
    for frame in frames:
        offsets.append(offsets[-1] + len(frame))
    pairs = np.concatenate(frames, axis=0) if offsets[-1] else np.empty((0, 2), dtype=np.int32)
    references = np.concatenate(reference_frames) if offsets[-1] else np.empty(0, dtype=np.float64)
    births = np.concatenate(birth_frames) if offsets[-1] else np.empty(0, dtype=np.float64)
    return (
        np.asarray(offsets, dtype=np.int32),
        np.asarray(pairs, dtype=np.int32),
        np.asarray(references, dtype=np.float64),
        np.asarray(births, dtype=np.float64),
    )


def _free_session(state: SimulationState) -> None:
    session = getattr(state, "_cuda_session", None)
    if session is None:
        return
    for value in vars(session).values():
        if isinstance(value, _Buffer) and getattr(value, "pointer", None):
            try:
                session.runtime.free(value.pointer)
            except Exception:
                pass
            value.pointer = None
    state._cuda_session = None


def _bar(fraction: float, width: int = 20) -> str:
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return "#" * filled + "-" * (width - filled)


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:02d}m{s:02d}s"


def resolve_seed_block(db_dir: Path, M: int, clot_index: int) -> int:
    reg_path = db_dir / "catalog.json"
    allocations: dict[str, int] = {}
    next_seed = 4018
    if reg_path.exists():
        try:
            data = json.loads(reg_path.read_text(encoding="utf-8"))
            allocations = {k: int(v) for k, v in data.get("seed_blocks", {}).items()}
            next_seed = int(data.get("next_available_seed", 4018))
        except Exception:
            pass
    key = f"M{M:02d}_clot{clot_index:02d}"
    if key in allocations:
        return allocations[key]
    if (M, clot_index) in DEFAULT_SEED_BLOCKS:
        return DEFAULT_SEED_BLOCKS[(M, clot_index)]
    assigned = max(next_seed, max(allocations.values(), default=4000))
    return assigned


def rebuild_catalog(db_dir: Path) -> None:
    clots = []
    seed_blocks: dict[str, int] = {
        f"M{m:02d}_clot{c:02d}": s for (m, c), s in DEFAULT_SEED_BLOCKS.items()
    }
    max_seed_used = 4017
    for clot_json in sorted(db_dir.glob("M*_clot*/clot.json")):
        try:
            meta = json.loads(clot_json.read_text(encoding="utf-8"))
            clots.append(meta)
            key = f"M{int(meta['M']):02d}_clot{int(meta['clot_index']):02d}"
            seed_blocks[key] = int(meta["seed_start"])
            max_seed_used = max(max_seed_used, int(meta["seed_end"]))
        except Exception:
            pass
    catalog = {
        "database": "2-D Layered Platelet-Fibrin Clot Database",
        "updated_local": dt.datetime.now().isoformat(timespec="seconds"),
        "domain_size_um": 2.0 * R_UM,
        "Lz_um": LZ_UM,
        "monomers_in_parallel_N": 1100.0,
        "next_available_seed": max_seed_used + 1,
        "seed_blocks": seed_blocks,
        "clots": clots,
    }
    (db_dir / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")


def run_slab(
    config: Program19Config,
    *,
    M: int,
    clot_index: int,
    slab_index: int,
    slab_path: Path,
    batch_done: int,
    batch_total: int,
) -> dict:
    z_um = (slab_index - 0.5 * (M - 1)) * config.layer_spacing_um
    t0 = time.perf_counter()
    state = build_initial_state(config)
    _session(state, config)

    rng_unused = np.random.default_rng(np.random.SeedSequence([config.seed, 1001]))
    frame_times = [0.0]
    node_frames = [state.nodes_um.copy()]
    platelet_frames = [state.platelets_um.copy()]
    cohesion_frames = [np.asarray(state.cohesion_pairs, dtype=np.int32).reshape((-1, 2)).copy()]
    cohesion_ref_frames = [np.asarray(state.cohesion_L0_um, dtype=np.float64).copy()]
    cohesion_birth_frames = [np.asarray(state.cohesion_birth_time_s, dtype=np.float64).copy()]
    filopodia_frames = [_filopodia_snapshot(state)]
    frame_wall_s = [0.0]
    frame_substeps = [0]

    next_output = config.output_interval_s
    last_print = time.perf_counter()
    gpu_start = time.perf_counter()

    try:
        while state.time_s < config.duration_s - 1e-15:
            interval = min(config.dt_s, config.duration_s - state.time_s)
            integrate_interval_cuda(state, config, rng_unused, interval)
            now = time.perf_counter()
            elapsed = now - gpu_start

            should_store = state.time_s >= next_output - 1e-12 or math.isclose(
                state.time_s, config.duration_s
            )
            if should_store:
                substeps_now = int(getattr(state, "cuda_noise_substep", state.step))
                frame_times.append(round(float(state.time_s), 6))
                node_frames.append(state.nodes_um.copy())
                platelet_frames.append(state.platelets_um.copy())
                cohesion_frames.append(
                    np.asarray(state.cohesion_pairs, dtype=np.int32).reshape((-1, 2)).copy()
                )
                cohesion_ref_frames.append(
                    np.asarray(state.cohesion_L0_um, dtype=np.float64).copy()
                )
                cohesion_birth_frames.append(
                    np.asarray(state.cohesion_birth_time_s, dtype=np.float64).copy()
                )
                filopodia_frames.append(_filopodia_snapshot(state))
                frame_wall_s.append(round(elapsed, 4))
                frame_substeps.append(substeps_now)
                while next_output <= state.time_s + 1e-12:
                    next_output += config.output_interval_s

            if now - last_print >= 2.0 or should_store:
                last_print = now
                frac = state.time_s / config.duration_s
                rate = state.step / max(elapsed, 1e-9)
                eta = (config.duration_s - state.time_s) / max(
                    state.time_s / max(elapsed, 1e-9), 1e-9
                )
                prog_line = (
                    f"[{batch_done + 1:2d}/{batch_total:2d}] M={M} Clot {clot_index} "
                    f"Slab {slab_index + 1}/{M} (seed {config.seed}) "
                    f"[{_bar(frac, 18)}] {100 * frac:5.1f}% | "
                    f"{state.time_s:5.0f}/{config.duration_s:.0f}s | "
                    f"wall {_fmt_time(elapsed)} (ETA {_fmt_time(eta)}) | "
                    f"{rate:5.0f} st/s | links {len(state.cohesion_pairs):4d}"
                )
                print("\r  " + prog_line, end="", flush=True)
                try:
                    (slab_path.parent.parent / "progress.txt").write_text(prog_line + "\n", encoding="utf-8")
                except OSError:
                    pass
        print()
    finally:
        total_substeps = int(getattr(state, "cuda_noise_substep", state.step))
        _free_session(state)

    offsets, pairs_flat, refs_flat, births_flat = _flatten_cohesion_frames(
        cohesion_frames, cohesion_ref_frames, cohesion_birth_frames
    )
    max_arms = max((f.shape[1] for f in filopodia_frames), default=0)
    padded_filo = np.full(
        (len(filopodia_frames), len(state.platelets_um), max_arms), -1, dtype=np.int32
    )
    for idx, frame in enumerate(filopodia_frames):
        padded_filo[idx, :, : frame.shape[1]] = frame

    fiber_offsets = [0]
    fiber_values: list[int] = []
    for identities in state.node_fiber_ids:
        fiber_values.extend(sorted(int(v) for v in identities))
        fiber_offsets.append(len(fiber_values))

    tmp_path = slab_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        M=np.int32(M),
        clot_index=np.int32(clot_index),
        slab_index_one_based=np.int32(slab_index + 1),
        seed=np.int32(config.seed),
        k_um=np.float64(config.effective_thickness_um),
        z_um=np.float64(z_um),
        domain_size_um=np.float64(config.domain_size_um),
        filopodia_f0_nN=np.float64(config.filopodia_f0_nN),
        times_s=np.asarray(frame_times, dtype=np.float64),
        nodes_um=np.asarray(node_frames, dtype=np.float64),
        platelets_um=np.asarray(platelet_frames, dtype=np.float64),
        backbone_pairs=np.asarray(state.backbone_pairs, dtype=np.int32),
        backbone_L0_um=np.asarray(state.backbone_L0_um, dtype=np.float64),
        node_fiber_id_offsets=np.asarray(fiber_offsets, dtype=np.int32),
        node_fiber_ids_flat=np.asarray(fiber_values, dtype=np.int32),
        fixed_nodes=np.asarray(state.fixed_nodes, dtype=bool),
        bending_triples=np.asarray(state.bending_triples, dtype=np.int32),
        bending_phi0_rad=np.asarray(state.bending_phi0_rad, dtype=np.float64),
        bending_dual_L0_um=np.asarray(state.bending_dual_L0_um, dtype=np.float64),
        cohesion_frame_offsets=offsets,
        cohesion_pairs_flat=pairs_flat,
        cohesion_L0_um_flat=refs_flat,
        cohesion_birth_time_s_flat=births_flat,
        final_cohesion_pairs=np.asarray(state.cohesion_pairs, dtype=np.int32).reshape((-1, 2)),
        final_cohesion_L0_um=np.asarray(state.cohesion_L0_um, dtype=np.float64),
        filopodia_node_ids=padded_filo,
        frame_wall_elapsed_s=np.asarray(frame_wall_s, dtype=np.float64),
        frame_gpu_substeps=np.asarray(frame_substeps, dtype=np.int64),
    )
    tmp_path.replace(slab_path)
    wall_s = time.perf_counter() - t0
    return {
        "slab_index": slab_index + 1,
        "seed": int(config.seed),
        "z_um": round(float(z_um), 4),
        "file": slab_path.name,
        "size_kb": round(slab_path.stat().st_size / 1024.0, 1),
        "nodes": int(len(state.nodes_um)),
        "segments": int(len(state.backbone_pairs)),
        "platelets": int(len(state.platelets_um)),
        "final_cohesion_links": int(len(state.cohesion_pairs)),
        "frames_saved": len(frame_times),
        "gpu_substeps": total_substeps,
        "wall_s": round(wall_s, 2),
    }


def load_existing_slab_info(slab_path: Path, slab_index: int, z_um: float) -> dict | None:
    if not slab_path.exists():
        return None
    try:
        with np.load(slab_path, allow_pickle=False) as d:
            return {
                "slab_index": slab_index + 1,
                "seed": int(d["seed"]),
                "z_um": round(float(z_um), 4),
                "file": slab_path.name,
                "size_kb": round(slab_path.stat().st_size / 1024.0, 1),
                "nodes": int(d["nodes_um"].shape[1]),
                "segments": int(len(d["backbone_pairs"])),
                "platelets": int(d["platelets_um"].shape[1]),
                "final_cohesion_links": int(len(d["final_cohesion_pairs"])),
                "frames_saved": int(len(d["times_s"])),
                "gpu_substeps": int(d["frame_gpu_substeps"][-1]),
                "wall_s": round(float(d["frame_wall_elapsed_s"][-1]), 2),
            }
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal CUDA Clot Database Generator")
    parser.add_argument("--M", type=int, nargs="+", default=[9, 6, 3], help="Slab counts M (default: 9 6 3)")
    parser.add_argument("--clot-index", type=int, default=1, help="Clot replica index (default: 1)")
    parser.add_argument("--duration", type=float, default=300.0, help="Simulated time in s (default: 300)")
    parser.add_argument("--interval", type=float, default=30.0, help="Trajectory frame interval in s (default: 30)")
    parser.add_argument("--f0", type=float, default=2.223, help="Filopodial baseline force in nN (default: 2.223)")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    db_dir = root_dir / "database"
    db_dir.mkdir(parents=True, exist_ok=True)

    batch_total = sum(int(m) for m in args.M)
    print("=" * 86)
    print(f"CLOT DATABASE GENERATOR | host={socket.gethostname()} | M={args.M} ({batch_total} slabs)")
    print(f"duration={args.duration:g}s | trajectory every {args.interval:g}s | f0={args.f0:g} nN | N=1100")
    print("=" * 86)

    t_init = time.perf_counter()
    runtime()
    print(f"CUDA kernel ready in {time.perf_counter() - t_init:.2f}s\n")

    batch_done = 0
    for M in args.M:
        seed_start = resolve_seed_block(db_dir, M, args.clot_index)
        seed_end = seed_start + M - 1
        clot_dir = db_dir / f"M{M:02d}_clot{args.clot_index:02d}_seeds{seed_start}_{seed_end}"
        clot_dir.mkdir(parents=True, exist_ok=True)

        base_cfg = make_slab_config(
            M,
            seed=seed_start,
            duration_s=args.duration,
            frame_interval_s=args.interval,
            f0_nN=args.f0,
        )
        targets = junction_degree_targets(base_cfg)
        print(
            f"--- Clot M={M} (clot #{args.clot_index}) | k={base_cfg.effective_thickness_um:.2f} um | "
            f"platelets={base_cfg.platelet_count}/slab | junctions={sum(targets.values())}/slab | "
            f"seeds={seed_start}..{seed_end} ---"
        )

        slab_records = []
        for slab_idx in range(M):
            slab_seed = seed_start + slab_idx
            z_um = (slab_idx - 0.5 * (M - 1)) * base_cfg.layer_spacing_um
            slab_path = clot_dir / f"slab_{slab_idx + 1:02d}_seed{slab_seed}.npz"
            existing = load_existing_slab_info(slab_path, slab_idx, z_um)
            if existing is not None:
                print(
                    f"  [{batch_done + 1:2d}/{batch_total:2d}] M={M} Clot {args.clot_index} "
                    f"Slab {slab_idx + 1}/{M} (seed {slab_seed}) -> ALREADY COMPLETE ({existing['size_kb']} KB)"
                )
                slab_records.append(existing)
                batch_done += 1
                continue

            slab_cfg = make_slab_config(
                M,
                seed=slab_seed,
                duration_s=args.duration,
                frame_interval_s=args.interval,
                f0_nN=args.f0,
            )
            info = run_slab(
                slab_cfg,
                M=M,
                clot_index=args.clot_index,
                slab_index=slab_idx,
                slab_path=slab_path,
                batch_done=batch_done,
                batch_total=batch_total,
            )
            slab_records.append(info)
            batch_done += 1

            clot_meta = {
                "clot_id": clot_dir.name,
                "M": int(M),
                "clot_index": int(args.clot_index),
                "status": "completed" if len(slab_records) == M else "partial",
                "slabs_completed": len(slab_records),
                "k_um": round(float(base_cfg.effective_thickness_um), 6),
                "Lz_um": LZ_UM,
                "domain_size_um": float(base_cfg.domain_size_um),
                "platelet_count_per_slab": int(base_cfg.platelet_count),
                "junction_degree_targets": {str(k): int(v) for k, v in targets.items()},
                "filopodia_f0_nN": float(args.f0),
                "monomers_in_parallel_N": float(base_cfg.monomers_in_parallel_N),
                "duration_s": float(args.duration),
                "frame_interval_s": float(args.interval),
                "seed_start": int(seed_start),
                "seed_end": int(seed_end),
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "total_wall_s": round(sum(r["wall_s"] for r in slab_records), 2),
                "total_size_kb": round(sum(r["size_kb"] for r in slab_records), 1),
                "slabs": slab_records,
            }
            (clot_dir / "clot.json").write_text(json.dumps(clot_meta, indent=2), encoding="utf-8")
            rebuild_catalog(db_dir)

        clot_meta = {
            "clot_id": clot_dir.name,
            "M": int(M),
            "clot_index": int(args.clot_index),
            "status": "completed",
            "slabs_completed": len(slab_records),
            "k_um": round(float(base_cfg.effective_thickness_um), 6),
            "Lz_um": LZ_UM,
            "domain_size_um": float(base_cfg.domain_size_um),
            "platelet_count_per_slab": int(base_cfg.platelet_count),
            "junction_degree_targets": {str(k): int(v) for k, v in targets.items()},
            "filopodia_f0_nN": float(args.f0),
            "monomers_in_parallel_N": float(base_cfg.monomers_in_parallel_N),
            "duration_s": float(args.duration),
            "frame_interval_s": float(args.interval),
            "seed_start": int(seed_start),
            "seed_end": int(seed_end),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "total_wall_s": round(sum(r["wall_s"] for r in slab_records), 2),
            "total_size_kb": round(sum(r["size_kb"] for r in slab_records), 1),
            "slabs": slab_records,
        }
        (clot_dir / "clot.json").write_text(json.dumps(clot_meta, indent=2), encoding="utf-8")
        rebuild_catalog(db_dir)
        print(
            f"=== Completed Clot M={M} ({M}/{M} slabs, {clot_meta['total_size_kb']/1024:.2f} MB) ===\n"
        )

    print("All requested clots generated and indexed in database/catalog.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
