"""Reproducible multi-layer mechanical pipeline for the Program19.3 2-D core."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import Program19Config
from .geometry import build_initial_state
from .state import SimulationState, StepDiagnostics, state_arrays


ProgressCallback = Callable[[dict], None]


def _default_output_root() -> Path:
    return Path(__file__).resolve().parents[1] / "results"


def _new_run_directory(config: Program19Config) -> Path:
    root = Path(config.output_root).expanduser() if config.output_root else _default_output_root()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = (
        f"program19_3_{config.mechanical_layers}mechanical_layers_"
        f"{config.equation_profile}_seed{config.seed}_{timestamp}"
    )
    candidate = root / base
    suffix = 1
    while candidate.exists():
        candidate = root / f"{base}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("numpy", "scipy", "nvidia-cuda-nvrtc-cu12"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _free_cuda_session(state: SimulationState | None) -> None:
    if state is None:
        return
    session = getattr(state, "_cuda_session", None)
    if session is None:
        return
    for value in vars(session).values():
        pointer = getattr(value, "pointer", None)
        if pointer:
            try:
                session.runtime.free(pointer)
            except Exception:
                pass
            try:
                value.pointer = None
            except Exception:
                pass
    state._cuda_session = None


def _write_manifest(run_dir: Path, config: Program19Config) -> None:
    package_root = Path(__file__).resolve().parent
    source_paths = (
        list(package_root.glob("*.py"))
        + list(package_root.glob("*.cu"))
        + list(package_root.glob("*.ptx"))
        + list(package_root.glob("*.cubin"))
        + [
            package_root.parent / "program19.py",
            package_root.parent / "program19_2.py",
            package_root.parent / "program19_3.py",
        ]
    )
    source_paths = [path for path in source_paths if path.exists()]
    sources = {
        str(path.relative_to(package_root.parent)): _sha256(path)
        for path in sorted(source_paths)
    }
    manifest = {
        "program": "Program19.3",
        "version": "0.6.0",
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "equation_profile": config.equation_profile,
        "configuration": config.to_dict(),
        "units": {"length": "um", "force": "nN", "time": "s"},
        "layer_seed_rule": "layer_seed = base_seed + zero_based_layer_index*layer_seed_stride",
        "stage_semantics": {
            "mechanics": "N explicit 2-D mechanical realizations, one per z layer",
            "autoregression": "not used",
            "interlayer_force_coupling": "none; layers evolve independently",
            "intermediate_states": (
                "positions and topology saved at output_interval_s inside each layer folder"
            ),
        },
        "equations": {
            "strain": "gamma=(L-L0)/L0; u=gamma/C",
            "wlc": "F=N*kBT/P*[1/(4*(1-gamma/C)^2)-1/4+gamma/C]",
            "paper_stiffness": "dF/dgamma=N*kBT/P*[C^2/(2*(C-gamma)^3)+1/C]",
            "bending": "E=0.5*(EI/h0)*(phi-phi0)^2; F=-grad(E)",
        },
        "dependencies": _dependency_versions(),
        "platform": {"python": sys.version, "platform": platform.platform()},
        "source_sha256": sources,
    }
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    config.save_json(run_dir / "config.json")


def _filopodia_snapshot(state: SimulationState) -> np.ndarray:
    maximum = max((len(arms) for arms in state.filopodia), default=0)
    result = np.full((len(state.filopodia), maximum), -1, dtype=int)
    for platelet_id, arms in enumerate(state.filopodia):
        for arm_id, node_id in enumerate(arms):
            if node_id is not None:
                result[platelet_id, arm_id] = int(node_id)
    return result


def _flatten_cohesion_frames(
    frames: list[np.ndarray], reference_frames: list[np.ndarray], birth_frames: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets = [0]
    for frame in frames:
        offsets.append(offsets[-1] + len(frame))
    pairs = np.concatenate(frames, axis=0) if offsets[-1] else np.empty((0, 2), dtype=int)
    references = np.concatenate(reference_frames) if offsets[-1] else np.empty(0)
    births = np.concatenate(birth_frames) if offsets[-1] else np.empty(0)
    return np.asarray(offsets, dtype=int), pairs, references, births


def _progress(
    callback: ProgressCallback | None, stage: str, fraction: float, **values
) -> None:
    if callback:
        callback({"type": "progress", "stage": stage, "fraction": float(fraction), **values})


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_trajectory(
    path: Path,
    state: SimulationState,
    config: Program19Config,
    frame_times: list[float],
    node_frames: list[np.ndarray],
    platelet_frames: list[np.ndarray],
    cohesion_frames: list[np.ndarray],
    cohesion_reference_frames: list[np.ndarray],
    cohesion_birth_frames: list[np.ndarray],
    filopodia_frames: list[np.ndarray],
) -> None:
    trajectory = {
        "times_s": np.asarray(frame_times),
        "nodes_um": np.asarray(node_frames),
        "platelets_um": np.asarray(platelet_frames),
        "backbone_pairs": state.backbone_pairs,
        "fixed_nodes": np.asarray(state.fixed_nodes, dtype=bool),
        "axis_order_for_positions": np.asarray("time,object,xy"),
    }
    if config.trajectory_detail == "topology":
        offsets, pairs, references, births = _flatten_cohesion_frames(
            cohesion_frames, cohesion_reference_frames, cohesion_birth_frames
        )
        maximum_arms = max((frame.shape[1] for frame in filopodia_frames), default=0)
        padded_filopodia = np.full(
            (len(filopodia_frames), len(state.platelets_um), maximum_arms), -1, dtype=int
        )
        for frame_id, frame in enumerate(filopodia_frames):
            padded_filopodia[frame_id, :, : frame.shape[1]] = frame
        trajectory.update(
            cohesion_frame_offsets=offsets,
            cohesion_pairs_flat=pairs,
            cohesion_reference_lengths_um_flat=references,
            cohesion_birth_times_s_flat=births,
            filopodia_node_ids=padded_filopodia,
        )
    np.savez_compressed(path, **trajectory)


def _integrate_interval(
    state: SimulationState,
    config: Program19Config,
    rng: np.random.Generator,
    interval_s: float,
) -> StepDiagnostics:
    if config.mechanics_backend == "cuda":
        from .cuda_mechanics import integrate_interval_cuda

        return integrate_interval_cuda(state, config, rng, interval_s)
    from .integrator import integrate_interval

    return integrate_interval(state, config, rng, interval_s)


def _run_mechanical_layer(
    config: Program19Config,
    layer_index: int,
    layer_count: int,
    layer_dir: Path,
    progress_callback: ProgressCallback | None,
) -> tuple[SimulationState, str, dict]:
    """Run and persist one independent explicit mechanical layer."""
    layer_dir.mkdir(parents=True, exist_ok=False)
    config.save_json(layer_dir / "config.json")
    state = build_initial_state(config)
    np.savez_compressed(layer_dir / "initial_state.npz", **state_arrays(state))
    rng_mechanics = np.random.default_rng(np.random.SeedSequence([config.seed, 1001]))
    frame_times = [0.0]
    node_frames = [state.nodes_um.copy()]
    platelet_frames = [state.platelets_um.copy()]
    cohesion_frames = [np.asarray(state.cohesion_pairs, dtype=int).reshape((-1, 2))]
    cohesion_reference_frames = [np.asarray(state.cohesion_L0_um, dtype=float)]
    cohesion_birth_frames = [np.asarray(state.cohesion_birth_time_s, dtype=float)]
    filopodia_frames = [_filopodia_snapshot(state)]
    rows: list[dict] = []
    layer_wall_start = time.perf_counter()
    next_output = config.output_interval_s
    last_diagnostics = StepDiagnostics()
    status = "running"
    error: Exception | None = None
    try:
        while state.time_s < config.duration_s - 1e-15:
            interval = min(config.dt_s, config.duration_s - state.time_s)
            last_diagnostics = _integrate_interval(state, config, rng_mechanics, interval)
            should_store = state.time_s >= next_output - 1e-12 or math.isclose(
                state.time_s, config.duration_s
            )
            if should_store:
                rows.append(
                    {
                        "step": state.step,
                        "simulated_time_s": state.time_s,
                        "wall_elapsed_s": time.perf_counter() - layer_wall_start,
                        **asdict(last_diagnostics),
                    }
                )
                frame_times.append(state.time_s)
                node_frames.append(state.nodes_um.copy())
                platelet_frames.append(state.platelets_um.copy())
                cohesion_frames.append(
                    np.asarray(state.cohesion_pairs, dtype=int).reshape((-1, 2)).copy()
                )
                cohesion_reference_frames.append(
                    np.asarray(state.cohesion_L0_um, dtype=float).copy()
                )
                cohesion_birth_frames.append(
                    np.asarray(state.cohesion_birth_time_s, dtype=float).copy()
                )
                filopodia_frames.append(_filopodia_snapshot(state))
                while next_output <= state.time_s + 1e-12:
                    next_output += config.output_interval_s
            layer_fraction = state.time_s / config.duration_s
            _progress(
                progress_callback,
                "mechanics",
                (layer_index + layer_fraction) / layer_count,
                layer_index=layer_index + 1,
                layer_count=layer_count,
                layer_fraction=layer_fraction,
                simulated_time_s=state.time_s,
                duration_s=config.duration_s,
                cohesion_links=len(state.cohesion_pairs),
                internal_substeps=last_diagnostics.internal_substeps,
                layer_directory=str(layer_dir),
            )
        status = "completed"
    except Exception as exc:
        status = "failed"
        error = exc
    finally:
        _free_cuda_session(state)

    np.savez_compressed(layer_dir / "final_state.npz", **state_arrays(state))
    if config.save_trajectory:
        _save_trajectory(
            layer_dir / "trajectory.npz", state, config, frame_times, node_frames,
            platelet_frames, cohesion_frames, cohesion_reference_frames,
            cohesion_birth_frames, filopodia_frames,
        )
    _write_csv(layer_dir / "diagnostics.csv", rows)
    layer_summary = {
        "layer_index_one_based": layer_index + 1,
        "z_um": layer_index * config.layer_spacing_um,
        "seed": config.seed,
        "status": status,
        "simulated_time_s": state.time_s,
        "completed_outer_steps": state.step,
        "saved_intermediate_states": len(frame_times),
        "node_count": len(state.nodes_um),
        "platelet_count": len(state.platelets_um),
        "cohesion_link_count": len(state.cohesion_pairs),
        "wall_elapsed_s": time.perf_counter() - layer_wall_start,
        "worker_process_id": os.getpid(),
        "mechanics_backend": config.mechanics_backend,
        "error": str(error) if error else None,
    }
    with (layer_dir / "layer_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(layer_summary, handle, indent=2)
    if error:
        raise RuntimeError(f"Layer {layer_index + 1} failed: {error}") from error
    return state, status, layer_summary


def _load_completed_layer_summary(layer_dir: Path, config: Program19Config) -> dict | None:
    summary_path = layer_dir / "layer_summary.json"
    final_path = layer_dir / "final_state.npz"
    initial_path = layer_dir / "initial_state.npz"
    traj_path = layer_dir / "trajectory.npz"
    if not (summary_path.exists() and final_path.exists() and initial_path.exists()):
        return None
    if config.save_trajectory and not traj_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if summary.get("status") != "completed":
        return None
    if not math.isclose(float(summary.get("simulated_time_s", -1.0)), config.duration_s, rel_tol=0.0, abs_tol=1e-6):
        return None
    return summary


def run_pipeline(
    config: Program19Config,
    progress_callback: ProgressCallback | None = None,
    *,
    run_dir: Path | None = None,
    resume: bool = True,
    **_ignored,
) -> tuple[SimulationState | None, Path, str]:
    """Run every mechanical layer and write its outputs."""
    config.validate()
    if run_dir is None:
        run_dir = _new_run_directory(config)
    else:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(run_dir, config)
    layers_dir = run_dir / "mechanical_layers"
    layers_dir.mkdir(exist_ok=True)
    wall_start = time.perf_counter()
    status = "running"
    error: Exception | None = None
    last_state: SimulationState | None = None
    completed_layers = 0
    layer_rows: list[dict] = []
    try:
        for layer_index in range(config.mechanical_layers):
            layer_seed = config.seed + layer_index * config.layer_seed_stride
            layer_config = replace(config, seed=layer_seed)
            layer_dir = layers_dir / f"layer_{layer_index + 1:03d}_seed{layer_seed}"
            existing_summary = _load_completed_layer_summary(layer_dir, layer_config) if resume else None
            if existing_summary is not None:
                completed_layers += 1
                layer_rows.append({**existing_summary, "folder": str(layer_dir.relative_to(run_dir))})
                _write_csv(run_dir / "layer_index.csv", layer_rows)
                _progress(
                    progress_callback,
                    "layer_skipped_complete",
                    (layer_index + 1) / config.mechanical_layers,
                    layer_index=layer_index + 1,
                    layer_count=config.mechanical_layers,
                    seed=layer_seed,
                    layer_directory=str(layer_dir),
                )
                continue
            if layer_dir.exists():
                shutil.rmtree(layer_dir)
            state, _, summary = _run_mechanical_layer(
                layer_config, layer_index, config.mechanical_layers, layer_dir, progress_callback
            )
            last_state = state
            completed_layers += 1
            layer_rows.append({**summary, "folder": str(layer_dir.relative_to(run_dir))})
            _write_csv(run_dir / "layer_index.csv", layer_rows)
        status = "completed"
        if completed_layers == 0:
            raise RuntimeError("No mechanical layer reached a saveable state")
        run_metrics = {
            "mechanical_layers_requested": config.mechanical_layers,
            "mechanical_layers_completed": completed_layers,
            "mechanics_backend": config.mechanics_backend,
            "autoregression_used": False,
            "interlayer_force_coupling": False,
            "filopodia_f0_nN": config.filopodia_f0_nN,
        }
        with (run_dir / "run_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(run_metrics, handle, indent=2)
        _progress(progress_callback, "complete", 1.0, run_directory=str(run_dir))
    except Exception as exc:
        status = "failed"
        error = exc
    finally:
        total_layer_wall = sum(float(r.get("wall_elapsed_s", 0.0)) for r in layer_rows)
        with (run_dir / "status.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "status": status,
                    "requested_mechanical_layers": config.mechanical_layers,
                    "completed_or_partial_layers_saved": completed_layers,
                    "session_wall_elapsed_s": time.perf_counter() - wall_start,
                    "cumulative_layers_wall_elapsed_s": total_layer_wall,
                    "error": str(error) if error else None,
                },
                handle,
                indent=2,
            )
    if error:
        raise RuntimeError(f"{error} Partial results were saved in {run_dir}") from error
    return last_state, run_dir, status
