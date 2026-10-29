"""30-minute clot replica library (m = 4, 8, 16 slabs | f0 = 0, 2.2, 4.4 nN | 10 replicas).

Every slab is an independent 2-D layer run with the trimmed CUDA core (no physics changes).
Per slab, one self-contained file  m{M}/f{f0}/rep{r}/slab{s}_seed{seed}.npz  holds (f0 folder: f2.2, f2.233):
  times_s (68)                       0..1800 every 30 s + 1, 2, 5, 10, 15, 20, 45 s
  nodes_um (F,N,2), platelets_um (F,P,2)   float64 positions at each snapshot
  filopodia_node_ids (F,P,A)         attached node per arm, -1 = free
  cohesion_count (F)                 links alive at each snapshot (links are only appended,
                                     so the first cohesion_count[f] rows of cohesion_* apply)
  cohesion_pairs / _L0_um / _birth_s all links with rest length and birth time
  backbone_pairs/_L0_um, bending_triples/_phi0_rad/_dual_L0_um, node_fiber_id_offsets/_flat,
  fixed_nodes, filopodia_arm_counts  topology
  config_json, core_commit, core_sha256, script_sha256, seed, M, slab, rep, f0, k_um, z_um
  state_*, cuda_noise_substep        full restart state (the final file is also a checkpoint)
While running, slab{s}_seed{seed}.ckpt.npz (same format, frames so far) is rewritten every
--ckpt-every simulated seconds. All writes go to *.tmp then os.replace (atomic).
Seeds: seed = SEED_BASE[M] + M*(rep-1) + (slab-1); the same seeds are used for every f0.
"""
from __future__ import annotations

import argparse, hashlib, json, os, socket, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np  # noqa: E402
from generate_clots import make_slab_config, _free_session, _bar, _fmt_time  # noqa: E402
from program19_core.cuda_mechanics import _session, integrate_interval_cuda, runtime  # noqa: E402
from program19_core.geometry import build_initial_state  # noqa: E402

VERSION = "library-1.0"
CORE_COMMIT = "a31ee69949799a97f6b86893a38b41e254e7b838"  # clot-database commit of src/program19_core
SEED_BASE = {4: 5000, 8: 5100, 16: 5200}  # m=4: 5000-5039, m=8: 5100-5179, m=16: 5200-5359
MS, FORCES, REPS, DURATION = (4, 8, 16), (0.0, 2.2, 4.4), 10, 1800.0
EXTRA_TIMES = (1, 2, 5, 10, 15, 20, 45)
DT = 0.001


def ftag(f0):
    """Folder/label for a force: 1 decimal when exact (old folders f0.0, f2.2, f4.4), else full value (f2.233)."""
    return f"{f0:.1f}" if abs(f0 - round(f0, 1)) < 1e-9 else f"{f0:g}"


def seed_of(M, rep, slab):
    base = SEED_BASE.get(M)
    if base is None:
        raise SystemExit(f"No seed block defined for m={M}")
    seed = base + M * (rep - 1) + (slab - 1)
    nxt = min([b for b in SEED_BASE.values() if b > base], default=None)
    if seed < 5000 or (nxt is not None and seed >= nxt):
        raise SystemExit(f"m={M} rep={rep} would leave its seed block ({seed})")
    return seed


def schedule(duration):
    t = set(np.arange(0.0, duration + 1e-9, 30.0).round(6)) | {float(x) for x in EXTRA_TIMES if x < duration}
    t.add(round(duration, 6))
    times = np.array(sorted(t), float)
    return times, np.rint(times / DT).astype(np.int64)


def sha256(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def provenance():
    core = HERE / "src" / "program19_core"
    files = [p for p in core.iterdir() if p.is_file() and p.suffix in (".py", ".cu", ".ptx", ".cubin")]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True,
                                text=True, timeout=10).stdout.strip() or CORE_COMMIT
    except Exception:
        commit = CORE_COMMIT
    return {"core_commit": commit, "core_sha256": sha256(files),
            "script_sha256": sha256([Path(__file__).resolve()]), "script_version": VERSION}


def atomic_savez(path: Path, **arrays):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_text(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8"); os.replace(tmp, path)


def update_catalog(path: Path, fn):
    """Read-modify-write catalog.json under a lock file, so parallel copies sharing one --out don't lose entries."""
    lock = path.with_name(path.name + ".lock")
    for _ in range(600):
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)); break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 60:
                    lock.unlink()  # stale lock from a killed process
            except OSError:
                pass
            time.sleep(0.1)
    try:
        cat = json.loads(path.read_text()) if path.exists() else {}
        fn(cat)
        atomic_text(path, json.dumps(cat, indent=1))
        return cat
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def filo_table(state, width):
    t = np.full((len(state.filopodia), width), -1, np.int32)
    for p, arms in enumerate(state.filopodia):
        for a, n in enumerate(arms):
            if n is not None:
                t[p, a] = n
    return t


def topology(state, cfg, M, rep, slab, prov):
    off, flat = [0], []
    for ids in state.node_fiber_ids:
        flat.extend(sorted(int(v) for v in ids)); off.append(len(flat))
    return dict(
        seed=np.int64(cfg.seed), M=np.int32(M), rep=np.int32(rep), slab=np.int32(slab),
        filopodia_f0_nN=np.float64(cfg.filopodia_f0_nN), k_um=np.float64(cfg.effective_thickness_um),
        z_um=np.float64((slab - 1 - 0.5 * (M - 1)) * cfg.layer_spacing_um),
        domain_size_um=np.float64(cfg.domain_size_um), platelet_count=np.int32(cfg.platelet_count),
        dt_s=np.float64(cfg.dt_s), duration_s=np.float64(cfg.duration_s),
        config_json=np.array(json.dumps(cfg.to_dict(), sort_keys=True, default=str)),
        **{k: np.array(v) for k, v in prov.items()},
        backbone_pairs=np.asarray(state.backbone_pairs, np.int32),
        backbone_L0_um=np.asarray(state.backbone_L0_um, np.float64),
        bending_triples=np.asarray(state.bending_triples, np.int32),
        bending_phi0_rad=np.asarray(state.bending_phi0_rad, np.float64),
        bending_dual_L0_um=np.asarray(state.bending_dual_L0_um, np.float64),
        node_fiber_id_offsets=np.asarray(off, np.int32), node_fiber_ids_flat=np.asarray(flat, np.int32),
        fixed_nodes=np.asarray(state.fixed_nodes, bool),
        filopodia_arm_counts=np.array([len(a) for a in state.filopodia], np.int16),
    )


def restore(state, ck):
    """Put a checkpoint's dynamic state back into a freshly built (same-seed) state."""
    if not np.array_equal(ck["backbone_pairs"], np.asarray(state.backbone_pairs, np.int32)):
        raise RuntimeError("checkpoint topology does not match the rebuilt geometry")
    state.nodes_um = ck["state_nodes_um"].copy()
    state.platelets_um = ck["state_platelets_um"].copy()
    state.filopodia, state.node_occupied_by = [], {}
    for p, (row, n_arms) in enumerate(zip(ck["state_filopodia"], ck["filopodia_arm_counts"])):
        arms = [None if int(x) < 0 else int(x) for x in row[:n_arms]]
        state.filopodia.append(arms)
        for a, n in enumerate(arms):
            if n is not None:
                state.node_occupied_by[n] = (p, a)
    pairs = [tuple(int(v) for v in r) for r in ck["cohesion_pairs"]]
    state.cohesion_pairs = pairs
    state.cohesion_L0_um = [float(x) for x in ck["cohesion_L0_um"]]
    state.cohesion_birth_time_s = [float(x) for x in ck["cohesion_birth_s"]]
    state.cohesion_pair_set = set(pairs)
    state.time_s = float(ck["state_time_s"])
    state.step = int(ck["state_step"])
    state.cuda_noise_substep = int(ck["cuda_noise_substep"])


def run_slab(cfg, path: Path, M, rep, slab, prov, ckpt_every, resume, stop_at, show):
    ckpt = path.with_name(path.stem + ".ckpt.npz")
    times, steps = schedule(cfg.duration_s)
    total = int(steps[-1])
    state = build_initial_state(cfg)
    width = max((len(a) for a in state.filopodia), default=1) or 1
    nodes, plts, filo, ncoh, wall, subs = [], [], [], [], [], []
    wall0 = 0.0
    if resume and ckpt.exists():
        with np.load(ckpt, allow_pickle=False) as d:
            ck = {k: d[k] for k in d.files}
        if not np.array_equal(ck["times_s"], times) or int(ck["seed"]) != cfg.seed:
            raise RuntimeError(f"{ckpt.name} was written with different settings")
        restore(state, ck)
        k = int(ck["frames_done"])
        nodes, plts, filo = list(ck["nodes_um"][:k]), list(ck["platelets_um"][:k]), list(ck["filopodia_node_ids"][:k])
        ncoh, wall, subs = list(ck["cohesion_count"][:k]), list(ck["frame_wall_s"][:k]), list(ck["frame_gpu_substeps"][:k])
        wall0 = float(ck["wall_s"])
        print(f"\n  resumed {ckpt.name} at t={state.time_s:.3f}s ({k}/{len(times)} frames)")
        if stop_at and state.step >= int(round(stop_at / DT)):
            return {"status": "stopped", "t_s": round(state.time_s, 3), "wall_s": round(wall0, 1)}
    topo = topology(state, cfg, M, rep, slab, prov)
    t0 = time.perf_counter() - wall0

    def snap():
        nodes.append(state.nodes_um.copy()); plts.append(state.platelets_um.copy())
        filo.append(filo_table(state, width)); ncoh.append(len(state.cohesion_pairs))
        wall.append(time.perf_counter() - t0); subs.append(int(getattr(state, "cuda_noise_substep", 0)))

    def save(target):
        k = len(nodes)
        atomic_savez(
            target, **topo, times_s=times, snapshot_steps=steps, frames_done=np.int32(k),
            nodes_um=np.asarray(nodes, np.float64).reshape(k, -1, 2),
            platelets_um=np.asarray(plts, np.float64).reshape(k, -1, 2),
            filopodia_node_ids=np.asarray(filo, np.int32).reshape(k, -1, width),
            cohesion_count=np.asarray(ncoh, np.int32),
            cohesion_pairs=np.asarray(state.cohesion_pairs, np.int32).reshape(-1, 2),
            cohesion_L0_um=np.asarray(state.cohesion_L0_um, np.float64),
            cohesion_birth_s=np.asarray(state.cohesion_birth_time_s, np.float64),
            state_nodes_um=np.asarray(state.nodes_um, np.float64),
            state_platelets_um=np.asarray(state.platelets_um, np.float64),
            state_filopodia=filo_table(state, width), state_time_s=np.float64(state.time_s),
            state_step=np.int64(state.step), cuda_noise_substep=np.int64(getattr(state, "cuda_noise_substep", 0)),
            frame_wall_s=np.asarray(wall, np.float64), frame_gpu_substeps=np.asarray(subs, np.int64),
            wall_s=np.float64(time.perf_counter() - t0))

    if state.step == 0:
        snap()
    ck_steps = int(round(ckpt_every / DT)) if ckpt_every > 0 else 0
    stop_step = int(round(stop_at / DT)) if stop_at else None
    _session(state, cfg)
    t_begin = state.time_s
    try:
        for i in range(state.step + 1, total + 1):
            integrate_interval_cuda(state, cfg, None, cfg.dt_s)
            if i == steps[len(nodes)]:
                snap()
            if i < total and ((ck_steps and i % ck_steps == 0) or i == stop_step):
                save(ckpt)
                if i == stop_step:
                    show(i * DT, t_begin, force=True)
                    print(f"\n  stopped at t={i*DT:g}s, kept {ckpt.name} for --resume")
                    return {"status": "stopped", "t_s": round(i * DT, 3), "seed": cfg.seed,
                            "links": len(state.cohesion_pairs), "wall_s": round(time.perf_counter() - t0, 1),
                            "ckpt_mb": round(ckpt.stat().st_size / 2**20, 2)}
            show(i * DT, t_begin)
    finally:
        _free_session(state)
    assert len(nodes) == len(times)
    save(path)
    if ckpt.exists():
        ckpt.unlink()
    return {"status": "done", "seed": cfg.seed, "nodes": len(state.nodes_um), "platelets": len(state.platelets_um),
            "links_final": len(state.cohesion_pairs), "frames": len(times),
            "wall_s": round(time.perf_counter() - t0, 1), "size_mb": round(path.stat().st_size / 2**20, 2)}


def merge(root: Path, others):
    """Fold parallel-copy folders into root (never overwrites) and rebuild catalog.json from the files."""
    for o in others:
        for f in sorted(o.glob("m*/f*/rep*/slab*.npz")):
            dst = root / f.relative_to(o)
            fin = dst.with_name(dst.name.replace(".ckpt", ""))
            if dst.exists() or (f.name.endswith(".ckpt.npz") and fin.exists()):
                print(f"  keep existing {dst.relative_to(root)}"); continue
            dst.parent.mkdir(parents=True, exist_ok=True); os.replace(f, dst)
    old = next((json.loads(p.read_text()) for p in [root / "catalog.json", *(o / "catalog.json" for o in others)]
                if p.exists()), {})
    cat = {k: v for k, v in old.items() if k not in ("slabs", "partial")}
    cat["slabs"], cat["partial"] = {}, {}
    for f in sorted(root.glob("m*/f*/rep*/slab*.npz")):
        with np.load(f, allow_pickle=False) as d:
            fin = not f.name.endswith(".ckpt.npz")
            info = {"status": "done" if fin else "stopped", "seed": int(d["seed"]), "M": int(d["M"]),
                    "f0_nN": float(d["filopodia_f0_nN"]), "rep": int(d["rep"]), "slab": int(d["slab"]),
                    "t_s": round(float(d["state_time_s"]), 3), "links": int(len(d["cohesion_pairs"])),
                    "wall_s": round(float(d["wall_s"]), 1), "size_mb": round(f.stat().st_size / 2**20, 2)}
        rel = str(f.relative_to(root)).replace("\\", "/").replace(".ckpt.npz", ".npz")
        cat["slabs" if fin else "partial"][rel] = info
    atomic_text(root / "catalog.json", json.dumps(cat, indent=1))
    print(f"merged: {len(cat['slabs'])} finished, {len(cat['partial'])} stopped slabs in {root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=r"D:\clot-data\library_30min")
    ap.add_argument("--m", type=int, nargs="+", default=list(MS))
    ap.add_argument("--f0", type=float, nargs="+", default=list(FORCES))
    ap.add_argument("--reps", type=int, nargs="+", default=list(range(1, REPS + 1)))
    ap.add_argument("--slabs", type=int, nargs="+", help="only these slab numbers (testing)")
    ap.add_argument("--duration", type=float, default=DURATION, help="override for tests (s)")
    ap.add_argument("--ckpt-every", type=float, default=300.0, help="checkpoint interval (simulated s)")
    ap.add_argument("--resume", action="store_true", help="continue *.ckpt.npz files (finished slabs are always skipped)")
    ap.add_argument("--tag", help="name for this copy when several run on one --out (progress_TAG.txt); "
                    "give each copy disjoint --reps/--f0 so no slab is shared")
    ap.add_argument("--merge", nargs="+", metavar="DIR", help="move slab files from other --out folders "
                    "(extra parallel copies) into --out and rebuild its catalog.json; runs nothing")
    ap.add_argument("--stop-at", type=float, help="run every selected slab only to this time, keep its checkpoint, "
                    "move on (continue later with --resume and no --stop-at)")
    a = ap.parse_args()
    root = Path(a.out); root.mkdir(parents=True, exist_ok=True)
    if a.merge:
        return merge(root, [Path(d) for d in a.merge])
    times, _ = schedule(a.duration)
    prov = provenance()
    settings = {"duration_s": a.duration, "dt_s": DT, "times_s": times.tolist(), "script_version": VERSION}
    cat_path = root / "catalog.json"
    prog_path = root / (f"progress_{a.tag}.txt" if a.tag else "progress.txt")

    def init(c):
        if not c:
            c.update(settings, seed_rule="seed = SEED_BASE[M] + M*(rep-1) + (slab-1), same seeds for every f0",
                     seed_base=SEED_BASE, **prov, slabs={})
    cat = update_catalog(cat_path, init)
    if any(cat.get(k) != v for k, v in settings.items()):
        raise SystemExit(f"{root} holds a library with different settings; use another --out")
    jobs = [(M, f0, r, s) for M in a.m for r in a.reps for f0 in a.f0 for s in range(1, M + 1)
            if not a.slabs or s in a.slabs]
    target = min(a.stop_at or a.duration, a.duration)
    sim_total = len(jobs) * target
    print(f"30-MIN LIBRARY | host={socket.gethostname()} | {len(jobs)} slabs, {len(times)} frames each -> {root}")
    print(f"core {prov['core_commit'][:7]} | script {prov['script_sha256'][:10]} | ckpt every {a.ckpt_every:g}s")
    runtime()
    t_start, done_sim, last = time.perf_counter(), 0.0, [0.0]
    for n, (M, f0, r, s) in enumerate(jobs, 1):
        seed = seed_of(M, r, s)
        rel = f"m{M}/f{ftag(f0)}/rep{r:02d}/slab{s:02d}_seed{seed}.npz"
        path = root / rel
        if path.exists():
            print(f"  [{n}/{len(jobs)}] {rel} already done"); sim_total -= target; continue
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = make_slab_config(M, seed=seed, duration_s=a.duration, frame_interval_s=30.0, f0_nN=f0)

        def show(t, t_begin, force=False, n=n, M=M, f0=f0, r=r, s=s, seed=seed):
            now = time.perf_counter()
            if now - last[0] < 2.0 and not force and t < a.duration:
                return
            last[0] = now
            el = now - t_start; f = (done_sim + t - t_begin) / max(sim_total, 1e-9)
            line = (f"[{n}/{len(jobs)}] m={M} f0={ftag(f0)} rep{r:02d} slab{s:02d} seed{seed} "
                    f"{t:7.1f}/{target:g}s | all [{_bar(f, 16)}] {100*f:5.1f}% | "
                    f"wall {_fmt_time(el)} ETA {_fmt_time(el / max(f, 1e-9) - el)}")
            print("\r  " + line[:160].ljust(160), end="", flush=True)
            try:
                atomic_text(prog_path, line + "\n")
            except OSError:
                pass

        info = run_slab(cfg, path, M, r, s, prov, a.ckpt_every, a.resume, a.stop_at, show)
        done_sim += target
        info.update(M=M, f0_nN=f0, rep=r, slab=s)

        def put(c, rel=rel, info=info):
            c.setdefault("partial", {}).pop(rel, None)
            c.setdefault("partial" if info["status"] == "stopped" else "slabs", {})[rel] = info
        update_catalog(cat_path, put)
    print("\nDone.")


if __name__ == "__main__":
    main()
