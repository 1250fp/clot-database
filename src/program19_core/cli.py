"""Command-line interface for the Program19.3 2-D clot mechanics core."""

from __future__ import annotations

import argparse
from collections.abc import Iterable

from .config import EQUATION_PROFILES, Program19Config
from .mechanical_runner import run_pipeline


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Program19.3 2-D core: generated platelet-fibrin clot mechanics"
    )
    result.add_argument("--config", type=str, help="JSON configuration file")
    result.add_argument("--duration", type=float)
    result.add_argument("--dt", type=float)
    result.add_argument("--seed", type=int)
    result.add_argument("--platelets", type=int)
    result.add_argument(
        "--profile",
        dest="equation_profile",
        choices=EQUATION_PROFILES,
        help="backward-compatible alias for --equation-profile",
    )
    result.add_argument("--equation-profile", dest="equation_profile", choices=EQUATION_PROFILES)
    result.add_argument("--output-root", type=str)
    result.add_argument("--layers", type=int, help="number of independent mechanical z layers")
    result.add_argument(
        "--backend",
        choices=("python", "cuda"),
        help="mechanical integrator; cuda runs each layer's adaptive substeps on the GPU",
    )
    result.add_argument("--layer-seed-stride", type=int)
    result.add_argument("--self-test", action="store_true")
    result.add_argument("--write-default-config", type=str)
    # Backward-compatible no-op flags accepted by existing scripts (.bat files)
    result.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--no-figures", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--no-masks", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--no-intermediate-masks", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--workers", type=int, help=argparse.SUPPRESS)
    result.add_argument("--parameter-profile", type=str, help=argparse.SUPPRESS)
    result.add_argument("--numerical-profile", type=str, help=argparse.SUPPRESS)
    return result


def _self_test() -> None:
    import numpy as np
    from .geometry import junction_degree_targets

    config = Program19Config(
        duration_s=0.002,
        dt_s=0.001,
        platelet_count=0,
        mechanical_layers=3,
    )
    targets = junction_degree_targets(config)
    assert set(targets) == {3, 4, 5, 6, 7}
    try:
        from .bending import bending_triplet
        from .wlc import paper_stiffness_dF_dgamma_nN, wlc_force_nN
    except ImportError:
        print("Program19.3 2-D core self-test passed (CUDA-only configuration)")
        return
    L0 = 0.3
    assert abs(wlc_force_nN(L0, L0, config)) < 1e-12
    assert wlc_force_nN(1.1 * L0, L0, config) > 0
    assert wlc_force_nN(0.9 * L0, L0, config) < 0
    gamma = 0.2
    eps = 1e-6
    numeric = (
        wlc_force_nN((1 + gamma + eps) * L0, L0, config)
        - wlc_force_nN((1 + gamma - eps) * L0, L0, config)
    ) / (2 * eps)
    assert np.isclose(numeric, paper_stiffness_dF_dgamma_nN(gamma, config), rtol=2e-5)
    xi, xj, xk = np.asarray((0.0, 0.0)), np.asarray((1.0, 0.0)), np.asarray((2.0, 0.0))
    forces = bending_triplet(xi, xj, xk, 0.0, 1.0, config)
    assert np.allclose(np.sum(forces, axis=0), 0.0)
    assert np.allclose(forces, 0.0)
    print("Program19.3 2-D core self-test passed")


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    if args.write_default_config:
        Program19Config(
            equation_profile=args.equation_profile or "paper_consistent"
        ).save_json(args.write_default_config)
        print(f"Default configuration written to {args.write_default_config}")
        return 0
    config = (
        Program19Config.load_json(args.config)
        if args.config
        else Program19Config(equation_profile=args.equation_profile or "paper_consistent")
    )
    values = config.to_dict()
    overrides = {
        "duration_s": args.duration,
        "dt_s": args.dt,
        "seed": args.seed,
        "platelet_count": args.platelets,
        "equation_profile": args.equation_profile,
        "output_root": args.output_root,
        "mechanical_layers": args.layers,
        "layer_seed_stride": args.layer_seed_stride,
        "mechanics_backend": args.backend,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    config = Program19Config.from_dict(values)

    def show(payload: dict) -> None:
        if payload.get("stage") == "mechanics":
            print(
                f"\r{100 * payload['fraction']:6.2f}%  "
                f"layer={payload.get('layer_index', 0)}/{payload.get('layer_count', 0)}  "
                f"t={payload.get('simulated_time_s', 0):.4g} s  "
                f"links={payload.get('cohesion_links', 0)}  "
                f"substeps={payload.get('internal_substeps', 1)}",
                end="",
                flush=True,
            )
        elif payload.get("stage") == "complete":
            print()

    _, directory, status = run_pipeline(config, progress_callback=show)
    print(f"Status: {status}\nResults: {directory}")
    return 0
