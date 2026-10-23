"""Validated configuration for the Program19.3 2-D clot mechanics core.

All mechanics use micrometres (um), nanonewtons (nN), and seconds (s).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any
import json
import math


EQUATION_PROFILES = ("paper_consistent", "codeocean_literal")
MECHANICS_BACKENDS = ("python", "cuda")
PHYSICS_PROFILES = EQUATION_PROFILES
MOBILITY_MODELS = ("effective_network", "free_stokes")
THERMAL_NOISE_MODELS = ("gaussian_independent", "codeocean_uniform_shared")
FILOPODIA_COUNT_MODELS = (
    "lognormal_clipped_round",
    "codeocean_lognormal_rejection",
    "fixed",
)
FILOPODIA_SELECTION_RULES = ("nearest_free", "codeocean_random")
FILOPODIA_FORCE_DISTRIBUTIONS = ("full_per_arm", "divide_by_attached")
FILOPODIA_HANDOVER_RULES = ("global_reacquire", "graph_neighbor")
TRAJECTORY_DETAILS = ("positions", "topology")
DEFAULT_DOMAIN_SIZE_UM = 48.0

# Fibrin junction densities N_V (um^-3) by node degree, from Michael & Pancaldi
# (2023) Code Ocean capsule, simulations/ input XML of sample 1: junction count
# of each degree divided by the volume of the R = 24 um sphere, 4/3*pi*24^3 =
# 57,906 um^3. Degrees 0-2 (10 of 1,000 junctions) are excluded.
CAPSULE_SAMPLE1_VOLUME_UM3 = 4.0 / 3.0 * math.pi * 24.0**3
CAPSULE_SAMPLE1_DEGREE_COUNTS = {3: 416, 4: 289, 5: 102, 6: 95, 7: 80}
CAPSULE_SAMPLE1_DEGREE_DENSITY_UM3 = {
    degree: count / CAPSULE_SAMPLE1_VOLUME_UM3
    for degree, count in CAPSULE_SAMPLE1_DEGREE_COUNTS.items()
}


@dataclass
class Program19Config:
    # Run controls
    duration_s: float = 1.0
    dt_s: float = 0.001
    output_interval_s: float = 0.10
    seed: int = 2035
    equation_profile: str = "paper_consistent"
    output_root: str = ""
    mechanics_backend: str = "python"
    save_trajectory: bool = True
    trajectory_detail: str = "topology"

    # Initial planar clot geometry (default R = 24 um -> domain 48 um)
    domain_size_um: float = DEFAULT_DOMAIN_SIZE_UM
    effective_thickness_um: float = 1.0
    node_degrees: tuple[int, ...] = (3, 4, 5, 6, 7)
    node_degree_density_um3: dict[int, float] = field(
        default_factory=lambda: dict(CAPSULE_SAMPLE1_DEGREE_DENSITY_UM3)
    )
    node_degree_counts: dict[int, int] = field(default_factory=dict)
    main_node_min_spacing_um: float = 1.0
    segment_target_um: float = 0.30
    platelet_count: int = 45
    platelet_near_node_fraction: float = 0.50

    # WLC: gamma=(L-L0)/L0 and u=gamma/C
    temperature_K: float = 300.0
    persistence_length_um: float = 0.001074033502990
    contour_multiplier_C: float = 2.351243606987286
    monomers_in_parallel_N: float = 1100.0
    gamma_max: float = 2.0

    # Bending and cohesion
    bending_EI_nN_um2: float = 0.002262593420685
    archive_bending_force_nN: float = 0.002262593420685
    cohesion_distance_um: float = 0.10
    cohesion_search_stride: int = 10
    cohesion_max_new_per_node_per_search: int = 4
    cohesion_max_total_neighbors: int = 4
    cohesion_exclude_same_fiber: bool = True
    cohesion_degree_includes_backbone: bool = False

    # Platelets and active Michael contact branches
    platelet_radius_um: float = 1.13
    platelet_core_fraction: float = 0.90
    fiber_radius_um: float = 0.05
    platelet_force_radius_um: float = 2.18
    platelet_fibrin_force_nN: float = 4.0
    platelet_platelet_force_parameter: float = 4.0
    filopodia_lognormal_mu: float = 1.85
    filopodia_lognormal_sigma: float = 0.27
    filopodia_min: int = 2
    filopodia_max: int = 13
    filopodia_count_model: str = "lognormal_clipped_round"
    filopodia_fixed_count: int = 3
    filopodia_selection_rule: str = "nearest_free"
    filopodia_force_distribution: str = "full_per_arm"
    filopodia_global_node_exclusivity: bool = True
    filopodia_handover_rule: str = "global_reacquire"
    filopodia_f0_nN: float = 2.233
    filopodia_a: float = 61.0
    filopodia_b: float = 3.582
    filopodia_c: float = 131.8

    # Overdamped motion. The effective values are interpreted as nN*s/um.
    mobility_model: str = "effective_network"
    viscosity_Pa_s: float = 0.004
    effective_node_drag_nN_s_per_um: float = 3.769911184308
    effective_platelet_drag_nN_s_per_um: float = 85.2
    brownian_motion: bool = True
    remove_global_translation: bool = True
    adaptive_substeps: bool = True
    thermal_noise_model: str = "gaussian_independent"
    adaptive_target_displacement_um: float = 0.01
    max_internal_substeps: int = 10_000
    abort_displacement_um: float = 1.0

    # Independent mechanical layers
    mechanical_layers: int = 50
    layer_spacing_um: float = 0.070
    layer_seed_stride: int = 1

    def __post_init__(self) -> None:
        if self.domain_size_um is None:
            self.domain_size_um = DEFAULT_DOMAIN_SIZE_UM

    def validate(self) -> None:
        positive = {
            "duration_s": self.duration_s,
            "dt_s": self.dt_s,
            "output_interval_s": self.output_interval_s,
            "domain_size_um": self.domain_size_um,
            "segment_target_um": self.segment_target_um,
            "temperature_K": self.temperature_K,
            "persistence_length_um": self.persistence_length_um,
            "contour_multiplier_C": self.contour_multiplier_C,
            "platelet_radius_um": self.platelet_radius_um,
            "platelet_force_radius_um": self.platelet_force_radius_um,
            "adaptive_target_displacement_um": self.adaptive_target_displacement_um,
            "mechanical_layers": self.mechanical_layers,
            "layer_spacing_um": self.layer_spacing_um,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive; received {value!r}")
        if self.equation_profile not in EQUATION_PROFILES:
            raise ValueError(f"equation_profile must be one of {EQUATION_PROFILES}")
        if self.mobility_model not in MOBILITY_MODELS:
            raise ValueError(f"mobility_model must be one of {MOBILITY_MODELS}")
        if self.thermal_noise_model not in THERMAL_NOISE_MODELS:
            raise ValueError(f"thermal_noise_model must be one of {THERMAL_NOISE_MODELS}")
        if self.filopodia_count_model not in FILOPODIA_COUNT_MODELS:
            raise ValueError(f"filopodia_count_model must be one of {FILOPODIA_COUNT_MODELS}")
        if self.filopodia_selection_rule not in FILOPODIA_SELECTION_RULES:
            raise ValueError(f"filopodia_selection_rule must be one of {FILOPODIA_SELECTION_RULES}")
        if self.filopodia_force_distribution not in FILOPODIA_FORCE_DISTRIBUTIONS:
            raise ValueError(
                f"filopodia_force_distribution must be one of {FILOPODIA_FORCE_DISTRIBUTIONS}"
            )
        if self.filopodia_handover_rule not in FILOPODIA_HANDOVER_RULES:
            raise ValueError(f"filopodia_handover_rule must be one of {FILOPODIA_HANDOVER_RULES}")
        if self.trajectory_detail not in TRAJECTORY_DETAILS:
            raise ValueError(f"trajectory_detail must be one of {TRAJECTORY_DETAILS}")
        if self.mechanics_backend not in MECHANICS_BACKENDS:
            raise ValueError(f"mechanics_backend must be one of {MECHANICS_BACKENDS}")
        if self.dt_s > self.duration_s:
            raise ValueError("dt_s cannot exceed duration_s")
        if not 0.0 < self.gamma_max < self.contour_multiplier_C:
            raise ValueError("gamma_max must satisfy 0 < gamma_max < C")
        if self.platelet_count < 0:
            raise ValueError("platelet_count cannot be negative")
        if not self.node_degrees or len(set(self.node_degrees)) != len(self.node_degrees):
            raise ValueError("node_degrees must list distinct degrees")
        for degree in self.node_degrees:
            if int(degree) != degree or degree < 1:
                raise ValueError(f"node degree {degree!r} must be a positive integer")
            if degree not in self.node_degree_counts and degree not in self.node_degree_density_um3:
                raise ValueError(f"degree {degree} has no density or explicit count")
        if any(float(value) < 0.0 for value in self.node_degree_density_um3.values()):
            raise ValueError("node_degree_density_um3 values cannot be negative")
        if any(int(value) < 0 for value in self.node_degree_counts.values()):
            raise ValueError("node_degree_counts values cannot be negative")
        if not 0.0 <= self.platelet_near_node_fraction <= 1.0:
            raise ValueError("platelet_near_node_fraction must be in [0,1]")
        if not 0.0 < self.platelet_core_fraction <= 1.0:
            raise ValueError("platelet_core_fraction must be in (0,1]")
        if self.platelet_force_radius_um <= self.platelet_radius_um + self.fiber_radius_um:
            raise ValueError("platelet_force_radius_um must exceed the fibrin contact shell")
        if self.cohesion_search_stride < 1:
            raise ValueError("cohesion_search_stride must be at least 1")
        if self.cohesion_max_new_per_node_per_search < 0:
            raise ValueError("cohesion_max_new_per_node_per_search cannot be negative")
        if self.cohesion_max_total_neighbors < 0:
            raise ValueError("cohesion_max_total_neighbors cannot be negative")
        if self.filopodia_fixed_count < 0:
            raise ValueError("filopodia_fixed_count cannot be negative")
        if self.max_internal_substeps < 1:
            raise ValueError("max_internal_substeps must be at least 1")
        if self.layer_seed_stride < 1:
            raise ValueError("layer_seed_stride must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Program19Config":
        values = dict(values)
        if "physics_profile" in values and "equation_profile" not in values:
            old = values.pop("physics_profile")
            values["equation_profile"] = (
                "codeocean_literal" if old == "archive_literal" else old
            )
        if values.get("domain_size_um") is None:
            values.pop("domain_size_um", None)
        if "node_degrees" in values:
            values["node_degrees"] = tuple(int(d) for d in values["node_degrees"])
        for name, cast in (("node_degree_density_um3", float), ("node_degree_counts", int)):
            if name in values:
                values[name] = {int(d): cast(v) for d, v in values[name].items()}
        known = {item.name for item in fields(cls)}
        filtered = {k: v for k, v in values.items() if k in known}
        result = cls(**filtered)
        result.validate()
        return result

    @classmethod
    def load_json(cls, path: str | Path) -> "Program19Config":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save_json(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
