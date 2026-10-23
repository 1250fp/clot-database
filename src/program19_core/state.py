"""State containers and shared state helpers for the Program19.3 2-D core."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from .config import Program19Config

K_B_J_PER_K = 1.380649e-23
J_PER_NN_UM = 1.0e-15


@dataclass
class SimulationState:
    nodes_um: np.ndarray
    platelets_um: np.ndarray
    backbone_pairs: np.ndarray
    backbone_L0_um: np.ndarray
    bending_triples: np.ndarray
    bending_phi0_rad: np.ndarray
    bending_dual_L0_um: np.ndarray
    node_fiber_ids: list[frozenset[int]]
    incident_backbone: list[list[int]]
    original_adjacency: list[list[int]]
    fixed_nodes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    cohesion_pairs: list[tuple[int, int]] = field(default_factory=list)
    cohesion_L0_um: list[float] = field(default_factory=list)
    cohesion_birth_time_s: list[float] = field(default_factory=list)
    cohesion_pair_set: set[tuple[int, int]] = field(default_factory=set)
    filopodia: list[list[int | None]] = field(default_factory=list)
    node_occupied_by: dict[int, tuple[int, int]] = field(default_factory=dict)
    time_s: float = 0.0
    step: int = 0


@dataclass
class ForceDiagnostics:
    wlc_rms_nN: float = 0.0
    bending_rms_nN: float = 0.0
    contact_rms_nN: float = 0.0
    filopodia_rms_nN: float = 0.0
    max_backbone_strain: float = 0.0
    max_total_node_force_nN: float = 0.0
    max_total_platelet_force_nN: float = 0.0
    net_internal_force_nN: float = 0.0
    attached_filopodia: int = 0
    platelet_overlap_count: int = 0
    minimum_platelet_center_distance_um: float = float("inf")


@dataclass
class StepDiagnostics(ForceDiagnostics):
    new_cohesion_links: int = 0
    total_cohesion_links: int = 0
    maximum_cohesion_degree: int = 0
    max_node_displacement_um: float = 0.0
    max_platelet_displacement_um: float = 0.0
    internal_substeps: int = 1
    node_diffusion_um2_s: float = 0.0
    platelet_diffusion_um2_s: float = 0.0


def kbt_nN_um(config: Program19Config) -> float:
    """Thermal energy k_B*T in nN*um (1 nN*um = 1e-15 J); ~4.14e-6 nN*um at 300 K."""
    return K_B_J_PER_K * config.temperature_K / J_PER_NN_UM


def mobility_and_diffusion(
    config: Program19Config,
) -> tuple[float, float, float, float]:
    """Return node/platelet drag (nN*s/um) and D (um^2/s)."""
    if config.mobility_model == "effective_network":
        node_drag = config.effective_node_drag_nN_s_per_um
        platelet_drag = config.effective_platelet_drag_nN_s_per_um
    else:
        node_radius_m = config.fiber_radius_um * 1e-6
        platelet_radius_m = config.platelet_radius_um * 1e-6
        node_drag = 6.0 * math.pi * config.viscosity_Pa_s * node_radius_m * 1e3
        platelet_drag = 6.0 * math.pi * config.viscosity_Pa_s * platelet_radius_m * 1e3
    thermal = kbt_nN_um(config)
    return node_drag, platelet_drag, thermal / node_drag, thermal / platelet_drag


def cohesion_degrees(state: SimulationState) -> np.ndarray:
    """Number of cohesion links attached to each node."""
    degree = np.zeros(len(state.nodes_um), dtype=int)
    for a, b in state.cohesion_pairs:
        degree[a] += 1
        degree[b] += 1
    return degree


def rms_vectors(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.sum(values * values, axis=1))))


def max_vector_norm(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(values, axis=1)))


def state_arrays(state: SimulationState) -> dict[str, np.ndarray]:
    cohesion_pairs = np.asarray(state.cohesion_pairs, dtype=int).reshape((-1, 2))
    max_arms = max((len(arms) for arms in state.filopodia), default=0)
    filopodia = np.full((len(state.filopodia), max_arms), -1, dtype=int)
    for platelet_id, arms in enumerate(state.filopodia):
        for arm_id, node_id in enumerate(arms):
            if node_id is not None:
                filopodia[platelet_id, arm_id] = int(node_id)
    fiber_offsets = [0]
    fiber_values: list[int] = []
    for identities in state.node_fiber_ids:
        fiber_values.extend(sorted(int(value) for value in identities))
        fiber_offsets.append(len(fiber_values))
    return {
        "nodes_um": np.asarray(state.nodes_um),
        "platelets_um": np.asarray(state.platelets_um),
        "backbone_pairs": np.asarray(state.backbone_pairs),
        "backbone_reference_lengths_um": np.asarray(state.backbone_L0_um),
        "bending_triples": np.asarray(state.bending_triples),
        "bending_rest_angles_rad": np.asarray(state.bending_phi0_rad),
        "bending_dual_reference_lengths_um": np.asarray(state.bending_dual_L0_um),
        "cohesion_pairs": cohesion_pairs,
        "cohesion_reference_lengths_um": np.asarray(state.cohesion_L0_um, dtype=float),
        "cohesion_birth_times_s": np.asarray(state.cohesion_birth_time_s, dtype=float),
        "fixed_nodes": np.asarray(state.fixed_nodes, dtype=bool),
        "filopodia_node_ids": filopodia,
        "node_fiber_id_offsets": np.asarray(fiber_offsets, dtype=int),
        "node_fiber_ids_flat": np.asarray(fiber_values, dtype=int),
        "time_s": np.asarray(state.time_s),
        "step": np.asarray(state.step),
    }
