"""Deterministic construction of the explicit 2-D clot geometry."""

from __future__ import annotations

import math
import numpy as np

from .config import Program19Config
from .state import SimulationState


def _sample_filopodia_count(
    rng: np.random.Generator, config: Program19Config
) -> int:
    if config.filopodia_count_model == "fixed":
        return int(config.filopodia_fixed_count)
    if config.filopodia_count_model == "lognormal_clipped_round":
        return int(
            np.clip(
                round(
                    rng.lognormal(
                        config.filopodia_lognormal_mu,
                        config.filopodia_lognormal_sigma,
                    )
                ),
                config.filopodia_min,
                config.filopodia_max,
            )
        )
    # CodeOcean's PDF option rejects draws outside [1,max+1] and then truncates.
    # The audited thesis profile instead uses the same bounds with a finite,
    # defensive retry limit.
    for _ in range(100_000):
        number = float(
            rng.lognormal(
                config.filopodia_lognormal_mu, config.filopodia_lognormal_sigma
            )
        )
        if config.filopodia_min <= number < config.filopodia_max + 1:
            return int(number)
    raise RuntimeError("Could not sample a bounded filopodia count")


def _lhs_1d(rng: np.random.Generator, n: int) -> np.ndarray:
    if n <= 0:
        return np.empty(0)
    values = (np.arange(n, dtype=float) + rng.random(n)) / n
    rng.shuffle(values)
    return values


def _disk_points(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 2))
    r = radius * np.sqrt(_lhs_1d(rng, n))
    theta = 2.0 * np.pi * _lhs_1d(rng, n)
    return np.column_stack((r * np.cos(theta), r * np.sin(theta)))


def _random_disk_point(rng: np.random.Generator, radius: float) -> np.ndarray:
    r = radius * math.sqrt(float(rng.random()))
    theta = 2.0 * math.pi * float(rng.random())
    return np.asarray((r * math.cos(theta), r * math.sin(theta)))


def _far_enough(candidate: np.ndarray, points: list[np.ndarray], spacing: float) -> bool:
    if not points or spacing <= 0.0:
        return True
    existing = np.asarray(points)
    return bool(np.all(np.sum((existing - candidate) ** 2, axis=1) >= spacing**2))


def _place_platelets(
    rng: np.random.Generator, count: int, clot_radius: float, platelet_radius: float
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2))
    usable = clot_radius - platelet_radius - 0.05
    spacing = 2.0 * platelet_radius + 0.30
    points: list[np.ndarray] = []
    for candidate in _disk_points(rng, max(16 * count, count), usable):
        if _far_enough(candidate, points, spacing):
            points.append(candidate)
            if len(points) == count:
                break
    attempts = 0
    while len(points) < count and attempts < 300_000:
        attempts += 1
        relaxation = 0.94 ** (attempts // 75_000)
        candidate = _random_disk_point(rng, usable)
        if _far_enough(candidate, points, spacing * relaxation):
            points.append(candidate)
    if len(points) != count:
        raise RuntimeError(f"Could place only {len(points)} of {count} platelets")
    return np.asarray(points)


def _distance_point_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    delta = b - a
    length2 = float(np.dot(delta, delta))
    if length2 <= 1e-18:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, delta) / length2, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * delta)))


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab, ac = b - a, c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def _crosses(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    eps = 1e-12
    return (
        _orientation(a, b, c) * _orientation(a, b, d) < -eps
        and _orientation(c, d, a) * _orientation(c, d, b) < -eps
    )


def _place_junctions(
    rng: np.random.Generator,
    count: int,
    radius: float,
    spacing: float,
    platelets: np.ndarray,
    exclusion: float,
    near_fraction: float,
    near_radius: float,
) -> np.ndarray:
    points: list[np.ndarray] = []

    def acceptable(candidate: np.ndarray, local_spacing: float) -> bool:
        if np.linalg.norm(candidate) > radius:
            return False
        if not _far_enough(candidate, points, local_spacing):
            return False
        return not len(platelets) or bool(
            np.all(np.linalg.norm(platelets - candidate, axis=1) >= exclusion)
        )

    near_count = int(round(count * near_fraction)) if len(platelets) else 0
    attempts = 0
    while len(points) < near_count and attempts < 200_000:
        attempts += 1
        center = platelets[int(rng.integers(0, len(platelets)))]
        candidate = center + _random_disk_point(rng, near_radius)
        if acceptable(candidate, spacing):
            points.append(candidate)
    for candidate in _disk_points(rng, max(12 * count, count), radius):
        if len(points) == count:
            break
        if acceptable(candidate, spacing):
            points.append(candidate)
    attempts = 0
    while len(points) < count and attempts < 400_000:
        attempts += 1
        relaxation = 0.94 ** (attempts // 100_000)
        candidate = _random_disk_point(rng, radius)
        if acceptable(candidate, spacing * relaxation):
            points.append(candidate)
    if len(points) != count:
        raise RuntimeError(f"Could place only {len(points)} of {count} fibrin junctions")
    return np.asarray(points)


def _pair_planar_stubs(
    rng: np.random.Generator,
    centers: np.ndarray,
    degrees: np.ndarray,
    platelets: np.ndarray,
    platelet_exclusion: float,
) -> list[tuple[int, int]]:
    remaining = degrees.astype(int).copy()
    distance = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    edges: list[tuple[int, int]] = []
    edge_set: set[tuple[int, int]] = set()
    for _ in range(200_000):
        available = np.flatnonzero(remaining > 0)
        if len(available) < 2:
            break
        maximum = int(np.max(remaining[available]))
        a = int(rng.choice(available[remaining[available] == maximum]))
        candidates: list[tuple[float, int]] = []
        for raw_b in available:
            b = int(raw_b)
            if b == a:
                continue
            key = (min(a, b), max(a, b))
            if key in edge_set:
                continue
            if len(platelets) and any(
                _distance_point_segment(p, centers[a], centers[b]) < platelet_exclusion
                for p in platelets
            ):
                continue
            if any(
                a not in (u, v)
                and b not in (u, v)
                and _crosses(centers[a], centers[b], centers[u], centers[v])
                for u, v in edges
            ):
                continue
            candidates.append((float(distance[a, b]), b))
        if not candidates:
            remaining[a] = 0
            continue
        candidates.sort()
        shortlist = candidates[: min(3, len(candidates))]
        b = shortlist[0 if rng.random() < 0.8 else int(rng.integers(len(shortlist)))][1]
        key = (min(a, b), max(a, b))
        edge_set.add(key)
        edges.append(key)
        remaining[a] -= 1
        remaining[b] -= 1
    return edges


def _turning_angle(xi: np.ndarray, xj: np.ndarray, xk: np.ndarray) -> float:
    u, v = xj - xi, xk - xj
    return float(np.arctan2(u[0] * v[1] - u[1] * v[0], np.dot(u, v)))


def junction_degree_targets(config: Program19Config) -> dict[int, int]:
    """Target junction count per degree, ``{degree: count}``, in ascending degree.

    Each count is N_V * pi * (domain_size_um / 2)^2 * effective_thickness_um
    (default domain 48 um, i.e. the stereology R = 24 um),
    rounded (no +E[H] term: junctions are points, not spheres).  An explicit
    count in ``node_degree_counts`` wins.
    """
    clot_radius = 0.5 * config.domain_size_um
    volume = math.pi * clot_radius**2 * config.effective_thickness_um
    targets = {}
    for degree in sorted(config.node_degrees):
        if degree in config.node_degree_counts:
            targets[degree] = int(config.node_degree_counts[degree])
        else:
            targets[degree] = int(round(config.node_degree_density_um3[degree] * volume))
    return targets


def build_initial_state(config: Program19Config) -> SimulationState:
    """Generate one planar mechanical realization with explicit fiber identity."""
    config.validate()
    sequence = np.random.SeedSequence(config.seed)
    rng_geometry, rng_filopodia = [np.random.default_rng(s) for s in sequence.spawn(2)]
    clot_radius = 0.5 * config.domain_size_um
    targets = junction_degree_targets(config)
    main_count = max(2, sum(targets.values()))
    platelets = _place_platelets(
        rng_geometry, config.platelet_count, clot_radius, config.platelet_radius_um
    )
    platelet_exclusion = config.platelet_radius_um + config.fiber_radius_um + 0.02
    junctions = _place_junctions(
        rng_geometry,
        main_count,
        clot_radius,
        config.main_node_min_spacing_um,
        platelets,
        platelet_exclusion,
        config.platelet_near_node_fraction,
        config.platelet_force_radius_um,
    )
    # Stub list in ascending degree; for 3/4/5 this equals the original
    # [3]*n3 + [4]*n4 + [5]*n5, so the RNG stream and outputs are unchanged.
    degrees = np.asarray(
        [degree for degree, count in targets.items() for _ in range(count)], dtype=int
    )
    if len(degrees) < main_count:
        degrees = np.pad(degrees, (0, main_count - len(degrees)), constant_values=3)
    rng_geometry.shuffle(degrees)
    main_edges = _pair_planar_stubs(
        rng_geometry, junctions, degrees, platelets, platelet_exclusion
    )
    if not main_edges:
        raise RuntimeError("Initial geometry did not produce fibrin edges")

    nodes = [point.copy() for point in junctions]
    node_fibers: list[set[int]] = [set() for _ in nodes]
    pairs: list[tuple[int, int]] = []
    lengths: list[float] = []
    triples: list[tuple[int, int, int]] = []
    rest_angles: list[float] = []
    dual_lengths: list[float] = []
    for fiber_id, (a, b) in enumerate(main_edges):
        start, finish = junctions[a], junctions[b]
        edge_length = float(np.linalg.norm(finish - start))
        segment_count = max(2, int(math.ceil(edge_length / config.segment_target_um)))
        chain = [a]
        for fraction in np.linspace(0.0, 1.0, segment_count + 1)[1:-1]:
            node = (1.0 - fraction) * start + fraction * finish
            chain.append(len(nodes))
            nodes.append(node)
            node_fibers.append(set())
        chain.append(b)
        for node_id in chain:
            node_fibers[node_id].add(fiber_id)
        for left, right in zip(chain[:-1], chain[1:]):
            pairs.append((left, right))
            lengths.append(float(np.linalg.norm(nodes[right] - nodes[left])))
        for left, middle, right in zip(chain[:-2], chain[1:-1], chain[2:]):
            triples.append((left, middle, right))
            rest_angles.append(_turning_angle(nodes[left], nodes[middle], nodes[right]))
            dual_lengths.append(
                0.5
                * (
                    np.linalg.norm(nodes[middle] - nodes[left])
                    + np.linalg.norm(nodes[right] - nodes[middle])
                )
            )

    node_array = np.asarray(nodes, dtype=float)
    pair_array = np.asarray(pairs, dtype=int).reshape((-1, 2))
    incident: list[list[int]] = [[] for _ in nodes]
    adjacency: list[list[int]] = [[] for _ in nodes]
    for edge_id, (a, b) in enumerate(pair_array):
        incident[int(a)].append(edge_id)
        incident[int(b)].append(edge_id)
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))

    # The earlier Program19 only bent interior subdivision nodes.  Michael's
    # builder also assigns torsion triplets at fiber junctions.  In 2-D we use
    # angularly ordered neighboring segments and store their projected rest
    # angles, adding this missing contour-preservation mechanism.
    for middle in range(main_count):
        neighbors = adjacency[middle]
        if len(neighbors) < 2:
            continue
        ordered = sorted(
            neighbors,
            key=lambda node: math.atan2(
                node_array[node, 1] - node_array[middle, 1],
                node_array[node, 0] - node_array[middle, 0],
            ),
        )
        neighbor_pairs = (
            [(ordered[0], ordered[1])]
            if len(ordered) == 2
            else list(zip(ordered, ordered[1:] + ordered[:1]))
        )
        for left, right in neighbor_pairs:
            triples.append((left, middle, right))
            rest_angles.append(
                _turning_angle(node_array[left], node_array[middle], node_array[right])
            )
            dual_lengths.append(
                0.5
                * (
                    np.linalg.norm(node_array[middle] - node_array[left])
                    + np.linalg.norm(node_array[right] - node_array[middle])
                )
            )
    filopodia: list[list[int | None]] = []
    for _ in platelets:
        count = _sample_filopodia_count(rng_filopodia, config)
        filopodia.append([None] * count)
    return SimulationState(
        nodes_um=node_array,
        platelets_um=platelets,
        backbone_pairs=pair_array,
        backbone_L0_um=np.asarray(lengths),
        bending_triples=np.asarray(triples, dtype=int).reshape((-1, 3)),
        bending_phi0_rad=np.asarray(rest_angles),
        bending_dual_L0_um=np.asarray(dual_lengths),
        node_fiber_ids=[frozenset(item) for item in node_fibers],
        incident_backbone=incident,
        original_adjacency=adjacency,
        fixed_nodes=np.zeros(len(node_array), dtype=bool),
        filopodia=filopodia,
    )
