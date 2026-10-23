"""Minimal CUDA-only 2-D platelet-fibrin mechanics core."""

from .config import Program19Config
from .geometry import build_initial_state, junction_degree_targets
from .state import SimulationState

__all__ = ["Program19Config", "SimulationState", "build_initial_state", "junction_degree_targets"]
