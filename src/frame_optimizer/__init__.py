"""frame_optimizer: gravity-load optimizer for fully pinned steel frames.

Part 1: Pynite FEA of the pinned frame (gravity only).
Part 2: AISC 360 LRFD checks + lightest-W-shape search.

Two building types share the pipeline: FrameConfig (conventional column grid)
and ClearSpanConfig (clear-span industrial building, no interior columns).
"""
from .clear_span import ClearSpanConfig
from .config import M_TO_FT, FrameConfig
from .export import (baseplate_inputs, building_configuration,
                     write_baseplate_json, write_building_json)
from .optimization import evaluate, geometry_for, optimize
from .results import OptimizationResult
from .sections import WShape, get_shapes, load_w_shapes

__all__ = [
    "ClearSpanConfig",
    "FrameConfig",
    "M_TO_FT",
    "OptimizationResult",
    "WShape",
    "baseplate_inputs",
    "building_configuration",
    "evaluate",
    "geometry_for",
    "get_shapes",
    "load_w_shapes",
    "optimize",
    "write_baseplate_json",
    "write_building_json",
]
