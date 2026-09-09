"""frame_optimizer: gravity-load optimizer for fully pinned steel frames.

Part 1: Pynite FEA of the pinned frame (gravity only).
Part 2: AISC 360 LRFD checks + lightest-section search (rolled W-shapes
        and square HSS).

Two building types share the pipeline: FrameConfig (conventional column grid)
and ClearSpanConfig (clear-span industrial building, no interior columns).
"""
from .clear_span import ClearSpanConfig
from .config import FrameConfig
from .export import (baseplate_inputs, building_configuration,
                     write_baseplate_json, write_building_json)
from .optimization import evaluate, geometry_for, optimize, optimize_layout
from .results import OptimizationResult
from .sections import (HSSShape, Section, WShape, get_shapes,
                       load_hss_shapes, load_sections, load_w_shapes)

__all__ = [
    "ClearSpanConfig",
    "FrameConfig",
    "HSSShape",
    "OptimizationResult",
    "Section",
    "WShape",
    "baseplate_inputs",
    "building_configuration",
    "evaluate",
    "geometry_for",
    "get_shapes",
    "load_hss_shapes",
    "load_sections",
    "load_w_shapes",
    "optimize",
    "optimize_layout",
    "write_baseplate_json",
    "write_building_json",
]
