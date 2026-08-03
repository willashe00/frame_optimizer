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
from .lateral_loads import (UnsupportedSDCError, all_strength_combos,
                            lateral_load_basis, lateral_strength_combos,
                            summarize_lateral_basis)
from .lateral_designer import (LateralDesignResult, design_lateral,
                               lateral_baseplate_inputs,
                               lateral_system_block,
                               write_lateral_baseplate_json)
from .lateral_system import (LateralSystem, LateralSystemConfig,
                             build_lateral_system)
from .optimization import evaluate, geometry_for, optimize, optimize_layout
from .results import OptimizationResult
from .sections import WShape, get_shapes, load_w_shapes
from .site import (SeismicHazard, SiteConfig, SiteHazards, resolve_seismic,
                   resolve_site_hazards, resolve_wind_speed)

__all__ = [
    "ClearSpanConfig",
    "FrameConfig",
    "M_TO_FT",
    "OptimizationResult",
    "SeismicHazard",
    "SiteConfig",
    "SiteHazards",
    "UnsupportedSDCError",
    "WShape",
    "all_strength_combos",
    "baseplate_inputs",
    "building_configuration",
    "evaluate",
    "geometry_for",
    "get_shapes",
    "lateral_load_basis",
    "lateral_strength_combos",
    "load_w_shapes",
    "optimize",
    "optimize_layout",
    "resolve_seismic",
    "resolve_site_hazards",
    "resolve_wind_speed",
    "summarize_lateral_basis",
    "write_baseplate_json",
    "write_building_json",
]
