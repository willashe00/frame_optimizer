"""JSON exports of an OptimizationResult for downstream modules.

Two machine-readable views of the optimized structure:

* baseplate_inputs()       - everything a (pinned-base) baseplate design
                             module needs per column: W-shape footprint
                             (d, bf, ...), base centerline location, and the
                             vertical base reaction per load combination.
* building_configuration() - the full optimized building for IFC authoring:
                             geometry (nodes + members), the selected W-shape
                             per design group with profile dimensions, loads,
                             material, and headline optimization results.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .analysis import build_model
from .analysis.frame_model import (SERVICE_LIVE_COMBO, SERVICE_TOTAL_COMBO,
                                   STRENGTH_COMBOS)
from .clear_span import ClearSpanConfig
from .config import COLUMN, IN_TO_MM, KIP_TO_KN, PLF_TO_KG_M, FrameConfig
from .geometry import FrameGeometry, MemberInfo, NodeInfo
from .optimization import geometry_for
from .results import OptimizationResult
from .sections import WShape, get_shapes

_SCHEMA_VERSION = 2
# baseplate_inputs is versioned separately: v3 added per-column base shear.
_BASEPLATE_SCHEMA_VERSION = 3

_IN2_TO_MM2 = IN_TO_MM ** 2


def _r(value: float, ndigits: int = 4) -> float:
    """Plain rounded float (also strips numpy scalar types for json)."""
    return round(float(value), ndigits)


def _require_config(result: OptimizationResult) -> object:
    if result.config is None:
        raise ValueError(
            "OptimizationResult.config is required to rebuild geometry for "
            "export; results produced by optimize()/evaluate() carry it."
        )
    return result.config


def _assignment(result: OptimizationResult) -> dict[str, WShape]:
    return {g: get_shapes([name])[0] for g, name in result.sections.items()}


def _section_dimensions(shape: WShape) -> dict:
    """The profile dimensions downstream geometry consumers need (SI; the
    name stays the AISC Manual label, which is an identifier, not a unit)."""
    return {
        "name": shape.name,
        "profile_type": "W-shape (AISC)",
        "depth_d_mm": _r(shape.d * IN_TO_MM, 2),
        "flange_width_bf_mm": _r(shape.bf * IN_TO_MM, 2),
        "flange_thickness_tf_mm": _r(shape.tf * IN_TO_MM, 2),
        "web_thickness_tw_mm": _r(shape.tw * IN_TO_MM, 2),
        "area_mm2": _r(shape.A * _IN2_TO_MM2, 1),
        "nominal_weight_kg_m": _r(shape.weight_plf * PLF_TO_KG_M, 2),
    }


def _base_columns(geometry: FrameGeometry) -> list[tuple[MemberInfo, NodeInfo]]:
    """(column member, its base node) for every column that lands on a base."""
    base_nodes = {n.name: n for n in geometry.nodes if n.is_base}
    return [(m, base_nodes[m.i_node]) for m in geometry.members
            if m.group == COLUMN and m.i_node in base_nodes]


def _base_reactions(result: OptimizationResult, geometry: FrameGeometry,
                    assignment: dict[str, WShape]
                    ) -> dict[str, dict[str, dict[str, float]]]:
    """Base reactions (kN) per node per combo, as {node: {'fy': ..., 'shear': ...}}.

    One linear solve of the final assignment; RxnFY is positive upward, i.e.
    equal to the axial compression the column delivers to the baseplate.
    'shear' is the horizontal resultant sqrt(RxnFX^2 + RxnFZ^2); under gravity
    on this fully pinned model it is numerical zero (the DX/DZ restraints only
    remove mechanism DOFs), and reporting it is what proves that.
    (This is the one solve that needs Pynite's reaction recovery, so it uses
    the public analyze().)
    """
    model = build_model(geometry, assignment, result.config)
    model.analyze(check_stability=True, check_statics=False, sparse=True)
    combos = list(STRENGTH_COMBOS) + [SERVICE_TOTAL_COMBO[0], SERVICE_LIVE_COMBO[0]]
    reactions = {}
    for node in geometry.nodes:
        if not node.is_base:
            continue
        n = model.nodes[node.name]
        reactions[node.name] = {
            "fy": {c: n.RxnFY[c] * KIP_TO_KN for c in combos},
            "shear": {c: math.hypot(n.RxnFX[c], n.RxnFZ[c]) * KIP_TO_KN
                      for c in combos},
        }
    return reactions


def baseplate_inputs(result: OptimizationResult) -> dict:
    """Per-column inputs for pinned-base baseplate design, as a dict."""
    config = _require_config(result)
    geometry = geometry_for(config)
    assignment = _assignment(result)
    reactions = _base_reactions(result, geometry, assignment)

    strength_combos = list(STRENGTH_COMBOS)
    total_combo = SERVICE_TOTAL_COMBO[0]
    live_combo = SERVICE_LIVE_COMBO[0]

    columns = []
    for member, base in sorted(_base_columns(geometry), key=lambda mb: mb[0].name):
        rxn = reactions[base.name]["fy"]
        shear = reactions[base.name]["shear"]
        by_combo = {c: _r(rxn[c]) for c in strength_combos}
        # linear analysis: service dead = (D+L) - L
        by_combo["D"] = _r(rxn[total_combo] - rxn[live_combo])
        by_combo[total_combo] = _r(rxn[total_combo])
        by_combo[live_combo] = _r(rxn[live_combo])
        columns.append({
            "member_id": member.name,
            "base_node": base.name,
            "section": _section_dimensions(assignment[member.group]),
            "centerline_location": {
                "x_mm": _r(base.x * IN_TO_MM, 1),
                "y_mm": _r(base.y * IN_TO_MM, 1),
                "z_mm": _r(base.z * IN_TO_MM, 1),
            },
            "column_height_mm": _r(member.length_in * IN_TO_MM, 1),
            "axial_compression_kN": {
                "Pu_governing_lrfd": _r(max(rxn[c] for c in strength_combos)),
                "by_combo": by_combo,
            },
            "base_shear_kN": {
                "Vu_governing_lrfd": _r(max(shear[c] for c in strength_combos)),
                "by_combo": {c: _r(shear[c]) for c in strength_combos},
            },
        })

    return {
        "schema": "frame_optimizer/baseplate_inputs",
        "schema_version": _BASEPLATE_SCHEMA_VERSION,
        "base_condition": "pinned",
        "units": {"length": "mm", "force": "kN", "stress": "MPa"},
        "sign_convention": (
            "axial_compression_kN values are vertical base reactions, "
            "positive in compression (bearing on the baseplate)"
        ),
        "notes": [
            "Gravity loads only; lateral (wind/seismic) base shear is out of "
            "scope of this model and must come from the lateral system design.",
            "base_shear_kN is the horizontal reaction this model actually "
            "produces, which is numerical zero by construction: the frame is "
            "fully pinned and its DX/DZ restraints exist only to remove "
            "mechanism DOFs. Design shear must be stated externally.",
            "Pu_governing_lrfd is the envelope over the LRFD strength combos "
            f"{strength_combos}; 'D', '{total_combo}', and '{live_combo}' are "
            "unfactored service-level values.",
            "Column plan orientation (web direction) is not defined by the "
            "gravity model; all sections are vertical W-shapes.",
        ],
        "material": {
            "Fy_MPa": _r(config.Fy_mpa),
            "Fu_MPa": _r(config.Fu_mpa),
            "E_MPa": _r(config.E_mpa),
        },
        "columns": columns,
    }


def _building_block(config: FrameConfig | ClearSpanConfig) -> dict:
    """Type-specific plan/elevation summary of the building."""
    if isinstance(config, ClearSpanConfig):
        roof = ("Pratt truss (top-chord bearing) spanning frame to frame"
                if config.is_truss else "girders")
        block = {
            "building_type": "clear_span",
            "description": ("transverse clear-span frames, no interior "
                            f"columns; one-way deck -> purlins -> {roof} "
                            "-> perimeter columns"),
            "roof_system": "truss" if config.is_truss else "girder",
            "span_m": _r(config.span_m),
            "length_m": _r(config.length_m),
            "height_m": _r(config.height_m),
            "n_frames": config.n_frames,
            "frame_spacing_m": _r(config.frame_spacing_m),
            "n_purlin_lines": config.n_purlin_spaces + 1,
            "purlin_spacing_m": _r(config.purlin_spacing_actual_m),
            "end_wall_columns_per_end": config.end_wall_columns,
        }
        if config.is_truss:
            block.update({
                "truss_bearing": "top_chord",
                "truss_depth_m": _r(config.truss_depth_actual_m),
                "n_truss_panels": config.n_truss_panels,
                "truss_panel_length_m": _r(config.truss_panel_m),
                "truss_camber_mm": _r(config.truss_camber_mm, 1),
            })
        else:
            block["girder_camber_mm"] = _r(config.girder_camber_mm, 1)
        return block
    return {
        "building_type": "grid_frame",
        "description": "conventional column grid, one-way deck on floor beams",
        "x_bays": config.x_bays,
        "x_bay_spacing_m": _r(config.x_bay_spacing_m),
        "z_bays": config.z_bays,
        "z_bay_spacing_m": _r(config.z_bay_spacing_m),
        "stories": config.stories,
        "story_heights_m": [_r(h) for h in config.story_heights_m],
        "deck_span_direction": config.deck_span_direction,
    }


def building_configuration(result: OptimizationResult) -> dict:
    """Full optimized building (geometry + sections) for IFC authoring."""
    config = _require_config(result)
    geometry = geometry_for(config)
    assignment = _assignment(result)

    group_rows = {row["group"]: row for _, row in result.group_summary.iterrows()}
    design_groups = {}
    for group, shape in assignment.items():
        row = group_rows[group]
        design_groups[group] = {
            "section": _section_dimensions(shape),
            "n_members": int(row["n_members"]),
            "weight_kg": _r(row["weight_kg"], 1),
            "max_unity_check": _r(row["max_uc"]),
            "governing_limit_state": row["governing_limitstate"],
        }

    nodes = [{
        "name": n.name,
        "x_mm": _r(n.x * IN_TO_MM, 1),
        "y_mm": _r(n.y * IN_TO_MM, 1),
        "z_mm": _r(n.z * IN_TO_MM, 1),
        "is_base": n.is_base,
    } for n in geometry.nodes]

    members = [{
        "name": m.name,
        "group": m.group,
        "section": assignment[m.group].name,
        "i_node": m.i_node,
        "j_node": m.j_node,
        "length_mm": _r(m.length_in * IN_TO_MM, 1),
    } for m in geometry.members]

    return {
        "schema": "frame_optimizer/building_configuration",
        "schema_version": _SCHEMA_VERSION,
        "units": {"length": "mm", "plan_dimensions": "m", "weight": "kg",
                  "load": "kPa", "stress": "MPa"},
        "coordinate_system": {
            "x": "plan (clear-span/girder direction for clear_span buildings)",
            "y": "vertical, up (gravity acts in -y)",
            "z": "plan (building length for clear_span buildings)",
            "origin": "base of the column at x=0, z=0",
        },
        "building": _building_block(config),
        "material": {
            "standard": "ASTM A992 defaults" if config.Fy_mpa == 345.0 else "user-specified",
            "Fy_MPa": _r(config.Fy_mpa),
            "Fu_MPa": _r(config.Fu_mpa),
            "E_MPa": _r(config.E_mpa),
        },
        "loads": {
            "superimposed_dead_kPa": _r(config.superimposed_dead_kpa),
            "live_kPa": _r(config.live_kpa),
            "self_weight": "included in analysis",
            "strength_combinations": list(STRENGTH_COMBOS),
            "lateral_loads": "out of scope (separate lateral system assumed)",
        },
        "connections": "all members pin-ended; column bases pinned",
        "design_groups": design_groups,
        "nodes": nodes,
        "members": members,
        "optimization": {
            "feasible": bool(result.feasible),
            "converged": bool(result.converged),
            "total_weight_kg": _r(result.total_weight_kg, 1),
        },
    }


def _write_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def write_baseplate_json(result: OptimizationResult | dict,
                         path: str | Path = "baseplate_inputs.json") -> Path:
    """Write baseplate_inputs(result) to `path`; returns the path written.

    An already-built baseplate_inputs() dict is accepted too, so a caller that
    also needs the data (the baseplate_design module does) can build it once
    and pay for only one reaction solve.
    """
    data = result if isinstance(result, dict) else baseplate_inputs(result)
    return _write_json(data, path)


def write_building_json(result: OptimizationResult,
                        path: str | Path = "building_configuration.json") -> Path:
    """Write building_configuration(result) to `path`; returns the path written."""
    return _write_json(building_configuration(result), path)
