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

Every numeric key carries an explicit unit suffix (_in, _ft, _kip, _psf,
_ksi, _lb, _plf) so downstream consumers never have to guess. Baseplate
reactions come from one extra FEA solve of the final assignment (linear
analysis, so service 'D' is recovered exactly as (D+L) - L).
"""
from __future__ import annotations

import json
from pathlib import Path

from .analysis import build_model
from .analysis.frame_model import (SERVICE_LIVE_COMBO, SERVICE_TOTAL_COMBO,
                                   STRENGTH_COMBOS)
from .clear_span import TRUSS, ClearSpanConfig
from .config import COLUMN, FrameConfig
from .geometry import FrameGeometry, MemberInfo, NodeInfo
from .optimization import geometry_for
from .results import OptimizationResult
from .sections import WShape, get_shapes

_SCHEMA_VERSION = 1


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
    """The profile dimensions downstream geometry consumers need."""
    return {
        "name": shape.name,
        "profile_type": "W-shape (AISC)",
        "depth_d_in": _r(shape.d),
        "flange_width_bf_in": _r(shape.bf),
        "flange_thickness_tf_in": _r(shape.tf),
        "web_thickness_tw_in": _r(shape.tw),
        "area_in2": _r(shape.A),
        "nominal_weight_plf": _r(shape.weight_plf),
    }


def _base_columns(geometry: FrameGeometry) -> list[tuple[MemberInfo, NodeInfo]]:
    """(column member, its base node) for every column that lands on a base."""
    base_nodes = {n.name: n for n in geometry.nodes if n.is_base}
    return [(m, base_nodes[m.i_node]) for m in geometry.members
            if m.group == COLUMN and m.i_node in base_nodes]


def _base_reactions(result: OptimizationResult, geometry: FrameGeometry,
                    assignment: dict[str, WShape]) -> dict[str, dict[str, dict[str, float]]]:
    """Base reactions per node per combo: {'FY': ..., 'FX': ...} (kip).

    One linear solve of the final assignment on the nominal-stiffness model
    (DAM notional loads are stability devices, not real loads, so they are
    excluded from exported reactions). RxnFY is positive upward, i.e. equal
    to the axial compression the column delivers to the baseplate. RxnFX is
    the horizontal thrust in the clear-span (X) direction — essentially zero
    for the pinned wide-flange system, real and self-equilibrating for rigid
    truss bents (frame action under gravity).
    """
    model = build_model(geometry, assignment, result.config)
    model.analyze(check_stability=True, check_statics=False, sparse=True)
    combos = list(STRENGTH_COMBOS) + [SERVICE_TOTAL_COMBO[0], SERVICE_LIVE_COMBO[0]]
    return {
        node.name: {c: {"FY": model.nodes[node.name].RxnFY[c],
                        "FX": model.nodes[node.name].RxnFX[c]}
                    for c in combos}
        for node in geometry.nodes if node.is_base
    }


def baseplate_inputs(result: OptimizationResult) -> dict:
    """Per-column inputs for pinned-base baseplate design, as a dict."""
    config = _require_config(result)
    geometry = geometry_for(config)
    assignment = _assignment(result)
    reactions = _base_reactions(result, geometry, assignment)

    strength_combos = list(STRENGTH_COMBOS)
    total_combo = SERVICE_TOTAL_COMBO[0]
    live_combo = SERVICE_LIVE_COMBO[0]

    def by_combo_block(rxn: dict, component: str) -> dict:
        block = {c: _r(rxn[c][component]) for c in strength_combos}
        # linear analysis: service dead = (D+L) - L
        block["D"] = _r(rxn[total_combo][component] - rxn[live_combo][component])
        block[total_combo] = _r(rxn[total_combo][component])
        block[live_combo] = _r(rxn[live_combo][component])
        return block

    columns = []
    for member, base in sorted(_base_columns(geometry), key=lambda mb: mb[0].name):
        rxn = reactions[base.name]
        columns.append({
            "member_id": member.name,
            "base_node": base.name,
            "section": _section_dimensions(assignment[member.group]),
            "centerline_location": {
                "x_in": _r(base.x), "y_in": _r(base.y), "z_in": _r(base.z),
            },
            "column_height_in": _r(member.length_in),
            "axial_compression_kip": {
                "Pu_governing_lrfd": _r(max(rxn[c]["FY"] for c in strength_combos)),
                "by_combo": by_combo_block(rxn, "FY"),
            },
            # thrust from rigid-bent frame action under gravity (truss
            # girders); essentially zero for the pinned wide-flange system
            "base_shear_x_kip": {
                "Vu_governing_lrfd": _r(max(
                    (abs(rxn[c]["FX"]) for c in strength_combos))),
                "by_combo": by_combo_block(rxn, "FX"),
            },
        })

    return {
        "schema": "frame_optimizer/baseplate_inputs",
        "schema_version": _SCHEMA_VERSION,
        "base_condition": "pinned",
        "units": {"length": "in", "force": "kip", "stress": "ksi"},
        "sign_convention": (
            "axial_compression_kip values are vertical base reactions, "
            "positive in compression (bearing on the baseplate); "
            "base_shear_x_kip values are signed horizontal reactions in the "
            "clear-span (X) direction"
        ),
        "notes": [
            "Gravity loads only; lateral (wind/seismic) base shear is out of "
            "scope of this model and must come from the lateral system design.",
            "base_shear_x_kip is the gravity-induced thrust of rigid truss "
            "bents (frame action); it is essentially zero for the pinned "
            "wide-flange system. DAM notional loads (fictitious stability "
            "loads) are excluded from exported reactions.",
            "Pu_governing_lrfd is the envelope over the LRFD strength combos "
            f"{strength_combos}; 'D', '{total_combo}', and '{live_combo}' are "
            "unfactored service-level values.",
            "Column plan orientation (web direction) is not defined by the "
            "gravity model; all sections are vertical W-shapes.",
        ],
        "material": {
            "Fy_ksi": _r(config.Fy_ksi),
            "Fu_ksi": _r(config.Fu_ksi),
            "E_ksi": _r(config.E_ksi),
        },
        "columns": columns,
    }


def _building_block(config: FrameConfig | ClearSpanConfig) -> dict:
    """Type-specific plan/elevation summary of the building."""
    if isinstance(config, ClearSpanConfig):
        trussed = config.girder_system == TRUSS
        girder_word = "Pratt truss girders" if trussed else "girders"
        block = {
            "building_type": "clear_span",
            "description": ("transverse clear-span frames, no interior "
                            f"columns; one-way deck -> purlins -> {girder_word} "
                            "-> perimeter columns"),
            "girder_system": config.girder_system,
            "span_ft": _r(config.span_ft),
            "length_ft": _r(config.length_ft),
            "eave_height_ft": _r(config.eave_height_ft),
            "n_frames": config.n_frames,
            "frame_spacing_ft": _r(config.frame_spacing_ft),
            "n_purlin_lines": config.n_purlin_spaces + 1,
            "purlin_spacing_ft": _r(config.purlin_spacing_actual_ft),
            "end_wall_columns_per_end": config.end_wall_columns,
            "girder_camber_in": _r(config.girder_camber_in),
        }
        if trussed:
            block.update({
                # parallel-chord Pratt: bottom chord bears at the eave, top
                # chord (and roof plane) at eave + depth
                "truss_depth_ft": _r(config.truss_depth_ft),
                "truss_n_panels": config.n_truss_panels,
                "truss_panel_ft": _r(config.truss_panel_actual_ft),
                "roof_elevation_ft": _r(config.eave_height_ft
                                        + config.truss_depth_ft),
            })
        return block
    return {
        "building_type": "grid_frame",
        "description": "conventional column grid, one-way deck on floor beams",
        "x_bays": config.x_bays,
        "x_bay_spacing_ft": _r(config.x_bay_spacing_ft),
        "z_bays": config.z_bays,
        "z_bay_spacing_ft": _r(config.z_bay_spacing_ft),
        "stories": config.stories,
        "story_heights_ft": [_r(h) for h in config.story_heights_ft],
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
            "weight_lb": _r(row["weight_lb"], 1),
            "max_unity_check": _r(row["max_uc"]),
            "governing_limit_state": row["governing_limitstate"],
        }

    nodes = [{
        "name": n.name,
        "x_in": _r(n.x), "y_in": _r(n.y), "z_in": _r(n.z),
        "is_base": n.is_base,
    } for n in geometry.nodes]

    members = [{
        "name": m.name,
        "group": m.group,
        "section": assignment[m.group].name,
        "i_node": m.i_node,
        "j_node": m.j_node,
        "length_in": _r(m.length_in),
    } for m in geometry.members]

    return {
        "schema": "frame_optimizer/building_configuration",
        "schema_version": _SCHEMA_VERSION,
        "units": {"length": "in", "plan_dimensions": "ft", "weight": "lb"},
        "coordinate_system": {
            "x": "plan (clear-span/girder direction for clear_span buildings)",
            "y": "vertical, up (gravity acts in -y)",
            "z": "plan (building length for clear_span buildings)",
            "origin": "base of the column at x=0, z=0",
        },
        "building": _building_block(config),
        "material": {
            "standard": "ASTM A992 defaults" if config.Fy_ksi == 50.0 else "user-specified",
            "Fy_ksi": _r(config.Fy_ksi),
            "Fu_ksi": _r(config.Fu_ksi),
            "E_ksi": _r(config.E_ksi),
        },
        "loads": {
            "superimposed_dead_psf": _r(config.superimposed_dead_psf),
            "live_psf": _r(config.live_psf),
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
            "total_weight_lb": _r(result.total_weight_lb, 1),
        },
    }


def _write_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def write_baseplate_json(result: OptimizationResult,
                         path: str | Path = "baseplate_inputs.json") -> Path:
    """Write baseplate_inputs(result) to `path`; returns the path written."""
    return _write_json(baseplate_inputs(result), path)


def write_building_json(result: OptimizationResult,
                        path: str | Path = "building_configuration.json") -> Path:
    """Write building_configuration(result) to `path`; returns the path written."""
    return _write_json(building_configuration(result), path)
