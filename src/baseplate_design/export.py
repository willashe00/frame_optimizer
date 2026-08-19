"""JSON export of a uniform baseplate design.

Mirrors `frame_optimizer.export`: one machine-readable view of the finished
design, every numeric key carrying its unit as a suffix. SI is primary, as
everywhere else in this project; the plate and rod dimensions additionally
carry their imperial values, because those are the fabrication increments the
design actually rounds to (1/2 in plan, 1/8 in thickness and rod diameter)
and a shop drawing will be dimensioned in them.

Consumers: fabrication / detailing, and the IFC authoring module -- the rod
positions and plate footprint are given relative to the column centerline, so
a plate can be placed directly on the base nodes of
`building_configuration.json`.
"""
from __future__ import annotations

import json
from pathlib import Path

from frame_optimizer.config import IN_TO_MM, KIP_TO_KN

from .baseplate_design import (FNV_C, MAX_CONFINEMENT, PHI_B, PHI_C, PHI_V,
                               BaseplateCheck, ColumnDemand)
from .config import BaseplateConfig
from .uniform_design import LIMIT_STATES, UniformBaseplateDesign

_SCHEMA_VERSION = 1

_IN2_TO_MM2 = IN_TO_MM ** 2


def _r(value: float, ndigits: int = 4) -> float:
    """Plain rounded float (also strips numpy scalar types for json)."""
    return round(float(value), ndigits)


def _design_basis(config: BaseplateConfig) -> dict:
    return {
        "codes": [
            "AISC 360-22 (Specification for Structural Steel Buildings)",
            "AISC Design Guide 1, 2nd Ed. (Base Plate and Anchor Rod Design)",
            "ACI 318-19 Table 22.9.4.2 (shear-friction coefficient)",
        ],
        "base_condition": "pinned",
        "resistance_factors": {
            "phi_bearing_concrete": PHI_C,
            "phi_plate_flexure": PHI_B,
            "phi_anchor_shear": PHI_V,
        },
        "methodology": [
            "Plan size from the DG1 Sec. 3.3 equal-cantilever optimum (m = n), "
            "held to detailing minimums so the plate covers the column and "
            "leaves room for a rod clear of each flange.",
            "Bearing per AISC 360-22 Eq. J8-2 with the confinement factor "
            f"sqrt(A2/A1) capped at {MAX_CONFINEMENT}; A2 is the largest area "
            "geometrically similar to and concentric with the plate that fits "
            "on the pier, solved together with the plate size.",
            "Thickness per DG1 Eq. 3.3.17 (LRFD) using the governing "
            "cantilever max(m, n, lambda*n').",
            "Anchor rod shear at Fnv = "
            f"{FNV_C}*Fu (AISC Table J3.2, threads in shear plane), after a "
            "shear-friction credit phi*mu*P taken on the MINIMUM compression "
            f"coincident with the shear ({config.friction_axial_factor:g}D), "
            "never on the maximum.",
            "One plate for every column: each column is designed, the "
            "dimensions are enveloped, and the single enveloped plate is then "
            "re-checked against every column individually.",
        ],
        "limit_states_checked": list(LIMIT_STATES),
        "limit_states_excluded": [
            "anchor rod tension and concrete breakout / pryout (ACI 318 Ch. 17)",
            "column-to-plate weld design",
            "moment at the base (e = Mu/Pu, DG1 Sec. 3.4) - the base is pinned",
            "grout bearing and leveling nut design",
        ],
    }


def _inputs_block(config: BaseplateConfig) -> dict:
    pedestal = (
        {"width_mm": _r(config.pedestal_width_mm, 1),
         "length_mm": _r(config.pedestal_length_mm, 1),
         "source": "given"}
        if config.has_explicit_pedestal else
        {"edge_projection_mm": _r(config.pedestal_edge_projection_mm, 1),
         "source": "assumed to project past the plate on all four sides"}
    )
    return {
        "concrete": {"fc_MPa": _r(config.fc_mpa, 2), "pedestal": pedestal},
        "plate_material": {"Fy_MPa": _r(config.plate_Fy_mpa, 1),
                           "grade": "ASTM A36 (nominal)"},
        "anchor_rods": {
            "count_per_column": config.n_rods,
            "Fu_MPa": _r(config.anchor_Fu_mpa, 1),
            "grade": "ASTM F1554 Gr. 36 (nominal)",
            "friction_coefficient_mu": _r(config.friction_coefficient, 3),
            "friction_axial_factor": _r(config.friction_axial_factor, 3),
        },
        "design_base_shear_kN": _r(config.design_base_shear_kN, 3),
        "detailing_mm": {
            "min_edge_distance": _r(config.min_edge_distance_mm, 2),
            "rod_clearance_from_flange": _r(config.rod_clearance_mm, 2),
            "min_plate_projection": _r(config.min_plate_projection_mm, 2),
            "edge_distance_rod_factor": _r(config.edge_distance_rod_factor, 3),
        },
        "fabrication_increments_mm": {
            "plan_dimensions": _r(config.plate_increment_mm, 3),
            "thickness": _r(config.thickness_increment_mm, 3),
            "min_thickness": _r(config.min_thickness_mm, 3),
            "rod_diameter": _r(config.rod_increment_mm, 3),
            "min_rod_diameter": _r(config.min_rod_diameter_mm, 3),
        },
    }


def _baseplate_block(design: UniformBaseplateDesign) -> dict:
    plate = design.plate
    return {
        "applies_to": f"all {design.n_columns} column base(s) - one detail, "
                      "one fabrication batch",
        "orientation": ("B is parallel to the column flange width bf, "
                        "N is parallel to the column depth d; the plate is "
                        "concentric with the column centerline"),
        "plate": {
            "width_B_mm": _r(plate.B * IN_TO_MM, 1),
            "length_N_mm": _r(plate.N * IN_TO_MM, 1),
            "thickness_tp_mm": _r(plate.tp * IN_TO_MM, 2),
            "width_B_in": _r(plate.B, 3),
            "length_N_in": _r(plate.N, 3),
            "thickness_tp_in": _r(plate.tp, 4),
            "bearing_area_A1_mm2": _r(plate.area * _IN2_TO_MM2, 0),
            "mass_kg": _r(design.plate_mass_kg, 2),
        },
        "anchor_rods": {
            "count": plate.n_rods,
            "diameter_mm": _r(plate.d_rod * IN_TO_MM, 2),
            "diameter_in": _r(plate.d_rod, 4),
            "edge_distance_mm": _r(plate.edge_distance * IN_TO_MM, 1),
            "layout": ("half the rods outside each column flange, spread "
                       "evenly across the plate width at the edge distance"),
            "positions_mm": [
                {"x_mm": _r(x * IN_TO_MM, 1), "y_mm": _r(y * IN_TO_MM, 1)}
                for x, y in plate.rod_positions()
            ],
            "position_note": ("offsets from the plate/column centerline; "
                              "x runs along B, y runs along N"),
        },
    }


def _column_row(demand: ColumnDemand, check: BaseplateCheck) -> dict:
    return {
        "member_id": demand.column_id,
        "base_node": demand.base_node,
        "section": demand.section_name,
        "centerline_location": {
            "x_mm": _r(demand.location_in[0] * IN_TO_MM, 1),
            "y_mm": _r(demand.location_in[1] * IN_TO_MM, 1),
            "z_mm": _r(demand.location_in[2] * IN_TO_MM, 1),
        },
        "demands_kN": {
            "Pu_governing_lrfd": _r(demand.Pu * KIP_TO_KN, 2),
            "Vu_design_shear": _r(demand.Vu * KIP_TO_KN, 3),
            "P_coincident_for_friction": _r(demand.P_friction * KIP_TO_KN, 2),
        },
        "capacity_kN": {
            "phiPp_bearing": _r(check.phiPp * KIP_TO_KN, 1),
            "phiVn_shear_total": _r(check.phiVn * KIP_TO_KN, 2),
            "friction_credit": _r(check.friction * KIP_TO_KN, 2),
            "shear_carried_by_rods": _r(check.V_rods * KIP_TO_KN, 2),
        },
        "plate_flexure": {
            "cantilever_m_mm": _r(check.m * IN_TO_MM, 1),
            "cantilever_n_mm": _r(check.n * IN_TO_MM, 1),
            "lambda_n_prime_mm": _r(check.lam_np * IN_TO_MM, 1),
            "governing_cantilever_mm": _r(check.ell * IN_TO_MM, 1),
            "t_required_mm": _r(check.t_req * IN_TO_MM, 2),
        },
        "dcr": {
            "bearing": _r(check.bearing_dcr, 4),
            "plate_flexure": _r(check.flexure_dcr, 4),
            "anchor_rod_shear": _r(check.shear_dcr, 4),
        },
        "governing_limit_state": check.governing_limit_state,
        "max_dcr": _r(check.max_dcr, 4),
        "PASS": bool(check.passes),
    }


def _governing_block(design: UniformBaseplateDesign) -> dict:
    gov, dcr = design.governing_column, design.max_dcr
    detail = {}
    for state in LIMIT_STATES:
        demand = design.demand_for(gov[state])
        detail[state] = {
            "member_id": gov[state],
            "dcr": _r(dcr[state], 4),
            "Pu_kN": _r(demand.Pu * KIP_TO_KN, 2),
            "Vu_kN": _r(demand.Vu * KIP_TO_KN, 3),
            "P_coincident_for_friction_kN": _r(demand.P_friction * KIP_TO_KN, 2),
        }
    pus = [d.Pu * KIP_TO_KN for d in design.demands]
    return {
        "overall_member_id": design.overall_governing_column,
        "by_limit_state": detail,
        "why_they_differ": (
            "Bearing and plate flexure are governed by the most heavily "
            "loaded column; anchor rod shear is governed by the LIGHTEST "
            "column, whose small shear-friction credit leaves its rods the "
            "most to carry."
        ),
        "column_axial_range_kN": {"min": _r(min(pus), 2), "max": _r(max(pus), 2)},
    }


def baseplate_design_configuration(design: UniformBaseplateDesign) -> dict:
    """The finished uniform baseplate design, as a dict."""
    checks_by_id = {c.column_id: c for c in design.checks}
    return {
        "schema": "baseplate_design/baseplate_design",
        "schema_version": _SCHEMA_VERSION,
        "units": {"length": "mm", "area": "mm^2", "force": "kN",
                  "stress": "MPa", "mass": "kg"},
        "source": {
            "module": "frame_optimizer",
            "contract": "frame_optimizer/baseplate_inputs",
            "note": ("column sections and base reactions come from the "
                     "finalized gravity member design; this file is written "
                     "automatically once that optimization completes"),
        },
        "design_basis": _design_basis(design.config),
        "inputs": _inputs_block(design.config),
        "baseplate": _baseplate_block(design),
        "governing": _governing_block(design),
        "verification": {
            "all_columns_pass": bool(design.feasible),
            "n_columns_checked": design.n_columns,
            "envelope_passes": design.envelope_passes,
            "envelope_dcr": {
                "bearing": _r(design.max_dcr["bearing"], 4),
                "plate_flexure": _r(design.max_dcr["plate flexure"], 4),
                "anchor_rod_shear": _r(design.max_dcr["anchor rod shear"], 4),
            },
            "failing_columns": [c.column_id for c in design.checks if not c.passes],
        },
        "columns": [_column_row(d, checks_by_id[d.column_id])
                    for d in design.demands],
        "totals": {
            "n_baseplates": design.n_columns,
            "plate_mass_kg_each": _r(design.plate_mass_kg, 2),
            "total_plate_mass_kg": _r(design.total_plate_mass_kg, 1),
        },
    }


def write_baseplate_design_json(design: UniformBaseplateDesign,
                                path: str | Path = "baseplate_design.json") -> Path:
    """Write baseplate_design_configuration(design) to `path`; returns it."""
    path = Path(path)
    path.write_text(
        json.dumps(baseplate_design_configuration(design), indent=2) + "\n",
        encoding="utf-8")
    return path
