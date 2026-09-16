"""JSON export of the single baseplate detail.

Only the parameters a modeler needs to build the one plate that serves every
column base:

    N        plate dimension along the column's local-x axis (parallel to
             the column depth d)
    B        plate dimension along the column's local-y axis (parallel to
             the column flange width bf)
    t_p      plate thickness
    e_min    distance from the outer edge of the plate to the centre of each
             anchor bolt (the same on all four edges)
    bolt_dia anchor bolt diameter
    n_bolts  number of anchor bolts (placed symmetrically, half outside
             each column flange, e_min in from the plate edges)

None of the design record -- codes, inputs, demands, limit-state checks -- is
written here; that stays in `design.summary()` and in the wireframe hover
cards. Placement is not in this file either: one plate detail serves every
column, concentric with the column centerline, so it drops straight onto the
base nodes already given in `building_configuration.json`.

SI, as everywhere else in this project: mm.
"""
from __future__ import annotations

import json
from pathlib import Path

from frame_optimizer.config import IN_TO_MM

from .uniform_design import UniformBaseplateDesign

_SCHEMA_VERSION = 2


def _r(value: float, ndigits: int = 2) -> float:
    """Plain rounded float (also strips numpy scalar types for json)."""
    return round(float(value), ndigits)


def baseplate_configuration(design: UniformBaseplateDesign) -> dict:
    """The one baseplate detail, as a flat dict of design parameters (mm)."""
    plate = design.plate
    return {
        "schema": "baseplate_design/baseplate_configuration",
        "schema_version": _SCHEMA_VERSION,
        "units": {"length": "mm"},
        "N": _r(plate.N * IN_TO_MM, 1),
        "B": _r(plate.B * IN_TO_MM, 1),
        "t_p": _r(plate.tp * IN_TO_MM, 2),
        "e_min": _r(plate.edge_distance * IN_TO_MM, 1),
        "bolt_dia": _r(plate.d_rod * IN_TO_MM, 2),
        "n_bolts": int(plate.n_rods),
    }


def write_baseplate_configuration_json(
        design: UniformBaseplateDesign,
        path: str | Path = "baseplate_configuration.json") -> Path:
    """Write baseplate_configuration(design) to `path`; returns it."""
    path = Path(path)
    path.write_text(
        json.dumps(baseplate_configuration(design), indent=2) + "\n",
        encoding="utf-8")
    return path
