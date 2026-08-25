"""JSON export of the single baseplate detail.

Only what a 3-D modeler needs to build one plate: its plan dimensions, its
thickness, and the anchor rods (count, diameter, and where they sit). None of
the design record -- codes, inputs, demands, limit-state checks -- is written
here; that stays in `design.summary()` and in the wireframe hover cards.

Placement is not in this file either. One plate detail serves every column
base, concentric with the column centerline, so it drops straight onto the
base nodes already given in `building_configuration.json`.

Rod positions are offsets from the plate center: x runs along the plate width
(parallel to the column flange width bf), y along the plate length (parallel
to the column depth d).

SI, as everywhere else in this project: mm.
"""
from __future__ import annotations

import json
from pathlib import Path

from frame_optimizer.config import IN_TO_MM

from .uniform_design import UniformBaseplateDesign

_SCHEMA_VERSION = 1


def _r(value: float, ndigits: int = 2) -> float:
    """Plain rounded float (also strips numpy scalar types for json)."""
    return round(float(value), ndigits)


def baseplate_configuration(design: UniformBaseplateDesign) -> dict:
    """The one baseplate detail, as a dict: plate geometry and anchor rods."""
    plate = design.plate
    return {
        "schema": "baseplate_design/baseplate_configuration",
        "schema_version": _SCHEMA_VERSION,
        "units": {"length": "mm"},
        "plate": {
            "width_mm": _r(plate.B * IN_TO_MM, 1),
            "length_mm": _r(plate.N * IN_TO_MM, 1),
            "thickness_mm": _r(plate.tp * IN_TO_MM, 2),
        },
        "anchor_rods": {
            "count": plate.n_rods,
            "diameter_mm": _r(plate.d_rod * IN_TO_MM, 2),
            "positions_mm": [
                {"x_mm": _r(x * IN_TO_MM, 1), "y_mm": _r(y * IN_TO_MM, 1)}
                for x, y in plate.rod_positions()
            ],
        },
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
