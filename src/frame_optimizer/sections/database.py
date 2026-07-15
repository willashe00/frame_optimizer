"""W-shape section catalog.

Loads the bundled CSV derived from the AISC Shapes Database (US customary
units: inches, in^2, in^3, in^4, in^6, plf). See tools/prepare_sections_csv.py
for provenance and the rts/ho computation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import pandas as pd


@dataclass(frozen=True)
class WShape:
    """Geometric properties of one rolled W-shape (US customary units)."""
    name: str
    weight_plf: float   # nominal weight, lb/ft
    A: float            # gross area, in^2
    d: float            # overall depth, in
    bf: float           # flange width, in
    tf: float           # flange thickness, in
    tw: float           # web thickness, in
    Ix: float           # major-axis moment of inertia, in^4
    Zx: float           # major-axis plastic modulus, in^3
    Sx: float           # major-axis elastic modulus, in^3
    rx: float           # major-axis radius of gyration, in
    Iy: float           # minor-axis moment of inertia, in^4
    Zy: float           # minor-axis plastic modulus, in^3
    Sy: float           # minor-axis elastic modulus, in^3
    ry: float           # minor-axis radius of gyration, in
    J: float            # torsional constant, in^4
    Cw: float           # warping constant, in^6
    bf_2tf: float       # flange slenderness bf/(2*tf)
    h_tw: float         # web slenderness h/tw
    rts: float          # effective radius of gyration for LTB, in (F2-7)
    ho: float           # distance between flange centroids, in


def normalize_name(name: str) -> str:
    return name.upper().replace(" ", "")


@lru_cache(maxsize=1)
def load_w_shapes() -> dict[str, WShape]:
    """Return the full catalog as {name: WShape}."""
    with resources.files("frame_optimizer.sections").joinpath("data/aisc_w_shapes.csv").open() as f:
        df = pd.read_csv(f)
    return {row["name"]: WShape(**row) for row in df.to_dict("records")}


def get_shapes(names: list[str]) -> list[WShape]:
    """Resolve candidate names to WShapes, sorted lightest-first.

    Raises ValueError listing every unrecognized name.
    """
    catalog = load_w_shapes()
    normalized = [normalize_name(n) for n in names]
    unknown = [n for n in normalized if n not in catalog]
    if unknown:
        raise ValueError(
            f"Unknown W-shape name(s): {unknown}. "
            "Use AISC Manual labels such as 'W18X35'."
        )
    shapes = [catalog[n] for n in dict.fromkeys(normalized)]  # dedupe, keep order
    return sorted(shapes, key=lambda s: s.weight_plf)
