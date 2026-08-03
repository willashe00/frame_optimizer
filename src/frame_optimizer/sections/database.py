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


# ---------------------------------------------------------------------------
# Threaded-rod family: the real-world section for tension-only X bracing.
#
# Labels: "ROD3/4", "ROD1", "ROD1-1/8", ... (diameter in inches, mixed
# fractions with '-'). Rods are represented as WShape records with exact
# round-bar geometry (A = pi d^2/4, I = pi d^4/64, r = d/4, ...) so they
# flow through the analysis/assignment plumbing unchanged; the plate-element
# slenderness fields are set fully compact (they do not exist on a bar) and
# the DESIGN of rods bypasses the W-shape flexure machinery entirely — see
# design/checker.py (tension-only per AISC D2/J3.6, L/r limit not
# applicable per the D1 user note).
# ---------------------------------------------------------------------------
ROD_PREFIX = "ROD"
_STEEL_PLF_PER_IN2 = 490.0 / 144.0   # lb/ft of length per in^2 of area


def is_rod(shape_or_name) -> bool:
    name = getattr(shape_or_name, "name", shape_or_name)
    return str(name).upper().startswith(ROD_PREFIX)


def _rod_diameter_in(label: str) -> float:
    """'ROD1-1/8' -> 1.125 (mixed fraction after the ROD prefix)."""
    body = label[len(ROD_PREFIX):]
    try:
        total = 0.0
        for part in body.split("-"):
            if "/" in part:
                num, den = part.split("/")
                total += float(num) / float(den)
            else:
                total += float(part)
        if total <= 0.0:
            raise ValueError
        return total
    except (ValueError, ZeroDivisionError):
        raise ValueError(
            f"Unrecognized rod label {label!r}. Use ROD<diameter> with a "
            "mixed fraction in inches, e.g. 'ROD3/4', 'ROD1', 'ROD1-1/8'."
        ) from None


def rod_shape(label: str) -> WShape:
    """Exact round-bar properties for one rod label (see block comment)."""
    import math
    d = _rod_diameter_in(label)
    A = math.pi * d**2 / 4.0
    I = math.pi * d**4 / 64.0
    return WShape(
        name=label, weight_plf=A * _STEEL_PLF_PER_IN2, A=A,
        d=d, bf=d, tf=d / 2.0, tw=d / 2.0,
        Ix=I, Zx=d**3 / 6.0, Sx=math.pi * d**3 / 32.0, rx=d / 4.0,
        Iy=I, Zy=d**3 / 6.0, Sy=math.pi * d**3 / 32.0, ry=d / 4.0,
        J=math.pi * d**4 / 32.0, Cw=0.0,
        bf_2tf=1.0, h_tw=1.0,          # a bar has no slender plate elements
        rts=d / 4.0, ho=d,             # keep the (unused) F2 terms finite
    )


@lru_cache(maxsize=1)
def load_w_shapes() -> dict[str, WShape]:
    """Return the full catalog as {name: WShape}."""
    with resources.files("frame_optimizer.sections").joinpath("data/aisc_w_shapes.csv").open() as f:
        df = pd.read_csv(f)
    return {row["name"]: WShape(**row) for row in df.to_dict("records")}


def get_shapes(names: list[str]) -> list[WShape]:
    """Resolve candidate names (W-shapes and RODs) sorted lightest-first.

    Raises ValueError listing every unrecognized name.
    """
    catalog = load_w_shapes()
    normalized = [normalize_name(n) for n in names]
    unknown = [n for n in normalized if n not in catalog and not is_rod(n)]
    if unknown:
        raise ValueError(
            f"Unknown section name(s): {unknown}. "
            "Use AISC Manual labels such as 'W18X35', or rod labels such "
            "as 'ROD3/4'."
        )
    shapes = [rod_shape(n) if is_rod(n) else catalog[n]
              for n in dict.fromkeys(normalized)]  # dedupe, keep order
    return sorted(shapes, key=lambda s: s.weight_plf)
