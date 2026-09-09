"""Section catalogs: rolled W-shapes and square HSS.

Loads the bundled CSVs derived from the AISC Shapes Database (US customary
units: inches, in^2, in^3, in^4, in^6, plf). See tools/prepare_sections_csv.py
for provenance, the rts/ho computation, and the square-HSS filter.

Both shape classes expose the properties the analysis layer consumes (name,
weight_plf, A, Ix, Iy, J) under the same names, so geometry, the FE model,
the optimizer and the checker never branch on section type. Only the AISC
strength equations do, in design/aisc_strengths.py.
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


@dataclass(frozen=True)
class HSSShape:
    """Geometric properties of one square HSS (US customary units).

    Area and both wall slendernesses are the DESIGN-wall values (tdes =
    0.93*tnom for ERW product), which is what AISC 360 Chapters E, F and G
    require. b_tdes and h_tdes are flat-width ratios: the flat widths
    themselves are recovered as b_tdes*tdes, keeping the corner radii
    consistent with the published ratios.

    Ht == B, Ix == Iy and rx == ry for every shape in this catalog; the two
    axes are kept separate anyway so the strength equations generalize to
    rectangular HSS without restructuring.
    """
    name: str
    weight_plf: float   # nominal weight, lb/ft
    A: float            # design-wall gross area, in^2
    Ht: float           # overall depth, in
    B: float            # overall width, in
    tnom: float         # nominal wall thickness, in
    tdes: float         # design wall thickness, in
    Ix: float           # major-axis moment of inertia, in^4
    Zx: float           # major-axis plastic modulus, in^3
    Sx: float           # major-axis elastic modulus, in^3
    rx: float           # major-axis radius of gyration, in
    Iy: float           # minor-axis moment of inertia, in^4
    Zy: float           # minor-axis plastic modulus, in^3
    Sy: float           # minor-axis elastic modulus, in^3
    ry: float           # minor-axis radius of gyration, in
    J: float            # torsional constant, in^4
    C: float            # HSS torsional shear constant, in^3
    b_tdes: float       # wall slenderness b/tdes (walls parallel to B)
    h_tdes: float       # wall slenderness h/tdes (walls parallel to Ht)

    @property
    def b_flat(self) -> float:
        """Flat width of the B-walls, in."""
        return self.b_tdes * self.tdes

    @property
    def h_flat(self) -> float:
        """Flat width of the Ht-walls, in."""
        return self.h_tdes * self.tdes


# Any catalog section. The analysis layer is written against this union and
# only touches the properties both members share.
Section = WShape | HSSShape


def normalize_name(name: str) -> str:
    return name.upper().replace(" ", "")


@lru_cache(maxsize=1)
def load_w_shapes() -> dict[str, WShape]:
    """Return the full W-shape catalog as {name: WShape}."""
    return _load("data/aisc_w_shapes.csv", WShape)


@lru_cache(maxsize=1)
def load_hss_shapes() -> dict[str, HSSShape]:
    """Return the full square-HSS catalog as {name: HSSShape}."""
    return _load("data/aisc_hss_square.csv", HSSShape)


@lru_cache(maxsize=1)
def load_sections() -> dict[str, Section]:
    """Both catalogs in one lookup. AISC Manual labels are unique across
    families ('W...' versus 'HSS...'), so the merge cannot collide."""
    return {**load_w_shapes(), **load_hss_shapes()}


def _load(resource: str, cls: type) -> dict:
    with resources.files("frame_optimizer.sections").joinpath(resource).open() as f:
        df = pd.read_csv(f)
    return {row["name"]: cls(**row) for row in df.to_dict("records")}


def get_shapes(names: list[str]) -> list[Section]:
    """Resolve candidate names to sections, sorted lightest-first.

    W-shapes and square HSS may be named interchangeably; a design group may
    mix families, since every downstream check dispatches per section.
    Raises ValueError listing every unrecognized name.
    """
    catalog = load_sections()
    normalized = [normalize_name(n) for n in names]
    unknown = [n for n in normalized if n not in catalog]
    if unknown:
        raise ValueError(
            f"Unknown section name(s): {unknown}. "
            "Use AISC Manual labels such as 'W18X35' or 'HSS8X8X1/2'."
        )
    shapes = [catalog[n] for n in dict.fromkeys(normalized)]  # dedupe, keep order
    return sorted(shapes, key=lambda s: s.weight_plf)
