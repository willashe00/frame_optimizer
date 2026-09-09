"""AISC 360-16/22 LRFD strength equations for rolled W-shapes and square HSS.

Pure functions: Section + material + lengths in, (phi*Rn, clause) out.
Units: kips, inches, ksi.

The five public capacity functions dispatch on section type; everything a
caller sees is family-independent, so design/checker.py never branches.

Implemented limit states:
    D2  tension yielding (rupture not checked - no connection geometry)
    E3  flexural buckling in compression (both axes, K = 1 pin-pin)
    E7  slender-element compression (effective areas; many light W-shapes
        have slender webs, and many thin HSS have slender walls, at Fy = 50)
    F2  W major-axis flexure: yielding + lateral-torsional buckling
    F3  W major-axis flexure: compression-flange local buckling
    F6  W minor-axis flexure: yielding + flange local buckling
    F7  square-HSS flexure: yielding + wall local buckling (both axes)
    G2  W web shear (Cv1; phi_v = 1.0 for stocky rolled webs)
    G4  HSS shear on the two walls parallel to the load (Cv2)
    H1  combined axial + flexure interaction

Torsional buckling (E4) is not evaluated: it does not govern doubly
symmetric W-shapes with KLz = KLy, and a closed HSS has a torsional
constant orders of magnitude too large for it to control.
"""
from __future__ import annotations

import math
from typing import NamedTuple

from ..sections import HSSShape, Section, WShape

PHI_C = 0.90   # compression
PHI_T = 0.90   # tension yielding
PHI_B = 0.90   # flexure
PHI_V_HSS = 0.90   # shear, G1 general provision


class Strength(NamedTuple):
    phi_Rn: float
    clause: str


def _unsupported(shape: Section) -> TypeError:
    name = getattr(shape, "name", "?")
    return TypeError(
        f"No AISC strength equations for section type {type(shape).__name__} "
        f"({name}). Supported: WShape, HSSShape."
    )


# ---------------------------------------------------------------------------
# Axial: tension
# ---------------------------------------------------------------------------

def tension_capacity(shape: Section, Fy: float) -> Strength:
    """D2 yielding on the gross section. Rupture (D2-2) needs net-area /
    connection data, which a member-level optimizer does not have; yielding
    governs for members without significant section loss. Family-independent."""
    return Strength(PHI_T * Fy * shape.A, "D2 (tension yielding)")


# ---------------------------------------------------------------------------
# Axial: compression
# ---------------------------------------------------------------------------

class _Element(NamedTuple):
    """One compression element of a cross-section, for the E7 check."""
    lam: float      # width-to-thickness ratio
    b: float        # full flat width, in
    t: float        # thickness, in
    lam_r: float    # nonslender limit (Table B4.1a)
    c1: float       # Table E7.1
    c2: float       # Table E7.1
    count: int      # how many identical elements the section has


def _e3_critical_stress(Fy: float, E: float, klr: float) -> float:
    """E3 flexural-buckling stress for one axis."""
    if klr < 1e-9:
        return Fy
    Fe = math.pi**2 * E / klr**2
    if Fy / Fe <= 2.25:
        return 0.658 ** (Fy / Fe) * Fy   # E3-2 (inelastic)
    return 0.877 * Fe                    # E3-3 (elastic)


def _w_elements(shape: WShape, Fy: float, E: float) -> tuple[_Element, ...]:
    """W-shape elements under uniform compression (Table B4.1a): each
    half-flange is an unstiffened element (lambda = bf/2tf, four outstands
    total), the web is a stiffened element (lambda = h/tw)."""
    return (
        _Element(shape.bf_2tf, shape.bf / 2.0, shape.tf,
                 0.56 * math.sqrt(E / Fy), 0.22, 1.49, 4),
        _Element(shape.h_tw, shape.h_tw * shape.tw, shape.tw,
                 1.49 * math.sqrt(E / Fy), 0.18, 1.31, 1),
    )


def _hss_elements(shape: HSSShape, Fy: float, E: float) -> tuple[_Element, ...]:
    """Square-HSS elements under uniform compression: all four walls are
    stiffened elements (Table B4.1a case 6, lambda_r = 1.40*sqrt(E/Fy)) of
    uniform thickness, so Table E7.1 gives c1 = 0.20, c2 = 1.38."""
    lam_r = 1.40 * math.sqrt(E / Fy)
    return (
        _Element(shape.b_tdes, shape.b_flat, shape.tdes, lam_r, 0.20, 1.38, 2),
        _Element(shape.h_tdes, shape.h_flat, shape.tdes, lam_r, 0.20, 1.38, 2),
    )


def _e7_effective_area(shape: Section, elements: tuple[_Element, ...],
                       Fy: float, Fcr: float) -> tuple[float, bool]:
    """E7 effective area given the E3 critical stress. Returns
    (Ae, any_reduction). The effective-width form is the same for every
    element type; only lambda_r and the Table E7.1 constants differ."""
    Ae = shape.A
    reduced = False
    for el in elements:
        if el.lam <= el.lam_r * math.sqrt(Fy / Fcr):
            continue                                       # E7-1: fully effective
        Fel = (el.c2 * el.lam_r / el.lam) ** 2 * Fy        # E7-5
        be = el.b * (1.0 - el.c1 * math.sqrt(Fel / Fcr)) * math.sqrt(Fel / Fcr)  # E7-3
        be = min(be, el.b)
        Ae -= el.count * (el.b - be) * el.t
        reduced = True
    return Ae, reduced


def compression_capacity(shape: Section, Fy: float, E: float,
                         KLx: float, KLy: float) -> Strength:
    """E3 flexural buckling about both axes + E7 effective-area reduction."""
    if isinstance(shape, WShape):
        elements = _w_elements(shape, Fy, E)
    elif isinstance(shape, HSSShape):
        elements = _hss_elements(shape, Fy, E)
    else:
        raise _unsupported(shape)

    klr_x = KLx / shape.rx
    klr_y = KLy / shape.ry
    if klr_y >= klr_x:
        klr, axis = klr_y, "y"
    else:
        klr, axis = klr_x, "x"
    Fcr = _e3_critical_stress(Fy, E, klr)
    Ae, reduced = _e7_effective_area(shape, elements, Fy, Fcr)
    clause = f"E7-{axis} (slender elements)" if reduced else f"E3-{axis} (flexural buckling)"
    return Strength(PHI_C * Fcr * Ae, clause)


# ---------------------------------------------------------------------------
# Flexure: W-shapes (F2/F3 major, F6 minor)
# ---------------------------------------------------------------------------

def _flb_major(shape: WShape, Fy: float, E: float) -> tuple[float, str] | None:
    """F3 compression-flange local buckling. None if the flange is compact."""
    lam = shape.bf_2tf
    lam_p = 0.38 * math.sqrt(E / Fy)
    lam_r = 1.0 * math.sqrt(E / Fy)
    if lam <= lam_p:
        return None
    Mp = Fy * shape.Zx
    if lam <= lam_r:
        Mn = Mp - (Mp - 0.7 * Fy * shape.Sx) * (lam - lam_p) / (lam_r - lam_p)  # F3-1
        return Mn, "F3-1 (noncompact flange)"
    kc = min(max(4.0 / math.sqrt(shape.h_tw), 0.35), 0.76)
    return 0.9 * E * kc * shape.Sx / lam**2, "F3-2 (slender flange)"           # F3-2


def _w_flexure_major(shape: WShape, Fy: float, E: float,
                     Lb: float, Cb: float) -> Strength:
    """F2 (yielding / LTB) capped by F3 (flange local buckling).

    Rolled W-shape webs are compact in flexure for Fy <= 65 ksi (lambda_pw =
    3.76*sqrt(E/Fy) ~ 79+, versus h/tw <= ~58 in the catalog), so F4/F5 web
    treatment is not needed; a clause note flags the (unreachable) exception.
    """
    Mp = Fy * shape.Zx

    Lp = 1.76 * shape.ry * math.sqrt(E / Fy)                                   # F2-5
    term = shape.J / (shape.Sx * shape.ho)   # c = 1 for doubly symmetric I
    Lr = 1.95 * shape.rts * (E / (0.7 * Fy)) * math.sqrt(
        term + math.sqrt(term**2 + 6.76 * (0.7 * Fy / E) ** 2))                # F2-6

    if Lb <= Lp:
        Mn, clause = Mp, "F2-1 (yielding)"
    elif Lb <= Lr:
        Mn = Cb * (Mp - (Mp - 0.7 * Fy * shape.Sx) * (Lb - Lp) / (Lr - Lp))    # F2-2
        Mn, clause = min(Mn, Mp), "F2-2 (inelastic LTB)"
    else:
        slend = Lb / shape.rts
        Fcr = Cb * math.pi**2 * E / slend**2 * math.sqrt(1.0 + 0.078 * term * slend**2)  # F2-4
        Mn, clause = min(Fcr * shape.Sx, Mp), "F2-3 (elastic LTB)"

    flb = _flb_major(shape, Fy, E)
    if flb is not None and flb[0] < Mn:
        Mn, clause = flb

    if shape.h_tw > 3.76 * math.sqrt(E / Fy):
        clause += " [web noncompact - F4 not implemented, verify manually]"

    return Strength(PHI_B * Mn, clause)


def _w_flexure_minor(shape: WShape, Fy: float, E: float) -> Strength:
    """F6: minor-axis yielding capped by flange local buckling. No LTB about
    the minor axis."""
    Mp = min(Fy * shape.Zy, 1.6 * Fy * shape.Sy)                               # F6-1
    clause = "F6-1 (yielding)"

    lam = shape.bf_2tf
    lam_p = 0.38 * math.sqrt(E / Fy)
    lam_r = 1.0 * math.sqrt(E / Fy)
    if lam > lam_p:
        if lam <= lam_r:
            Mn = Mp - (Mp - 0.7 * Fy * shape.Sy) * (lam - lam_p) / (lam_r - lam_p)  # F6-2
            clause = "F6-2 (noncompact flange)"
        else:
            Mn = (0.69 * E / lam**2) * shape.Sy                                # F6-3/F6-4
            clause = "F6-3 (slender flange)"
        Mp = min(Mp, Mn)

    return Strength(PHI_B * Mp, clause)


# ---------------------------------------------------------------------------
# Flexure: square HSS (F7, both axes)
# ---------------------------------------------------------------------------

def _hss_effective_S(I: float, A: float, D: float, t: float,
                     b: float, be: float) -> float:
    """Effective section modulus of a box section whose compression flange is
    only be wide (F7-3). The ineffective strip (b - be)*t sits at the
    compression wall, so the neutral axis shifts away from it and the extreme
    compression fiber moves further out; both are accounted for exactly.
    Origin at the gross centroid, compression side positive.
    """
    Ar = (b - be) * t                      # removed (ineffective) area
    if Ar <= 0.0:
        return I / (D / 2.0)
    Ae = A - Ar
    if Ae <= 0.0:
        return 0.0
    yr = (D - t) / 2.0                     # centroid of the compression wall
    Io = I - (Ar * yr**2 + (b - be) * t**3 / 12.0)   # about the gross centroid
    shift = Ar * yr / Ae                   # centroid moves this far in -y
    Ie = Io - Ae * shift**2                # about the effective centroid
    c_comp = D / 2.0 + shift               # extreme compression fiber
    if Ie <= 0.0 or c_comp <= 0.0:
        return 0.0
    return Ie / c_comp


def _hss_flexure(shape: HSSShape, Fy: float, E: float, axis: str) -> Strength:
    """F7 flexure of a square/rectangular HSS about one axis.

    F7.4 lateral-torsional buckling is not evaluated: it does not apply to a
    square HSS (Ix == Iy leaves nothing to buckle into) nor to bending about
    the minor axis, and this catalog is square-only. Lb therefore does not
    enter the HSS flexural capacity at all.
    """
    if axis == "x":
        Z, S, I = shape.Zx, shape.Sx, shape.Ix
        lam_f, lam_w = shape.b_tdes, shape.h_tdes
        b_f, D = shape.b_flat, shape.Ht
    else:
        Z, S, I = shape.Zy, shape.Sy, shape.Iy
        lam_f, lam_w = shape.h_tdes, shape.b_tdes
        b_f, D = shape.h_flat, shape.B
    t = shape.tdes

    Mp = Fy * Z                                                    # F7-1 yielding
    Mn, clause = Mp, "F7-1 (yielding)"

    # F7.2 flange local buckling (the walls normal to the bending axis)
    fp = 1.12 * math.sqrt(E / Fy)
    fr = 1.40 * math.sqrt(E / Fy)
    if lam_f > fp:
        if lam_f <= fr:
            Mn_f = min(Mp - (Mp - Fy * S)
                       * (3.57 * lam_f * math.sqrt(Fy / E) - 4.0), Mp)         # F7-2
            clause_f = "F7-2 (noncompact flange)"
        else:
            be = min(1.92 * t * math.sqrt(E / Fy)
                     * (1.0 - 0.38 / lam_f * math.sqrt(E / Fy)), b_f)          # F7-4
            Mn_f = Fy * _hss_effective_S(I, shape.A, D, t, b_f, be)            # F7-3
            clause_f = "F7-3 (slender flange)"
        if Mn_f < Mn:
            Mn, clause = Mn_f, clause_f

    # F7.3 web local buckling (the walls parallel to the bending axis)
    wp = 2.42 * math.sqrt(E / Fy)
    wr = 5.70 * math.sqrt(E / Fy)
    if lam_w > wp:
        Mn_w = min(Mp - (Mp - Fy * S)
                   * (0.305 * lam_w * math.sqrt(Fy / E) - 0.738), Mp)          # F7-6
        if Mn_w < Mn:
            Mn, clause = Mn_w, "F7-6 (noncompact web)"
        if lam_w > wr:
            # unreachable in this catalog (max h/tdes ~ 75, lambda_rw ~ 137)
            clause += " [web slender - F7.3(c) not implemented, verify manually]"

    return Strength(PHI_B * Mn, clause)


# ---------------------------------------------------------------------------
# Flexure: public dispatch
# ---------------------------------------------------------------------------

def flexure_major_capacity(shape: Section, Fy: float, E: float,
                           Lb: float, Cb: float = 1.0) -> Strength:
    """Major-axis flexural capacity. W-shapes: F2 capped by F3. Square HSS:
    F7, for which Lb and Cb are inert (no LTB limit state)."""
    if isinstance(shape, WShape):
        return _w_flexure_major(shape, Fy, E, Lb, Cb)
    if isinstance(shape, HSSShape):
        return _hss_flexure(shape, Fy, E, "x")
    raise _unsupported(shape)


def flexure_minor_capacity(shape: Section, Fy: float, E: float) -> Strength:
    """Minor-axis flexural capacity. W-shapes: F6. Square HSS: F7 about y,
    which equals the major-axis result for a square."""
    if isinstance(shape, WShape):
        return _w_flexure_minor(shape, Fy, E)
    if isinstance(shape, HSSShape):
        return _hss_flexure(shape, Fy, E, "y")
    raise _unsupported(shape)


# ---------------------------------------------------------------------------
# Shear
# ---------------------------------------------------------------------------

def _w_shear(shape: WShape, Fy: float, E: float) -> Strength:
    """G2 web shear with the real web area d*tw (not an area fraction guess).
    phi_v = 1.00 with Cv1 = 1.0 for stocky rolled-I webs (G2.1(a)); otherwise
    phi_v = 0.90 with Cv1 per G2-3/G2-4 (kv = 5.34, unstiffened web)."""
    Aw = shape.d * shape.tw
    if shape.h_tw <= 2.24 * math.sqrt(E / Fy):
        return Strength(1.00 * 0.6 * Fy * Aw, "G2-1 (phi=1.0, Cv1=1.0)")
    kv = 5.34
    limit = 1.10 * math.sqrt(kv * E / Fy)
    Cv1 = 1.0 if shape.h_tw <= limit else limit / shape.h_tw
    return Strength(0.90 * 0.6 * Fy * Aw * Cv1, "G2 (slender web)")


def _hss_shear(shape: HSSShape, Fy: float, E: float) -> Strength:
    """G4 shear on the two walls parallel to the major-axis shear force.
    Aw = 2*h*tdes with h the flat wall width; kv = 5 and Cv2 per G2.2."""
    Aw = 2.0 * shape.h_flat * shape.tdes
    lam = shape.h_tdes
    kv = 5.0
    root = math.sqrt(kv * E / Fy)
    if lam <= 1.10 * root:
        Cv2, note = 1.0, "G4 (Cv2=1.0)"                                # G2-9
    elif lam <= 1.37 * root:
        Cv2, note = 1.10 * root / lam, "G4 (inelastic wall buckling)"  # G2-10
    else:
        Cv2, note = 1.51 * kv * E / (lam**2 * Fy), "G4 (elastic wall buckling)"  # G2-11
    return Strength(PHI_V_HSS * 0.6 * Fy * Aw * Cv2, note)


def shear_capacity(shape: Section, Fy: float, E: float) -> Strength:
    """Major-axis shear capacity. W-shapes: G2. Square HSS: G4."""
    if isinstance(shape, WShape):
        return _w_shear(shape, Fy, E)
    if isinstance(shape, HSSShape):
        return _hss_shear(shape, Fy, E)
    raise _unsupported(shape)


# ---------------------------------------------------------------------------
# Interaction
# ---------------------------------------------------------------------------

def interaction_h1(Pu: float, Mux: float, Muy: float,
                   phi_Pc: float, phi_Mcx: float, phi_Mcy: float) -> tuple[float, str]:
    """H1-1 combined axial + biaxial flexure. Pu signed (tension positive);
    the caller supplies phi_Pc consistent with the sign of Pu."""
    p = abs(Pu) / max(phi_Pc, 1e-12)
    mx = abs(Mux) / max(phi_Mcx, 1e-12)
    my = abs(Muy) / max(phi_Mcy, 1e-12)
    if p >= 0.2:
        return p + (8.0 / 9.0) * (mx + my), "H1-1a"
    return p / 2.0 + (mx + my), "H1-1b"
