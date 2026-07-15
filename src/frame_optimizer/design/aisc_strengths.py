"""AISC 360-16/22 LRFD strength equations for rolled W-shapes.

Pure functions: WShape + material + lengths in, (phi*Rn, clause) out.
Units: kips, inches, ksi.

Implemented limit states:
    D2  tension yielding (rupture not checked - no connection geometry)
    E3  flexural buckling in compression (both axes, K = 1 pin-pin)
    E7  slender-element compression (effective areas; many light W-shapes
        have slender webs under uniform compression at Fy = 50)
    F2  major-axis flexure: yielding + lateral-torsional buckling
    F3  major-axis flexure: compression-flange local buckling
    F6  minor-axis flexure: yielding + flange local buckling
    G2  web shear (Cv1; phi_v = 1.0 for stocky rolled webs)
    H1  combined axial + flexure interaction
"""
from __future__ import annotations

import math
from typing import NamedTuple

from ..sections import WShape

PHI_C = 0.90   # compression
PHI_T = 0.90   # tension yielding
PHI_B = 0.90   # flexure


class Strength(NamedTuple):
    phi_Rn: float
    clause: str


# ---------------------------------------------------------------------------
# Axial
# ---------------------------------------------------------------------------

def tension_capacity(shape: WShape, Fy: float) -> Strength:
    """D2 yielding on the gross section. Rupture (D2-2) needs net-area /
    connection data, which a member-level optimizer doesn't have; yielding
    governs for members without significant section loss."""
    return Strength(PHI_T * Fy * shape.A, "D2 (tension yielding)")


def _e3_critical_stress(Fy: float, E: float, klr: float) -> float:
    """E3 flexural-buckling stress for one axis."""
    if klr < 1e-9:
        return Fy
    Fe = math.pi**2 * E / klr**2
    if Fy / Fe <= 2.25:
        return 0.658 ** (Fy / Fe) * Fy   # E3-2 (inelastic)
    return 0.877 * Fe                    # E3-3 (elastic)


def _e7_effective_area(shape: WShape, Fy: float, E: float, Fcr: float) -> tuple[float, bool]:
    """E7 effective area given the E3 critical stress.

    W-shape elements (Table B4.1a): each half-flange is an unstiffened element
    (lambda = bf/2tf, four outstands total), the web is a stiffened element
    (lambda = h/tw). Returns (Ae, any_reduction).
    """
    Ae = shape.A
    reduced = False
    # (slenderness, full width b, thickness, lambda_r, c1, c2, count)
    elements = (
        (shape.bf_2tf, shape.bf / 2.0, shape.tf, 0.56 * math.sqrt(E / Fy), 0.22, 1.49, 4),
        (shape.h_tw, shape.h_tw * shape.tw, shape.tw, 1.49 * math.sqrt(E / Fy), 0.18, 1.31, 1),
    )
    for lam, b, t, lam_r, c1, c2, count in elements:
        if lam <= lam_r * math.sqrt(Fy / Fcr):
            continue                                    # E7-1: fully effective
        Fel = (c2 * lam_r / lam) ** 2 * Fy              # E7-5
        be = b * (1.0 - c1 * math.sqrt(Fel / Fcr)) * math.sqrt(Fel / Fcr)  # E7-3
        be = min(be, b)
        Ae -= count * (b - be) * t
        reduced = True
    return Ae, reduced


def compression_capacity(shape: WShape, Fy: float, E: float,
                         KLx: float, KLy: float) -> Strength:
    """E3 flexural buckling about both axes + E7 effective-area reduction.
    Torsional buckling (E4) does not govern doubly symmetric W-shapes with
    KLz = KLy, so it is not evaluated."""
    klr_x = KLx / shape.rx
    klr_y = KLy / shape.ry
    if klr_y >= klr_x:
        klr, axis = klr_y, "y"
    else:
        klr, axis = klr_x, "x"
    Fcr = _e3_critical_stress(Fy, E, klr)
    Ae, reduced = _e7_effective_area(shape, Fy, E, Fcr)
    clause = f"E7-{axis} (slender elements)" if reduced else f"E3-{axis} (flexural buckling)"
    return Strength(PHI_C * Fcr * Ae, clause)


# ---------------------------------------------------------------------------
# Flexure
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


def flexure_major_capacity(shape: WShape, Fy: float, E: float,
                           Lb: float, Cb: float = 1.0) -> Strength:
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


def flexure_minor_capacity(shape: WShape, Fy: float, E: float) -> Strength:
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
# Shear
# ---------------------------------------------------------------------------

def shear_capacity(shape: WShape, Fy: float, E: float) -> Strength:
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
