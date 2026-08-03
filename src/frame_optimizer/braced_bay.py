"""Closed-form design mechanics of the longitudinal LFRS: tension-only
X-braced sidewall bays, collectors, and the end-bay roof truss.

Why closed-form: a tension-only X-braced panel is statically determinate —
under a given story shear the active diagonal's tension, the strut forces,
and the column couple follow from statics alone, with no dependence on
member stiffness. That makes hand-verifiable design values (this module)
strictly better than a 3D solve for this subsystem, and sidesteps the
mechanism-stabilization DZ restraints of the gravity model, which would
silently absorb applied longitudinal loads. (The same determinacy argument
underpins the gravity model's representative strip.)

Load path and distribution (single-story, all lateral force at the roof
line; diaphragm = rod-braced roof, i.e. FLEXIBLE, so tributary
distribution applies):

* Wind z: the end walls span vertically base-to-roof, so HALF the wall
  wind reaches the roof plane (the rest goes straight into the end-wall
  column bases). The roof-level force enters at the two ends: the windward
  end carries its Cp share, the leeward end its suction share (computed
  from the actual pressure split, not assumed 50/50), each delivered by
  that end's roof-truss bay to the two sidewalls and dragged by the eave
  strut into the NEAREST braced bay.
* Seismic z: the roof-line inertia is distributed along the length, so
  each braced bay takes the tributary length between midpoints to its
  neighboring braced bays (V_line = V/2 per sidewall).
* Every braced bay is designed for the worst of its wind and seismic
  share; both carry factor 1.0 in the governing LRFD combos, so the
  member-force maximum is the design value.

Per-bay statics (bay width b, wall height H, `tiers` stacked X panels,
story shear V constant over height):

* active diagonal tension  T = V * L_d / b     (L_d = diagonal length)
* intermediate wall strut  P = -/+ V           (full shear transfer)
* column axial couple      dP = -/+ V * H / b  (overturning, at the base)
* foundation, braced bay:  FZ = V at the base of the loaded corner; the
  tier-1 diagonal also hangs V * (H/tiers) / b of uplift on its anchor.

Elastic drift by virtual work over the determinate truss (unit load at the
roof line; diagonals + intermediate struts + column axial):

  delta = V * [ tiers * L_d^3 / (b^2 E A_d)
                + (tiers - 1) * b / (E A_s)
                + 2 H^3 / (3 b^2 E A_c) ]

(the column term integrates the linearly varying overturning axial over
both columns). Seismic drift is amplified by Cd/Ie per ASCE 7-16 12.8.6;
wind drift uses the ~10-year (0.7V)^2 factor, both checked by the designer.

End-bay roof truss (horizontal, span = building width, depth = the end
bay s_f, chords = the two bounding girder/top-chord lines, posts = the
purlins, diagonals = the roof braces): with its end-wall load spread along
the span, max panel shear is half the entering force at the sidewall
supports, diagonal tension scales by L_d/s_f, and the mid-span chord force
is M/s_f = W*span/(8*s_f) — added to the bounding girders'/chords' axial
recheck by the designer.

Units: ft and kips at the interface; E in ksi with areas in in^2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import FT
from .lateral_loads import WindPressures


@dataclass(frozen=True)
class BracedBayForces:
    """Design actions of one braced bay under story shear V_kip."""
    V_kip: float
    bay_ft: float
    wall_height_ft: float
    tiers: int
    diag_len_ft: float
    diag_tension_kip: float     # active diagonal, per tier (equal all tiers)
    strut_axial_kip: float      # +/- at every intermediate tier level
    column_couple_kip: float    # +/- overturning axial at the column bases
    anchor_uplift_kip: float    # vertical pull of the tier-1 diagonal anchor
    base_shear_kip: float       # horizontal reaction at the braced bay


def bay_forces(V_kip: float, bay_ft: float, wall_height_ft: float,
               tiers: int) -> BracedBayForces:
    tier_h = wall_height_ft / tiers
    diag_len = math.hypot(bay_ft, tier_h)
    return BracedBayForces(
        V_kip=V_kip, bay_ft=bay_ft, wall_height_ft=wall_height_ft,
        tiers=tiers, diag_len_ft=diag_len,
        diag_tension_kip=V_kip * diag_len / bay_ft,
        strut_axial_kip=V_kip,
        column_couple_kip=V_kip * wall_height_ft / bay_ft,
        anchor_uplift_kip=V_kip * tier_h / bay_ft,
        base_shear_kip=V_kip,
    )


def bay_drift_in(V_kip: float, bay_ft: float, wall_height_ft: float,
                 tiers: int, A_diag_in2: float, A_strut_in2: float,
                 A_col_in2: float, E_ksi: float) -> float:
    """Roof-line drift of one braced bay (virtual work, tension-only)."""
    b = bay_ft * FT
    H = wall_height_ft * FT
    L_d = math.hypot(b, H / tiers)
    delta = tiers * L_d**3 / (b**2 * E_ksi * A_diag_in2)
    if tiers > 1:
        delta += (tiers - 1) * b / (E_ksi * A_strut_in2)
    delta += 2.0 * H**3 / (3.0 * b**2 * E_ksi * A_col_in2)
    return V_kip * delta


# ---------------------------------------------------------------------------
# Share of the line shear per braced bay
# ---------------------------------------------------------------------------

def windward_fraction(wp: WindPressures) -> float:
    """Windward wall's share of the total wind force (the rest is leeward
    suction) — sets how much of the roof-level z-wind enters at each end."""
    ww = sum(b.pressure_psf * (b.to_ft - b.from_ft) for b in wp.windward_bands)
    lw = abs(wp.leeward_psf) * wp.h_ft
    return ww / (ww + lw)


def roof_level_line_shear_kip(wp: WindPressures) -> float:
    """Roof-level z-wind per sidewall line: half the (possibly 27.1.5-
    floored) base shear reaches the roof plane, split between two lines."""
    return wp.base_shear_kip / 2.0 / 2.0


def wind_bay_shares_kip(wp: WindPressures, braced: list[int],
                        n_bays: int) -> dict[int, float]:
    """Wind z story shear per braced bay on ONE sidewall line: the windward
    end's share goes to the braced bay nearest bay 0, the leeward share to
    the one nearest the far end (flexible diaphragm, load enters at the
    ends)."""
    V_line = roof_level_line_shear_kip(wp)
    f_ww = windward_fraction(wp)
    shares = {b: 0.0 for b in braced}
    shares[min(braced)] += f_ww * V_line
    shares[max(braced)] += (1.0 - f_ww) * V_line
    return shares


def seismic_bay_shares_kip(V_line_kip: float, braced: list[int],
                           n_bays: int, bay_ft: float) -> dict[int, float]:
    """Seismic story shear per braced bay on one line: tributary length
    between midpoints to the neighboring braced bays (flexible diaphragm,
    inertia distributed along the length)."""
    length = n_bays * bay_ft
    centers = [(b + 0.5) * bay_ft for b in braced]
    bounds = [0.0] + [(a + c) / 2.0 for a, c in zip(centers, centers[1:])] + [length]
    return {b: V_line_kip * (hi - lo) / length
            for b, lo, hi in zip(braced, bounds[:-1], bounds[1:])}


def design_bay_shears_kip(wind_shares: dict[int, float],
                          seismic_shares: dict[int, float]) -> dict[int, float]:
    """Per-bay design story shear: worst of wind and seismic (both carry
    factor 1.0 in their governing combos)."""
    return {b: max(wind_shares[b], seismic_shares[b]) for b in wind_shares}


# ---------------------------------------------------------------------------
# End-bay roof truss
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoofTrussForces:
    entering_force_kip: float    # worst end's roof-level wind
    diag_tension_kip: float      # worst diagonal (support panel)
    chord_axial_kip: float       # +/- on the bounding girder/chord lines


def roof_truss_forces(wp_z: WindPressures, span_ft: float,
                      bay_ft: float, panel_w_ft: float) -> RoofTrussForces:
    """Design forces of one end-bay roof truss under its end-wall wind.

    W = the worse end's roof-level share (windward Cp share governs),
    treated as uniformly distributed along the end wall. Max panel shear
    W/2 at the sidewall supports; T = shear * L_d / depth with depth = the
    bay width s_f; chord force = W*span/(8*s_f). Seismic never governs
    these members: the roof-level end-tributary inertia is far below the
    end-wall wind for any realistic building, and wind is what this truss
    exists for."""
    V_roof_total = wp_z.base_shear_kip / 2.0          # roof-level, both lines
    W = max(windward_fraction(wp_z), 1.0 - windward_fraction(wp_z)) * V_roof_total
    diag_len = math.hypot(panel_w_ft, bay_ft)
    return RoofTrussForces(
        entering_force_kip=W,
        diag_tension_kip=(W / 2.0) * diag_len / bay_ft,
        chord_axial_kip=W * span_ft / (8.0 * bay_ft),
    )
