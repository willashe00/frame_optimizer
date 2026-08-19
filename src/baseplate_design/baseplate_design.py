#!/usr/bin/env python3
"""
Steel Column Baseplate Auto-Design
AISC 360-22 + AISC Design Guide 1 (2nd Ed.)

Scope: compression + shear on a pinned base, with F1554 Gr. 36 anchor rods
(Fu = 58 ksi typical). Plate dimensions, plate thickness and anchor rod
diameter are all solved from the governing loads and the column size.

Methodology:
  1. Plate sizing uses the equal-cantilever optimum (m = n) per DG1 Sec. 3.3,
     then holds the result to the detailing minimums that make the plate
     buildable: the plate must project past the column, and the anchor rods
     must sit clear of the flanges with a real edge distance.
  2. A1,req is the closed-form governing minimum bearing area, with the
     confinement factor sqrt(A2/A1) capped at 2.0. A2 is the largest area
     geometrically similar to and concentric with the plate that fits on the
     pier, so it tracks the plate size -- plate and A2 are solved together by
     a short ratcheting fixed-point iteration.
  3. B, N round UP to the plate increment (1/2 in); tp UP to the thickness
     increment (1/8 in, 1/2 in floor); d_rod UP to the rod increment (1/8 in,
     3/4 in floor).
  4. Rod shear sizing credits friction (phi*mu*P), where P is the MINIMUM
     compression coincident with the shear (0.9D), not the maximum -- the
     maximum is unconservative for this check. The remainder is distributed
     to all rods at Fnv = 0.45*Fu (AISC Table J3.2, threads in shear plane).

This module designs and checks ONE plate. `uniform_design.py` drives it over
every column of an optimized frame to land on a single plate for all of them.

NOT included: rod tension, concrete breakout/pryout (ACI 318 Ch. 17), weld
design, moment cases (e = Mu/Pu, DG1 Sec. 3.4). These cases are out of scope.

Units: this module is internally consistent in kips and inches, the system
AISC 360 and DG1 are written in. SI conversion happens at the BaseplateConfig
boundary (config.py) and in the JSON export (export.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import BaseplateConfig

# Fixed code constants (LRFD)
PHI_C = 0.65     # bearing on concrete resistance factor (AISC 360-22 J8)
PHI_B = 0.90     # plate flexure resistance factor (AISC 360-22 F1)
PHI_V = 0.75     # anchor shear resistance factor (AISC 360-22 J3.7)
FNV_C = 0.45     # Fnv = 0.45*Fu (threads in shear plane, AISC Table J3.2)
MAX_CONFINEMENT = 2.0   # cap on sqrt(A2/A1), AISC 360-22 Eq. J8-2

STEEL_DENSITY_PCI = 0.490 / 12.0 ** 3   # kip/in^3 (490 pcf), for plate weight

# The plate/A2 fixed point closes in 2-3 passes; 25 leaves a wide margin and
# still fails loudly if an explicit pedestal is too small for the plate.
_MAX_SIZING_ITERATIONS = 25


# A value sitting exactly on an increment can land a few ulps above it, which
# math.ceil would charge a whole extra increment for. Snap before rounding.
_SNAP_TOL = 1e-9


def ceil_to(x: float, step: float) -> float:
    """this function rounds up to nearest step

    Both arguments arrive from SI conversions that are exact in mm but not in
    binary floating point (a 1/2 in increment is 12.7 mm, and 12.7/25.4 is
    0.49999999999999994), so a dimension already on the increment must not be
    rounded up again -- that would silently buy half an inch of plate on every
    column.
    """
    n = x / step
    nearest = round(n)
    if abs(n - nearest) <= _SNAP_TOL * max(1.0, abs(n)):
        n = float(nearest)
    return math.ceil(n) * step


@dataclass(frozen=True)
class ColumnDemand:
    """One column's baseplate demands, in the internal kip/inch system.

    Pu drives bearing and plate flexure; P_friction drives the shear-friction
    credit and is deliberately a DIFFERENT (smaller) axial force -- see the
    module docstring, item 4.
    """
    column_id: str
    d: float            # in    column depth (W-shape overall depth)
    bf: float           # in    column flange width
    Pu: float           # kip   governing LRFD axial compression (+ = compression)
    Vu: float           # kip   design base shear
    P_friction: float   # kip   minimum compression coincident with Vu
    section_name: str = ""
    base_node: str = ""
    location_in: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.d <= 0 or self.bf <= 0:
            raise ValueError(f"{self.column_id}: column d and bf must be positive.")
        if self.Pu < 0 or self.Vu < 0 or self.P_friction < 0:
            raise ValueError(
                f"{self.column_id}: Pu, Vu and P_friction must be >= 0 "
                "(net uplift is out of scope for this pinned-base module)."
            )


@dataclass(frozen=True)
class PlateGeometry:
    """A physical baseplate: the numbers a fabricator needs. Inches."""
    B: float            # plate width,  parallel to the column flange width bf
    N: float            # plate length, parallel to the column depth d
    tp: float           # plate thickness
    d_rod: float        # anchor rod diameter
    n_rods: int         # anchor rods per column base
    edge_distance: float  # rod centerline to plate edge

    @property
    def area(self) -> float:
        """Bearing area A1, in^2."""
        return self.B * self.N

    @property
    def weight_kip(self) -> float:
        return self.area * self.tp * STEEL_DENSITY_PCI

    def rod_positions(self) -> list[tuple[float, float]]:
        """Anchor rod (x, y) offsets from the plate center, in inches.

        x runs along B, y runs along N. Half the rods sit outside each column
        flange at the edge distance, which is what N_min guarantees room for;
        within a row they are spread evenly across the available width.
        """
        per_row = self.n_rods // 2
        y = self.N / 2.0 - self.edge_distance
        x_max = self.B / 2.0 - self.edge_distance
        if per_row == 1:
            xs = [0.0]
        else:
            step = 2.0 * x_max / (per_row - 1)
            xs = [-x_max + i * step for i in range(per_row)]
        return [(x, sy * y) for sy in (-1.0, 1.0) for x in xs]


@dataclass(frozen=True)
class BaseplateCheck:
    """Every limit state of one plate under one column's demands. kip/inch."""
    column_id: str
    # bearing (AISC 360-22 Eq. J8-2)
    A1: float
    A2_effective: float
    confinement: float
    phiPp: float
    bearing_dcr: float
    # plate flexure (DG1 Eq. 3.3.17)
    m: float
    n: float
    lam: float
    lam_np: float
    ell: float
    t_req: float
    flexure_dcr: float
    # anchor rod shear (AISC 360-22 J3.7 + shear friction)
    friction: float
    V_rods: float
    phiRnv: float
    phiVn: float
    shear_dcr: float
    # detailing
    rod_clearance: float   # clear distance, rod centerline to flange face
    clearance_ok: bool

    @property
    def bearing_ok(self) -> bool:
        return self.bearing_dcr <= 1.0

    @property
    def flexure_ok(self) -> bool:
        return self.flexure_dcr <= 1.0

    @property
    def shear_ok(self) -> bool:
        return self.shear_dcr <= 1.0

    @property
    def passes(self) -> bool:
        return (self.bearing_ok and self.flexure_ok and self.shear_ok
                and self.clearance_ok)

    @property
    def max_dcr(self) -> float:
        return max(self.bearing_dcr, self.flexure_dcr, self.shear_dcr)

    @property
    def governing_limit_state(self) -> str:
        states = {"bearing": self.bearing_dcr, "plate flexure": self.flexure_dcr,
                  "anchor rod shear": self.shear_dcr}
        return max(states, key=states.get)


# ---------------------------------------------------------------------------
# concrete bearing area
# ---------------------------------------------------------------------------
def effective_A2(B: float, N: float, config: BaseplateConfig) -> float:
    """A2 for AISC 360-22 J8: the largest area geometrically similar to and
    concentric with the plate that still fits on the pier (in^2).

    Similarity is what J8 requires, so a pier that is generous in one
    direction and tight in the other is governed by the tight one: scaling
    the plate by k = min(Bp/B, Np/N) gives A2 = k^2 * A1.
    """
    pedestal = config.pedestal_in
    if pedestal is None:
        c = config.pedestal_edge_projection_in
        pedestal = (B + 2.0 * c, N + 2.0 * c)
    k = min(pedestal[0] / B, pedestal[1] / N)
    if k < 1.0:
        raise ValueError(
            f"Baseplate {B:.1f} x {N:.1f} in does not fit on the "
            f"{pedestal[0]:.1f} x {pedestal[1]:.1f} in pier. Enlarge "
            "pedestal_width_mm / pedestal_length_mm, or raise fc_mpa so a "
            "smaller plate works."
        )
    return k * k * B * N


def required_A1(Pu: float, fc: float, A2: float) -> float:
    """Minimum bearing area for Pu (in^2), AISC 360-22 Eq. J8-2 inverted.

    Two branches, whichever needs more area: the confinement factor capped at
    2.0, and the uncapped phiPp = phi*0.85*fc*sqrt(A1*A2) form.
    """
    q = PHI_C * 0.85 * fc
    return max(Pu / (q * MAX_CONFINEMENT), Pu ** 2 / (q ** 2 * A2))


# ---------------------------------------------------------------------------
# limit-state checks of a given plate
# ---------------------------------------------------------------------------
def check_plate(plate: PlateGeometry, demand: ColumnDemand,
                config: BaseplateConfig) -> BaseplateCheck:
    """Check one plate against one column's demands. This is the single source
    of truth for every limit state; the sizing routines below invert it."""
    A1 = plate.area
    A2 = effective_A2(plate.B, plate.N, config)
    conf = min(math.sqrt(A2 / A1), MAX_CONFINEMENT)
    phiPp = PHI_C * 0.85 * config.fc_ksi * A1 * conf
    bearing_dcr = demand.Pu / phiPp

    # cantilever projections and the DG1 lambda*n' yield-line term
    m = (plate.N - 0.95 * demand.d) / 2.0
    n = (plate.B - 0.80 * demand.bf) / 2.0
    X = (4.0 * demand.d * demand.bf / (demand.d + demand.bf) ** 2) * bearing_dcr
    lam = 1.0 if X >= 1.0 else min(2.0 * math.sqrt(X) / (1.0 + math.sqrt(1.0 - X)), 1.0)
    lam_np = lam * math.sqrt(demand.d * demand.bf) / 4.0
    ell = max(m, n, lam_np)
    t_req = ell * math.sqrt(2.0 * demand.Pu / (PHI_B * config.plate_Fy_ksi * A1))
    flexure_dcr = t_req / plate.tp

    friction = PHI_V * config.friction_coefficient * demand.P_friction
    V_rods = max(0.0, demand.Vu - friction)
    A_rod = math.pi * plate.d_rod ** 2 / 4.0
    phiRnv = PHI_V * FNV_C * config.anchor_Fu_ksi * A_rod * plate.n_rods
    phiVn = friction + phiRnv                    # DG1 Sec. 3.5: friction + rods
    shear_dcr = demand.Vu / phiVn if phiVn > 0.0 else (0.0 if demand.Vu == 0.0
                                                      else math.inf)

    # rods must land clear of the flanges to be installable
    clear = (plate.N / 2.0 - plate.edge_distance) - demand.d / 2.0
    return BaseplateCheck(
        column_id=demand.column_id,
        A1=A1, A2_effective=A2, confinement=conf, phiPp=phiPp,
        bearing_dcr=bearing_dcr,
        m=m, n=n, lam=lam, lam_np=lam_np, ell=ell, t_req=t_req,
        flexure_dcr=flexure_dcr,
        friction=friction, V_rods=V_rods, phiRnv=phiRnv, phiVn=phiVn,
        shear_dcr=shear_dcr,
        rod_clearance=clear,
        clearance_ok=clear >= config.rod_clearance_in - 1e-9,
    )


# ---------------------------------------------------------------------------
# sizing (inverting each limit state)
# ---------------------------------------------------------------------------
def _rod_diameter(demand: ColumnDemand, config: BaseplateConfig,
                  floor: float = 0.0) -> float:
    """Anchor rod diameter for the shear left over after the friction credit."""
    friction = PHI_V * config.friction_coefficient * demand.P_friction
    V_rods = max(0.0, demand.Vu - friction)
    Ab_req = V_rods / (config.n_rods * PHI_V * FNV_C * config.anchor_Fu_ksi)
    d_calc = math.sqrt(4.0 * Ab_req / math.pi)
    return max(config.min_rod_diameter_in, floor,
               ceil_to(d_calc, config.rod_increment_in))


def _edge_distance(d_rod: float, config: BaseplateConfig) -> float:
    """Rod centerline to plate edge: the detailing minimum, but never less
    than a multiple of the rod diameter."""
    return max(config.min_edge_distance_in, config.edge_distance_rod_factor * d_rod)


def _plan_dimensions(A1_req: float, demand: ColumnDemand, edge: float,
                     config: BaseplateConfig) -> tuple[float, float]:
    """B and N for a required bearing area (DG1 Sec. 3.3 equal-cantilever
    optimum), held to the detailing minimums and rounded up to plate stock.

    The closed form is exact: with delta = 0.95d - 0.80bf, B*(B + delta) =
    A1_req and N - B = delta, which is m = n.
    """
    delta = 0.95 * demand.d - 0.80 * demand.bf
    B_opt = (-delta + math.sqrt(delta ** 2 + 4.0 * A1_req)) / 2.0

    # minimums: cover the column with a real projection, and leave room for a
    # rod outside each flange at its edge distance
    B_min = max(0.80 * demand.bf + 2.0 * edge,
                demand.bf + 2.0 * config.min_plate_projection_in)
    N_min = max(0.95 * demand.d + 2.0 * edge,
                demand.d + 2.0 * config.min_plate_projection_in,
                demand.d + 2.0 * (config.rod_clearance_in + edge))

    step = config.plate_increment_in
    B = ceil_to(max(B_opt, B_min), step)
    N = ceil_to(max(B_opt + delta, A1_req / B, N_min), step)
    return B, N


def design_plate(demand: ColumnDemand, config: BaseplateConfig,
                 floor: PlateGeometry | None = None) -> PlateGeometry:
    """Smallest buildable plate that carries `demand`, never below `floor`.

    `floor` is what lets one plate be grown to cover a whole set of columns
    (see uniform_design.py): every dimension is held at or above it.

    Bearing area and A2 are mutually dependent -- a bigger plate sits on a
    bigger pier, which raises the confinement factor, which reduces the
    required area. The loop below ratchets (dimensions only ever grow), so it
    terminates on the quantized plate increments and ends up self-consistent:
    the A2 finally used is the A2 the final plate actually gets.
    """
    d_rod = _rod_diameter(demand, config, floor.d_rod if floor else 0.0)
    edge = _edge_distance(d_rod, config)

    B, N = _plan_dimensions(0.0, demand, edge, config)   # detailing minimum
    if floor is not None:
        B, N = max(B, floor.B), max(N, floor.N)

    for _ in range(_MAX_SIZING_ITERATIONS):
        A1_req = required_A1(demand.Pu, config.fc_ksi, effective_A2(B, N, config))
        B_new, N_new = _plan_dimensions(A1_req, demand, edge, config)
        B_new, N_new = max(B, B_new), max(N, N_new)
        if (B_new, N_new) == (B, N):
            break
        B, N = B_new, N_new
    else:
        raise RuntimeError(
            f"{demand.column_id}: plate/A2 sizing did not converge at "
            f"{B:.1f} x {N:.1f} in. The pier is likely too small to develop "
            "the bearing area this load needs."
        )

    # thickness follows from the final plan size: check_plate computes the
    # required thickness for any tp, so size against a unit plate and round up
    probe = PlateGeometry(B, N, 1.0, d_rod, config.n_rods, edge)
    t_req = check_plate(probe, demand, config).t_req
    tp = max(config.min_thickness_in, floor.tp if floor else 0.0,
             ceil_to(t_req, config.thickness_increment_in))
    return PlateGeometry(B=B, N=N, tp=tp, d_rod=d_rod, n_rods=config.n_rods,
                         edge_distance=edge)


# ---------------------------------------------------------------------------
# standalone demo: the module still runs on its own with hand-entered loads
# ---------------------------------------------------------------------------
def design(Pu: float = 350.0, Vu: float = 25.0, d: float = 10.1,
           bf: float = 10.0, P_friction: float | None = None,
           config: BaseplateConfig | None = None) -> PlateGeometry:
    """Design and report one baseplate from hand-entered kip/inch loads.

    Kept so this file stays runnable by itself; the frame-driven path is
    `uniform_design.design_uniform_baseplate()`.
    """
    config = config or BaseplateConfig()
    demand = ColumnDemand(
        column_id="(hand-entered)", d=d, bf=bf, Pu=Pu, Vu=Vu,
        P_friction=Pu if P_friction is None else P_friction,
    )
    plate = design_plate(demand, config)
    check = check_plate(plate, demand, config)

    print("=" * 56)
    print(" STEEL COLUMN BASEPLATE - DESIGN SUMMARY")
    print(" AISC 360-22 + AISC DG1 (2nd Ed.)")
    print("=" * 56)
    print(f" Inputs:  Pu = {demand.Pu:.0f} kips   Vu = {demand.Vu:.0f} kips")
    print(f"          Column {d:g} x {bf:g} in   f'c = {config.fc_ksi:.2f} ksi"
          f"   Fy = {config.plate_Fy_ksi:.1f} ksi")
    print("-" * 56)
    print(f" Plate          : {plate.B:g} in x {plate.N:g} in x {plate.tp:g} in")
    print(f" Anchor rods    : {plate.n_rods} - {plate.d_rod:g} in dia.")
    print(f" Bearing area A1: {check.A1:g} in^2  "
          f"(A2,eff = {check.A2_effective:.1f} in^2)")
    print("-" * 56)
    print(f" Bearing DCR    : {check.bearing_dcr:.3f}   "
          f"{'PASS' if check.bearing_ok else 'FAIL'}")
    print(f" Flexure DCR    : {check.flexure_dcr:.3f}   "
          f"{'PASS' if check.flexure_ok else 'FAIL'}")
    print(f" Shear DCR      : {check.shear_dcr:.3f}   "
          f"{'PASS' if check.shear_ok else 'FAIL'}")
    print("-" * 56)
    print(f" GOVERNING STATUS: "
          f"{'DESIGN ADEQUATE' if check.passes else 'REVISE INPUTS'}")
    print("=" * 56)
    return plate


if __name__ == "__main__":
    design()
