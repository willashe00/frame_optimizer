"""ASCE 7-16 lateral loads for the clear-span building: wind MWFRS
(directional procedure) + seismic ELF, and the LRFD combinations that pair
them with the existing gravity cases.

Scope and assumptions (v1):

* Building idealization: enclosed rectangular box, flat roof (theta < 10
  deg), no parapets, no topographic speed-up unless Kzt is given. Walls
  extend to the roof plane — the eave height for wide-flange buildings, the
  top-chord level for truss buildings.
* Wind: Chapter 27 directional procedure for the MWFRS. qz = 0.00256 Kz Kzt
  Kd Ke V^2 (26.10-1) with Kd = 0.85 (Table 26.6-1), gust factor G = 0.85
  (rigid building, 26.11 — these stocky single-story frames are well above
  1 Hz), enclosed GCpi = +/-0.18 (Table 26.13-1). Wall/roof Cp from Figure
  27.3-1: windward +0.8 with qz, leeward by L/B with qh, side walls -0.7,
  roof suction bands from the windward edge interpolated on h/L (the
  optional 0.8x area reduction on the -1.3 zone is NOT taken —
  conservative). Internal pressure cancels on the frame's windward+leeward
  pair, so line shears use external pressures; roof net uplift includes
  +GCpi (internal pressure has no opposing surface there — slab on grade).
  The 27.1.5 minimum (16 psf on the projected wall area; the roof projects
  to nothing on a vertical plane for a flat roof) floors each direction's
  base shear.
* Seismic: Section 12.8 ELF. Single-story, so the whole base shear acts at
  the roof line. Default R = 3 ("steel systems not specifically detailed
  for seismic resistance", Table 12.2-1 item H.1) — permitted in SDC B/C
  only, which is exactly the v1 scope: SDC D and above raises
  UnsupportedSDCError instead of producing an unlicensed design. SDC A
  needs only the Section 1.4.2 minimum (Cs = 0.01). Ta = Ct*hn^x with the
  'all other systems' coefficients by default; rho = 1.0 (SDC A-C,
  12.3.4.1). The effective seismic weight takes the optimized steel weight
  + superimposed dead over the roof + the upper half of the wall cladding;
  roof live load is excluded per 12.7.2 (set include_live_fraction = 0.2
  when live_psf is actually a flat-roof snow load over 30 psf).
* Load cases are named Wx+/-, Wz+/-, Ex+/-, Ez+/- : x is the clear-span
  (transverse frame) direction, z the building length (longitudinal braced
  bays); the sign is the loading direction. Combinations follow 2.3.1 and
  2.3.6 with the vertical seismic effect folded into the D factor
  ((1.2 + 0.2 SDS)D etc.) and the roof live/snow lumped into the existing
  'L' case exactly as the gravity module does.

This module produces LOADS and COMBINATIONS only — structured, unit-tagged
data for the load-application/analysis phase — plus the printable
lateral_load_basis report. Nothing here touches Pynite.

Interface units: ft, psf, mph, kip. West longitudes negative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .analysis.frame_model import STRENGTH_COMBOS
from .clear_span import ClearSpanConfig
from .config import TRUSS
from .site import SiteConfig, SeismicHazard, SiteHazards

# --- wind constants (ASCE 7-16) ---
# Table 26.11-1 terrain exposure constants: {exposure: (alpha, zg_ft)}
EXPOSURE_CONSTANTS = {"B": (7.0, 1200.0), "C": (9.5, 900.0), "D": (11.5, 700.0)}
KD_BUILDING_MWFRS = 0.85       # Table 26.6-1, buildings (MWFRS and C&C)
G_RIGID = 0.85                 # 26.11.1, rigid buildings
GCPI_ENCLOSED = 0.18           # Table 26.13-1, +/- for enclosed buildings
CP_WINDWARD_WALL = 0.8         # Figure 27.3-1 (all L/B)
CP_SIDE_WALL = -0.7
MIN_WALL_PRESSURE_PSF = 16.0   # 27.1.5 MWFRS minimum on projected wall area

# --- seismic constants (ASCE 7-16) ---
R_NOT_DETAILED = 3.0    # Table 12.2-1 H.1: steel systems not specifically
                        # detailed for seismic resistance; SDC B/C only
CT_ALL_OTHER = 0.02     # Table 12.8-2 'all other structural systems'
X_ALL_OTHER = 0.75

WIND_CASES = ("Wx+", "Wx-", "Wz+", "Wz-")
SEISMIC_CASES = ("Ex+", "Ex-", "Ez+", "Ez-")


class UnsupportedSDCError(ValueError):
    """Raised for SDC D/E/F sites: v1's R = 3 basis is limited to SDC B/C."""


# ---------------------------------------------------------------------------
# Wind: velocity pressure and external pressure coefficients
# ---------------------------------------------------------------------------

def kz(z_ft: float, exposure: str) -> float:
    """Velocity pressure exposure coefficient, Table 26.10-1:
    Kz = 2.01 (z/zg)^(2/alpha), evaluated at z = 15 ft for anything lower."""
    alpha, zg = EXPOSURE_CONSTANTS[exposure]
    return 2.01 * (max(z_ft, 15.0) / zg) ** (2.0 / alpha)


def velocity_pressure_psf(z_ft: float, V_mph: float, exposure: str,
                          Kzt: float = 1.0, Kd: float = KD_BUILDING_MWFRS,
                          Ke: float = 1.0) -> float:
    """qz per Eq. 26.10-1 (V in mph, result in psf)."""
    return 0.00256 * kz(z_ft, exposure) * Kzt * Kd * Ke * V_mph**2


def leeward_wall_cp(L_over_B: float) -> float:
    """Leeward wall Cp, Figure 27.3-1: -0.5 up to L/B = 1, -0.3 at 2,
    -0.2 from 4 on, linearly interpolated between."""
    if L_over_B <= 1.0:
        return -0.5
    if L_over_B <= 2.0:
        return -0.5 + 0.2 * (L_over_B - 1.0)
    if L_over_B <= 4.0:
        return -0.3 + 0.1 * (L_over_B - 2.0) / 2.0
    return -0.2


def roof_cp_bands(h_ft: float, L_ft: float) -> list[tuple[float, float, float]]:
    """Flat-roof suction Cp bands (from the windward edge), Figure 27.3-1,
    wind normal or parallel to ridge with theta < 10 deg.

    The figure tabulates two band sets — h/L <= 0.5 (-0.9/-0.9/-0.5/-0.3 at
    0..h/2, h/2..h, h..2h, beyond) and h/L >= 1.0 (-1.3 to h/2, -0.7
    beyond) — with linear interpolation between. The 0.8x area reduction
    allowed on the -1.3 zone is not taken (conservative). Returns
    [(x_from_ft, x_to_ft, Cp)] covering the full along-wind depth L.
    """
    low = ((0.5 * h_ft, -0.9), (h_ft, -0.9), (2.0 * h_ft, -0.5),
           (math.inf, -0.3))
    high = ((0.5 * h_ft, -1.3), (math.inf, -0.7))

    def value(bands, x: float) -> float:
        for upper, cp in bands:
            if x < upper:
                return cp
        return bands[-1][1]

    t = min(max((h_ft / L_ft - 0.5) / 0.5, 0.0), 1.0)
    cuts = sorted({c for c in (0.5 * h_ft, h_ft, 2.0 * h_ft) if c < L_ft})
    edges = [0.0, *cuts, L_ft]
    bands: list[tuple[float, float, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mid = 0.5 * (lo + hi)
        cp = value(low, mid) + t * (value(high, mid) - value(low, mid))
        if bands and abs(bands[-1][2] - cp) < 1e-12:
            bands[-1] = (bands[-1][0], hi, cp)   # merge equal neighbors
        else:
            bands.append((lo, hi, cp))
    return bands


# ---------------------------------------------------------------------------
# Wind: design pressures for one wind direction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PressureBand:
    """One constant-pressure band. Wall bands run over height (ft above
    grade); roof bands run over distance from the windward edge. Pressures
    are DESIGN external pressures q*G*Cp — signed positive toward the
    surface, negative for suction."""
    from_ft: float
    to_ft: float
    pressure_psf: float


@dataclass(frozen=True)
class WindPressures:
    """MWFRS design pressures for one wind direction on the enclosed box.

    direction 'x': wind across the clear span — loads the transverse
    frames; B (width normal to wind) is the building length. direction 'z':
    wind along the building — loads the longitudinal braced bays; B is the
    span. Internal pressure (+/- gcpi_qh_psf) cancels between windward and
    leeward walls, so base_shear_kip uses external pressures; it does NOT
    cancel on the roof, so roof_net_uplift_kip includes +GCpi.
    """
    direction: str            # 'x' | 'z'
    V_mph: float
    exposure: str
    B_ft: float               # width normal to the wind
    L_ft: float               # along-wind depth
    h_ft: float               # mean roof height (flat roof: the roof height)
    qh_psf: float
    gcpi_qh_psf: float        # magnitude of the internal pressure, qh*GCpi
    windward_bands: tuple[PressureBand, ...]    # over height, positive
    leeward_psf: float                          # uniform over height, negative
    side_wall_psf: float                        # negative
    roof_bands: tuple[PressureBand, ...]        # external, negative (suction)
    base_shear_computed_kip: float              # from pressures
    base_shear_kip: float                       # after the 27.1.5 floor
    governed_by_minimum: bool
    roof_net_uplift_kip: float                  # external + internal (+GCpi)


def design_wind_pressures(direction: str, V_mph: float, exposure: str,
                          span_ft: float, length_ft: float, h_ft: float,
                          Kzt: float = 1.0, Ke: float = 1.0,
                          Kd: float = KD_BUILDING_MWFRS,
                          G: float = G_RIGID,
                          gcpi: float = GCPI_ENCLOSED) -> WindPressures:
    """Directional-procedure MWFRS pressures for one wind direction."""
    if direction not in ("x", "z"):
        raise ValueError(f"direction must be 'x' or 'z', got {direction!r}.")
    B, L = ((length_ft, span_ft) if direction == "x" else (span_ft, length_ft))

    qh = velocity_pressure_psf(h_ft, V_mph, exposure, Kzt=Kzt, Kd=Kd, Ke=Ke)

    # windward wall: qz varies with height — stepped bands at the customary
    # 15 ft first step then every 5 ft, each priced at its top (conservative)
    edges = [0.0]
    z = 15.0
    while z < h_ft:
        edges.append(z)
        z += 5.0
    edges.append(h_ft)
    windward = tuple(
        PressureBand(lo, hi, velocity_pressure_psf(hi, V_mph, exposure,
                                                   Kzt=Kzt, Kd=Kd, Ke=Ke)
                     * G * CP_WINDWARD_WALL)
        for lo, hi in zip(edges[:-1], edges[1:]))

    leeward = qh * G * leeward_wall_cp(L / B)
    side = qh * G * CP_SIDE_WALL
    roof = tuple(PressureBand(lo, hi, qh * G * cp)
                 for lo, hi, cp in roof_cp_bands(h_ft, L))

    shear_kip = (sum(b.pressure_psf * (b.to_ft - b.from_ft) for b in windward)
                 + abs(leeward) * h_ft) * B / 1000.0
    min_shear_kip = MIN_WALL_PRESSURE_PSF * B * h_ft / 1000.0
    governed = min_shear_kip > shear_kip

    gcpi_qh = gcpi * qh
    uplift_kip = sum((-b.pressure_psf + gcpi_qh) * (b.to_ft - b.from_ft)
                    for b in roof) * B / 1000.0

    return WindPressures(
        direction=direction, V_mph=V_mph, exposure=exposure,
        B_ft=B, L_ft=L, h_ft=h_ft, qh_psf=qh, gcpi_qh_psf=gcpi_qh,
        windward_bands=windward, leeward_psf=leeward, side_wall_psf=side,
        roof_bands=roof,
        base_shear_computed_kip=shear_kip,
        base_shear_kip=max(shear_kip, min_shear_kip),
        governed_by_minimum=governed,
        roof_net_uplift_kip=uplift_kip,
    )


# ---------------------------------------------------------------------------
# Distribution to lateral-resisting lines
# ---------------------------------------------------------------------------

def tributary_line_lengths_ft(n_lines: int, total_ft: float) -> list[float]:
    """One-way tributary length per equally spaced line (ends take half)."""
    if n_lines < 2:
        return [total_ft]
    s = total_ft / (n_lines - 1)
    return [s / 2.0] + [s] * (n_lines - 2) + [s / 2.0]


def transverse_line_shears_kip(total_kip: float,
                               config: ClearSpanConfig) -> list[float]:
    """Direction-x story force per transverse frame, by tributary length
    along the building (end frames take half a bay) — valid for wind
    pressure and for seismic mass, both uniform along the length."""
    tribs = tributary_line_lengths_ft(config.n_frames, config.length_ft)
    return [total_kip * t / config.length_ft for t in tribs]


# ---------------------------------------------------------------------------
# Seismic: ELF
# ---------------------------------------------------------------------------

def roof_height_ft(config: ClearSpanConfig) -> float:
    """Roof-plane height: the eave for wide-flange girders, the top-chord
    level for trusses (walls and roof mass sit there)."""
    extra = config.truss_depth_ft if config.girder_system == TRUSS else 0.0
    return config.eave_height_ft + extra


def approximate_period_s(hn_ft: float, Ct: float = CT_ALL_OTHER,
                         x: float = X_ALL_OTHER) -> float:
    """Ta = Ct * hn^x (Eq. 12.8-7)."""
    return Ct * hn_ft**x


def effective_seismic_weight_kip(config: ClearSpanConfig,
                                 steel_weight_lb: float,
                                 wall_weight_psf: float = 0.0,
                                 include_live_fraction: float = 0.0) -> float:
    """W per 12.7.2: all optimized steel + superimposed dead over the roof
    + the upper half of the wall cladding (tributary to the roof line).
    Counting the full column weight is slightly conservative and keeps the
    accounting trivial. include_live_fraction adds that fraction of
    live_psf — use 0.2 when live_psf is a flat-roof snow load > 30 psf,
    leave 0 for roof live load (excluded by 12.7.2)."""
    area = config.span_ft * config.length_ft
    perimeter = 2.0 * (config.span_ft + config.length_ft)
    return (steel_weight_lb
            + config.superimposed_dead_psf * area
            + wall_weight_psf * perimeter * roof_height_ft(config) / 2.0
            + include_live_fraction * config.live_psf * area) / 1000.0


def seismic_response_coefficient(sds: float, sd1: float, s1: float,
                                 T_s: float, R: float, Ie: float,
                                 tl_s: float) -> tuple[float, str]:
    """Cs per 12.8.1.1 with its cap and floors; returns (Cs, governing eq)."""
    r_over_i = R / Ie
    cs = sds / r_over_i
    governing = "12.8-2"
    if T_s <= tl_s:
        cap = sd1 / (T_s * r_over_i)
        cap_eq = "12.8-3"
    else:
        cap = sd1 * tl_s / (T_s**2 * r_over_i)
        cap_eq = "12.8-4"
    if cap < cs:
        cs, governing = cap, cap_eq
    floor = max(0.044 * sds * Ie, 0.01)
    if cs < floor:
        cs, governing = floor, "12.8-5"
    if s1 >= 0.6:
        floor_s1 = 0.5 * s1 / r_over_i
        if cs < floor_s1:
            cs, governing = floor_s1, "12.8-6"
    return cs, governing


@dataclass(frozen=True)
class SeismicELF:
    """Equivalent-lateral-force basis: V = Cs * W, all at the roof line."""
    sds: float
    sd1: float
    s1: float
    sdc: str
    tl_s: float
    R: float
    Ie: float
    rho: float
    hn_ft: float
    Ta_s: float
    Cs: float
    Cs_governing: str
    W_kip: float
    V_kip: float


def seismic_elf(hazard: SeismicHazard, site: SiteConfig, W_kip: float,
                hn_ft: float, R: float = R_NOT_DETAILED,
                Ct: float = CT_ALL_OTHER, x: float = X_ALL_OTHER) -> SeismicELF:
    """ELF base shear for the single-story building. Raises
    UnsupportedSDCError outside SDC A-C (see module docstring)."""
    if hazard.sdc not in ("A", "B", "C"):
        raise UnsupportedSDCError(
            f"Site is Seismic Design Category {hazard.sdc} (SDS = "
            f"{hazard.sds} g, SD1 = {hazard.sd1} g). The v1 design basis — "
            f"R = {R_NOT_DETAILED} 'steel systems not specifically detailed "
            "for seismic resistance' (ASCE 7-16 Table 12.2-1) — is "
            "permitted in SDC B and C only. SDC D+ requires a seismically "
            "detailed system (OCBF/SCBF + AISC 341), which is out of scope "
            "of this version.")
    Ta = approximate_period_s(hn_ft, Ct=Ct, x=x)
    if hazard.sdc == "A":
        # 11.7: SDC A is exempt from the ELF; 1.4.2 minimum lateral force
        cs, governing = 0.01, "1.4.2 (SDC A minimum)"
    else:
        cs, governing = seismic_response_coefficient(
            hazard.sds, hazard.sd1, hazard.s1, Ta, R, site.Ie, hazard.tl_s)
    return SeismicELF(
        sds=hazard.sds, sd1=hazard.sd1, s1=hazard.s1, sdc=hazard.sdc,
        tl_s=hazard.tl_s, R=R, Ie=site.Ie,
        rho=1.0,   # 12.3.4.1: rho = 1.0 in SDC A-C (the v1 scope)
        hn_ft=hn_ft, Ta_s=Ta, Cs=cs, Cs_governing=governing,
        W_kip=W_kip, V_kip=cs * W_kip,
    )


# ---------------------------------------------------------------------------
# LRFD load combinations (2.3.1 / 2.3.6)
# ---------------------------------------------------------------------------

def lateral_strength_combos(sds: float, rho: float = 1.0,
                            live_is_snow: bool = False) -> dict[str, dict[str, float]]:
    """Strength combos involving wind or seismic, as {name: {case: factor}}.

    The gravity module's 'L' case is the governing roof live/snow surface
    load, so it takes the 0.5 companion factor in the wind combos (2.3.1
    combos 3 and 4 collapse together for a roof-only building) and appears
    in the seismic combos only as 0.2S when live_is_snow. The seismic
    vertical effect Ev = 0.2*SDS*D is folded into the D factor per 2.3.6;
    rho multiplies the horizontal effect (1.0 in SDC A-C).
    """
    combos: dict[str, dict[str, float]] = {}
    for w in WIND_CASES:
        combos[f"1.2D+1.6L+0.5{w}"] = {"D": 1.2, "L": 1.6, w: 0.5}
        combos[f"1.2D+{w}+0.5L"] = {"D": 1.2, "L": 0.5, w: 1.0}
        combos[f"0.9D+{w}"] = {"D": 0.9, w: 1.0}
    d_plus = round(1.2 + 0.2 * sds, 4)
    d_minus = round(0.9 - 0.2 * sds, 4)
    for e in SEISMIC_CASES:
        up: dict[str, float] = {"D": d_plus, e: rho}
        if live_is_snow:
            up["L"] = 0.2
        combos[f"(1.2+0.2SDS)D+{e}" + ("+0.2S" if live_is_snow else "")] = up
        combos[f"(0.9-0.2SDS)D+{e}"] = {"D": d_minus, e: rho}
    return combos


def all_strength_combos(sds: float, rho: float = 1.0,
                        live_is_snow: bool = False) -> dict[str, dict[str, float]]:
    """Gravity combos (1.4D, 1.2D+1.6L) + every lateral combo — the full
    strength envelope for the load-application phase."""
    return {**STRENGTH_COMBOS,
            **lateral_strength_combos(sds, rho=rho, live_is_snow=live_is_snow)}


# ---------------------------------------------------------------------------
# The lateral load basis: structured object + JSON dict
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LateralLoads:
    """The resolved lateral loads as OBJECTS — the working input of the
    analysis/design phase (lateral_load_basis() serializes this)."""
    elf: SeismicELF
    wind_x: WindPressures
    wind_z: WindPressures
    combos: dict[str, dict[str, float]]
    steel_weight_lb: float
    live_is_snow: bool


def compute_lateral_loads(config: ClearSpanConfig, steel_weight_lb: float,
                          hazards: SiteHazards, R: float = R_NOT_DETAILED,
                          Ct: float = CT_ALL_OTHER, x: float = X_ALL_OTHER,
                          wall_weight_psf: float = 0.0,
                          include_live_fraction: float = 0.0,
                          live_is_snow: bool = False,
                          Kzt: float = 1.0, Ke: float = 1.0) -> LateralLoads:
    """Resolve every ASCE 7-16 lateral load for one optimized building."""
    site = hazards.site
    h = roof_height_ft(config)
    W = effective_seismic_weight_kip(
        config, steel_weight_lb, wall_weight_psf=wall_weight_psf,
        include_live_fraction=include_live_fraction)
    elf = seismic_elf(hazards.seismic, site, W, h, R=R, Ct=Ct, x=x)
    wind = {
        d: design_wind_pressures(
            d, hazards.basic_wind_speed_mph, site.exposure,
            config.span_ft, config.length_ft, h, Kzt=Kzt, Ke=Ke)
        for d in ("x", "z")
    }
    return LateralLoads(
        elf=elf, wind_x=wind["x"], wind_z=wind["z"],
        combos=all_strength_combos(elf.sds, rho=elf.rho,
                                   live_is_snow=live_is_snow),
        steel_weight_lb=steel_weight_lb, live_is_snow=live_is_snow,
    )


def _r(value: float, ndigits: int = 4) -> float:
    return round(float(value), ndigits)


def _bands_json(bands: tuple[PressureBand, ...]) -> list[dict]:
    return [{"from_ft": _r(b.from_ft), "to_ft": _r(b.to_ft),
             "pressure_psf": _r(b.pressure_psf)} for b in bands]


def _wind_json(wp: WindPressures, config: ClearSpanConfig) -> dict:
    block = {
        "width_normal_to_wind_B_ft": _r(wp.B_ft),
        "along_wind_depth_L_ft": _r(wp.L_ft),
        "mean_roof_height_ft": _r(wp.h_ft),
        "qh_psf": _r(wp.qh_psf),
        "internal_pressure_qh_GCpi_psf": _r(wp.gcpi_qh_psf),
        "windward_wall_bands": _bands_json(wp.windward_bands),
        "leeward_wall_psf": _r(wp.leeward_psf),
        "side_wall_psf": _r(wp.side_wall_psf),
        "roof_external_bands": _bands_json(wp.roof_bands),
        "base_shear_computed_kip": _r(wp.base_shear_computed_kip),
        "base_shear_kip": _r(wp.base_shear_kip),
        "governed_by_16psf_minimum": wp.governed_by_minimum,
        "roof_net_uplift_kip": _r(wp.roof_net_uplift_kip),
    }
    if wp.direction == "x":
        block["per_frame_line_kip"] = [
            _r(v) for v in transverse_line_shears_kip(wp.base_shear_kip, config)]
    else:
        block["per_sidewall_line_kip"] = _r(wp.base_shear_kip / 2.0)
    return block


def lateral_load_basis(config: ClearSpanConfig, steel_weight_lb: float,
                       hazards: SiteHazards, R: float = R_NOT_DETAILED,
                       Ct: float = CT_ALL_OTHER, x: float = X_ALL_OTHER,
                       wall_weight_psf: float = 0.0,
                       include_live_fraction: float = 0.0,
                       live_is_snow: bool = False,
                       Kzt: float = 1.0, Ke: float = 1.0) -> dict:
    """The complete ASCE 7-16 lateral load basis for one optimized building:
    seismic ELF, wind pressures and line shears for both directions, and the
    strength combinations. This dict is the data contract consumed by the
    load-application/analysis phase (and written to JSON by
    lateral_design.py). steel_weight_lb is OptimizationResult.total_weight_lb
    from the gravity run.
    """
    site = hazards.site
    loads = compute_lateral_loads(
        config, steel_weight_lb, hazards, R=R, Ct=Ct, x=x,
        wall_weight_psf=wall_weight_psf,
        include_live_fraction=include_live_fraction,
        live_is_snow=live_is_snow, Kzt=Kzt, Ke=Ke)
    elf = loads.elf
    wind = {"x": loads.wind_x, "z": loads.wind_z}

    return {
        "schema": "frame_optimizer/lateral_load_basis",
        "schema_version": 1,
        "reference_standard": "ASCE 7-16",
        "directions": {
            "x": "clear-span direction: transverse frames resist",
            "z": "building length: longitudinal braced bays resist",
        },
        "site": {
            "latitude": site.latitude, "longitude": site.longitude,
            "risk_category": site.risk_category,
            "site_class": site.site_class,
            "exposure": site.exposure,
            "Ie": site.Ie,
        },
        "seismic": {
            "parameters": {
                "sds_g": _r(elf.sds), "sd1_g": _r(elf.sd1),
                "s1_g": _r(elf.s1), "sdc": elf.sdc, "tl_s": _r(elf.tl_s),
                "source": hazards.seismic.source,
            },
            "elf": {
                "R": elf.R, "rho": elf.rho, "Ie": elf.Ie,
                "hn_ft": _r(elf.hn_ft), "Ta_s": _r(elf.Ta_s),
                "Cs": _r(elf.Cs), "Cs_governing": elf.Cs_governing,
                "W_kip": _r(elf.W_kip), "V_kip": _r(elf.V_kip),
                "weight_inputs": {
                    "steel_weight_lb": _r(steel_weight_lb, 1),
                    "wall_weight_psf": _r(wall_weight_psf),
                    "include_live_fraction": _r(include_live_fraction),
                },
            },
            "per_frame_line_kip": [
                _r(v) for v in transverse_line_shears_kip(elf.V_kip, config)],
            "per_sidewall_line_kip": _r(elf.V_kip / 2.0),
        },
        "wind": {
            "V_mph": hazards.basic_wind_speed_mph,
            "exposure": site.exposure,
            "Kzt": Kzt, "Ke": Ke, "Kd": KD_BUILDING_MWFRS, "G": G_RIGID,
            "GCpi": GCPI_ENCLOSED,
            "x": _wind_json(wind["x"], config),
            "z": _wind_json(wind["z"], config),
        },
        "load_combinations": all_strength_combos(
            elf.sds, rho=elf.rho, live_is_snow=live_is_snow),
        "notes": [
            "Loads and combinations only: member load application, drift, "
            "and LFRS sizing are the later phases.",
            "Wind: MWFRS directional procedure, enclosed flat-roof box; "
            "internal pressure cancels on the windward/leeward pair, is "
            "included in roof_net_uplift_kip.",
            "Seismic: ELF, single story, R = 3 both directions (SDC A-C "
            "only), rho = 1.0.",
            "'L' load case = governing roof live/snow surface load, as in "
            "the gravity module.",
        ],
    }


def summarize_lateral_basis(basis: dict) -> str:
    """Human-readable report of a lateral_load_basis() dict."""
    s, w = basis["seismic"], basis["wind"]
    p, e = s["parameters"], s["elf"]
    lines = [
        "=" * 62,
        "frame_optimizer - ASCE 7-16 lateral load basis",
        "=" * 62,
        f"Site:    ({basis['site']['latitude']}, {basis['site']['longitude']})  "
        f"Risk {basis['site']['risk_category']}, Site Class "
        f"{basis['site']['site_class']}, Exposure {basis['site']['exposure']}",
        f"Seismic: SDS = {p['sds_g']:.3f} g, SD1 = {p['sd1_g']:.3f} g, "
        f"SDC {p['sdc']}  [{p['source']}]",
        f"         W = {e['W_kip']:.1f} kip, Ta = {e['Ta_s']:.3f} s, "
        f"Cs = {e['Cs']:.4f} ({e['Cs_governing']}), R = {e['R']:.1f}",
        f"         V = {e['V_kip']:.1f} kip  ->  "
        f"{s['per_sidewall_line_kip']:.1f} kip per sidewall brace line (z)",
        f"Wind:    V = {w['V_mph']:.0f} mph, "
        f"qh = {w['x']['qh_psf']:.1f} psf @ h = {w['x']['mean_roof_height_ft']:.1f} ft",
    ]
    for d, label in (("x", "transverse"), ("z", "longitudinal")):
        blk = w[d]
        governed = "  (16 psf minimum governs)" if blk["governed_by_16psf_minimum"] else ""
        lines.append(
            f"         {label} ({d}): base shear = {blk['base_shear_kip']:.1f} kip"
            f"{governed}, roof net uplift = {blk['roof_net_uplift_kip']:.1f} kip")
    n_frames = len(s["per_frame_line_kip"])
    gov_x = ("wind" if w["x"]["base_shear_kip"] > s["elf"]["V_kip"] else "seismic")
    gov_z = ("wind" if w["z"]["base_shear_kip"] > s["elf"]["V_kip"] else "seismic")
    lines += [
        f"Frames:  {n_frames} transverse lines; interior frame takes "
        f"wind {max(w['x']['per_frame_line_kip']):.1f} kip / "
        f"seismic {max(s['per_frame_line_kip']):.1f} kip",
        f"Governs: {gov_x} in x (frames), {gov_z} in z (braced bays) "
        "(by base shear; member-level verdicts need the analysis phase)",
        f"Combos:  {len(basis['load_combinations'])} strength combinations "
        "(gravity + wind + seismic)",
        "=" * 62,
    ]
    return "\n".join(lines)
