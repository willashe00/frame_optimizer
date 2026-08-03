"""Lateral load application to the 3D Pynite model + drift extraction.

The transverse (x) direction is analyzed on the FULL augmented building
geometry in one model — the representative strip cannot be used because the
braced bays make the building non-uniform along its length, and a one-shot
certification analysis (as opposed to the gravity layout search) can afford
the full model.

Load cases applied here (case names match lateral_loads combo generation):

* Wx+/Wx-  — wall pressures on the windward/leeward columns of every
  transverse frame (stepped qz bands x the frame's tributary length) plus
  the roof net-uplift bands on the purlins/eave struts (position along the
  span picks the band). Internal pressure is included in the roof uplift
  (+GCpi, no opposing surface over slab-on-grade) and omitted on the walls,
  where it cancels between the windward/leeward pair. When the ASCE 7-16
  27.1.5 minimum governed a direction's base shear, the wall pressures are
  scaled up so the analyzed shear equals the floored value.
* Wz+/Wz-  — ROOF UPLIFT ONLY: the in-plane longitudinal wall forces are
  resisted by the braced bays, which are designed closed-form
  (braced_bay.py) because the 3D model's mechanism-stabilization DZ
  restraints would silently absorb any applied FZ. The uplift bands (which
  run along the building length for z-wind) drive the 0.9D+W purlin/girder
  force-reversal checks.
* Ex+/Ex-  — the ELF story force per frame (tributary share of V) at the
  frame's two top corners, exactly where the DAM notional loads act.
* Ez cases — no loads in this model (closed-form braced bays); combos that
  reference only Ez are dropped from the model via active_strength_combos()
  to avoid paying P-Delta solves for load-free combinations.

DAM note: build_model's notional-load augmentation adds ND/NL to EVERY
combo containing D/L factors, including the lateral ones. AISC C2.2b(4)
only requires notional loads in gravity-only combos for modest-sway frames;
adding them everywhere is conservative and keeps one code path.

Drift: a parallel NOMINAL-stiffness model (the same one that serves
serviceability deflections) carries two extra combos —
'drift:E' = 1.0*Ex+ (the strength-level E, giving delta_xe for
Delta = Cd*delta_xe/Ie per 12.8.6) and 'drift:W' = WIND_DRIFT_FACTOR*Wx+
(0.7V ~ 10-year MRI wind, factor (0.7)^2 ~ 0.49, per ASCE 7-16 Appendix CC
practice; the limit itself lives in lateral_designer.py). Frame sway is
read at the top corner nodes.

Units: kips and inches internally; the WindPressures bands arrive in ft/psf.
"""
from __future__ import annotations

from Pynite import FEModel3D

from ..config import COLUMN, FT
from ..geometry import FrameGeometry
from ..lateral_loads import LateralLoads, WindPressures, tributary_line_lengths_ft
from ..sections import WShape
from .frame_model import build_model, extract_demands

WIND_DRIFT_FACTOR = 0.49   # (0.7V)^2: ~10-yr MRI wind for serviceability
DRIFT_COMBOS = {"drift:E": {"Ex+": 1.0}, "drift:W": {"Wx+": WIND_DRIFT_FACTOR}}

ROOF_LOADED_GROUPS = ("purlin", "eave_strut")   # members carrying roof wind


# cases whose combos are NOT solved in the 3D model: Ez acts only on the
# closed-form braced bays, and the negative-direction cases are exact
# mirrors — the building is symmetric about both plan axes (identical
# frames; the braced-bay layout is mirror-symmetric by construction), so
# Wx-/Wz-/Ex- produce member demands that are the +direction demands
# mapped onto the mirrored members. The GROUP envelopes that drive sizing
# are therefore identical, exactly as for the one-direction DAM notional
# loads; only the per-member table columns reflect the +direction loading.
# Halving the combo set halves the P-Delta solve count.
_SKIPPED_CASES = ("Wx-", "Wz-", "Ex-", "Ez+", "Ez-")


def active_strength_combos(combos: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Combos analyzed in the 3D model (see _SKIPPED_CASES for why the
    mirrored and braced-bay-only combinations are excluded)."""
    return {name: factors for name, factors in combos.items()
            if not any(case in _SKIPPED_CASES for case in factors)}


def _wall_scale(wp: WindPressures) -> float:
    """Scale on the applied wall pressures so the analyzed base shear
    matches the 27.1.5-floored design value when the minimum governs."""
    if wp.governed_by_minimum and wp.base_shear_computed_kip > 0.0:
        return wp.base_shear_kip / wp.base_shear_computed_kip
    return 1.0


def _band_pressure_psf(bands, position_ft: float) -> float:
    for b in bands:
        if position_ft < b.to_ft:
            return b.pressure_psf
    return bands[-1].pressure_psf


def _psf_to_kip_in(pressure_psf: float, trib_ft: float) -> float:
    """psf x tributary width (ft) -> kip/in of member length."""
    return pressure_psf * trib_ft / 1000.0 / 12.0


def _frame_columns(geometry: FrameGeometry) -> list[tuple[str, int, int]]:
    """(member name, side, frame index) for the transverse-frame columns
    (C{side}.{j}; gable columns CG* are end-wall gravity posts and carry no
    x-wind — the end walls are parallel to x)."""
    out = []
    for m in geometry.members:
        if m.group != COLUMN or not m.name.startswith("C") or m.name.startswith("CG"):
            continue
        side_s, j_s = m.name[1:].split(".")
        out.append((m.name, int(side_s), int(j_s)))
    return out


def _frame_top_corners(geometry: FrameGeometry) -> dict[int, list[str]]:
    """frame index (by z order) -> its two top-corner node names."""
    frames: dict[float, list] = {}
    for n in geometry.nodes:
        frames.setdefault(round(n.z, 6), []).append(n)
    corners: dict[int, list[str]] = {}
    for j, z in enumerate(sorted(frames)):
        nodes = frames[z]
        y_top = max(n.y for n in nodes)
        top = [n for n in nodes if abs(n.y - y_top) < 1e-6]
        x_lo = min(n.x for n in top)
        x_hi = max(n.x for n in top)
        corners[j] = [n.name for n in top
                      if abs(n.x - x_lo) < 1e-6 or abs(n.x - x_hi) < 1e-6]
    return corners


def add_lateral_load_cases(model: FEModel3D, geometry: FrameGeometry,
                           config, loads: LateralLoads) -> None:
    """Apply Wx+/-, Wz+/- (roof uplift), and Ex+/- to an assembled model."""
    tribs_ft = tributary_line_lengths_ft(config.n_frames, config.length_ft)
    span_ft = config.span_ft
    length_ft = config.length_ft
    node_by_name = {n.name: n for n in geometry.nodes}

    # ---- wall wind, x-direction (windward qz bands + leeward suction) ----
    wx = loads.wind_x
    scale = _wall_scale(wx)
    for name, side, j in _frame_columns(geometry):
        trib = tribs_ft[j]
        lee_w = _psf_to_kip_in(-wx.leeward_psf, trib) * scale   # suction -> +x
        for case, ww_side, sign in (("Wx+", 0, 1.0), ("Wx-", 1, -1.0)):
            if side == ww_side:
                for band in wx.windward_bands:
                    w = sign * _psf_to_kip_in(band.pressure_psf, trib) * scale
                    model.add_member_dist_load(
                        name, "FX", w, w,
                        band.from_ft * FT, band.to_ft * FT, case=case)
            else:
                w = sign * lee_w
                model.add_member_dist_load(name, "FX", w, w, case=case)

    # ---- roof net uplift (external suction + internal +GCpi) ----
    for m in geometry.members:
        if m.group not in ROOF_LOADED_GROUPS or m.trib_width_in <= 0.0:
            continue
        trib_ft = m.trib_width_in / 12.0
        x_ft = node_by_name[m.i_node].x / FT           # purlins run along z
        mid_z_ft = (node_by_name[m.i_node].z + node_by_name[m.j_node].z) / 2.0 / FT
        for case, wp, dist_ft in (
                ("Wx+", loads.wind_x, x_ft),
                ("Wx-", loads.wind_x, span_ft - x_ft),
                ("Wz+", loads.wind_z, mid_z_ft),
                ("Wz-", loads.wind_z, length_ft - mid_z_ft)):
            p_net = _band_pressure_psf(wp.roof_bands, dist_ft) - wp.gcpi_qh_psf
            w_up = _psf_to_kip_in(-p_net, trib_ft)     # suction -> upward +FY
            if w_up > 0.0:
                model.add_member_dist_load(m.name, "FY", w_up, w_up, case=case)

    # ---- seismic story forces at the frame top corners ----
    corners = _frame_top_corners(geometry)
    V = loads.elf.V_kip
    for j, nodes in corners.items():
        F = V * tribs_ft[j] / length_ft / len(nodes)
        for node in nodes:
            model.add_node_load(node, "FX", F, case="Ex+")
            model.add_node_load(node, "FX", -F, case="Ex-")


# X-diagonal groups whose stiffness is neutralized in the 3D model
_SOFT_GROUPS = ("brace", "roof_brace")
_SOFT_SCALE = 1e-4


def _soften_brace_stiffness(model: FEModel3D) -> None:
    """Neutralize the X diagonals' stiffness in the 3D model (their
    self-weight LOADS stay).

    The diagonals are tension-only by design and sized closed-form for the
    longitudinal system (braced_bay.py). Modeled with full elastic
    stiffness they do two wrong things under transverse wind: they attract
    large FRAME-COMPATIBILITY forces (end frames sway less than interior
    ones, and stiff diagonals try to splint the difference — forces that
    real rod/angle X bracing with tension-only action and oversized holes
    simply sheds), and they relieve the frames of tributary load the
    frames must be able to carry alone. Scaling their section properties
    to negligible makes every frame carry 100% of its tributary lateral
    load (conservative) and leaves the braces with self-weight statics
    only, exactly matching the stated design basis."""
    for group in _SOFT_GROUPS:
        section = model.sections.get(group)
        if section is not None:
            section.A *= _SOFT_SCALE
            section.Iy *= _SOFT_SCALE
            section.Iz *= _SOFT_SCALE
            section.J *= _SOFT_SCALE


def analyze_lateral(geometry: FrameGeometry, assignment: dict[str, WShape],
                    config, loads: LateralLoads):
    """One full lateral certification analysis.

    Returns (demands, drift_by_frame, service_model): demands are enveloped
    over the active strength combos from the DAM P-Delta model; drift is
    {frame_j: {'seismic': delta_xe_in, 'wind': delta_10yr_in}} from the
    nominal-stiffness model; the solved service model is returned for base-
    reaction export.
    """
    combos = active_strength_combos(loads.combos)

    strength = build_model(geometry, assignment, config, rigid_strength=True,
                           strength_combos=combos)
    _soften_brace_stiffness(strength)
    add_lateral_load_cases(strength, geometry, config, loads)
    strength.analyze_PDelta(check_stability=True, sparse=True)

    service = build_model(geometry, assignment, config,
                          strength_combos=combos)
    _soften_brace_stiffness(service)
    add_lateral_load_cases(service, geometry, config, loads)
    for name, factors in DRIFT_COMBOS.items():
        service.add_load_combo(name, factors)
    service.analyze(check_stability=True, check_statics=False, sparse=True)

    demands = extract_demands(strength, geometry, assignment,
                              service_model=service, strength_combos=combos)

    drift: dict[int, dict[str, float]] = {}
    for j, nodes in _frame_top_corners(geometry).items():
        drift[j] = {
            "seismic": max(abs(service.nodes[n].DX["drift:E"]) for n in nodes),
            "wind": max(abs(service.nodes[n].DX["drift:W"]) for n in nodes),
        }
    return demands, drift, service
