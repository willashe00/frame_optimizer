"""Truss-girder clear-span building: rigid bents, DAM, hand-calc anchors, e2e.

Truss frames are RIGID transverse bents (full-height columns tied to both
chords) analyzed per the AISC Direct Analysis Method. Anchor philosophy
matches the other test modules, applied to truss statics:

* factored base reactions equal the total factored gravity load exactly, and
  the horizontal thrust reactions self-equilibrate (equal and opposite);
* chord axial forces reproduce M/d truss statics, a few percent below the
  simple-span midspan value (partial end fixity from frame action, the
  governing section cut at the panel point nearest midspan, and the small
  moment share the continuous chords carry in bending);
* frame action REVERSES the chord force near the frame corners — the bottom
  chord must show end compression alongside its midspan tension, and the
  checker must verify both extremes;
* truss sag brackets the equivalent-beam value 5wL^4/384EI_truss: web-member
  axial strain adds deflection beam theory omits, partial end fixity removes
  some.

If the mechanism DX restraints ever swallowed the diagonals' horizontal
components again, the chord-force anchor would read near zero and fail
loudly.
"""
import math

import pytest

from frame_optimizer import ClearSpanConfig, evaluate, optimize, optimize_layout
from frame_optimizer.analysis import analyze_frame, build_model
from frame_optimizer.analysis.frame_model import (NOTIONAL_RATIO,
                                                  STIFFNESS_REDUCTION)
from frame_optimizer.clear_span import (TRUSS, TRUSS_BOT_CHORD,
                                        TRUSS_TOP_CHORD, TRUSS_WEB,
                                        WIDE_FLANGE,
                                        build_clear_span_geometry,
                                        candidate_layouts,
                                        candidate_truss_depths,
                                        clear_span_check_params,
                                        derive_truss_depth_ft,
                                        derive_truss_panel_ft)
from frame_optimizer.config import COLUMN, FT
from frame_optimizer.optimization.optimizer import wide_flange_infeasible_reason
from frame_optimizer.sections import get_shapes

PURLIN = "purlin"

CAT = {s.name: s for s in get_shapes(["W8X10", "W8X24", "W10X33", "W12X53"])}


def truss_cfg(**kw):
    base = dict(
        girder_candidates=[],            # unused with girder_system="truss"
        purlin_candidates=["W8X10", "W12X16"],
        column_candidates=["W10X33", "W12X53"],
        girder_system=TRUSS,
        truss_chord_candidates=["W8X24", "W10X33", "W12X53"],
        truss_web_candidates=["W8X10", "W8X24", "W10X33"],
        span_ft=50.0, length_ft=60.0, n_frames=3,
        eave_height_ft=20.0, purlin_spacing_ft=5.0,
        superimposed_dead_psf=15.0, live_psf=25.0,
        purlin_Lb_ft=0.0,
    )
    base.update(kw)
    return ClearSpanConfig(**base)


def analyzed(config, chord="W10X33", web="W8X24", purlin="W8X10",
             column="W10X33"):
    geo = build_clear_span_geometry(config)
    assignment = {COLUMN: CAT[column], TRUSS_TOP_CHORD: CAT[chord],
                  TRUSS_BOT_CHORD: CAT[chord], TRUSS_WEB: CAT[web],
                  PURLIN: CAT[purlin]}
    return geo, assignment, analyze_frame(geo, assignment, config)


def web_length_ft(config):
    """Total web steel per frame: k-1 interior verticals + k diagonals (the
    column segments between eave and top chord replace the end posts)."""
    k = config.n_truss_panels
    return ((k - 1) * config.truss_depth_ft
            + k * math.hypot(config.truss_panel_actual_ft,
                             config.truss_depth_ft))


def frame_line_loads_plf(config, chord="W10X33", web="W8X24", purlin="W8X10"):
    """(dead, live) line load an interior frame's truss carries, smearing the
    purlin and web self-weight over the span like the girder anchor test."""
    n_lines = config.n_purlin_spaces + 1
    purlin_psf = CAT[purlin].weight_plf * n_lines / config.span_ft
    web_plf = CAT[web].weight_plf * web_length_ft(config) / config.span_ft
    s_f = config.frame_spacing_ft
    w_dead = ((15.0 + purlin_psf) * s_f + web_plf + 2.0 * CAT[chord].weight_plf)
    return w_dead, 25.0 * s_f


# ---------------------------------------------------------------- topology

def test_truss_topology_and_counts():
    config = truss_cfg()
    geo = build_clear_span_geometry(config)
    node = {n.name: n for n in geo.nodes}
    span_in = config.span_ft * FT
    k = config.n_truss_panels

    # perimeter columns only — the clear span stays clear
    for m in geo.members_in_group(COLUMN):
        x = node[m.i_node].x
        assert x == pytest.approx(0.0) or x == pytest.approx(span_in)
    # chords are full-span physical members loaded only through shared nodes
    for group in (TRUSS_TOP_CHORD, TRUSS_BOT_CHORD):
        for m in geo.members_in_group(group):
            assert m.length_in == pytest.approx(span_in)
            assert m.trib_width_in == 0.0
    assert geo.groups == (COLUMN, TRUSS_TOP_CHORD, TRUSS_BOT_CHORD,
                          TRUSS_WEB, PURLIN)
    nf = config.n_frames
    assert len(geo.members_in_group(COLUMN)) == 2 * nf
    assert len(geo.members_in_group(TRUSS_TOP_CHORD)) == nf
    assert len(geo.members_in_group(TRUSS_BOT_CHORD)) == nf
    # k-1 interior verticals + k diagonals per frame; the column segments
    # between eave and top chord replace the end posts (rigid bent)
    assert len(geo.members_in_group(TRUSS_WEB)) == (2 * k - 1) * nf
    n_lines = config.n_purlin_spaces + 1
    assert len(geo.members_in_group(PURLIN)) == n_lines * (nf - 1)

    # full-height columns: base to the top-chord level, through the eave node
    col_len = (config.eave_height_ft + config.truss_depth_ft) * FT
    for m in geo.members_in_group(COLUMN):
        assert m.length_in == pytest.approx(col_len)
        assert node[m.j_node].y == pytest.approx(col_len)

    # top chord at eave + depth, bottom chord at the eave (clear height kept)
    y_top = (config.eave_height_ft + config.truss_depth_ft) * FT
    for name, n in node.items():
        if name.startswith(("NT", "NP")):
            assert n.y == pytest.approx(y_top)
        if name.startswith(("NC", "NE")):
            assert n.y == pytest.approx(config.eave_height_ft * FT)

    # diagonals descend toward midspan and never land on a bearing
    for m in geo.members:
        if m.name.startswith("TD"):
            hi, lo = node[m.i_node], node[m.j_node]
            assert hi.y > lo.y
            assert abs(lo.x - span_in / 2.0) < abs(hi.x - span_in / 2.0)
            assert not lo.name.startswith("NE")


def test_truss_dof_flags():
    config = truss_cfg()
    span_in = config.span_ft * FT
    geo = build_clear_span_geometry(config)
    for n in geo.nodes:
        if n.is_base:
            # pinned bases ground the bent: DX restrained, rotations clamped
            assert not n.free_dx and not n.free_rotations
        elif n.name.startswith("NE"):
            # eave nodes: the continuous column passes through (rotational
            # stiffness) and the bent sways freely (frame action, no roller
            # asymmetry — thrust is real and self-equilibrating)
            assert n.free_dx and n.free_rotations
        elif n.name.startswith(("NC", "NP")):
            assert n.free_dx and n.free_rotations   # chord interiors
        elif n.name.startswith("NT"):
            assert n.free_dx              # every top-chord node carries chord force
            interior = not (n.x == pytest.approx(0.0)
                            or n.x == pytest.approx(span_in))
            assert n.free_rotations == interior


def test_purlin_lines_reuse_coincident_panel_nodes():
    # span 60 @ 5 ft purlins, panels derived at 5 ft: every purlin line sits
    # on a panel point, so no separate NP nodes exist at all
    config = truss_cfg(span_ft=60.0, length_ft=60.0)
    assert config.truss_panel_actual_ft == pytest.approx(5.0)
    geo = build_clear_span_geometry(config)
    assert not any(n.name.startswith("NP") for n in geo.nodes)
    # default config (panels 4.17 ft vs purlins 5 ft): only midspan coincides
    default = truss_cfg()
    geo = build_clear_span_geometry(default)
    per_frame = sum(1 for n in geo.nodes if n.name.endswith(".0")
                    and n.name.startswith("NP"))
    assert per_frame == default.n_purlin_spaces - 1 - 1   # 8 of 9 interior lines


def test_truss_derivations_and_proportions():
    assert derive_truss_depth_ft(120.0) == pytest.approx(10.0)
    assert derive_truss_panel_ft(120.0, 10.0) == pytest.approx(10.0)  # 12 panels
    # panel count is even (mirror-symmetric Pratt) and never below the floor
    assert 120.0 / derive_truss_panel_ft(120.0, 11.0) == 10
    assert 20.0 / derive_truss_panel_ft(20.0, 10.0) == 4
    config = truss_cfg()
    assert config.truss_depth_ft == pytest.approx(50.0 / 12.0)
    assert config.n_truss_panels == 12
    assert config.truss_diagonal_deg == pytest.approx(45.0)
    pinned = truss_cfg(truss_depth_ft=5.0, truss_panel_ft=10.0)
    assert (pinned.truss_depth_ft, pinned.n_truss_panels) == (5.0, 5)
    # auto-derived truss proportions are layout-search variables; pinned
    # values are honored and searched as a single option
    assert {"truss_depth_ft", "truss_panel_ft"} <= config.auto_layout_fields
    assert "truss_depth_ft" not in pinned.auto_layout_fields
    assert candidate_truss_depths(config) == pytest.approx(
        [50.0 / 10.0, 50.0 / 12.0, 50.0 / 15.0])
    assert candidate_truss_depths(pinned) == [5.0]


def test_truss_config_validation():
    with pytest.raises(ValueError, match="girder_system"):
        truss_cfg(girder_system="space_frame")
    with pytest.raises(ValueError, match="truss_chord_candidates"):
        truss_cfg(truss_chord_candidates=None)
    with pytest.raises(ValueError, match="truss_web_candidates"):
        truss_cfg(truss_web_candidates=[])
    with pytest.raises(ValueError, match="end_girder"):
        truss_cfg(end_girder_candidates=["W12X16"])
    with pytest.raises(ValueError, match="end_wall_columns"):
        truss_cfg(end_wall_columns=2)
    with pytest.raises(ValueError, match="truss_depth_ft"):
        truss_cfg(truss_depth_ft=-4.0)
    with pytest.raises(ValueError, match="truss_panel_ft"):
        truss_cfg(truss_panel_ft=40.0)   # > span/2
    with pytest.raises(ValueError, match="only apply"):
        ClearSpanConfig(   # pinned W-girder mode rejects the truss fields
            girder_candidates=["W24X76"], purlin_candidates=["W8X10"],
            column_candidates=["W10X33"], span_ft=50.0, length_ft=60.0,
            girder_system=WIDE_FLANGE,
            truss_chord_candidates=["W10X33"],
        )


def test_truss_check_params_rules():
    config = truss_cfg(girder_camber_in=1.5, truss_bottom_brace_ft=12.5)
    params = clear_span_check_params(config)
    panel_in = config.truss_panel_actual_ft * FT
    sp_in = config.purlin_spacing_actual_ft * FT

    top = params.rules_for(TRUSS_TOP_CHORD)
    assert top.KLx_in == pytest.approx(panel_in)     # in-plane: panel length
    assert top.KLy_in == pytest.approx(sp_in)        # out-of-plane: purlins
    assert top.Lb_in == pytest.approx(sp_in)
    assert top.camber_in == 1.5 and top.check_deflection

    bot = params.rules_for(TRUSS_BOT_CHORD)
    assert bot.KLx_in == pytest.approx(panel_in)
    assert bot.KLy_in == pytest.approx(12.5 * FT)    # explicit bridging spacing
    assert bot.Lb_in == pytest.approx(12.5 * FT)
    assert bot.camber_in == 1.5 and bot.check_deflection

    # bridging defaults to every panel point when not specified
    default_bot = clear_span_check_params(truss_cfg()).rules_for(TRUSS_BOT_CHORD)
    assert default_bot.KLy_in == pytest.approx(panel_in)

    web = params.rules_for(TRUSS_WEB)
    assert web.KLx_in is None and web.KLy_in is None   # true member length
    assert not web.check_deflection and not web.check_slenderness
    assert params.rules_for(COLUMN).check_slenderness


# ------------------------------------------------------- hand-calc anchors

def test_truss_statics_closure_and_thrust_equilibrium():
    config = truss_cfg()
    geo = build_clear_span_geometry(config)
    assignment = {COLUMN: CAT["W10X33"], TRUSS_TOP_CHORD: CAT["W10X33"],
                  TRUSS_BOT_CHORD: CAT["W10X33"], TRUSS_WEB: CAT["W8X24"],
                  PURLIN: CAT["W8X10"]}
    # base reactions on the nominal model close exactly (the notional loads
    # of the DAM strength model are fictitious and live only there)
    model = build_model(geo, assignment, config)
    model.analyze(check_stability=True, check_statics=False, sparse=True)
    bases = [n.name for n in geo.nodes if n.is_base]
    total_fy = sum(model.nodes[b].RxnFY["1.2D+1.6L"] for b in bases)
    total_fx = sum(model.nodes[b].RxnFX["1.2D+1.6L"] for b in bases)

    area = config.span_ft * config.length_ft
    n_lines = config.n_purlin_spaces + 1
    purlin_lb = CAT["W8X10"].weight_plf * n_lines * config.length_ft
    chord_lb = CAT["W10X33"].weight_plf * config.span_ft * 2 * config.n_frames
    web_lb = CAT["W8X24"].weight_plf * web_length_ft(config) * config.n_frames
    col_h_ft = config.eave_height_ft + config.truss_depth_ft
    col_lb = CAT["W10X33"].weight_plf * col_h_ft * 2 * config.n_frames
    dead_kip = (15.0 * area + purlin_lb + chord_lb + web_lb + col_lb) / 1000.0
    live_kip = 25.0 * area / 1000.0

    assert total_fy == pytest.approx(1.2 * dead_kip + 1.6 * live_kip, rel=1e-3)
    # rigid-bent thrust: real per-column horizontal reactions that cancel in
    # the global sum (gravity applies no net horizontal load)
    assert total_fx == pytest.approx(0.0, abs=1e-6)
    fx0 = model.nodes["NB0.1"].RxnFX["1.2D+1.6L"]
    fx1 = model.nodes["NB1.1"].RxnFX["1.2D+1.6L"]
    assert fx0 == pytest.approx(-fx1, rel=1e-6)
    assert abs(fx0) > 0.1        # frame action exists (not a pinned frame)


def test_truss_chord_forces_match_M_over_d_with_end_reversal():
    config = truss_cfg()
    _, _, demands = analyzed(config)
    by_name = {d.name: d for d in demands}
    w_d, w_l = frame_line_loads_plf(config)
    w_u = (1.2 * w_d + 1.6 * w_l) / 12000.0            # kip/in
    L = config.span_ft * FT
    M_over_d = (w_u * L**2 / 8.0) / (config.truss_depth_ft * FT)

    bot, top = by_name["BC1"], by_name["TC1"]
    assert bot.Pu > 0.0        # gravity puts the bottom chord in TENSION
    assert top.Pu < 0.0        # ... and the top chord in compression
    # a DX-restraint regression would drop these ratios toward zero; partial
    # end fixity from frame action keeps them a few percent below 1.0
    assert 0.85 <= bot.Pu_max / M_over_d <= 1.00
    assert 0.88 <= -top.Pu_min / M_over_d <= 1.02
    # rigid-bent frame action: the bottom chord reverses into compression
    # near the frame corners — the demand must expose it for checking
    assert bot.Pu_min < 0.0
    assert -bot.Pu_min < 0.25 * bot.Pu_max   # ... but far below the tension


def test_truss_sag_matches_equivalent_beam_plus_web_strain():
    config = truss_cfg()
    _, _, demands = analyzed(config)
    by_name = {d.name: d for d in demands}
    w_d, w_l = frame_line_loads_plf(config)
    w_serv = (w_d + w_l) / 12000.0                     # kip/in
    L = config.span_ft * FT
    half_d = config.truss_depth_ft * FT / 2.0
    shape = CAT["W10X33"]
    I_tr = 2.0 * (shape.A * half_d**2 + shape.Ix)
    delta_bend = 5.0 * w_serv * L**4 / (384.0 * 29000.0 * I_tr)

    sag = by_name["BC1"].defl_total_in
    assert by_name["TC1"].defl_total_in == pytest.approx(sag, rel=0.05)
    # web strain raises the sag above beam theory; the partial end fixity of
    # the rigid bent pulls it back down a little
    assert 0.95 * delta_bend <= sag <= 1.45 * delta_bend


def test_truss_web_force_pattern_is_pratt():
    config = truss_cfg()
    _, _, demands = analyzed(config)
    by_name = {d.name: d for d in demands}
    k = config.n_truss_panels

    assert by_name[f"TV{k // 2}.1"].Pu < 0.0    # verticals in compression
    for i in range(k):
        assert by_name[f"TD{i}.1"].Pu > 0.0     # every diagonal in tension
    # shear anchor: the first diagonal carries most of the end-panel shear
    # (the full-height column, which replaced the end post, takes the rest)
    w_d, w_l = frame_line_loads_plf(config)
    w_u = (1.2 * w_d + 1.6 * w_l) / 12000.0
    L = config.span_ft * FT
    sin_t = math.sin(math.radians(config.truss_diagonal_deg))
    V_over_sin = (w_u * L / 2.0) / sin_t
    assert 0.70 * V_over_sin <= by_name["TD0.1"].Pu <= 1.05 * V_over_sin


def test_rigid_bent_columns_carry_moment():
    config = truss_cfg()
    _, _, demands = analyzed(config)
    col = next(d for d in demands if d.name == "C0.1")
    assert col.Pu < 0.0            # compression, as always
    assert col.Mux > 5.0 * FT      # real frame-action bending (> 5 kip-ft)
    assert col.Vu > 0.5            # and the shear that goes with it


def test_dam_strength_model_vs_nominal_service_model():
    config = truss_cfg()
    geo = build_clear_span_geometry(config)
    assignment = {COLUMN: CAT["W10X33"], TRUSS_TOP_CHORD: CAT["W10X33"],
                  TRUSS_BOT_CHORD: CAT["W10X33"], TRUSS_WEB: CAT["W8X24"],
                  PURLIN: CAT["W8X10"]}

    strength = build_model(geo, assignment, config, rigid_strength=True)
    nominal = build_model(geo, assignment, config)
    # C2.3: 0.8-reduced stiffness in the strength model only
    assert strength.materials["steel"].E == pytest.approx(
        STIFFNESS_REDUCTION * config.E_ksi)
    assert nominal.materials["steel"].E == pytest.approx(config.E_ksi)
    # C2.2b + C2.3(c): notional cases ride the strength combos with their
    # parent-case factors; the nominal model carries no notional loads
    factors = strength.load_combos["1.2D+1.6L"].factors
    assert factors["ND"] == 1.2 and factors["NL"] == 1.6
    assert strength.load_combos["1.4D"].factors["ND"] == 1.4
    assert "ND" not in nominal.load_combos["1.2D+1.6L"].factors
    assert NOTIONAL_RATIO == pytest.approx(0.003)


def test_checker_verifies_both_axial_extremes():
    """A member in net tension that also sees a small compression excursion
    must be checked against buckling too — the compression side can govern
    even at a fraction of the tension magnitude (slender member)."""
    from frame_optimizer.analysis import MemberDemand
    from frame_optimizer.design import CheckParams, GroupRules, check_member

    shape = get_shapes(["W12X16"])[0]     # L/ry = 240/0.773 ~ 310: very slender
    demand = MemberDemand(
        name="M1", group="chord", story=1, length_in=240.0, trib_width_in=0.0,
        shape_used=shape.name, Ix_used=shape.Ix,
        Pu=100.0, Mux=0.0, Muy=0.0, Vu=0.0,
        defl_total_in=0.0, defl_live_in=0.0,
        Pu_min=-15.0, Pu_max=100.0,
    )
    params = CheckParams(Fy=50.0, Fu=65.0, E=29000.0,
                         group_rules={"chord": GroupRules(check_deflection=False)})
    row = check_member(shape, demand, params)
    # tension side: 100/212 ~ 0.47; compression side: 15 kip vs ~11 kip
    # buckling capacity -> compression governs despite the smaller force
    assert row["UC_axial"] > 1.0
    assert row["axial_clause"].startswith("E")   # buckling clause reported


# ------------------------------------------------------------- end-to-end

def test_optimize_truss_feasible_and_certified():
    result = optimize(truss_cfg())
    assert result.feasible and result.converged
    assert bool(result.member_table["PASS"].all())
    assert set(result.sections) == {COLUMN, TRUSS_TOP_CHORD, TRUSS_BOT_CHORD,
                                    TRUSS_WEB, PURLIN}
    assert result.total_weight_lb > 0


def test_optimize_layout_truss_never_searches_gables():
    config = truss_cfg(n_frames=None, purlin_spacing_ft=None, length_ft=45.0)
    layouts = candidate_layouts(config)
    assert len(layouts) > 1
    assert all(gables == 0 for _, _, gables in layouts)
    result = optimize_layout(config)
    assert result.feasible
    assert result.config.girder_system == TRUSS
    assert result.config.end_wall_columns == 0


def test_evaluate_truss_explicit_design():
    sections = {COLUMN: "W10X33", TRUSS_TOP_CHORD: "W10X33",
                TRUSS_BOT_CHORD: "W10X33", TRUSS_WEB: "W8X24",
                PURLIN: "W8X10"}
    result = evaluate(truss_cfg(), sections)
    assert result.total_weight_lb > 0
    assert set(result.sections) == set(sections)
    with pytest.raises(ValueError, match="group"):
        evaluate(truss_cfg(), {COLUMN: "W10X33", PURLIN: "W8X10"})


# ------------------------------------- automatic girder-system selection

def auto_cfg(**kw):
    """girder_system='auto' (default) with both W and truss candidates."""
    base = dict(
        girder_candidates=["W12X16", "W16X26", "W21X44"],
        purlin_candidates=["W8X10", "W12X16"],
        column_candidates=["W10X33", "W12X53"],
        truss_chord_candidates=["W8X24", "W10X33", "W12X53"],
        truss_web_candidates=["W8X10", "W8X24", "W10X33"],
        span_ft=30.0, length_ft=45.0, eave_height_ft=16.0,
        superimposed_dead_psf=15.0, live_psf=25.0,
        purlin_Lb_ft=0.0,
    )
    base.update(kw)
    return ClearSpanConfig(**base)


def test_auto_config_validation():
    with pytest.raises(ValueError, match="together"):
        auto_cfg(truss_web_candidates=None)      # chords without webs
    with pytest.raises(ValueError, match="truss_chord_candidates"):
        auto_cfg(truss_chord_candidates=None, truss_web_candidates=None,
                 truss_depth_ft=8.0)             # depth without any sections
    with pytest.raises(ValueError, match="non-empty"):
        auto_cfg(truss_chord_candidates=[], truss_web_candidates=[])


def test_wide_flange_infeasible_reason_bounds():
    # 30 ft: W21X44 easily plausible -> screen must NOT rule the system out
    assert wide_flange_infeasible_reason(auto_cfg()) is None
    # 100 ft: no candidate meets L/360 live deflection under any layout
    long = auto_cfg(span_ft=100.0, length_ft=100.0)
    assert "deflection" in wide_flange_infeasible_reason(long)
    # with deflection checks off, the strength bound still rules it out
    strength = auto_cfg(span_ft=100.0, length_ft=100.0, check_deflection=False)
    assert "moment" in wide_flange_infeasible_reason(strength)


def test_auto_prefers_wide_flange_when_feasible():
    result = optimize_layout(auto_cfg())
    assert result.feasible
    assert result.config.girder_system == WIDE_FLANGE
    assert all(e["girder_system"] == WIDE_FLANGE for e in result.layout_search)
    systems = {e["girder_system"]: e for e in result.system_search}
    assert systems[WIDE_FLANGE]["evaluated"] and systems[WIDE_FLANGE]["chosen"]
    assert not systems[TRUSS]["evaluated"]     # never built a single truss
    assert "System: wide_flange" in result.summary()


def test_auto_falls_back_to_truss_when_no_w_shape_spans():
    config = auto_cfg(span_ft=60.0, length_ft=60.0, n_frames=3,
                      purlin_spacing_ft=5.0, girder_candidates=["W12X16"])
    result = optimize_layout(config)
    assert result.feasible
    assert result.config.girder_system == TRUSS
    systems = {e["girder_system"]: e for e in result.system_search}
    # the screen proved the W system hopeless without evaluating a layout
    assert not systems[WIDE_FLANGE]["evaluated"]
    assert "deflection" in systems[WIDE_FLANGE]["note"]
    assert systems[TRUSS]["evaluated"] and systems[TRUSS]["chosen"]
    assert all(e["girder_system"] == TRUSS for e in result.layout_search)
    # the auto depth ranged over the span/10..span/15 practice band
    depths = {round(e["truss_depth_ft"], 2) for e in result.layout_search}
    assert depths == {6.0, 5.0, 4.0}
    assert "System: wide_flange skipped" in result.summary()
    assert "System: truss" in result.summary()
