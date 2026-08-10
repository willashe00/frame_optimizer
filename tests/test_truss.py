"""Pratt roof truss: topology, hand-calc anchors, checker units, end-to-end.

The truss replaces the interior W girders when the clear span outgrows every
rolled shape (roof_system="auto") or on request (roof_system="truss"). The
anchors follow the existing philosophy: the FEA must reproduce classical
parallel-chord Pratt statics — chord force M/d with M = wL^2/8, end-diagonal
tension V/sin(theta) — and the factored base reactions must equal the total
factored gravity load exactly.
"""
import math

import pytest

from frame_optimizer import ClearSpanConfig, evaluate, optimize, optimize_layout
from frame_optimizer.analysis import MemberDemand, analyze_frame
from frame_optimizer.clear_span import (BOTTOM_CHORD, END_GIRDER, GIRDER,
                                        PURLIN, TOP_CHORD, TRUSS_WEB,
                                        build_clear_span_geometry,
                                        clear_span_check_params)
from frame_optimizer.config import (COLUMN, KPA_TO_KSI, M_TO_IN,
                                    PLF_TO_KIP_PER_IN)
from frame_optimizer.design import CheckParams, GroupRules, check_member
from frame_optimizer.design.aisc_strengths import (compression_capacity,
                                                   flexure_major_capacity)
from frame_optimizer.optimization.optimizer import _bound_demands
from frame_optimizer.sections import get_shapes

CAT = {s.name: s for s in get_shapes(
    ["W8X24", "W10X33", "W12X14", "W12X16", "W12X40", "W12X53", "W14X61",
     "W16X26", "W14X22"])}


def tcfg(**kw):
    """Small pinned-layout truss building: 24 m span -> depth 2 m, 12 panels
    of 2 m (panel points coincide with every 4th purlin line at 6 m). The
    length matches the span so the footprint normalization never swaps the
    clear-span direction out from under the truss assertions."""
    base = dict(
        girder_candidates=["W24X76"],   # required by "auto"; unused as truss
        purlin_candidates=["W12X14", "W12X16", "W14X22"],
        column_candidates=["W12X40", "W12X53", "W14X61"],
        end_girder_candidates=["W12X16", "W16X26"],
        top_chord_candidates=["W12X40", "W12X53"],
        bottom_chord_candidates=["W12X40", "W12X53"],
        truss_web_candidates=["W8X24", "W10X33", "W12X40"],
        roof_system="truss",
        span_m=24.0, length_m=24.0, height_m=8.0,
        n_frames=4, purlin_spacing_m=1.5, end_wall_columns=4,
        superimposed_dead_kpa=0.72, live_kpa=1.20,
        purlin_Lb_m=0.0,
    )
    base.update(kw)
    return ClearSpanConfig(**base)


def analyzed(config, top="W12X53", bottom="W12X40", web="W8X24",
             purlin="W12X14", column="W14X61", end_girder="W16X26"):
    geo = build_clear_span_geometry(config)
    assignment = {COLUMN: CAT[column], TOP_CHORD: CAT[top],
                  BOTTOM_CHORD: CAT[bottom], END_GIRDER: CAT[end_girder],
                  TRUSS_WEB: CAT[web], PURLIN: CAT[purlin]}
    return geo, assignment, analyze_frame(geo, assignment, config)


def truss_line_load_kip_in(config, assignment):
    """Equivalent uniform factored line load on one interior truss frame:
    roof surface + smeared purlin self + smeared truss self-weight."""
    q_d = config.superimposed_dead_kpa * KPA_TO_KSI
    q_l = config.live_kpa * KPA_TO_KSI
    sp = config.purlin_spacing_actual_m * M_TO_IN
    s_f = config.frame_spacing_m * M_TO_IN
    span = config.span_m * M_TO_IN
    depth = config.truss_depth_actual_m * M_TO_IN
    n = config.n_truss_panels
    panel = span / n
    diag = math.hypot(panel, depth)
    self_len = (span                        # top chord
                + (span - 2.0 * panel)      # bottom chord
                + (n - 1) * depth + n * diag)   # webs
    w_truss = (assignment[TOP_CHORD].weight_plf * span
               + assignment[BOTTOM_CHORD].weight_plf * (span - 2.0 * panel)
               + assignment[TRUSS_WEB].weight_plf * (self_len - span
                                                     - (span - 2.0 * panel))
               ) / span * PLF_TO_KIP_PER_IN
    w_purlin = assignment[PURLIN].weight_plf * PLF_TO_KIP_PER_IN / sp
    w_d = (q_d + w_purlin) * s_f + w_truss
    w_l = q_l * s_f
    return max(1.4 * w_d, 1.2 * w_d + 1.6 * w_l)


# ---------------------------------------------------------------- topology

def test_truss_topology_and_derived_geometry():
    config = tcfg()
    assert config.truss_depth_actual_m == pytest.approx(2.0)   # span/12
    assert config.n_truss_panels == 12
    assert config.truss_panel_m == pytest.approx(2.0)

    geo = build_clear_span_geometry(config)
    assert geo.groups == (COLUMN, END_GIRDER, TOP_CHORD, BOTTOM_CHORD,
                          TRUSS_WEB, PURLIN)

    n = config.n_truss_panels
    n_int = config.n_frames - 2   # interior (truss) frames
    # per interior frame: 1 top chord + 1 bottom chord, n-1 verticals
    # (no end posts - the end diagonals carry the shear into the chord),
    # n diagonals
    assert len(geo.members_in_group(TOP_CHORD)) == n_int
    assert len(geo.members_in_group(BOTTOM_CHORD)) == n_int
    webs = geo.members_in_group(TRUSS_WEB)
    assert len(webs) == n_int * ((n - 1) + n)
    # end frames keep their end girders and gable columns
    assert len(geo.members_in_group(END_GIRDER)) == 2
    assert len(geo.members_in_group(COLUMN)) == 2 * config.n_frames + 2 * 4

    node = {nd.name: nd for nd in geo.nodes}
    span_in = config.span_m * M_TO_IN
    depth_in = config.truss_depth_actual_m * M_TO_IN
    panel_in = span_in / n

    tc = geo.members_in_group(TOP_CHORD)[0]
    assert (tc.i_node, tc.j_node) == ("NE0.1", "NE1.1")
    assert tc.length_in == pytest.approx(span_in)
    bc = geo.members_in_group(BOTTOM_CHORD)[0]
    assert (bc.i_node, bc.j_node) == ("NC1.1", f"NC{n - 1}.1")
    assert bc.length_in == pytest.approx(span_in - 2.0 * panel_in)

    # bottom chord hangs at eave - depth on interior panel points only
    bottoms = [nd for nd in geo.nodes if nd.name.startswith("NC")]
    assert len(bottoms) == n_int * (n - 1)
    for nd in bottoms:
        assert nd.y == pytest.approx(config.height_m * M_TO_IN - depth_in)
    # chord END nodes keep clamped rotations (moment-released member ends);
    # interior chord nodes ride the continuous chord
    assert not node["NC1.1"].free_rotations
    assert not node[f"NC{n - 1}.1"].free_rotations
    assert node["NC2.1"].free_rotations

    # panel points landing on a purlin line reuse that node (2 m panels on
    # 1.5 m purlins coincide every 6 m -> panels 3, 6, 9 = purlins 4, 8, 12)
    names = {nd.name for nd in geo.nodes}
    assert {"NT3.1", "NT6.1", "NT9.1"}.isdisjoint(names)
    vert_ends = {m.i_node for m in webs if m.name.startswith("TV")}
    assert {"NP4.1", "NP8.1", "NP12.1"} <= vert_ends

    # verticals span depth; diagonals slope toward midspan (Pratt)
    diag_in = math.hypot(panel_in, depth_in)
    for m in webs:
        expected = depth_in if m.name.startswith("TV") else diag_in
        assert m.length_in == pytest.approx(expected)


def test_truss_frames_are_pin_plus_roller_end_frames_untouched():
    config = tcfg()
    geo = build_clear_span_geometry(config)
    for nd in geo.nodes:
        frame = int(nd.name.split(".")[-1])
        if frame in (0, config.n_frames - 1):
            assert not nd.free_dx          # end frames: today's model exactly
        elif nd.name.startswith("NE0"):
            assert not nd.free_dx          # the pin bearing keeps DX
        elif nd.name.startswith(("NE1", "NP", "NT", "NC")):
            assert nd.free_dx              # roller side + panel points


def test_purlin_wiring_is_identical_to_girder_mode():
    truss_geo = build_clear_span_geometry(tcfg())
    girder_geo = build_clear_span_geometry(tcfg(roof_system="girder"))

    def wiring(geo):
        return {(m.name, m.i_node, m.j_node, m.trib_width_in)
                for m in geo.members_in_group(PURLIN)}

    assert wiring(truss_geo) == wiring(girder_geo)


# ------------------------------------------------------------ statics anchors

def test_chord_forces_match_pratt_statics():
    """Max chord axial ~ M/d with M = w*L^2/8: top chord compression, bottom
    chord tension. The continuous chords carry a little of the load in local
    bending, so the pin-jointed value is a few percent off, never more."""
    config = tcfg()
    _, assignment, demands = analyzed(config)
    by_name = {d.name: d for d in demands}
    span_in = config.span_m * M_TO_IN
    depth_in = config.truss_depth_actual_m * M_TO_IN
    w_u = truss_line_load_kip_in(config, assignment)
    P = w_u * span_in**2 / 8.0 / depth_in

    assert by_name["TC1"].Pu == pytest.approx(-P, rel=6e-2)
    assert by_name["BC1"].Pu == pytest.approx(P, rel=6e-2)
    # the top chord's local bending between panel points is real but bounded
    # by the panel-span value
    assert 0.0 < by_name["TC1"].Mux < w_u * (span_in / 12)**2


def test_end_diagonal_carries_the_support_shear():
    config = tcfg()
    _, assignment, demands = analyzed(config)
    by_name = {d.name: d for d in demands}
    span_in = config.span_m * M_TO_IN
    depth_in = config.truss_depth_actual_m * M_TO_IN
    panel_in = span_in / config.n_truss_panels
    w_u = truss_line_load_kip_in(config, assignment)
    V = w_u * span_in / 2.0
    sin_theta = depth_in / math.hypot(panel_in, depth_in)

    # the diagonal carries the first-panel shear: at least the panel-point
    # value (load in the first half panel goes straight into the bearing),
    # at most the full support reaction
    lo = (V - w_u * panel_in / 2.0) / sin_theta
    hi = V / sin_theta
    assert lo * 0.98 <= by_name["TD0.1"].Pu <= hi * 1.02
    # web force pattern: all diagonals tension, all verticals compression
    for d in demands:
        if d.group != TRUSS_WEB:
            continue
        if d.name.startswith("TD"):
            assert d.Pu > 0.0, f"{d.name} should be tension"
        else:
            assert d.Pu < 0.0, f"{d.name} should be compression"


def test_factored_base_reactions_equal_total_load_with_truss():
    config = tcfg(end_wall_columns=0)   # no gable columns: clean closure sum
    geo, assignment, demands = analyzed(config)
    total_axial = sum(-d.Pu for d in demands if d.group == COLUMN)

    area_in2 = (config.span_m * M_TO_IN) * (config.length_m * M_TO_IN)
    dead = config.superimposed_dead_kpa * KPA_TO_KSI * area_in2
    live = config.live_kpa * KPA_TO_KSI * area_in2
    for m in geo.members:   # every member's self-weight is explicit dead load
        dead += assignment[m.group].weight_plf * PLF_TO_KIP_PER_IN * m.length_in

    assert total_axial == pytest.approx(1.2 * dead + 1.6 * live, rel=1e-3)


def test_truss_sag_is_reported_on_the_full_span_top_chord():
    """The top chord is one full-span horizontal member, so its chord-relative
    sag IS the truss midspan deflection and the span ratios are the real
    L/360, L/240."""
    config = tcfg()
    _, _, demands = analyzed(config)
    tc = next(d for d in demands if d.name == "TC1")
    assert tc.length_in == pytest.approx(config.span_m * M_TO_IN)
    assert tc.defl_total_in > tc.defl_live_in > 0.0
    rules = clear_span_check_params(config).rules_for(TOP_CHORD)
    assert rules.check_deflection and rules.defl_scale_axial
    for group in (BOTTOM_CHORD, TRUSS_WEB):
        assert not clear_span_check_params(config).rules_for(
            group).check_deflection


# --------------------------------------------------------------- checker units

def _demand(**kw):
    base = dict(name="m", group="g", story=1, length_in=240.0,
                trib_width_in=0.0, shape_used="W12X40",
                Ix_used=CAT["W12X40"].Ix, A_used=CAT["W12X40"].A,
                Pu=0.0, Mux=0.0, Muy=0.0, Vu=0.0,
                defl_total_in=0.0, defl_live_in=0.0)
    base.update(kw)
    return MemberDemand(**base)


def _params(rules):
    return CheckParams(Fy=50.0, Fu=65.0, E=29000.0, group_rules={"g": rules})


def test_b1_amplifies_compression_chord_moment():
    shape = CAT["W12X40"]
    KLx = 120.0
    rules = GroupRules(Lb_in=60.0, check_deflection=False,
                       KLx_in=KLx, KLy_in=60.0, apply_B1=True)
    demand = _demand(Pu=-200.0, Mux=800.0)
    row = check_member(shape, demand, _params(rules))

    Pe1 = math.pi**2 * 29000.0 * shape.Ix / KLx**2
    B1 = 1.0 / (1.0 - 200.0 / Pe1)
    assert B1 > 1.0
    assert row["B1"] == pytest.approx(B1)
    phi_Mnx, _ = flexure_major_capacity(shape, 50.0, 29000.0, Lb=60.0, Cb=1.0)
    assert row["UC_Mx"] == pytest.approx(B1 * 800.0 / phi_Mnx)

    # tension members are never amplified; without apply_B1 nothing changes
    assert check_member(shape, _demand(Pu=200.0, Mux=800.0),
                        _params(rules))["B1"] == 1.0
    plain = GroupRules(Lb_in=60.0, check_deflection=False,
                       KLx_in=KLx, KLy_in=60.0)
    assert check_member(shape, demand, _params(plain))["B1"] == 1.0
    assert check_member(shape, demand, _params(plain))["UC_Mx"] == \
        pytest.approx(800.0 / phi_Mnx)


def test_b1_beyond_euler_load_fails_but_still_ranks():
    shape = CAT["W8X24"]
    KLx = 400.0
    Pe1 = math.pi**2 * 29000.0 * shape.Ix / KLx**2
    rules = GroupRules(check_deflection=False, KLx_in=KLx, KLy_in=100.0,
                       apply_B1=True)
    row = check_member(shape, _demand(Pu=-1.5 * Pe1, Mux=10.0), _params(rules))
    assert not row["PASS"]
    assert row["governing_uc"] > 10.0
    assert math.isfinite(row["governing_uc"])


def test_slenderness_limit_is_200_compression_300_tension():
    shape = CAT["W8X24"]
    L = 250.0 * shape.ry          # L/r = 250: fails E2, passes D1
    rules = GroupRules(Lb_in=0.0, check_deflection=False,
                       check_slenderness=True)
    compression = check_member(shape, _demand(length_in=L, Pu=-1.0),
                               _params(rules))
    tension = check_member(shape, _demand(length_in=L, Pu=1.0), _params(rules))
    assert compression["UC_slenderness"] == pytest.approx(250.0 / 200.0)
    assert not compression["PASS"]
    assert tension["UC_slenderness"] == pytest.approx(250.0 / 300.0)
    assert tension["PASS"]


def test_kl_overrides_set_the_compression_capacity():
    shape = CAT["W10X33"]
    rules = GroupRules(Lb_in=0.0, check_deflection=False,
                       KLx_in=90.0, KLy_in=60.0)
    row = check_member(shape, _demand(length_in=480.0, Pu=-50.0),
                       _params(rules))
    phi_Pn, _ = compression_capacity(shape, 50.0, 29000.0, KLx=90.0, KLy=60.0)
    assert row["phiPn_kN"] == pytest.approx(phi_Pn * 4.4482216152605)
    # default (no overrides) uses the full member length
    default = GroupRules(Lb_in=0.0, check_deflection=False)
    row_def = check_member(shape, _demand(length_in=480.0, Pu=-50.0),
                           _params(default))
    phi_Pn_def, _ = compression_capacity(shape, 50.0, 29000.0,
                                         KLx=480.0, KLy=480.0)
    assert row_def["phiPn_kN"] == pytest.approx(phi_Pn_def * 4.4482216152605)
    assert row["phiPn_kN"] > row_def["phiPn_kN"]


def test_deflection_projection_by_area_for_truss_chords():
    light, heavy = CAT["W12X40"], CAT["W14X61"]
    rules = GroupRules(Lb_in=0.0, defl_live_ratio=360.0, defl_total_ratio=240.0,
                       defl_scale_axial=True)
    demand = _demand(length_in=7200.0, Ix_used=light.Ix, A_used=light.A,
                     defl_total_in=2.0, defl_live_in=1.0)
    row = check_member(heavy, demand, _params(rules))
    scale = light.A / heavy.A     # truss sag ~ 1/A_chord, NOT 1/Ix
    assert row["UC_defl_live"] == pytest.approx(
        (1.0 * scale) / (7200.0 / 360.0))
    assert row["UC_defl_total"] == pytest.approx(
        (2.0 * scale) / (7200.0 / 240.0))


# ------------------------------------------------------- roof-system selection

def test_config_validation():
    with pytest.raises(ValueError, match="roof_system"):
        tcfg(roof_system="dome")
    with pytest.raises(ValueError, match="truss_web_candidates"):
        tcfg(truss_web_candidates=None)
    with pytest.raises(ValueError, match="end_girder_candidates"):
        tcfg(end_girder_candidates=None, end_wall_columns=0)
    with pytest.raises(ValueError, match="truss depth"):
        tcfg(truss_depth_m=9.0)   # >= the 8 m eave
    with pytest.raises(ValueError, match="n_frames >= 3"):
        tcfg(n_frames=2, length_m=8.0)
    with pytest.raises(ValueError, match="girder_candidates"):
        tcfg(girder_candidates=[], roof_system="auto")
    # a pure truss config does not need girder candidates
    assert tcfg(girder_candidates=[]).is_truss


def test_auto_keeps_the_girder_when_one_can_carry_the_span():
    """Backward compatibility: on a girder-buildable footprint, "auto" with
    truss candidates produces the identical design to a config without them."""
    legacy = tcfg(roof_system="auto", span_m=15.0, length_m=16.0,
                  girder_candidates=["W21X44", "W24X76"],
                  top_chord_candidates=None, bottom_chord_candidates=None,
                  truss_web_candidates=None, end_wall_columns=2)
    with_truss = tcfg(roof_system="auto", span_m=15.0, length_m=16.0,
                      girder_candidates=["W21X44", "W24X76"],
                      end_wall_columns=2)
    a, b = optimize(legacy), optimize(with_truss)
    assert not a.config.is_truss and not b.config.is_truss
    assert a.sections == b.sections
    assert a.total_weight_kg == pytest.approx(b.total_weight_kg)


def test_auto_switches_to_truss_when_no_girder_can_span():
    config = tcfg(roof_system="auto", girder_candidates=["W12X16"])
    result = optimize(config, second_order=True)
    assert result.config.is_truss
    assert result.feasible
    assert TOP_CHORD in result.sections and GIRDER not in result.sections
    assert result.second_order is not None
    assert result.second_order["verified"] is True
    assert bool(result.member_table["PASS"].all())


def test_explicit_roof_system_is_honored():
    truss = optimize(tcfg())
    assert truss.config.is_truss and truss.feasible
    girder = optimize(tcfg(roof_system="girder",
                           girder_candidates=["W24X76", "W30X108"]))
    assert not girder.config.is_truss
    assert GIRDER in girder.sections


def test_evaluate_explicit_truss_design():
    result = evaluate(tcfg(), {
        "column": "W14X61", "top_chord": "W12X53", "bottom_chord": "W12X40",
        "end_girder": "W16X26", "truss_web": "W10X33", "purlin": "W12X14"})
    assert set(result.sections) == {COLUMN, TOP_CHORD, BOTTOM_CHORD,
                                    END_GIRDER, TRUSS_WEB, PURLIN}
    assert result.total_weight_kg > 0


# ------------------------------------------------- layout search + second order

def test_layout_search_resolves_to_truss_and_verifies_second_order():
    config = ClearSpanConfig(
        girder_candidates=["W12X16"],          # hopeless for a 14 m span
        purlin_candidates=["W12X14", "W12X16", "W14X22"],
        column_candidates=["W12X40", "W12X53", "W14X61"],
        end_girder_candidates=["W12X16", "W16X26"],
        top_chord_candidates=["W12X40", "W12X53"],
        bottom_chord_candidates=["W12X40", "W12X53"],
        truss_web_candidates=["W8X24", "W10X33", "W12X40"],
        span_m=14.0, length_m=20.0, height_m=8.0,
        superimposed_dead_kpa=0.72, live_kpa=1.20,
        purlin_Lb_m=0.0,
    )
    result = optimize_layout(config)
    assert result.config.is_truss
    assert result.feasible
    assert bool(result.member_table["PASS"].all())
    assert result.second_order is not None and result.second_order["verified"]
    assert result.layout_search
    # the truss run keeps end girders propped: unpropped 20 m end-girder
    # layouts (0 gables) are pruned by statics, never analyzed
    zero_gable = [r for r in result.layout_search
                  if r["end_wall_columns"] == 0]
    assert zero_gable and all(not r["analyzed"] for r in zero_gable)
    assert "Pratt truss" in "\n".join(result.config.describe())
    assert "2nd order" in result.summary()


def test_bounds_omit_truss_groups_but_still_hold_for_the_rest():
    config = tcfg()
    geo, assignment, demands = analyzed(config)
    candidates = {g: [assignment[g]] for g in geo.groups}
    bounds = _bound_demands(config, geo, candidates)
    assert bounds   # purlin + column at least
    for g in (TOP_CHORD, BOTTOM_CHORD, TRUSS_WEB, GIRDER):
        assert g not in bounds
    for group, bound in bounds.items():
        actual = [d for d in demands if d.group == group]
        assert bound.Mux <= max(d.Mux for d in actual) + 1e-9
        assert abs(bound.Pu) <= max(abs(d.Pu) for d in actual) + 1e-9
