"""N-group generalization: per-group design rules and group validation.

These tests exercise the Phase-1 refactor that removed the hardcoded
beam/column pair: GroupRules gate the AISC E2 KL/r limit, the LTB unbraced
length, and the serviceability deflection checks per design group, and the
optimizer refuses inconsistent group/candidate/rule sets.
"""
import pytest

from frame_optimizer import FrameConfig, evaluate
from frame_optimizer.analysis import MemberDemand
from frame_optimizer.design import CheckParams, GroupRules, check_member
from frame_optimizer.geometry import build_geometry
from frame_optimizer.sections import get_shapes

SHAPE = get_shapes(["W12X16"])[0]
L_IN = 240.0   # 20 ft


def demand(group="girder", **kw):
    base = dict(
        name="M1", group=group, story=1, length_in=L_IN, trib_width_in=0.0,
        shape_used=SHAPE.name, Ix_used=SHAPE.Ix,   # scale factor 1.0
        Pu=0.0, Mux=0.0, Muy=0.0, Vu=0.0,
        defl_total_in=0.0, defl_live_in=0.0,
    )
    base.update(kw)
    return MemberDemand(**base)


def params(group_rules):
    return CheckParams(Fy=50.0, Fu=65.0, E=29000.0, group_rules=group_rules)


def small_config():
    return FrameConfig(
        beam_candidates=["W10X12", "W12X16", "W14X22"],
        column_candidates=["W6X9", "W8X24"],
        x_bays=1, x_bay_spacing_ft=20.0,
        z_bays=1, z_bay_spacing_ft=20.0,
        stories=1, story_height_ft=10.0,
        superimposed_dead_psf=20.0, live_psf=50.0,
        deck_span_direction="z",
        beam_Lb_ft=0.0,
    )


def test_geometry_reports_its_groups():
    geo = build_geometry(small_config())
    assert geo.groups == ("column", "beam")


def test_unknown_group_raises_clear_error():
    p = params({"beam": GroupRules()})
    with pytest.raises(KeyError, match="girder"):
        check_member(SHAPE, demand(group="girder"), p)


def test_slenderness_check_gated_per_group():
    # W12X16 over 20 ft: L/ry = 240/0.773 ~ 310 > 200, so the check bites
    # when enabled and must be absent when the group's rules disable it.
    row_on = check_member(SHAPE, demand(), params({"girder": GroupRules(check_slenderness=True)}))
    row_off = check_member(SHAPE, demand(), params({"girder": GroupRules()}))
    assert row_on["UC_slenderness"] > 1.0
    assert "UC_slenderness" not in row_off


def test_unbraced_length_credit_per_group():
    # Lb_in=0 (continuously braced) must give more major-axis capacity than
    # the conservative default Lb = full 20-ft span (well past Lr for W12X16).
    braced = check_member(SHAPE, demand(Mux=100.0), params({"girder": GroupRules(Lb_in=0.0)}))
    unbraced = check_member(SHAPE, demand(Mux=100.0), params({"girder": GroupRules()}))
    assert braced["phiMnx_kipft"] > unbraced["phiMnx_kipft"]


def test_deflection_rules_per_group():
    d = demand(defl_live_in=1.0, defl_total_in=1.0)
    p = params({"girder": GroupRules(check_deflection=True,
                                     defl_live_ratio=360.0, defl_total_ratio=240.0)})
    row = check_member(SHAPE, d, p)
    assert row["UC_defl_live"] == pytest.approx(1.0 / (L_IN / 360.0))
    assert row["UC_defl_total"] == pytest.approx(1.0 / (L_IN / 240.0))
    # serviceability is on by default (safe default); opting out is explicit
    row_default = check_member(SHAPE, d, params({"girder": GroupRules()}))
    assert "UC_defl_live" in row_default
    row_off = check_member(SHAPE, d, params({"girder": GroupRules(check_deflection=False)}))
    assert "UC_defl_live" not in row_off


def test_simple_span_Cb_scales_elastic_ltb_capacity():
    # W12X16 unbraced over 20 ft is deep in elastic LTB, so capacity scales
    # linearly with Cb = 12.5/11 (AISC F1-1, parabolic diagram) when enabled
    d = demand(Mux=100.0)
    off = check_member(SHAPE, d, params({"girder": GroupRules()}))
    on = check_member(SHAPE, d, params({"girder": GroupRules(Cb_simple_span=True)}))
    assert on["phiMnx_kipft"] == pytest.approx(
        off["phiMnx_kipft"] * 12.5 / 11.0, rel=1e-9)
    # braced members (multiple unbraced segments / Lb < L) keep Cb = 1.0
    braced_off = check_member(SHAPE, d, params({"girder": GroupRules(Lb_in=60.0)}))
    braced_on = check_member(SHAPE, d, params(
        {"girder": GroupRules(Lb_in=60.0, Cb_simple_span=True)}))
    assert braced_on["phiMnx_kipft"] == pytest.approx(braced_off["phiMnx_kipft"])


def test_camber_credits_total_deflection_never_below_live():
    d = demand(defl_live_in=0.5, defl_total_in=1.2)
    base = check_member(SHAPE, d, params({"girder": GroupRules(check_deflection=True)}))
    cambered = check_member(SHAPE, d, params(
        {"girder": GroupRules(check_deflection=True, camber_in=0.4)}))
    limit_total = L_IN / 240.0
    assert cambered["UC_defl_total"] == pytest.approx((1.2 - 0.4) / limit_total)
    assert cambered["UC_defl_live"] == base["UC_defl_live"]   # live unaffected
    # over-declared camber floors at the live-load deflection
    excessive = check_member(SHAPE, d, params(
        {"girder": GroupRules(check_deflection=True, camber_in=5.0)}))
    assert excessive["UC_defl_total"] == pytest.approx(0.5 / limit_total)


def test_evaluate_rejects_incomplete_or_extra_groups():
    config = small_config()
    with pytest.raises(ValueError, match="group"):
        evaluate(config, {"beam": "W14X22"})                       # missing column
    with pytest.raises(ValueError, match="group"):
        evaluate(config, {"beam": "W14X22", "column": "W8X24",
                          "girder": "W18X35"})                     # no such members
