"""Phase 4+5: braced-bay statics anchors, drift virtual work, uplift
reversal, wind statics closure of the loaded 3D model, and the end-to-end
lateral design of a small building.

Hand-calc anchors follow the repo philosophy: every closed-form number is
recomputed on paper in the test comments.
"""
import math

import pytest

from frame_optimizer import (ClearSpanConfig, LateralSystemConfig, SiteConfig,
                             design_lateral, optimize, resolve_site_hazards)
from frame_optimizer.analysis import build_model
from frame_optimizer.analysis.lateral_model import (active_strength_combos,
                                                    add_lateral_load_cases)
from frame_optimizer.braced_bay import (bay_drift_in, bay_forces,
                                        roof_truss_forces,
                                        seismic_bay_shares_kip,
                                        wind_bay_shares_kip,
                                        windward_fraction)
from frame_optimizer.lateral_designer import uplift_uc
from frame_optimizer.lateral_loads import (all_strength_combos,
                                           compute_lateral_loads,
                                           design_wind_pressures)
from frame_optimizer.lateral_system import build_lateral_system
from frame_optimizer.sections import get_shapes


def cfg(**kw):
    base = dict(
        girder_candidates=["W24X76", "W30X108", "W33X130", "W40X167"],
        purlin_candidates=["W8X10", "W12X16", "W14X22"],
        column_candidates=["W10X33", "W12X53", "W14X90", "W14X145"],
        # gable columns + end girders: the pinned wide-flange building the
        # lateral phase converts to portal frames must handle them
        end_girder_candidates=["W12X16", "W16X26", "W21X44"],
        end_wall_columns=2,
        span_ft=50.0, length_ft=60.0, n_frames=3,
        eave_height_ft=20.0, purlin_spacing_ft=5.0,
        superimposed_dead_psf=15.0, live_psf=25.0,
        purlin_Lb_ft=0.0, girder_system="wide_flange",
    )
    base.update(kw)
    return ClearSpanConfig(**base)


def lat_cfg(**kw):
    base = dict(
        # wall braces as W-shapes and roof braces as rods, so the
        # end-to-end run exercises BOTH design paths
        brace_candidates=["W8X24", "W10X33", "W12X40"],
        roof_brace_candidates=["ROD3/4", "ROD7/8", "ROD1", "ROD1-1/4"],
        eave_strut_candidates=["W8X10", "W12X16", "W14X22"],
        wall_strut_candidates=["W8X24", "W10X33"],
    )
    base.update(kw)
    return LateralSystemConfig(**base)


def hazards(**kw):
    base = dict(latitude=29.76, longitude=-95.37, basic_wind_speed_mph=115.0,
                sds_override=0.2, sd1_override=0.1, s1_override=0.08)
    base.update(kw)
    site = SiteConfig(**base)
    return resolve_site_hazards(site)   # fully manual: no network


# --------------------------------------------------- braced-bay statics

def test_bay_forces_hand_calc():
    # V = 10 kip, b = 25 ft, H = 20 ft, 1 tier: L_d = 32.016 ft
    f = bay_forces(10.0, 25.0, 20.0, 1)
    assert f.diag_len_ft == pytest.approx(math.hypot(25.0, 20.0))
    assert f.diag_tension_kip == pytest.approx(10.0 * 32.0156 / 25.0, abs=0.01)
    assert f.strut_axial_kip == pytest.approx(10.0)
    assert f.column_couple_kip == pytest.approx(10.0 * 20.0 / 25.0)
    assert f.anchor_uplift_kip == pytest.approx(8.0)      # tier_h = H
    # 2 tiers: tier_h = 10 ft, L_d = 26.926 ft, anchor pulls V*10/25
    f2 = bay_forces(10.0, 25.0, 20.0, 2)
    assert f2.diag_tension_kip == pytest.approx(10.0 * 26.9258 / 25.0, abs=0.01)
    assert f2.anchor_uplift_kip == pytest.approx(4.0)


def test_bay_drift_hand_calc():
    # V=10, b=300 in, H=240 in, 1 tier, A_d=7.08 (W8X24), A_c=15.6 (W12X53):
    # L_d = 384.187 in
    # diag: 384.187^3/(300^2 * 29000 * 7.08) = 3.068e-3
    # cols: 2*240^3/(3 * 300^2 * 29000 * 15.6) = 2.263e-4
    # delta = 10 * 3.295e-3 = 0.0330 in
    d = bay_drift_in(10.0, 25.0, 20.0, 1, 7.08, 7.08, 15.6, 29000.0)
    assert d == pytest.approx(0.0330, abs=0.0005)


def test_wind_and_seismic_bay_shares():
    wp = design_wind_pressures("z", 115.0, "C", 50.0, 60.0, 20.0)
    f_ww = windward_fraction(wp)
    assert 0.55 < f_ww < 0.70          # 0.8qz vs 0.46qh split
    shares = wind_bay_shares_kip(wp, [0, 1], 2)
    V_line = wp.base_shear_kip / 4.0   # half to the roof, half per line
    assert shares[0] == pytest.approx(f_ww * V_line)
    assert shares[1] == pytest.approx((1 - f_ww) * V_line)
    assert sum(shares.values()) == pytest.approx(V_line)

    seis = seismic_bay_shares_kip(5.0, [0, 1], 2, 30.0)
    assert seis == {0: pytest.approx(2.5), 1: pytest.approx(2.5)}
    seis = seismic_bay_shares_kip(10.0, [0, 6], 7, 25.0)   # tributary halves
    assert seis[0] == pytest.approx(5.0) and seis[6] == pytest.approx(5.0)


def test_roof_truss_forces_hand_calc():
    wp = design_wind_pressures("z", 115.0, "C", 50.0, 60.0, 20.0)
    tr = roof_truss_forces(wp, 50.0, 30.0, 25.0)
    W = max(windward_fraction(wp), 1 - windward_fraction(wp)) \
        * wp.base_shear_kip / 2.0
    assert tr.entering_force_kip == pytest.approx(W)
    assert tr.diag_tension_kip == pytest.approx(
        W / 2.0 * math.hypot(25.0, 30.0) / 30.0)
    assert tr.chord_axial_kip == pytest.approx(W * 50.0 / (8.0 * 30.0))


# --------------------------------------------------------- uplift reversal

def test_uplift_uc_hand_calc():
    shape = get_shapes(["W8X10"])[0]
    # trib 5 ft, SDL 15 psf: wD = 10 + 75 = 85 plf; p_up = 30 psf ->
    # w_net = 150 - 0.9*85 = 73.5 plf up; Mu = 73.5*30^2/8 = 8.27 kip-ft
    uc = uplift_uc(shape, 360.0, 60.0, 0.0, 15.0, 30.0, 50.0, 29000.0)
    assert uc > 0.0
    # a light suction that cannot overcome 0.9D gives exactly zero
    assert uplift_uc(shape, 360.0, 60.0, 0.0, 15.0, 5.0, 50.0, 29000.0) == 0.0
    # a stronger section strictly reduces the unity check
    assert uplift_uc(get_shapes(["W12X16"])[0], 360.0, 60.0, 0.0, 15.0, 30.0,
                     50.0, 29000.0) < uc


def test_active_combos_drop_mirrors_and_pure_ez():
    # Ez combos are braced-bay-only (closed-form); the negative-direction
    # cases are exact mirrors of the positive ones on this doubly symmetric
    # building, so the 3D model solves only the + set: 10 of 22 combos
    combos = all_strength_combos(0.25)
    active = active_strength_combos(combos)
    assert len(combos) == 22 and len(active) == 10
    kept_cases = {c for f in active.values() for c in f}
    assert kept_cases == {"D", "L", "Wx+", "Wz+", "Ex+"}
    # the gravity pair and one of each lateral family survive
    for name in ("1.4D", "1.2D+1.6L", "0.9D+Wx+", "1.2D+Wz++0.5L",
                 "(1.2+0.2SDS)D+Ex+"):
        assert name in active


# ---------------------------------------------------- threaded-rod family

def test_rod_shapes_exact_geometry():
    rod = get_shapes(["ROD1"])[0]
    assert rod.A == pytest.approx(math.pi / 4.0)
    assert rod.weight_plf == pytest.approx(math.pi / 4.0 * 490.0 / 144.0)
    assert rod.ry == pytest.approx(0.25)
    assert get_shapes(["ROD1-1/8"])[0].d == pytest.approx(1.125)
    # rods sort into mixed candidate lists by weight like any section
    mixed = get_shapes(["W8X24", "ROD1"])
    assert [s.name for s in mixed] == ["ROD1", "W8X24"]
    with pytest.raises(ValueError):
        get_shapes(["RODX"])


def test_rod_tension_capacity_hand_calc():
    from frame_optimizer.design.aisc_strengths import rod_tension_capacity
    rod = get_shapes(["ROD1"])[0]
    # A36: yield 0.9*36*0.7854 = 25.45 kip vs threads
    # 0.75*0.75*58*0.7854 = 25.63 kip -> gross yielding governs
    phi_Tn, clause = rod_tension_capacity(rod)
    assert phi_Tn == pytest.approx(25.45, abs=0.02)
    assert "D2" in clause


def test_rod_member_check_is_tension_only():
    from frame_optimizer.analysis.frame_model import MemberDemand
    from frame_optimizer.design import CheckParams, GroupRules, check_member
    rod = get_shapes(["ROD1"])[0]
    demand = MemberDemand(
        name="RB0.0.a", group="roof_brace", story=1, length_in=400.0,
        trib_width_in=0.0, shape_used="ROD1", Ix_used=rod.Ix,
        Pu=20.0, Mux=500.0, Muy=100.0, Vu=5.0,   # flexure/shear must be moot
        defl_total_in=0.0, defl_live_in=0.0, Pu_min=-0.5, Pu_max=20.0)
    params = CheckParams(Fy=50.0, Fu=65.0, E=29000.0,
                         group_rules={"roof_brace": GroupRules()})
    row = check_member(rod, demand, params)
    assert row["governing_limitstate"] == "axial"
    assert row["governing_uc"] == pytest.approx(20.0 / 25.45, abs=0.001)
    assert row["PASS"]
    assert row["UC_Mx"] == 0.0 and row["UC_V"] == 0.0   # not rod limit states


# ------------------------------------------ 3D model wind statics closure

def test_lateral_model_statics():
    config = cfg()
    system = build_lateral_system(config, lat_cfg())
    geometry, a_config = system.geometry, system.analysis_config
    shapes = {s.name: s for s in get_shapes(
        ["W8X24", "W12X53", "W30X108", "W16X26", "W8X10"])}
    assignment = {g: shapes[{"column": "W12X53", "girder": "W30X108",
                             "end_girder": "W16X26",
                             "purlin": "W8X10", "eave_strut": "W8X10",
                             "brace": "W8X24", "roof_brace": "W8X24"}[g]]
                  for g in geometry.groups}

    loads = compute_lateral_loads(config, 30000.0, hazards())
    combos = active_strength_combos(loads.combos)
    model = build_model(geometry, assignment, a_config,
                        strength_combos=combos)   # nominal: no notional loads
    add_lateral_load_cases(model, geometry, a_config, loads)
    model.analyze(check_stability=True, check_statics=False, sparse=True)

    def rxn(component, combo):
        return sum(getattr(model.nodes[n.name], f"Rxn{component}")[combo]
                   for n in geometry.nodes if n.is_base)

    # wind x: the reactions must resist exactly the applied base shear
    assert -rxn("FX", "0.9D+Wx+") == pytest.approx(
        loads.wind_x.base_shear_kip, rel=1e-6)
    # seismic x: the reactions must resist exactly V
    assert -rxn("FX", "(0.9-0.2SDS)D+Ex+") == pytest.approx(
        loads.elf.V_kip, rel=1e-6)
    # z-wind roof uplift lightens the building under 0.9D+Wz
    assert rxn("FY", "0.9D+Wz+") < 0.9 * rxn("FY", "D+L")


# ------------------------------------------------- end-to-end lateral design

def test_design_lateral_end_to_end():
    gravity = optimize(cfg())
    assert gravity.feasible

    result = design_lateral(gravity, hazards(), lat_cfg(),
                            wall_weight_psf=3.0)

    assert result.converged
    assert result.feasible, result.summary()
    assert bool(result.member_table["PASS"].all())
    assert bool(result.drift_table["PASS"].all())

    # ratchet: gravity members never shrink under the combined actions
    catalog = {s.name: s for s in get_shapes(
        list(result.sections.values()) + list(gravity.sections.values()))}
    for g, sel in gravity.sections.items():
        assert catalog[result.sections[g]].weight_plf >= catalog[sel].weight_plf

    # roof braces landed on a rod, sized by the tension-only path
    assert result.sections["roof_brace"].startswith("ROD")

    # every geometry group got a section and appears in the check table
    assert set(result.sections) == set(result.system.geometry.groups)
    assert set(result.member_table["group"]) == set(result.sections)

    # drift table covers both directions
    assert {"x", "z"} == set(result.drift_table["direction"])

    # base reactions: statics closure of the certified service model
    total = sum(v["D+L"]["FY"] for v in result.base_reactions.values())
    from frame_optimizer.lateral_designer import _weights
    steel_lb, _ = _weights(result.system.geometry,
                           {g: catalog[n] for g, n in result.sections.items()})
    area = result.analysis_config.span_ft * result.analysis_config.length_ft
    expected = (steel_lb + (15.0 + 25.0) * area) / 1000.0
    assert total == pytest.approx(expected, rel=1e-4)

    # exports build and serialize
    import json
    from frame_optimizer import lateral_baseplate_inputs, lateral_system_block
    from frame_optimizer.export import building_configuration
    bp = lateral_baseplate_inputs(result)
    assert any("braced_bay_z" in c for c in bp["columns"])
    json.dumps(bp)
    combined = result.as_optimization_result()
    bldg = building_configuration(combined, geometry=result.system.geometry,
                                  lateral_block=lateral_system_block(result))
    assert bldg["lateral_system"]["braced_bay_indices"] == [0, 1]
    json.dumps(bldg)
    assert result.summary()   # renders without error
