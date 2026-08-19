"""Baseplate design: closed-form anchors, detailing rules, end-to-end.

Anchors follow the existing test philosophy: every capacity equation is
reproduced by hand from AISC 360-22 / DG1 in the backend's native kips and
inches, and the sizing routines are checked by proving they invert those
equations. The uniform-design layer is checked on the property that matters
structurally -- ONE plate has to satisfy EVERY column, including the ones it
was not sized for.
"""
import json
import math
from dataclasses import replace

import pytest

from baseplate_design import (BaseplateConfig, ColumnDemand, PlateGeometry,
                              baseplate_design_configuration, check_plate,
                              demands_from_inputs, design_plate,
                              design_uniform_baseplate, effective_A2,
                              required_A1, write_baseplate_design_json)
from baseplate_design.baseplate_design import (FNV_C, MAX_CONFINEMENT, PHI_B,
                                               PHI_C, PHI_V, ceil_to)
from baseplate_design.config import KN_TO_KIP
from frame_optimizer import ClearSpanConfig, baseplate_inputs, optimize_layout
from frame_optimizer.config import IN_TO_MM, KIP_TO_KN, MM_TO_IN

# f'c = 4.0 ksi and Fy = 36 ksi exactly, so hand calcs stay in imperial
FC_KSI = 4.0
FY_KSI = 36.0
KSI_TO_MPA = 6.894757293168361


def config(**overrides) -> BaseplateConfig:
    """Config whose imperial values are exact, for hand-calc comparison."""
    base = dict(fc_mpa=FC_KSI * KSI_TO_MPA, plate_Fy_mpa=FY_KSI * KSI_TO_MPA,
                anchor_Fu_mpa=58.0 * KSI_TO_MPA)
    return BaseplateConfig(**{**base, **overrides})


def demand(**overrides) -> ColumnDemand:
    base = dict(column_id="C1", d=10.1, bf=10.0, Pu=350.0, Vu=25.0,
                P_friction=350.0)
    return ColumnDemand(**{**base, **overrides})


# ---------------------------------------------------------------------------
# rounding
# ---------------------------------------------------------------------------
def test_ceil_to_rounds_up():
    assert ceil_to(12.01, 0.5) == pytest.approx(12.5)
    assert ceil_to(1.13, 0.125) == pytest.approx(1.25)


def test_ceil_to_does_not_charge_an_extra_increment_for_float_dust():
    """12.7 mm -> in is 0.49999999999999994, so a dimension already on the
    increment must not round up again (half an inch on every plate)."""
    step = 12.7 * MM_TO_IN            # nominal 1/2 in, one ulp low
    assert ceil_to(12.0, step) == pytest.approx(12.0, abs=1e-9)
    assert ceil_to(16.0, step) == pytest.approx(16.0, abs=1e-9)
    # a real overshoot still rounds up
    assert ceil_to(12.01, step) == pytest.approx(12.5, abs=1e-9)


# ---------------------------------------------------------------------------
# bearing: AISC 360-22 Eq. J8-2
# ---------------------------------------------------------------------------
def test_required_A1_matches_hand_calc():
    Pu = 350.0
    q = PHI_C * 0.85 * FC_KSI
    # confinement-capped branch governs when A2 is generous
    assert required_A1(Pu, FC_KSI, 10_000.0) == pytest.approx(Pu / (q * 2.0))
    # sqrt(A1*A2) branch governs when A2 is tight
    assert required_A1(Pu, FC_KSI, 200.0) == pytest.approx(Pu ** 2 / (q ** 2 * 200.0))


def test_bearing_capacity_matches_eq_J8_2():
    cfg = config(pedestal_width_mm=1000.0, pedestal_length_mm=1000.0)
    plate = PlateGeometry(B=14.0, N=18.0, tp=1.0, d_rod=0.75, n_rods=4,
                          edge_distance=1.5)
    check = check_plate(plate, demand(), cfg)
    A1 = 14.0 * 18.0
    conf = min(math.sqrt(check.A2_effective / A1), MAX_CONFINEMENT)
    assert check.phiPp == pytest.approx(PHI_C * 0.85 * FC_KSI * A1 * conf)
    assert check.bearing_dcr == pytest.approx(350.0 / check.phiPp)


def test_A2_is_the_largest_geometrically_similar_area():
    """AISC J8 needs A2 similar to and concentric with A1, so the tight pier
    direction governs both directions."""
    cfg = config(pedestal_width_mm=40.0 * IN_TO_MM,   # generous in B
                 pedestal_length_mm=20.0 * IN_TO_MM)  # tight in N
    A2 = effective_A2(B=10.0, N=16.0, config=cfg)
    k = min(40.0 / 10.0, 20.0 / 16.0)                 # = 1.25, the N direction
    assert A2 == pytest.approx(k ** 2 * 10.0 * 16.0)


def test_plate_too_big_for_the_pier_is_an_error():
    cfg = config(pedestal_width_mm=200.0, pedestal_length_mm=200.0)
    with pytest.raises(ValueError, match="does not fit"):
        effective_A2(B=20.0, N=24.0, config=cfg)


# ---------------------------------------------------------------------------
# plate flexure: DG1 Eq. 3.3.17
# ---------------------------------------------------------------------------
def test_thickness_matches_DG1_hand_calc():
    cfg = config(pedestal_width_mm=1000.0, pedestal_length_mm=1000.0)
    dm = demand()
    plate = PlateGeometry(B=14.0, N=18.0, tp=1.0, d_rod=0.75, n_rods=4,
                          edge_distance=1.5)
    c = check_plate(plate, dm, cfg)

    m = (18.0 - 0.95 * dm.d) / 2.0
    n = (14.0 - 0.80 * dm.bf) / 2.0
    X = (4.0 * dm.d * dm.bf / (dm.d + dm.bf) ** 2) * c.bearing_dcr
    lam = min(2.0 * math.sqrt(X) / (1.0 + math.sqrt(1.0 - X)), 1.0)
    ell = max(m, n, lam * math.sqrt(dm.d * dm.bf) / 4.0)
    assert c.m == pytest.approx(m)
    assert c.n == pytest.approx(n)
    assert c.ell == pytest.approx(ell)
    assert c.t_req == pytest.approx(
        ell * math.sqrt(2.0 * dm.Pu / (PHI_B * FY_KSI * 14.0 * 18.0)))


def test_lambda_is_capped_at_one_when_bearing_is_fully_utilized():
    """X >= 1 has no real sqrt(1-X); lambda must clamp, not blow up."""
    cfg = config(pedestal_edge_projection_mm=0.0)
    plate = PlateGeometry(B=10.0, N=12.0, tp=1.0, d_rod=0.75, n_rods=4,
                          edge_distance=1.5)
    c = check_plate(plate, demand(Pu=2000.0), cfg)
    assert c.bearing_dcr > 1.0          # deliberately overloaded
    assert c.lam == 1.0


def test_designed_plate_is_never_below_its_own_required_thickness():
    cfg = config()
    for Pu in (50.0, 350.0, 900.0):
        dm = demand(Pu=Pu, P_friction=Pu)
        plate = design_plate(dm, cfg)
        check = check_plate(plate, dm, cfg)
        assert check.flexure_dcr <= 1.0
        # and not wastefully thick: one increment less would fail
        thinner = replace(plate, tp=plate.tp - cfg.thickness_increment_in)
        assert (thinner.tp < cfg.min_thickness_in
                or check_plate(thinner, dm, cfg).flexure_dcr > 1.0)


# ---------------------------------------------------------------------------
# anchor rod shear
# ---------------------------------------------------------------------------
def test_shear_capacity_is_friction_plus_rods():
    cfg = config()
    plate = PlateGeometry(B=14.0, N=18.0, tp=1.0, d_rod=0.75, n_rods=4,
                          edge_distance=1.5)
    dm = demand(Vu=60.0, P_friction=100.0)
    c = check_plate(plate, dm, cfg)

    friction = PHI_V * cfg.friction_coefficient * 100.0
    A_rod = math.pi * 0.75 ** 2 / 4.0
    phiRnv = PHI_V * FNV_C * 58.0 * A_rod * 4
    assert c.friction == pytest.approx(friction)
    assert c.V_rods == pytest.approx(60.0 - friction)
    assert c.phiVn == pytest.approx(friction + phiRnv)
    assert c.shear_dcr == pytest.approx(60.0 / (friction + phiRnv))


def test_friction_uses_the_coincident_axial_not_the_governing_one():
    """The credit mu*P must ride on the MINIMUM compression present with the
    shear. Two columns with equal Pu but different dead load must not get the
    same rod check."""
    cfg = config()
    plate = PlateGeometry(B=14.0, N=18.0, tp=1.0, d_rod=0.75, n_rods=4,
                          edge_distance=1.5)
    heavy = check_plate(plate, demand(Vu=60.0, P_friction=200.0), cfg)
    light = check_plate(plate, demand(Vu=60.0, P_friction=20.0), cfg)
    assert light.shear_dcr > heavy.shear_dcr


def test_rods_grow_when_friction_cannot_cover_the_shear():
    cfg = config()
    small = design_plate(demand(Vu=5.0, P_friction=200.0), cfg)
    big = design_plate(demand(Vu=400.0, P_friction=0.0), cfg)
    assert small.d_rod == pytest.approx(cfg.min_rod_diameter_in)
    assert big.d_rod > small.d_rod
    assert check_plate(big, demand(Vu=400.0, P_friction=0.0), cfg).shear_dcr <= 1.0


# ---------------------------------------------------------------------------
# detailing: the plate has to be buildable
# ---------------------------------------------------------------------------
def test_plate_covers_the_column_with_a_real_projection():
    cfg = config()
    dm = demand(Pu=1.0, P_friction=1.0)          # bearing demands nothing
    plate = design_plate(dm, cfg)
    assert plate.B >= dm.bf + 2 * cfg.min_plate_projection_in - 1e-9
    assert plate.N >= dm.d + 2 * cfg.min_plate_projection_in - 1e-9


def test_rods_land_clear_of_the_column_flanges():
    cfg = config()
    dm = demand(Pu=1.0, P_friction=1.0)
    plate = design_plate(dm, cfg)
    check = check_plate(plate, dm, cfg)
    assert check.clearance_ok
    assert check.rod_clearance >= cfg.rod_clearance_in - 1e-9
    # every rod is outside the column footprint in the N direction
    for _, y in plate.rod_positions():
        assert abs(y) >= dm.d / 2.0


def test_edge_distance_tracks_the_rod_diameter():
    cfg = config(min_edge_distance_mm=1.0)       # detailing floor removed
    plate = design_plate(demand(Vu=400.0, P_friction=0.0), cfg)
    assert plate.edge_distance == pytest.approx(
        cfg.edge_distance_rod_factor * plate.d_rod)


def test_rod_positions_are_symmetric_and_inside_the_plate():
    plate = PlateGeometry(B=14.0, N=18.0, tp=1.0, d_rod=0.75, n_rods=4,
                          edge_distance=1.5)
    positions = plate.rod_positions()
    assert len(positions) == 4
    assert sum(x for x, _ in positions) == pytest.approx(0.0)
    assert sum(y for _, y in positions) == pytest.approx(0.0)
    for x, y in positions:
        assert abs(x) == pytest.approx(14.0 / 2 - 1.5)
        assert abs(y) == pytest.approx(18.0 / 2 - 1.5)


def test_six_rods_split_evenly_outside_each_flange():
    plate = PlateGeometry(B=20.0, N=18.0, tp=1.0, d_rod=0.75, n_rods=6,
                          edge_distance=1.5)
    positions = plate.rod_positions()
    assert len(positions) == 6
    assert sorted({round(x, 6) for x, _ in positions}) == [-8.5, 0.0, 8.5]
    assert sorted({round(y, 6) for _, y in positions}) == [-7.5, 7.5]


def test_uplift_is_rejected_rather_than_designed():
    with pytest.raises(ValueError, match="must be >= 0"):
        demand(Pu=-10.0)


# ---------------------------------------------------------------------------
# design_plate inverts the checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("Pu,Vu,P_fric,d,bf", [
    (50.0, 0.0, 45.0, 9.73, 7.96),        # light W10x33
    (350.0, 25.0, 120.0, 10.1, 10.0),
    (1200.0, 200.0, 400.0, 14.78, 15.5),  # heavy W14x145
    (2000.0, 500.0, 0.0, 14.78, 15.5),    # no friction credit at all
])
def test_designed_plate_passes_every_limit_state(Pu, Vu, P_fric, d, bf):
    cfg = config()
    dm = demand(Pu=Pu, Vu=Vu, P_friction=P_fric, d=d, bf=bf)
    check = check_plate(design_plate(dm, cfg), dm, cfg)
    assert check.passes, (check.bearing_dcr, check.flexure_dcr, check.shear_dcr)


def test_design_respects_a_floor_plate():
    cfg = config()
    dm = demand(Pu=50.0, P_friction=50.0)
    floor = PlateGeometry(B=30.0, N=36.0, tp=2.0, d_rod=1.5, n_rods=4,
                          edge_distance=2.25)
    grown = design_plate(dm, cfg, floor=floor)
    assert grown.B >= floor.B and grown.N >= floor.N
    assert grown.tp >= floor.tp and grown.d_rod >= floor.d_rod


def test_A2_and_plate_size_are_self_consistent():
    """The A2 the design used must be the A2 the final plate actually gets."""
    cfg = config(pedestal_edge_projection_mm=100.0)
    dm = demand(Pu=1500.0, P_friction=500.0)
    plate = design_plate(dm, cfg)
    check = check_plate(plate, dm, cfg)
    assert check.A2_effective == pytest.approx(
        effective_A2(plate.B, plate.N, cfg))
    assert check.bearing_dcr <= 1.0


# ---------------------------------------------------------------------------
# uniform design: one plate, every column
# ---------------------------------------------------------------------------
def _inputs(columns) -> dict:
    """Minimal baseplate_inputs() payload for the given (id, Pu_kN, D_kN)."""
    return {
        "schema": "frame_optimizer/baseplate_inputs",
        "columns": [{
            "member_id": cid,
            "base_node": f"NB{cid}",
            "section": {"name": "W14X145", "depth_d_mm": 14.78 * IN_TO_MM,
                        "flange_width_bf_mm": 15.5 * IN_TO_MM},
            "centerline_location": {"x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0},
            "axial_compression_kN": {"Pu_governing_lrfd": Pu, "by_combo": {"D": D}},
            "base_shear_kN": {"Vu_governing_lrfd": 0.0},
        } for cid, Pu, D in columns],
    }


SPREAD = [("C0", 90.0, 55.0), ("C1", 740.0, 280.0), ("C2", 1500.0, 600.0)]


def test_one_plate_satisfies_every_column():
    design = design_uniform_baseplate(_inputs(SPREAD),
                                      config(design_base_shear_kN=60.0))
    assert design.feasible
    assert design.n_columns == 3
    # the single plate, re-checked independently against each column
    for dm in design.demands:
        assert check_plate(design.plate, dm, design.config).passes


def test_heaviest_column_governs_the_plate_lightest_governs_the_rods():
    design = design_uniform_baseplate(_inputs(SPREAD),
                                      config(design_base_shear_kN=60.0))
    gov = design.governing_column
    assert gov["bearing"] == "C2"           # largest Pu
    assert gov["plate flexure"] == "C2"
    assert gov["anchor rod shear"] == "C0"  # smallest friction credit


def test_uniform_plate_is_the_envelope_of_the_individual_designs():
    cfg = config(design_base_shear_kN=60.0)
    design = design_uniform_baseplate(_inputs(SPREAD), cfg)
    individually = [design_plate(dm, cfg) for dm in design.demands]
    assert design.plate.B >= max(p.B for p in individually)
    assert design.plate.N >= max(p.N for p in individually)
    assert design.plate.tp >= max(p.tp for p in individually)
    assert design.plate.d_rod >= max(p.d_rod for p in individually)


def test_envelope_accounts_for_thickness_growing_with_plan_size():
    """Growing B and N for the heavy column lengthens the cantilevers, so the
    other columns need a thicker plate than their own design asked for. The
    fixed point has to catch that, not just max the first-pass results."""
    cfg = config(design_base_shear_kN=60.0)
    design = design_uniform_baseplate(_inputs(SPREAD), cfg)
    for dm in design.demands:
        alone = design_plate(dm, cfg)
        at_final_size = check_plate(design.plate, dm, cfg)
        assert at_final_size.flexure_dcr <= 1.0
        if alone.B < design.plate.B or alone.N < design.plate.N:
            # its own thickness was computed on a smaller plate
            assert design.plate.tp >= alone.tp


def test_identical_columns_give_the_same_plate_as_designing_one():
    cfg = config()
    same = [("C0", 740.0, 280.0), ("C1", 740.0, 280.0)]
    design = design_uniform_baseplate(_inputs(same), cfg)
    assert design.plate == design_plate(design.demands[0], cfg)


def test_demands_use_the_factored_dead_load_for_friction():
    cfg = config(friction_axial_factor=0.9)
    demands = demands_from_inputs(_inputs(SPREAD), cfg)
    assert demands[1].P_friction == pytest.approx(0.9 * 280.0 * KN_TO_KIP)
    assert demands[1].Pu == pytest.approx(740.0 * KN_TO_KIP)


def test_design_shear_comes_from_the_config_not_the_gravity_model():
    demands = demands_from_inputs(_inputs(SPREAD),
                                  config(design_base_shear_kN=75.0))
    assert all(d.Vu == pytest.approx(75.0 * KN_TO_KIP) for d in demands)


def test_no_columns_is_a_clear_error():
    with pytest.raises(ValueError, match="no 'columns'"):
        design_uniform_baseplate({"columns": []}, config())


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------
def test_export_round_trips_and_reports_the_governing_column(tmp_path):
    design = design_uniform_baseplate(_inputs(SPREAD),
                                      config(design_base_shear_kN=60.0))
    path = write_baseplate_design_json(design, tmp_path / "baseplate_design.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["schema"] == "baseplate_design/baseplate_design"
    assert data["verification"]["all_columns_pass"] is True
    assert data["verification"]["n_columns_checked"] == 3
    assert len(data["columns"]) == 3
    assert data["governing"]["by_limit_state"]["bearing"]["member_id"] == "C2"
    assert (data["governing"]["by_limit_state"]["anchor rod shear"]["member_id"]
            == "C0")

    plate = data["baseplate"]["plate"]
    assert plate["width_B_mm"] == pytest.approx(design.plate.B * IN_TO_MM, abs=0.1)
    assert plate["thickness_tp_mm"] == pytest.approx(design.plate.tp * IN_TO_MM,
                                                     abs=0.01)
    assert len(data["baseplate"]["anchor_rods"]["positions_mm"]) == 4
    assert data["totals"]["n_baseplates"] == 3


def test_export_dcrs_match_the_checks():
    design = design_uniform_baseplate(_inputs(SPREAD),
                                      config(design_base_shear_kN=60.0))
    data = baseplate_design_configuration(design)
    for row in data["columns"]:
        cid = row["member_id"]
        check, dm = design.check_for(cid), design.demand_for(cid)
        assert row["dcr"]["bearing"] == pytest.approx(check.bearing_dcr, abs=1e-4)
        assert row["dcr"]["plate_flexure"] == pytest.approx(check.flexure_dcr,
                                                            abs=1e-4)
        assert row["dcr"]["anchor_rod_shear"] == pytest.approx(check.shear_dcr,
                                                               abs=1e-4)
        assert row["demands_kN"]["Pu_governing_lrfd"] == pytest.approx(
            dm.Pu * KIP_TO_KN, abs=0.01)
        assert row["PASS"] is True


# ---------------------------------------------------------------------------
# end to end, off a real optimization
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def optimized():
    cfg = ClearSpanConfig(
        span_m=20.0, length_m=24.0, height_m=6.0,
        girder_candidates=["W24X76", "W30X108"],
        purlin_candidates=["W8X10", "W12X14"],
        column_candidates=["W10X33", "W12X53"],
        superimposed_dead_kpa=0.72, live_kpa=1.2,
    )
    return optimize_layout(cfg, verbose=False)


def test_baseplate_inputs_carry_base_shear(optimized):
    inputs = baseplate_inputs(optimized)
    assert inputs["schema_version"] == 3
    # the fully pinned gravity model produces no base shear, by construction
    for col in inputs["columns"]:
        assert col["base_shear_kN"]["Vu_governing_lrfd"] == pytest.approx(0.0,
                                                                          abs=1e-6)


def test_design_runs_off_an_optimization_result(optimized):
    design = design_uniform_baseplate(optimized, config())
    assert design.feasible
    assert design.n_columns == len(baseplate_inputs(optimized)["columns"])
    assert not design.has_design_shear          # no lateral in the gravity model
    assert "uniform column baseplate result" in design.summary()


def test_result_and_prebuilt_inputs_give_the_same_design(optimized):
    cfg = config(design_base_shear_kN=30.0)
    from_result = design_uniform_baseplate(optimized, cfg)
    from_dict = design_uniform_baseplate(baseplate_inputs(optimized), cfg)
    assert from_result.plate == from_dict.plate


def test_governing_column_is_the_most_loaded_one(optimized):
    design = design_uniform_baseplate(optimized, config())
    heaviest = max(design.demands, key=lambda d: d.Pu).column_id
    assert design.governing_column["plate flexure"] == heaviest
