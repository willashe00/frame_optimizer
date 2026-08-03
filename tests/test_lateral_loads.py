"""ASCE 7-16 lateral loads: hand-calculation anchors.

Wind anchors reproduce Table 26.10-1 Kz values and a full directional-
procedure base-shear hand calculation; seismic anchors walk Cs through each
governing equation of 12.8.1.1. Combination factors are checked against
2.3.1/2.3.6 written out by hand. The frame_model combo pass-through is
verified on a one-bay frame where doubling the combo factor must double the
enveloped moment.
"""
import json

import pytest

from frame_optimizer import ClearSpanConfig, FrameConfig
from frame_optimizer.analysis import analyze_frame
from frame_optimizer.geometry import build_geometry
from frame_optimizer.lateral_loads import (
    UnsupportedSDCError, all_strength_combos, approximate_period_s,
    design_wind_pressures, effective_seismic_weight_kip, kz,
    lateral_load_basis, lateral_strength_combos, leeward_wall_cp,
    roof_cp_bands, roof_height_ft, seismic_elf, seismic_response_coefficient,
    summarize_lateral_basis, transverse_line_shears_kip,
    tributary_line_lengths_ft, velocity_pressure_psf)
from frame_optimizer.sections import get_shapes
from frame_optimizer.site import SeismicHazard, SiteConfig, SiteHazards


def cs_cfg(**kw):
    base = dict(
        girder_candidates=["W24X76", "W30X108", "W33X130"],
        purlin_candidates=["W8X10", "W12X16"],
        column_candidates=["W10X33", "W12X53"],
        span_ft=50.0, length_ft=60.0, n_frames=3,
        eave_height_ft=20.0, purlin_spacing_ft=5.0,
        superimposed_dead_psf=15.0, live_psf=25.0,
        purlin_Lb_ft=0.0,
    )
    base.update(kw)
    return ClearSpanConfig(**base)


def hazard(**kw):
    base = dict(sds=0.25, sd1=0.12, s1=0.08, sdc="B", tl_s=12.0)
    base.update(kw)
    return SeismicHazard(**base)


def houston_site(**kw):
    base = dict(latitude=29.76, longitude=-95.37, basic_wind_speed_mph=115.0)
    base.update(kw)
    return SiteConfig(**base)


# -------------------------------------------------- wind: Kz and qz anchors

@pytest.mark.parametrize("z, exposure, expected", [
    (30.0, "C", 0.98),   # Table 26.10-1 values
    (15.0, "C", 0.85),
    (10.0, "C", 0.85),   # below 15 ft evaluates at 15 ft
    (15.0, "B", 0.57),
    (30.0, "D", 1.16),
])
def test_kz_matches_table(z, exposure, expected):
    assert kz(z, exposure) == pytest.approx(expected, abs=0.006)


def test_velocity_pressure_hand_calc():
    # qz = 0.00256 * 0.9823 * 0.85 * 115^2 = 28.27 psf
    q = velocity_pressure_psf(30.0, 115.0, "C")
    assert q == pytest.approx(28.27, abs=0.05)


@pytest.mark.parametrize("l_over_b, expected", [
    (0.5, -0.5), (1.0, -0.5), (1.5, -0.4), (2.0, -0.3),
    (3.0, -0.25), (4.0, -0.2), (6.0, -0.2),
])
def test_leeward_cp(l_over_b, expected):
    assert leeward_wall_cp(l_over_b) == pytest.approx(expected)


def test_roof_bands_low_rise():
    # h/L = 0.2 <= 0.5: -0.9 to h, -0.5 to 2h, -0.3 beyond (Fig 27.3-1)
    assert roof_cp_bands(20.0, 100.0) == [
        (0.0, 20.0, pytest.approx(-0.9)),
        (20.0, 40.0, pytest.approx(-0.5)),
        (40.0, 100.0, pytest.approx(-0.3)),
    ]


def test_roof_bands_interpolate_on_h_over_L():
    # h/L = 0.75 -> halfway between the 0.5 and 1.0 band sets
    assert roof_cp_bands(30.0, 40.0) == [
        (0.0, 15.0, pytest.approx(-1.1)),
        (15.0, 30.0, pytest.approx(-0.8)),
        (30.0, 40.0, pytest.approx(-0.6)),
    ]


def test_roof_bands_tall():
    # h/L >= 1.0: -1.3 to h/2, -0.7 beyond
    assert roof_cp_bands(50.0, 40.0) == [
        (0.0, 25.0, pytest.approx(-1.3)),
        (25.0, 40.0, pytest.approx(-0.7)),
    ]


def test_directional_procedure_hand_calc():
    """50 x 60 ft, h = 20 ft, V = 115 mph, Exposure C, wind across the span.

    Hand calc: q15 = 24.43, qh = q20 = 25.95 psf. Windward 0.85*0.8*qz:
    16.61 psf (0-15 ft), 17.65 psf (15-20 ft); leeward L/B = 50/60 ->
    Cp = -0.5, p = -11.03 psf. Base shear = (16.61*15 + 17.65*5 + 11.03*20)
    * 60 ft / 1000 = 33.5 kip. Roof (h/L = 0.4): external -0.9/-0.5/-0.3 *
    qh*G with +GCpi = 0.18*qh inside -> uplift = 55.0 kip.
    """
    wp = design_wind_pressures("x", 115.0, "C", 50.0, 60.0, 20.0)
    assert wp.B_ft == 60.0 and wp.L_ft == 50.0
    assert wp.qh_psf == pytest.approx(25.95, abs=0.05)
    assert wp.windward_bands[0].pressure_psf == pytest.approx(16.61, abs=0.05)
    assert wp.windward_bands[1].pressure_psf == pytest.approx(17.65, abs=0.05)
    assert wp.leeward_psf == pytest.approx(-11.03, abs=0.05)
    assert wp.base_shear_kip == pytest.approx(33.48, abs=0.15)
    assert not wp.governed_by_minimum
    assert wp.roof_net_uplift_kip == pytest.approx(55.05, abs=0.3)


def test_minimum_wind_pressure_floor():
    # V = 40 mph scales the computed shear by (40/115)^2 -> ~4 kip, below
    # the 27.1.5 floor of 16 psf * 60 ft * 20 ft = 19.2 kip
    wp = design_wind_pressures("x", 40.0, "C", 50.0, 60.0, 20.0)
    assert wp.governed_by_minimum
    assert wp.base_shear_kip == pytest.approx(19.2)
    assert wp.base_shear_computed_kip < 19.2


def test_wind_direction_z_swaps_plan_dimensions():
    wp = design_wind_pressures("z", 115.0, "C", 50.0, 60.0, 20.0)
    assert wp.B_ft == 50.0 and wp.L_ft == 60.0
    assert wp.leeward_psf == pytest.approx(
        25.95 * 0.85 * leeward_wall_cp(60.0 / 50.0), abs=0.05)


# ------------------------------------------------------------- distribution

def test_tributary_line_lengths():
    assert tributary_line_lengths_ft(4, 60.0) == [10.0, 20.0, 20.0, 10.0]
    assert tributary_line_lengths_ft(2, 25.0) == [12.5, 12.5]
    assert tributary_line_lengths_ft(1, 40.0) == [40.0]


def test_transverse_line_shears_sum_to_total():
    shears = transverse_line_shears_kip(30.0, cs_cfg())
    assert sum(shears) == pytest.approx(30.0)
    assert shears[0] == pytest.approx(shears[1] / 2.0)   # end frame half trib


# ------------------------------------------------------------------- seismic

def test_approximate_period():
    assert approximate_period_s(30.0) == pytest.approx(0.2564, abs=0.001)


def test_effective_seismic_weight_hand_calc():
    # steel 20 kip + SDL 15 psf * 3000 ft2 = 45 kip + walls 3 psf * 220 ft
    # perimeter * 10 ft (upper half of 20 ft) = 6.6 kip
    W = effective_seismic_weight_kip(cs_cfg(), 20000.0, wall_weight_psf=3.0)
    assert W == pytest.approx(20.0 + 45.0 + 6.6)
    # snow option: +0.2 * 25 psf * 3000 ft2 = 15 kip
    W_snow = effective_seismic_weight_kip(cs_cfg(), 20000.0,
                                          wall_weight_psf=3.0,
                                          include_live_fraction=0.2)
    assert W_snow == pytest.approx(W + 15.0)


@pytest.mark.parametrize("sds, sd1, s1, T, R, tl, cs, eq", [
    (1.0, 0.4, 0.3, 0.2564, 3.0, 12.0, 1.0 / 3.0, "12.8-2"),   # plateau
    (1.0, 0.4, 0.3, 1.5, 3.0, 12.0, 0.0889, "12.8-3"),         # 1/T cap
    (1.0, 1.0, 0.3, 13.0, 1.0, 12.0, 12.0 / 169.0, "12.8-4"),  # beyond TL
    (0.3, 0.05, 0.1, 2.0, 3.0, 12.0, 0.0132, "12.8-5"),        # 0.044*SDS
    (1.0, 0.4, 0.65, 3.0, 3.0, 12.0, 0.1083, "12.8-6"),        # S1 >= 0.6
])
def test_cs_governing_equations(sds, sd1, s1, T, R, tl, cs, eq):
    got, governing = seismic_response_coefficient(sds, sd1, s1, T, R, 1.0, tl)
    assert got == pytest.approx(cs, abs=0.0005)
    assert governing == eq


def test_seismic_elf_basics():
    elf = seismic_elf(hazard(), houston_site(), W_kip=100.0, hn_ft=20.0)
    assert elf.Cs == pytest.approx(0.25 / 3.0)
    assert elf.V_kip == pytest.approx(100.0 * 0.25 / 3.0)
    assert elf.rho == 1.0 and elf.R == 3.0


def test_sdc_a_uses_1_4_2_minimum():
    elf = seismic_elf(hazard(sds=0.07, sd1=0.06, s1=0.04, sdc="A"),
                      houston_site(), W_kip=100.0, hn_ft=20.0)
    assert elf.Cs == pytest.approx(0.01)
    assert "1.4.2" in elf.Cs_governing
    assert elf.V_kip == pytest.approx(1.0)


def test_sdc_d_refused():
    with pytest.raises(UnsupportedSDCError, match="SDC B and C"):
        seismic_elf(hazard(sds=1.0, sd1=0.6, s1=0.5, sdc="D"),
                    houston_site(), W_kip=100.0, hn_ft=20.0)


def test_roof_height_includes_truss_depth():
    assert roof_height_ft(cs_cfg()) == 20.0
    truss = cs_cfg(girder_system="truss", girder_candidates=[],
                   truss_chord_candidates=["W10X33", "W12X53"],
                   truss_web_candidates=["W8X10"], truss_depth_ft=5.0)
    assert roof_height_ft(truss) == 25.0


# ---------------------------------------------------------------- combos

def test_lateral_combos_factors_written_by_hand():
    combos = lateral_strength_combos(0.5)
    assert len(combos) == 4 * 3 + 4 * 2
    assert combos["1.2D+1.6L+0.5Wx+"] == {"D": 1.2, "L": 1.6, "Wx+": 0.5}
    assert combos["1.2D+Wx-+0.5L"] == {"D": 1.2, "L": 0.5, "Wx-": 1.0}
    assert combos["0.9D+Wz+"] == {"D": 0.9, "Wz+": 1.0}
    assert combos["(1.2+0.2SDS)D+Ex+"] == {"D": 1.3, "Ex+": 1.0}
    assert combos["(0.9-0.2SDS)D+Ez-"] == {"D": 0.8, "Ez-": 1.0}


def test_snow_joins_seismic_combos_only_when_declared():
    dry = lateral_strength_combos(0.5)
    snow = lateral_strength_combos(0.5, live_is_snow=True)
    assert all("L" not in f for n, f in dry.items() if "E" in n)
    assert snow["(1.2+0.2SDS)D+Ex++0.2S"]["L"] == pytest.approx(0.2)
    assert "L" not in snow["(0.9-0.2SDS)D+Ex+"]


def test_all_strength_combos_keeps_gravity():
    combos = all_strength_combos(0.25)
    assert combos["1.4D"] == {"D": 1.4}
    assert combos["1.2D+1.6L"] == {"D": 1.2, "L": 1.6}
    assert len(combos) == 2 + 20


# ------------------------------------------------ frame_model combo plumbing

def test_analyze_frame_accepts_custom_combos():
    config = FrameConfig(
        beam_candidates=["W18X35"], column_candidates=["W10X33"],
        superimposed_dead_psf=20.0, live_psf=10.0,
    )
    geometry = build_geometry(config)
    assignment = {"beam": get_shapes(["W18X35"])[0],
                  "column": get_shapes(["W10X33"])[0]}
    single = analyze_frame(geometry, assignment, config,
                           strength_combos={"1.0D": {"D": 1.0}})
    double = analyze_frame(geometry, assignment, config,
                           strength_combos={"2.0D": {"D": 2.0}})
    beams = {d.name: d for d in single if d.group == "beam"}
    for d in double:
        if d.group == "beam" and d.Mux > 0:
            assert d.Mux == pytest.approx(2.0 * beams[d.name].Mux, rel=1e-6)


# ------------------------------------------------------- the full load basis

def test_lateral_load_basis_end_to_end():
    config = cs_cfg()
    hazards = SiteHazards(site=houston_site(), seismic=hazard(),
                          basic_wind_speed_mph=115.0)
    basis = lateral_load_basis(config, steel_weight_lb=20000.0,
                               hazards=hazards, wall_weight_psf=3.0)

    elf = basis["seismic"]["elf"]
    assert elf["V_kip"] == pytest.approx(elf["Cs"] * elf["W_kip"], abs=0.01)
    frames = basis["seismic"]["per_frame_line_kip"]
    assert len(frames) == config.n_frames
    assert sum(frames) == pytest.approx(elf["V_kip"], abs=0.01)
    assert basis["seismic"]["per_sidewall_line_kip"] == pytest.approx(
        elf["V_kip"] / 2.0, abs=0.01)

    x = basis["wind"]["x"]
    assert sum(x["per_frame_line_kip"]) == pytest.approx(
        x["base_shear_kip"], abs=0.01)
    assert basis["wind"]["z"]["per_sidewall_line_kip"] == pytest.approx(
        basis["wind"]["z"]["base_shear_kip"] / 2.0, abs=0.01)

    assert len(basis["load_combinations"]) == 22
    json.dumps(basis)   # must be JSON-serializable as written

    report = summarize_lateral_basis(basis)
    assert "base shear" in report and "SDC B" in report
