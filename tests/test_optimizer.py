import pytest

from frame_optimizer import FrameConfig, evaluate, optimize
from frame_optimizer.config import FT_TO_M

# exact psf -> kPa, so the imperial hand-calc anchors in the comments hold
PSF_TO_KPA = 4.4482216152605 / 0.3048**2 / 1000.0


def small_config(**kw):
    base = dict(
        beam_candidates=["W10X12", "W12X16", "W14X22"],
        column_candidates=["W6X9", "W8X24"],
        x_bays=1, x_bay_spacing_m=20.0 * FT_TO_M,
        z_bays=1, z_bay_spacing_m=20.0 * FT_TO_M,
        stories=1, story_height_m=10.0 * FT_TO_M,
        superimposed_dead_kpa=20.0 * PSF_TO_KPA, live_kpa=50.0 * PSF_TO_KPA,
        deck_span_direction="z",
        beam_Lb_m=0.0,   # deck-braced compression flange
    )
    base.update(kw)
    return FrameConfig(**base)


def test_iterative_finds_feasible_lightest():
    result = optimize(small_config())
    assert result.feasible and result.converged
    assert bool(result.member_table["PASS"].all())
    # W10X12 fails flexure on the loaded beams at this load/span
    # (phiMp = 47.2 < Mu ~ 53 kip-ft), W12X16 passes; the shared beam size is
    # governed by the loaded members while the zero-tributary ones ride along
    assert result.sections == {"beam": "W12X16", "column": "W6X9"}


def test_iterative_matches_exhaustive():
    it = optimize(small_config())
    ex = optimize(small_config(), method="exhaustive")
    assert ex.feasible
    assert it.sections == ex.sections
    assert it.total_weight_kg == pytest.approx(ex.total_weight_kg)


def test_infeasible_candidates_reported():
    result = optimize(small_config(beam_candidates=["W10X12"]))
    assert not result.feasible
    assert not bool(result.member_table["PASS"].all())


def test_unbraced_beam_needs_heavier_section():
    braced = optimize(small_config())
    unbraced = optimize(small_config(beam_Lb_m=None))   # Lb = full 20-ft span
    assert unbraced.total_weight_kg >= braced.total_weight_kg
    if unbraced.feasible:
        assert unbraced.member_table["PASS"].all()


def test_evaluate_explicit_design():
    result = evaluate(small_config(), {"beam": "W14X22", "column": "W8X24"})
    assert result.sections == {"beam": "W14X22", "column": "W8X24"}
    assert bool(result.member_table["PASS"].all())
    assert result.total_weight_kg > 0


def test_warm_start_reaches_the_same_design():
    cold = optimize(small_config())
    warm = optimize(small_config(), warm_start=cold.sections)
    assert warm.feasible and warm.converged
    assert warm.sections == cold.sections
    # a valid warm start converges on the first fixed-point iteration
    assert len(warm.iterations) == 1
    # a bogus warm start is ignored, not an error
    ignored = optimize(small_config(), warm_start={"beam": "W44X230"})
    assert ignored.sections == cold.sections


def test_deflection_can_govern():
    # long span, light loads: strength is easy but L/360 live deflection bites
    config = small_config(
        beam_candidates=["W12X16", "W14X22", "W16X26", "W18X35"],
        x_bay_spacing_m=30.0 * FT_TO_M,
        superimposed_dead_kpa=5.0 * PSF_TO_KPA, live_kpa=40.0 * PSF_TO_KPA,
    )
    with_defl = optimize(config)
    config_no = small_config(
        beam_candidates=["W12X16", "W14X22", "W16X26", "W18X35"],
        x_bay_spacing_m=30.0 * FT_TO_M,
        superimposed_dead_kpa=5.0 * PSF_TO_KPA, live_kpa=40.0 * PSF_TO_KPA,
        check_deflection=False,
    )
    without_defl = optimize(config_no)
    assert with_defl.feasible and without_defl.feasible
    # the serviceability constraint can only push the design up
    assert with_defl.total_weight_kg >= without_defl.total_weight_kg
