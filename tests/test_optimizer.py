import pytest

from frame_optimizer import FrameConfig, evaluate, optimize


def small_config(**kw):
    base = dict(
        beam_candidates=["W10X12", "W12X16", "W14X22"],
        column_candidates=["W6X9", "W8X24"],
        x_bays=1, x_bay_spacing_ft=20.0,
        z_bays=1, z_bay_spacing_ft=20.0,
        stories=1, story_height_ft=10.0,
        superimposed_dead_psf=20.0, live_psf=50.0,
        deck_span_direction="z",
        beam_Lb_ft=0.0,   # deck-braced compression flange
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
    assert it.total_weight_lb == pytest.approx(ex.total_weight_lb)


def test_infeasible_candidates_reported():
    result = optimize(small_config(beam_candidates=["W10X12"]))
    assert not result.feasible
    assert not bool(result.member_table["PASS"].all())


def test_unbraced_beam_needs_heavier_section():
    braced = optimize(small_config())
    unbraced = optimize(small_config(beam_Lb_ft=None))   # Lb = full 20-ft span
    assert unbraced.total_weight_lb >= braced.total_weight_lb
    if unbraced.feasible:
        assert unbraced.member_table["PASS"].all()


def test_evaluate_explicit_design():
    result = evaluate(small_config(), {"beam": "W14X22", "column": "W8X24"})
    assert result.sections == {"beam": "W14X22", "column": "W8X24"}
    assert bool(result.member_table["PASS"].all())
    assert result.total_weight_lb > 0


def test_deflection_can_govern():
    # long span, light loads: strength is easy but L/360 live deflection bites
    config = small_config(
        beam_candidates=["W12X16", "W14X22", "W16X26", "W18X35"],
        x_bay_spacing_ft=30.0,
        superimposed_dead_psf=5.0, live_psf=40.0,
    )
    with_defl = optimize(config)
    config_no = small_config(
        beam_candidates=["W12X16", "W14X22", "W16X26", "W18X35"],
        x_bay_spacing_ft=30.0,
        superimposed_dead_psf=5.0, live_psf=40.0,
        check_deflection=False,
    )
    without_defl = optimize(config_no)
    assert with_defl.feasible and without_defl.feasible
    # the serviceability constraint can only push the design up
    assert with_defl.total_weight_lb >= without_defl.total_weight_lb
