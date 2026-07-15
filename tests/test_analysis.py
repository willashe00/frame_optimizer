"""FEA demands vs closed-form statics for a single-bay, single-story frame.

Layout: 1 x-bay @ 30 ft, 1 z-bay @ 20 ft, one 12-ft story, deck spans z.
All horizontal members share one 'beam' section (W18X35, 35 plf).
Loaded beams (x-running, span 30 ft) are all on edge z-lines: tributary = 10 ft.
Unloaded beams (z-running, span 20 ft, trib = 0) carry self-weight only.

Hand values (kips, ft):
    loaded beam:   w_D = 35 + 20*10 = 235 plf, w_L = 50*10 = 500 plf
                   wu  = 1.2*235 + 1.6*500 = 1082 plf  (1.2D+1.6L governs 1.4D = 329)
                   Mu  = wu*L^2/8 = 121.725 kip-ft,  Vu = wu*L/2 = 16.23 kip
                   service D+L: w = 735 plf -> midspan sag 5wL^4/384EI = 0.9057 in
    unloaded beam: dead only, 1.4D governs: wu = 1.4*35 = 49 plf -> Mu = 2.45 kip-ft
    column: P = loaded-beam rxn + unloaded-beam rxn + column self (1.2D+1.6L)
"""
import pytest

from frame_optimizer.analysis import analyze_frame
from frame_optimizer.config import FrameConfig
from frame_optimizer.geometry import BEAM, COLUMN, build_geometry
from frame_optimizer.sections import load_w_shapes

CAT = load_w_shapes()
E = 29000.0


@pytest.fixture(scope="module")
def demands():
    config = FrameConfig(
        beam_candidates=["W18X35"],
        column_candidates=["W10X33"],
        x_bays=1, x_bay_spacing_ft=30.0,
        z_bays=1, z_bay_spacing_ft=20.0,
        stories=1, story_height_ft=12.0,
        superimposed_dead_psf=20.0, live_psf=50.0,
        deck_span_direction="z",
    )
    geometry = build_geometry(config)
    assignment = {BEAM: CAT["W18X35"], COLUMN: CAT["W10X33"]}
    return analyze_frame(geometry, assignment, config)


def by_group(demands, group):
    return [d for d in demands if d.group == group]


def loaded_beams(demands):
    return [d for d in by_group(demands, BEAM) if d.trib_width_in > 0.0]


def unloaded_beams(demands):
    return [d for d in by_group(demands, BEAM) if d.trib_width_in == 0.0]


def test_beam_moment_shear(demands):
    wu = (1.2 * 235.0 + 1.6 * 500.0) / 1000.0   # kip/ft
    L = 30.0
    assert len(loaded_beams(demands)) == 2
    for d in loaded_beams(demands):
        assert d.Mux == pytest.approx(wu * L**2 / 8.0 * 12.0, rel=1e-3)   # kip-in
        assert d.Vu == pytest.approx(wu * L / 2.0, rel=1e-3)
        assert abs(d.Pu) < 0.05                     # pinned gravity beams: no axial
        assert d.Muy == pytest.approx(0.0, abs=0.5)


def test_unloaded_beam_self_weight_only(demands):
    # zero-tributary beams carry dead load only, so 1.4D governs the envelope
    wu = 1.4 * 35.0 / 1000.0
    L = 20.0
    assert len(unloaded_beams(demands)) == 2
    for d in unloaded_beams(demands):
        assert d.Mux == pytest.approx(wu * L**2 / 8.0 * 12.0, rel=1e-3)


def test_column_axial_is_compression_with_correct_magnitude(demands):
    # each corner column: one loaded-beam end + one unloaded-beam end + self
    beam_rxn = (1.2 * 235.0 + 1.6 * 500.0) / 1000.0 * 30.0 / 2.0
    unloaded_rxn = 1.2 * 35.0 / 1000.0 * 20.0 / 2.0
    col_self = 1.2 * 33.0 / 1000.0 * 12.0
    expected = beam_rxn + unloaded_rxn + col_self
    for d in by_group(demands, COLUMN):
        assert d.Pu < 0.0                            # compression is negative
        assert -d.Pu == pytest.approx(expected, rel=1e-3)
        assert d.Mux == pytest.approx(0.0, abs=0.5)  # pin-pin, no transverse load


def test_beam_service_deflection_matches_closed_form(demands):
    w = 735.0 / 12000.0        # kip/in, D+L service
    L = 360.0
    Ix = CAT["W18X35"].Ix
    expected = 5.0 * w * L**4 / (384.0 * E * Ix)
    for d in loaded_beams(demands):
        assert d.defl_total_in == pytest.approx(expected, rel=1e-3)
        assert d.defl_live_in == pytest.approx(expected * 500.0 / 735.0, rel=1e-3)
