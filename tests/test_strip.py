"""Performance machinery must not change results.

* Representative strip: pin-ended purlins are statically determinate, so
  transverse frames are exactly decoupled — the optimizer analyzes a 3-frame
  strip and broadcasts demands to the full building. These tests assert the
  broadcast demands match a full-building analysis member for member, for
  both girder systems (rigid truss bents solved with P-Delta included).
* Parallel layout search: a process pool over independent layout variants
  must reproduce the serial search exactly (same winner, same record).
"""
import pytest

from frame_optimizer import ClearSpanConfig, optimize_layout
from frame_optimizer.analysis import analyze_frame
from frame_optimizer.clear_span import (END_GIRDER, GIRDER, PURLIN, TRUSS,
                                        TRUSS_BOT_CHORD, TRUSS_TOP_CHORD,
                                        TRUSS_WEB)
from frame_optimizer.config import COLUMN
from frame_optimizer.optimization.optimizer import _prepare
from frame_optimizer.sections import get_shapes

CAT = {s.name: s for s in get_shapes(
    ["W8X10", "W8X24", "W10X33", "W12X53", "W16X26", "W30X108"])}


def assert_demands_match(fast, full, rel, abs_tol):
    assert len(fast) == len(full)
    by_name = {d.name: d for d in full}
    for d in fast:
        ref = by_name[d.name]
        assert d.group == ref.group and d.length_in == ref.length_in
        for field in ("Pu", "Pu_min", "Pu_max", "Mux", "Muy", "Vu",
                      "defl_total_in", "defl_live_in"):
            assert getattr(d, field) == pytest.approx(
                getattr(ref, field), rel=rel, abs=abs_tol), (d.name, field)


def test_strip_matches_full_building_truss_rigid():
    config = ClearSpanConfig(
        girder_candidates=[], purlin_candidates=["W8X10"],
        column_candidates=["W12X53"], girder_system=TRUSS,
        truss_chord_candidates=["W10X33"], truss_web_candidates=["W8X24"],
        span_ft=60.0, length_ft=150.0, n_frames=6,
        eave_height_ft=20.0, purlin_spacing_ft=5.0,
        superimposed_dead_psf=15.0, live_psf=25.0, purlin_Lb_ft=0.0,
    )
    assignment = {COLUMN: CAT["W12X53"], TRUSS_TOP_CHORD: CAT["W10X33"],
                  TRUSS_BOT_CHORD: CAT["W10X33"], TRUSS_WEB: CAT["W8X24"],
                  PURLIN: CAT["W8X10"]}
    geometry, _, analyze = _prepare(config)
    assert analyze is not analyze_frame          # strip is active (> 3 frames)
    fast = analyze(geometry, assignment, config)
    full = analyze_frame(geometry, assignment, config)
    # P-Delta solves on both sides: tolerance covers iteration convergence
    assert_demands_match(fast, full, rel=1e-4, abs_tol=1e-4)


def test_strip_matches_full_building_wide_flange_with_gables():
    config = ClearSpanConfig(
        girder_candidates=["W30X108"], purlin_candidates=["W8X10"],
        column_candidates=["W10X33"],
        end_girder_candidates=["W16X26"], end_wall_columns=2,
        span_ft=50.0, length_ft=125.0, n_frames=6,
        eave_height_ft=20.0, purlin_spacing_ft=5.0,
        superimposed_dead_psf=15.0, live_psf=25.0, purlin_Lb_ft=0.0,
    )
    assignment = {COLUMN: CAT["W10X33"], GIRDER: CAT["W30X108"],
                  END_GIRDER: CAT["W16X26"], PURLIN: CAT["W8X10"]}
    geometry, _, analyze = _prepare(config)
    assert analyze is not analyze_frame
    fast = analyze(geometry, assignment, config)
    full = analyze_frame(geometry, assignment, config)
    # residual coupling: purlin torsion (retained to prevent spin mechanisms)
    # ties neighboring girder rotations at ~1e-5 relative magnitude
    assert_demands_match(fast, full, rel=1e-4, abs_tol=1e-4)


def test_short_buildings_skip_the_strip():
    config = ClearSpanConfig(
        girder_candidates=["W30X108"], purlin_candidates=["W8X10"],
        column_candidates=["W10X33"],
        span_ft=50.0, length_ft=60.0, n_frames=3,
        eave_height_ft=20.0, purlin_spacing_ft=5.0,
        superimposed_dead_psf=15.0, live_psf=25.0, purlin_Lb_ft=0.0,
    )
    _, _, analyze = _prepare(config)
    assert analyze is analyze_frame              # nothing to reduce


def test_parallel_layout_search_matches_serial():
    config = ClearSpanConfig(
        girder_candidates=["W24X76", "W30X108"],
        purlin_candidates=["W8X10", "W12X16"],
        column_candidates=["W10X33", "W12X53"],
        end_girder_candidates=["W12X16", "W16X26", "W21X44"],
        span_ft=50.0, length_ft=45.0, eave_height_ft=20.0,
        superimposed_dead_psf=15.0, live_psf=25.0, purlin_Lb_ft=0.0,
    )
    serial = optimize_layout(config, parallel=False)
    pooled = optimize_layout(config, parallel=True)
    assert pooled.feasible == serial.feasible
    assert pooled.sections == serial.sections
    assert pooled.total_weight_lb == pytest.approx(serial.total_weight_lb)
    assert ([e["feasible"] for e in pooled.layout_search]
            == [e["feasible"] for e in serial.layout_search])
