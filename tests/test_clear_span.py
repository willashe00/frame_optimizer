"""Clear-span industrial building: topology, hand-calc anchors, end-to-end.

Anchors follow the existing test philosophy: purlins must reproduce simple-
beam statics (wL^2/8, wL/2, 5wL^4/384EI) through the FEA, girders must match
the discrete-point-load statics, and the factored base reactions must equal
the total factored gravity load exactly.

Config inputs are SI (meters / kPa); the values are exact conversions of the
original imperial anchors (50 ft span, 15/25 psf, ...), so every hand
calculation below can stay in the backend's native kips/feet/inches.
"""
from dataclasses import replace

import pytest

from frame_optimizer import ClearSpanConfig, evaluate, optimize, optimize_layout
from frame_optimizer.analysis import analyze_frame
from frame_optimizer.clear_span import (MAX_FRAME_SPACING_M,
                                        MAX_PURLIN_SPACING_M,
                                        MIN_FRAME_SPACING_M,
                                        MIN_PURLIN_SPACING_M,
                                        END_GIRDER, GIRDER, PURLIN,
                                        build_clear_span_geometry,
                                        candidate_layouts,
                                        clear_span_check_params,
                                        derive_end_wall_columns,
                                        derive_n_frames,
                                        derive_purlin_spacing_m)
from frame_optimizer.config import COLUMN, FT_TO_M, M_TO_IN
from frame_optimizer.optimization.optimizer import (_PROOF_MARGIN,
                                                    _bound_demands,
                                                    _distinct_demands)
from frame_optimizer.sections import get_shapes

CAT = {s.name: s for s in get_shapes(["W8X10", "W12X16", "W16X26", "W21X44",
                                      "W24X76", "W30X108", "W33X130",
                                      "W10X33", "W12X53"])}

# exact psf -> kPa so hand calcs can keep using 15 / 25 psf
PSF_TO_KPA = 4.4482216152605 / 0.3048**2 / 1000.0


def cfg(**kw):
    base = dict(
        girder_candidates=["W24X76", "W30X108", "W33X130"],
        purlin_candidates=["W8X10", "W12X16"],
        column_candidates=["W10X33", "W12X53"],
        span_m=50.0 * FT_TO_M, length_m=60.0 * FT_TO_M, n_frames=3,
        height_m=20.0 * FT_TO_M, purlin_spacing_m=5.0 * FT_TO_M,
        superimposed_dead_kpa=15.0 * PSF_TO_KPA, live_kpa=25.0 * PSF_TO_KPA,
        purlin_Lb_m=0.0,
    )
    base.update(kw)
    return ClearSpanConfig(**base)


def gable_cfg(**kw):
    return cfg(end_wall_columns=2,
               end_girder_candidates=["W12X16", "W16X26", "W21X44"], **kw)


def auto_cfg(**kw):
    """Footprint-only config: the layout is derived, not given."""
    base = dict(
        girder_candidates=["W24X76", "W30X108", "W33X130"],
        purlin_candidates=["W8X10", "W12X16"],
        column_candidates=["W10X33", "W12X53"],
        span_m=50.0 * FT_TO_M, length_m=60.0 * FT_TO_M,
        height_m=20.0 * FT_TO_M,
        superimposed_dead_kpa=15.0 * PSF_TO_KPA, live_kpa=25.0 * PSF_TO_KPA,
        purlin_Lb_m=0.0,
    )
    base.update(kw)
    return ClearSpanConfig(**base)


def analyzed(config, girder="W30X108", purlin="W8X10", column="W10X33",
             end_girder="W16X26"):
    geo = build_clear_span_geometry(config)
    assignment = {COLUMN: CAT[column], GIRDER: CAT[girder], PURLIN: CAT[purlin]}
    if config.has_end_girder_group:
        assignment[END_GIRDER] = CAT[end_girder]
    return geo, assignment, analyze_frame(geo, assignment, config)


def factored_plf(w_dead_plf: float, w_live_plf: float) -> float:
    return max(1.4 * w_dead_plf, 1.2 * w_dead_plf + 1.6 * w_live_plf)


def span_ft(config) -> float:
    return config.span_m / FT_TO_M


# ---------------------------------------------------------------- topology

def test_all_columns_on_perimeter_no_interior_supports():
    config = cfg()
    geo = build_clear_span_geometry(config)
    node = {n.name: n for n in geo.nodes}
    span_in = config.span_m * M_TO_IN
    for m in geo.members_in_group(COLUMN):
        x = node[m.i_node].x
        assert x == pytest.approx(0.0) or x == pytest.approx(span_in)
    # each girder runs wall to wall in one piece: full clear span
    for m in geo.members_in_group(GIRDER):
        assert m.length_in == pytest.approx(span_in)
        assert m.trib_width_in == 0.0   # all roof load arrives through purlins
    assert geo.groups == (COLUMN, GIRDER, PURLIN)
    assert len(geo.members_in_group(COLUMN)) == 2 * config.n_frames
    assert len(geo.members_in_group(GIRDER)) == config.n_frames
    n_lines = config.n_purlin_spaces + 1
    assert len(geo.members_in_group(PURLIN)) == n_lines * (config.n_frames - 1)


def test_interior_girder_nodes_have_free_rotations():
    geo = build_clear_span_geometry(gable_cfg())
    for n in geo.nodes:
        on_girder_interior = n.name.startswith(("NP", "NG")) and not n.is_base
        assert n.free_rotations == on_girder_interior


def test_gable_columns_on_end_walls_only():
    config = gable_cfg()
    geo = build_clear_span_geometry(config)
    node = {n.name: n for n in geo.nodes}
    gables = [m for m in geo.members if m.name.startswith("CG")]
    assert len(gables) == 2 * config.end_wall_columns
    span_in = config.span_m * M_TO_IN
    length_in = config.length_m * M_TO_IN
    for m in gables:
        n = node[m.i_node]
        assert m.group == COLUMN
        assert 0.0 < n.x < span_in                       # interior of the span...
        assert n.z == pytest.approx(0.0) or n.z == pytest.approx(length_in)
        # ...but only on the two exterior end walls
    assert geo.groups == (COLUMN, END_GIRDER, GIRDER, PURLIN)
    assert len(geo.members_in_group(END_GIRDER)) == 2
    assert len(geo.members_in_group(GIRDER)) == config.n_frames - 2


def test_gable_column_on_purlin_line_reuses_node():
    # span 50 @ 5 ft purlins: 4 gable columns land exactly on purlin lines
    config = cfg(end_wall_columns=4,
                 end_girder_candidates=["W12X16", "W16X26"])
    geo = build_clear_span_geometry(config)
    tops = [n for n in geo.nodes
            if n.name.startswith("NG") and not n.name.startswith("NGB")]
    assert tops == []           # every gable top merged into a purlin node
    _, _, demands = analyzed(config)   # and the merged model analyzes fine
    assert demands


def test_config_validation():
    with pytest.raises(ValueError):
        cfg(n_frames=1)
    with pytest.raises(ValueError):
        cfg(purlin_spacing_m=30.0)    # > span/2
    with pytest.raises(ValueError):
        cfg(girder_candidates=[])
    with pytest.raises(ValueError):
        cfg(end_wall_columns=2)       # needs end_girder_candidates
    with pytest.raises(ValueError):
        cfg(girder_camber_mm=-1.0)


# ---------------------------------------- layout derived from the footprint

def test_layout_rules_track_practice_bands():
    # bays as close to 7.62 m (25 ft) as the length allows, never above 9.14 m
    assert derive_n_frames(98.4 * FT_TO_M) == 5      # 4 bays @ 24.6 ft
    assert derive_n_frames(60.0 * FT_TO_M) == 3      # 2 bays @ 30.0 ft
    assert derive_n_frames(62.5 * FT_TO_M) == 4      # extra bay keeps spacing in band
    assert derive_n_frames(25.0 * FT_TO_M) == 2      # single bay
    assert derive_n_frames(12.0 * FT_TO_M) == 2      # tiny building: still a 1x1 bay
    # purlins at ~1.52 m (5 ft) unless the two-space minimum forces less
    assert derive_purlin_spacing_m(50.0 * FT_TO_M) == pytest.approx(5.0 * FT_TO_M)
    assert derive_purlin_spacing_m(8.0 * FT_TO_M) == pytest.approx(4.0 * FT_TO_M)
    # gable columns keep end-girder segments <= 7.62 m, and require the group
    assert derive_end_wall_columns(65.6 * FT_TO_M, True) == 2
    assert derive_end_wall_columns(20.0 * FT_TO_M, True) == 0
    assert derive_end_wall_columns(65.6 * FT_TO_M, False) == 0


def test_auto_layout_fields_derived_and_tracked():
    config = auto_cfg()
    assert config.auto_layout_fields == frozenset(
        {"n_frames", "purlin_spacing_m", "end_wall_columns"})
    assert config.n_frames == 3            # 60 ft -> 2 bays @ 30 ft
    assert config.purlin_spacing_m == pytest.approx(5.0 * FT_TO_M)
    assert config.end_wall_columns == 0    # no end-girder group given
    # explicit values are honored, marked pinned, and not searched over
    assert cfg().auto_layout_fields == frozenset({"end_wall_columns"})
    pinned = gable_cfg()
    assert pinned.auto_layout_fields == frozenset()
    assert candidate_layouts(pinned) == [(3, 5.0 * FT_TO_M, 2)]


def test_footprint_orientation_normalized():
    # girders must clear-span the shorter plan dimension; a span > length
    # input is auto-swapped and produces the identical building
    swapped = auto_cfg(span_m=60.0 * FT_TO_M, length_m=50.0 * FT_TO_M)
    right = auto_cfg()   # same footprint, correctly oriented (50 x 60)
    assert (swapped.span_m, swapped.length_m) == (50.0 * FT_TO_M, 60.0 * FT_TO_M)
    assert swapped.n_frames == right.n_frames
    assert swapped.purlin_spacing_m == right.purlin_spacing_m
    assert any("swapped" in line for line in swapped.describe())
    assert not any("swapped" in line for line in right.describe())


def test_small_footprint_collapses_to_single_bay():
    config = auto_cfg(span_m=20.0 * FT_TO_M, length_m=24.0 * FT_TO_M,
                      height_m=16.0 * FT_TO_M)
    assert (config.n_frames, config.end_wall_columns) == (2, 0)
    geo = build_clear_span_geometry(config)
    assert len(geo.members_in_group(GIRDER)) == 2
    assert len(geo.members_in_group(COLUMN)) == 4   # 1x1 bay: corner columns only


def test_candidate_layouts_stay_realistic_and_valid():
    config = auto_cfg(end_girder_candidates=["W12X16", "W16X26", "W21X44"])
    layouts = candidate_layouts(config)
    assert len(layouts) == len(set(layouts)) > 1
    for n_frames, spacing, gables in layouts:
        assert MIN_FRAME_SPACING_M <= config.length_m / (n_frames - 1) \
            <= MAX_FRAME_SPACING_M + 1e-9
        assert MIN_PURLIN_SPACING_M - 1e-9 <= spacing <= MAX_PURLIN_SPACING_M + 1e-9
        assert gables in (0, 1, 2)   # span 50 ft: up to 2 keeps segments >= ~15 ft
        # every candidate layout must construct as a valid config
        replace(config, n_frames=n_frames, purlin_spacing_m=spacing,
                end_wall_columns=gables)
    assert {g for _, _, g in layouts} == {0, 1, 2}


def test_optimize_layout_picks_lightest_feasible():
    config = auto_cfg(span_m=30.0 * FT_TO_M, length_m=45.0 * FT_TO_M)
    result = optimize_layout(config)
    assert result.feasible
    feasible = [r for r in result.layout_search if r["feasible"]]
    assert feasible
    assert result.total_weight_kg == pytest.approx(
        min(r["total_weight_kg"] for r in feasible))
    # the chosen layout is baked into the returned config, now fully concrete
    assert result.config.n_frames == 3               # 45 ft -> 2 bays
    assert result.config.auto_layout_fields == frozenset()
    assert "Layout:" in result.summary()


def test_optimize_layout_honors_pinned_fields():
    config = auto_cfg(span_m=30.0 * FT_TO_M, length_m=45.0 * FT_TO_M,
                      purlin_spacing_m=5.0 * FT_TO_M)
    assert all(sp == 5.0 * FT_TO_M for _, sp, _ in candidate_layouts(config))
    with pytest.raises(TypeError):
        optimize_layout("not a clear-span config")


def test_check_params_rules_per_group():
    config = gable_cfg(girder_camber_mm=25.4,
                       girder_defl_live_ratio=240.0, girder_defl_total_ratio=180.0)
    params = clear_span_check_params(config)
    girder = params.rules_for(GIRDER)
    assert girder.Lb_in == pytest.approx(config.purlin_spacing_actual_m * M_TO_IN)
    assert girder.camber_in == pytest.approx(1.0)
    assert (girder.defl_live_ratio, girder.defl_total_ratio) == (240.0, 180.0)
    end_girder = params.rules_for(END_GIRDER)
    assert end_girder.camber_in == 0.0            # no camber on supported spans
    assert end_girder.defl_live_ratio == 240.0    # shares the girder ratios
    purlin = params.rules_for(PURLIN)
    assert purlin.Lb_in == 0.0                    # purlin_Lb_m=0.0
    assert purlin.defl_live_ratio == 360.0        # global default, no override
    assert params.rules_for(COLUMN).check_slenderness is True


# ------------------------------------------------------- hand-calc anchors

def test_purlin_demands_match_simple_beam_statics():
    config = cfg()
    _, _, demands = analyzed(config)
    sp_ft = config.purlin_spacing_actual_m / FT_TO_M
    L = config.frame_spacing_m * M_TO_IN

    interior = next(d for d in demands if d.name == "P1.b0")
    w_d = (CAT["W8X10"].weight_plf + 15.0 * sp_ft) / 12000.0
    w_l = 25.0 * sp_ft / 12000.0
    w_u = factored_plf(CAT["W8X10"].weight_plf + 15.0 * sp_ft, 25.0 * sp_ft) / 12000.0
    assert interior.Mux == pytest.approx(w_u * L ** 2 / 8.0, rel=1e-3)
    assert interior.Vu == pytest.approx(w_u * L / 2.0, rel=1e-3)
    EI = 29000.0 * CAT["W8X10"].Ix
    assert interior.defl_total_in == pytest.approx(
        5.0 * (w_d + w_l) * L ** 4 / (384.0 * EI), rel=1e-2)

    eave = next(d for d in demands if d.name == "P0.b0")
    assert eave.trib_width_in == pytest.approx(sp_ft * 12.0 / 2.0)


def test_girder_demand_matches_point_load_statics():
    config = cfg()
    _, _, demands = analyzed(config)
    s_f = config.frame_spacing_m / FT_TO_M
    L = config.span_m * M_TO_IN

    # interior purlin reactions are P = (q + purlin self) * sp * s_f at
    # spacing sp along the girder. For this pattern (interior point loads
    # with tributary sp, eave strips going straight to the columns) the exact
    # midspan moment equals the full uniform-load value w*L^2/8: the load
    # concentration and the missing eave strips cancel exactly.
    n_lines = config.n_purlin_spaces + 1
    purlin_psf = CAT["W8X10"].weight_plf * n_lines / span_ft(config)
    w_u = factored_plf(CAT["W30X108"].weight_plf + (15.0 + purlin_psf) * s_f,
                       25.0 * s_f) / 12000.0            # kip/in
    interior = next(d for d in demands if d.name == "G1")
    assert interior.Mux == pytest.approx(w_u * L ** 2 / 8.0, rel=1e-2)
    assert interior.Mux <= w_u * L ** 2 / 8.0 * 1.001   # never above the bound


def test_factored_base_reactions_equal_total_load():
    config = cfg()
    _, _, demands = analyzed(config)
    total_axial = sum(-d.Pu for d in demands if d.group == COLUMN)

    length_ft = config.length_m / FT_TO_M
    eave_ft = config.height_m / FT_TO_M
    area = span_ft(config) * length_ft
    n_lines = config.n_purlin_spaces + 1
    purlin_lb = CAT["W8X10"].weight_plf * n_lines * length_ft
    girder_lb = CAT["W30X108"].weight_plf * span_ft(config) * config.n_frames
    column_lb = CAT["W10X33"].weight_plf * eave_ft * 2 * config.n_frames
    dead_kip = (15.0 * area + purlin_lb + girder_lb + column_lb) / 1000.0
    live_kip = 25.0 * area / 1000.0

    assert total_axial == pytest.approx(1.2 * dead_kip + 1.6 * live_kip, rel=1e-3)


def test_column_axials_match_tributary_hand_calc():
    config = cfg()
    _, _, demands = analyzed(config)
    by_name = {d.name: d for d in demands}
    n_lines = config.n_purlin_spaces + 1
    purlin_psf = CAT["W8X10"].weight_plf * n_lines / span_ft(config)
    eave_ft = config.height_m / FT_TO_M
    col_self = 1.2 * CAT["W10X33"].weight_plf * eave_ft / 1000.0

    def expected(trib_ft):
        w_d = CAT["W30X108"].weight_plf + (15.0 + purlin_psf) * trib_ft
        w_u = factored_plf(w_d, 25.0 * trib_ft) / 1000.0    # kip/ft
        return w_u * span_ft(config) / 2.0 + col_self       # girder reaction + self

    s_f = config.frame_spacing_m / FT_TO_M
    assert -by_name["C0.1"].Pu == pytest.approx(expected(s_f), rel=1e-3)
    assert -by_name["C0.0"].Pu == pytest.approx(expected(s_f / 2.0), rel=1e-3)


def test_end_girders_see_a_fraction_of_interior_demand():
    config = gable_cfg()
    _, _, demands = analyzed(config)
    interior = next(d for d in demands if d.name == "G1")
    end = next(d for d in demands if d.name == "G0")
    assert end.group == END_GIRDER
    # half the tributary width AND gable-column support at the third points
    assert end.Mux < 0.2 * interior.Mux
    assert end.defl_total_in < 0.2 * interior.defl_total_in


# ------------------------------------------------------------- end-to-end

def test_optimize_clear_span_feasible_and_certified():
    result = optimize(cfg())
    assert result.feasible and result.converged
    assert bool(result.member_table["PASS"].all())
    assert set(result.sections) == {COLUMN, GIRDER, PURLIN}
    assert result.total_weight_kg > 0


def test_optimize_with_gable_columns_lightens_end_girders():
    result = optimize(gable_cfg())
    assert result.feasible and result.converged
    assert bool(result.member_table["PASS"].all())
    shapes = {s.name: s for s in get_shapes(
        [result.sections[GIRDER], result.sections[END_GIRDER]])}
    assert (shapes[result.sections[END_GIRDER]].weight_plf
            < shapes[result.sections[GIRDER]].weight_plf)


def test_camber_can_only_lighten_the_girder():
    plain = optimize(cfg())
    cambered = optimize(cfg(girder_camber_mm=25.4))
    assert cambered.feasible
    assert cambered.total_weight_kg <= plain.total_weight_kg


def test_iterative_matches_exhaustive():
    it = optimize(cfg())
    ex = optimize(cfg(), method="exhaustive")
    assert ex.feasible
    assert it.sections == ex.sections
    assert it.total_weight_kg == pytest.approx(ex.total_weight_kg)


def test_evaluate_explicit_design_and_group_validation():
    result = evaluate(cfg(), {"column": "W12X53", "girder": "W33X130",
                              "purlin": "W12X16"})
    assert result.total_weight_kg > 0
    assert set(result.sections) == {COLUMN, GIRDER, PURLIN}
    with pytest.raises(ValueError, match="group"):
        evaluate(cfg(), {"column": "W12X53", "girder": "W33X130"})


# ------------------------------------------- FEA-free infeasibility screening

def test_statics_bounds_never_exceed_the_fea_demands():
    """The pre-screen may only reject a layout when NO section could pass, so
    _bound_demands must sit at or below what the FEA actually reports for
    every group and every component. Anything above would let a feasible
    layout be thrown away."""
    no_gable_end_girder = cfg(
        end_wall_columns=0,
        end_girder_candidates=["W12X16", "W16X26", "W21X44"])
    for config in (cfg(), gable_cfg(), cfg(span_m=20.0, length_m=25.0),
                   auto_cfg(span_m=12.0, length_m=14.0, height_m=5.0),
                   # the end_girder bound only applies with no gable columns
                   # (a plain simple span), so it needs its own case
                   no_gable_end_girder,
                   cfg(span_m=20.0, length_m=25.0, end_wall_columns=0,
                       end_girder_candidates=["W12X16", "W16X26"])):
        geo, assignment, demands = analyzed(config)
        candidates = {g: [assignment[g]] for g in geo.groups}
        bounds = _bound_demands(config, geo, candidates)
        assert bounds, "expected bounds for at least one group"
        if config.has_end_girder_group and config.end_wall_columns == 0:
            assert END_GIRDER in bounds   # the case added above must be live
        for group, bound in bounds.items():
            actual = [d for d in demands if d.group == group]
            assert bound.Mux <= max(d.Mux for d in actual) + 1e-9
            assert bound.Vu <= max(d.Vu for d in actual) + 1e-9
            assert abs(bound.Pu) <= max(abs(d.Pu) for d in actual) + 1e-9
            # deflection x Ix is the section-independent invariant, since
            # check_member projects sag onto candidates via Ix_used/Ix
            for field in ("defl_total_in", "defl_live_in"):
                b = getattr(bound, field) * bound.Ix_used
                a = max(getattr(d, field) * d.Ix_used for d in actual)
                assert b <= a * _PROOF_MARGIN


def test_prescreen_does_not_change_the_chosen_design():
    config = auto_cfg(span_m=30.0 * FT_TO_M, length_m=45.0 * FT_TO_M,
                      end_girder_candidates=["W12X16", "W16X26", "W21X44"])
    fast = optimize_layout(config, prescreen=True)
    full = optimize_layout(config, prescreen=False)
    assert fast.feasible == full.feasible
    assert fast.sections == full.sections
    assert fast.total_weight_kg == pytest.approx(full.total_weight_kg)
    assert fast.config.n_frames == full.config.n_frames
    assert fast.config.end_wall_columns == full.config.end_wall_columns
    # every layout is still accounted for in the search record
    assert len(fast.layout_search) == len(full.layout_search)


def test_prescreen_skips_hopeless_layouts_but_still_certifies_one():
    # columns far too slender for any candidate (KL/r >> 200 regardless of
    # load) - a demand-independent limit state the screen catches exactly
    config = auto_cfg(span_m=40.0, length_m=50.0, height_m=40.0,
                      end_girder_candidates=["W12X16", "W16X26", "W21X44"])
    result = optimize_layout(config)
    assert not result.feasible
    analyzed_layouts = [r for r in result.layout_search if r["analyzed"]]
    assert len(analyzed_layouts) < len(result.layout_search)
    # the reported best attempt is still a real, FEA-certified check table
    assert len(analyzed_layouts) >= 1
    assert not result.member_table.empty
    assert bool((~result.member_table["PASS"]).any())


def test_distinct_demands_keeps_one_of_each_exact_duplicate():
    from frame_optimizer.analysis import MemberDemand

    def d(name, Mux):
        return MemberDemand(name=name, group=PURLIN, story=1, length_in=120.0,
                            trib_width_in=0.0, shape_used="W8X10", Ix_used=30.8,
                            Pu=0.0, Mux=Mux, Muy=0.0, Vu=0.0,
                            defl_total_in=0.0, defl_live_in=0.0)

    kept = _distinct_demands([d("a", 10.0), d("b", 10.0), d("c", 10.1)])
    assert len(kept) == 2                      # the exact duplicate collapsed
    assert {k.Mux for k in kept} == {10.0, 10.1}
    # values that merely look close are NOT merged
    assert len(_distinct_demands([d("a", 10.0), d("b", 10.0 + 1e-12)])) == 2
