"""Lateral system topology: practice-band derivations, geometry counts and
wiring, portal-frame conversion, and Pynite stability/statics closure of the
augmented model.

The statics anchor follows the existing test philosophy: the augmented
geometry (braces, struts, re-tagged eave lines, moment-connected knees)
must still deliver the EXACT total gravity load to the foundations, and the
portal conversion must produce real knee moments where the pinned gravity
frame had none.
"""
import math

import pytest

from frame_optimizer import ClearSpanConfig, LateralSystemConfig, build_lateral_system
from frame_optimizer.analysis import analyze_frame, build_model
from frame_optimizer.lateral_system import (BRACE, EAVE_STRUT, ROOF_BRACE,
                                            WALL_STRUT,
                                            MAX_BAYS_BETWEEN_BRACED,
                                            derive_brace_tiers,
                                            derive_braced_bays,
                                            derive_roof_panels)
from frame_optimizer.sections import get_shapes


def cfg(**kw):
    base = dict(
        girder_candidates=["W24X76", "W30X108", "W33X130"],
        purlin_candidates=["W8X10", "W12X16"],
        column_candidates=["W10X33", "W12X53"],
        span_ft=50.0, length_ft=60.0, n_frames=3,
        eave_height_ft=20.0, purlin_spacing_ft=5.0,
        superimposed_dead_psf=15.0, live_psf=25.0,
        purlin_Lb_ft=0.0, girder_system="wide_flange",
    )
    base.update(kw)
    return ClearSpanConfig(**base)


def truss_cfg(**kw):
    base = dict(
        girder_candidates=None, end_girder_candidates=None,
        girder_system="truss",
        truss_chord_candidates=["W10X33", "W12X53"],
        truss_web_candidates=["W8X10"],
    )
    base.update(kw)
    return cfg(**base)


def lat_cfg(**kw):
    base = dict(
        brace_candidates=["W8X10"],
        roof_brace_candidates=["W8X10"],
        eave_strut_candidates=["W8X10"],
        wall_strut_candidates=["W8X10"],
    )
    base.update(kw)
    return LateralSystemConfig(**base)


# ------------------------------------------------------------- derivations

@pytest.mark.parametrize("n_bays, expected", [
    (1, [0]), (2, [0, 1]), (7, [0, 6]), (13, [0, 6, 12]),
])
def test_derive_braced_bays(n_bays, expected):
    assert derive_braced_bays(n_bays) == expected


@pytest.mark.parametrize("n_bays", [3, 8, 11, 17, 24])
def test_braced_bays_run_limit_and_symmetry(n_bays):
    bays = derive_braced_bays(n_bays)
    assert bays[0] == 0 and bays[-1] == n_bays - 1
    runs = [b2 - b1 - 1 for b1, b2 in zip(bays, bays[1:])]
    assert all(r <= MAX_BAYS_BETWEEN_BRACED for r in runs)
    assert bays == sorted(n_bays - 1 - b for b in bays)   # symmetric


@pytest.mark.parametrize("H, bay, tiers", [
    (20.0, 30.0, 1),     # 33.7 deg, in band
    (55.0, 20.5, 3),     # 41.8 deg after tiering
    (63.7, 23.4, 3),     # 42.2 deg
    (100.0, 10.0, 10),   # 45 deg exactly
    (5.0, 30.0, 1),      # shallow but 1 tier is the floor
])
def test_derive_brace_tiers(H, bay, tiers):
    assert derive_brace_tiers(H, bay) == tiers
    if tiers > 1:   # tiered walls must land inside the practice band
        angle = math.degrees(math.atan2(H / tiers, bay))
        assert 30.0 <= angle <= 60.0


def test_derive_roof_panels():
    assert derive_roof_panels(50.0, 30.0) == 2
    assert derive_roof_panels(131.2, 23.4) == 6


# ------------------------------------------------- topology: W-girder system

def test_wide_flange_topology_counts_and_wiring():
    config = cfg()
    system = build_lateral_system(config, lat_cfg())
    geo = system.geometry

    assert system.braced_bay_indices == [0, 1]
    assert system.n_tiers == 1
    assert system.counts == {EAVE_STRUT: 4, BRACE: 8, ROOF_BRACE: 8}
    assert WALL_STRUT not in geo.groups            # single tier: no struts
    assert system.roof_panel_line_indices == [0, 5, 10]

    node_names = {n.name for n in geo.nodes}
    for m in geo.members:                          # no orphan connections
        assert m.i_node in node_names and m.j_node in node_names

    by_name = {m.name: m for m in geo.members}
    # X pair of tier 1, bay 0, wall 0: base -> opposite column top
    assert by_name["DB0.0.t1.a"].i_node == "NB0.0"
    assert by_name["DB0.0.t1.a"].j_node == "NE0.1"
    assert by_name["DB0.0.t1.b"].i_node == "NB0.1"
    assert by_name["DB0.0.t1.b"].j_node == "NE0.0"
    # eave lines re-tagged, interior purlins untouched
    assert by_name["P0.b0"].group == EAVE_STRUT
    assert by_name["P10.b1"].group == EAVE_STRUT
    assert by_name["P5.b0"].group == "purlin"
    assert by_name["P0.b0"].trib_width_in == pytest.approx(2.5 * 12.0)


def test_wide_flange_portal_conversion():
    config = cfg()
    system = build_lateral_system(config, lat_cfg())
    by_name = {m.name: m for m in system.geometry.members}
    assert by_name["G1"].fixed_i and by_name["G1"].fixed_j
    assert by_name["C0.1"].fixed_j and not by_name["C0.1"].fixed_i
    assert not by_name["P5.b0"].fixed_i
    nodes = {n.name: n for n in system.geometry.nodes}
    assert nodes["NE0.1"].free_rotations and nodes["NE0.1"].free_dx
    assert nodes["NP3.1"].free_dx                  # whole frame translates
    assert not nodes["NB0.0"].free_dx              # bases stay supported
    assert system.analysis_config.transverse_moment_frame
    assert not config.transverse_moment_frame      # input config untouched


def test_gable_columns_stay_gravity_posts():
    config = cfg(end_wall_columns=2,
                 end_girder_candidates=["W12X16", "W16X26"])
    system = build_lateral_system(config, lat_cfg())
    by_name = {m.name: m for m in system.geometry.members}
    assert by_name["G0"].fixed_i and by_name["G0"].fixed_j   # end girder knee
    assert not by_name["CG1.0"].fixed_i and not by_name["CG1.0"].fixed_j


# --------------------------------------------------- topology: truss system

def test_truss_topology_reuses_rigid_bents():
    config = truss_cfg()
    system = build_lateral_system(config, lat_cfg())
    assert system.analysis_config is config        # already rigid: unchanged
    assert all(not m.fixed_i and not m.fixed_j
               for m in system.geometry.members)
    by_name = {m.name: m for m in system.geometry.members}
    assert by_name["DB0.0.t1.a"].j_node == "NT0.1"   # braces to top chord
    assert by_name["P0.b0"].group == EAVE_STRUT


def test_pinned_tiers_add_wall_nodes_and_struts():
    system = build_lateral_system(truss_cfg(), lat_cfg(n_brace_tiers=2))
    geo = system.geometry
    nodes = {n.name: n for n in geo.nodes}
    top_y = nodes["NT0.0"].y
    assert nodes["NW0.1.t1"].y == pytest.approx(top_y / 2.0)
    assert nodes["NW0.1.t1"].free_rotations and nodes["NW0.1.t1"].free_dx
    by_name = {m.name: m for m in geo.members}
    assert by_name["WS0.0.t1"].i_node == "NW0.0.t1"
    assert by_name["WS0.0.t1"].j_node == "NW0.1.t1"
    assert system.counts[WALL_STRUT] == 2 * 2 * 1  # 2 walls x 2 bays x 1 level
    assert system.counts[BRACE] == 2 * 2 * 2 * 2


def test_multi_tier_needs_strut_candidates():
    with pytest.raises(ValueError, match="wall_strut_candidates"):
        build_lateral_system(truss_cfg(),
                             lat_cfg(wall_strut_candidates=None,
                                     n_brace_tiers=3))


def test_unresolved_system_rejected():
    with pytest.raises(ValueError, match="resolved"):
        build_lateral_system(cfg(girder_system="auto"), lat_cfg())


@pytest.mark.parametrize("config", [
    cfg(),
    truss_cfg(),
    # real-building proportions: 22 purlin spaces / 6 roof panels do NOT
    # divide evenly, so the roof-brace panels have unequal widths — every
    # stored member length must still match the node geometry exactly
    truss_cfg(span_ft=131.234, length_ft=163.9, n_frames=8,
              purlin_spacing_ft=5.965, eave_height_ft=55.0),
])
def test_member_lengths_match_node_geometry(config):
    system = build_lateral_system(config, lat_cfg())
    nodes = {n.name: n for n in system.geometry.nodes}
    for m in system.geometry.members:
        ni, nj = nodes[m.i_node], nodes[m.j_node]
        dist = math.dist((ni.x, ni.y, ni.z), (nj.x, nj.y, nj.z))
        assert m.length_in == pytest.approx(dist, abs=1e-6), m.name
    widths = system.roof_panel_widths_ft
    assert sum(widths) == pytest.approx(config.span_ft)


# ------------------------------- analysis: stability, statics, portal action

def assignment_for(geometry):
    shapes = {s.name: s for s in get_shapes(
        ["W8X10", "W12X53", "W30X108", "W12X16"])}
    pick = {"column": "W12X53", "girder": "W30X108", "end_girder": "W12X16",
            "purlin": "W8X10", EAVE_STRUT: "W8X10", BRACE: "W8X10",
            WALL_STRUT: "W8X10", ROOF_BRACE: "W8X10"}
    return {g: shapes[pick[g]] for g in geometry.groups}


def total_gravity_kip(geometry, assignment, config):
    total = 0.0
    for m in geometry.members:
        L_ft = m.length_in / 12.0
        trib_ft = m.trib_width_in / 12.0
        psf = config.superimposed_dead_psf + config.live_psf
        total += (assignment[m.group].weight_plf * L_ft
                  + psf * trib_ft * L_ft) / 1000.0
    return total


@pytest.mark.parametrize("tiers", [None, 2])
def test_augmented_model_is_stable_and_statics_close(tiers):
    system = build_lateral_system(cfg(), lat_cfg(n_brace_tiers=tiers))
    geometry, config = system.geometry, system.analysis_config
    assignment = assignment_for(geometry)

    model = build_model(geometry, assignment, config)
    model.analyze(check_stability=True, check_statics=False, sparse=True)

    reactions = sum(model.nodes[n.name].RxnFY["D+L"]
                    for n in geometry.nodes if n.is_base)
    assert reactions == pytest.approx(
        total_gravity_kip(geometry, assignment, config), rel=1e-6)
    thrust = sum(model.nodes[n.name].RxnFX["D+L"]
                 for n in geometry.nodes if n.is_base)
    assert thrust == pytest.approx(0.0, abs=1e-6)   # self-equilibrating


def test_portal_frames_develop_knee_moments():
    config = cfg()
    system = build_lateral_system(config, lat_cfg())
    assignment = assignment_for(system.geometry)

    portal = build_model(system.geometry, assignment, system.analysis_config)
    portal.analyze(check_stability=True, check_statics=False, sparse=True)
    knee_moment = portal.members["G1"].moment("Mz", 0.0, "D+L")
    assert abs(knee_moment) > 100.0                 # kip-in: real knee moment

    # the DAM strength path (0.8E + notional + P-Delta) also solves
    demands = analyze_frame(system.geometry, assignment,
                            system.analysis_config)
    assert len(demands) == len(system.geometry.members)
    girder = next(d for d in demands if d.name == "G1")
    # moment connection relieves the midspan: envelope < the simple-span
    # moment the pinned gravity model would give (wL^2/8 with the purlin
    # point loads), but stays nonzero
    assert 0.0 < girder.Mux
