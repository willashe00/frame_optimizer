import pytest

from frame_optimizer.config import FT_TO_M, FrameConfig
from frame_optimizer.geometry import BEAM, COLUMN, build_geometry


def make_config(**kw):
    base = dict(
        beam_candidates=["W18X35"],
        column_candidates=["W10X33"],
        x_bays=2, x_bay_spacing_m=30.0 * FT_TO_M,
        z_bays=3, z_bay_spacing_m=25.0 * FT_TO_M,
        stories=2, story_height_m=13.0 * FT_TO_M,
        superimposed_dead_kpa=1.0, live_kpa=2.4,
        deck_span_direction="z",
    )
    base.update(kw)
    return FrameConfig(**base)


def loaded(geo):
    return [m for m in geo.members_in_group(BEAM) if m.trib_width_in > 0.0]


def unloaded(geo):
    return [m for m in geo.members_in_group(BEAM) if m.trib_width_in == 0.0]


def test_member_and_node_counts():
    geo = build_geometry(make_config())
    assert len(geo.nodes) == 3 * 4 * 3                      # nx * nz * (stories+1)
    assert len(geo.members_in_group(COLUMN)) == 3 * 4 * 2   # lines * stories
    # all horizontal members are one 'beam' group:
    # x-running (2 x-bays * 4 z-lines) + z-running (3 z-bays * 3 x-lines), 2 levels
    assert len(geo.members_in_group(BEAM)) == (2 * 4 + 3 * 3) * 2
    assert len(loaded(geo)) == 2 * 4 * 2     # deck spans z -> x-running loaded
    assert len(unloaded(geo)) == 3 * 3 * 2


def test_one_way_tributary_widths():
    geo = build_geometry(make_config())
    # deck spans z (25-ft bays): edge z-lines get half a bay, interior a full
    # bay — 12.5 ft and 25 ft, i.e. 150 in and 300 in internal
    tribs = sorted({round(b.trib_width_in, 6) for b in loaded(geo)})
    assert tribs == [pytest.approx(12.5 * 12), pytest.approx(25.0 * 12)]
    assert all(m.trib_width_in == 0.0 for m in unloaded(geo))
    assert all(c.trib_width_in == 0.0 for c in geo.members_in_group(COLUMN))


def test_deck_direction_swaps_loaded_members():
    geo_z = build_geometry(make_config(deck_span_direction="z"))
    geo_x = build_geometry(make_config(deck_span_direction="x"))
    # the beam group is the same set of members either way; swapping the deck
    # direction only swaps which of them carry tributary load
    assert len(geo_z.members_in_group(BEAM)) == len(geo_x.members_in_group(BEAM))
    assert len(loaded(geo_z)) == len(unloaded(geo_x))
    assert len(unloaded(geo_z)) == len(loaded(geo_x))


def test_per_story_heights():
    geo = build_geometry(make_config(
        stories=2, story_height_m=[15.0 * FT_TO_M, 12.0 * FT_TO_M]))
    cols = geo.members_in_group(COLUMN)
    h1 = {c.length_in for c in cols if c.story == 1}
    h2 = {c.length_in for c in cols if c.story == 2}
    assert len(h1) == len(h2) == 1
    assert h1.pop() == pytest.approx(15.0 * 12)
    assert h2.pop() == pytest.approx(12.0 * 12)


def test_config_validation():
    with pytest.raises(ValueError):
        make_config(x_bays=0)
    with pytest.raises(ValueError):
        make_config(deck_span_direction="y")
    with pytest.raises(ValueError):
        make_config(story_height_m=[4.0])            # wrong list length for 2 stories
    with pytest.raises(NotImplementedError):
        make_config(infill_beams_per_bay=2)
