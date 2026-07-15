import pytest

from frame_optimizer.sections import get_shapes, load_w_shapes


def test_catalog_loads():
    catalog = load_w_shapes()
    assert len(catalog) == 283
    assert all(s.A > 0 and s.Ix > 0 and s.weight_plf > 0 for s in catalog.values())


def test_w18x35_properties_match_manual():
    s = load_w_shapes()["W18X35"]
    assert s.A == pytest.approx(10.3)
    assert s.d == pytest.approx(17.7)
    assert s.Zx == pytest.approx(66.5)
    assert s.Sx == pytest.approx(57.6)
    assert s.ry == pytest.approx(1.22)
    # computed columns vs AISC Manual: rts = 1.51 in, ho = 17.3 in
    assert s.rts == pytest.approx(1.51, rel=0.01)
    assert s.ho == pytest.approx(17.3, rel=0.01)


def test_get_shapes_sorts_normalizes_dedupes():
    shapes = get_shapes(["w21x44", "W12X26 ", "W16X31", "W12X26"])
    assert [s.name for s in shapes] == ["W12X26", "W16X31", "W21X44"]
    weights = [s.weight_plf for s in shapes]
    assert weights == sorted(weights)


def test_get_shapes_unknown_name_raises():
    with pytest.raises(ValueError, match="W99X99"):
        get_shapes(["W18X35", "W99X99"])
