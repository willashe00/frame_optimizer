"""Square-HSS catalog and AISC 360 strength equations.

Anchors follow the existing philosophy: each limit state is reproduced from
its clause by hand, so a change to the implementation has to disagree with
the code book, not just with a stored number. (Fy = 50 ksi, E = 29000 ksi.)
"""
import math

import pytest

from frame_optimizer.design import (compression_capacity,
                                    flexure_major_capacity,
                                    flexure_minor_capacity, shear_capacity,
                                    tension_capacity)
from frame_optimizer.sections import (HSSShape, WShape, get_shapes,
                                      load_hss_shapes)

FY, E = 50.0, 29000.0
CAT = load_hss_shapes()

# HSS8X8X1/4: noncompact flange in flexure (b/tdes = 31.3 vs lambda_p = 27.0),
# nonslender in compression (vs lambda_r = 33.7) - the interesting middle case.
NONCOMPACT = "HSS8X8X1/4"
# HSS10X10X1/4: b/tdes = 39.9, slender both in compression (E7) and in
# flexure (F7-3).
SLENDER = "HSS10X10X1/4"
# HSS8X8X5/8: b/tdes = 10.8, compact everywhere.
COMPACT = "HSS8X8X5/8"


# ---------------------------------- catalog ----------------------------------

def test_catalog_loads_square_only():
    assert len(CAT) == 107
    for s in CAT.values():
        assert s.Ht == pytest.approx(s.B)          # square
        assert s.Ix == pytest.approx(s.Iy)
        assert s.rx == pytest.approx(s.ry)
        assert s.b_tdes == pytest.approx(s.h_tdes)
        assert s.A > 0 and s.weight_plf > 0
        assert s.tdes == pytest.approx(0.93 * s.tnom, rel=0.02)


def test_hss8x8x1_4_properties_match_manual():
    s = CAT[NONCOMPACT]
    assert s.A == pytest.approx(7.10)
    assert s.Ht == pytest.approx(8.0)
    assert s.tdes == pytest.approx(0.233)
    assert s.Zx == pytest.approx(20.5)
    assert s.Sx == pytest.approx(17.7)
    assert s.rx == pytest.approx(3.15)
    # flat width recovered from the published ratio, not re-assumed
    assert s.b_flat == pytest.approx(s.b_tdes * s.tdes)


def test_get_shapes_resolves_both_families_together():
    shapes = get_shapes(["hss8x8x1/4 ", "W18X35", "HSS8X8X1/4"])   # dedupes
    # one list, sorted lightest-first across both families (25.8 vs 35.0 plf)
    assert [s.name for s in shapes] == ["HSS8X8X1/4", "W18X35"]
    assert isinstance(shapes[0], HSSShape)
    assert isinstance(shapes[1], WShape)


def test_unknown_name_names_both_families():
    with pytest.raises(ValueError, match="HSS8X8X1/2"):
        get_shapes(["HSS9X9X9"])


# -------------------------------- compression --------------------------------

def test_compression_nonslender_matches_e3_by_hand():
    s = CAT[NONCOMPACT]
    KL = 170.6                                   # a truss vertical, ~4.3 m
    klr = KL / s.rx
    Fe = math.pi**2 * E / klr**2
    Fcr = 0.658 ** (FY / Fe) * FY                # E3-2, inelastic
    phi_pn, clause = compression_capacity(s, FY, E, KLx=KL, KLy=KL)
    assert phi_pn == pytest.approx(0.90 * Fcr * s.A)
    assert "E3" in clause                        # walls fully effective


def test_compression_slender_walls_reduce_area():
    s = CAT[SLENDER]
    assert s.b_tdes > 1.40 * math.sqrt(E / FY)   # slender per Table B4.1a
    KL = 170.6
    Fcr = 0.658 ** (FY / (math.pi**2 * E / (KL / s.rx) ** 2)) * FY
    phi_pn, clause = compression_capacity(s, FY, E, KLx=KL, KLy=KL)
    assert "E7" in clause
    assert phi_pn < 0.90 * Fcr * s.A             # E7 effective area bit


def test_hss8x8x1_2_matches_aisc_manual_tables():
    """Independent anchors: the Manual tabulates square HSS at Fy = 46 ksi
    (ASTM A500 Gr. B), so these are checked at 46, not at the 50 this project
    designs to. Table 4-3 phi*Pn: 559 kips at zero length, 425 kips at
    Lc = 16 ft. Table 3-12 phi_b*Mn: 129 kip-ft."""
    s = CAT["HSS8X8X1/2"]
    fy46 = 46.0

    squash, _ = compression_capacity(s, fy46, E, KLx=0.0, KLy=0.0)
    assert squash == pytest.approx(559.0, rel=0.005)

    Lc = 16.0 * 12
    phi_pn, _ = compression_capacity(s, fy46, E, KLx=Lc, KLy=Lc)
    assert phi_pn == pytest.approx(425.0, rel=0.01)

    phi_mn, clause = flexure_major_capacity(s, fy46, E, Lb=0.0)
    assert phi_mn / 12.0 == pytest.approx(129.0, rel=0.005)
    assert "F7-1" in clause          # compact at 46 ksi and at 50


def test_tension_is_gross_area_yielding():
    s = CAT[NONCOMPACT]
    phi_tn, clause = tension_capacity(s, FY)
    assert phi_tn == pytest.approx(0.90 * FY * s.A)
    assert "D2" in clause


# ---------------------------------- flexure ----------------------------------

def test_compact_section_reaches_mp():
    s = CAT[COMPACT]
    assert s.b_tdes <= 1.12 * math.sqrt(E / FY)
    phi_mn, clause = flexure_major_capacity(s, FY, E, Lb=200.0)
    assert phi_mn == pytest.approx(0.90 * FY * s.Zx)
    assert "F7-1" in clause


def test_noncompact_flange_matches_f7_2_by_hand():
    s = CAT[NONCOMPACT]
    Mp = FY * s.Zx
    Mn = Mp - (Mp - FY * s.Sx) * (3.57 * s.b_tdes * math.sqrt(FY / E) - 4.0)
    phi_mn, clause = flexure_major_capacity(s, FY, E, Lb=100.0)
    assert phi_mn == pytest.approx(0.90 * Mn)
    assert "F7-2" in clause


def test_slender_flange_uses_effective_modulus():
    s = CAT[SLENDER]
    phi_mn, clause = flexure_major_capacity(s, FY, E, Lb=100.0)
    assert "F7-3" in clause
    Se = phi_mn / 0.90 / FY
    assert 0.0 < Se < s.Sx                       # effective width bites
    assert phi_mn < 0.90 * FY * s.Zx


@pytest.mark.parametrize("name", [COMPACT, NONCOMPACT, SLENDER])
def test_lb_is_inert_and_axes_are_equal_for_a_square(name):
    """AISC F7.4 has no LTB limit state for a square HSS, so the unbraced
    length must not change the answer, and both axes must agree."""
    s = CAT[name]
    ref = flexure_major_capacity(s, FY, E, Lb=0.0).phi_Rn
    for Lb in (50.0, 500.0, 5000.0):
        assert flexure_major_capacity(s, FY, E, Lb=Lb).phi_Rn == pytest.approx(ref)
    assert flexure_minor_capacity(s, FY, E).phi_Rn == pytest.approx(ref)


# ----------------------------------- shear -----------------------------------

def test_shear_uses_both_walls_g4():
    s = CAT[NONCOMPACT]
    assert s.h_tdes <= 1.10 * math.sqrt(5.0 * E / FY)     # Cv2 = 1.0
    Aw = 2.0 * s.h_flat * s.tdes
    phi_vn, clause = shear_capacity(s, FY, E)
    assert phi_vn == pytest.approx(0.90 * 0.6 * FY * Aw)
    assert "G4" in clause
