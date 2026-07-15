"""Strength functions vs AISC Manual anchor values and hand calculations
(Fy = 50 ksi, E = 29000 ksi throughout)."""
import math

import pytest

from frame_optimizer.design import (
    compression_capacity,
    flexure_major_capacity,
    flexure_minor_capacity,
    interaction_h1,
    shear_capacity,
    tension_capacity,
)
from frame_optimizer.sections import load_w_shapes

FY, FU, E = 50.0, 65.0, 29000.0
CAT = load_w_shapes()


# ---------------- flexure, W18X35 anchors (Manual Table 3-2) ----------------

def test_w18x35_phi_mp():
    # Table 3-2: phi_b*Mp = 249 kip-ft
    s = CAT["W18X35"]
    phi_mn, clause = flexure_major_capacity(s, FY, E, Lb=0.0)
    assert phi_mn / 12.0 == pytest.approx(249.0, rel=0.01)
    assert "F2-1" in clause


def test_w18x35_ltb_zones_and_lr():
    s = CAT["W18X35"]
    # Manual: Lp = 4.31 ft, Lr = 12.3 ft
    lp_in = 1.76 * s.ry * math.sqrt(E / FY)
    assert lp_in / 12.0 == pytest.approx(4.31, rel=0.01)

    _, clause_plastic = flexure_major_capacity(s, FY, E, Lb=4.0 * 12)
    assert "F2-1" in clause_plastic
    _, clause_inelastic = flexure_major_capacity(s, FY, E, Lb=8.0 * 12)
    assert "F2-2" in clause_inelastic
    _, clause_elastic = flexure_major_capacity(s, FY, E, Lb=13.0 * 12)
    assert "F2-3" in clause_elastic

    # hand-computed elastic LTB anchor at Lb = 25 ft: phi_Mn ~ 50.4 kip-ft
    phi_mn, _ = flexure_major_capacity(s, FY, E, Lb=25.0 * 12)
    assert phi_mn / 12.0 == pytest.approx(50.4, rel=0.02)


def test_flexure_monotonic_in_lb():
    s = CAT["W21X44"]
    values = [flexure_major_capacity(s, FY, E, Lb=lb * 12.0)[0] for lb in range(0, 41, 2)]
    assert all(a >= b - 1e-9 for a, b in zip(values, values[1:]))


def test_flexure_minor_w18x35():
    # Mp_y = min(Fy*Zy, 1.6*Fy*Sy) = min(403, 409.6) = 403 kip-in; compact flange
    s = CAT["W18X35"]
    phi_mn, clause = flexure_minor_capacity(s, FY, E)
    assert phi_mn == pytest.approx(0.9 * 403.0, rel=0.005)
    assert "F6-1" in clause


# ---------------- shear ----------------

def test_w18x35_shear_matches_manual():
    # Manual: phi_v*Vn = 159 kips (phi = 1.00, Cv1 = 1.0)
    s = CAT["W18X35"]
    phi_vn, _ = shear_capacity(s, FY, E)
    assert phi_vn == pytest.approx(1.0 * 0.6 * 50.0 * 17.7 * 0.300, rel=1e-9)
    assert phi_vn == pytest.approx(159.0, rel=0.01)


def test_slender_web_shear_reduced():
    # W16X26: h/tw = 56.8 > 2.24*sqrt(E/Fy) = 53.9 -> phi = 0.9 branch
    s = CAT["W16X26"]
    phi_vn, clause = shear_capacity(s, FY, E)
    assert "slender web" in clause
    assert phi_vn < 1.0 * 0.6 * FY * s.d * s.tw


# ---------------- compression ----------------

def test_squash_load_nonslender():
    # W14X82 (bf/2tf = 5.92, h/tw = 22.4: nonslender): KL ~ 0 -> phi*Fy*A
    s = CAT["W14X82"]
    phi_pn, clause = compression_capacity(s, FY, E, KLx=1e-6, KLy=1e-6)
    assert phi_pn == pytest.approx(0.9 * FY * s.A, rel=1e-6)
    assert "E3" in clause


def test_w12x26_column_hand_calc():
    # KL = 13 ft, governed by y-axis: KL/ry = 156/1.51 = 103.3
    # Fe = pi^2*29000/103.3^2 = 26.83 ksi; Fcr = 0.658^(50/26.83)*50 = 22.9 ksi
    # phi_Pn = 0.9*22.9*7.65 = 158 kips (matches Manual Table 4-1a ~ 158)
    s = CAT["W12X26"]
    phi_pn, clause = compression_capacity(s, FY, E, KLx=156.0, KLy=156.0)
    assert phi_pn == pytest.approx(158.0, rel=0.02)
    assert clause.startswith("E3-y") or clause.startswith("E7-y")


def test_e7_reduces_slender_web_shape():
    # W18X35 web (h/tw = 53.5) is slender under uniform compression at low
    # slenderness (lambda_r = 1.49*sqrt(E/Fy) = 35.9): Ae < A must engage.
    s = CAT["W18X35"]
    phi_pn, clause = compression_capacity(s, FY, E, KLx=1e-6, KLy=1e-6)
    assert phi_pn < 0.9 * FY * s.A * 0.999
    assert "E7" in clause


def test_e7_inactive_when_stress_low():
    # Same W18X35 but very slender (Fcr small): elements fully effective,
    # capacity must equal the plain E3 value.
    s = CAT["W18X35"]
    KL = 25.0 * 12
    klr = KL / s.ry
    Fe = math.pi**2 * E / klr**2
    Fcr = 0.877 * Fe if FY / Fe > 2.25 else 0.658 ** (FY / Fe) * FY
    phi_pn, clause = compression_capacity(s, FY, E, KLx=KL, KLy=KL)
    assert phi_pn == pytest.approx(0.9 * Fcr * s.A, rel=1e-6)
    assert "E3" in clause


# ---------------- tension & interaction ----------------

def test_tension_yielding():
    s = CAT["W18X35"]
    phi_pn, _ = tension_capacity(s, FY)
    assert phi_pn == pytest.approx(0.9 * 50.0 * 10.3)


def test_h1_branches():
    uc_a, eq_a = interaction_h1(Pu=-30.0, Mux=50.0, Muy=0.0,
                                phi_Pc=100.0, phi_Mcx=100.0, phi_Mcy=100.0)
    assert eq_a == "H1-1a"
    assert uc_a == pytest.approx(0.3 + (8.0 / 9.0) * 0.5)

    uc_b, eq_b = interaction_h1(Pu=-10.0, Mux=50.0, Muy=10.0,
                                phi_Pc=100.0, phi_Mcx=100.0, phi_Mcy=100.0)
    assert eq_b == "H1-1b"
    assert uc_b == pytest.approx(0.05 + 0.5 + 0.1)
