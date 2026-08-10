"""Per-member design verification: MemberDemand + candidate WShape -> unity checks.

The same check_member() drives both the optimizer's candidate screening and
the final reported check table, so there is exactly one code path for design
acceptance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from ..analysis import MemberDemand
from ..config import (BEAM, COLUMN, IN_TO_M, KIP_IN_TO_KNM, KIP_TO_KN,
                      M_TO_IN, MPA_TO_KSI, FrameConfig)
from ..sections import WShape
from . import aisc_strengths as st

# AISC 360 Eq. F1-1 evaluated at the quarter points of a parabolic moment
# diagram (uniformly loaded simple span, single unbraced segment):
# Cb = 12.5*M / (2.5*M + 3*(0.75M) + 4*M + 3*(0.75M)) = 12.5/11
CB_SIMPLE_SPAN = 12.5 / 11.0


@dataclass(frozen=True)
class GroupRules:
    """Design rules for one member group.

    The defaults are the conservative end of each rule, so a group whose real
    bracing/service conditions were not stated can only be over-designed:

    * ``Lb_in=None`` — LTB unbraced length = full member length (no bracing
      credited; AISC 360 F2). Set to the physical brace spacing (deck
      fasteners, purlins, ...) to credit bracing; 0.0 means continuously
      braced.
    * ``check_deflection`` — vertical-sag serviceability limits
      (IBC Table 1604.3 style span/ratio), on by default at the floor-member
      ratios (span/360 live, span/240 total — stricter than typical roof
      limits, so the default can only over-design). Ratios are
      span/`defl_live_ratio` under live and span/`defl_total_ratio` under
      total service load. Vertical members report zero sag, so the check is
      harmless where it does not apply; disable it only deliberately.
    * ``check_slenderness`` — KL/r <= 200 proportioning limit (AISC 360 E2
      user note); enable for groups that resist compression as their primary
      action (columns), not for flexural members with incidental axial.
      (Off by default because it is advisory, and the E3 strength check
      already penalizes slender compression members.)
    * ``Cb_simple_span`` — when the member is a single unbraced segment
      (Lb >= member length), use Cb = 12.5/11 per AISC F1-1 for a parabolic
      (uniformly loaded, simply supported) moment diagram instead of the
      conservative Cb = 1.0. Enable only for groups whose members really are
      gravity-loaded simple spans. When braces subdivide the span (Lb < L)
      Cb stays 1.0: the governing near-midspan segment has an almost uniform
      moment diagram, for which Cb = 1.0 is essentially exact.
    * ``camber_in`` — fabrication camber credited against the TOTAL-load
      deflection check only (live-load deflection is unaffected). The credit
      never reduces the checked total below the live-load deflection, so
      over-declared camber cannot hide a live-load problem. Specify no more
      than the dead-load deflection (typically ~75%% of it).
    * ``KLx_in`` / ``KLy_in`` — compression effective lengths per axis
      (AISC E2). ``None`` keeps the conservative default KL = member length
      (K = 1, pin-pin, unbraced). Truss chords set these to the panel length
      (in-plane) and the physical brace spacing (out-of-plane), where the
      member length would be the full truss span. Also used by the
      slenderness proportioning check.
    * ``slenderness_tension_limit`` — L/r limit applied when the member is
      in TENSION and ``check_slenderness`` is on (AISC D1 user note: 300,
      advisory). Compression keeps the E2 limit of 200.
    * ``apply_B1`` — amplify the major-axis moment of compression members by
      the AISC Appendix 8 B1 multiplier (member-level second order, Cm = 1.0
      for transverse loading; B2 = 1 because the model is gravity-only and
      non-sway). Enable for truss chords, where axial and bending interact.
    * ``defl_scale_axial`` — project deflections onto candidate sections by
      the AREA ratio instead of the Ix ratio. Truss sag is dominated by
      chord axial flexibility (delta ~ 1/A); the chord's own Ix is nearly
      unrelated to truss stiffness. The scale only steers screening — the
      final table is always an FEA of the final assignment, where the scale
      is exactly 1 either way.
    """
    Lb_in: float | None = None
    check_deflection: bool = True
    defl_live_ratio: float = 360.0
    defl_total_ratio: float = 240.0
    check_slenderness: bool = False
    Cb_simple_span: bool = False
    camber_in: float = 0.0
    KLx_in: float | None = None
    KLy_in: float | None = None
    slenderness_tension_limit: float = 300.0
    apply_B1: bool = False
    defl_scale_axial: bool = False


@dataclass(frozen=True)
class CheckParams:
    """Material (internal ksi) + per-group design rules. Built from an SI
    config via from_material()/from_config(); the strength equations run in
    the AISC-native kip/inch system."""
    Fy: float
    Fu: float
    E: float
    group_rules: dict[str, GroupRules]

    @classmethod
    def from_material(cls, config, group_rules: dict[str, GroupRules]) -> "CheckParams":
        """Convert any config's SI material (MPa) to internal ksi and attach
        the given per-group rules."""
        return cls(
            Fy=config.Fy_mpa * MPA_TO_KSI,
            Fu=config.Fu_mpa * MPA_TO_KSI,
            E=config.E_mpa * MPA_TO_KSI,
            group_rules=group_rules,
        )

    @classmethod
    def from_config(cls, config: FrameConfig) -> "CheckParams":
        """Rules for the conventional grid frame's two groups, replicating the
        long-standing behavior: beams may credit deck bracing and carry the
        deflection checks; columns carry the KL/r limit."""
        beam_rules = GroupRules(
            Lb_in=None if config.beam_Lb_m is None else config.beam_Lb_m * M_TO_IN,
            check_deflection=config.check_deflection,
            defl_live_ratio=config.defl_live_ratio,
            defl_total_ratio=config.defl_total_ratio,
            Cb_simple_span=True,   # every floor member is a uniformly loaded
                                   # simple span (pinned ends, line loads only)
        )
        column_rules = GroupRules(
            check_deflection=False,   # columns: no sag check (they report 0 anyway)
            check_slenderness=config.enforce_slenderness_limit,
        )
        return cls.from_material(
            config, group_rules={COLUMN: column_rules, BEAM: beam_rules})

    def rules_for(self, group: str) -> GroupRules:
        try:
            return self.group_rules[group]
        except KeyError:
            raise KeyError(
                f"No design rules defined for member group '{group}'. Every "
                "group in the geometry needs a GroupRules entry in "
                f"CheckParams.group_rules (have: {sorted(self.group_rules)})."
            ) from None


def _unbraced_length(demand: MemberDemand, rules: GroupRules) -> float:
    """Lb for LTB: the group's brace spacing if one was declared, otherwise
    the full member length (conservative — no bracing credited)."""
    if rules.Lb_in is not None:
        return rules.Lb_in
    return demand.length_in


def check_member(shape: WShape, demand: MemberDemand, params: CheckParams) -> dict:
    """All applicable unity checks for one member with a candidate section.

    Deflections were computed by FEA with section demand.Ix_used; elastic
    deflection scales as 1/I, so they are projected onto the candidate with
    the ratio Ix_used/Ix (exact when the candidate is the analyzed section,
    since loads are re-derived each optimizer iteration).
    """
    Fy, Fu, E = params.Fy, params.Fu, params.E
    rules = params.rules_for(demand.group)
    L = demand.length_in
    Lb = _unbraced_length(demand, rules)
    # compression effective lengths: the group's declared bracing, else the
    # conservative KL = member length (K = 1, pin-pin). Explicit None checks:
    # 0.0 means continuously braced, same convention as Lb_in.
    KLx = rules.KLx_in if rules.KLx_in is not None else L
    KLy = rules.KLy_in if rules.KLy_in is not None else L

    # axial capacity consistent with the force sign
    if demand.Pu < 0.0:
        phi_Pn, ax_clause = st.compression_capacity(shape, Fy, E, KLx=KLx, KLy=KLy)
    else:
        phi_Pn, ax_clause = st.tension_capacity(shape, Fy)
    uc_axial = abs(demand.Pu) / max(phi_Pn, 1e-12)

    # AISC Appendix 8 B1: member-level second-order amplification of the
    # major-axis moment for compression members (Cm = 1.0, transverse
    # loading; B2 = 1, non-sway gravity model). Pu at/above the in-plane
    # Euler load collapses the denominator, producing an enormous amplified
    # moment that correctly fails the member yet still ranks candidates.
    B1 = 1.0
    if rules.apply_B1 and demand.Pu < 0.0 and demand.Mux > 0.0:
        Pe1 = math.pi**2 * E * shape.Ix / max(KLx, 1e-9)**2
        B1 = max(1.0, 1.0 / max(1.0 - abs(demand.Pu) / Pe1, 1e-9))
    Mux = B1 * demand.Mux

    Cb = CB_SIMPLE_SPAN if (rules.Cb_simple_span and Lb >= L) else 1.0
    phi_Mnx, mx_clause = st.flexure_major_capacity(shape, Fy, E, Lb=Lb, Cb=Cb)
    uc_mx = Mux / max(phi_Mnx, 1e-12)

    phi_Mny, my_clause = st.flexure_minor_capacity(shape, Fy, E)
    uc_my = demand.Muy / max(phi_Mny, 1e-12)

    phi_Vn, v_clause = st.shear_capacity(shape, Fy, E)
    uc_v = demand.Vu / max(phi_Vn, 1e-12)

    uc_h1, h1_clause = st.interaction_h1(demand.Pu, Mux, demand.Muy,
                                         phi_Pn, phi_Mnx, phi_Mny)

    ucs = {
        "UC_axial": uc_axial,
        "UC_Mx": uc_mx,
        "UC_My": uc_my,
        "UC_V": uc_v,
        "UC_H1": uc_h1,
    }

    # Slenderness proportioning: KL/r <= 200 in compression (E2 user note),
    # L/r <= 300 in tension (D1 user note), enabled per group. Uses the same
    # per-axis effective lengths as the strength check.
    if rules.check_slenderness:
        slend = max(KLx / shape.rx, KLy / shape.ry)
        limit = 200.0 if demand.Pu <= 0.0 else rules.slenderness_tension_limit
        ucs["UC_slenderness"] = slend / limit

    if rules.check_deflection:
        if rules.defl_scale_axial and demand.A_used > 0.0:
            scale = demand.A_used / shape.A
        else:
            scale = demand.Ix_used / shape.Ix
        d_live = demand.defl_live_in * scale
        d_total = demand.defl_total_in * scale
        if rules.camber_in > 0.0:
            # camber offsets dead-load sag; never credit below the live sag
            d_total = max(d_total - rules.camber_in, d_live)
        ucs["UC_defl_live"] = d_live / (L / rules.defl_live_ratio)
        ucs["UC_defl_total"] = d_total / (L / rules.defl_total_ratio)

    governing_limit, governing_uc = max(ucs.items(), key=lambda kv: kv[1])

    # demands/capacities are computed in kips and inches; report them in SI
    return {
        "member": demand.name,
        "group": demand.group,
        "story": demand.story,
        "profile": shape.name,
        "length_m": demand.length_in * IN_TO_M,
        "Pu_kN": demand.Pu * KIP_TO_KN,
        "phiPn_kN": phi_Pn * KIP_TO_KN,
        "axial_clause": ax_clause,
        "Mux_kNm": Mux * KIP_IN_TO_KNM,   # B1-amplified when apply_B1 is on
        "B1": B1,
        "phiMnx_kNm": phi_Mnx * KIP_IN_TO_KNM,
        "Mx_clause": mx_clause,
        "Muy_kNm": demand.Muy * KIP_IN_TO_KNM,
        "phiMny_kNm": phi_Mny * KIP_IN_TO_KNM,
        "My_clause": my_clause,
        "Vu_kN": demand.Vu * KIP_TO_KN,
        "phiVn_kN": phi_Vn * KIP_TO_KN,
        "V_clause": v_clause,
        "H1_clause": h1_clause,
        **ucs,
        "governing_uc": governing_uc,
        "governing_limitstate": governing_limit.replace("UC_", ""),
        "PASS": governing_uc <= 1.0,
    }


def member_passes(shape: WShape, demand: MemberDemand, params: CheckParams) -> bool:
    return bool(check_member(shape, demand, params)["PASS"])


def check_all(demands: list[MemberDemand], assignment: dict[str, WShape],
              params: CheckParams) -> pd.DataFrame:
    """Full check table for a {group: shape} assignment."""
    rows = [check_member(assignment[d.group], d, params) for d in demands]
    return pd.DataFrame(rows)
