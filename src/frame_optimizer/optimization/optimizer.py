from __future__ import annotations

import itertools
import math
import os
from dataclasses import replace

import pandas as pd

from typing import Callable, Union

from ..analysis import MemberDemand, analyze_frame
from ..clear_span import (BOTTOM_CHORD, END_GIRDER, GIRDER, PURLIN, TOP_CHORD,
                          TRUSS_WEB, ClearSpanConfig,
                          build_clear_span_geometry, candidate_layouts,
                          clear_span_check_params)
from ..config import (COLUMN, FT, KPA_TO_KSI, LB_TO_KG, M_TO_IN, MPA_TO_KSI,
                      PLF_TO_KIP_PER_IN, FrameConfig)
from ..design import CheckParams, check_all, check_member
from ..geometry import FrameGeometry, build_geometry
from ..results import OptimizationResult
from ..sections import WShape, get_shapes

AnyConfig = Union[FrameConfig, ClearSpanConfig]
AnalyzeFn = Callable[..., list[MemberDemand]]


# A layout is only ruled out without FEA when its lower-bound UC clears 1.0 by this margin
_PROOF_MARGIN = 1.02

# Startup cost
_SPAWN_COST_S = 1.5

# Estimated solve-and-check cost per model member per layout, seconds.
_COST_PER_MEMBER_S = 9e-2

# When every layout is proven infeasible, how many of the closest ones still
# get a real FEA so the report has a certified check table.
_HOPELESS_ANALYSES = 1

# Iteration cap for those diagnosis-only analyses (they never converge).
_HOPELESS_ITERATIONS = 1


def geometry_for(config: AnyConfig) -> FrameGeometry:
    """Rebuild the frame geometry for any supported config type."""
    if isinstance(config, ClearSpanConfig):
        return build_clear_span_geometry(config)
    return build_geometry(config)


def _prepare(config: AnyConfig) -> tuple[FrameGeometry, CheckParams, AnalyzeFn]:
    """Geometry, per-group design rules, and analysis function for a config.
    Both building types solve in one Pynite model (clear-span purlins are
    explicit members), so they share analyze_frame."""
    if isinstance(config, ClearSpanConfig):
        return geometry_for(config), clear_span_check_params(config), analyze_frame
    return geometry_for(config), CheckParams.from_config(config), analyze_frame


def _candidates_by_group(config: AnyConfig) -> dict[str, list[WShape]]:
    return {g: get_shapes(names) for g, names in config.candidates_by_group.items()}


def _validate_groups(geometry: FrameGeometry, candidates: dict[str, object],
                     params: CheckParams) -> None:
    """Every design group in the geometry needs a candidate list and design
    rules, and every candidate list must correspond to real members —
    otherwise part of the structure would go unchecked or unsized."""
    geo_groups = set(geometry.groups)
    cand_groups = set(candidates)
    if geo_groups != cand_groups:
        raise ValueError(
            f"Candidate groups {sorted(cand_groups)} do not match the "
            f"geometry's member groups {sorted(geo_groups)}."
        )
    missing_rules = geo_groups - set(params.group_rules)
    if missing_rules:
        raise ValueError(
            f"No GroupRules defined for group(s) {sorted(missing_rules)}; "
            "every member group needs design rules (Lb, deflection, KL/r)."
        )


def _group_demands(demands: list[MemberDemand],
                   groups: list[str]) -> dict[str, list[MemberDemand]]:
    grouped: dict[str, list[MemberDemand]] = {g: [] for g in groups}
    for d in demands:
        grouped[d.group].append(d)
    return grouped


def _distinct_demands(demands: list[MemberDemand]) -> list[MemberDemand]:
    """Drop members whose demand is bit-identical to one already in the list.

    check_member()'s unity checks are a pure function of (shape, params, group)
    and the numeric fields keyed below — `name` and `story` only ride along
    into the output row. Symmetric structures produce many members with
    exactly equal demands (all interior purlins of a bay, both columns of a
    frame, ...), so screening one of each is exactly equivalent to screening
    all of them, not an approximation. Equality is exact: near-identical
    values are kept as separate entries and still checked individually.
    """
    unique: dict[tuple, MemberDemand] = {}
    for d in demands:
        unique.setdefault(
            (d.length_in, d.Ix_used, d.Pu, d.Mux, d.Muy, d.Vu,
             d.defl_total_in, d.defl_live_in), d)
    return list(unique.values())


def _screen_group(candidates: list[WShape], demands: list[MemberDemand],
                  params: CheckParams) -> tuple[WShape, bool, float]:
    """Lightest candidate passing every member of the group under the given
    demands. If none passes, return the candidate with the smallest worst-case
    governing UC so iteration can continue, flagged infeasible. The third
    element is that candidate's worst governing UC (<= 1.0 when feasible)."""
    demands = _distinct_demands(demands)
    best_shape, best_worst_uc = None, float("inf")
    for shape in candidates:  # already sorted lightest-first
        worst = max(check_member(shape, d, params)["governing_uc"] for d in demands)
        if worst <= 1.0:
            return shape, True, worst
        if worst < best_worst_uc:
            best_shape, best_worst_uc = shape, worst
    return best_shape, False, best_worst_uc


# ---------------------------------------------------------------------------
# Statics-based starting assignment (clear-span buildings)
# ---------------------------------------------------------------------------

def _presize_clear_span(config: ClearSpanConfig,
                        candidates: dict[str, list[WShape]],
                        params: CheckParams) -> dict[str, WShape]:
    """Starting assignment from closed-form tributary statics — no FEA.

    The clear-span load path is one-way and statically determinate (deck ->
    purlins -> girders -> columns), so simple-span formulas reproduce the FEA
    demands to within the self-weight terms (the hand-calc anchor tests prove
    the match). Screening the candidate lists against these estimated demands
    skips sections that could never pass, which typically saves one to two
    full FEA iterations per layout. The fixed-point loop then verifies and
    corrects the guess against the real model — this is only a starting
    point, never a substitute for the certified final solve.

    All statics here are in the internal kip/inch system.
    """
    E = config.E_mpa * MPA_TO_KSI
    q_d = config.superimposed_dead_kpa * KPA_TO_KSI   # kip/in^2
    q_l = config.live_kpa * KPA_TO_KSI
    sp = config.purlin_spacing_actual_m * M_TO_IN
    s_f = config.frame_spacing_m * M_TO_IN
    span = config.span_m * M_TO_IN
    height = config.height_m * M_TO_IN

    def factored(w_d: float, w_l: float) -> float:
        return max(1.4 * w_d, 1.2 * w_d + 1.6 * w_l)

    def simple_span(group: str, name: str, shape: WShape, L: float,
                    w_d: float, w_l: float, length_in: float | None = None) -> MemberDemand:
        """Uniformly loaded simple-span pseudo-demand referenced to `shape`
        (the checker projects deflections onto other candidates via 1/Ix)."""
        w_u = factored(w_d, w_l)
        EI = E * shape.Ix
        return MemberDemand(
            name=name, group=group, story=1,
            length_in=L if length_in is None else length_in,
            trib_width_in=0.0, shape_used=shape.name, Ix_used=shape.Ix,
            Pu=0.0, Mux=w_u * L**2 / 8.0, Muy=0.0, Vu=w_u * L / 2.0,
            defl_total_in=5.0 * (w_d + w_l) * L**4 / (384.0 * EI),
            defl_live_in=5.0 * w_l * L**4 / (384.0 * EI),
        )

    purlin0 = candidates[PURLIN][0]
    column0 = candidates[COLUMN][0]
    w_purlin = purlin0.weight_plf * PLF_TO_KIP_PER_IN
    w_column = column0.weight_plf * PLF_TO_KIP_PER_IN

    demands: dict[str, list[MemberDemand]] = {}

    # interior purlin: trib = one purlin space, span = frame spacing
    demands[PURLIN] = [simple_span(
        PURLIN, "~P", purlin0, s_f,
        w_d=q_d * sp + w_purlin, w_l=q_l * sp)]

    smear = w_purlin / sp   # purlin self-weight as an equivalent surface load
    if config.is_truss:
        # Pratt truss statics: midspan chord force ~ M/d with M = w*L^2/8,
        # end web forces from the support shear V = w*L/2 (diagonal V/sin
        # theta in tension, first vertical V in compression). The chords'
        # local bending between panel points is ~ w*p^2/8. Truss midspan sag
        # uses the parallel-chord identity I ~ A_chord*d^2/2, consistent
        # with the checker's area-based deflection projection.
        top0 = candidates[TOP_CHORD][0]
        bot0 = candidates[BOTTOM_CHORD][0]
        web0 = candidates[TRUSS_WEB][0]
        depth = config.truss_depth_actual_m * M_TO_IN
        panel = config.truss_panel_m * M_TO_IN
        diag_len = math.hypot(panel, depth)
        w_top = top0.weight_plf * PLF_TO_KIP_PER_IN
        w_bot = bot0.weight_plf * PLF_TO_KIP_PER_IN
        t_wd = (q_d + smear) * s_f + w_top + w_bot
        t_wl = q_l * s_f
        w_u = factored(t_wd, t_wl)
        P_chord = w_u * span**2 / 8.0 / depth
        V = w_u * span / 2.0
        EI = E * (top0.A * depth**2 / 2.0)

        demands[TOP_CHORD] = [MemberDemand(
            name="~TC", group=TOP_CHORD, story=1, length_in=span,
            trib_width_in=0.0, shape_used=top0.name, Ix_used=top0.Ix,
            A_used=top0.A,
            Pu=-P_chord, Mux=w_u * panel**2 / 8.0, Muy=0.0,
            Vu=w_u * panel / 2.0,
            defl_total_in=5.0 * (t_wd + t_wl) * span**4 / (384.0 * EI),
            defl_live_in=5.0 * t_wl * span**4 / (384.0 * EI))]
        demands[BOTTOM_CHORD] = [MemberDemand(
            name="~BC", group=BOTTOM_CHORD, story=1, length_in=span,
            trib_width_in=0.0, shape_used=bot0.name, Ix_used=bot0.Ix,
            A_used=bot0.A,
            Pu=P_chord, Mux=0.0, Muy=0.0, Vu=0.0,
            defl_total_in=0.0, defl_live_in=0.0)]
        demands[TRUSS_WEB] = [
            MemberDemand(
                name="~TD", group=TRUSS_WEB, story=1, length_in=diag_len,
                trib_width_in=0.0, shape_used=web0.name, Ix_used=web0.Ix,
                A_used=web0.A,
                Pu=V * diag_len / depth, Mux=0.0, Muy=0.0, Vu=0.0,
                defl_total_in=0.0, defl_live_in=0.0),
            MemberDemand(
                name="~TV", group=TRUSS_WEB, story=1, length_in=depth,
                trib_width_in=0.0, shape_used=web0.name, Ix_used=web0.Ix,
                A_used=web0.A,
                Pu=-V, Mux=0.0, Muy=0.0, Vu=0.0,
                defl_total_in=0.0, defl_live_in=0.0),
        ]
        w_roof_self = w_top + w_bot   # per-frame roof spanning self-weight
    else:
        girder0 = candidates[GIRDER][0]
        w_girder = girder0.weight_plf * PLF_TO_KIP_PER_IN
        # interior girder: equivalent uniform load (roof + smeared purlin
        # self) x frame spacing + girder self; the exact-cancellation statics
        # of the point-load pattern make w*L^2/8 the true midspan moment.
        g_wd = (q_d + smear) * s_f + w_girder
        g_wl = q_l * s_f
        demands[GIRDER] = [simple_span(GIRDER, "~G", girder0, span,
                                       w_d=g_wd, w_l=g_wl)]
        w_roof_self = w_girder

    if config.has_end_girder_group:
        # end wall: half tributary width, segments between gable columns
        # treated as simple spans (conservative vs. continuity)
        e0 = candidates[END_GIRDER][0]
        seg = span / (config.end_wall_columns + 1)
        demands[END_GIRDER] = [simple_span(
            END_GIRDER, "~EG", e0, seg,
            w_d=(q_d + smear) * s_f / 2.0 + e0.weight_plf * PLF_TO_KIP_PER_IN,
            w_l=q_l * s_f / 2.0, length_in=span)]

    # interior perimeter column: half the roof tributary of one full bay
    # (everything on its side of the span reaches it, via girder/truss or
    # eave purlin) + member self-weights
    p_d = (q_d + smear) * s_f * span / 2.0 + w_roof_self * span / 2.0 \
        + w_column * height
    p_l = q_l * s_f * span / 2.0
    demands[COLUMN] = [MemberDemand(
        name="~C", group=COLUMN, story=1, length_in=height, trib_width_in=0.0,
        shape_used=column0.name, Ix_used=column0.Ix,
        Pu=-factored(p_d, p_l), Mux=0.0, Muy=0.0, Vu=0.0,
        defl_total_in=0.0, defl_live_in=0.0)]

    return {group: _screen_group(cands, demands[group], params)[0]
            for group, cands in candidates.items()}


# ---------------------------------------------------------------------------
# FEA-free infeasibility proof (clear-span buildings)
# ---------------------------------------------------------------------------

def _bound_demands(config: ClearSpanConfig, geometry: FrameGeometry,
                   candidates: dict[str, list[WShape]]) -> dict[str, MemberDemand]:
    """Rigorous LOWER bounds on each design group's governing member demands.

    Every value below is a quantity the real structure must carry *at least*,
    obtained from the determinate one-way load path with all self-weight
    dropped:

    * purlin  - interior line, uniformly loaded simple span of the frame
                spacing carrying q x purlin spacing.
    * girder  - the interior purlin reactions as point loads on a simple span
                (exact superposition: M = sum P*a/2, delta = sum
                P*a*(3L^2-4a^2)/48EI). Eave lines are excluded because they
                run column-to-column, not onto the girder.
    * column  - half the applied roof load on one frame's tributary strip;
                every perimeter column carries at least that by symmetry.

    Dropping self-weight strictly lowers D, and max(1.4D, 1.2D+1.6L) is
    increasing in D, so the factored bounds stay below the true demands.
    Midspan values bound the member maxima from below. `end_girder` is not
    bounded (gable-column continuity makes it awkward); omitting a group only
    weakens the proof, never invalidates it.

    Deflections are referenced to each group's first candidate, matching the
    Ix_used/Ix projection check_member() applies, so the bound is valid for
    every candidate in the list.
    """
    E = config.E_mpa * MPA_TO_KSI
    q_d = config.superimposed_dead_kpa * KPA_TO_KSI   # kip/in^2
    q_l = config.live_kpa * KPA_TO_KSI
    span = config.span_m * M_TO_IN
    height = config.height_m * M_TO_IN
    s_f = config.frame_spacing_m * M_TO_IN
    n_sp = config.n_purlin_spaces
    sp = span / n_sp
    present = set(geometry.groups)

    def factored(w_d: float, w_l: float) -> float:
        return max(1.4 * w_d, 1.2 * w_d + 1.6 * w_l)

    bounds: dict[str, MemberDemand] = {}

    if PURLIN in present and PURLIN in candidates:
        ref = candidates[PURLIN][0]
        w_d, w_l = q_d * sp, q_l * sp          # kip/in, interior line
        w_u = factored(w_d, w_l)
        EI = E * ref.Ix
        bounds[PURLIN] = MemberDemand(
            name="~purlin", group=PURLIN, story=1, length_in=s_f,
            trib_width_in=0.0, shape_used=ref.name, Ix_used=ref.Ix,
            Pu=0.0, Mux=w_u * s_f**2 / 8.0, Muy=0.0, Vu=w_u * s_f / 2.0,
            defl_total_in=5.0 * (w_d + w_l) * s_f**4 / (384.0 * EI),
            defl_live_in=5.0 * w_l * s_f**4 / (384.0 * EI))

    if GIRDER in present and GIRDER in candidates:
        ref = candidates[GIRDER][0]
        P_d, P_l = q_d * sp * s_f, q_l * sp * s_f   # one interior purlin reaction
        P_u = factored(P_d, P_l)
        EI = E * ref.Ix
        M = d_total = d_live = 0.0
        for i in range(1, n_sp):                    # interior purlin lines only
            a = min(i * sp, span - i * sp)
            M += P_u * a / 2.0
            shape_fn = a * (3.0 * span**2 - 4.0 * a**2) / (48.0 * EI)
            d_total += (P_d + P_l) * shape_fn
            d_live += P_l * shape_fn
        bounds[GIRDER] = MemberDemand(
            name="~girder", group=GIRDER, story=1, length_in=span,
            trib_width_in=0.0, shape_used=ref.name, Ix_used=ref.Ix,
            Pu=0.0, Mux=M, Muy=0.0, Vu=P_u * (n_sp - 1) / 2.0,
            defl_total_in=d_total, defl_live_in=d_live)

    if COLUMN in present and COLUMN in candidates:
        ref = candidates[COLUMN][0]
        # an interior frame's perimeter column takes a full bay of tributary;
        # with only end frames every column takes half a bay
        trib = s_f if config.n_frames >= 3 else s_f / 2.0
        p_d, p_l = q_d * trib * span / 2.0, q_l * trib * span / 2.0
        bounds[COLUMN] = MemberDemand(
            name="~column", group=COLUMN, story=1, length_in=height,
            trib_width_in=0.0, shape_used=ref.name, Ix_used=ref.Ix,
            Pu=-factored(p_d, p_l), Mux=0.0, Muy=0.0, Vu=0.0,
            defl_total_in=0.0, defl_live_in=0.0)

    # An end girder is only boundable when nothing holds it up mid-span: with
    # gable columns it runs continuous over them, and a trustworthy lower
    # bound would need their stiffnesses. With none it is the interior-girder
    # case at half the tributary width (one adjacent bay of purlins instead of
    # two), so the identical simple-span superposition applies.
    #
    # Worth doing because the no-gable variants are exactly the ones that tend
    # to fail - an unsupported end girder spans the full building width - and
    # an infeasible layout is the expensive kind to analyze: the fixed-point
    # loop cannot settle on a section that does not exist, so it runs to its
    # iteration cap, every iteration a full FEA.
    if (END_GIRDER in present and END_GIRDER in candidates
            and config.end_wall_columns == 0):
        ref = candidates[END_GIRDER][0]
        P_d, P_l = q_d * sp * s_f / 2.0, q_l * sp * s_f / 2.0
        P_u = factored(P_d, P_l)
        EI = E * ref.Ix
        M = d_total = d_live = 0.0
        for i in range(1, n_sp):
            a = min(i * sp, span - i * sp)
            M += P_u * a / 2.0
            shape_fn = a * (3.0 * span**2 - 4.0 * a**2) / (48.0 * EI)
            d_total += (P_d + P_l) * shape_fn
            d_live += P_l * shape_fn
        bounds[END_GIRDER] = MemberDemand(
            name="~end_girder", group=END_GIRDER, story=1, length_in=span,
            trib_width_in=0.0, shape_used=ref.name, Ix_used=ref.Ix,
            Pu=0.0, Mux=M, Muy=0.0, Vu=P_u * (n_sp - 1) / 2.0,
            defl_total_in=d_total, defl_live_in=d_live)

    return bounds


def _end_girder_estimate(config: ClearSpanConfig, geometry: FrameGeometry,
                         candidates: dict[str, list[WShape]],
                         params: CheckParams) -> float:
    """Advisory end-girder UC — used ONLY to rank layouts, never to reject one.

    The end girders run over their gable columns as continuous members, and a
    trustworthy lower bound on a continuous member's demands needs support
    stiffnesses this stage does not have. So this deliberately goes the other
    way and OVER-estimates, treating each gable-column segment as an
    independent simple span carrying half a bay.

    Ranking needs it because the rigorous bounds are blind to gable columns
    entirely: without this term, layouts with no gable columns (whose end
    girders then span the full width unsupported) look exactly as promising as
    well-supported ones, and the "closest attempt" reported when every layout
    fails could be an obviously silly one.
    """
    if END_GIRDER not in geometry.groups or END_GIRDER not in candidates:
        return 0.0
    ref = candidates[END_GIRDER][0]
    span = config.span_m * M_TO_IN
    s_f = config.frame_spacing_m * M_TO_IN
    seg = span / (config.end_wall_columns + 1)
    q_d = config.superimposed_dead_kpa * KPA_TO_KSI
    q_l = config.live_kpa * KPA_TO_KSI
    w_d = q_d * s_f / 2.0 + ref.weight_plf * PLF_TO_KIP_PER_IN   # half a bay
    w_l = q_l * s_f / 2.0
    w_u = max(1.4 * w_d, 1.2 * w_d + 1.6 * w_l)
    EI = config.E_mpa * MPA_TO_KSI * ref.Ix
    estimate = MemberDemand(
        name="~end_girder", group=END_GIRDER, story=1,
        length_in=span,          # deflection limits key off the member length
        trib_width_in=0.0, shape_used=ref.name, Ix_used=ref.Ix,
        Pu=0.0, Mux=w_u * seg**2 / 8.0, Muy=0.0, Vu=w_u * seg / 2.0,
        defl_total_in=5.0 * (w_d + w_l) * seg**4 / (384.0 * EI),
        defl_live_in=5.0 * w_l * seg**4 / (384.0 * EI))
    return _screen_group(candidates[END_GIRDER], [estimate], params)[2]


def _infeasibility_proof(config: ClearSpanConfig, geometry: FrameGeometry,
                         candidates: dict[str, list[WShape]],
                         params: CheckParams) -> tuple[bool, float]:
    """(provably_infeasible, distance_from_feasible) for one layout, with no FEA.

    Screens every candidate against the lower-bound demands above. Capacities
    depend only on the shape, the group rules and the member length — all
    identical to the real members — so a candidate failing here fails against
    the true (larger) demands too. When no candidate of some group clears its
    lower bound by _PROOF_MARGIN, that group cannot be satisfied by any
    assignment and the layout is infeasible with certainty.

    This also catches the demand-independent limit states exactly rather than
    by bound: KL/r <= 200 on columns depends only on length and shape, so a
    too-tall column rules out every layout outright.

    The second element ranks how far the layout is from feasible; it folds in
    the advisory end-girder estimate so the ranking is not blind to gable
    columns. It is used ONLY to choose which layout to report when every one
    of them is proven infeasible — never to reject one.
    """
    proven = False
    distance = 0.0
    for group, bound in _bound_demands(config, geometry, candidates).items():
        _, ok, uc = _screen_group(candidates[group], [bound], params)
        distance = max(distance, uc)
        if not ok and uc > _PROOF_MARGIN:
            proven = True
    distance = max(distance,
                   _end_girder_estimate(config, geometry, candidates, params))
    return proven, distance


def _initial_assignment(config: AnyConfig, candidates: dict[str, list[WShape]],
                        params: CheckParams,
                        warm_start: dict[str, str] | None) -> dict[str, WShape]:
    """Starting point for the fixed-point loop: a warm-start assignment when
    one is given (e.g. the previous layout's winner), else tributary-statics
    pre-sizing for clear-span buildings, else the lightest candidates."""
    if warm_start is not None:
        by_name = {g: {s.name: s for s in cands} for g, cands in candidates.items()}
        if set(warm_start) == set(candidates) and all(
                warm_start[g] in by_name[g] for g in candidates):
            return {g: by_name[g][warm_start[g]] for g in candidates}
    if isinstance(config, ClearSpanConfig):
        return _presize_clear_span(config, candidates, params)
    return {g: cands[0] for g, cands in candidates.items()}


def _weights(geometry: FrameGeometry, assignment: dict[str, WShape]) -> tuple[float, dict[str, float]]:
    """Total and per-group steel weight in kg (catalog weights are lb/ft)."""
    by_group = {g: 0.0 for g in assignment}
    for m in geometry.members:
        by_group[m.group] += (assignment[m.group].weight_plf
                              * (m.length_in / FT) * LB_TO_KG)
    return sum(by_group.values()), by_group


def _build_result(config: AnyConfig, geometry: FrameGeometry,
                  assignment: dict[str, WShape], demands: list[MemberDemand],
                  params: CheckParams, iterations: list[dict],
                  converged: bool, feasible: bool) -> OptimizationResult:
    table = check_all(demands, assignment, params)
    total, by_group = _weights(geometry, assignment)

    rows = []
    for group in geometry.groups:
        sub = table[table["group"] == group]
        worst = sub.loc[sub["governing_uc"].idxmax()]
        rows.append({
            "group": group,
            "profile": assignment[group].name,
            "n_members": len(sub),
            "weight_kg": by_group[group],
            "max_uc": worst["governing_uc"],
            "governing_limitstate": worst["governing_limitstate"],
            "governing_member": worst["member"],
            "all_pass": bool(sub["PASS"].all()),
        })

    return OptimizationResult(
        feasible=feasible and bool(table["PASS"].all()),
        converged=converged,
        sections={g: s.name for g, s in assignment.items()},
        total_weight_kg=total,
        weight_by_group_kg=by_group,
        member_table=table,
        group_summary=pd.DataFrame(rows),
        iterations=iterations,
        config=config,
    )


def _optimize_iterative(config: AnyConfig, geometry: FrameGeometry,
                        candidates: dict[str, list[WShape]], params: CheckParams,
                        analyze: AnalyzeFn, max_iterations: int, verbose: bool,
                        warm_start: dict[str, str] | None = None) -> OptimizationResult:
    assignment = _initial_assignment(config, candidates, params, warm_start)
    history = [tuple(s.name for s in assignment.values())]
    iterations: list[dict] = []
    converged = False
    feasible = True
    # stability depends on the support/release topology only, so the scan
    # runs once per geometry; later solves of the same model skip it
    demands = analyze(geometry, assignment, config, check_stability=True)

    for it in range(1, max_iterations + 1):
        grouped = _group_demands(demands, list(candidates))
        new_assignment: dict[str, WShape] = {}
        feasible = True
        for group, cands in candidates.items():
            shape, ok, _ = _screen_group(cands, grouped[group], params)
            new_assignment[group] = shape
            feasible &= ok

        iterations.append({
            "iteration": it,
            "assignment": {g: s.name for g, s in new_assignment.items()},
            "feasible_screen": feasible,
        })
        if verbose:
            print(f"[iter {it}] {iterations[-1]['assignment']} feasible={feasible}")

        if all(new_assignment[g] is assignment[g] for g in assignment):
            converged = True   # demands already reflect this exact assignment
            break

        key = tuple(s.name for s in new_assignment.values())
        if key in history:
            # Oscillation between assignments (self-weight feedback): take the
            # heavier shape per group, which can only be conservative.
            new_assignment = {
                g: max(assignment[g], new_assignment[g], key=lambda s: s.weight_plf)
                for g in assignment
            }
            if all(new_assignment[g] is assignment[g] for g in assignment):
                # Damping landed back on the assignment already analyzed, so
                # `demands` describe it exactly and every further iteration
                # would re-solve an identical model and re-derive an identical
                # screen. This is a genuine fixed point of the update rule -
                # breaking here yields the same sections, weights and check
                # table, without the repeated solves. Groups with no passing
                # candidate oscillate like this until the iteration cap, which
                # is what makes infeasible layouts the expensive ones.
                converged = True
                break
        history.append(key)
        assignment = new_assignment
        demands = analyze(geometry, assignment, config, check_stability=False)

    return _build_result(config, geometry, assignment, demands, params,
                         iterations, converged, feasible)


def _optimize_exhaustive(config: AnyConfig, geometry: FrameGeometry,
                         candidates: dict[str, list[WShape]], params: CheckParams,
                         analyze: AnalyzeFn, verbose: bool) -> OptimizationResult:
    groups = list(candidates)
    best = None  # (weight, assignment, demands)
    iterations: list[dict] = []

    first = True
    for combo in itertools.product(*(candidates[g] for g in groups)):
        assignment = dict(zip(groups, combo))
        demands = analyze(geometry, assignment, config, check_stability=first)
        first = False
        table = check_all(demands, assignment, params)
        ok = bool(table["PASS"].all())
        weight, _ = _weights(geometry, assignment)
        iterations.append({
            "assignment": {g: s.name for g, s in assignment.items()},
            "weight_kg": weight,
            "feasible": ok,
        })
        if verbose:
            print(f"[exhaustive] {iterations[-1]}")
        if ok and (best is None or weight < best[0]):
            best = (weight, assignment, demands)

    if best is None:
        # nothing passed; report the last combination as the failed attempt
        return _build_result(config, geometry, assignment, demands, params,
                             iterations, converged=True, feasible=False)
    _, assignment, demands = best
    return _build_result(config, geometry, assignment, demands, params,
                         iterations, converged=True, feasible=True)


def optimize(config: AnyConfig, method: str = "iterative",
             max_iterations: int = 10, verbose: bool = False,
             warm_start: dict[str, str] | None = None,
             second_order: bool = False) -> OptimizationResult:
    """Find the lightest per-group W-shape assignment that passes every
    AISC 360 LRFD check (and serviceability, if enabled). Members of a design
    group share one section; the conventional grid frame has two groups
    ('column', 'beam'), the clear-span building three ('column', 'girder',
    'purlin') — or five with a Pratt-truss roof ('column', 'top_chord',
    'bottom_chord', 'truss_web', 'purlin', plus 'end_girder').

    warm_start ({group: shape name}) seeds the iterative search's starting
    assignment (it is ignored if it does not cover the geometry's groups
    with known candidates). The fixed point reached is verified by FEA
    either way; a warm start only reduces the number of solves.

    second_order=True runs the P-Delta verification on a feasible truss
    design (see _verify_second_order); it is a no-op for girder configs."""
    if isinstance(config, ClearSpanConfig):
        config = _resolve_roof_system(config)
    geometry, params, analyze = _prepare(config)
    candidates = _candidates_by_group(config)
    _validate_groups(geometry, candidates, params)

    if method == "iterative":
        result = _optimize_iterative(config, geometry, candidates, params,
                                     analyze, max_iterations, verbose,
                                     warm_start=warm_start)
    elif method == "exhaustive":
        result = _optimize_exhaustive(config, geometry, candidates, params,
                                      analyze, verbose)
    else:
        raise ValueError("method must be 'iterative' or 'exhaustive'.")

    if (second_order and isinstance(config, ClearSpanConfig)
            and config.is_truss and result.feasible):
        result = _verify_second_order(result)
    return result


# ---------------------------------------------------------------------------
# Layout search
# ---------------------------------------------------------------------------

def _optimize_layout_chunk(payload: tuple) -> list[OptimizationResult]:
    """Optimize a contiguous chunk of candidate layouts serially, warm-
    starting each from the previous winner (adjacent layouts differ little,
    so most re-converge in a single FEA solve). Module-level so worker
    processes can pickle it."""
    config, layouts, method, max_iterations = payload
    results = []
    warm = None
    for n_frames, spacing, gables in layouts:
        variant = replace(config, n_frames=n_frames,
                          purlin_spacing_m=spacing, end_wall_columns=gables)
        result = optimize(variant, method=method,
                          max_iterations=max_iterations, verbose=False,
                          warm_start=warm)
        if result.feasible:
            warm = result.sections
        results.append(result)
    return results


def _run_layouts(config: ClearSpanConfig, layouts: list[tuple[int, float, int]],
                 method: str, max_iterations: int,
                 n_jobs: int | None) -> list[OptimizationResult]:
    """Evaluate every candidate layout, in parallel worker processes when
    that pays off. Layouts are independent, so parallelism cannot change any
    result; chunks stay contiguous to preserve warm-start locality.

    Measured best-of-3, parallel wins at every size tried (1.7x on a small
    footprint, 3.4x on a mid-size one), so the only reason to stay serial is
    having nothing worth splitting: a single chunk, or so little queued work
    that the ~1.5 s per-worker startup (a spawn platform re-imports numpy,
    pandas and Pynite in each) would not be repaid. Estimate the serial cost
    from the work actually queued and compare.
    """
    if not layouts:
        return []
    # Never spawn from inside a worker. On spawn platforms each worker
    # re-imports the caller's __main__ module, so a user script that calls
    # optimize_layout() at module level (no `if __name__ == "__main__"`
    # guard) would otherwise have every worker start another pool, and so on
    # - a fork bomb rather than a slow run. Falling back to serial here keeps
    # such a script merely wasteful instead of fatal.
    import multiprocessing
    if multiprocessing.parent_process() is not None:
        return _optimize_layout_chunk((config, layouts, method, max_iterations))
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1
    # at least ~2 layouts per worker so the per-process import cost and the
    # cold (un-warm-started) first layout are amortized
    n_chunks = max(1, min(n_jobs, math.ceil(len(layouts) / 2)))

    n_members = len(geometry_for(_variant(config, layouts[0])).members)
    serial_s = len(layouts) * n_members * _COST_PER_MEMBER_S
    saving_s = serial_s - serial_s / n_chunks
    if n_chunks <= 1 or saving_s < _SPAWN_COST_S * n_chunks:
        return _optimize_layout_chunk((config, layouts, method, max_iterations))

    bounds = [round(i * len(layouts) / n_chunks) for i in range(n_chunks + 1)]
    chunks = [layouts[bounds[i]:bounds[i + 1]] for i in range(n_chunks)
              if bounds[i] < bounds[i + 1]]
    payloads = [(config, chunk, method, max_iterations) for chunk in chunks]
    try:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=len(payloads)) as pool:
            chunk_results = list(pool.map(_optimize_layout_chunk, payloads))
        return [r for chunk in chunk_results for r in chunk]
    except Exception:
        # multiprocessing unavailable (sandboxed interpreter, pickling issue,
        # ...): the serial path produces identical results, just slower
        return _optimize_layout_chunk((config, layouts, method, max_iterations))


def _variant(config: ClearSpanConfig, layout: tuple[int, float, int]) -> ClearSpanConfig:
    n_frames, spacing, gables = layout
    return replace(config, n_frames=n_frames, purlin_spacing_m=spacing,
                   end_wall_columns=gables)


def _with_roof_system(config: ClearSpanConfig, roof_system: str) -> ClearSpanConfig:
    """replace() with the roof system, keeping auto-derived layout fields
    auto. A plain replace() passes their concrete derived values back in,
    which would silently pin the layout search to a single layout."""
    auto = {name: None for name in config.auto_layout_fields}
    return replace(config, roof_system=roof_system, **auto)


def _resolve_roof_system(config: ClearSpanConfig) -> ClearSpanConfig:
    """Resolve roof_system="auto" to a concrete "girder" or "truss" — no FEA.

    The girder keeps the roof unless the lower-bound proof (the exact
    machinery of _infeasibility_proof) shows that in EVERY candidate layout
    no girder candidate can carry the span; only then, and only when truss
    candidate lists are configured, does the Pratt truss take over. A layout
    whose girder could pass is never taken away from the girder path, so
    "auto" is exactly the historical behavior for buildable girder spans.
    (If the girder search still ends infeasible — the bounds cannot prove
    everything — optimize_layout() falls back to a truss search afterwards.)
    """
    if config.roof_system != "auto":
        return config
    girder_cfg = _with_roof_system(config, "girder")
    if not config.has_truss_groups:
        return girder_cfg
    girder_shapes = get_shapes(config.girder_candidates)
    for layout in candidate_layouts(girder_cfg):
        variant = _variant(girder_cfg, layout)
        geometry = geometry_for(variant)
        params = clear_span_check_params(variant)
        bounds = _bound_demands(variant, geometry, {GIRDER: girder_shapes})
        if GIRDER not in bounds:   # no interior girder members to bound
            return girder_cfg
        _, ok, uc = _screen_group(girder_shapes, [bounds[GIRDER]], params)
        if ok or uc <= _PROOF_MARGIN:
            return girder_cfg      # this layout might carry a W girder
    return _with_roof_system(config, "truss")


def _merge_second_order(first: list[MemberDemand],
                        second: list[MemberDemand]) -> list[MemberDemand]:
    """Second-order verification demands: P-Delta AXIAL forces (enveloped
    with first order), everything else first-order.

    In a non-sway gravity truss the system second-order effect is an axial
    redistribution — that is what the geometrically nonlinear solve is
    trusted for. Member-level moment amplification (P-small-delta) is
    covered by the Appendix 8 B1 factor inside check_member(), per AISC 360
    C2's approximate-method allowance. Pynite's P-Delta moment recovery is
    NOT usable here: for members with released ends it reports order-P*L
    moments that a pin-ended two-force member cannot physically carry."""
    merged = []
    for d1, d2 in zip(first, second):
        assert d1.name == d2.name
        Pu = d2.Pu if abs(d2.Pu) > abs(d1.Pu) else d1.Pu
        merged.append(replace(d1, Pu=Pu))
    return merged


def _verify_second_order(result: OptimizationResult) -> OptimizationResult:
    """Second-order verification of a finished truss design (AISC 360 C2).

    The search already screens with the Appendix 8 B1 amplifier (member
    second order; B2 = 1 for this non-sway gravity model). This step
    additionally re-solves the winning assignment geometrically nonlinear
    (Pynite P-Delta) and re-runs every member check with the second-order
    axial forces (see _merge_second_order) — the reported check table IS
    that certification. If a member fails, one corrective re-screen picks
    heavier sections against the second-order demands and re-verifies."""
    config = result.config
    geometry, params, analyze = _prepare(config)
    candidates = _candidates_by_group(config)
    by_name = {g: {s.name: s for s in cands} for g, cands in candidates.items()}
    assignment = {g: by_name[g][name] for g, name in result.sections.items()}
    info = {"method": "P-Delta axial + App. 8 B1", "verified": False,
            "passes": 0, "sections_changed": False, "note": ""}

    def second_order_demands(assignment):
        first = analyze(geometry, assignment, config, check_stability=False)
        second = analyze(geometry, assignment, config,
                         check_stability=False, second_order=True)
        return _merge_second_order(first, second)

    try:
        demands = second_order_demands(assignment)
        info["passes"] = 1
        table = check_all(demands, assignment, params)
        if not bool(table["PASS"].all()):
            grouped = _group_demands(demands, list(candidates))
            new_assignment = {g: _screen_group(cands, grouped[g], params)[0]
                              for g, cands in candidates.items()}
            info["sections_changed"] = any(
                new_assignment[g] is not assignment[g] for g in assignment)
            assignment = new_assignment
            demands = second_order_demands(assignment)
            info["passes"] = 2
            table = check_all(demands, assignment, params)
        info["verified"] = bool(table["PASS"].all())
    except RuntimeError as exc:
        # divergence: keep the first-order design, flagged unverified
        info["note"] = str(exc)
        result.second_order = info
        return result

    verified = _build_result(config, geometry, assignment, demands, params,
                             result.iterations, result.converged,
                             feasible=info["verified"])
    verified.second_order = info
    return verified


def _prescreen(config: ClearSpanConfig, layouts: list[tuple[int, float, int]],
               verbose: bool) -> tuple[list[tuple[int, float, int]],
                                       dict[tuple, float], bool]:
    """Split the candidate layouts into those needing FEA and those already
    proven infeasible by statics (see _infeasibility_proof).

    Returns (layouts to analyze, {skipped layout: distance}, hopeless). Any
    layout that could possibly be feasible is always analyzed, so the design
    this search settles on is unchanged.

    When statics rules out *every* layout there is no design to find, only a
    diagnosis to report, so the _HOPELESS_ANALYSES closest layouts are
    analyzed anyway: the caller then still returns a complete, FEA-certified
    check table showing what fails and by how much, instead of a bare
    "nothing works". `hopeless` flags that case so the caller can also cap
    the iteration count (see _HOPELESS_ITERATIONS).
    """
    to_run: list[tuple[int, float, int]] = []
    skipped: dict[tuple, float] = {}
    for layout in layouts:
        variant = _variant(config, layout)
        geometry = geometry_for(variant)
        params = clear_span_check_params(variant)
        candidates = _candidates_by_group(variant)
        is_proven, distance = _infeasibility_proof(variant, geometry,
                                                   candidates, params)
        if is_proven:
            skipped[layout] = distance
        else:
            to_run.append(layout)

    hopeless = not to_run and bool(skipped)
    if hopeless:
        closest = sorted(skipped, key=lambda k: skipped[k])[:_HOPELESS_ANALYSES]
        for layout in closest:
            del skipped[layout]
        # keep the caller's original layout order for a stable report
        to_run = [l for l in layouts if l in set(closest)]
    if verbose and skipped:
        print(f"[prescreen] {len(skipped)} of {len(layouts)} layout(s) ruled "
              f"out by statics without FEA; {len(to_run)} analyzed"
              + (" for diagnosis only (no layout can be feasible)"
                 if hopeless else ""))
    return to_run, skipped, hopeless


def optimize_layout(config: ClearSpanConfig, method: str = "iterative",
                    max_iterations: int = 10, verbose: bool = False,
                    n_jobs: int | None = None,
                    prescreen: bool = True) -> OptimizationResult:
    """Clear-span layout search: determine the building layout (n_frames,
    purlin_spacing_m, end_wall_columns) from the footprint by optimizing
    every realistic layout and keeping the lightest feasible design.

    The footprint (span_m, length_m, height_m) is the fixed input.
    Layout fields left to auto-derive on the config are the search variables,
    ranging over the practice bands in clear_span.py; any field set
    explicitly is honored as-is. The winning result's .config carries the
    chosen layout and .layout_search records every layout considered (each
    entry flagged `analyzed` — False for layouts statics ruled out without a
    solve). If no layout yields a feasible design, the closest attempt is
    returned with feasible=False; see the `prescreen` note below for exactly
    what "closest" means then.

    n_jobs: worker processes for the layout search (None = one per CPU,
    capped at one per ~2 layouts; 1 = serial). Parallel and serial runs
    produce identical designs.

    prescreen: skip the FEA for layouts that closed-form statics prove no
    candidate section can satisfy (see _infeasibility_proof). Set False to
    force a full analysis of every layout (validation only — it is strictly
    slower). The guarantee is:

    * Whenever a feasible design exists, the same lightest one is returned.
      A layout is only skipped once it is certain no candidate can satisfy
      it, so nothing that could have won is ever discarded.
    * When NO layout is feasible, the result is a diagnosis rather than a
      design, and the layout it reports may differ from the exhaustive
      search's pick: both are infeasible, and the reported one is the
      closest by statics rather than the closest by full analysis. It is
      still a real, FEA-certified check table.
    """
    if not isinstance(config, ClearSpanConfig):
        raise TypeError(
            "optimize_layout() searches clear-span building layouts; a "
            "conventional grid frame (FrameConfig) has its layout given "
            "explicitly — use optimize() for it."
        )

    was_auto = config.roof_system == "auto"
    config = _resolve_roof_system(config)
    if verbose and was_auto and config.is_truss:
        print("[roof] no W girder candidate can carry the span in any "
              "layout — switching to the Pratt truss roof system")

    layouts = candidate_layouts(config)
    if prescreen:
        analyzed, skipped, hopeless = _prescreen(config, layouts, verbose)
    else:
        analyzed, skipped, hopeless = layouts, {}, False
    # a proven-infeasible layout cannot become feasible by iterating, so the
    # diagnosis analyses do not chase a fixed point they will never reach
    iterations = _HOPELESS_ITERATIONS if hopeless else max_iterations
    results = dict(zip(analyzed,
                       _run_layouts(config, analyzed, method, iterations, n_jobs)))

    best = best_key = None
    fallback = fallback_uc = None
    search: list[dict] = []
    for layout in layouts:
        n_frames, spacing, gables = layout
        result = results.get(layout)
        variant = result.config if result is not None else _variant(config, layout)
        # a screened-out layout has no check table; its statics distance is a
        # lower bound on the worst UC any assignment could achieve there
        worst_uc = (float(result.member_table["governing_uc"].max())
                    if result is not None else skipped[layout])
        search.append({
            "n_frames": n_frames,
            "frame_spacing_m": variant.frame_spacing_m,
            "purlin_spacing_m": variant.purlin_spacing_actual_m,
            "end_wall_columns": gables,
            "feasible": bool(result is not None and result.feasible),
            "total_weight_kg": result.total_weight_kg if result is not None else None,
            "worst_uc": worst_uc,
            "analyzed": result is not None,
        })
        if verbose:
            if result is None:
                outcome = f"infeasible by statics (bound UC >= {worst_uc:.2f})"
            elif result.feasible:
                outcome = f"{result.total_weight_kg:,.0f} kg"
            else:
                outcome = f"infeasible (worst UC {worst_uc:.2f})"
            print(f"[layout] {n_frames} frames @ {variant.frame_spacing_m:.2f} m, "
                  f"purlins @ {variant.purlin_spacing_actual_m:.2f} m, "
                  f"{gables} gable col(s)/end -> {outcome}")

        if result is None:
            continue
        if result.feasible:
            # lightest wins; weights within a kilogram are a practical tie,
            # broken toward fewer members (less fabrication and erection)
            key = (round(result.total_weight_kg), len(result.member_table))
            if best is None or key < best_key:
                best, best_key = result, key
        elif fallback is None or worst_uc < fallback_uc:
            fallback, fallback_uc = result, worst_uc

    chosen = best if best is not None else fallback
    # second-order certification of the winner only: the search screened with
    # the B1 amplifier; the final truss design is re-verified by a real
    # P-Delta solve (and rebuilt from it, so the table IS second-order)
    if chosen.feasible and chosen.config.is_truss:
        chosen = _verify_second_order(chosen)
    chosen.layout_search = search

    # "auto" safety net: the girder proof is one-sided (bounds can only rule
    # layouts OUT), so a girder search can still end infeasible for reasons
    # the statics cannot see. With truss candidates available, the truss
    # search gets its turn before reporting failure.
    if (was_auto and not chosen.feasible and not config.is_truss
            and config.has_truss_groups):
        if verbose:
            print("[roof] no feasible W-girder design — retrying with the "
                  "Pratt truss roof system")
        truss_result = optimize_layout(
            _with_roof_system(config, "truss"), method=method,
            max_iterations=max_iterations, verbose=verbose, n_jobs=n_jobs,
            prescreen=prescreen)
        if truss_result.feasible:
            return truss_result
    return chosen


def evaluate(config: AnyConfig, sections: dict[str, str]) -> OptimizationResult:
    """Analyze + check one explicit {'column': 'W12X40', 'beam': 'W18X35'}
    assignment without searching."""
    geometry, params, analyze = _prepare(config)
    catalog = {g: get_shapes([name])[0] for g, name in sections.items()}
    _validate_groups(geometry, catalog, params)
    demands = analyze(geometry, catalog, config)
    return _build_result(config, geometry, catalog, demands, params,
                         iterations=[], converged=True, feasible=True)
