"""Lightest-section search over the candidate catalogs.

Default method 'iterative' exploits the physics of a fully pinned gravity
frame: member demands are statically determinate apart from self-weight, so
they barely move when sections change. A fixed-point loop therefore converges
in a handful of FEA solves:

    assign the lightest candidate to every group
    repeat:
        FEA with the current assignment (self-weight included)
        per group: pick the lightest candidate whose checks pass for every
                   member of the group under the current demands
    until the assignment stops changing

Convergence note: when the loop exits because the assignment repeated, the
screening demands came from an FEA of exactly that assignment, so the final
check table IS the certification - no extra solve needed.

Method 'exhaustive' re-analyzes every combination (Cartesian product) and is
provided for validation on small candidate lists.
"""
from __future__ import annotations

import itertools
from dataclasses import replace

import pandas as pd

from typing import Callable, Union

from ..analysis import MemberDemand, analyze_frame
from ..clear_span import (ClearSpanConfig, build_clear_span_geometry,
                          candidate_layouts, clear_span_check_params)
from ..config import FT, FrameConfig
from ..design import CheckParams, check_all, check_member
from ..geometry import FrameGeometry, build_geometry
from ..results import OptimizationResult
from ..sections import WShape, get_shapes

AnyConfig = Union[FrameConfig, ClearSpanConfig]
AnalyzeFn = Callable[[FrameGeometry, dict[str, WShape], AnyConfig], list[MemberDemand]]


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


def _screen_group(candidates: list[WShape], demands: list[MemberDemand],
                  params: CheckParams) -> tuple[WShape, bool]:
    """Lightest candidate passing every member of the group under the given
    demands. If none passes, return the candidate with the smallest worst-case
    governing UC so iteration can continue, flagged infeasible."""
    best_shape, best_worst_uc = None, float("inf")
    for shape in candidates:  # already sorted lightest-first
        worst = max(check_member(shape, d, params)["governing_uc"] for d in demands)
        if worst <= 1.0:
            return shape, True
        if worst < best_worst_uc:
            best_shape, best_worst_uc = shape, worst
    return best_shape, False


def _weights(geometry: FrameGeometry, assignment: dict[str, WShape]) -> tuple[float, dict[str, float]]:
    by_group = {g: 0.0 for g in assignment}
    for m in geometry.members:
        by_group[m.group] += assignment[m.group].weight_plf * (m.length_in / FT)
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
            "weight_lb": by_group[group],
            "max_uc": worst["governing_uc"],
            "governing_limitstate": worst["governing_limitstate"],
            "governing_member": worst["member"],
            "all_pass": bool(sub["PASS"].all()),
        })

    return OptimizationResult(
        feasible=feasible and bool(table["PASS"].all()),
        converged=converged,
        sections={g: s.name for g, s in assignment.items()},
        total_weight_lb=total,
        weight_by_group_lb=by_group,
        member_table=table,
        group_summary=pd.DataFrame(rows),
        iterations=iterations,
        config=config,
    )


def _optimize_iterative(config: AnyConfig, geometry: FrameGeometry,
                        candidates: dict[str, list[WShape]], params: CheckParams,
                        analyze: AnalyzeFn,
                        max_iterations: int, verbose: bool) -> OptimizationResult:
    assignment = {g: cands[0] for g, cands in candidates.items()}
    history = [tuple(s.name for s in assignment.values())]
    iterations: list[dict] = []
    converged = False
    feasible = True
    demands = analyze(geometry, assignment, config)

    for it in range(1, max_iterations + 1):
        grouped = _group_demands(demands, list(candidates))
        new_assignment: dict[str, WShape] = {}
        feasible = True
        for group, cands in candidates.items():
            shape, ok = _screen_group(cands, grouped[group], params)
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
        history.append(key)
        assignment = new_assignment
        demands = analyze(geometry, assignment, config)

    return _build_result(config, geometry, assignment, demands, params,
                         iterations, converged, feasible)


def _optimize_exhaustive(config: AnyConfig, geometry: FrameGeometry,
                         candidates: dict[str, list[WShape]], params: CheckParams,
                         analyze: AnalyzeFn, verbose: bool) -> OptimizationResult:
    groups = list(candidates)
    best = None  # (weight, assignment, demands)
    iterations: list[dict] = []

    for combo in itertools.product(*(candidates[g] for g in groups)):
        assignment = dict(zip(groups, combo))
        demands = analyze(geometry, assignment, config)
        table = check_all(demands, assignment, params)
        ok = bool(table["PASS"].all())
        weight, _ = _weights(geometry, assignment)
        iterations.append({
            "assignment": {g: s.name for g, s in assignment.items()},
            "weight_lb": weight,
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
             max_iterations: int = 10, verbose: bool = False) -> OptimizationResult:
    """Find the lightest per-group W-shape assignment that passes every
    AISC 360 LRFD check (and serviceability, if enabled). Members of a design
    group share one section; the conventional grid frame has two groups
    ('column', 'beam'), the clear-span building three ('column', 'girder',
    'purlin')."""
    geometry, params, analyze = _prepare(config)
    candidates = _candidates_by_group(config)
    _validate_groups(geometry, candidates, params)

    if method == "iterative":
        return _optimize_iterative(config, geometry, candidates, params,
                                   analyze, max_iterations, verbose)
    if method == "exhaustive":
        return _optimize_exhaustive(config, geometry, candidates, params,
                                    analyze, verbose)
    raise ValueError("method must be 'iterative' or 'exhaustive'.")


def optimize_layout(config: ClearSpanConfig, method: str = "iterative",
                    max_iterations: int = 10,
                    verbose: bool = False) -> OptimizationResult:
    """Clear-span layout search: determine the building layout (n_frames,
    purlin_spacing_ft, end_wall_columns) from the footprint by optimizing
    every realistic layout and keeping the lightest feasible design.

    The footprint (span_ft, length_ft, eave_height_ft) is the fixed input.
    Layout fields left to auto-derive on the config are the search variables,
    ranging over the practice bands in clear_span.py; any field set
    explicitly is honored as-is. The winning result's .config carries the
    chosen layout and .layout_search records every layout tried. If no
    layout yields a feasible design, the attempt that came closest (smallest
    worst unity check) is returned with feasible=False."""
    if not isinstance(config, ClearSpanConfig):
        raise TypeError(
            "optimize_layout() searches clear-span building layouts; a "
            "conventional grid frame (FrameConfig) has its layout given "
            "explicitly — use optimize() for it."
        )

    best = best_key = None
    fallback = fallback_uc = None
    search: list[dict] = []
    for n_frames, spacing, gables in candidate_layouts(config):
        variant = replace(config, n_frames=n_frames,
                          purlin_spacing_ft=spacing, end_wall_columns=gables)
        result = optimize(variant, method=method,
                          max_iterations=max_iterations, verbose=False)
        worst_uc = float(result.member_table["governing_uc"].max())
        search.append({
            "n_frames": n_frames,
            "frame_spacing_ft": variant.frame_spacing_ft,
            "purlin_spacing_ft": variant.purlin_spacing_actual_ft,
            "end_wall_columns": gables,
            "feasible": result.feasible,
            "total_weight_lb": result.total_weight_lb,
            "worst_uc": worst_uc,
        })
        if verbose:
            outcome = (f"{result.total_weight_lb:,.0f} lb" if result.feasible
                       else f"infeasible (worst UC {worst_uc:.2f})")
            print(f"[layout] {n_frames} frames @ {variant.frame_spacing_ft:.1f} ft, "
                  f"purlins @ {variant.purlin_spacing_actual_ft:.2f} ft, "
                  f"{gables} gable col(s)/end -> {outcome}")

        if result.feasible:
            # lightest wins; weights within a pound are a practical tie,
            # broken toward fewer members (less fabrication and erection)
            key = (round(result.total_weight_lb), len(result.member_table))
            if best is None or key < best_key:
                best, best_key = result, key
        elif fallback is None or worst_uc < fallback_uc:
            fallback, fallback_uc = result, worst_uc

    chosen = best if best is not None else fallback
    chosen.layout_search = search
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
