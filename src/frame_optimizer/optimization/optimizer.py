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
import multiprocessing
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace

import pandas as pd

from typing import Callable, Union

from ..analysis import MemberDemand, analyze_frame
from ..clear_span import (AUTO, MIN_FRAME_SPACING_FT, TRUSS, WIDE_FLANGE,
                          ClearSpanConfig, build_clear_span_geometry,
                          candidate_layouts, candidate_truss_depths,
                          clear_span_check_params)
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


# ---------------------------------------------------------------------------
# Representative-strip analysis (clear-span buildings).
#
# The purlins are pin-ended, statically determinate members, so their support
# reactions do NOT depend on the stiffness of the frames they land on: every
# transverse frame is EXACTLY decoupled from its neighbors (chord-relative
# purlin sag is likewise a property of the purlin's own load only — the same
# theorem the deflection extraction already relies on). A 3-frame strip with
# the true frame spacing therefore reproduces, member for member:
#   * every interior frame  -> the strip's middle frame (full tributary),
#   * both end frames       -> the strip's end frames (half tributary,
#                              gable columns / end girders included),
#   * every purlin bay      -> a strip bay (all bays carry identical loads).
# Analyzing the strip and broadcasting its demands onto the full member list
# makes the FEA cost independent of building length — the dominant speedup
# for long buildings. It is exact up to one deliberately-retained coupling:
# purlin TORSION (kept attached to prevent spin mechanisms) ties neighboring
# frames' girder rotations together at parts-per-million magnitude (~1e-5
# relative on member demands, tested) — orders of magnitude below any design
# significance.
# ---------------------------------------------------------------------------
_STRIP_FRAMES = 3
_BARE_FRAME_NAME = re.compile(r"^([A-Za-z]+)(\d+)$")   # G7, TC3, BC0, ...


def _strip_frame(j: int, nf: int) -> int:
    if j == 0:
        return 0
    return 2 if j == nf - 1 else 1


def _strip_member_name(name: str, nf: int) -> str:
    """Full-building member name -> its representative in the 3-frame strip."""
    if ".b" in name:                       # purlin P{line}.b{bay}
        head, bay = name.rsplit(".b", 1)
        return f"{head}.b{0 if int(bay) == 0 else 1}"
    if "." in name:                        # C0.{j}, CG1.{j}, TV4.{j}, TD2.{j}
        head, j = name.rsplit(".", 1)
        return f"{head}.{_strip_frame(int(j), nf)}"
    match = _BARE_FRAME_NAME.match(name)   # G{j}, TC{j}, BC{j}
    return f"{match.group(1)}{_strip_frame(int(match.group(2)), nf)}"


def _clear_span_analyze_fn(config: ClearSpanConfig) -> AnalyzeFn:
    """analyze_frame, routed through the exact 3-frame strip when the
    building has more frames than the strip."""
    if config.n_frames <= _STRIP_FRAMES:
        return analyze_frame
    strip_geometry = build_clear_span_geometry(config, n_frames=_STRIP_FRAMES)

    def strip_analyze(geometry: FrameGeometry, assignment: dict[str, WShape],
                      cfg: AnyConfig) -> list[MemberDemand]:
        strip_demands = analyze_frame(strip_geometry, assignment, cfg)
        by_name = {d.name: d for d in strip_demands}
        return [replace(by_name[_strip_member_name(m.name, cfg.n_frames)],
                        name=m.name)
                for m in geometry.members]

    return strip_analyze


def _prepare(config: AnyConfig) -> tuple[FrameGeometry, CheckParams, AnalyzeFn]:
    """Geometry, per-group design rules, and analysis function for a config.
    Both building types solve in one Pynite model (clear-span purlins are
    explicit members); clear-span buildings longer than 3 frames analyze an
    exact representative strip instead of the whole building."""
    if isinstance(config, ClearSpanConfig):
        return (geometry_for(config), clear_span_check_params(config),
                _clear_span_analyze_fn(config))
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
    monotone = False   # set on first oscillation: sections then only ratchet up
    demands = analyze(geometry, assignment, config)

    def merged_up(new: dict[str, WShape]) -> dict[str, WShape]:
        return {g: max(assignment[g], new[g], key=lambda s: s.weight_plf)
                for g in assignment}

    for it in range(1, max_iterations + 1):
        grouped = _group_demands(demands, list(candidates))
        new_assignment: dict[str, WShape] = {}
        feasible = True
        for group, cands in candidates.items():
            shape, ok = _screen_group(cands, grouped[group], params)
            new_assignment[group] = shape
            feasible &= ok
        if monotone:
            new_assignment = merged_up(new_assignment)

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
        if key in history and not monotone:
            # Oscillation between assignments (self-weight / frame-stiffness
            # feedback — rigid bents redistribute demands as sections change).
            # From here on take the heavier shape per group, which can only be
            # conservative; the finite candidate lists then force termination
            # within a few strictly-heavier steps instead of bouncing to the
            # iteration cap.
            monotone = True
            new_assignment = merged_up(new_assignment)
            if all(new_assignment[g] is assignment[g] for g in assignment):
                converged = True   # merge landed on the current assignment
                break
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
    'purlin') — or five with girder_system='truss' ('column',
    'truss_top_chord', 'truss_bot_chord', 'truss_web', 'purlin')."""
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


def _layout_variants(config: ClearSpanConfig) -> list[ClearSpanConfig]:
    """One concrete config per candidate layout — times the depth options of
    the span/10..span/15 practice band for trusses whose depth is auto."""
    variants = []
    reopen_panel = "truss_panel_ft" in config.auto_layout_fields
    for n_frames, spacing, gables in candidate_layouts(config):
        base = dict(n_frames=n_frames, purlin_spacing_ft=spacing,
                    end_wall_columns=gables)
        if config.girder_system == TRUSS:
            for depth in candidate_truss_depths(config):
                kwargs = dict(base, truss_depth_ft=depth)
                if reopen_panel:   # panel count re-derives for each depth
                    kwargs["truss_panel_ft"] = None
                variants.append(replace(config, **kwargs))
        else:
            variants.append(replace(config, **base))
    return variants


def wide_flange_infeasible_reason(config: ClearSpanConfig) -> str | None:
    """Certain-infeasibility screen for the W-girder system, or None.

    Uses bounds strictly OPTIMISTIC for the girder — the least tributary
    width any layout in the practice band could assign, demand without
    girder/purlin self-weight, the full plastic moment for capacity, live
    deflection only (camber can never offset it) — so a non-None reason
    proves that no candidate W-shape can carry the clear span under ANY
    layout, and the W search may be skipped without losing a feasible
    design. None means the W system might work: search it.
    """
    shapes = get_shapes(config.girder_candidates)
    L = config.span_ft * FT
    if "n_frames" in config.auto_layout_fields:
        s_min_ft = MIN_FRAME_SPACING_FT
    else:
        s_min_ft = config.frame_spacing_ft

    if config.check_deflection and config.live_psf > 0:
        ratio = (config.defl_live_ratio if config.girder_defl_live_ratio is None
                 else config.girder_defl_live_ratio)
        w_live = config.live_psf * s_min_ft / 12000.0            # kip/in
        # 5wL^4/384EI <= L/ratio  ->  Ix >= 5wL^3*ratio/384E
        Ix_req = 5.0 * w_live * L**3 * ratio / (384.0 * config.E_ksi)
        best = max(shapes, key=lambda s: s.Ix)
        if Ix_req > best.Ix:
            return (f"no girder candidate can meet the L/{ratio:.0f} live-load "
                    f"deflection limit over {config.span_ft:.0f} ft (needs "
                    f"Ix >= ~{Ix_req:,.0f} in^4; best candidate {best.name} "
                    f"offers {best.Ix:,.0f})")

    w_u = ((1.2 * config.superimposed_dead_psf + 1.6 * config.live_psf)
           * s_min_ft / 12000.0)                                 # kip/in
    Mu = w_u * L**2 / 8.0
    best = max(shapes, key=lambda s: s.Zx)
    phi_Mp = 0.9 * config.Fy_ksi * best.Zx
    if Mu > phi_Mp:
        return (f"no girder candidate can carry the factored moment over "
                f"{config.span_ft:.0f} ft (needs phi*Mn >= ~{Mu / FT:,.0f} "
                f"kip-ft; best candidate {best.name} offers at most "
                f"{phi_Mp / FT:,.0f})")
    return None


def _pin_system(config: ClearSpanConfig, system: str) -> ClearSpanConfig:
    """Copy of an 'auto'-system config pinned to one girder system, with the
    auto-derived layout fields reopened so the layout search re-ranges them.
    The truss variant drops the end-girder/gable option (not available with
    trusses); the wide-flange variant drops the truss-only fields."""
    kwargs: dict = {name: None for name in config.auto_layout_fields}
    kwargs["girder_system"] = system
    if system == WIDE_FLANGE:
        kwargs.update(truss_chord_candidates=None, truss_web_candidates=None,
                      truss_depth_ft=None, truss_panel_ft=None,
                      truss_bottom_brace_ft=None)
    else:
        kwargs.update(end_girder_candidates=None, end_wall_columns=None)
    return replace(config, **kwargs)


# parallel layout evaluation: below this many variants the process-pool
# startup (each worker imports Pynite/pandas) outweighs the solve savings
_PARALLEL_MIN_VARIANTS = 4


def _optimize_variant(args) -> "OptimizationResult":
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    variant, method, max_iterations = args
    return optimize(variant, method=method, max_iterations=max_iterations,
                    verbose=False)


def _optimize_system_layouts(config: ClearSpanConfig, method: str,
                             max_iterations: int, verbose: bool,
                             parallel: bool | None = None):
    """Layout search for ONE concrete girder system. Returns
    (best_feasible | None, closest_infeasible | None, search_entries).

    Layout variants are independent, so they evaluate on a process pool when
    there are enough of them to amortize the worker startup (parallel=None:
    auto; True/False force it). Results are collected in variant order, so
    the search record and the winner are identical either way.

    NOTE (Windows/spawn): a script that calls optimize_layout() at module
    top level must use the standard ``if __name__ == "__main__":`` guard
    (gravity_design.py does) — Python re-imports the main script in each
    worker. Calls from within a worker process fall back to serial
    automatically, and a broken pool degrades to serial with a warning."""
    variants = _layout_variants(config)
    in_worker = multiprocessing.current_process().name != "MainProcess"
    use_parallel = (len(variants) >= _PARALLEL_MIN_VARIANTS
                    if parallel is None else parallel) and not in_worker
    jobs = [(v, method, max_iterations) for v in variants]
    results = None
    if use_parallel and len(variants) > 1:
        workers = min(len(variants), max(1, (os.cpu_count() or 2) - 1))
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_optimize_variant, jobs))
        except BrokenProcessPool:
            warnings.warn(
                "parallel layout search failed (broken process pool) — "
                "falling back to serial. If this is a script, make sure the "
                "optimize_layout() call sits under "
                "'if __name__ == \"__main__\":'.")
            results = None
    if results is None:
        results = [_optimize_variant(job) for job in jobs]

    best = best_key = None
    fallback = fallback_uc = None
    search: list[dict] = []
    for variant, result in zip(variants, results):
        worst_uc = float(result.member_table["governing_uc"].max())
        entry = {
            "girder_system": variant.girder_system,
            "n_frames": variant.n_frames,
            "frame_spacing_ft": variant.frame_spacing_ft,
            "purlin_spacing_ft": variant.purlin_spacing_actual_ft,
            "end_wall_columns": variant.end_wall_columns,
            "feasible": result.feasible,
            "total_weight_lb": result.total_weight_lb,
            "worst_uc": worst_uc,
        }
        truss_note = ""
        if variant.girder_system == TRUSS:
            entry["truss_depth_ft"] = variant.truss_depth_ft
            entry["n_truss_panels"] = variant.n_truss_panels
            truss_note = f", truss depth {variant.truss_depth_ft:.1f} ft"
        search.append(entry)
        if verbose:
            outcome = (f"{result.total_weight_lb:,.0f} lb" if result.feasible
                       else f"infeasible (worst UC {worst_uc:.2f})")
            print(f"[layout] ({variant.girder_system}) "
                  f"{variant.n_frames} frames @ {variant.frame_spacing_ft:.1f} ft, "
                  f"purlins @ {variant.purlin_spacing_actual_ft:.2f} ft, "
                  f"{variant.end_wall_columns} gable col(s)/end{truss_note} "
                  f"-> {outcome}")

        if result.feasible:
            # lightest wins; weights within a pound are a practical tie,
            # broken toward fewer members (less fabrication and erection)
            key = (round(result.total_weight_lb), len(result.member_table))
            if best is None or key < best_key:
                best, best_key = result, key
        elif fallback is None or worst_uc < fallback_uc:
            fallback, fallback_uc = result, worst_uc
    return best, fallback, search


def _worst_uc(result: OptimizationResult) -> float:
    return float(result.member_table["governing_uc"].max())


def _optimize_auto_system(config: ClearSpanConfig, method: str,
                          max_iterations: int, verbose: bool,
                          parallel: bool | None = None) -> OptimizationResult:
    """girder_system='auto' with truss candidates: search W-girder layouts
    first and use them whenever ANY W layout is feasible; escalate to truss
    girders only when none is (or when the screen proves none can be)."""
    system_search: list[dict] = []
    search: list[dict] = []
    wf_fallback = None

    wf_config = _pin_system(config, WIDE_FLANGE)
    reason = wide_flange_infeasible_reason(wf_config)
    if reason is None:
        best, wf_fallback, wf_search = _optimize_system_layouts(
            wf_config, method, max_iterations, verbose, parallel)
        search += wf_search
        n_feasible = sum(1 for e in wf_search if e["feasible"])
        system_search.append({
            "girder_system": WIDE_FLANGE, "evaluated": True,
            "layouts": len(wf_search), "feasible_layouts": n_feasible,
            "chosen": best is not None,
            "note": None if best is not None else "no layout feasible",
        })
        if best is not None:
            system_search.append({
                "girder_system": TRUSS, "evaluated": False,
                "layouts": 0, "feasible_layouts": 0, "chosen": False,
                "note": "not needed — wide-flange girders are feasible",
            })
            best.layout_search = search
            best.system_search = system_search
            return best
    else:
        if verbose:
            print(f"[system] wide_flange skipped: {reason}")
        system_search.append({
            "girder_system": WIDE_FLANGE, "evaluated": False,
            "layouts": 0, "feasible_layouts": 0, "chosen": False,
            "note": reason,
        })

    best, tr_fallback, tr_search = _optimize_system_layouts(
        _pin_system(config, TRUSS), method, max_iterations, verbose, parallel)
    search += tr_search
    n_feasible = sum(1 for e in tr_search if e["feasible"])
    system_search.append({
        "girder_system": TRUSS, "evaluated": True,
        "layouts": len(tr_search), "feasible_layouts": n_feasible,
        "chosen": best is not None,
        "note": None if best is not None else "no layout feasible",
    })
    if best is not None:
        chosen = best
    else:
        # neither system works: report the attempt that came closest
        attempts = [r for r in (wf_fallback, tr_fallback) if r is not None]
        chosen = min(attempts, key=_worst_uc)
    chosen.layout_search = search
    chosen.system_search = system_search
    return chosen


def optimize_layout(config: ClearSpanConfig, method: str = "iterative",
                    max_iterations: int = 10, verbose: bool = False,
                    parallel: bool | None = None) -> OptimizationResult:
    """Clear-span layout search: determine the building layout (n_frames,
    purlin_spacing_ft, end_wall_columns — plus truss_depth_ft for trusses)
    from the footprint by optimizing every realistic layout and keeping the
    lightest feasible design.

    Performance: each candidate layout analyzes an exact 3-frame
    representative strip instead of the full building (pin-ended purlins
    decouple the bents — see _clear_span_analyze_fn), and independent
    layouts evaluate on a process pool (`parallel`: None = auto, on when the
    search has enough variants to amortize worker startup; results are
    deterministic and identical to a serial run).

    Girder-system selection (girder_system='auto' with truss candidates):
    rolled W girders are preferred — the layout search runs on them first and
    the lightest feasible W design wins outright. Truss girders are used ONLY
    when no W layout is feasible; a closed-form screen
    (wide_flange_infeasible_reason) skips the W search entirely when no
    candidate W-shape could carry the clear span under any layout. The
    outcome of the system decision is recorded in .system_search and shown
    in the result summary. A config pinned to one system searches only it.

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

    if config.girder_system == AUTO and config.truss_chord_candidates is not None:
        return _optimize_auto_system(config, method, max_iterations, verbose,
                                     parallel)

    best, fallback, search = _optimize_system_layouts(
        config, method, max_iterations, verbose, parallel)
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
