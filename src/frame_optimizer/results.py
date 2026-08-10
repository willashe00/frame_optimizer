"""Structured result of an optimization run + human-readable report."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class OptimizationResult:
    feasible: bool
    converged: bool
    sections: dict[str, str]          # group -> shape name
    total_weight_kg: float
    weight_by_group_kg: dict[str, float]
    member_table: pd.DataFrame        # one row per member, all unity checks
    group_summary: pd.DataFrame       # one row per group
    iterations: list[dict] = field(default_factory=list)
    config: object | None = None      # FrameConfig | ClearSpanConfig (any config
                                      # with candidates_by_group and describe())
    layout_search: list[dict] = field(default_factory=list)
                                      # optimize_layout() only: one entry per
                                      # candidate layout tried (this result is
                                      # the winner)
    second_order: dict | None = None  # truss winners only: P-Delta
                                      # verification record (method, verified,
                                      # passes, sections_changed, note)

    def summary(self) -> str:
        lines = ["=" * 62, "frame_optimizer - gravity frame optimization result", "=" * 62]

        status = "FEASIBLE" if self.feasible else "INFEASIBLE (no candidate passes; showing best attempt)"
        conv = f"converged in {len(self.iterations)} iteration(s)" if self.converged \
            else f"did NOT converge in {len(self.iterations)} iteration(s)"
        lines.append(f"Status: {status}, {conv}")

        if self.config is not None:
            lines.extend(self.config.describe())
        if self.layout_search:
            n_feasible = sum(1 for r in self.layout_search if r["feasible"])
            lines.append(
                f"Layout: lightest of {n_feasible} feasible / "
                f"{len(self.layout_search)} realistic layout(s) searched "
                "for the footprint")
        if self.second_order is not None:
            so = self.second_order
            if so["verified"]:
                resized = (", sections resized against second-order demands"
                           if so["sections_changed"] else "")
                lines.append(
                    f"2nd order: {so['method']} verification PASSED "
                    f"({so['passes']} pass(es){resized})")
            else:
                note = f" - {so['note']}" if so.get("note") else ""
                lines.append(
                    f"WARNING: {so['method']} second-order verification "
                    f"FAILED{note}")

        lines.append("-" * 62)
        lines.append("Selected sections:")
        for _, row in self.group_summary.iterrows():
            lines.append(
                f"  {row['group']:<7} {row['profile']:<9} "
                f"({int(row['n_members'])} members, {row['weight_kg']:,.0f} kg)  "
                f"max UC = {row['max_uc']:.3f} [{row['governing_limitstate']}"
                f" @ {row['governing_member']}]"
            )
        lines.append("-" * 62)
        lines.append(
            f"Total steel weight: {self.total_weight_kg:,.0f} kg "
            f"({self.total_weight_kg / 1000.0:,.2f} t)"
        )
        n_fail = int((~self.member_table["PASS"]).sum())
        if n_fail:
            lines.append(f"WARNING: {n_fail} member(s) FAIL their checks - see member_table.")
        lines.append("=" * 62)
        return "\n".join(lines)
