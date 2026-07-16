"""Structured result of an optimization run + human-readable report."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class OptimizationResult:
    feasible: bool
    converged: bool
    sections: dict[str, str]          # group -> shape name
    total_weight_lb: float
    weight_by_group_lb: dict[str, float]
    member_table: pd.DataFrame        # one row per member, all unity checks
    group_summary: pd.DataFrame       # one row per group
    iterations: list[dict] = field(default_factory=list)
    config: object | None = None      # FrameConfig | ClearSpanConfig (any config
                                      # with candidates_by_group and describe())
    layout_search: list[dict] = field(default_factory=list)
                                      # optimize_layout() only: one entry per
                                      # candidate layout tried (this result is
                                      # the winner)

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

        lines.append("-" * 62)
        lines.append("Selected sections:")
        for _, row in self.group_summary.iterrows():
            lines.append(
                f"  {row['group']:<7} {row['profile']:<9} "
                f"({int(row['n_members'])} members, {row['weight_lb']:,.0f} lb)  "
                f"max UC = {row['max_uc']:.3f} [{row['governing_limitstate']}"
                f" @ {row['governing_member']}]"
            )
        lines.append("-" * 62)
        lines.append(
            f"Total steel weight: {self.total_weight_lb:,.0f} lb "
            f"({self.total_weight_lb / 2000.0:,.2f} tons)"
        )
        n_fail = int((~self.member_table["PASS"]).sum())
        if n_fail:
            lines.append(f"WARNING: {n_fail} member(s) FAIL their checks - see member_table.")
        lines.append("=" * 62)
        return "\n".join(lines)
