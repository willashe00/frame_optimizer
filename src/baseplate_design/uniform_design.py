"""One baseplate detail for every column of an optimized frame.

Industrial buildings do not detail a different baseplate per column: one plate
is drawn, fabricated in a single batch, and set at every base. So the design
question is not "what does each column need" but "which column governs, and
what does its plate cost the rest of them" -- answered here by designing every
column, enveloping the result, and then proving the single enveloped plate
against every column individually.

The envelope is a ratcheting fixed point rather than a one-shot max, because
the dimensions are coupled: growing B and N to satisfy the heaviest column
lengthens the plate cantilevers m and n, which raises the thickness the OTHER
columns need. Iterating until nothing grows removes that trap.

Which column governs is not a single answer, and this is the reason the
envelope exists at all:

* bearing and plate flexure are governed by the column with the LARGEST axial
  load -- more load, more bearing area, thicker plate;
* anchor rod shear is governed by the column with the SMALLEST axial load --
  the shear-friction credit mu*P is what the rods do not have to carry, so
  the lightest column leaves its rods with the most work.

Input is `frame_optimizer.baseplate_inputs()` -- either an OptimizationResult
to derive it from, or the already-built dict (or a loaded baseplate_inputs.json,
so a design can be re-run offline without touching the FE model).
"""
from __future__ import annotations

from dataclasses import dataclass

from frame_optimizer.config import IN_TO_MM, KIP_TO_KN, LB_TO_KG, MM_TO_IN

from .baseplate_design import (BaseplateCheck, ColumnDemand, PlateGeometry,
                               check_plate, design_plate)
from .config import KN_TO_KIP, BaseplateConfig

# The envelope closes in 2-3 passes (rod size settles on the first pass, plan
# size and thickness on the next); 10 is a loud backstop, not a working limit.
_MAX_ENVELOPE_PASSES = 10

# Reactions below this are numerical zero, not uplift.
_ZERO_TOL_KN = 1e-6

LIMIT_STATES = ("bearing", "plate flexure", "anchor rod shear")


@dataclass(frozen=True)
class UniformBaseplateDesign:
    """One plate, proven against every column base in the building."""
    plate: PlateGeometry
    config: BaseplateConfig
    demands: list[ColumnDemand]
    checks: list[BaseplateCheck]     # same order as `demands`
    envelope_passes: int             # passes the envelope fixed point took

    @property
    def feasible(self) -> bool:
        return all(c.passes for c in self.checks)

    @property
    def n_columns(self) -> int:
        return len(self.demands)

    @property
    def has_design_shear(self) -> bool:
        """False when no column carries any base shear, i.e. the anchor rods
        are at their detailing minimum and their check is not exercised."""
        return any(d.Vu > 0.0 for d in self.demands)

    def check_for(self, column_id: str) -> BaseplateCheck:
        for check in self.checks:
            if check.column_id == column_id:
                return check
        raise KeyError(f"No column {column_id!r} in this design.")

    def demand_for(self, column_id: str) -> ColumnDemand:
        for demand in self.demands:
            if demand.column_id == column_id:
                return demand
        raise KeyError(f"No column {column_id!r} in this design.")

    @property
    def max_dcr(self) -> dict[str, float]:
        """Envelope DCR per limit state over every column."""
        return {
            "bearing": max(c.bearing_dcr for c in self.checks),
            "plate flexure": max(c.flexure_dcr for c in self.checks),
            "anchor rod shear": max(c.shear_dcr for c in self.checks),
        }

    @property
    def governing_column(self) -> dict[str, str]:
        """The column that drives each limit state (highest DCR)."""
        keys = {"bearing": lambda c: c.bearing_dcr,
                "plate flexure": lambda c: c.flexure_dcr,
                "anchor rod shear": lambda c: c.shear_dcr}
        return {state: max(self.checks, key=key).column_id
                for state, key in keys.items()}

    @property
    def overall_governing_column(self) -> str:
        return max(self.checks, key=lambda c: c.max_dcr).column_id

    @property
    def plate_mass_kg(self) -> float:
        """Mass of one plate, kg (1 kip of steel weight = 1000 lb)."""
        return self.plate.weight_kip * 1000.0 * LB_TO_KG

    @property
    def total_plate_mass_kg(self) -> float:
        """Mass of the whole batch: one plate at every column base, kg."""
        return self.plate_mass_kg * self.n_columns

    def summary(self) -> str:
        plate, cfg = self.plate, self.config
        gov = self.governing_column
        dcr = self.max_dcr
        lines = ["=" * 62,
                 "baseplate_design - uniform column baseplate result",
                 "=" * 62]
        status = ("ADEQUATE" if self.feasible else
                  "INADEQUATE (no plate satisfies every column; see below)")
        lines.append(f"Status: {status}, envelope closed in "
                     f"{self.envelope_passes} pass(es)")
        lines.append(f"Applied to all {self.n_columns} column base(s), "
                     f"{self.demands[0].section_name or 'W-shape'} columns, "
                     "pinned")
        lines.extend(cfg.describe())
        lines.append("-" * 62)
        lines.append("Baseplate (one detail, every column):")
        lines.append(f"  Plate       {plate.B * IN_TO_MM:,.0f} x "
                     f"{plate.N * IN_TO_MM:,.0f} x {plate.tp * IN_TO_MM:,.1f} mm"
                     f"   ({plate.B:g} x {plate.N:g} x {plate.tp:g} in)")
        lines.append(f"  Anchor rods {plate.n_rods} - "
                     f"{plate.d_rod * IN_TO_MM:,.1f} mm dia."
                     f"   ({plate.d_rod:g} in), edge distance "
                     f"{plate.edge_distance * IN_TO_MM:,.0f} mm")
        lines.append(f"  Plate mass  {self.plate_mass_kg:,.1f} kg each, "
                     f"{self.total_plate_mass_kg:,.0f} kg total")
        lines.append("-" * 62)
        lines.append("Governing check per limit state:")
        for state in LIMIT_STATES:
            column = gov[state]
            demand = self.demand_for(column)
            driver = (f"Vu = {demand.Vu * KIP_TO_KN:,.1f} kN on "
                      f"{demand.P_friction * KIP_TO_KN:,.0f} kN of friction"
                      if state == "anchor rod shear"
                      else f"Pu = {demand.Pu * KIP_TO_KN:,.0f} kN")
            flag = "" if dcr[state] <= 1.0 else "   <-- FAILS"
            lines.append(f"  {state:<17} DCR = {dcr[state]:.3f}  "
                         f"[{column}, {driver}]{flag}")
        lines.append("-" * 62)
        span = self._demand_span()
        lines.append(f"Column axial range: {span[0]:,.0f} - {span[1]:,.0f} kN "
                     f"(the plate is sized by {gov['plate flexure']}, "
                     f"the rods by {gov['anchor rod shear']})")
        if not self.has_design_shear:
            lines.append(
                "NOTE: no design base shear stated (the gravity model has "
                "none). Rods are at the detailing minimum; set "
                "BaseplateConfig.design_base_shear_kN from the lateral "
                "system design to size them.")
        if not self.feasible:
            failing = [c.column_id for c in self.checks if not c.passes]
            lines.append(f"WARNING: {len(failing)} column(s) FAIL: "
                         f"{', '.join(failing[:6])}"
                         f"{' ...' if len(failing) > 6 else ''}")
        lines.append("=" * 62)
        return "\n".join(lines)

    def _demand_span(self) -> tuple[float, float]:
        pus = [d.Pu * KIP_TO_KN for d in self.demands]
        return min(pus), max(pus)


def _envelope(plates: list[PlateGeometry]) -> PlateGeometry:
    """Largest of every dimension. Edge distance is a function of the rod
    diameter, so its max belongs to the max rod and stays consistent."""
    return PlateGeometry(
        B=max(p.B for p in plates),
        N=max(p.N for p in plates),
        tp=max(p.tp for p in plates),
        d_rod=max(p.d_rod for p in plates),
        n_rods=plates[0].n_rods,
        edge_distance=max(p.edge_distance for p in plates),
    )


def _positive(value: float, column_id: str, what: str) -> float:
    """Clamp numerical-zero reactions to 0; reject real uplift."""
    if value < -_ZERO_TOL_KN:
        raise ValueError(
            f"{column_id}: {what} is {value:.3f} kN (net uplift). This module "
            "designs compression + shear on a pinned base only; rod tension "
            "and concrete breakout (ACI 318 Ch. 17) are out of scope."
        )
    return max(0.0, value)


def demands_from_inputs(inputs: dict,
                        config: BaseplateConfig) -> list[ColumnDemand]:
    """Turn a `frame_optimizer.baseplate_inputs()` dict into kip/inch demands.

    This is the whole seam between the two modules: the frame optimizer's
    documented JSON contract in, design demands out. Nothing here needs the
    FE model, so a saved baseplate_inputs.json designs just as well.
    """
    columns = inputs.get("columns")
    if not columns:
        raise ValueError(
            "baseplate_inputs has no 'columns'; nothing to design. Check that "
            "the optimization produced a frame with column bases."
        )

    demands = []
    for col in columns:
        cid = col["member_id"]
        axial = col["axial_compression_kN"]
        by_combo = axial["by_combo"]
        if "D" not in by_combo:
            raise ValueError(
                f"{cid}: baseplate_inputs is missing the service dead load "
                "'D', which the shear-friction credit needs."
            )
        Pu_kN = _positive(axial["Pu_governing_lrfd"], cid, "governing compression")
        dead_kN = _positive(by_combo["D"], cid, "service dead reaction")

        # The gravity model reports no real base shear; the design value comes
        # from the config (lateral system), enveloped with whatever the model
        # did report so a future lateral-capable model flows through unchanged.
        model_shear = col.get("base_shear_kN", {}).get("Vu_governing_lrfd", 0.0)
        Vu_kN = max(config.design_base_shear_kN, abs(model_shear))

        section = col["section"]
        demands.append(ColumnDemand(
            column_id=cid,
            d=section["depth_d_mm"] * MM_TO_IN,
            bf=section["flange_width_bf_mm"] * MM_TO_IN,
            Pu=Pu_kN * KN_TO_KIP,
            Vu=Vu_kN * KN_TO_KIP,
            # friction rides on the MINIMUM coincident compression (0.9D),
            # never on Pu -- see baseplate_design.py, methodology item 4
            P_friction=config.friction_axial_factor * dead_kN * KN_TO_KIP,
            section_name=section["name"],
            base_node=col.get("base_node", ""),
            location_in=tuple(col["centerline_location"][k] * MM_TO_IN
                              for k in ("x_mm", "y_mm", "z_mm")),
        ))
    return demands


def design_uniform_baseplate(source, config: BaseplateConfig | None = None
                             ) -> UniformBaseplateDesign:
    """Design ONE baseplate that works for every column base in the frame.

    `source` is an OptimizationResult (its baseplate inputs are derived from
    the final assignment) or an already-built baseplate_inputs dict.
    """
    config = config or BaseplateConfig()
    inputs = source if isinstance(source, dict) else _inputs_from_result(source)
    demands = demands_from_inputs(inputs, config)

    plate = None
    for taken in range(1, _MAX_ENVELOPE_PASSES + 1):
        candidate = _envelope([design_plate(d, config, floor=plate) for d in demands])
        if candidate == plate:
            break
        plate = candidate
    else:
        raise RuntimeError(
            f"Uniform baseplate envelope did not settle in "
            f"{_MAX_ENVELOPE_PASSES} passes (last: {plate}). This should not "
            "happen for gravity demands; please report the config."
        )

    return UniformBaseplateDesign(
        plate=plate, config=config, demands=demands,
        checks=[check_plate(plate, d, config) for d in demands],
        envelope_passes=taken,
    )


def _inputs_from_result(result) -> dict:
    """Late import: only the result-driven path needs the FE machinery."""
    from frame_optimizer import baseplate_inputs
    return baseplate_inputs(result)
