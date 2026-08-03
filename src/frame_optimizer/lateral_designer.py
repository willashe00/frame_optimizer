"""Phase 4+5 of the lateral system: size the LFRS members, re-certify every
gravity member under the full ASCE 7-16 combination set, check drift, and
package the certified design for export.

design_lateral() is the entry point. The algorithm mirrors the gravity
optimizer's fixed-point iteration, extended with the lateral physics:

  build the augmented geometry (lateral_system)
  assignment = gravity sections (RATCHET FLOORS: a gravity member may grow
               under combined actions but never shrink) + lightest lateral
               candidates
  repeat:
      lateral loads from the CURRENT total steel weight (seismic W feedback)
      3D DAM P-Delta analysis of the full building over the active combos
          (analysis/lateral_model.py: wall wind, roof uplift, ELF forces)
      closed-form longitudinal actions (braced_bay.py) AUGMENT the 3D
          demands: brace tension, strut shear transfer, braced-line column
          couples, eave-strut collector axial, roof-truss diagonal tension
          and chord axial on the bounding girder/top-chord lines
      per-group screen with check_member — the same single acceptance path
          as the gravity design — plus the closed-form UPLIFT check for
          purlins/eave struts/W girders (0.9D + 1.0W net uplift bent about
          an UNBRACED bottom flange: Lb = full span, Cb = 1) and a brace
          bump loop for longitudinal drift
  until the assignment repeats (the final analysis then certifies exactly
  the reported sections; an oscillating assignment switches to the same
  monotone merge-up ratchet the gravity optimizer uses);
  transverse drift check (seismic Cd*delta_xe/Ie <= 0.025 hsx per 12.8.6 /
  Table 12.12-1 — a code limit; wind ~10-yr delta <= h/100 per the AISC
  Design Guide 3 customary band for metal-clad industrial buildings — a
  serviceability default, tightened via wind_drift_denom when finishes
  demand it) with a bounded stiffness-escalation loop that raises the
  column (then girder/chord) floors when a frame drifts too far.

Design-basis notes (v1, stated in the exports):

* R = 3 / Cd = 3 "steel systems not specifically detailed for seismic
  resistance" both directions, SDC A-C only (enforced upstream); AISC 341
  is intentionally not invoked.
* Wall/roof X pairs are tension-only by design; L/r <= 300 (D1) is
  enforced on the diagonals, KL/r <= 200 on the compression struts.
* Moment-frame girders lose the simple-span Cb credit (their moment
  diagrams reverse at the knees); everything else keeps its gravity rules.
* Connections, baseplates, anchor rods, and foundation ties are exported
  as demands (including net uplift and braced-bay shear), not designed.

Units: interface ft/psf/kip; internals kips and inches.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

from .analysis.frame_model import (MemberDemand, SERVICE_LIVE_COMBO,
                                   SERVICE_TOTAL_COMBO)
from .analysis.lateral_model import (_band_pressure_psf, active_strength_combos,
                                     analyze_lateral)
from .braced_bay import (BracedBayForces, RoofTrussForces, bay_drift_in,
                         bay_forces, design_bay_shears_kip,
                         roof_truss_forces, seismic_bay_shares_kip,
                         wind_bay_shares_kip)
from .clear_span import (END_GIRDER, GIRDER, PURLIN, TRUSS, TRUSS_TOP_CHORD,
                         ClearSpanConfig, clear_span_check_params)
from .config import COLUMN, FT
from .design import CheckParams, GroupRules, check_all, check_member
from .design.aisc_strengths import flexure_major_capacity
from .export import _r, _section_dimensions, _write_json
from .lateral_loads import LateralLoads, compute_lateral_loads, lateral_load_basis, roof_height_ft
from .lateral_system import (BRACE, EAVE_STRUT, ROOF_BRACE, WALL_STRUT,
                             LateralSystem, LateralSystemConfig,
                             build_lateral_system)
from .results import OptimizationResult
from .sections import WShape, get_shapes
from .site import SiteHazards

# default drift criteria (overridable in design_lateral)
SEISMIC_DRIFT_RATIO = 0.025    # Delta_a = 0.025 hsx (Table 12.12-1) — CODE limit
WIND_DRIFT_DENOM = 100.0       # h/100 under ~10-yr wind: AISC Design Guide 3's
                               # customary band for METAL-CLAD industrial
                               # buildings is H/60-H/100 (bare frame, 10-yr).
                               # Serviceability, not code — tighten via
                               # wind_drift_denom (e.g. 400) for buildings
                               # with cranes, masonry, or brittle finishes.
CD_NOT_DETAILED = 3.0          # Table 12.2-1 H.1: Cd = 3 alongside R = 3

UPLIFT_GROUPS_W = (PURLIN, EAVE_STRUT, GIRDER, END_GIRDER)
UPLIFT_GROUPS_TRUSS = (PURLIN, EAVE_STRUT)   # chord reversal is captured by
                                             # the 3D signed-axial envelope


# ---------------------------------------------------------------------------
# Group design rules for the lateral phase
# ---------------------------------------------------------------------------

def lateral_check_params(config: ClearSpanConfig,
                         system: LateralSystem) -> CheckParams:
    """Gravity rules extended with the LFRS groups. Moment-frame girders
    lose Cb_simple_span (knee moments break the parabolic simple-span
    diagram); eave struts inherit the purlin bracing/serviceability rules
    (same deck attachment, same roof role)."""
    params = clear_span_check_params(config)
    rules = dict(params.group_rules)
    if config.girder_system != TRUSS:
        for g in (GIRDER, END_GIRDER):
            if g in rules:
                rules[g] = replace(rules[g], Cb_simple_span=False)
    # The deck edge-fastens to the eave strut continuously, so whatever LTB
    # bracing the purlins credit (purlin_Lb_ft) equally braces the strut's
    # WEAK AXIS in compression — without this, the collector axial is judged
    # against weak-axis buckling over the full bay and no rolled shape of
    # purlin-like weight can work. Strong-axis KL stays the full bay span.
    rules[EAVE_STRUT] = replace(rules[PURLIN], KLy_in=rules[PURLIN].Lb_in)
    rules[BRACE] = GroupRules(check_deflection=False, check_slenderness=True,
                              slenderness_limit=300.0)
    rules[ROOF_BRACE] = GroupRules(check_deflection=False,
                                   check_slenderness=True,
                                   slenderness_limit=300.0)
    if WALL_STRUT in system.counts:
        rules[WALL_STRUT] = GroupRules(check_deflection=False,
                                       check_slenderness=True)
    return CheckParams(Fy=params.Fy, Fu=params.Fu, E=params.E,
                       group_rules=rules)


# ---------------------------------------------------------------------------
# Closed-form augmentation of the 3D demands
# ---------------------------------------------------------------------------

def _add_axial(d: MemberDemand, compression_kip: float = 0.0,
               tension_kip: float = 0.0) -> MemberDemand:
    """Fold closed-form axial actions into a demand's signed envelope."""
    pu_min = d.Pu_compression - compression_kip
    pu_max = d.Pu_tension + tension_kip
    pu = pu_min if -pu_min >= pu_max else pu_max
    return replace(d, Pu=pu, Pu_min=pu_min, Pu_max=pu_max)


def _augment_demands(demands: list[MemberDemand], system: LateralSystem,
                     config: ClearSpanConfig,
                     forces_by_bay: dict[int, BracedBayForces],
                     roof_tr: RoofTrussForces,
                     collector_kip: float) -> list[MemberDemand]:
    trussed = config.girder_system == TRUSS
    n_bays = config.n_frames - 1
    chord_frames = {0, 1, n_bays - 1, n_bays}
    chord_prefix = "TC" if trussed else "G"
    couple_by_line: dict[int, float] = {}
    for b, f in forces_by_bay.items():
        for line in (b, b + 1):
            couple_by_line[line] = max(couple_by_line.get(line, 0.0),
                                       f.column_couple_kip)

    out = []
    for d in demands:
        if d.group == BRACE:                        # DB{side}.{bay}.t{t}.{tag}
            bay = int(d.name.split(".")[1])
            out.append(_add_axial(d, tension_kip=forces_by_bay[bay].diag_tension_kip))
        elif d.group == WALL_STRUT:                 # WS{side}.{bay}.t{t}
            v = forces_by_bay[int(d.name.split(".")[1])].strut_axial_kip
            out.append(_add_axial(d, compression_kip=v, tension_kip=v))
        elif d.group == ROOF_BRACE:                 # tension-only diagonals
            out.append(_add_axial(d, tension_kip=roof_tr.diag_tension_kip))
        elif d.group == EAVE_STRUT:                 # collector drag, +/-
            out.append(_add_axial(d, compression_kip=collector_kip,
                                  tension_kip=collector_kip))
        elif d.group == COLUMN and not d.name.startswith("CG"):
            line = int(d.name.split(".")[1])
            dp = couple_by_line.get(line, 0.0)
            out.append(_add_axial(d, compression_kip=dp, tension_kip=dp)
                       if dp else d)
        elif (d.name.startswith(chord_prefix)
              and d.name[len(chord_prefix):].isdigit()
              and int(d.name[len(chord_prefix):]) in chord_frames):
            out.append(_add_axial(d, compression_kip=roof_tr.chord_axial_kip,
                                  tension_kip=roof_tr.chord_axial_kip))
        else:
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Closed-form wind-uplift reversal check (unbraced bottom flange)
# ---------------------------------------------------------------------------

def uplift_uc(shape: WShape, span_in: float, trib_in: float,
              extra_dead_plf: float, sdl_psf: float, p_up_psf: float,
              Fy: float, E: float) -> float:
    """0.9D + 1.0W net-uplift unity check as a simple span bent about an
    unbraced bottom flange (Lb = span, Cb = 1 — conservative for the
    moment-connected girders, exact for the pin-ended purlins)."""
    trib_ft = trib_in / 12.0
    L_ft = span_in / 12.0
    w_dead_plf = shape.weight_plf + extra_dead_plf + sdl_psf * trib_ft
    w_net_plf = p_up_psf * trib_ft - 0.9 * w_dead_plf
    if w_net_plf <= 0.0:
        return 0.0
    Mu = w_net_plf / 1000.0 * L_ft**2 / 8.0 * 12.0   # kip-in
    phi_Mn = flexure_major_capacity(shape, Fy, E, Lb=span_in, Cb=1.0).phi_Rn
    return Mu / max(phi_Mn, 1e-12)


def _worst_uplift_psf(loads: LateralLoads, config: ClearSpanConfig,
                      x_ft: float | None, mid_z_ft: float | None) -> float:
    """Worst net uplift pressure (positive = up) over the four wind cases
    at a roof position; None for a coordinate means take the worst band
    anywhere in that direction (used for girders, which span the full x)."""
    worst = 0.0
    for wp, dist in ((loads.wind_x, x_ft),
                     (loads.wind_x, None if x_ft is None else config.span_ft - x_ft),
                     (loads.wind_z, mid_z_ft),
                     (loads.wind_z, None if mid_z_ft is None else config.length_ft - mid_z_ft)):
        if dist is None:
            p_ext = min(b.pressure_psf for b in wp.roof_bands)
        else:
            p_ext = _band_pressure_psf(wp.roof_bands, dist)
        worst = max(worst, -(p_ext - wp.gcpi_qh_psf))
    return worst


def _uplift_extras(system: LateralSystem, config: ClearSpanConfig,
                   loads: LateralLoads,
                   assignment: dict[str, WShape]) -> dict[str, tuple]:
    """member name -> (span_in, trib_in, extra_dead_plf, p_up_psf) for every
    member that gets the closed-form uplift check. span_in is the UPLIFT
    span: the member length, except end girders supported by gable columns,
    whose uplift spans are the segments between supports (checking them
    over the full clear span would spuriously condemn every candidate)."""
    geometry = system.geometry
    node = {n.name: n for n in geometry.nodes}
    groups = (UPLIFT_GROUPS_TRUSS if config.girder_system == TRUSS
              else UPLIFT_GROUPS_W)
    sp_ft = config.purlin_spacing_actual_ft
    out: dict[str, tuple] = {}
    for m in geometry.members:
        if m.group not in groups:
            continue
        if m.group in (PURLIN, EAVE_STRUT):
            x_ft = node[m.i_node].x / FT
            z_ft = (node[m.i_node].z + node[m.j_node].z) / 2.0 / FT
            out[m.name] = (m.length_in, m.trib_width_in, 0.0,
                           _worst_uplift_psf(loads, config, x_ft, z_ft))
        else:   # W girders: purlin dead weight included in the resisting D
            j = int(m.name[1:])
            trib_ft = (config.frame_spacing_ft
                       if 0 < j < config.n_frames - 1
                       else config.frame_spacing_ft / 2.0)
            purlin_plf = assignment[PURLIN].weight_plf * trib_ft / sp_ft
            span_in = m.length_in
            if m.group == END_GIRDER and config.end_wall_columns:
                span_in /= (config.end_wall_columns + 1)
            out[m.name] = (span_in, trib_ft * 12.0, purlin_plf,
                           _worst_uplift_psf(loads, config, None, None))
    return out


# ---------------------------------------------------------------------------
# Screening (the gravity optimizer's screen + the uplift extra check)
# ---------------------------------------------------------------------------

def _screen_group(cands: list[WShape], demands: list[MemberDemand],
                  params: CheckParams, uplift: dict[str, tuple],
                  config: ClearSpanConfig) -> tuple[WShape, bool]:
    best, best_uc = None, float("inf")
    for shape in cands:                    # lightest-first
        worst = 0.0
        for d in demands:
            uc = check_member(shape, d, params)["governing_uc"]
            if d.name in uplift:
                span_in, trib_in, extra_plf, p_up = uplift[d.name]
                uc = max(uc, uplift_uc(shape, span_in, trib_in, extra_plf,
                                       config.superimposed_dead_psf, p_up,
                                       params.Fy, params.E))
            worst = max(worst, uc)
        if worst <= 1.0:
            return shape, True
        if worst < best_uc:
            best, best_uc = shape, worst
    return best, False


def _weights(geometry, assignment) -> tuple[float, dict[str, float]]:
    by_group: dict[str, float] = {g: 0.0 for g in assignment}
    for m in geometry.members:
        by_group[m.group] += assignment[m.group].weight_plf * (m.length_in / FT)
    return sum(by_group.values()), by_group


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class LateralDesignResult:
    feasible: bool
    converged: bool
    sections: dict[str, str]
    total_weight_lb: float
    weight_by_group_lb: dict[str, float]
    member_table: pd.DataFrame
    drift_table: pd.DataFrame
    basis: dict
    system: LateralSystem
    gravity: OptimizationResult
    analysis_config: ClearSpanConfig
    bay_shears_kip: dict[int, float]
    bay_forces: dict[int, BracedBayForces]
    roof_truss: RoofTrussForces
    base_reactions: dict
    iterations: list[dict] = field(default_factory=list)

    def as_optimization_result(self) -> OptimizationResult:
        """The combined design in the gravity result shape, so the existing
        exports/wireframe work on it (pass system.geometry to them)."""
        rows = []
        for group in self.system.geometry.groups:
            sub = self.member_table[self.member_table["group"] == group]
            worst = sub.loc[sub["governing_uc"].idxmax()]
            rows.append({
                "group": group, "profile": self.sections[group],
                "n_members": len(sub),
                "weight_lb": self.weight_by_group_lb[group],
                "max_uc": worst["governing_uc"],
                "governing_limitstate": worst["governing_limitstate"],
                "governing_member": worst["member"],
                "all_pass": bool(sub["PASS"].all()),
            })
        return OptimizationResult(
            feasible=self.feasible, converged=self.converged,
            sections=self.sections, total_weight_lb=self.total_weight_lb,
            weight_by_group_lb=self.weight_by_group_lb,
            member_table=self.member_table,
            group_summary=pd.DataFrame(rows),
            iterations=self.iterations, config=self.analysis_config,
        )

    def summary(self) -> str:
        opt = self.as_optimization_result()
        lines = ["=" * 62,
                 "frame_optimizer - lateral design result (ASCE 7-16)",
                 "=" * 62]
        status = ("FEASIBLE" if self.feasible
                  else "INFEASIBLE (best attempt shown)")
        lines.append(f"Status: {status}, "
                     f"{len(self.iterations)} sizing iteration(s)")
        lines.extend(self.analysis_config.describe())
        lines.extend(self.system.describe())
        elf = self.basis["seismic"]["elf"]
        lines.append(
            f"Loads:  seismic V = {elf['V_kip']} kip (Cs = {elf['Cs']}), "
            f"wind base shear x/z = "
            f"{self.basis['wind']['x']['base_shear_kip']}/"
            f"{self.basis['wind']['z']['base_shear_kip']} kip")
        lines.append("-" * 62)
        lines.append("Certified sections (gravity members re-checked under "
                     "all combos):")
        for _, row in opt.group_summary.iterrows():
            marker = ""
            if row["group"] in self.gravity.sections and \
                    self.gravity.sections[row["group"]] != row["profile"]:
                marker = f"  [gravity: {self.gravity.sections[row['group']]}]"
            lines.append(
                f"  {row['group']:<15} {row['profile']:<9} "
                f"({int(row['n_members'])} members, "
                f"{row['weight_lb']:,.0f} lb)  max UC = {row['max_uc']:.3f} "
                f"[{row['governing_limitstate']} @ {row['governing_member']}]"
                + marker)
        lines.append("-" * 62)
        n_fail_drift = int((~self.drift_table["PASS"]).sum())
        worst = self.drift_table.loc[self.drift_table["uc"].idxmax()]
        lines.append(
            f"Drift:  {len(self.drift_table)} checks, worst UC = "
            f"{worst['uc']:.2f} ({worst['direction']}, {worst['location']}, "
            f"{worst['case']})" + (f", {n_fail_drift} FAIL" if n_fail_drift
                                   else ", all PASS"))
        lines.append(
            f"Total steel weight: {self.total_weight_lb:,.0f} lb "
            f"({self.total_weight_lb / 2000.0:,.2f} tons), "
            f"+{self.total_weight_lb - self.gravity.total_weight_lb:,.0f} lb "
            "over the gravity-only design")
        n_fail = int((~self.member_table["PASS"]).sum())
        if n_fail:
            lines.append(f"WARNING: {n_fail} member(s) FAIL - see member_table.")
        lines.append("=" * 62)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The design loop
# ---------------------------------------------------------------------------

def design_lateral(gravity: OptimizationResult, hazards: SiteHazards,
                   lateral_config: LateralSystemConfig, *,
                   R: float | None = None, Cd: float = CD_NOT_DETAILED,
                   wall_weight_psf: float = 0.0,
                   include_live_fraction: float = 0.0,
                   live_is_snow: bool = False,
                   Kzt: float = 1.0, Ke: float = 1.0,
                   seismic_drift_ratio: float = SEISMIC_DRIFT_RATIO,
                   wind_drift_denom: float = WIND_DRIFT_DENOM,
                   max_iterations: int = 8,
                   verbose: bool = False) -> LateralDesignResult:
    """Complete lateral design off the back of a gravity optimization; see
    the module docstring for the algorithm and design basis."""
    if not gravity.feasible:
        raise ValueError("the gravity design is infeasible; fix it before "
                         "designing the lateral system.")
    config: ClearSpanConfig = gravity.config
    system = build_lateral_system(config, lateral_config)
    geometry, a_config = system.geometry, system.analysis_config
    params = lateral_check_params(a_config, system)

    catalog = {g: get_shapes(names)
               for g, names in system.candidates_by_group().items()}
    # ratchet floors: gravity groups may only grow
    floors: dict[str, float] = {}
    for g, name in gravity.sections.items():
        sel = get_shapes([name])[0]
        floors[g] = sel.weight_plf
        if all(s.name != sel.name for s in catalog[g]):
            catalog[g] = sorted(catalog[g] + [sel], key=lambda s: s.weight_plf)
    def usable(g):
        return [s for s in catalog[g] if s.weight_plf >= floors.get(g, 0.0)]

    assignment = {g: (get_shapes([gravity.sections[g]])[0]
                      if g in gravity.sections else catalog[g][0])
                  for g in catalog}

    braced = system.braced_bay_indices
    n_bays = config.n_frames - 1
    bay_ft = config.frame_spacing_ft
    wall_h_ft = roof_height_ft(config)
    tiers = system.n_tiers
    # widest actual panel -> longest diagonal -> worst tension for the
    # one-size roof-brace group
    panel_w_ft = max(system.roof_panel_widths_ft)
    Ie = hazards.site.Ie

    def loads_and_shares(assign):
        steel_lb, _ = _weights(geometry, assign)
        loads = compute_lateral_loads(
            config, steel_lb, hazards,
            **({"R": R} if R is not None else {}),
            wall_weight_psf=wall_weight_psf,
            include_live_fraction=include_live_fraction,
            live_is_snow=live_is_snow, Kzt=Kzt, Ke=Ke)
        shares = design_bay_shears_kip(
            wind_bay_shares_kip(loads.wind_z, braced, n_bays),
            seismic_bay_shares_kip(loads.elf.V_kip / 2.0, braced, n_bays,
                                   bay_ft))
        return loads, shares

    iterations: list[dict] = []
    converged = False
    demands = drift_raw = loads = shares = None
    forces_by_bay: dict[int, BracedBayForces] = {}
    roof_tr: RoofTrussForces | None = None

    def bay_drift_pair_in(assign, b: float, loads) -> tuple[float, float]:
        """(amplified seismic, 10-yr wind) roof-line drift of braced bay b."""
        strut_A = assign.get(WALL_STRUT, assign[BRACE]).A

        def drift(V):
            return bay_drift_in(V, bay_ft, wall_h_ft, tiers, assign[BRACE].A,
                                strut_A, assign[COLUMN].A, params.E)

        v_seis = seismic_bay_shares_kip(loads.elf.V_kip / 2.0, braced,
                                        n_bays, bay_ft)[b]
        v_wind = wind_bay_shares_kip(loads.wind_z, braced, n_bays)[b]
        return drift(v_seis) * Cd / Ie, 0.49 * drift(v_wind)

    def longitudinal_drift_ok(assign, loads) -> bool:
        for b in braced:
            seis, wind = bay_drift_pair_in(assign, b, loads)
            if seis > seismic_drift_ratio * wall_h_ft * FT:
                return False
            if wind > wall_h_ft * FT / wind_drift_denom:
                return False
        return True

    for escalation in range(10):     # transverse-drift stiffness escalation
        # oscillation guard, exactly like the gravity optimizer: once an
        # assignment repeats, only ratchet upward — the finite candidate
        # lists then force termination in a few strictly-heavier steps
        history = [tuple(assignment[g].name for g in sorted(assignment))]
        monotone = False

        def merged_up(new):
            return {g: max(assignment[g], new[g], key=lambda s: s.weight_plf)
                    for g in assignment}

        for it in range(1, max_iterations + 1):
            loads, shares = loads_and_shares(assignment)
            forces_by_bay = {b: bay_forces(v, bay_ft, wall_h_ft, tiers)
                             for b, v in shares.items()}
            roof_tr = roof_truss_forces(loads.wind_z, config.span_ft, bay_ft,
                                        panel_w_ft)
            collector = max(shares.values())

            demands3d, drift_raw, service = analyze_lateral(
                geometry, assignment, a_config, loads)
            demands = _augment_demands(demands3d, system, config,
                                       forces_by_bay, roof_tr, collector)
            uplift = _uplift_extras(system, config, loads, assignment)

            grouped: dict[str, list[MemberDemand]] = {g: [] for g in catalog}
            for d in demands:
                grouped[d.group].append(d)

            new_assignment: dict[str, WShape] = {}
            screen_ok = True
            for g in catalog:
                shape, ok = _screen_group(usable(g), grouped[g], params,
                                          uplift, config)
                new_assignment[g] = shape
                screen_ok &= ok

            # longitudinal drift: bump the brace until the bay is stiff enough
            while (not longitudinal_drift_ok(new_assignment, loads)
                   and any(s.weight_plf > new_assignment[BRACE].weight_plf
                           for s in usable(BRACE))):
                new_assignment[BRACE] = next(
                    s for s in usable(BRACE)
                    if s.weight_plf > new_assignment[BRACE].weight_plf)

            if monotone:
                new_assignment = merged_up(new_assignment)

            iterations.append({
                "iteration": len(iterations) + 1,
                "assignment": {g: s.name for g, s in new_assignment.items()},
                "feasible_screen": screen_ok,
            })
            if verbose:
                print(f"[lateral iter {len(iterations)}] "
                      f"{iterations[-1]['assignment']} feasible={screen_ok}")
            if all(new_assignment[g] is assignment[g] for g in assignment):
                converged = True
                break
            key = tuple(new_assignment[g].name for g in sorted(new_assignment))
            if key in history and not monotone:
                monotone = True
                new_assignment = merged_up(new_assignment)
                if all(new_assignment[g] is assignment[g] for g in assignment):
                    converged = True   # merge landed on the current assignment
                    break
            history.append(key)
            assignment = new_assignment

        # transverse drift (single story: hsx = wall height)
        h_in = wall_h_ft * FT
        worst_ratio = max(
            max(d["seismic"] * Cd / Ie / (seismic_drift_ratio * h_in),
                d["wind"] * wind_drift_denom / h_in)
            for d in drift_raw.values())
        if worst_ratio <= 1.0:
            break
        # stiffen the columns first, then the girder/chords. Frame sway
        # scales ~1/EI, so JUMP the floor to the candidate whose Ix covers
        # the worst drift ratio in one step instead of one catalog step per
        # (expensive) re-analysis; the next pass corrects any overshoot the
        # discrete catalog forces.
        stiff_groups = [COLUMN] + ([GIRDER] if config.girder_system != TRUSS
                                   else [TRUSS_TOP_CHORD])
        bumped = False
        for g in stiff_groups:
            heavier = [s for s in usable(g)
                       if s.weight_plf > assignment[g].weight_plf]
            if heavier:
                target_Ix = assignment[g].Ix * worst_ratio
                pick = next((s for s in heavier if s.Ix >= target_Ix),
                            heavier[-1])
                floors[g] = pick.weight_plf
                assignment[g] = pick
                bumped = True
                if verbose:
                    print(f"[drift] worst drift ratio {worst_ratio:.2f} — "
                          f"raising {g} to {pick.name}")
                break
        if not bumped:
            break   # nothing left to stiffen: report infeasible drift

    # ---- final tables (demands correspond to the converged assignment) ----
    member_table = check_all(demands, assignment, params)
    uplift = _uplift_extras(system, config, loads, assignment)
    uc_uplift = []
    for d in demands:
        if d.name in uplift:
            span_in, trib_in, extra_plf, p_up = uplift[d.name]
            uc_uplift.append(uplift_uc(
                assignment[d.group], span_in, trib_in, extra_plf,
                config.superimposed_dead_psf, p_up, params.Fy, params.E))
        else:
            uc_uplift.append(0.0)
    member_table["UC_uplift"] = uc_uplift
    governs = member_table["UC_uplift"] > member_table["governing_uc"]
    member_table.loc[governs, "governing_uc"] = member_table.loc[governs, "UC_uplift"]
    member_table.loc[governs, "governing_limitstate"] = "uplift"
    member_table["PASS"] = member_table["governing_uc"] <= 1.0

    h_in = wall_h_ft * FT
    drift_rows = []
    for j, d in drift_raw.items():
        seis = d["seismic"] * Cd / Ie
        drift_rows.append({"direction": "x", "location": f"frame {j}",
                           "case": "seismic (Cd*d/Ie)", "drift_in": seis,
                           "limit_in": seismic_drift_ratio * h_in,
                           "uc": seis / (seismic_drift_ratio * h_in)})
        drift_rows.append({"direction": "x", "location": f"frame {j}",
                           "case": "wind 10-yr", "drift_in": d["wind"],
                           "limit_in": h_in / wind_drift_denom,
                           "uc": d["wind"] / (h_in / wind_drift_denom)})
    for b in braced:
        seis, wind = bay_drift_pair_in(assignment, b, loads)
        drift_rows.append({"direction": "z", "location": f"braced bay {b}",
                           "case": "seismic (Cd*d/Ie)", "drift_in": seis,
                           "limit_in": seismic_drift_ratio * h_in,
                           "uc": seis / (seismic_drift_ratio * h_in)})
        drift_rows.append({"direction": "z", "location": f"braced bay {b}",
                           "case": "wind 10-yr", "drift_in": wind,
                           "limit_in": h_in / wind_drift_denom,
                           "uc": wind / (h_in / wind_drift_denom)})
    drift_table = pd.DataFrame(drift_rows)
    drift_table["PASS"] = drift_table["uc"] <= 1.0

    combos = list(active_strength_combos(loads.combos)) + [
        SERVICE_TOTAL_COMBO[0], SERVICE_LIVE_COMBO[0]]
    base_reactions = {
        n.name: {c: {"FX": _r(service.nodes[n.name].RxnFX[c]),
                     "FY": _r(service.nodes[n.name].RxnFY[c])}
                 for c in combos}
        for n in geometry.nodes if n.is_base}

    steel_lb, by_group = _weights(geometry, assignment)
    basis = lateral_load_basis(
        config, steel_lb, hazards, **({"R": R} if R is not None else {}),
        wall_weight_psf=wall_weight_psf,
        include_live_fraction=include_live_fraction,
        live_is_snow=live_is_snow, Kzt=Kzt, Ke=Ke)

    return LateralDesignResult(
        feasible=(bool(member_table["PASS"].all())
                  and bool(drift_table["PASS"].all()) and converged),
        converged=converged,
        sections={g: s.name for g, s in assignment.items()},
        total_weight_lb=steel_lb,
        weight_by_group_lb=by_group,
        member_table=member_table,
        drift_table=drift_table,
        basis=basis,
        system=system,
        gravity=gravity,
        analysis_config=a_config,
        bay_shears_kip=shares,
        bay_forces=forces_by_bay,
        roof_truss=roof_tr,
        base_reactions=base_reactions,
        iterations=iterations,
    )


# ---------------------------------------------------------------------------
# Phase 5 exports
# ---------------------------------------------------------------------------

def lateral_baseplate_inputs(result: LateralDesignResult) -> dict:
    """Baseplate/anchorage demands for the COMBINED design: vertical and
    x-shear reactions per LRFD combo from the certified model (net uplift
    shows up as negative FY under the 0.9D+W combos), plus the closed-form
    braced-bay actions (z-shear, overturning couple, tier-1 diagonal anchor
    uplift) on the braced lines."""
    geometry = result.system.geometry
    assignment = {g: get_shapes([n])[0] for g, n in result.sections.items()}
    base_members = {}
    base_nodes = {n.name: n for n in geometry.nodes if n.is_base}
    for m in geometry.members:
        if m.group == COLUMN and m.i_node in base_nodes:
            base_members[m.name] = base_nodes[m.i_node]

    braced_lines: dict[int, float] = {}
    anchor_uplift: dict[int, float] = {}
    for b, f in result.bay_forces.items():
        for line in (b, b + 1):
            braced_lines[line] = max(braced_lines.get(line, 0.0), f.base_shear_kip)
            anchor_uplift[line] = max(anchor_uplift.get(line, 0.0),
                                      f.anchor_uplift_kip + f.column_couple_kip)

    columns = []
    for name, node in sorted(base_members.items()):
        rxn = result.base_reactions[node.name]
        fy = {c: v["FY"] for c, v in rxn.items()}
        entry = {
            "member_id": name,
            "base_node": node.name,
            "section": _section_dimensions(assignment[COLUMN]),
            "centerline_location": {"x_in": _r(node.x), "y_in": _r(node.y),
                                    "z_in": _r(node.z)},
            "vertical_reaction_kip_by_combo": {c: _r(v) for c, v in fy.items()},
            "max_compression_kip": _r(max(fy.values())),
            "net_uplift_kip": _r(min(0.0, min(fy.values()))),
            "shear_x_kip_by_combo": {c: _r(v["FX"]) for c, v in rxn.items()},
        }
        if not name.startswith("CG"):
            line = int(name.split(".")[1])
            if line in braced_lines:
                couple = max(f.column_couple_kip
                             for b, f in result.bay_forces.items()
                             if line in (b, b + 1))
                entry["braced_bay_z"] = {
                    "shear_z_kip": _r(braced_lines[line]),
                    "overturning_couple_kip": _r(couple),
                    "diagonal_anchor_uplift_kip": _r(anchor_uplift[line]),
                }
        columns.append(entry)

    return {
        "schema": "frame_optimizer/lateral_baseplate_inputs",
        "schema_version": 1,
        "base_condition": "pinned",
        "units": {"length": "in", "force": "kip"},
        "notes": [
            "Reactions from the certified nominal-stiffness model over every "
            "active LRFD combo (DAM notional loads excluded). Negative "
            "vertical values are NET UPLIFT — anchor rods must be designed "
            "for them.",
            "braced_bay_z entries are the closed-form longitudinal actions "
            "on braced lines: design the anchorage and foundation ties for "
            "the z-shear and the diagonal anchor uplift concurrently with "
            "the tabulated vertical reactions.",
            "Ez load cases act only on the braced bays (closed-form); their "
            "combos do not appear in the by-combo tables.",
        ],
        "columns": columns,
    }


def write_lateral_baseplate_json(result: LateralDesignResult, path) -> object:
    return _write_json(lateral_baseplate_inputs(result), path)


def lateral_system_block(result: LateralDesignResult) -> dict:
    """The 'lateral_system' block for building_configuration()."""
    s = result.system
    return {
        "transverse": s.transverse_system,
        "braced_bay_indices": s.braced_bay_indices,
        "n_brace_tiers": s.n_tiers,
        "brace_angle_deg": _r(s.brace_angle_deg, 1),
        "roof_brace_bays": s.roof_brace_bays,
        "design_bay_shears_kip": {str(b): _r(v)
                                  for b, v in result.bay_shears_kip.items()},
        "roof_truss_diag_tension_kip": _r(result.roof_truss.diag_tension_kip),
        "seismic_basis": "R = 3, Cd = 3 (not specifically detailed), SDC A-C",
    }
