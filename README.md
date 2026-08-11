# frame_optimizer

Gravity-load optimizer for fully pinned steel frames (AISC W-shapes).
Pipeline: [Pynite](https://github.com/JWock82/Pynite) 3-D FEA → AISC 360 LRFD
checks → lightest-section search over candidate section combinations.

Primary entry point: **[gravity_design.py](gravity_design.py)** — clear-span
industrial building (equipment enclosure, no interior columns).

## Quick start

```bash
pip install -e .[viz]      # [viz] adds plotly for the wireframe (optional);
                           # core needs only numpy, pandas, PyniteFEA
python gravity_design.py
```

All inputs live in the `ClearSpanConfig` block of `gravity_design.py`.
No CLI args.

**Units — SI in, SI out.** Interface units: meters, kPa (kN/m²) for surface
loads, MPa for material, millimeters for camber. Results report kN, kN·m,
kg, and mm. Internally the FE solver and the AISC strength equations run in
the AISC-native consistent kip/inch system; the exact conversion factors in
[config.py](src/frame_optimizer/config.py) are applied once at the config
boundary and once when reporting, so no mixed-unit arithmetic can occur.

The only geometric inputs are the building **footprint**: `span_m`,
`length_m`, `height_m`. The frame layout is **derived**, not user-specified: `optimize_layout()`
searches the realistic layout band for the footprint (bays ~6–9 m, purlins
~1.2–1.8 m, end-girder segments ≤ ~7.6 m) and keeps the lightest feasible
design. A footprint no longer than one bay collapses to a minimal 1×1-bay
enclosure (2 frames, no gable columns). Footprint orientation is
self-correcting: if `span_m > length_m` the two are swapped, so girders
always clear-span the shorter plan dimension (girder demand grows with
span², so spanning the long way is never lighter).

## What gravity_design.py does

1. Defines a `ClearSpanConfig`: 25 m × 35 m plan footprint, 9.14 m eave,
   candidate W-shapes per design group, roof loads.
2. Calls `optimize_layout(config)` — derives the layout from the footprint
   and returns the lightest feasible `OptimizationResult`.
3. Emits (to the git-ignored `output/` directory):

| Output | Content | Consumer |
|---|---|---|
| `result.summary()` (stdout) | selected sections, weights, governing checks | humans |
| `member_checks_clear_span.csv` | one row per member, all unity checks (kN, kN·m, m) | review |
| `baseplate_inputs.json` | per-column footprint + base reactions (mm, kN) | baseplate module |
| `building_configuration.json` | full geometry + sections (mm, m, kg, kPa, MPa) | IFC authoring module |
| `clear_span_wireframe.html` | interactive 3-D wireframe (m, kN) | visual check (needs `[viz]`) |

## Building topology

- X = clear-span direction, Z = building length, Y = up. Origin at base of
  the x=0, z=0 column.
- Transverse frames at `length/(n_frames-1)` spacing. Each frame: two
  perimeter columns + one clear-span roof girder. Interior stays empty.
- Purlins run in Z between girders, spaced along the span. Eave lines carry
  half tributary width.
- Optional gable columns on the two end walls only (count chosen by the
  layout search). They support the end girders, which then form their own
  lighter design group — providing `end_girder_candidates` is what enables
  this option.
- One-way load path: deck → purlins → girders → perimeter columns.

### Pratt-truss roof for very long spans

> Design process with the governing math: **[docs/truss_design.md](docs/truss_design.md)**

When the clear span outgrows every rolled W-shape (`roof_system="auto"`, the
default, proves this with the FEA-free girder bound before any solve — or set
`roof_system="truss"` explicitly), the interior girders are replaced by
parallel-chord **Pratt trusses**:

- **Top-chord bearing**: the top chord stays at the girder elevation and
  bears on the same column tops; the truss depth (`truss_depth_m`, default
  span/12) hangs below the eave. Roof plane, purlins, end walls, and columns
  are untouched — interior clearance under the truss is reduced by the depth.
- Even panel count with panel length ≈ depth (diagonals ≈ 45°); the bottom
  chord ends at the first interior panel points and the end diagonals carry
  the support shear (no members on the column axis). Diagonals are in
  tension under gravity, verticals in compression.
- The top chord is one continuous full-span member (like the girder it
  replaces), so its chord-relative sag IS the truss deflection check and
  `truss_camber_mm` is credited the same way girder camber is.
- Second order: the search amplifies compression-chord moments with the
  AISC Appendix 8 **B1** factor (B2 = 1, non-sway gravity); the winning
  design is then re-verified with second-order **axial** forces from a
  Pynite P-Delta solve and the check table is rebuilt from them (see
  `_verify_second_order`). The summary reports the verification outcome.
- End frames keep their gable-column-propped W end girders
  (`end_girder_candidates` is required in truss mode).

Design groups (one shared section per group; heaviest-loaded member governs):

| Group | Members | Notes |
|---|---|---|
| `column` | perimeter + gable columns | KL/r ≤ 200 check, no deflection check |
| `girder` | interior clear-span girders | Lb = purlin spacing, camber credit |
| `end_girder` | the two end-wall girders | own candidates; lighter when gable columns exist |
| `purlin` | roof purlins | `purlin_Lb_m=0` = deck-braced top flange |
| `top_chord` | truss top chords (truss mode) | KLx = panel, KLy/Lb = purlin spacing, B1, camber, deflection check (δ ∝ 1/A) |
| `bottom_chord` | truss bottom chords (truss mode) | tension, L/r ≤ 300; braced out-of-plane at `bottom_chord_brace_spacing_m` (assumed struts, default every panel) |
| `truss_web` | truss verticals + diagonals (truss mode) | one shared shape; pin-ended, checked over own length |

## Pipeline

`optimize_layout(config)` in
[optimizer.py](src/frame_optimizer/optimization/optimizer.py) is the
clear-span entry point: it enumerates every realistic layout for the
footprint (`candidate_layouts()` in `clear_span.py`), runs `optimize()` on
each, and returns the lightest feasible design (weight ties break toward
fewer members). Layout fields set explicitly on the config are pinned and
excluded from the search. Per layout, `optimize(config)`:

1. **Geometry** — [clear_span.py](src/frame_optimizer/clear_span.py)
   `build_clear_span_geometry()`: nodes + members tagged with group and
   tributary width. Pure data (`FrameGeometry`), no FEA objects.
2. **Analysis** — [frame_model.py](src/frame_optimizer/analysis/frame_model.py)
   `analyze_frame()`: one Pynite model for the whole building. All member
   ends moment-released (fully pinned). Load cases D (self-weight + SDL) and
   L; combos 1.4D, 1.2D+1.6L (strength), D+L, L (deflection). Returns one
   `MemberDemand` per member: enveloped Pu, Mux, Muy, Vu + chord-relative sag.
3. **Checks** — [checker.py](src/frame_optimizer/design/checker.py)
   `check_member()`: unity checks per member (axial, flexure w/ LTB, shear,
   H1 interaction, deflection, slenderness). Per-group knobs in `GroupRules`.
   Strength equations are pure functions in
   [aisc_strengths.py](src/frame_optimizer/design/aisc_strengths.py).
4. **Search** — fixed-point iteration, not brute force. Demands are nearly
   statically determinate (only self-weight feedback), so: FEA → pick
   lightest passing candidate per group → re-FEA → repeat until stable.
   Typically 1–2 solves. Last iteration doubles as certification.
   `method="exhaustive"` cross-checks by enumeration (small lists only).
5. **Result** — [results.py](src/frame_optimizer/results.py):
   `OptimizationResult` with `sections`, `total_weight_kg`, `member_table`
   (DataFrame), `group_summary`, `feasible`/`converged` flags, and the config.

`evaluate(config, {"girder": "W30X108", ...})` checks one explicit assignment
without searching.

### Runtime engineering

**Guarantee: whenever a feasible design exists, the search returns the same
lightest one it always would.** Work is only ever skipped once it is certain
it cannot change that answer. Every design is still certified by a real FEA
solve and the full AISC check table.

- **FEA-free infeasibility proof** (`_infeasibility_proof`): before any
  solve, closed-form statics put a rigorous **lower bound** on each group's
  demands — the load the structure must carry at minimum, with all
  self-weight dropped (interior purlin reactions as point loads on the
  girder, `M = Σ P·a/2`, `δ = Σ P·a(3L²−4a²)/48EI`). Capacities are
  unchanged (same shape, rules and member length), so a candidate failing
  the bound must fail the real demands. When no candidate of some group can
  clear its bound, that layout is infeasible with certainty and is skipped.
  This also catches demand-independent limit states exactly: `KL/r ≤ 200`
  depends only on length and shape, so an over-tall column rules out every
  layout at once. Rejection requires clearing 1.0 by a 2% margin
  (`_PROOF_MARGIN`), covering solver round-off and deflection sampling: a
  144-configuration sweep put the worst bound overshoot at 1.9×10⁻³, an
  order of magnitude inside the margin. Disable with
  `optimize_layout(prescreen=False)` to force the exhaustive path.
- **Statics pre-sizing** (`_presize_clear_span`): the same determinate load
  path gives the fixed-point loop its starting assignment, so it begins at
  (usually) the answer and converges in 1–2 solves instead of 3–4. Only a
  starting guess; the certified table is unaffected.
- **Exact demand deduplication** (`_distinct_demands`): symmetric structures
  produce many members whose demands are bit-identical (all interior purlins
  of a bay, both columns of a frame). Screening one of each is exactly
  equivalent to screening all — equality is exact, so near-identical values
  are still checked separately.
- **Warm starts across layouts**: adjacent candidate layouts differ little,
  so each layout seeds its search from the previous winner
  (`optimize(..., warm_start=...)`).
- **Reactions-free screening solves** (`solve_model` in frame_model.py):
  Pynite's `analyze()` spends ~40% of its time recovering reactions the
  optimizer never reads. Member results are identical; the baseplate export
  still uses the full `analyze()` for its one reactions solve.
- **Parallel layout search**: independent layouts run in worker processes
  (`optimize_layout(..., n_jobs=...)`). Measured best-of-3, parallel wins at
  every size tried — 1.7× on a 15 × 20 m footprint up to 3.4× on 30 × 45 m —
  so `_run_layouts` only falls back to serial when there is genuinely nothing
  to split (a single chunk, or work too small to repay the ~1.5 s per-worker
  import). Parallel and serial runs produce identical designs; anything that
  prevents multiprocessing falls back to serial automatically.
- **Fork-bomb guard**: on spawn platforms every worker re-imports the
  caller's `__main__`, so a script calling `optimize_layout()` at module
  level without an `if __name__ == "__main__"` guard would have each worker
  start another pool. `_run_layouts` refuses to spawn from inside a worker,
  making such a script merely slow instead of fatal. Guard your entry point
  anyway — `gravity_design.py` does.

One behavioral note, for infeasible footprints only: when *no* layout is
feasible there is no design to find, only a diagnosis to report. The
optimizer analyzes the closest few layouts (`_HOPELESS_ANALYSES`) and
returns the best of those, so the layout named in the report can differ from
what an exhaustive search would name — both are infeasible, and the
governing limit states and check table are real either way.

## JSON exports

[export.py](src/frame_optimizer/export.py). Every numeric key has an SI unit
suffix (`_mm`, `_m`, `_kN`, `_kPa`, `_MPa`, `_kg`, `_kg_m`). Both files carry
`schema` + `schema_version` (**2** — version 1 used US customary units).

**`baseplate_inputs.json`** — `write_baseplate_json(result)`. One entry per
column landing on a base (includes gable columns). Per column:

- `section`: name (AISC label), `depth_d_mm`, `flange_width_bf_mm`, tf, tw,
  `area_mm2`, `nominal_weight_kg_m`
- `centerline_location`: base-node x/y/z in mm
- `axial_compression_kN`: governing LRFD value + per-combo breakdown
  (`1.4D`, `1.2D+1.6L`, and service `D`, `L`, `D+L`)

Reactions come from one extra linear solve of the final assignment; vertical
base reaction = column axial. Compression-positive. Base condition: pinned.
No lateral shear — out of model scope. Column web orientation not defined by
the gravity model.

**`building_configuration.json`** — `write_building_json(result)`:

- `building`: span, length, eave height, frame count/spacing, purlin lines,
  gable columns, camber (m / mm)
- `design_groups`: selected W-shape per group with profile dimensions in mm
  (enough for a parametric IFC I-section), member count, weight (kg), max UC
- `nodes` / `members`: complete analysis topology (names, coordinates in mm,
  connectivity, group, section)
- material (MPa), loads (kPa) + combos, connection assumption, headline results

## Engineering assumptions (must-read)

- **Gravity only. Fully pinned.** The frame is a lateral mechanism; nodes are
  restrained in DX/DZ/rotations purely to remove mechanism DOFs. Valid only
  because those restraints attract no force under gravity. **Never add
  lateral loads to this model.** Wind/seismic need a separate system —
  a tall single-story shell is usually wind-governed. (Truss frames are the
  one exception to the blanket DX restraint: their panel nodes must
  translate for chord action, so each truss frame is supported pin + roller
  — one bearing keeps DX, everything else on that frame frees it.)
- Truss mode: top chord braced out-of-plane by the purlins, bottom chord by
  assumed bracing struts at `bottom_chord_brace_spacing_m` (not modeled);
  webs are checked pin-ended over their own length, K = 1. Connection and
  gusset weight is not included in the reported steel mass.
- Purlins are explicit pin-ended members; they deliver true point reactions
  to the girders at shared nodes. Girders are Pynite physical members:
  subdivided at purlin nodes, checked over the full span, self-weight only
  as direct load. Purlin nodes get free rotations (the continuous girder
  stabilizes them) — clamping them would falsify girder bending.
  Statics close exactly (tested to 0.1%).
- Columns: pin–pin, K = 1.0, L = eave height.
- `live_kpa` = governing of ASCE 7 roof live (Lr) and snow.
- Girder Lb defaults to actual purlin spacing (purlins brace the compression
  flange). Purlin Lb defaults to full span (conservative); set
  `purlin_Lb_m=0` for through-fastened deck.
- Cb = 12.5/11 (AISC F1-1, parabolic diagram) when a member is a single
  unbraced segment; 1.0 otherwise.
- Camber (`girder_camber_mm`): credited against the total-deflection check
  only, never below the live-load deflection. Keep ≤ dead-load sag.
- Deflection defaults are the strict floor ratios (L/360 live, L/240 total);
  relax per group via `girder_defl_*_ratio` / `purlin_defl_*_ratio` when
  roof limits apply.
- Not modeled: crane loads, hanging equipment, drifted snow, connections.

## Section database

[sections/data/aisc_w_shapes.csv](src/frame_optimizer/sections/) — 283
W-shapes from AISC Shapes Database v15.0 (US units; the backend works in
kips/inches, so the catalog is used as-is — only the interface is SI).
`rts`, `ho` computed from their exact definitions. Regenerate with
[tools/prepare_sections_csv.py](tools/prepare_sections_csv.py).

## Layout

```
gravity_design.py                entry point: clear-span building (this README)
src/frame_optimizer/
├── clear_span.py                ClearSpanConfig, layout derivation, geometry builder, group rules
├── config.py                    FrameConfig + SI/imperial conversion constants (exact)
├── geometry.py                  NodeInfo/MemberInfo/FrameGeometry dataclasses; grid builder
├── analysis/frame_model.py      Pynite model build, combos, MemberDemand extraction
├── design/aisc_strengths.py     AISC 360 capacity equations (pure functions)
├── design/checker.py            check_member(), GroupRules, CheckParams
├── optimization/optimizer.py    layout search + iterative/exhaustive section search
├── export.py                    baseplate + building-configuration JSON writers
├── results.py                   OptimizationResult + summary()
└── sections/                    W-shape catalog: CSV + WShape loader
modeler/                         plotly wireframe (optional, delete-able)
tests/                           hand-calc, AISC Manual anchors, regression
```

## Tests

```bash
pip install pytest
pytest tests/
```

Coverage: FEA vs closed-form statics (wL²/8, wL/2, 5wL⁴/384EI, tributary
axials), strength functions vs AISC Manual anchors, clear-span statics
closure, iterative-vs-exhaustive agreement, warm-start invariance, and the
screening invariants — that the statics bounds never exceed the FEA demands
(the property the pre-screen's correctness rests on) and that
`prescreen=True` picks the same design as the exhaustive path. Test configs
feed exact SI conversions of the imperial anchor values, so the hand
calculations remain in the backend's native kip/inch system.
