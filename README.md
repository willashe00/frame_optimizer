# frame_optimizer

Gravity-load optimizer for fully pinned steel frames (AISC W-shapes).
Pipeline: [Pynite](https://github.com/JWock82/Pynite) 3-D FEA → AISC 360 LRFD
checks → lightest-section search over candidate section combinations.

Primary entry point: **[gravity_design.py](gravity_design.py)** — clear-span
industrial building (equipment enclosure, no interior columns)..

## Quick start

```bash
pip install -e .[viz]      # [viz] adds plotly for the wireframe (optional);
                           # core needs only numpy, pandas, PyniteFEA
python gravity_design.py
```

All inputs live in the `ClearSpanConfig` block of `gravity_design.py`.
No CLI args. Interface units: feet and psf. Internal units: kips, inches, ksi.
Metric plan dimensions via `M_TO_FT`.

The only geometric inputs are the building **footprint**: `span_ft`,
`length_ft`, `eave_height_ft`. The frame layout is **derived**, not user-specified: `optimize_layout()`
searches the realistic layout band for the footprint (bays ~20–30 ft, purlins
~4–6 ft, end-girder segments ≤ ~25 ft) and keeps the lightest feasible design.
A footprint no longer than one bay collapses to a minimal 1×1-bay enclosure
(2 frames, no gable columns). Footprint orientation is self-correcting: if
`span_ft > length_ft` the two are swapped, so girders always clear-span the
shorter plan dimension (girder demand grows with span², so spanning the long
way is never lighter).

## What gravity_design.py does

1. Defines a `ClearSpanConfig`: 20 m × 30 m plan footprint, 9.14 m (30 ft)
   eave, candidate W-shapes per design group, roof loads.
2. Calls `optimize_layout(config)` — derives the layout from the footprint
   and returns the lightest feasible `OptimizationResult`.
3. Emits (to the git-ignored `output/` directory):

| Output | Content | Consumer |
|---|---|---|
| `result.summary()` (stdout) | selected sections, weights, governing checks | humans |
| `member_checks_clear_span.csv` | one row per member, all unity checks | review |
| `baseplate_inputs.json` | per-column footprint + base reactions | baseplate module |
| `building_configuration.json` | full geometry + sections | IFC authoring module |
| `clear_span_wireframe.html` | interactive 3-D wireframe | visual check (needs `[viz]`) |

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

Design groups (one shared section per group; heaviest-loaded member governs):

| Group | Members | Notes |
|---|---|---|
| `column` | perimeter + gable columns | KL/r ≤ 200 check, no deflection check |
| `girder` | interior clear-span girders | Lb = purlin spacing, camber credit |
| `end_girder` | the two end-wall girders | own candidates; lighter when gable columns exist |
| `purlin` | roof purlins | `purlin_Lb_ft=0` = deck-braced top flange |

### Truss girders (automatic fallback, or `girder_system="truss"`)

Rolled W girders top out around a 90–100 ft clear span — deflection demands
Ix that grows with span⁴, and even a W44 falls far short at 170 ft. With
`girder_system="auto"` (the default) the choice is automatic: supply
`truss_chord_candidates` + `truss_web_candidates` alongside the girder list,
and `optimize_layout()` uses rolled W girders whenever **any** W layout is
feasible, escalating to trusses **only when none is** — a closed-form screen
(`wide_flange_infeasible_reason`) proves certain infeasibility from the
candidate list's best Ix/Zx and skips the futile W search on long spans. The
decision is recorded in `result.system_search` and printed in the summary.
Pin `girder_system` to `"wide_flange"` or `"truss"` to force either system.

**The girder system also sets the frame behavior.** Wide-flange girders keep
the fully pinned gravity frame with the long-standing checks. Truss girders
make every frame a **rigid transverse bent**: columns run full height to the
top-chord level and the truss ties into each column at *both* chord
elevations (bottom chord at the eave, top chord at the column top) — each
tie is a pin, but the pair, one truss depth apart, forms the moment
connection (the classic mill-bent detail). The bent then provides its own
in-plane stability, so strength design follows the **AISC 360 Chapter C
Direct Analysis Method**: 0.8-reduced stiffness, notional lateral loads
0.003·Yi per gravity case (0.002 per C2.2b + 0.001 per C2.3(c) in lieu of
τb), second-order P-Δ analysis, and K = 1 member checks. Serviceability
uses a parallel nominal-stiffness model. Frame action puts real bending and
shear in the columns (checked as beam-columns under H1), produces real
gravity thrust at the bases (exported for anchorage design), and reverses
the chord forces near the frame corners — the checker verifies **both**
signed axial extremes of every member (tension yielding *and* flexural
buckling), which the single-envelope check of a pinned frame never needed.

In truss mode every frame carries a parallel-chord **Pratt truss** (the
custom-fabricated analog of SJI DLH/SLH long-span joists, with W-shape
chords and webs as is customary for heavy long-span roof trusses). The
bottom chord ties in at the eave, so `eave_height_ft` stays the true
clear height; the depth rises above the eave and the purlins ride the top
chord. An auto-derived depth joins the layout search over the practice band
span/10–span/15 (panels even-count near 45° diagonals re-derive per depth).
Both chords are full-span physical members — purlin lines that miss a panel
point load the continuous top chord in local bending, captured directly —
and the chord sag *is* the checked truss deflection (camber credit applies).
Chord compression uses segment effective lengths: one panel in plane, the
brace spacing out of plane (purlins on top; bottom-chord bridging, assumed
at every panel point per `truss_bottom_brace_ft`, required in practice and
load-free so not modeled). In truss mode `girder_candidates` is ignored;
the groups become:

| Group | Members | Notes |
|---|---|---|
| `truss_top_chord` | top chords (full span) | compression + local bending; KLx = panel, KLy/Lb = purlin spacing |
| `truss_bot_chord` | bottom chords (full span) | tension; braced by bridging (`truss_bottom_brace_ft`) |
| `truss_web` | verticals + diagonals | pin-ended axial members, checked at full length |

v1 scope: end frames carry the same trusses at half tributary width; gable
columns / `end_girder` are not available with trusses. Chord/web candidates
are W-shapes (WT/HSS/double-angle families need AISC F9/F7 clauses — future
work).

## Pipeline

`optimize_layout(config)` in
[optimizer.py](src/frame_optimizer/optimization/optimizer.py) is the
clear-span entry point: it enumerates every realistic layout for the
footprint (`candidate_layouts()` in `clear_span.py`), runs `optimize()` on
each, and returns the lightest feasible design (weight ties break toward
fewer members). Layout fields set explicitly on the config are pinned and
excluded from the search.

Performance: pin-ended purlins are statically determinate, so the transverse
frames are exactly decoupled — each layout analyzes a **3-frame
representative strip** and broadcasts demands to the full member list
(member-for-member identical to the full building to ~1e-5, the residual
being the purlin torsional stiffness retained against spin mechanisms;
tested). Independent layouts evaluate on a **process pool** (`parallel=None`
auto-enables it for big searches; scripts need the standard
`if __name__ == "__main__":` guard, as `gravity_design.py` has). An
oscillating section search ratchets monotonically upward instead of bouncing
to the iteration cap. Net effect: the full 27-layout truss sweep of a
195×170 ft building — DAM P-Δ analysis included — runs in ~3 minutes.

Per layout, `optimize(config)`:

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
   Typically 2–3 solves. Last iteration doubles as certification.
   `method="exhaustive"` cross-checks by enumeration (small lists only).
5. **Result** — [results.py](src/frame_optimizer/results.py):
   `OptimizationResult` with `sections`, `total_weight_lb`, `member_table`
   (DataFrame), `group_summary`, `feasible`/`converged` flags, and the config.

`evaluate(config, {"girder": "W30X108", ...})` checks one explicit assignment
without searching.

## JSON exports

[export.py](src/frame_optimizer/export.py). Every numeric key has a unit
suffix (`_in`, `_ft`, `_kip`, `_psf`, `_ksi`, `_lb`, `_plf`). Both files carry
`schema` + `schema_version`.

**`baseplate_inputs.json`** — `write_baseplate_json(result)`. One entry per
column landing on a base (includes gable columns). Per column:

- `section`: name, `depth_d_in`, `flange_width_bf_in`, tf, tw, area, weight
- `centerline_location`: base-node x/y/z in inches
- `axial_compression_kip`: governing LRFD value + per-combo breakdown
  (`1.4D`, `1.2D+1.6L`, and service `D`, `L`, `D+L`)

Reactions come from one extra linear solve of the final assignment; vertical
base reaction = column axial. Compression-positive. Base condition: pinned.
`base_shear_x_kip` carries the gravity thrust of rigid truss bents
(essentially zero for the pinned wide-flange system); wind/seismic base
shear remains out of model scope, and DAM notional loads are excluded from
exported reactions. Column web orientation not defined by the gravity model.

**`building_configuration.json`** — `write_building_json(result)`:

- `building`: span, length, eave height, frame count/spacing, purlin lines,
  gable columns, camber
- `design_groups`: selected W-shape per group with profile dimensions
  (enough for a parametric IFC I-section), member count, weight, max UC
- `nodes` / `members`: complete analysis topology (names, coordinates,
  connectivity, group, section)
- material, loads + combos, connection assumption, headline results

## Engineering assumptions (must-read)

- **Gravity only.** Wind/seismic are never applied and need a building-level
  lateral design — a tall single-story shell is usually wind-governed. The
  frame behavior depends on the girder system:
  - **Wide-flange girders: fully pinned.** The frame is a lateral mechanism;
    nodes are restrained in DX/DZ/rotations purely to remove mechanism DOFs.
    Valid only because those restraints attract no force under gravity.
    **Never add lateral loads to this model.**
  - **Truss girders: rigid transverse bents.** Each bent is self-stable in
    its plane (frame-plane DX restraints are released), analyzed per the
    AISC Direct Analysis Method: 0.8EI/0.8EA, notional loads 0.003·Yi (the
    only "lateral" loads ever present — fictitious stability devices, kept
    out of exported reactions), P-Δ second order, K = 1 checks. Gravity
    thrust at the bases is real and exported. Out-of-plane (DZ) restraints
    remain mechanism devices exactly as in the pinned scheme.
- Purlins are explicit pin-ended members; they deliver true point reactions
  to the girders at shared nodes. Girders are Pynite physical members:
  subdivided at purlin nodes, checked over the full span, self-weight only
  as direct load. Purlin nodes get free rotations (the continuous girder
  stabilizes them) — clamping them would falsify girder bending.
  Statics close exactly (tested to 0.1%).
- Truss mode: every non-base node of a bent keeps its X translation
  (`NodeInfo.free_dx`) — unlike a beam node, a truss-frame node must
  equilibrate the diagonals' horizontal components through the chords, and
  the bent's own frame action supplies the in-plane stiffness the blanket DX
  mechanism restraint would otherwise fake. Columns are continuous physical
  members through the eave node (both chord ties are pins; the pair is the
  moment connection). Chord tension/compression reproduces M/d, sag
  reproduces 5wL⁴/384EI_truss + web strain, thrust reactions
  self-equilibrate, and the bottom chord's end-compression reversal is
  checked (all tested).
- Columns: pin–pin, K = 1.0, L = eave height.
- `live_psf` = governing of ASCE 7 roof live (Lr) and snow.
- Girder Lb defaults to actual purlin spacing (purlins brace the compression
  flange). Purlin Lb defaults to full span (conservative); set
  `purlin_Lb_ft=0` for through-fastened deck.
- Cb = 12.5/11 (AISC F1-1, parabolic diagram) when a member is a single
  unbraced segment; 1.0 otherwise.
- Camber (`girder_camber_in`): credited against the total-deflection check
  only, never below the live-load deflection. Keep ≤ dead-load sag.
- Deflection defaults are the strict floor ratios (L/360 live, L/240 total);
  relax per group via `girder_defl_*_ratio` / `purlin_defl_*_ratio` when
  roof limits apply.
- Not modeled: crane loads, hanging equipment, drifted snow, connections.

## Lateral design (lateral_design.py — complete)

`lateral_design.py` runs off the back of `gravity_design.py`: it imports the
same config, runs the gravity optimization, resolves the site hazards from
geographic location (the only required new inputs are latitude/longitude +
one looked-up wind speed), builds and sizes the lateral force-resisting
system, and **re-certifies every gravity member under the complete ASCE
7-16 wind + seismic combination set** — including drift and wind-uplift
force-reversal checks. Outputs: the member check table CSV, the load-basis
JSON, baseplate/anchorage demands (net uplift + braced-bay shear), the
building configuration incl. LFRS, and a wireframe with the braces.

* **Seismic** ([site.py](src/frame_optimizer/site.py)): latitude/longitude →
  SDS/SD1/SDC via the USGS Design Maps web service (cached; every parameter
  manually overridable for offline use). Where ASCE 7-16 §11.4.8 demands a
  site-specific study the service returns nulls and the module *refuses to
  guess* — supply overrides from the study. ELF per §12.8 with R = 3
  ("steel systems not specifically detailed"), valid in SDC A–C only; SDC
  D+ raises instead of producing an unlicensed design.
* **Wind** ([lateral_loads.py](src/frame_optimizer/lateral_loads.py)):
  MWFRS directional procedure (Ch. 27) for the enclosed flat-roof box —
  Kz/qz profiles, leeward Cp by L/B, roof suction bands interpolated on
  h/L, the 16 psf floor, internal pressure in the roof net uplift. The
  basic wind speed has **no free official API** and is a required manual
  input (look it up at https://ascehazardtool.org/).
* **Combinations**: §2.3.1/2.3.6 LRFD combos (wind ±x/±z, seismic ±x/±z,
  Ev folded into the D factor) generated by `all_strength_combos()`;
  `analysis/frame_model.py` now accepts any combo set, so the coming
  phases plug lateral load cases straight into the existing DAM machinery.
* **Sizing + certification** ([lateral_designer.py](src/frame_optimizer/lateral_designer.py)):
  `design_lateral()` runs the fixed-point sizing loop: full-building DAM
  P-Δ analysis over the active combos ([analysis/lateral_model.py](src/frame_optimizer/analysis/lateral_model.py)
  applies the wall-pressure bands, roof net-uplift bands, and ELF story
  forces), closed-form braced-bay actions ([braced_bay.py](src/frame_optimizer/braced_bay.py)
  — statically determinate tension-only X panels, so brace tension, strut
  shear, column couples, collector drag, and roof-truss forces are exact
  hand-checkable statics) folded into the member demands, then the same
  `check_member()` acceptance path as gravity. Gravity sections ratchet up,
  never down. Purlins/eave struts/W girders additionally get the 0.9D+W
  net-uplift check about an unbraced bottom flange (Lb = span). Drift:
  seismic Cd·δ/Ie ≤ 0.025h (12.8.6, code) and ~10-yr wind ≤ h/100 (AISC
  Design Guide 3 customary band for metal-clad industrial buildings —
  tighten `wind_drift_denom` for cranes or brittle finishes), with a
  bounded stiffness-escalation loop. Tests verify exact wind/seismic
  base-shear closure of the loaded model and end-to-end feasibility.
* **LFRS topology** ([lateral_system.py](src/frame_optimizer/lateral_system.py)):
  `build_lateral_system()` augments the gravity geometry with the lateral
  system. Transverse: the frames themselves — truss buildings keep their
  rigid bents; W-girder buildings get moment-connected knees
  (`MemberInfo.fixed_i/j`, `transverse_moment_frame=True`) and run through
  the same Direct Analysis Method path. Longitudinal: tension-only X-braced
  sidewall bays (end bays + ≤5-bay runs, mirror-symmetric, tiered so
  diagonals stay 30–60°, wall struts closing intermediate tiers), the eave
  purlin lines re-tagged as an `eave_strut` collector group, and end-bay
  roof X-bracing panelized to ~45° in plan. The augmented model is
  tested for stability, exact gravity statics closure, and real portal
  knee moments. Topology + candidates go to `output/lateral_system.json`.

## Section database

[sections/data/aisc_w_shapes.csv](src/frame_optimizer/sections/) — 283
W-shapes from AISC Shapes Database v15.0 (US units). `rts`, `ho` computed
from their exact definitions. Regenerate with
[tools/prepare_sections_csv.py](tools/prepare_sections_csv.py).

**Threaded rods** (the as-built reality for X bracing): labels like
`ROD3/4`, `ROD1-1/8` resolve to exact round-bar properties on the fly.
Rods are designed tension-only — ASTM A36 gross yielding (D2) vs. thread
rupture (J3.6) — with no flexure/compression/slenderness checks: the D1
L/r limit expressly does not apply to rods, self-weight rides on the
installation draw, and a compression cycle just slackens the rod while its
X partner carries the reversed load. Mixed W + rod candidate lists work.

## Layout

```
gravity_design.py                entry point: clear-span building (this README)
lateral_design.py                entry point: ASCE 7-16 lateral load basis (phases 1+2)
main.py                          entry point: conventional grid frame
src/frame_optimizer/
├── clear_span.py                ClearSpanConfig, layout derivation, geometry builder, group rules
├── config.py                    FrameConfig + shared constants (FT, M_TO_FT, group names)
├── geometry.py                  NodeInfo/MemberInfo/FrameGeometry dataclasses; grid builder
├── site.py                      SiteConfig, USGS seismic fetch + cache, wind-speed resolution
├── lateral_loads.py             wind MWFRS + seismic ELF + LRFD combos, lateral_load_basis()
├── lateral_system.py            LFRS topology: braced bays, tiers, roof bracing, portal knees
├── braced_bay.py                closed-form tension-only braced-bay statics + drift
├── lateral_designer.py          design_lateral(): sizing loop, rechecks, drift, exports
├── analysis/lateral_model.py    3D lateral load application + DAM P-Δ + drift extraction
├── analysis/frame_model.py      Pynite model build, combos (parameterizable), MemberDemand extraction
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
closure, iterative-vs-exhaustive agreement.
