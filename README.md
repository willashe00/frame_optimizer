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
`length_ft`, `eave_height_ft`. The layout — `n_frames`, `purlin_spacing_ft`,
`end_wall_columns` — is **derived**, not user-specified: `optimize_layout()`
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
No lateral shear — out of model scope. Column web orientation not defined by
the gravity model.

**`building_configuration.json`** — `write_building_json(result)`:

- `building`: span, length, eave height, frame count/spacing, purlin lines,
  gable columns, camber
- `design_groups`: selected W-shape per group with profile dimensions
  (enough for a parametric IFC I-section), member count, weight, max UC
- `nodes` / `members`: complete analysis topology (names, coordinates,
  connectivity, group, section)
- material, loads + combos, connection assumption, headline results

## Engineering assumptions (must-read)

- **Gravity only. Fully pinned.** The frame is a lateral mechanism; nodes are
  restrained in DX/DZ/rotations purely to remove mechanism DOFs. Valid only
  because those restraints attract no force under gravity. **Never add
  lateral loads to this model.** Wind/seismic need a separate system —
  a tall single-story shell is usually wind-governed.
- Purlins are explicit pin-ended members; they deliver true point reactions
  to the girders at shared nodes. Girders are Pynite physical members:
  subdivided at purlin nodes, checked over the full span, self-weight only
  as direct load. Purlin nodes get free rotations (the continuous girder
  stabilizes them) — clamping them would falsify girder bending.
  Statics close exactly (tested to 0.1%).
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

## Section database

[sections/data/aisc_w_shapes.csv](src/frame_optimizer/sections/) — 283
W-shapes from AISC Shapes Database v15.0 (US units). `rts`, `ho` computed
from their exact definitions. Regenerate with
[tools/prepare_sections_csv.py](tools/prepare_sections_csv.py).

## Layout

```
gravity_design.py                entry point: clear-span building (this README)
main.py                          entry point: conventional grid frame
src/frame_optimizer/
├── clear_span.py                ClearSpanConfig, layout derivation, geometry builder, group rules
├── config.py                    FrameConfig + shared constants (FT, M_TO_FT, group names)
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
closure, iterative-vs-exhaustive agreement.
