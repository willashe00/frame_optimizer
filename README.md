# frame_optimizer

Gravity-load optimizer for fully pinned steel frames (AISC W-shapes).
Pipeline: [Pynite](https://github.com/JWock82/Pynite) 3-D FEA → AISC 360 LRFD
checks → lightest-section search over candidate section combinations →
pinned-base column baseplates
([src/baseplate_design/](src/baseplate_design/)) off the resulting base
reactions.

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
3. Calls `design_uniform_baseplate(result, baseplate_config)` — the column
   baseplates follow automatically from the finalized member design
   ([src/baseplate_design/](src/baseplate_design/)): every column is designed,
   the dimensions are enveloped, and the single enveloped plate is re-checked
   against every column. The heaviest column governs bearing and plate
   flexure; the **lightest** governs anchor rod shear, because the
   shear-friction credit μ·P is what its rods do *not* have to carry.
4. Emits (to the git-ignored `output/` directory):

| Output | Content | Consumer |
|---|---|---|
| `result.summary()` (stdout) | selected sections, weights, governing checks | humans |
| `member_checks_clear_span.csv` | one row per member, all unity checks (kN, kN·m, m) | review |
| `baseplate_inputs.json` | per-column footprint + base reactions (mm, kN) | baseplate module |
| `building_configuration.json` | full geometry + sections (mm, m, kg, kPa, MPa) | IFC authoring module |
| `baseplate_design.json` | the one baseplate detail + per-column checks (mm, kN, MPa) | fabrication / IFC |
| `baseplates.summary()` (stdout) | plate size, governing column per limit state | humans |
| `clear_span_wireframe.html` | interactive 3-D wireframe + baseplates (m, kN) | visual check (needs `[viz]`) |

## JSON exports

[export.py](src/frame_optimizer/export.py) and
[baseplate_design/export.py](src/baseplate_design/export.py). Every numeric
key has an SI unit suffix (`_mm`, `_m`, `_kN`, `_kPa`, `_MPa`, `_kg`,
`_kg_m`). Every file carries `schema` + `schema_version`
(`building_configuration` **2** — version 1 used US customary units;
`baseplate_inputs` **3** — version 3 added `base_shear_kN`).

**`baseplate_inputs.json`** — `write_baseplate_json(result)`. One entry per
column landing on a base (includes gable columns). Per column:

- `section`: name (AISC label), `depth_d_mm`, `flange_width_bf_mm`, tf, tw,
  `area_mm2`, `nominal_weight_kg_m`
- `centerline_location`: base-node x/y/z in mm
- `axial_compression_kN`: governing LRFD value + per-combo breakdown
  (`1.4D`, `1.2D+1.6L`, and service `D`, `L`, `D+L`)
- `base_shear_kN`: horizontal reaction resultant, which is numerical **zero**
  by construction — the frame is fully pinned and its DX/DZ restraints only
  remove mechanism DOFs. Reported so that fact is visible to consumers rather
  than assumed; design shear must be stated externally.

Reactions come from one extra linear solve of the final assignment; vertical
base reaction = column axial. Compression-positive. Base condition: pinned.
Column web orientation not defined by the gravity model.

**`building_configuration.json`** — `write_building_json(result)`:

- `building`: span, length, eave height, frame count/spacing, purlin lines,
  gable columns, camber (m / mm)
- `design_groups`: selected W-shape per group with profile dimensions in mm
  (enough for a parametric IFC I-section), member count, weight (kg), max UC
- `nodes` / `members`: complete analysis topology (names, coordinates in mm,
  connectivity, group, section)
- material (MPa), loads (kPa) + combos, connection assumption, headline results

**`baseplate_design.json`** — `write_baseplate_design_json(design)`:

- `design_basis`: codes, φ factors, methodology, and the limit states
  explicitly **excluded** — read this before using the file
- `inputs`: concrete, pier, plate/rod materials, detailing minimums,
  fabrication increments (everything from `BaseplateConfig`)
- `baseplate`: the single detail applied everywhere — B × N × tp in mm *and*
  inches, bearing area, mass, rod count/diameter/edge distance, and rod
  `positions_mm` as x/y offsets from the column centerline (so a plate can be
  placed directly on the base nodes of `building_configuration.json`)
- `governing`: which column drives each limit state, its Pu/Vu, and why they
  differ
- `verification`: `all_columns_pass`, envelope DCRs, `failing_columns`
- `columns`: every column base — demands, capacities, cantilevers, the three
  DCRs, governing limit state, `PASS`

## Engineering assumptions

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
- Baseplates: pinned base, compression + shear only (AISC 360-22 + DG1 2nd
  Ed.). The plate carries no moment, and rod tension / concrete breakout
  (ACI 318 Ch. 17), welds and grout are **not** designed — the JSON lists
  every exclusion. The shear-friction credit rides on the *minimum*
  coincident compression (0.9D), never on Pu. The design shear is an input
  (`design_base_shear_kN`), not a model output, because the gravity model has
  none; while it is 0 the rods sit at their 3/4 in detailing minimum.
  In the wireframe the plates are drawn at true scale with their rods, and
  hovering one shows its dimensions and all three DCRs; the plan orientation
  there is an assumption, since the model does not fix column web direction.
- Not modeled: crane loads, hanging equipment, drifted snow, connections.

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
├── export.py                    baseplate-inputs + building-configuration JSON writers
├── results.py                   OptimizationResult + summary()
└── sections/                    W-shape catalog: CSV + WShape loader
src/baseplate_design/            pinned-base baseplates, off the back of the above
├── config.py                    BaseplateConfig (SI in, kip/inch internally)
├── baseplate_design.py          AISC 360 / DG1 design + check of ONE plate
├── uniform_design.py            governing column -> one plate for every column
├── export.py                    baseplate-design JSON writer
└── __main__.py                  `python -m baseplate_design [baseplate_inputs.json]`
modeler/                         plotly wireframe + baseplates (optional, delete-able)
tests/                           hand-calc, AISC Manual anchors, regression
```

## Tests

To run the pytests, follow the commands below:

```bash
pip install pytest
pytest tests/
```