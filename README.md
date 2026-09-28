# frame_optimizer

Gravity-load optimizer for fully pinned steel frames (AISC W-shapes and
square HSS).
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

**Units — SI in, SI out.** Interface units: meters, kPa (kN/m²) for surface
loads, MPa for material, millimeters for camber. Results report kN, kN·m,
kg, and mm.

## What gravity_design.py does

1. Defines a `ClearSpanConfig`: 25 m × 35 m plan footprint, 9.14 m eave,
   candidate sections per design group, roof loads.
2. Calls `optimize_layout(config)` — derives the layout from the footprint
   and returns the lightest feasible `OptimizationResult`.
3. Calls `design_uniform_baseplate(result, baseplate_config)` — the column
   baseplates follow automatically from the finalized member design
   ([src/baseplate_design/](src/baseplate_design/)): every column is designed,
   the dimensions are enveloped, and the single enveloped plate is re-checked
   against every column. The heaviest column governs bearing and plate
   flexure; the **lightest** governs anchor rod shear, because the
   shear-friction credit μ·P is what its rods do *not* have to carry.
4. Outputs (to the git-ignored `output/` directory):

| Output | Content | Consumer |
|---|---|---|
| `result.summary()` (stdout) | selected sections, weights, governing checks | humans |
| `member_checks_clear_span.csv` | one row per member, all unity checks (kN, kN·m, m) | review |
| `baseplate_inputs.json` | per-column footprint + base reactions (mm, kN) | baseplate module |
| `building_configuration.json` | full geometry + sections (mm, m, kg, kPa, MPa) | IFC authoring module |
| `baseplate_configuration.json` | the one baseplate detail: plate size + anchor rods (mm) | 3-D modeling / IFC |
| `baseplates.summary()` (stdout) | plate size, governing column per limit state | humans |
| `clear_span_wireframe.html` | interactive 3-D wireframe + baseplates (m, kN) | visual check (needs `[viz]`) |

## Tests

To run the pytests, follow the commands below:

```bash
pip install pytest
pytest tests/
```