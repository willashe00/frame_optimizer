"""Clear-span industrial building entry point.

A steel enclosure over large industrial equipment: 20 m x 30 m plan assumed with
NO interior columns. Transverse frames (two perimeter columns + one clear-span
roof girder) repeat along the building length; purlins span frame-to-frame and
carry the one-way roof deck.

Only the building footprint (span, length, eave height) is a geometric input.
The layout — frame count, purlin spacing, and gable columns per end wall — is
determined by optimize_layout(): it searches the realistic layout band for the
footprint (bays ~20-30 ft, purlins ~4-6 ft, end-girder segments <= ~25 ft) and
keeps the lightest feasible design. A footprint no longer than one bay
naturally collapses to a minimal 1x1-bay enclosure (2 frames, no gables).

The girder system is chosen automatically too: rolled W girders whenever any
candidate W-shape can carry the clear span, parallel-chord Pratt truss girders
(built from the truss chord/web candidates below) when none can — which is
what happens for clear spans beyond roughly 90-100 ft.

Running this script executes the FULL pipeline: once the gravity design
converges, lateral_design.py is invoked automatically (site location and
wind speed are configured there) — it sizes the lateral force-resisting
system, re-certifies every member under the ASCE 7-16 wind + seismic
combinations, and finishes by rendering the full-building wireframe.
"""
from pathlib import Path

from frame_optimizer import (ClearSpanConfig, M_TO_FT, optimize_layout,
                             write_baseplate_json, write_building_json)

# all files produced by this script land here (git-ignored)
OUTPUT_DIR = Path(__file__).parent / "output"

# the "modeler" folder can be deleted once this module is inside Alchemy.
# It also needs plotly, which the core package deliberately does not:
# install with  pip install -e .[viz]  (or  pip install plotly)
try:
    from modeler import visualize_result
except ImportError:   # plotly not installed: pip install -e .[viz]
    visualize_result = None

config = ClearSpanConfig(
    # ------- candidate W-shapes (AISC Manual labels) -------
    girder_candidates=[
        "W24X76", "W27X84", "W30X90", "W30X99", "W30X108",
        "W30X116", "W33X118", "W33X130", "W36X135", "W40X149",
        "W40X167", "W44X230"
    ],
    purlin_candidates=[
        "W8X10", "W10X12", "W12X14", "W12X16", "W14X22",
    ],
    column_candidates=[
        "W10X33", "W10X39", "W12X40", "W12X53", "W14X61",
        "W12X65", "W14X90", "W14X109",
        "W14X120", "W14X132", "W14X145", "W14X159", "W14X176",
        "W14X193", "W14X211", "W14X233", "W14X257", "W14X283",
        # headroom for the lateral phase: transverse wind drift at the
        # 55-ft eave governs the columns, one notch past the strength pick
        "W14X311", "W14X342",
    ],
    end_girder_candidates=[
        "W12X16", "W14X22", "W16X26", "W18X35", "W21X44",
    ],
    # ------- truss-girder fallback candidates (chords + webs) -------
    truss_chord_candidates=[
        "W10X33", "W12X40", "W12X53", "W14X61", "W14X82", "W14X109",
        "W14X120", "W14X132",   # drift-escalation headroom (lateral phase)
    ],
    truss_web_candidates=[
        "W8X10", "W8X24", "W10X33", "W12X40",
    ],

    # ------- building footprint (configure these ONLY) -------
    span_ft=40.0 * M_TO_FT,      # clear span: girder direction
    length_ft=50.0 * M_TO_FT,    # building length
    eave_height_ft=55.0,         # clearance over the equipment



    # ------- Other inputs (defaults) -------

    # ------- gravity loads -------
    superimposed_dead_psf=15.0,  # roof deck + insulation + collateral (MEP etc.)
    live_psf=25.0,               # governing of ASCE 7 roof live (Lr) and snow

    # ------- optional design settings (defaults shown unless noted) -------
    Fy_ksi=50.0, Fu_ksi=65.0, E_ksi=29000.0,   # ASTM A992
    girder_Lb_ft=None,           # None = braced at every purlin (the default)
    purlin_Lb_ft=0.0,            # steel deck braces the top flange
    girder_camber_in=1.0,        # shop camber on the interior girders; credited
                                 # against the total-deflection check only
    check_deflection=True,       # L/360 live, L/240 total (floor-strict values;
                                 # relax via girder_/purlin_defl_*_ratio when
                                 # roof limits, e.g. L/240 & L/180, apply)
    enforce_slenderness_limit=True,   # KL/r <= 200 on columns
)

if __name__ == "__main__":
    print("[1/2] Gravity design: searching layouts, sizing members...")
    result = optimize_layout(config)
    print(result.summary())

    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / "member_checks_clear_span.csv"
    result.member_table.to_csv(csv_path, index=False)
    write_baseplate_json(result, OUTPUT_DIR / "baseplate_inputs.json")
    write_building_json(result, OUTPUT_DIR / "building_configuration.json")
    files = [csv_path.name, "baseplate_inputs.json", "building_configuration.json"]
    if visualize_result is not None:
        # written for reference but not opened: the FULL wireframe (gravity
        # + lateral system) renders at the end of the lateral design below
        visualize_result(result, path=str(OUTPUT_DIR / "clear_span_wireframe.html"),
                         show=False)
        files.append("clear_span_wireframe.html")
    print(f"Gravity outputs -> {OUTPUT_DIR}\\{{{', '.join(files)}}}")

    # ------- lateral design runs automatically off this gravity result -------
    # (site location + wind speed are configured in lateral_design.py; it
    # sizes the LFRS, re-certifies every member under the ASCE 7-16 wind +
    # seismic combos, writes the lateral exports, and opens the full-building
    # wireframe. Imported here, not at the top, so the module import in the
    # layout search's worker processes stays lightweight.)
    print("\n[2/2] Lateral design: ASCE 7-16 wind + seismic, LFRS sizing, "
          "member re-certification...")
    from lateral_design import run_lateral_design
    run_lateral_design(result)
