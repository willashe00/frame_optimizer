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
    _viz_skip_reason = None
except ImportError as exc:
    visualize_result = None
    _viz_skip_reason = exc

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
    ],
    end_girder_candidates=[
        # giving the end girders their own (lighter) group lets the layout
        # search consider gable columns on the two end walls
        "W12X16", "W14X22", "W16X26", "W18X35", "W21X44",
    ],


    # ------- building footprint (configure these ONLY) -------
    span_ft=25.0 * M_TO_FT,      # clear span: girder direction
    length_ft=35.0 * M_TO_FT,    # building length
    eave_height_ft=30.0,         # clearance over the equipment


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
    result = optimize_layout(config, verbose=True)
    print()
    print(result.summary())

    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / "member_checks_clear_span.csv"
    result.member_table.to_csv(csv_path, index=False)
    print(f"\nFull per-member check table written to {csv_path}")

    bp_path = write_baseplate_json(result, OUTPUT_DIR / "baseplate_inputs.json")
    print(f"Baseplate design inputs (pinned base) written to {bp_path}")
    ifc_path = write_building_json(result, OUTPUT_DIR / "building_configuration.json")
    print(f"Building configuration for IFC authoring written to {ifc_path}")

    if visualize_result is not None:
        html_path = visualize_result(result, path=str(OUTPUT_DIR / "clear_span_wireframe.html"),
                                     show=True)
        print(f"Interactive wireframe written to {html_path}")
    else:
        print(f"(wireframe visualization skipped: {_viz_skip_reason} - "
              f"run 'pip install -e .[viz]' to enable it)")
