"""Clear-span industrial building entry point.

A steel enclosure over large industrial equipment with NO interior columns.
Transverse frames (two perimeter columns + one clear-span roof member) repeat
along the building length; purlins span frame-to-frame and carry the one-way
roof deck. The roof member is a W-shape girder when one can carry the span;
when none can (roof_system="auto" proves this by statics — e.g. the 52 m
span below), each interior frame gets a parallel-chord Pratt roof truss
bearing at the top chord on the same column tops, and the final design is
re-verified with second-order (P-Delta) axial forces.

Once the member design is final, the column baseplates are designed
automatically off its base reactions: one pinned-base plate detail sized by
the governing column and applied at every column base.

Only the building footprint (span, length, eave height) is a geometric input.

All inputs and outputs are SI: meters
"""
from pathlib import Path

from baseplate_design import (BaseplateConfig, design_uniform_baseplate,
                              write_baseplate_design_json)
from frame_optimizer import (ClearSpanConfig, baseplate_inputs,
                             optimize_layout, write_baseplate_json,
                             write_building_json)

# all files produced by this script land here (git-ignored)
OUTPUT_DIR = Path(__file__).parent / "output"

# the "modeler" folder can be deleted once this module is inside Alchemy.
try:
    from modeler import visualize_result
    _viz_skip_reason = None
except ImportError as exc:
    visualize_result = None
    _viz_skip_reason = exc

config = ClearSpanConfig(
    # ------- building footprint (the primary inputs) -------
    span_m=52.0,                 # clear span: girder direction
    length_m=57.0,               # building length
    height_m=20.0,          # clearance over the equipment




    # ------- Other inputs (defaults) -------

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
        "W14X145", "W14X176", "W14X211",
    ],
    end_girder_candidates=[
        "W12X16", "W14X22", "W16X26", "W18X35", "W21X44",
    ],
    # ------- Pratt truss roof (engages automatically when no W girder works) -------
    top_chord_candidates=[
        "W12X40", "W12X53", "W12X65", "W14X61", "W14X82",
        "W14X99", "W14X109",
    ],
    bottom_chord_candidates=[
        "W12X40", "W12X53", "W12X65", "W14X61", "W14X82",
        "W14X99", "W14X109",
    ],
    truss_web_candidates=[
        "W8X24", "W8X28", "W10X33", "W10X39", "W12X40", "W12X53",
    ],

    # ------- gravity loads -------
    superimposed_dead_kpa=0.72,  # roof deck + insulation + collateral (MEP etc.)

    live_kpa=1.20,               # governing of ASCE 7 roof live (Lr) and snow

    # ------- optional design settings (defaults shown unless noted) -------
    Fy_mpa=345.0, Fu_mpa=450.0, E_mpa=200000.0,   # ASTM A992

    girder_Lb_m=None,            # None = braced at every purlin (the default)

    purlin_Lb_m=0.0,             # steel deck braces the top flange

    girder_camber_mm=25.0,       # shop camber on the interior girders

    truss_camber_mm=40.0,        # shop camber on the trusses

    check_deflection=True,       # L/360 live, L/240 tota

    enforce_slenderness_limit=True,   # KL/r <= 200 on columns
)

# ---------------------------------------------------------------------------
# Column baseplates. Designed automatically from the base reactions of the
# finalized member design above - nothing here is a geometric input; the
# column section and its loads come from the optimization.
#
# One plate detail is produced for the whole building: every column is
# designed, the dimensions are enveloped, and the single plate is re-checked
# against every column. Bearing and plate flexure are governed by the heaviest
# column; anchor rod shear by the LIGHTEST one, whose small friction credit
# leaves its rods the most to carry.
# ---------------------------------------------------------------------------
baseplate_config = BaseplateConfig(
    # ------- concrete support -------
    fc_mpa=27.6,                 # f'c (4.0 ksi normal-weight concrete)

    pedestal_width_mm=None,      # None = pier assumed to project past the
    pedestal_length_mm=None,     # plate by pedestal_edge_projection_mm; give
    pedestal_edge_projection_mm=100.0,   # both to design on a known pier

    # ------- plate and anchor materials -------
    plate_Fy_mpa=248.0,          # ASTM A36 plate (36 ksi)

    anchor_Fu_mpa=400.0,         # ASTM F1554 Gr. 36 (58 ksi)

    n_rods=4,                    # rods per base (4 per OSHA 1926.755)

    # ------- design base shear (kN per column) -------
    design_base_shear_kN=0.0,
    # The gravity model is a lateral mechanism and produces NO base shear, so
    # the rods land on their 3/4 in detailing minimum while this is 0. Set it
    # from the separate lateral (wind/seismic) system design to size them.
)

if __name__ == "__main__":
    result = optimize_layout(config, verbose=True)
    print()
    print(result.summary())

    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / "member_checks_clear_span.csv"
    result.member_table.to_csv(csv_path, index=False)
    print(f"\nFull per-member check table written to {csv_path}")

    # base reactions of the final design: one solve, written out and then
    # handed straight to the baseplate design below
    bp_inputs = baseplate_inputs(result)
    bp_path = write_baseplate_json(bp_inputs, OUTPUT_DIR / "baseplate_inputs.json")
    print(f"Baseplate design inputs (pinned base) written to {bp_path}")
    ifc_path = write_building_json(result, OUTPUT_DIR / "building_configuration.json")
    print(f"Building configuration for IFC authoring written to {ifc_path}")

    # ------- baseplate design, off the back of the member design -------
    baseplates = design_uniform_baseplate(bp_inputs, baseplate_config)
    print()
    print(baseplates.summary())
    bpd_path = write_baseplate_design_json(
        baseplates, OUTPUT_DIR / "baseplate_design.json")
    print(f"\nBaseplate design written to {bpd_path}")

    if visualize_result is not None:
        html_path = visualize_result(result, path=str(OUTPUT_DIR / "clear_span_wireframe.html"),
                                     show=True, baseplates=baseplates)
        print(f"Interactive wireframe written to {html_path}")
    else:
        print(f"(wireframe visualization skipped: {_viz_skip_reason} - "
              f"run 'pip install -e .[viz]' to enable it)")
