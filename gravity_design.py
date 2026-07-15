"""Clear-span industrial building entry point.

A steel enclosure over large industrial equipment: 20 m x 30 m plan assumed with
NO interior columns. Transverse frames (two perimeter columns + one clear-span
roof girder) repeat along the 30 m length; purlins span frame-to-frame and
carry the one-way roof deck.
"""
from frame_optimizer import (ClearSpanConfig, M_TO_FT, optimize,
                             write_baseplate_json, write_building_json)

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
    ],
    purlin_candidates=[
        "W8X10", "W10X12", "W12X14", "W12X16", "W14X22",
    ],
    column_candidates=[
        "W10X33", "W10X39", "W12X40", "W12X53", "W14X61",
    ],
    end_girder_candidates=[
        "W12X16", "W14X22", "W16X26", "W18X35", "W21X44",
    ],

    # ------- geometry (20 m x 30 m plan, no interior columns) -------
    span_ft=20.0 * M_TO_FT,      # clear span: girder direction
    length_ft=30.0 * M_TO_FT,    # building length
    n_frames=5,                  # frames @ 7.5 m; all columns on the perimeter
    eave_height_ft=30.0,         # clearance over the equipment
    purlin_spacing_ft=5.0,       # target spacing along the girder
    end_wall_columns=2,          # gable columns per end wall (exterior only —
                                 # the interior stays completely clear)

    # ------- gravity loads -------
    superimposed_dead_psf=15.0,  # roof deck + insulation + collateral (MEP etc.)
    live_psf=25.0,               # governing of ASCE 7 roof live (Lr) and snow

    # ------- optional design settings (defaults shown unless noted) -------
    Fy_ksi=50.0, Fu_ksi=65.0, E_ksi=29000.0,   # ASTM A992
    girder_Lb_ft=None,           # None = braced at every purlin (the default)
    purlin_Lb_ft=0.0,            # through-fastened deck braces the top flange
    girder_camber_in=1.0,        # shop camber on the interior girders; credited
                                 # against the total-deflection check only
    check_deflection=True,       # L/360 live, L/240 total (floor-strict values;
                                 # relax via girder_/purlin_defl_*_ratio when
                                 # roof limits, e.g. L/240 & L/180, apply)
    enforce_slenderness_limit=True,   # KL/r <= 200 on columns
)

if __name__ == "__main__":
    result = optimize(config, verbose=True)
    print()
    print(result.summary())
    result.member_table.to_csv("member_checks_clear_span.csv", index=False)
    print("\nFull per-member check table written to member_checks_clear_span.csv")

    # machine-readable handoffs to the downstream modules
    bp_path = write_baseplate_json(result, "baseplate_inputs.json")
    print(f"Baseplate design inputs (pinned base) written to {bp_path}")
    ifc_path = write_building_json(result, "building_configuration.json")
    print(f"Building configuration for IFC authoring written to {ifc_path}")

    if visualize_result is not None:
        html_path = visualize_result(result, path="clear_span_wireframe.html", show=True)
        print(f"Interactive wireframe written to {html_path}")
    else:
        print(f"(wireframe visualization skipped: {_viz_skip_reason} - "
              f"run 'pip install -e .[viz]' to enable it)")
