"""Lateral design entry point: runs off the back of gravity_design.py.

Full pipeline (phases 1-5): resolve the site hazards from geographic
location, run the gravity optimization for the SAME building (the config is
imported from gravity_design.py — nothing is re-entered), build the lateral
force-resisting system, size it, and re-certify every gravity member under
the complete ASCE 7-16 wind + seismic combination set, including drift
checks and the wind-uplift force-reversal checks.

Outputs (all in ./output):
  member_checks_lateral.csv      every member, every unity check, all combos
  lateral_load_basis.json        the ASCE 7-16 load derivation
  lateral_baseplate_inputs.json  anchorage demands incl. net uplift + braced-
                                 bay shear (supersedes the gravity baseplate
                                 file for foundation design)
  building_configuration_lateral.json  full building incl. LFRS for IFC
  lateral_wireframe.html         interactive wireframe with the braces

Site inputs: latitude/longitude drive the seismic parameters via the USGS
web service (cached in output/hazards_cache.json). The basic wind speed has
no free official API — look it up once at https://ascehazardtool.org/ for
the site and risk category and record it below.
"""

from pathlib import Path

from frame_optimizer import (LateralSystemConfig, SiteConfig, design_lateral,
                             optimize_layout, resolve_site_hazards,
                             write_lateral_baseplate_json)
from frame_optimizer.export import building_configuration
from frame_optimizer.lateral_designer import lateral_system_block
from gravity_design import config   # the same building; run-code is guarded

try:
    from modeler import visualize_result
except ImportError:   # plotly not installed: pip install -e .[viz]
    visualize_result = None

OUTPUT_DIR = Path(__file__).parent / "output"

site = SiteConfig(
    # ------- geographic location (configure these ONLY) -------
    latitude=39.96,               # example: Columbus, OH — replace with the site
    longitude=-83.00,

    # ------- code defaults (override when the project says otherwise) -------
    risk_category="II",
    site_class="D",               # ASCE 7-16 11.4.3 default when soil unknown
    exposure="C",                 # open terrain, typical industrial site
    basic_wind_speed_mph=107.0,   # ascehazardtool.org value for the example
                                  # coordinates (Risk II) — ALWAYS look up
                                  # your own site; there is no API for wind
    cache_path=OUTPUT_DIR / "hazards_cache.json",
)

lateral_config = LateralSystemConfig(
    # ------- candidate sections for the LFRS groups -------
    # X diagonals: threaded A36 rods — the as-built reality for industrial
    # roof/wall X bracing. Designed tension-only per AISC D2 (gross yield)
    # and J3.6 (thread rupture); the D1 L/r limit does not apply to rods,
    # and self-weight rides on installation draw (sag), not member flexure.
    # Struts stay rolled W (they carry compression + their own weight).
    brace_candidates=["ROD3/4", "ROD7/8", "ROD1", "ROD1-1/8", "ROD1-1/4",
                      "ROD1-3/8", "ROD1-1/2", "ROD1-3/4", "ROD2"],
    roof_brace_candidates=["ROD3/4", "ROD7/8", "ROD1", "ROD1-1/8",
                           "ROD1-1/4", "ROD1-3/8", "ROD1-1/2"],
    eave_strut_candidates=["W8X10", "W10X12", "W12X14", "W12X16", "W14X22",
                           "W16X26", "W18X35", "W21X44", "W24X76"],
    wall_strut_candidates=["W8X24", "W10X33", "W12X40", "W12X53"],
    # braced bays / tiers / roof panels derive from the footprint
)

def run_lateral_design(gravity, show_wireframe: bool = True):
    """Everything downstream of the gravity optimization: resolve hazards,
    design + certify the lateral system, write all exports, and render the
    FULL wireframe (gravity members + LFRS). Called automatically at the
    end of a gravity_design.py run, and by this script's own __main__.
    Returns the LateralDesignResult."""
    import json

    OUTPUT_DIR.mkdir(exist_ok=True)

    hazards = resolve_site_hazards(site)
    result = design_lateral(
        gravity, hazards, lateral_config,
        wall_weight_psf=3.0,
    )
    print(result.summary())

    csv_path = OUTPUT_DIR / "member_checks_lateral.csv"
    result.member_table.to_csv(csv_path, index=False)
    result.drift_table.to_csv(OUTPUT_DIR / "drift_checks_lateral.csv",
                              index=False)
    (OUTPUT_DIR / "lateral_load_basis.json").write_text(
        json.dumps(result.basis, indent=2) + "\n", encoding="utf-8")
    write_lateral_baseplate_json(
        result, OUTPUT_DIR / "lateral_baseplate_inputs.json")
    combined = result.as_optimization_result()
    building = building_configuration(
        combined, geometry=result.system.geometry,
        lateral_block=lateral_system_block(result))
    (OUTPUT_DIR / "building_configuration_lateral.json").write_text(
        json.dumps(building, indent=2) + "\n", encoding="utf-8")
    files = [csv_path.name, "drift_checks_lateral.csv",
             "lateral_load_basis.json", "lateral_baseplate_inputs.json",
             "building_configuration_lateral.json"]

    if visualize_result is not None:
        visualize_result(
            combined, path=str(OUTPUT_DIR / "lateral_wireframe.html"),
            geometry=result.system.geometry, show=show_wireframe)
        files.append("lateral_wireframe.html (opened)")
    print(f"Lateral outputs -> {OUTPUT_DIR}\\{{{', '.join(files)}}}")

    return result


if __name__ == "__main__":
    print("[1/2] Gravity design: searching layouts, sizing members...")
    gravity = optimize_layout(config)
    print(gravity.summary())
    print("\n[2/2] Lateral design: ASCE 7-16 wind + seismic, LFRS sizing, "
          "member re-certification...")
    run_lateral_design(gravity)
