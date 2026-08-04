"""User-facing configuration for a frame optimization run.

Interface units are SI: meters, kPa (kN/m^2) for surface loads, MPa for
material properties, and millimeters for camber. Everything downstream of the
config works in the AISC-native consistent unit system (kips and inches);
conversion happens exactly once, at the config boundary, using the exact
conversion factors below. Results and exports are reported back in SI
(kN, kN*m, mm, kg).

Coordinate system: X and Z are the plan directions, Y is vertical (gravity
acts in -Y).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --- exact unit conversions (interface SI <-> internal kip/inch) ---
M_TO_IN = 1000.0 / 25.4          # inches per meter (exact)
MM_TO_IN = 1.0 / 25.4            # inches per millimeter (exact)
IN_TO_M = 0.0254                 # meters per inch (exact)
IN_TO_MM = 25.4                  # millimeters per inch (exact)
FT_TO_M = 0.3048                 # meters per foot (exact)
MPA_TO_KSI = 1.0 / 6.894757293168361   # 1 ksi = 6.894757... MPa (exact chain)
KPA_TO_KSI = MPA_TO_KSI / 1000.0
KIP_TO_KN = 4.4482216152605      # 1 kip = 4.448221... kN (exact)
KIP_IN_TO_KNM = KIP_TO_KN * IN_TO_M   # kip-in -> kN*m
LB_TO_KG = 0.45359237            # kilograms per pound (exact)
PLF_TO_KG_M = LB_TO_KG / FT_TO_M      # lb/ft -> kg/m (section nominal weight)

# --- internal (backend) constants; the section catalog is US customary ---
FT = 12.0                        # inches per foot
PLF_TO_KIP_PER_IN = 1.0 / (1000.0 * 12.0)

# Design-group names for the conventional grid frame. A design group is a set
# of members that share one section and one set of design rules; geometry
# builders may define additional groups (e.g. 'girder', 'purlin').
COLUMN = "column"
BEAM = "beam"


@dataclass
class FrameConfig:
    # --- candidate sections (AISC Manual labels) ---
    # beam_candidates covers every horizontal floor member: one shared size,
    # no beam/girder distinction.
    beam_candidates: list[str]
    column_candidates: list[str]

    # --- geometry (m) ---
    x_bays: int = 1
    x_bay_spacing_m: float = 9.0
    z_bays: int = 1
    z_bay_spacing_m: float = 9.0
    stories: int = 1
    story_height_m: float | list[float] = 4.0  # scalar or one value per story

    # --- loads (kPa = kN/m^2 over the floor plan) ---
    superimposed_dead_kpa: float = 0.0
    live_kpa: float = 0.0
    deck_span_direction: str = "z"   # deck spans in this direction; the members
                                     # running PERPENDICULAR to it are the loaded beams

    # --- material, MPa (default ASTM A992: Fy 345, Fu 450, E 200 GPa) ---
    Fy_mpa: float = 345.0
    Fu_mpa: float = 450.0
    E_mpa: float = 200000.0

    # --- design options ---
    beam_Lb_m: float | None = None    # unbraced length of loaded beams;
                                      # None -> full span (conservative, no deck bracing)
    check_deflection: bool = True
    defl_live_ratio: float = 360.0    # live-load limit = span / this
    defl_total_ratio: float = 240.0   # total-load limit = span / this
    enforce_slenderness_limit: bool = True   # KL/r <= 200 for compression members

    # --- future feature placeholder ---
    infill_beams_per_bay: int = 0

    def __post_init__(self) -> None:
        if not self.beam_candidates or not self.column_candidates:
            raise ValueError("beam_candidates and column_candidates must be non-empty.")
        if self.x_bays < 1 or self.z_bays < 1 or self.stories < 1:
            raise ValueError("x_bays, z_bays, and stories must all be >= 1.")
        if self.x_bay_spacing_m <= 0 or self.z_bay_spacing_m <= 0:
            raise ValueError("Bay spacings must be positive.")
        if self.deck_span_direction not in ("x", "z"):
            raise ValueError("deck_span_direction must be 'x' or 'z'.")
        if self.superimposed_dead_kpa < 0 or self.live_kpa < 0:
            raise ValueError("Loads must be non-negative.")
        if self.infill_beams_per_bay != 0:
            raise NotImplementedError(
                "infill_beams_per_bay > 0 (point loads on supporting beams) "
                "is not implemented in v1."
            )
        heights = self.story_heights_m
        if len(heights) != self.stories or any(h <= 0 for h in heights):
            raise ValueError(
                "story_height_m must be a positive scalar or a list of "
                f"{self.stories} positive values."
            )

    def describe(self) -> list[str]:
        """Human-readable configuration lines for result summaries."""
        return [
            f"Frame:  {self.x_bays} x-bay(s) @ {self.x_bay_spacing_m} m  x  "
            f"{self.z_bays} z-bay(s) @ {self.z_bay_spacing_m} m,  "
            f"{self.stories} story(ies), deck spans '{self.deck_span_direction}'",
            f"Loads:  SDL = {self.superimposed_dead_kpa} kPa, LL = {self.live_kpa} kPa "
            f"(1.4D, 1.2D+1.6L) + self-weight",
        ]

    @property
    def candidates_by_group(self) -> dict[str, list[str]]:
        """Candidate section labels per design group. Key order sets the
        reporting order in results and the wireframe legend."""
        return {COLUMN: self.column_candidates, BEAM: self.beam_candidates}

    @property
    def story_heights_m(self) -> list[float]:
        if isinstance(self.story_height_m, (int, float)):
            return [float(self.story_height_m)] * self.stories
        return [float(h) for h in self.story_height_m]
