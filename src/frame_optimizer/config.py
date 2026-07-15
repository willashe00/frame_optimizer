"""User-facing configuration for a frame optimization run.

Interface units are feet and psf; everything downstream of FrameConfig works
in kips and inches.

Coordinate system: X and Z are the plan directions, Y is vertical (gravity
acts in -Y).
"""
from __future__ import annotations

from dataclasses import dataclass, field

FT = 12.0                      # in per ft
M_TO_FT = 1000.0 / 25.4 / 12.0  # ft per meter (exact), for metric plan inputs
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

    # --- geometry ---
    x_bays: int = 1
    x_bay_spacing_ft: float = 30.0
    z_bays: int = 1
    z_bay_spacing_ft: float = 30.0
    stories: int = 1
    story_height_ft: float | list[float] = 13.0  # scalar or one value per story

    # --- loads ---
    superimposed_dead_psf: float = 0.0
    live_psf: float = 0.0
    deck_span_direction: str = "z"   # deck spans in this direction; the members
                                     # running PERPENDICULAR to it are the loaded beams

    # --- material (default ASTM A992) ---
    Fy_ksi: float = 50.0
    Fu_ksi: float = 65.0
    E_ksi: float = 29000.0

    # --- design options ---
    beam_Lb_ft: float | None = None   # unbraced length of loaded beams;
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
        if self.x_bay_spacing_ft <= 0 or self.z_bay_spacing_ft <= 0:
            raise ValueError("Bay spacings must be positive.")
        if self.deck_span_direction not in ("x", "z"):
            raise ValueError("deck_span_direction must be 'x' or 'z'.")
        if self.superimposed_dead_psf < 0 or self.live_psf < 0:
            raise ValueError("Loads must be non-negative.")
        if self.infill_beams_per_bay != 0:
            raise NotImplementedError(
                "infill_beams_per_bay > 0 (point loads on supporting beams) "
                "is not implemented in v1."
            )
        heights = self.story_heights_ft
        if len(heights) != self.stories or any(h <= 0 for h in heights):
            raise ValueError(
                "story_height_ft must be a positive scalar or a list of "
                f"{self.stories} positive values."
            )

    def describe(self) -> list[str]:
        """Human-readable configuration lines for result summaries."""
        return [
            f"Frame:  {self.x_bays} x-bay(s) @ {self.x_bay_spacing_ft} ft  x  "
            f"{self.z_bays} z-bay(s) @ {self.z_bay_spacing_ft} ft,  "
            f"{self.stories} story(ies), deck spans '{self.deck_span_direction}'",
            f"Loads:  SDL = {self.superimposed_dead_psf} psf, LL = {self.live_psf} psf "
            f"(1.4D, 1.2D+1.6L) + self-weight",
        ]

    @property
    def candidates_by_group(self) -> dict[str, list[str]]:
        """Candidate section labels per design group. Key order sets the
        reporting order in results and the wireframe legend."""
        return {COLUMN: self.column_candidates, BEAM: self.beam_candidates}

    @property
    def story_heights_ft(self) -> list[float]:
        if isinstance(self.story_height_ft, (int, float)):
            return [float(self.story_height_ft)] * self.stories
        return [float(h) for h in self.story_height_ft]
