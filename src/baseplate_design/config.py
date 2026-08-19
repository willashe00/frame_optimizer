"""User-facing configuration for a baseplate design run.

Interface units are SI, matching `frame_optimizer`: millimeters for lengths,
MPa for material strengths, kN for forces. The design math itself runs in the
AISC-native consistent kip/inch system (AISC 360 and Design Guide 1 are
written in it, and plate stock / anchor rods are supplied in imperial
increments); conversion happens exactly once, at this config boundary, using
the exact factors imported from `frame_optimizer.config` so the two modules
can never drift apart.

Plan-direction naming (a plate lies flat, so both are plan dimensions):

    B = plate width,  parallel to the column flange width bf
    N = plate length, parallel to the column depth d
"""
from __future__ import annotations

from dataclasses import dataclass

from frame_optimizer.config import IN_TO_MM, KIP_TO_KN, MM_TO_IN, MPA_TO_KSI

# --- SI <-> internal kip/inch, derived from frame_optimizer's exact factors ---
KN_TO_KIP = 1.0 / KIP_TO_KN
KSI_TO_MPA = 1.0 / MPA_TO_KSI

# Imperial fabrication increments, exact in mm. Plate stock is ordered in
# 1/2 in plan increments and 1/8 in thicknesses; anchor rods come in 1/8 in
# diameter steps, with 3/4 in the smallest size used on a column base.
_HALF_IN_MM = 0.5 * IN_TO_MM        # 12.7
_EIGHTH_IN_MM = 0.125 * IN_TO_MM    # 3.175
_THREE_QTR_IN_MM = 0.75 * IN_TO_MM  # 19.05
_ONE_IN_MM = IN_TO_MM               # 25.4
_ONE_HALF_IN_MM = 1.5 * IN_TO_MM    # 38.1


@dataclass
class BaseplateConfig:
    """Everything the baseplate design needs that the frame model does not know.

    The frame model supplies the column section and its base reactions; this
    config supplies the concrete, the anchorage, and the fabrication rules.
    Every default is a common industrial-building value, annotated with its
    imperial equivalent since the governing code is written in imperial.
    """

    # --- concrete support (MPa / mm) ---
    fc_mpa: float = 27.6             # f'c, 4.0 ksi normal-weight concrete
    pedestal_width_mm: float | None = None   # pier dimension parallel to B
    pedestal_length_mm: float | None = None  # pier dimension parallel to N
    pedestal_edge_projection_mm: float = 100.0
    # When either pedestal dimension is None the pier is assumed to project
    # this far past the plate on all four sides (~4 in), which is what sets
    # the AISC J8 confinement factor sqrt(A2/A1). Give both pedestal
    # dimensions explicitly to design against a known pier instead.

    # --- plate material (MPa) ---
    plate_Fy_mpa: float = 248.0      # ASTM A36 plate, 36 ksi

    # --- anchor rods ---
    anchor_Fu_mpa: float = 400.0     # ASTM F1554 Gr. 36, 58 ksi
    n_rods: int = 4                  # rods per column base (4 per OSHA 1926.755)
    friction_coefficient: float = 0.40
    # mu for shear friction, ACI 318-19 Table 22.9.4.2 (concrete placed
    # against hardened concrete, not intentionally roughened)
    friction_axial_factor: float = 0.9
    # The friction credit mu*P must use the MINIMUM compression coincident
    # with the shear, not the maximum: 0.9 is the LRFD dead-load factor of
    # the 0.9D + lateral combination. Set to 0.0 to ignore friction entirely
    # and carry all shear on the rods.

    # --- design base shear (kN per column) ---
    design_base_shear_kN: float = 0.0
    # The gravity model is a lateral mechanism and produces no base shear, so
    # any shear the rods must carry has to be stated here (it comes from the
    # separate lateral system design). The larger of this value and the shear
    # actually reported by the frame model is used.

    # --- plate / anchorage detailing (mm) ---
    min_edge_distance_mm: float = _ONE_HALF_IN_MM      # 1.5 in, rod CL to plate edge
    rod_clearance_mm: float = _ONE_HALF_IN_MM          # 1.5 in, flange face to rod CL
    min_plate_projection_mm: float = _ONE_IN_MM        # 1 in of plate past the column
    edge_distance_rod_factor: float = 1.5
    # Edge distance is also held to this multiple of the rod diameter.

    # --- fabrication increments (mm) ---
    plate_increment_mm: float = _HALF_IN_MM            # B, N round up to 1/2 in
    thickness_increment_mm: float = _EIGHTH_IN_MM      # tp rounds up to 1/8 in
    min_thickness_mm: float = _HALF_IN_MM              # 1/2 in minimum plate
    rod_increment_mm: float = _EIGHTH_IN_MM            # rod dia rounds up to 1/8 in
    min_rod_diameter_mm: float = _THREE_QTR_IN_MM      # 3/4 in minimum rod

    def __post_init__(self) -> None:
        if self.fc_mpa <= 0 or self.plate_Fy_mpa <= 0 or self.anchor_Fu_mpa <= 0:
            raise ValueError("fc_mpa, plate_Fy_mpa and anchor_Fu_mpa must be positive.")
        if self.n_rods < 2:
            raise ValueError("n_rods must be >= 2 (4 is the OSHA minimum for a column).")
        if self.n_rods % 2:
            raise ValueError(
                "n_rods must be even: rods are placed in a symmetric pattern, "
                "half outside each column flange."
            )
        if not 0.0 <= self.friction_coefficient <= 1.0:
            raise ValueError("friction_coefficient must be between 0 and 1.")
        if not 0.0 <= self.friction_axial_factor <= 1.0:
            raise ValueError("friction_axial_factor must be between 0 and 1.")
        if self.design_base_shear_kN < 0:
            raise ValueError("design_base_shear_kN must be >= 0.")
        if min(self.min_edge_distance_mm, self.rod_clearance_mm,
               self.min_plate_projection_mm, self.edge_distance_rod_factor) < 0:
            raise ValueError("Detailing dimensions must be >= 0.")
        if min(self.plate_increment_mm, self.thickness_increment_mm,
               self.rod_increment_mm) <= 0:
            raise ValueError("Fabrication increments must be positive.")
        if self.min_thickness_mm <= 0 or self.min_rod_diameter_mm <= 0:
            raise ValueError("min_thickness_mm and min_rod_diameter_mm must be positive.")
        pedestal = (self.pedestal_width_mm, self.pedestal_length_mm)
        if any(p is not None and p <= 0 for p in pedestal):
            raise ValueError("Pedestal dimensions must be positive when given.")
        if self.pedestal_edge_projection_mm < 0:
            raise ValueError("pedestal_edge_projection_mm must be >= 0.")

    @property
    def has_explicit_pedestal(self) -> bool:
        return self.pedestal_width_mm is not None and self.pedestal_length_mm is not None

    def describe(self) -> list[str]:
        """Human-readable configuration lines for the design summary."""
        if self.has_explicit_pedestal:
            pier = (f"{self.pedestal_width_mm:.0f} x {self.pedestal_length_mm:.0f} mm "
                    "pier (given)")
        else:
            pier = (f"pier projecting {self.pedestal_edge_projection_mm:.0f} mm "
                    "past the plate (assumed)")
        return [
            f"Concrete: f'c = {self.fc_mpa:.1f} MPa on a {pier}",
            f"Plate:    Fy = {self.plate_Fy_mpa:.0f} MPa",
            f"Anchors:  {self.n_rods} rods, Fu = {self.anchor_Fu_mpa:.0f} MPa, "
            f"friction mu = {self.friction_coefficient:.2f} on "
            f"{self.friction_axial_factor:.1f}D",
            f"Shear:    design base shear = {self.design_base_shear_kN:.1f} kN/column",
        ]

    # --- internal (kip/inch) views of the SI inputs -----------------------
    @property
    def fc_ksi(self) -> float:
        return self.fc_mpa * MPA_TO_KSI

    @property
    def plate_Fy_ksi(self) -> float:
        return self.plate_Fy_mpa * MPA_TO_KSI

    @property
    def anchor_Fu_ksi(self) -> float:
        return self.anchor_Fu_mpa * MPA_TO_KSI

    @property
    def design_base_shear_kip(self) -> float:
        return self.design_base_shear_kN * KN_TO_KIP

    @property
    def min_edge_distance_in(self) -> float:
        return self.min_edge_distance_mm * MM_TO_IN

    @property
    def rod_clearance_in(self) -> float:
        return self.rod_clearance_mm * MM_TO_IN

    @property
    def min_plate_projection_in(self) -> float:
        return self.min_plate_projection_mm * MM_TO_IN

    @property
    def pedestal_edge_projection_in(self) -> float:
        return self.pedestal_edge_projection_mm * MM_TO_IN

    @property
    def pedestal_in(self) -> tuple[float, float] | None:
        """(width parallel to B, length parallel to N) in inches, or None."""
        if not self.has_explicit_pedestal:
            return None
        return (self.pedestal_width_mm * MM_TO_IN, self.pedestal_length_mm * MM_TO_IN)

    @property
    def plate_increment_in(self) -> float:
        return self.plate_increment_mm * MM_TO_IN

    @property
    def thickness_increment_in(self) -> float:
        return self.thickness_increment_mm * MM_TO_IN

    @property
    def min_thickness_in(self) -> float:
        return self.min_thickness_mm * MM_TO_IN

    @property
    def rod_increment_in(self) -> float:
        return self.rod_increment_mm * MM_TO_IN

    @property
    def min_rod_diameter_in(self) -> float:
        return self.min_rod_diameter_mm * MM_TO_IN
