"""Lateral force-resisting system (LFRS) topology for the clear-span
building: which members resist the ASCE 7 wind/seismic loads of
lateral_loads.py, and where they go.

System concept (standard practice for single-story industrial steel
buildings; see also the plan in lateral_loads.py):

* TRANSVERSE (x, across the clear span): the frames themselves — bracing
  would obstruct the clear span. Truss-girder frames are already rigid
  mill-type bents. W-girder frames are converted to PORTAL frames here:
  the girder-to-column knees become moment connections (MemberInfo
  fixed_i/fixed_j; bases stay pinned) and the config copy gets
  transverse_moment_frame=True so the analysis layer applies the same AISC
  360 Chapter C Direct Analysis Method it already uses for truss bents
  (0.8E, notional loads, P-Delta, K = 1).
* LONGITUDINAL (z, along the building): tension-only X-braced bays in both
  sidewalls + collector struts. The two eave purlin lines are re-tagged as
  an 'eave_strut' design group — they keep their gravity tributary load
  and, in the lateral phases, additionally drag the roof-level longitudinal
  force to the braced bays (the classic eave-strut double duty).
* ROOF: X-bracing in the two end bays forms a horizontal truss spanning
  the building width; it collects the end-wall wind at roof level and
  delivers it to the sidewall eave struts -> braced bays. Diagonals connect
  purlin-line nodes chosen so the plan panels are roughly square (~45 deg
  diagonals) instead of one flat diagonal per purlin space.

Practice bands enforced by the derivations (all overridable on the
LateralSystemConfig, which is intended for tests/validation — normal use
derives everything from the footprint, mirroring the gravity layout):

* Braced bays sit in BOTH end bays (shortest drag path from the end-wall
  wind) and, for long buildings, at additional interior bays so that no
  unbraced run exceeds MAX_BAYS_BETWEEN_BRACED bays. The same bays are
  braced in both sidewalls — an asymmetric layout would twist the building.
* Wall X-bracing is TIERED: tall eaves over ordinary bay widths would give
  a single X an uselessly steep diagonal, so the wall is stacked into
  panels whose diagonals stay inside BRACE_ANGLE_MIN/MAX (30-60 deg, ~45
  target), with a horizontal wall strut closing every intermediate tier
  (the X corners need a compression/tension chord to complete the truss).
  Braces run from the column bases to the COLUMN TOP (the roof plane: the
  eave for W-girder buildings, the top-chord level for trusses).
* Roof-brace plan panels group ~s_f/purlin_spacing purlin spaces so the
  diagonal angle lands near 45 deg.

Analysis-model notes (what makes the augmented geometry sound in Pynite):

* Every non-base node becomes free_dx: the transverse frames are now ALL
  rigid (portal or truss bent) and supply their own in-plane stiffness —
  the same DOF scheme the truss bents already use.
* W-girder knee nodes (NE*) get free_rotations: the moment connection
  transfers rotation into the columns; clamping would falsify the frame.
* New tier nodes subdivide the (physical) columns exactly like purlin
  nodes subdivide girders: free_rotations, stiffness from the continuous
  column through them.
* Braces/struts are pin-ended members with zero tributary width: under
  gravity they carry only self-weight (the blanket DZ restraints keep the
  longitudinal direction dormant until the lateral load phase, which
  analyzes the braced-bay planes; X-brace gravity pickup from column
  shortening is conventionally neglected). Tension-only idealization of
  the wall/roof X pairs is a DESIGN rule applied in the sizing phase — one
  diagonal of each pair acts per load direction.
* The optimizer's representative-strip analysis must never see this
  geometry (braced bays make the building non-uniform along its length);
  it is built directly, not via geometry_for(), so the strip stays a
  gravity-only device.

v1 scope notes: braces come from the W-shape catalog (rods/angles/HSS are
a future section-family addition); foundation ties at the brace bases and
end-wall girts are out of scope; gable columns (when present) remain
gravity posts pending the C&C/wall-wind phase.

Interface units are feet; geometry is emitted in inches like everything
else.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .clear_span import (PURLIN, TRUSS, ClearSpanConfig,
                         _COINCIDENT_TOL_IN, build_clear_span_geometry)
from .config import FT
from .geometry import FrameGeometry, MemberInfo, NodeInfo

BRACE = "brace"                # sidewall X diagonals (tension-only pairs)
WALL_STRUT = "wall_strut"      # horizontal struts closing intermediate tiers
ROOF_BRACE = "roof_brace"      # roof-plane X diagonals (end bays)
EAVE_STRUT = "eave_strut"      # re-tagged eave purlin lines (collectors)

MAX_BAYS_BETWEEN_BRACED = 5    # longest unbraced run between braced bays
BRACE_ANGLE_MIN_DEG = 30.0     # practice band for X diagonals (45 target)
BRACE_ANGLE_MAX_DEG = 60.0


def derive_braced_bays(n_bays: int) -> list[int]:
    """Braced-bay indices: both end bays, plus evenly spaced interior bays
    so no unbraced run exceeds MAX_BAYS_BETWEEN_BRACED. Each placement is
    unioned with its mirror image, so the layout is symmetric about
    midlength by construction (an asymmetric brace layout would twist the
    building); the extra bay this sometimes adds only shortens runs."""
    if n_bays <= 1:
        return [0]
    segments = max(1, math.ceil((n_bays - 1) / (MAX_BAYS_BETWEEN_BRACED + 1)))
    bays: set[int] = set()
    for q in range(segments + 1):
        p = math.floor(q * (n_bays - 1) / segments + 0.5)
        bays.update((p, n_bays - 1 - p))
    return sorted(bays)


def derive_brace_tiers(wall_height_ft: float, bay_ft: float) -> int:
    """Tier count putting the X diagonals as close to 45 deg as possible,
    clamped into the 30-60 deg practice band (1 tier minimum)."""
    def angle(tiers: int) -> float:
        return math.degrees(math.atan2(wall_height_ft / tiers, bay_ft))

    tiers = max(1, round(wall_height_ft / bay_ft))
    while angle(tiers) > BRACE_ANGLE_MAX_DEG:
        tiers += 1
    while tiers > 1 and angle(tiers) < BRACE_ANGLE_MIN_DEG:
        tiers -= 1
    return tiers


def derive_roof_panels(span_ft: float, frame_spacing_ft: float) -> int:
    """Roof-brace plan panels across the span: near-square panels (~45 deg
    diagonals), i.e. one panel per frame-spacing of width."""
    return max(1, round(span_ft / frame_spacing_ft))


@dataclass
class LateralSystemConfig:
    """Candidate sections for the LFRS groups + optional layout pins.

    Like the gravity config, the LAYOUT (braced bays, tiers, roof panels)
    derives from the building; pinning a value is for tests/validation.
    wall_strut_candidates may be omitted for single-tier walls (no
    intermediate struts exist then).
    """
    brace_candidates: list[str]
    roof_brace_candidates: list[str]
    eave_strut_candidates: list[str]
    wall_strut_candidates: list[str] | None = None
    braced_bay_indices: list[int] | None = None   # None -> derived
    n_brace_tiers: int | None = None              # None -> derived
    n_roof_panels: int | None = None              # None -> derived

    def __post_init__(self) -> None:
        for name in ("brace_candidates", "roof_brace_candidates",
                     "eave_strut_candidates"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty.")
        if self.wall_strut_candidates is not None and not self.wall_strut_candidates:
            raise ValueError("wall_strut_candidates must be non-empty when given.")
        if self.braced_bay_indices is not None:
            if not self.braced_bay_indices:
                raise ValueError("braced_bay_indices must be non-empty when given.")
            if sorted(set(self.braced_bay_indices)) != sorted(self.braced_bay_indices):
                raise ValueError("braced_bay_indices must be sorted and unique.")
        if self.n_brace_tiers is not None and self.n_brace_tiers < 1:
            raise ValueError("n_brace_tiers must be >= 1.")
        if self.n_roof_panels is not None and self.n_roof_panels < 1:
            raise ValueError("n_roof_panels must be >= 1.")


@dataclass
class LateralSystem:
    """The augmented building: gravity geometry + LFRS members, plus the
    config copy the analysis layer must use (transverse_moment_frame set
    for W-girder buildings)."""
    geometry: FrameGeometry
    analysis_config: ClearSpanConfig
    lateral_config: LateralSystemConfig
    braced_bay_indices: list[int]
    n_tiers: int
    tier_height_ft: float
    brace_angle_deg: float
    roof_brace_bays: list[int]
    roof_panel_line_indices: list[int]   # purlin-line indices bounding panels
    roof_panel_widths_ft: list[float]    # ACTUAL widths (boundaries snap to
                                         # purlin lines, so they can differ)
    roof_brace_angle_deg: float
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def transverse_system(self) -> str:
        if self.analysis_config.girder_system == TRUSS:
            return "rigid truss bents (existing)"
        return "portal frames (knees moment-connected by the lateral phase)"

    def candidates_by_group(self) -> dict[str, list[str]]:
        """Gravity groups + LFRS groups, for the sizing phase. Key order
        extends the gravity reporting order."""
        groups = dict(self.analysis_config.candidates_by_group)
        lat = self.lateral_config
        groups[EAVE_STRUT] = lat.eave_strut_candidates
        groups[BRACE] = lat.brace_candidates
        if self.n_tiers > 1:
            groups[WALL_STRUT] = lat.wall_strut_candidates
        groups[ROOF_BRACE] = lat.roof_brace_candidates
        return groups

    def describe(self) -> list[str]:
        bays = ", ".join(str(b) for b in self.braced_bay_indices)
        lines = [
            f"LFRS:   transverse (x): {self.transverse_system}",
            f"        longitudinal (z): X-braced sidewall bays [{bays}] "
            f"(both walls), {self.n_tiers} tier(s) @ "
            f"{self.tier_height_ft:.1f} ft, diagonals "
            f"{self.brace_angle_deg:.0f} deg",
            f"        roof: X-bracing in end bay(s) "
            f"{self.roof_brace_bays}, "
            f"{len(self.roof_panel_line_indices) - 1} plan panel(s), "
            f"diagonals {self.roof_brace_angle_deg:.0f} deg; eave purlin "
            "lines re-tagged as eave struts (collectors)",
            "        added members: "
            + ", ".join(f"{n} {g}" for g, n in self.counts.items()),
        ]
        return lines


def _column_top_node(config: ClearSpanConfig, side: int, j: int) -> str:
    """The column-top node at the roof plane: the eave/knee node for
    W-girder buildings, the top-chord tie-in for truss bents."""
    if config.girder_system == TRUSS:
        panel = 0 if side == 0 else config.n_truss_panels
        return f"NT{panel}.{j}"
    return f"NE{side}.{j}"


def _roof_line_node(config: ClearSpanConfig, i: int, j: int) -> str:
    """Roof-plane node on purlin line i at frame j — mirrors the line_node
    helpers of the geometry builders (including the truss builder's reuse
    of panel-point nodes for coincident purlin lines)."""
    n_sp = config.n_purlin_spaces
    if config.girder_system != TRUSS:
        if i == 0:
            return f"NE0.{j}"
        if i == n_sp:
            return f"NE1.{j}"
        return f"NP{i}.{j}"
    k = config.n_truss_panels
    if i == 0:
        return f"NT0.{j}"
    if i == n_sp:
        return f"NT{k}.{j}"
    sp = config.span_ft * FT / n_sp
    p = config.span_ft * FT / k
    x = i * sp
    panel = round(x / p)
    if abs(x - panel * p) < _COINCIDENT_TOL_IN:
        return f"NT{panel}.{j}"
    return f"NP{i}.{j}"


def _eave_line_indices(config: ClearSpanConfig) -> tuple[int, int]:
    return (0, config.n_purlin_spaces)


def build_lateral_system(config: ClearSpanConfig,
                         lateral: LateralSystemConfig) -> LateralSystem:
    """Augment the gravity geometry with the LFRS (see module docstring).

    `config` must be a RESOLVED building (a concrete girder system and
    layout — i.e. the .config of the winning gravity OptimizationResult),
    not an unsolved 'auto' config.
    """
    if config.girder_system not in ("wide_flange", TRUSS):
        raise ValueError(
            "build_lateral_system needs a resolved girder system "
            "('wide_flange' or 'truss') — pass the winning gravity "
            f"result's .config, not girder_system={config.girder_system!r}.")
    trussed = config.girder_system == TRUSS
    base = build_clear_span_geometry(config)

    n_bays = config.n_frames - 1
    bay_ft = config.frame_spacing_ft
    braced = (derive_braced_bays(n_bays) if lateral.braced_bay_indices is None
              else sorted(set(lateral.braced_bay_indices)))
    if braced[0] < 0 or braced[-1] > n_bays - 1:
        raise ValueError(
            f"braced_bay_indices {braced} outside bays 0..{n_bays - 1}.")

    top_y_in = (config.eave_height_ft
                + (config.truss_depth_ft if trussed else 0.0)) * FT
    wall_height_ft = top_y_in / FT
    tiers = (derive_brace_tiers(wall_height_ft, bay_ft)
             if lateral.n_brace_tiers is None else lateral.n_brace_tiers)
    if tiers > 1 and lateral.wall_strut_candidates is None:
        raise ValueError(
            f"{tiers} brace tiers need intermediate wall struts — provide "
            "wall_strut_candidates.")
    tier_h_ft = wall_height_ft / tiers
    brace_angle = math.degrees(math.atan2(tier_h_ft, bay_ft))

    n_sp = config.n_purlin_spaces
    n_panels = (derive_roof_panels(config.span_ft, bay_ft)
                if lateral.n_roof_panels is None else lateral.n_roof_panels)
    n_panels = min(n_panels, n_sp)   # can't have finer panels than lines
    panel_lines = sorted({round(q * n_sp / n_panels) for q in range(n_panels + 1)})
    roof_bays = sorted({0, n_bays - 1})
    # panel boundaries snap to purlin lines, so ACTUAL widths vary — member
    # lengths must use them (a uniform span/n_panels length would disagree
    # with the node geometry and falsify the diagonals)
    sp_ft = config.span_ft / n_sp
    panel_widths_ft = [(i1 - i0) * sp_ft
                       for i0, i1 in zip(panel_lines[:-1], panel_lines[1:])]
    mean_w_ft = config.span_ft / (len(panel_lines) - 1)
    roof_angle = math.degrees(math.atan2(bay_ft, mean_w_ft))

    s_f_in = bay_ft * FT
    span_in = config.span_ft * FT

    # ------- nodes: DOF upgrade + tier nodes on braced-bay columns -------
    knee = () if trussed else tuple(
        f"NE{side}.{j}" for side in (0, 1) for j in range(config.n_frames))
    nodes: list[NodeInfo] = []
    for n in base.nodes:
        if n.is_base:
            nodes.append(n)
            continue
        nodes.append(replace(
            n, free_dx=True,
            free_rotations=n.free_rotations or n.name in knee))

    braced_lines = sorted({j for b in braced for j in (b, b + 1)})
    for side, x in ((0, 0.0), (1, span_in)):
        for j in braced_lines:
            for t in range(1, tiers):
                nodes.append(NodeInfo(
                    f"NW{side}.{j}.t{t}", x, top_y_in * t / tiers, j * s_f_in,
                    is_base=False, free_rotations=True, free_dx=True))

    def wall_point(side: int, j: int, level: int) -> str:
        if level == 0:
            return f"NB{side}.{j}"
        if level == tiers:
            return _column_top_node(config, side, j)
        return f"NW{side}.{j}.t{level}"

    # ------- members: retag eave lines, then add the LFRS -------
    eave_i = _eave_line_indices(config)
    eave_names = {f"P{i}.b{b}" for i in eave_i for b in range(n_bays)}
    members: list[MemberInfo] = []
    for m in base.members:
        if m.group == PURLIN and m.name in eave_names:
            members.append(replace(m, group=EAVE_STRUT))
        elif not trussed and m.name.startswith("G"):
            # W-girder portal frames: girder (and end-girder) knees fixed
            members.append(replace(m, fixed_i=True, fixed_j=True))
        elif not trussed and m.name.startswith("C") and not m.name.startswith("CG"):
            members.append(replace(m, fixed_j=True))   # column top at the knee
        else:
            members.append(m)

    counts = {EAVE_STRUT: len(eave_names), BRACE: 0, WALL_STRUT: 0,
              ROOF_BRACE: 0}
    diag_len = math.hypot(s_f_in, top_y_in / tiers)
    for side in (0, 1):
        for b in braced:
            for t in range(1, tiers + 1):
                for tag, (i_nd, j_nd) in (
                        ("a", (wall_point(side, b, t - 1),
                               wall_point(side, b + 1, t))),
                        ("b", (wall_point(side, b + 1, t - 1),
                               wall_point(side, b, t)))):
                    members.append(MemberInfo(
                        name=f"DB{side}.{b}.t{t}.{tag}", group=BRACE,
                        i_node=i_nd, j_node=j_nd, length_in=diag_len,
                        story=1, trib_width_in=0.0))
                    counts[BRACE] += 1
            for t in range(1, tiers):
                members.append(MemberInfo(
                    name=f"WS{side}.{b}.t{t}", group=WALL_STRUT,
                    i_node=wall_point(side, b, t),
                    j_node=wall_point(side, b + 1, t),
                    length_in=s_f_in, story=1, trib_width_in=0.0))
                counts[WALL_STRUT] += 1

    for b in roof_bays:
        for q, (i0, i1) in enumerate(zip(panel_lines[:-1], panel_lines[1:])):
            diag_len_q = math.hypot(panel_widths_ft[q] * FT, s_f_in)
            for tag, (i_nd, j_nd) in (
                    ("a", (_roof_line_node(config, i0, b),
                           _roof_line_node(config, i1, b + 1))),
                    ("b", (_roof_line_node(config, i1, b),
                           _roof_line_node(config, i0, b + 1)))):
                members.append(MemberInfo(
                    name=f"RB{q}.{b}.{tag}", group=ROOF_BRACE,
                    i_node=i_nd, j_node=j_nd, length_in=diag_len_q,
                    story=1, trib_width_in=0.0))
                counts[ROOF_BRACE] += 1
    if tiers == 1:
        del counts[WALL_STRUT]

    analysis_config = (config if trussed
                       else replace(config, transverse_moment_frame=True))
    return LateralSystem(
        geometry=FrameGeometry(nodes=nodes, members=members),
        analysis_config=analysis_config,
        lateral_config=lateral,
        braced_bay_indices=braced,
        n_tiers=tiers,
        tier_height_ft=tier_h_ft,
        brace_angle_deg=brace_angle,
        roof_brace_bays=roof_bays,
        roof_panel_line_indices=panel_lines,
        roof_panel_widths_ft=panel_widths_ft,
        roof_brace_angle_deg=roof_angle,
        counts=counts,
    )
