"""Clear-span industrial building: config, geometry, and design rules.

Intended for enclosures over large equipment where interior columns are not
allowed. Topology (X = clear-span direction, Z = building length, Y up):
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import COLUMN, FT_TO_M, M_TO_IN, MM_TO_IN
from .design import CheckParams, GroupRules
from .geometry import FrameGeometry, MemberInfo, NodeInfo

GIRDER = "girder"
END_GIRDER = "end_girder"
PURLIN = "purlin"
# Pratt-truss roof system (roof_system="truss"): replaces the interior W
# girders when the clear span outgrows every rolled shape. Verticals and
# diagonals share one 'truss_web' group (one shape — fabrication-uniform).
TOP_CHORD = "top_chord"
BOTTOM_CHORD = "bottom_chord"
TRUSS_WEB = "truss_web"

# two girder-axis nodes closer than this are treated as the same point
_COINCIDENT_TOL_IN = 1e-6

# ---------------------------------------------------------------------------
# Layout practice bands (single-story industrial steel buildings).
#
# The building footprint (span, length, eave height) is the input; the layout
# (frame count, purlin spacing, gable columns) is derived from it. These bands
# bound the derivation to configurations a fabricator would actually build.
# The values are the customary US-practice bands (20-30 ft bays, 4-6 ft
# purlins, 15-25 ft end-girder segments), converted exactly to meters:
#
# * Bay (frame) spacing: hot-rolled W purlins span frame-to-frame economically
#   at ~6-9 m; ~7.6 m is the customary sweet spot for industrial bays.
# * Purlin spacing: one-way roof deck spans economically at ~1.2-1.8 m o.c.
# * End-wall (gable) columns: end-girder segments between supports typically
#   run ~4.6-7.6 m along the end wall.
# ---------------------------------------------------------------------------
MIN_FRAME_SPACING_M = 20.0 * FT_TO_M      # 6.096 m
TARGET_FRAME_SPACING_M = 25.0 * FT_TO_M   # 7.62 m
MAX_FRAME_SPACING_M = 30.0 * FT_TO_M      # 9.144 m
MIN_PURLIN_SPACING_M = 4.0 * FT_TO_M      # 1.219 m
TARGET_PURLIN_SPACING_M = 5.0 * FT_TO_M   # 1.524 m
MAX_PURLIN_SPACING_M = 6.0 * FT_TO_M      # 1.829 m
MIN_END_GIRDER_SEGMENT_M = 15.0 * FT_TO_M  # 4.572 m
MAX_END_GIRDER_SEGMENT_M = 25.0 * FT_TO_M  # 7.62 m

_LAYOUT_FIELDS = ("n_frames", "purlin_spacing_m", "end_wall_columns")


def derive_n_frames(length_m: float) -> int:
    """Frame count putting bays as close to TARGET_FRAME_SPACING_M as the
    length allows, never above MAX_FRAME_SPACING_M. Buildings no longer than
    a single bay collapse to 2 frames — the minimal '1x1 bay' enclosure."""
    n_bays = max(1, round(length_m / TARGET_FRAME_SPACING_M))
    while length_m / n_bays > MAX_FRAME_SPACING_M:
        n_bays += 1
    return n_bays + 1


def derive_purlin_spacing_m(span_m: float) -> float:
    """Target purlin spacing: TARGET_PURLIN_SPACING_M, unless the span is so
    short that the minimum of two purlin spaces forces it smaller."""
    return min(TARGET_PURLIN_SPACING_M, span_m / 2.0)


def derive_end_wall_columns(span_m: float, has_end_girder_group: bool) -> int:
    """Gable columns per end wall so no end-girder segment exceeds
    MAX_END_GIRDER_SEGMENT_M. Zero when one segment covers the span, or when
    no separate end-girder group exists — without a lighter end-girder group
    gable columns cannot pay off (see ClearSpanConfig.__post_init__)."""
    if not has_end_girder_group:
        return 0
    return max(0, math.ceil(span_m / MAX_END_GIRDER_SEGMENT_M) - 1)


def candidate_layouts(config: "ClearSpanConfig") -> list[tuple[int, float, int]]:
    """Realistic (n_frames, purlin_spacing_m, end_wall_columns) combinations
    for the footprint — the search space of optimize_layout(). Every spacing
    stays inside the practice bands above. Layout fields the user pinned to an
    explicit value contribute exactly that value; auto-derived fields range
    over their band."""
    if "n_frames" in config.auto_layout_fields:
        n_lo = max(1, math.ceil(config.length_m / MAX_FRAME_SPACING_M))
        n_hi = max(n_lo, math.floor(config.length_m / MIN_FRAME_SPACING_M))
        frame_opts = [n + 1 for n in range(n_lo, n_hi + 1)]
    else:
        frame_opts = [config.n_frames]

    if "purlin_spacing_m" in config.auto_layout_fields:
        # dedupe targets that round to the same purlin-space count
        by_spaces: dict[int, float] = {}
        for target in (TARGET_PURLIN_SPACING_M, MIN_PURLIN_SPACING_M,
                       MAX_PURLIN_SPACING_M):
            t = min(target, config.span_m / 2.0)
            by_spaces.setdefault(max(2, round(config.span_m / t)), t)
        purlin_opts = list(by_spaces.values())
    else:
        purlin_opts = [config.purlin_spacing_m]

    if "end_wall_columns" in config.auto_layout_fields:
        if config.has_end_girder_group:
            k_lo = max(0, math.ceil(config.span_m / MAX_END_GIRDER_SEGMENT_M) - 1)
            k_hi = max(k_lo, math.floor(config.span_m / MIN_END_GIRDER_SEGMENT_M) - 1)
            gable_opts = sorted({0, *range(k_lo, k_hi + 1)})
        else:
            gable_opts = [0]
    else:
        gable_opts = [config.end_wall_columns]

    return [(nf, sp, gc) for nf in frame_opts for sp in purlin_opts
            for gc in gable_opts]


@dataclass
class ClearSpanConfig:
    # --- candidate sections (AISC Manual labels), one list per design group ---
    girder_candidates: list[str]
    purlin_candidates: list[str]
    column_candidates: list[str]
    # optional separate group for the two end-wall girders; give it a list to
    # let them be sized lighter than the interior girders (required when
    # end_wall_columns > 0, where the benefit is largest)
    end_girder_candidates: list[str] | None = None

    # --- building footprint (m) — the geometric inputs ---
    # Orientation is normalized on construction: if span_m > length_m the
    # two are swapped so girders always clear-span the shorter plan
    # dimension (see __post_init__).
    span_m: float = 20.0         # clear span, girder direction (no interior columns)
    length_m: float = 30.0       # building length
    eave_height_m: float = 9.0

    # --- layout — derived from the footprint, NOT user inputs ---
    # Leave these as None (the default): __post_init__ derives realistic
    # values from the footprint via the practice bands above, and
    # optimize_layout() searches those bands for the lightest feasible
    # design. Setting one explicitly pins it (intended for tests and
    # validation studies, not for normal use).
    n_frames: int | None = None          # transverse frame lines incl. both ends (>= 2)
    purlin_spacing_m: float | None = None   # target; actual = span_m / n_purlin_spaces
    end_wall_columns: int | None = None  # interior gable columns per end wall
                                         # (exterior walls only — the clear
                                         # span stays clear)

    # --- gravity loads (kPa = kN/m^2 over the roof plan) ---
    superimposed_dead_kpa: float = 0.0   # deck + insulation + collateral
    live_kpa: float = 0.0                # governing roof live (Lr) or snow

    # --- material, MPa (default ASTM A992: Fy 345, Fu 450, E 200 GPa) ---
    Fy_mpa: float = 345.0
    Fu_mpa: float = 450.0
    E_mpa: float = 200000.0

    # --- design options ---
    girder_Lb_m: float | None = None    # None -> actual purlin spacing (purlins
                                        # brace the girder compression flange)
    purlin_Lb_m: float | None = None    # None -> full purlin span (conservative);
                                        # 0 = through-fastened deck braces top flange
    girder_camber_mm: float = 0.0       # fabrication camber on interior girders,
                                        # credited against total-load deflection
                                        # only (keep <= the dead-load deflection)
    check_deflection: bool = True
    defl_live_ratio: float = 360.0      # IBC Table 1604.3 floor values by default
    defl_total_ratio: float = 240.0
    # optional per-group relaxations (None -> the global pair above); e.g.
    # roof members not supporting a ceiling may justify L/240 and L/180
    girder_defl_live_ratio: float | None = None
    girder_defl_total_ratio: float | None = None
    purlin_defl_live_ratio: float | None = None
    purlin_defl_total_ratio: float | None = None
    enforce_slenderness_limit: bool = True   # KL/r <= 200 on columns

    # --- roof system: W girder vs. Pratt truss ---
    # "auto" (default) designs W girders when any candidate can carry the
    # span, and switches to a parallel-chord Pratt truss when the FEA-free
    # lower-bound proof shows none can. The truss bears at the TOP chord on
    # the existing column tops (roof plane, purlins, end walls, and columns
    # keep their elevations; the truss depth hangs below the eave). End
    # frames keep their gable-column-propped W end girders either way.
    roof_system: str = "auto"                # "auto" | "girder" | "truss"
    top_chord_candidates: list[str] | None = None
    bottom_chord_candidates: list[str] | None = None
    truss_web_candidates: list[str] | None = None
    truss_depth_m: float | None = None       # None -> span/12
    truss_camber_mm: float = 0.0             # credited like girder camber,
                                             # against total deflection only
    bottom_chord_brace_spacing_m: float | None = None  # out-of-plane bracing
                                             # of the bottom chord (assumed
                                             # struts, not modeled);
                                             # None -> every panel point

    def __post_init__(self) -> None:
        if self.roof_system not in ("auto", "girder", "truss"):
            raise ValueError('roof_system must be "auto", "girder", or "truss".')
        # girder candidates stay required for "auto" (it must be able to
        # prove girder infeasibility before switching) and for "girder"
        required = ("purlin_candidates", "column_candidates")
        if self.roof_system != "truss":
            required = ("girder_candidates",) + required
        for name in required:
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty.")
        if self.end_girder_candidates is not None and not self.end_girder_candidates:
            raise ValueError("end_girder_candidates must be non-empty when given.")
        if self.roof_system == "truss":
            for name in ("top_chord_candidates", "bottom_chord_candidates",
                         "truss_web_candidates"):
                if not getattr(self, name):
                    raise ValueError(
                        f'{name} must be non-empty when roof_system="truss".')
            if self.end_girder_candidates is None:
                raise ValueError(
                    'roof_system="truss" requires end_girder_candidates: the '
                    "end frames keep gable-column-propped W end girders (a "
                    "truss on a propped wall line has no purpose)."
                )
        if self.span_m <= 0 or self.length_m <= 0 or self.eave_height_m <= 0:
            raise ValueError("span_m, length_m, and eave_height_m must be positive.")

        # Girders always clear-span the SHORTER plan dimension: girder moment
        # grows with span^2 (deflection with span^4), while purlins, columns,
        # and the clear interior are orientation-agnostic — so spanning the
        # long way is never lighter, and the swap is just the framing plan
        # rotated 90 degrees on the same footprint. Normalize automatically.
        self._footprint_swapped = self.span_m > self.length_m
        if self._footprint_swapped:
            self.span_m, self.length_m = self.length_m, self.span_m

        # Layout fields left as None are derived from the footprint; remember
        # which ones so optimize_layout() knows its free search variables.
        self._auto_layout = frozenset(
            name for name in _LAYOUT_FIELDS if getattr(self, name) is None)
        if self.n_frames is None:
            self.n_frames = derive_n_frames(self.length_m)
        if self.purlin_spacing_m is None:
            self.purlin_spacing_m = derive_purlin_spacing_m(self.span_m)
        if self.end_wall_columns is None:
            self.end_wall_columns = derive_end_wall_columns(
                self.span_m, self.has_end_girder_group)

        if self.end_wall_columns and self.end_girder_candidates is None:
            raise ValueError(
                "end_wall_columns > 0 requires end_girder_candidates: gable "
                "columns only pay off when the supported end girders form "
                "their own (lighter) design group."
            )
        if self.n_frames < 2:
            raise ValueError("n_frames must be >= 2 (both end walls need a frame).")
        if not (0.0 < self.purlin_spacing_m <= self.span_m / 2.0):
            raise ValueError("purlin_spacing_m must be in (0, span_m/2].")
        if self.end_wall_columns < 0:
            raise ValueError("end_wall_columns must be >= 0.")
        if self.superimposed_dead_kpa < 0 or self.live_kpa < 0:
            raise ValueError("Loads must be non-negative.")
        if self.girder_camber_mm < 0:
            raise ValueError("girder_camber_mm must be >= 0.")
        if self.truss_camber_mm < 0:
            raise ValueError("truss_camber_mm must be >= 0.")
        if self.roof_system == "truss" and self.n_frames < 3:
            raise ValueError(
                'roof_system="truss" needs an interior frame to carry a truss '
                "(n_frames >= 3); a 2-frame building is all end walls, where "
                "gable-column-propped end girders already cover the span."
            )
        if self.has_truss_groups or self.roof_system == "truss":
            if not (0.0 < self.truss_depth_actual_m < self.eave_height_m):
                raise ValueError(
                    "truss depth must be in (0, eave_height_m): the truss "
                    "bears at the top chord and hangs below the eave."
                )
            if (self.bottom_chord_brace_spacing_m is not None
                    and self.bottom_chord_brace_spacing_m <= 0.0):
                raise ValueError(
                    "bottom_chord_brace_spacing_m must be positive when given.")

    # --- derived geometry ---
    @property
    def auto_layout_fields(self) -> frozenset[str]:
        """Layout fields that were derived from the footprint rather than
        given explicitly — the free variables of optimize_layout()."""
        return self._auto_layout

    @property
    def frame_spacing_m(self) -> float:
        return self.length_m / (self.n_frames - 1)

    @property
    def n_purlin_spaces(self) -> int:
        return max(2, round(self.span_m / self.purlin_spacing_m))

    @property
    def purlin_spacing_actual_m(self) -> float:
        return self.span_m / self.n_purlin_spaces

    @property
    def has_end_girder_group(self) -> bool:
        return self.end_girder_candidates is not None

    # --- Pratt truss (derived as properties, never written back, so
    # dataclasses.replace() re-derives them consistently) ---
    @property
    def is_truss(self) -> bool:
        """True only for a RESOLVED truss config. "auto" behaves as girder
        everywhere downstream until the optimizer resolves it (see
        optimizer._resolve_roof_system)."""
        return self.roof_system == "truss"

    @property
    def has_truss_groups(self) -> bool:
        return all(getattr(self, name) for name in (
            "top_chord_candidates", "bottom_chord_candidates",
            "truss_web_candidates"))

    @property
    def truss_depth_actual_m(self) -> float:
        """Parallel-chord truss depth: span/12 is the customary economical
        depth for long-span roof trusses unless the user pins one."""
        if self.truss_depth_m is not None:
            return self.truss_depth_m
        return self.span_m / 12.0

    @property
    def n_truss_panels(self) -> int:
        """Even panel count (vertical at midspan) with panel length as close
        to the depth as possible — diagonals near the customary 45 degrees.
        Minimum 4: the bottom chord spans the interior panel points (it ends
        one panel in from each bearing), which needs at least three."""
        return max(4, 2 * round(self.span_m / self.truss_depth_actual_m / 2.0))

    @property
    def truss_panel_m(self) -> float:
        return self.span_m / self.n_truss_panels

    @property
    def candidates_by_group(self) -> dict[str, list[str]]:
        """Candidate section labels per design group. Key order sets the
        reporting order in results and the wireframe legend."""
        if self.is_truss:
            groups = {COLUMN: self.column_candidates,
                      TOP_CHORD: self.top_chord_candidates,
                      BOTTOM_CHORD: self.bottom_chord_candidates}
            if self.has_end_girder_group:
                groups[END_GIRDER] = self.end_girder_candidates
            groups[TRUSS_WEB] = self.truss_web_candidates
            groups[PURLIN] = self.purlin_candidates
            return groups
        groups = {COLUMN: self.column_candidates, GIRDER: self.girder_candidates}
        if self.has_end_girder_group:
            groups[END_GIRDER] = self.end_girder_candidates
        groups[PURLIN] = self.purlin_candidates
        return groups

    def describe(self) -> list[str]:
        gable = (f", {self.end_wall_columns} gable column(s)/end wall"
                 if self.end_wall_columns else "")
        if self.is_truss:
            camber = (f", truss camber {self.truss_camber_mm:.0f} mm"
                      if self.truss_camber_mm else "")
            roof = (f"Roof:   Pratt truss (top-chord bearing), depth "
                    f"{self.truss_depth_actual_m:.2f} m, {self.n_truss_panels} "
                    f"panels @ {self.truss_panel_m:.2f} m; purlins @ "
                    f"{self.purlin_spacing_actual_m:.2f} m "
                    f"({self.n_purlin_spaces + 1} lines){camber}")
        else:
            camber = (f", girder camber {self.girder_camber_mm:.0f} mm"
                      if self.girder_camber_mm else "")
            roof = (f"Roof:   purlins @ {self.purlin_spacing_actual_m:.2f} m "
                    f"({self.n_purlin_spaces + 1} lines), one-way deck -> "
                    f"purlin -> girder{camber}")
        lines = [
            f"Frame:  clear span {self.span_m:.1f} m x length {self.length_m:.1f} m, "
            f"{self.n_frames} frames @ {self.frame_spacing_m:.2f} m, "
            f"eave {self.eave_height_m:.1f} m (NO interior columns{gable})",
            roof,
            f"Loads:  SDL = {self.superimposed_dead_kpa} kPa, "
            f"roof L/S = {self.live_kpa} kPa (1.4D, 1.2D+1.6L) + self-weight",
        ]
        if self._auto_layout:
            lines.append(
                "Layout: " + ", ".join(sorted(self._auto_layout))
                + " derived from the footprint (not user inputs)")
        if self._footprint_swapped:
            lines.append(
                "Note:   span/length inputs were swapped so the girders "
                "clear-span the shorter plan dimension")
        return lines


def clear_span_check_params(config: ClearSpanConfig) -> CheckParams:
    """Per-group AISC/serviceability rules for the clear-span building.

    Girders default to Lb = the actual purlin spacing (each purlin line is a
    top-flange brace point under gravity); purlins default to the conservative
    full-span Lb unless the deck attachment justifies purlin_Lb_m = 0. All
    flexural groups are gravity-loaded simple spans, so the single-unbraced-
    segment Cb of 12.5/11 (AISC F1-1, parabolic diagram) applies when unbraced.
    """
    def ratio(override: float | None, fallback: float) -> float:
        return fallback if override is None else override

    g_live = ratio(config.girder_defl_live_ratio, config.defl_live_ratio)
    g_total = ratio(config.girder_defl_total_ratio, config.defl_total_ratio)
    p_live = ratio(config.purlin_defl_live_ratio, config.defl_live_ratio)
    p_total = ratio(config.purlin_defl_total_ratio, config.defl_total_ratio)

    sp_in = config.purlin_spacing_actual_m * M_TO_IN
    girder_Lb = sp_in if config.girder_Lb_m is None else config.girder_Lb_m * M_TO_IN
    rules = {
        COLUMN: GroupRules(
            check_deflection=False,   # columns: no sag check (they report 0 anyway)
            check_slenderness=config.enforce_slenderness_limit,
        ),
        PURLIN: GroupRules(
            Lb_in=None if config.purlin_Lb_m is None else config.purlin_Lb_m * M_TO_IN,
            check_deflection=config.check_deflection,
            defl_live_ratio=p_live, defl_total_ratio=p_total,
            Cb_simple_span=True,
        ),
    }
    if config.is_truss:
        panel_in = config.truss_panel_m * M_TO_IN
        brace_in = (panel_in if config.bottom_chord_brace_spacing_m is None
                    else config.bottom_chord_brace_spacing_m * M_TO_IN)
        # Top chord: compression + local bending (purlins land between panel
        # points). In-plane buckling over one panel, out-of-plane and LTB
        # braced by the purlins. Its chord-relative sag IS the truss midspan
        # deflection (one full-span physical member), so the deflection check
        # and camber credit apply here, projected by area (truss sag is
        # governed by chord axial stiffness, delta ~ 1/A).
        rules[TOP_CHORD] = GroupRules(
            Lb_in=sp_in,
            check_deflection=config.check_deflection,
            defl_live_ratio=g_live, defl_total_ratio=g_total,
            camber_in=config.truss_camber_mm * MM_TO_IN,
            check_slenderness=True,
            KLx_in=panel_in, KLy_in=sp_in,
            apply_B1=True,
            defl_scale_axial=True,
        )
        # Bottom chord: tension under gravity (L/r <= 300); braced
        # out-of-plane by assumed bracing struts (not modeled) at
        # bottom_chord_brace_spacing_m, default every panel point.
        rules[BOTTOM_CHORD] = GroupRules(
            Lb_in=brace_in,
            check_deflection=False,
            check_slenderness=True,
            KLx_in=panel_in, KLy_in=brace_in,
            apply_B1=True,
        )
        # Webs: pin-ended axial members checked over their own length
        # (K = 1, no bracing credit — conservative).
        rules[TRUSS_WEB] = GroupRules(
            check_deflection=False,
            check_slenderness=True,
        )
    else:
        rules[GIRDER] = GroupRules(
            Lb_in=girder_Lb,
            check_deflection=config.check_deflection,
            defl_live_ratio=g_live, defl_total_ratio=g_total,
            Cb_simple_span=True,
            camber_in=config.girder_camber_mm * MM_TO_IN,
        )
    if config.has_end_girder_group:
        # same bracing/serviceability rules as the interior girders, but no
        # camber: gable-column support makes their effective spans short
        rules[END_GIRDER] = GroupRules(
            Lb_in=girder_Lb,
            check_deflection=config.check_deflection,
            defl_live_ratio=g_live, defl_total_ratio=g_total,
            Cb_simple_span=True,
        )
    return CheckParams.from_material(config, group_rules=rules)


def build_clear_span_geometry(config: ClearSpanConfig) -> FrameGeometry:
    span = config.span_m * M_TO_IN
    height = config.eave_height_m * M_TO_IN
    s_f = config.frame_spacing_m * M_TO_IN
    n_sp = config.n_purlin_spaces
    sp = span / n_sp
    nf = config.n_frames
    end_frames = (0, nf - 1)

    nodes: list[NodeInfo] = []
    members: list[MemberInfo] = []

    # truss frames must translate in-plane (chords shorten/stretch), so their
    # nodes drop the blanket DX restraint: each interior frame is supported
    # pin (NE0 keeps DX) + roller (NE1 and every panel/purlin node free it)
    truss = config.is_truss

    def is_truss_frame(j: int) -> bool:
        return truss and j not in end_frames

    for j in range(nf):
        z = j * s_f
        for side, x in ((0, 0.0), (1, span)):
            nodes.append(NodeInfo(f"NB{side}.{j}", x, 0.0, z, is_base=True))
            nodes.append(NodeInfo(f"NE{side}.{j}", x, height, z, is_base=False,
                                  free_dx=(side == 1 and is_truss_frame(j))))
        # interior purlin-line nodes sit on the girder axis: Pynite splits the
        # physical girder there, and the continuous girder provides their
        # rotational stiffness (free_rotations - see analysis/frame_model.py)
        for i in range(1, n_sp):
            nodes.append(NodeInfo(f"NP{i}.{j}", i * sp, height, z,
                                  is_base=False, free_rotations=True,
                                  free_dx=is_truss_frame(j)))

    def girder_group(j: int) -> str:
        if config.has_end_girder_group and j in end_frames:
            return END_GIRDER
        return GIRDER

    for j in range(nf):
        for side in (0, 1):
            members.append(MemberInfo(
                name=f"C{side}.{j}", group=COLUMN,
                i_node=f"NB{side}.{j}", j_node=f"NE{side}.{j}",
                length_in=height, story=1, trib_width_in=0.0,
            ))
        if is_truss_frame(j):
            continue   # interior truss frames are assembled below
        # girders carry only self-weight directly; ALL roof load arrives as
        # purlin point reactions at the shared nodes
        members.append(MemberInfo(
            name=f"G{j}", group=girder_group(j),
            i_node=f"NE0.{j}", j_node=f"NE1.{j}",
            length_in=span, story=1, trib_width_in=0.0,
        ))

    # Pratt truss on every interior frame (top-chord bearing): the top chord
    # replaces the girder as ONE continuous full-span member at eave height —
    # purlin wiring, chord-relative sag (= truss midspan deflection), and
    # camber credit all carry over from the girder scheme. The bottom chord
    # hangs at eave - depth and ends at the FIRST interior panel points: the
    # end diagonals carry the support shear from the bearings down to it.
    # (Bottom-chord nodes at x = 0 would sit on the column axis, where Pynite
    # would subdivide the physical column and feed chord force into parasitic
    # column bending.) Webs are pin-ended; diagonals slope down toward
    # midspan (tension under gravity), verticals carry compression.
    if truss:
        depth = config.truss_depth_actual_m * M_TO_IN
        n_pan = config.n_truss_panels
        panel = span / n_pan
        diag_len = math.hypot(panel, depth)
        purlin_xs = {i: i * sp for i in range(1, n_sp)}

        for j in range(1, nf - 1):
            z = j * s_f

            # top-chord panel points: reuse a coincident purlin node, else add
            top_names: dict[int, str] = {0: f"NE0.{j}", n_pan: f"NE1.{j}"}
            for k in range(1, n_pan):
                x = k * panel
                for i, xi in purlin_xs.items():
                    if abs(x - xi) < _COINCIDENT_TOL_IN:
                        top_names[k] = f"NP{i}.{j}"
                        break
                else:
                    nodes.append(NodeInfo(f"NT{k}.{j}", x, height, z,
                                          is_base=False, free_rotations=True,
                                          free_dx=True))
                    top_names[k] = f"NT{k}.{j}"

            # bottom-chord nodes k = 1..n-1: interior ones sit on the
            # continuous chord (free rotations); the END nodes (k = 1, n-1)
            # are member ends of a moment-released chord and must keep the
            # rotational restraint
            for k in range(1, n_pan):
                nodes.append(NodeInfo(f"NC{k}.{j}", k * panel, height - depth,
                                      z, is_base=False,
                                      free_rotations=(1 < k < n_pan - 1),
                                      free_dx=True))

            members.append(MemberInfo(
                name=f"TC{j}", group=TOP_CHORD,
                i_node=f"NE0.{j}", j_node=f"NE1.{j}",
                length_in=span, story=1, trib_width_in=0.0,
            ))
            members.append(MemberInfo(
                name=f"BC{j}", group=BOTTOM_CHORD,
                i_node=f"NC1.{j}", j_node=f"NC{n_pan - 1}.{j}",
                length_in=span - 2.0 * panel, story=1, trib_width_in=0.0,
            ))
            for k in range(1, n_pan):
                members.append(MemberInfo(
                    name=f"TV{k}.{j}", group=TRUSS_WEB,
                    i_node=top_names[k], j_node=f"NC{k}.{j}",
                    length_in=depth, story=1, trib_width_in=0.0,
                ))
            for i in range(n_pan):
                if i < n_pan // 2:
                    i_node, j_node = top_names[i], f"NC{i + 1}.{j}"
                else:
                    i_node, j_node = top_names[i + 1], f"NC{i}.{j}"
                members.append(MemberInfo(
                    name=f"TD{i}.{j}", group=TRUSS_WEB,
                    i_node=i_node, j_node=j_node,
                    length_in=diag_len, story=1, trib_width_in=0.0,
                ))

    # end-wall (gable) columns: exterior members under the two end girders.
    # A gable column that lands on a purlin line reuses that node.
    if config.end_wall_columns:
        purlin_xs = {i: i * sp for i in range(1, n_sp)}
        for j in end_frames:
            for k in range(1, config.end_wall_columns + 1):
                x = span * k / (config.end_wall_columns + 1)
                top = None
                for i, xi in purlin_xs.items():
                    if abs(x - xi) < _COINCIDENT_TOL_IN:
                        top, x = f"NP{i}.{j}", xi
                        break
                if top is None:
                    top = f"NG{k}.{j}"
                    nodes.append(NodeInfo(top, x, height, j * s_f,
                                          is_base=False, free_rotations=True))
                nodes.append(NodeInfo(f"NGB{k}.{j}", x, 0.0, j * s_f, is_base=True))
                members.append(MemberInfo(
                    name=f"CG{k}.{j}", group=COLUMN,
                    i_node=f"NGB{k}.{j}", j_node=top,
                    length_in=height, story=1, trib_width_in=0.0,
                ))

    def line_node(i: int, j: int) -> str:
        if i == 0:
            return f"NE0.{j}"
        if i == n_sp:
            return f"NE1.{j}"
        return f"NP{i}.{j}"

    for i in range(n_sp + 1):
        trib = sp if 0 < i < n_sp else sp / 2.0   # eave lines carry half a space
        for j in range(nf - 1):
            members.append(MemberInfo(
                name=f"P{i}.b{j}", group=PURLIN,
                i_node=line_node(i, j), j_node=line_node(i, j + 1),
                length_in=s_f, story=1, trib_width_in=trib,
            ))

    return FrameGeometry(nodes=nodes, members=members)
