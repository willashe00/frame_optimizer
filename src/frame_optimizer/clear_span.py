"""Clear-span industrial building: config, geometry, and design rules.

Intended for enclosures over large equipment where interior columns are not
allowed. Topology (X = clear-span direction, Z = building length, Y up):

* Transverse frames at spacing s_f = length/(n_frames - 1). Each frame is two
  perimeter columns plus ONE clear-span roof girder — no interior columns or
  interior column-supported beams anywhere.
* Purlin lines run in Z at spacing s_p = span/n_spaces along the girder,
  each spanning s_f between adjacent girders. The two lines at x = 0 and
  x = span are eave purlins (half tributary width) spanning column-to-column.
* Optional end-wall (gable) columns — exterior, on the two end walls only —
  support the end girders at interior points. When used, the end girders form
  their own 'end_girder' design group so they can be sized lighter than the
  interior clear-span girders.
* One-way load path: deck -> purlins -> girders -> perimeter columns.

Analysis model (explicit purlins):

* Everything is solved in ONE Pynite model with the same fully pinned
  gravity-only scheme as the grid frame (analysis/frame_model.py). Purlins
  are pin-ended members carrying the deck as a one-way line load
  (q x purlin spacing; half for the eave lines) and deliver their reactions
  to the girders as true point loads at shared nodes.
* Girders are physical members: Pynite subdivides them internally at the
  purlin (and gable-column) nodes but reports moments, shears, and
  deflections over the whole span. Those interior nodes are created with
  free rotations — the continuous girder stabilizes them — so the
  mechanism-stabilization supports do not falsify girder bending.
* Girders therefore carry only their self-weight as a line load; all roof
  load reaches them through the purlins. Total statics close exactly.
* `live_psf` is the governing roof live/snow surface load (ASCE 7 roof live
  Lr is 20 psf minimum; use the governing of Lr and the flat-roof snow load
  for the site). Lateral loads remain out of scope exactly as for the grid
  frame — a separate system must provide wind/seismic resistance.

Interface units are feet and psf (use M_TO_FT for metric plan dimensions);
everything internal is kips and inches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import COLUMN, FT
from .design import CheckParams, GroupRules
from .geometry import FrameGeometry, MemberInfo, NodeInfo

GIRDER = "girder"
END_GIRDER = "end_girder"
PURLIN = "purlin"

# two girder-axis nodes closer than this are treated as the same point
_COINCIDENT_TOL_IN = 1e-6

# ---------------------------------------------------------------------------
# Layout practice bands (single-story industrial steel buildings).
#
# The building footprint (span, length, eave height) is the input; the layout
# (frame count, purlin spacing, gable columns) is derived from it. These bands
# bound the derivation to configurations a fabricator would actually build:
#
# * Bay (frame) spacing: hot-rolled W purlins span frame-to-frame economically
#   at ~20-30 ft; ~25 ft is the customary sweet spot for industrial bays.
# * Purlin spacing: one-way roof deck spans economically at ~4-6 ft o.c.
# * End-wall (gable) columns: end-girder segments between supports typically
#   run ~15-25 ft along the end wall.
# ---------------------------------------------------------------------------
MIN_FRAME_SPACING_FT = 20.0
TARGET_FRAME_SPACING_FT = 25.0
MAX_FRAME_SPACING_FT = 30.0
MIN_PURLIN_SPACING_FT = 4.0
TARGET_PURLIN_SPACING_FT = 5.0
MAX_PURLIN_SPACING_FT = 6.0
MIN_END_GIRDER_SEGMENT_FT = 15.0
MAX_END_GIRDER_SEGMENT_FT = 25.0

_LAYOUT_FIELDS = ("n_frames", "purlin_spacing_ft", "end_wall_columns")


def derive_n_frames(length_ft: float) -> int:
    """Frame count putting bays as close to TARGET_FRAME_SPACING_FT as the
    length allows, never above MAX_FRAME_SPACING_FT. Buildings no longer than
    a single bay collapse to 2 frames — the minimal '1x1 bay' enclosure."""
    n_bays = max(1, round(length_ft / TARGET_FRAME_SPACING_FT))
    while length_ft / n_bays > MAX_FRAME_SPACING_FT:
        n_bays += 1
    return n_bays + 1


def derive_purlin_spacing_ft(span_ft: float) -> float:
    """Target purlin spacing: TARGET_PURLIN_SPACING_FT, unless the span is so
    short that the minimum of two purlin spaces forces it smaller."""
    return min(TARGET_PURLIN_SPACING_FT, span_ft / 2.0)


def derive_end_wall_columns(span_ft: float, has_end_girder_group: bool) -> int:
    """Gable columns per end wall so no end-girder segment exceeds
    MAX_END_GIRDER_SEGMENT_FT. Zero when one segment covers the span, or when
    no separate end-girder group exists — without a lighter end-girder group
    gable columns cannot pay off (see ClearSpanConfig.__post_init__)."""
    if not has_end_girder_group:
        return 0
    return max(0, math.ceil(span_ft / MAX_END_GIRDER_SEGMENT_FT) - 1)


def candidate_layouts(config: "ClearSpanConfig") -> list[tuple[int, float, int]]:
    """Realistic (n_frames, purlin_spacing_ft, end_wall_columns) combinations
    for the footprint — the search space of optimize_layout(). Every spacing
    stays inside the practice bands above. Layout fields the user pinned to an
    explicit value contribute exactly that value; auto-derived fields range
    over their band."""
    if "n_frames" in config.auto_layout_fields:
        n_lo = max(1, math.ceil(config.length_ft / MAX_FRAME_SPACING_FT))
        n_hi = max(n_lo, math.floor(config.length_ft / MIN_FRAME_SPACING_FT))
        frame_opts = [n + 1 for n in range(n_lo, n_hi + 1)]
    else:
        frame_opts = [config.n_frames]

    if "purlin_spacing_ft" in config.auto_layout_fields:
        # dedupe targets that round to the same purlin-space count
        by_spaces: dict[int, float] = {}
        for target in (TARGET_PURLIN_SPACING_FT, MIN_PURLIN_SPACING_FT,
                       MAX_PURLIN_SPACING_FT):
            t = min(target, config.span_ft / 2.0)
            by_spaces.setdefault(max(2, round(config.span_ft / t)), t)
        purlin_opts = list(by_spaces.values())
    else:
        purlin_opts = [config.purlin_spacing_ft]

    if "end_wall_columns" in config.auto_layout_fields:
        if config.has_end_girder_group:
            k_lo = max(0, math.ceil(config.span_ft / MAX_END_GIRDER_SEGMENT_FT) - 1)
            k_hi = max(k_lo, math.floor(config.span_ft / MIN_END_GIRDER_SEGMENT_FT) - 1)
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

    # --- building footprint (ft) — the geometric inputs ---
    # Orientation is normalized on construction: if span_ft > length_ft the
    # two are swapped so girders always clear-span the shorter plan
    # dimension (see __post_init__).
    span_ft: float = 65.0        # clear span, girder direction (no interior columns)
    length_ft: float = 98.0      # building length
    eave_height_ft: float = 30.0

    # --- layout — derived from the footprint, NOT user inputs ---
    # Leave these as None (the default): __post_init__ derives realistic
    # values from the footprint via the practice bands above, and
    # optimize_layout() searches those bands for the lightest feasible
    # design. Setting one explicitly pins it (intended for tests and
    # validation studies, not for normal use).
    n_frames: int | None = None          # transverse frame lines incl. both ends (>= 2)
    purlin_spacing_ft: float | None = None   # target; actual = span_ft / n_purlin_spaces
    end_wall_columns: int | None = None  # interior gable columns per end wall
                                         # (exterior walls only — the clear
                                         # span stays clear)

    # --- gravity loads (psf over the roof plan) ---
    superimposed_dead_psf: float = 0.0   # deck + insulation + collateral
    live_psf: float = 0.0                # governing roof live (Lr) or snow

    # --- material (default ASTM A992) ---
    Fy_ksi: float = 50.0
    Fu_ksi: float = 65.0
    E_ksi: float = 29000.0

    # --- design options ---
    girder_Lb_ft: float | None = None   # None -> actual purlin spacing (purlins
                                        # brace the girder compression flange)
    purlin_Lb_ft: float | None = None   # None -> full purlin span (conservative);
                                        # 0 = through-fastened deck braces top flange
    girder_camber_in: float = 0.0       # fabrication camber on interior girders,
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

    def __post_init__(self) -> None:
        for name in ("girder_candidates", "purlin_candidates", "column_candidates"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty.")
        if self.end_girder_candidates is not None and not self.end_girder_candidates:
            raise ValueError("end_girder_candidates must be non-empty when given.")
        if self.span_ft <= 0 or self.length_ft <= 0 or self.eave_height_ft <= 0:
            raise ValueError("span_ft, length_ft, and eave_height_ft must be positive.")

        # Girders always clear-span the SHORTER plan dimension: girder moment
        # grows with span^2 (deflection with span^4), while purlins, columns,
        # and the clear interior are orientation-agnostic — so spanning the
        # long way is never lighter, and the swap is just the framing plan
        # rotated 90 degrees on the same footprint. Normalize automatically.
        self._footprint_swapped = self.span_ft > self.length_ft
        if self._footprint_swapped:
            self.span_ft, self.length_ft = self.length_ft, self.span_ft

        # Layout fields left as None are derived from the footprint; remember
        # which ones so optimize_layout() knows its free search variables.
        self._auto_layout = frozenset(
            name for name in _LAYOUT_FIELDS if getattr(self, name) is None)
        if self.n_frames is None:
            self.n_frames = derive_n_frames(self.length_ft)
        if self.purlin_spacing_ft is None:
            self.purlin_spacing_ft = derive_purlin_spacing_ft(self.span_ft)
        if self.end_wall_columns is None:
            self.end_wall_columns = derive_end_wall_columns(
                self.span_ft, self.has_end_girder_group)

        if self.end_wall_columns and self.end_girder_candidates is None:
            raise ValueError(
                "end_wall_columns > 0 requires end_girder_candidates: gable "
                "columns only pay off when the supported end girders form "
                "their own (lighter) design group."
            )
        if self.n_frames < 2:
            raise ValueError("n_frames must be >= 2 (both end walls need a frame).")
        if not (0.0 < self.purlin_spacing_ft <= self.span_ft / 2.0):
            raise ValueError("purlin_spacing_ft must be in (0, span_ft/2].")
        if self.end_wall_columns < 0:
            raise ValueError("end_wall_columns must be >= 0.")
        if self.superimposed_dead_psf < 0 or self.live_psf < 0:
            raise ValueError("Loads must be non-negative.")
        if self.girder_camber_in < 0:
            raise ValueError("girder_camber_in must be >= 0.")

    # --- derived geometry ---
    @property
    def auto_layout_fields(self) -> frozenset[str]:
        """Layout fields that were derived from the footprint rather than
        given explicitly — the free variables of optimize_layout()."""
        return self._auto_layout

    @property
    def frame_spacing_ft(self) -> float:
        return self.length_ft / (self.n_frames - 1)

    @property
    def n_purlin_spaces(self) -> int:
        return max(2, round(self.span_ft / self.purlin_spacing_ft))

    @property
    def purlin_spacing_actual_ft(self) -> float:
        return self.span_ft / self.n_purlin_spaces

    @property
    def has_end_girder_group(self) -> bool:
        return self.end_girder_candidates is not None

    @property
    def candidates_by_group(self) -> dict[str, list[str]]:
        """Candidate section labels per design group. Key order sets the
        reporting order in results and the wireframe legend."""
        groups = {COLUMN: self.column_candidates, GIRDER: self.girder_candidates}
        if self.has_end_girder_group:
            groups[END_GIRDER] = self.end_girder_candidates
        groups[PURLIN] = self.purlin_candidates
        return groups

    def describe(self) -> list[str]:
        gable = (f", {self.end_wall_columns} gable column(s)/end wall"
                 if self.end_wall_columns else "")
        camber = (f", girder camber {self.girder_camber_in} in"
                  if self.girder_camber_in else "")
        lines = [
            f"Frame:  clear span {self.span_ft:.1f} ft x length {self.length_ft:.1f} ft, "
            f"{self.n_frames} frames @ {self.frame_spacing_ft:.1f} ft, "
            f"eave {self.eave_height_ft:.1f} ft (NO interior columns{gable})",
            f"Roof:   purlins @ {self.purlin_spacing_actual_ft:.2f} ft "
            f"({self.n_purlin_spaces + 1} lines), one-way deck -> purlin -> girder"
            f"{camber}",
            f"Loads:  SDL = {self.superimposed_dead_psf} psf, "
            f"roof L/S = {self.live_psf} psf (1.4D, 1.2D+1.6L) + self-weight",
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
    full-span Lb unless the deck attachment justifies purlin_Lb_ft = 0. All
    flexural groups are gravity-loaded simple spans, so the single-unbraced-
    segment Cb of 12.5/11 (AISC F1-1, parabolic diagram) applies when unbraced.
    """
    def ratio(override: float | None, fallback: float) -> float:
        return fallback if override is None else override

    g_live = ratio(config.girder_defl_live_ratio, config.defl_live_ratio)
    g_total = ratio(config.girder_defl_total_ratio, config.defl_total_ratio)
    p_live = ratio(config.purlin_defl_live_ratio, config.defl_live_ratio)
    p_total = ratio(config.purlin_defl_total_ratio, config.defl_total_ratio)

    sp_in = config.purlin_spacing_actual_ft * FT
    girder_Lb = sp_in if config.girder_Lb_ft is None else config.girder_Lb_ft * FT
    girder_rules = GroupRules(
        Lb_in=girder_Lb,
        check_deflection=config.check_deflection,
        defl_live_ratio=g_live, defl_total_ratio=g_total,
        Cb_simple_span=True,
        camber_in=config.girder_camber_in,
    )
    rules = {
        COLUMN: GroupRules(
            check_deflection=False,   # columns: no sag check (they report 0 anyway)
            check_slenderness=config.enforce_slenderness_limit,
        ),
        GIRDER: girder_rules,
        PURLIN: GroupRules(
            Lb_in=None if config.purlin_Lb_ft is None else config.purlin_Lb_ft * FT,
            check_deflection=config.check_deflection,
            defl_live_ratio=p_live, defl_total_ratio=p_total,
            Cb_simple_span=True,
        ),
    }
    if config.has_end_girder_group:
        # same bracing/serviceability rules as the interior girders, but no
        # camber: gable-column support makes their effective spans short
        rules[END_GIRDER] = GroupRules(
            Lb_in=girder_Lb,
            check_deflection=config.check_deflection,
            defl_live_ratio=g_live, defl_total_ratio=g_total,
            Cb_simple_span=True,
        )
    return CheckParams(Fy=config.Fy_ksi, Fu=config.Fu_ksi, E=config.E_ksi,
                       group_rules=rules)


def build_clear_span_geometry(config: ClearSpanConfig) -> FrameGeometry:
    span = config.span_ft * FT
    height = config.eave_height_ft * FT
    s_f = config.frame_spacing_ft * FT
    n_sp = config.n_purlin_spaces
    sp = span / n_sp
    nf = config.n_frames
    end_frames = (0, nf - 1)

    nodes: list[NodeInfo] = []
    members: list[MemberInfo] = []

    for j in range(nf):
        z = j * s_f
        for side, x in ((0, 0.0), (1, span)):
            nodes.append(NodeInfo(f"NB{side}.{j}", x, 0.0, z, is_base=True))
            nodes.append(NodeInfo(f"NE{side}.{j}", x, height, z, is_base=False))
        # interior purlin-line nodes sit on the girder axis: Pynite splits the
        # physical girder there, and the continuous girder provides their
        # rotational stiffness (free_rotations - see analysis/frame_model.py)
        for i in range(1, n_sp):
            nodes.append(NodeInfo(f"NP{i}.{j}", i * sp, height, z,
                                  is_base=False, free_rotations=True))

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
        # girders carry only self-weight directly; ALL roof load arrives as
        # purlin point reactions at the shared nodes
        members.append(MemberInfo(
            name=f"G{j}", group=girder_group(j),
            i_node=f"NE0.{j}", j_node=f"NE1.{j}",
            length_in=span, story=1, trib_width_in=0.0,
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
