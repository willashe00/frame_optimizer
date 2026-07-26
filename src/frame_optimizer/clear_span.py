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

Truss girders (girder_system="truss", or the "auto" fallback):

Rolled W girders top out around a 90-100 ft clear span (deflection needs
Ix ~ span^4, and even the deepest W44 falls short well before 170 ft). With
girder_system="auto" (the default), optimize_layout() prefers the W girders
and switches to trusses only when no candidate W-shape can carry the clear
span. The girder system also sets the FRAME BEHAVIOR: W girders keep the
fully pinned gravity frame; truss girders make every frame a RIGID
transverse bent. In truss mode every frame carries a parallel-chord Pratt
truss — the custom-fabricated analog of SJI DLH/SLH long-span joists, with
W-shape chords and webs as is customary for heavy long-span roof trusses:

* Rigid bent, mill-building style: the columns run full height from the
  base to the top-chord level, and the truss ties into each column at BOTH
  chord elevations — bottom chord at the eave, top chord at the column top.
  Each tie is a pin, but the pair of them (one truss depth apart) IS the
  moment connection: the chord-force couple restrains the truss end
  rotation and the column resists in bending, so the bent carries its own
  in-plane stability. eave_height_ft stays the true clear height under the
  bottom chord; the truss depth (~span/12) rises above it and the purlins
  ride the top chord. Verticals sit at interior panel points; diagonals
  descend toward midspan, putting them in tension (verticals in
  compression).
* Because the bent is self-stable in its plane, strength design follows the
  AISC 360 Chapter C Direct Analysis Method: 0.8-reduced stiffness,
  notional lateral loads (0.003*Yi), second-order P-Delta analysis, and
  K = 1 member checks. Serviceability uses a parallel nominal-stiffness
  model. See analysis/frame_model.py. Frame action also reverses chord
  forces near the frame corners (bottom chord: midspan tension, end
  compression) — the checker verifies both signed axial extremes.
* Both chords are single Pynite physical members over the full span —
  continuous through their panel and purlin nodes, moment-released only at
  the ends. Purlin lines that miss a panel point simply load the continuous
  top chord in local bending, which the physical member captures directly,
  and the chord-relative sag of either chord IS the truss deflection.
* Truss-frame nodes keep their X translation (NodeInfo.free_dx): the
  blanket DX mechanism restraint would absorb the diagonals' horizontal
  components AND hide the frame action; the bent supplies its own X
  stiffness. Under gravity the frame develops real horizontal base
  reactions (thrust) — exported for anchorage design.
* Design groups: 'truss_top_chord' and 'truss_bot_chord' (one candidate list,
  screened independently) and 'truss_web'. Chord compression uses segment
  effective lengths (panel in plane, brace spacing out of plane) via the
  GroupRules KLx/KLy overrides. The bottom chord is assumed braced by
  bridging lines at every panel point unless truss_bottom_brace_ft says
  otherwise — bridging is required in practice (as SJI mandates for joists)
  and carries no gravity load, so it is not modeled.
* v1 scope: end frames carry the same trusses (at half tributary width);
  gable columns / the end_girder group are not available with trusses.
  Wind/seismic remain out of scope for BOTH systems: the rigid bents are
  designed for gravity + code-required stability effects only, and a
  building-level lateral design must still be provided.

Interface units are feet and psf (use M_TO_FT for metric plan dimensions);
everything internal is kips and inches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import AUTO, COLUMN, FT, TRUSS, WIDE_FLANGE
from .design import CheckParams, GroupRules
from .geometry import FrameGeometry, MemberInfo, NodeInfo

GIRDER = "girder"
END_GIRDER = "end_girder"
PURLIN = "purlin"
TRUSS_TOP_CHORD = "truss_top_chord"
TRUSS_BOT_CHORD = "truss_bot_chord"
TRUSS_WEB = "truss_web"

# girder_system values (defined in config.py, re-exported here). AUTO (the
# default) prefers rolled W girders and falls back to truss girders only when
# no candidate W-shape can carry the clear span (see optimize_layout); the
# other two pin the system explicitly.

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

# ---------------------------------------------------------------------------
# Truss practice bands (parallel-chord roof trusses).
#
# * Depth: economical span/depth for long clear-span roof trusses runs
#   ~10-15 (span/12 target). Long-span joist catalogs push shallower, but at
#   the spans that force a truss here, deflection rewards depth.
# * Panels: diagonals work best near 45 degrees, so the target panel length
#   equals the depth. An even panel count keeps the Pratt diagonal layout
#   mirror-symmetric about midspan (odd counts still build; the center panel
#   diagonal is then assigned the left-half orientation).
# ---------------------------------------------------------------------------
TARGET_TRUSS_SPAN_TO_DEPTH = 12.0
TRUSS_SPAN_TO_DEPTH_OPTIONS = (10.0, 12.0, 15.0)   # depth search band
MIN_TRUSS_PANELS = 4

_LAYOUT_FIELDS = ("n_frames", "purlin_spacing_ft", "end_wall_columns")
# truss proportions join the layout search when auto-derived in truss mode
_TRUSS_LAYOUT_FIELDS = ("truss_depth_ft", "truss_panel_ft")


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


def derive_truss_depth_ft(span_ft: float) -> float:
    """Truss depth at the practice-band target (span/12)."""
    return span_ft / TARGET_TRUSS_SPAN_TO_DEPTH


def derive_truss_panel_ft(span_ft: float, depth_ft: float) -> float:
    """Panel length from an even panel count that puts the diagonals as close
    to 45 degrees (panel = depth) as the span allows."""
    n_panels = max(MIN_TRUSS_PANELS, 2 * round(span_ft / depth_ft / 2.0))
    return span_ft / n_panels


def candidate_truss_depths(config: "ClearSpanConfig") -> list[float]:
    """Depth options for the truss layout search: the span/10..span/15
    practice band when the depth is auto-derived, else the pinned value."""
    if "truss_depth_ft" in config.auto_layout_fields:
        return [config.span_ft / r for r in TRUSS_SPAN_TO_DEPTH_OPTIONS]
    return [config.truss_depth_ft]


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

    # --- girder system: automatic (default), rolled W girders, or trusses ---
    # "auto": optimize_layout() prefers the rolled W girders and falls back to
    # parallel-chord Pratt trusses (see module docstring) ONLY when no
    # candidate W-shape can carry the clear span (~>90-100 ft, deflection
    # governs) — supply truss_chord/web_candidates to enable the fallback.
    # "truss" pins every frame to trusses (girder_candidates then ignored,
    # may be []); "wide_flange" pins rolled girders and rejects the truss
    # fields. girder_Lb_ft, girder_camber_in, and the girder deflection
    # ratios apply to the chords when trusses are used.
    girder_system: str = AUTO
    truss_chord_candidates: list[str] | None = None   # both chords; screened
                                                      # independently per group
    truss_web_candidates: list[str] | None = None     # verticals + diagonals
    truss_depth_ft: float | None = None    # None -> span/12 (practice target)
    truss_panel_ft: float | None = None    # target; actual = span/n_panels
                                           # (None -> even count, diagonals ~45 deg)
    truss_bottom_brace_ft: float | None = None   # bottom-chord lateral bracing
                                                 # (bridging) spacing; None ->
                                                 # braced at every panel point

    def __post_init__(self) -> None:
        if self.girder_system not in (AUTO, WIDE_FLANGE, TRUSS):
            raise ValueError(
                f"girder_system must be one of '{AUTO}', '{WIDE_FLANGE}', "
                f"'{TRUSS}', got {self.girder_system!r}."
            )
        required = ["purlin_candidates", "column_candidates"]
        if self.girder_system == TRUSS:
            required.extend(["truss_chord_candidates", "truss_web_candidates"])
        else:
            required.append("girder_candidates")   # 'auto' prefers W girders
        for name in required:
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty.")
        truss_fields = ("truss_chord_candidates", "truss_web_candidates",
                        "truss_depth_ft", "truss_panel_ft",
                        "truss_bottom_brace_ft")
        if self.girder_system == WIDE_FLANGE:
            set_anyway = [n for n in truss_fields if getattr(self, n) is not None]
            if set_anyway:
                raise ValueError(
                    f"{set_anyway} only apply with girder_system='{TRUSS}' "
                    f"(or the '{AUTO}' fallback).")
        if self.girder_system == AUTO:
            has_chords = self.truss_chord_candidates is not None
            has_webs = self.truss_web_candidates is not None
            if has_chords != has_webs:
                raise ValueError(
                    "provide truss_chord_candidates and truss_web_candidates "
                    "together — the truss fallback needs both.")
            if has_chords and not (self.truss_chord_candidates
                                   and self.truss_web_candidates):
                raise ValueError(
                    "truss candidate lists must be non-empty when given.")
            if not has_chords:
                orphaned = [n for n in truss_fields[2:]
                            if getattr(self, n) is not None]
                if orphaned:
                    raise ValueError(
                        f"{orphaned} require truss_chord_candidates and "
                        "truss_web_candidates (the truss fallback needs "
                        "sections to build from).")
        if self.girder_system == TRUSS and self.end_girder_candidates is not None:
            raise ValueError(
                "end_girder_candidates is not supported with "
                f"girder_system='{TRUSS}': end frames carry the same trusses "
                "(at half tributary width), so there is no separate end-girder "
                "group to lighten and no gable columns."
            )
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
        # which ones so optimize_layout() knows its free search variables. In
        # truss mode the truss proportions are layout too: an auto-derived
        # depth ranges over the span/10..span/15 band in the search.
        tracked = _LAYOUT_FIELDS
        if self.girder_system == TRUSS:
            tracked = tracked + _TRUSS_LAYOUT_FIELDS
        self._auto_layout = frozenset(
            name for name in tracked if getattr(self, name) is None)

        # Truss proportions derive from the (normalized) span like the layout
        # fields do; explicit values are honored as-is.
        if self.girder_system == TRUSS:
            if self.truss_depth_ft is None:
                self.truss_depth_ft = derive_truss_depth_ft(self.span_ft)
            if self.truss_depth_ft <= 0:
                raise ValueError("truss_depth_ft must be positive.")
            if self.truss_panel_ft is None:
                self.truss_panel_ft = derive_truss_panel_ft(
                    self.span_ft, self.truss_depth_ft)
            if not (0.0 < self.truss_panel_ft <= self.span_ft / 2.0):
                raise ValueError("truss_panel_ft must be in (0, span_ft/2].")
            if self.truss_bottom_brace_ft is not None and self.truss_bottom_brace_ft <= 0:
                raise ValueError("truss_bottom_brace_ft must be positive.")
        if self.n_frames is None:
            self.n_frames = derive_n_frames(self.length_ft)
        if self.purlin_spacing_ft is None:
            self.purlin_spacing_ft = derive_purlin_spacing_ft(self.span_ft)
        if self.end_wall_columns is None:
            self.end_wall_columns = derive_end_wall_columns(
                self.span_ft, self.has_end_girder_group)

        if self.girder_system == TRUSS and self.end_wall_columns:
            raise ValueError(
                "end_wall_columns > 0 is not supported with "
                f"girder_system='{TRUSS}' (end frames carry full trusses)."
            )
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

    # --- truss proportions (girder_system="truss" only) ---
    @property
    def n_truss_panels(self) -> int:
        return max(2, round(self.span_ft / self.truss_panel_ft))

    @property
    def truss_panel_actual_ft(self) -> float:
        return self.span_ft / self.n_truss_panels

    @property
    def truss_diagonal_deg(self) -> float:
        """Diagonal inclination from horizontal (~45 deg at the target
        proportions)."""
        return math.degrees(
            math.atan2(self.truss_depth_ft, self.truss_panel_actual_ft))

    @property
    def candidates_by_group(self) -> dict[str, list[str]]:
        """Candidate section labels per design group. Key order sets the
        reporting order in results and the wireframe legend."""
        groups: dict[str, list[str]] = {COLUMN: self.column_candidates}
        if self.girder_system == TRUSS:
            groups[TRUSS_TOP_CHORD] = self.truss_chord_candidates
            groups[TRUSS_BOT_CHORD] = self.truss_chord_candidates
            groups[TRUSS_WEB] = self.truss_web_candidates
        else:
            groups[GIRDER] = self.girder_candidates
            if self.has_end_girder_group:
                groups[END_GIRDER] = self.end_girder_candidates
        groups[PURLIN] = self.purlin_candidates
        return groups

    def describe(self) -> list[str]:
        gable = (f", {self.end_wall_columns} gable column(s)/end wall"
                 if self.end_wall_columns else "")
        camber = (f", girder camber {self.girder_camber_in} in"
                  if self.girder_camber_in else "")
        girder_word = "truss" if self.girder_system == TRUSS else "girder"
        lines = [
            f"Frame:  clear span {self.span_ft:.1f} ft x length {self.length_ft:.1f} ft, "
            f"{self.n_frames} frames @ {self.frame_spacing_ft:.1f} ft, "
            f"eave {self.eave_height_ft:.1f} ft (NO interior columns{gable})",
            f"Roof:   purlins @ {self.purlin_spacing_actual_ft:.2f} ft "
            f"({self.n_purlin_spaces + 1} lines), one-way deck -> purlin -> {girder_word}"
            f"{camber}",
            f"Loads:  SDL = {self.superimposed_dead_psf} psf, "
            f"roof L/S = {self.live_psf} psf (1.4D, 1.2D+1.6L) + self-weight",
        ]
        if self.girder_system == TRUSS:
            lines.insert(2, (
                f"Truss:  parallel-chord Pratt, depth {self.truss_depth_ft:.1f} ft "
                f"above the eave, {self.n_truss_panels} panels @ "
                f"{self.truss_panel_actual_ft:.1f} ft (diagonals "
                f"{self.truss_diagonal_deg:.0f} deg); rigid bents, AISC "
                "Direct Analysis (0.8EI, 0.003Yi notional, P-Delta)"))
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

    Truss girders: the chords are full-span physical members, so their
    compression effective lengths come from the GroupRules KLx/KLy overrides
    instead of the member length — one panel in plane (web members restrain
    the chord at every panel point) and the lateral brace spacing out of
    plane (purlin lines for the top chord; bottom-chord bridging, assumed at
    every panel point unless truss_bottom_brace_ft is set). The chords carry
    the deflection check (their chord-relative sag IS the truss sag) and the
    camber credit. Webs are pin-ended axial members checked at their full
    length; their KL/r <= 200 proportioning check stays off like every
    flexural group's, because the E3/E7 strength check already penalizes
    slender compression members (see GroupRules).
    """
    def ratio(override: float | None, fallback: float) -> float:
        return fallback if override is None else override

    g_live = ratio(config.girder_defl_live_ratio, config.defl_live_ratio)
    g_total = ratio(config.girder_defl_total_ratio, config.defl_total_ratio)
    p_live = ratio(config.purlin_defl_live_ratio, config.defl_live_ratio)
    p_total = ratio(config.purlin_defl_total_ratio, config.defl_total_ratio)

    sp_in = config.purlin_spacing_actual_ft * FT
    girder_Lb = sp_in if config.girder_Lb_ft is None else config.girder_Lb_ft * FT
    rules = {
        COLUMN: GroupRules(
            check_deflection=False,   # columns: no sag check (they report 0 anyway)
            check_slenderness=config.enforce_slenderness_limit,
        ),
    }
    if config.girder_system == TRUSS:
        panel_in = config.truss_panel_actual_ft * FT
        bot_brace = (panel_in if config.truss_bottom_brace_ft is None
                     else config.truss_bottom_brace_ft * FT)
        chord_common = dict(
            check_deflection=config.check_deflection,
            defl_live_ratio=g_live, defl_total_ratio=g_total,
            camber_in=config.girder_camber_in,
            KLx_in=panel_in,
        )
        rules[TRUSS_TOP_CHORD] = GroupRules(
            Lb_in=girder_Lb, KLy_in=girder_Lb, **chord_common)
        rules[TRUSS_BOT_CHORD] = GroupRules(
            Lb_in=bot_brace, KLy_in=bot_brace, **chord_common)
        rules[TRUSS_WEB] = GroupRules(check_deflection=False)
    else:
        rules[GIRDER] = GroupRules(
            Lb_in=girder_Lb,
            check_deflection=config.check_deflection,
            defl_live_ratio=g_live, defl_total_ratio=g_total,
            Cb_simple_span=True,
            camber_in=config.girder_camber_in,
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
    rules[PURLIN] = GroupRules(
        Lb_in=None if config.purlin_Lb_ft is None else config.purlin_Lb_ft * FT,
        check_deflection=config.check_deflection,
        defl_live_ratio=p_live, defl_total_ratio=p_total,
        Cb_simple_span=True,
    )
    return CheckParams(Fy=config.Fy_ksi, Fu=config.Fu_ksi, E=config.E_ksi,
                       group_rules=rules)


def build_clear_span_geometry(config: ClearSpanConfig,
                              n_frames: int | None = None) -> FrameGeometry:
    """Building geometry; `n_frames` overrides the frame COUNT while keeping
    the config's frame spacing — used for the exact representative-strip
    analysis (see optimization/optimizer.py), where a 3-frame strip stands in
    for the full building."""
    if config.girder_system == TRUSS:
        return _build_truss_clear_span(config, n_frames)
    span = config.span_ft * FT
    height = config.eave_height_ft * FT
    s_f = config.frame_spacing_ft * FT
    n_sp = config.n_purlin_spaces
    sp = span / n_sp
    nf = config.n_frames if n_frames is None else n_frames
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


def _build_truss_clear_span(config: ClearSpanConfig,
                            n_frames: int | None = None) -> FrameGeometry:
    """Truss-girder variant: every frame is a RIGID transverse bent — two
    full-height perimeter columns with a parallel-chord Pratt truss tied to
    each at BOTH chord levels (see module docstring).

    Per frame j (k = panel count, p = panel length, d = truss depth):

    * Columns are single physical members from the base to the top-chord
      level (eave + d), continuous through the eave node NE where the bottom
      chord ties in; the top chord ties in at the column top NT0/NTk. Both
      connections are pins, but the pair of them — one truss depth apart —
      forms the moment connection: the chord-force couple restrains the
      truss end rotation and the column resists in bending. This is the
      classic mill-bent detail; no end-post web members exist (the column
      segment between eave and top chord does that job).
    * Both chords are single full-span physical members, continuous through
      their interior nodes. Top-chord panel nodes NT0..NTk; bottom-chord
      interior panel nodes NC1..NC(k-1). Verticals TVi at interior panel
      points; diagonals TDi descend toward midspan (tension under gravity),
      meeting in a V at the center bottom node for even k.
    * Purlin lines ride the top chord: lines on a panel point share its node,
      others get NP nodes that subdivide the physical chord, loading it in
      local bending exactly as purlin nodes load a W girder.
    * DOFs: EVERY non-base node of a truss frame keeps DX (free_dx) — the
      bent provides its own in-plane stiffness, and the blanket mechanism
      restraint would both absorb the diagonals' horizontal components and
      hide the frame action the rigid system is designed for. Nodes a
      continuous member passes through (chord interiors, the eave point on
      the column) also keep their rotations; nodes where every member end is
      moment-released (column tops, bases) stay rotationally clamped.
      Analysis of this system follows the AISC Direct Analysis Method — see
      analysis/frame_model.py.

    Note for candidate screening: check_member projects FEA deflections onto
    candidate sections with the Ix ratio, but truss sag actually scales with
    the chord AREA (delta ~ 1/(A*d^2)). The screening step is therefore only
    approximate for the chord groups; the converged design is re-analyzed
    with its own sections, so the certified check table is exact.
    """
    span = config.span_ft * FT
    height = config.eave_height_ft * FT
    depth = config.truss_depth_ft * FT
    s_f = config.frame_spacing_ft * FT
    n_sp = config.n_purlin_spaces
    sp = span / n_sp
    k = config.n_truss_panels
    p = span / k
    y_top = height + depth
    nf = config.n_frames if n_frames is None else n_frames

    nodes: list[NodeInfo] = []
    members: list[MemberInfo] = []

    def top_node(i: int, j: int) -> str:
        return f"NT{i}.{j}"

    def bot_node(i: int, j: int) -> str:
        if i == 0:
            return f"NE0.{j}"
        if i == k:
            return f"NE1.{j}"
        return f"NC{i}.{j}"

    purlin_node: dict[tuple[int, int], str] = {}
    for j in range(nf):
        z = j * s_f
        for side, x in ((0, 0.0), (1, span)):
            nodes.append(NodeInfo(f"NB{side}.{j}", x, 0.0, z, is_base=True))
            # eave node: the continuous column passes through (rotations
            # stabilized by it), the bottom chord ties in here
            nodes.append(NodeInfo(f"NE{side}.{j}", x, height, z,
                                  is_base=False, free_rotations=True,
                                  free_dx=True))
        for i in range(k + 1):
            nodes.append(NodeInfo(top_node(i, j), i * p, y_top, z,
                                  is_base=False, free_rotations=(0 < i < k),
                                  free_dx=True))
        for i in range(1, k):
            nodes.append(NodeInfo(f"NC{i}.{j}", i * p, height, z,
                                  is_base=False, free_rotations=True,
                                  free_dx=True))
        # purlin-line nodes on the top chord; a line on a panel point reuses it
        for i in range(1, n_sp):
            x = i * sp
            panel = round(x / p)
            if abs(x - panel * p) < _COINCIDENT_TOL_IN:
                purlin_node[i, j] = top_node(panel, j)
            else:
                purlin_node[i, j] = f"NP{i}.{j}"
                nodes.append(NodeInfo(f"NP{i}.{j}", x, y_top, z,
                                      is_base=False, free_rotations=True,
                                      free_dx=True))

    diag_len = math.hypot(p, depth)
    for j in range(nf):
        # full-height columns: base -> top chord, through the eave node
        for side, top in ((0, top_node(0, j)), (1, top_node(k, j))):
            members.append(MemberInfo(
                name=f"C{side}.{j}", group=COLUMN,
                i_node=f"NB{side}.{j}", j_node=top,
                length_in=height + depth, story=1, trib_width_in=0.0,
            ))
        # chords carry only self-weight directly; roof load arrives as purlin
        # point reactions on the top chord, exactly like the W-girder case
        members.append(MemberInfo(
            name=f"TC{j}", group=TRUSS_TOP_CHORD,
            i_node=top_node(0, j), j_node=top_node(k, j),
            length_in=span, story=1, trib_width_in=0.0,
        ))
        members.append(MemberInfo(
            name=f"BC{j}", group=TRUSS_BOT_CHORD,
            i_node=f"NE0.{j}", j_node=f"NE1.{j}",
            length_in=span, story=1, trib_width_in=0.0,
        ))
        for i in range(1, k):
            members.append(MemberInfo(
                name=f"TV{i}.{j}", group=TRUSS_WEB,
                i_node=bot_node(i, j), j_node=top_node(i, j),
                length_in=depth, story=1, trib_width_in=0.0,
            ))
        for i in range(k):
            if i < k / 2.0:   # left half: down toward midspan
                i_nd, j_nd = top_node(i, j), bot_node(i + 1, j)
            else:             # right half: mirror image
                i_nd, j_nd = top_node(i + 1, j), bot_node(i, j)
            members.append(MemberInfo(
                name=f"TD{i}.{j}", group=TRUSS_WEB,
                i_node=i_nd, j_node=j_nd,
                length_in=diag_len, story=1, trib_width_in=0.0,
            ))

    def line_node(i: int, j: int) -> str:
        if i == 0:
            return top_node(0, j)
        if i == n_sp:
            return top_node(k, j)
        return purlin_node[i, j]

    for i in range(n_sp + 1):
        trib = sp if 0 < i < n_sp else sp / 2.0   # eave lines carry half a space
        for j in range(nf - 1):
            members.append(MemberInfo(
                name=f"P{i}.b{j}", group=PURLIN,
                i_node=line_node(i, j), j_node=line_node(i, j + 1),
                length_in=s_f, story=1, trib_width_in=trib,
            ))

    return FrameGeometry(nodes=nodes, members=members)
