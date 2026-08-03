"""Pynite FEA of the gravity frame: fully pinned, or rigid truss bents.

Two analysis modes, keyed off the config's girder system:

* Pinned (grid frames and wide-flange clear-span buildings): one linear
  model, the long-standing scheme described below.
* Rigid (clear-span buildings with truss girders): each transverse frame is
  a rigid bent — full-height columns with both truss chords tied in — that
  provides its own in-plane stability, so strength design follows the AISC
  360 Chapter C Direct Analysis Method:
    - stiffness reduced to 0.8*EI / 0.8*EA (C2.3; applied to all members via
      E* = 0.8E, as the C2.3 user note permits, avoiding artificial
      distortion),
    - notional lateral loads Ni = 0.003*Yi per gravity case at each frame's
      top corners (0.002*Yi per C2.2b plus 0.001*Yi per C2.3(c) in lieu of
      the tau_b stiffness iteration),
    - second-order P-Delta analysis (Pynite analyze_PDelta; the interior
      nodes that subdivide the physical chords and columns also let the
      geometric stiffness capture member-curvature P-delta effects),
    - member checks then use K = 1 (AISC C3).
  Serviceability deflections come from a parallel nominal-stiffness linear
  model without notional loads, per AISC C2.3 (reduced stiffness applies to
  strength/stability, not serviceability). Notional loads are stability
  devices, not real loads — exported reactions use the nominal model.

Modeling scheme (see README "Engineering assumptions"):

* Every member has its end bending rotations (Ry, Rz) released at both ends —
  the frame is fully pinned. Torsion stays attached to avoid a spin mechanism.
* A fully pinned frame is a lateral mechanism, so every node is restrained in
  DX, DZ and all rotations (except nodes flagged free_rotations, which sit on
  the interior of a continuous physical member — e.g. purlin points along a
  clear-span girder — and get rotational stiffness from the member itself);
  base nodes are additionally restrained in DY. Because all members are
  moment-released at their ends, these restraints attract no member force
  under gravity — they only remove mechanism DOFs. Lateral stability is
  explicitly out of scope (assumed provided by a separate system).
* Nodes flagged free_dx keep their X translation: every non-base node of a
  rigid truss bent. Unlike a beam node, a truss-frame node MUST equilibrate
  the diagonals' horizontal components through the chords — a DX restraint
  there would absorb them, falsify every chord force, and suppress the frame
  action the rigid bent is designed for. The bent supplies its own in-plane
  stiffness, and its columns deliver real horizontal base reactions (thrust)
  under gravity.
* Loads: 'D' = member self-weight + superimposed dead on beams (one-way
  tributary line load); 'L' = live on beams. LRFD combos 1.4D and 1.2D+1.6L
  for strength; D+L and L for serviceability.
* Pynite sign conventions (verified against hand calculations): member axial
  is POSITIVE IN COMPRESSION; a sagging major-axis moment is negative Mz;
  'dy' deflection is the absolute displacement (includes column shortening),
  so serviceability uses deflection relative to the member-end chord.

Units: kips and inches throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from Pynite import FEModel3D

from ..config import PLF_TO_KIP_PER_IN, TRUSS, FrameConfig
from ..geometry import FrameGeometry
from ..sections import WShape

STRENGTH_COMBOS = {"1.4D": {"D": 1.4}, "1.2D+1.6L": {"D": 1.2, "L": 1.6}}
SERVICE_TOTAL_COMBO = ("D+L", {"D": 1.0, "L": 1.0})
SERVICE_LIVE_COMBO = ("L", {"L": 1.0})

STEEL_DENSITY_KCI = 0.490 / 12.0**3  # kip/in^3, only used for Pynite's material record

# --- AISC 360 Chapter C Direct Analysis Method (rigid truss bents only) ---
STIFFNESS_REDUCTION = 0.8     # C2.3: 0.8*EI and 0.8*EA on all members
NOTIONAL_RATIO = 0.003        # C2.2b 0.002*Yi + C2.3(c) 0.001*Yi (tau_b = 1)
_NOTIONAL_CASE = {"D": "ND", "L": "NL"}   # gravity case -> its notional case


def _rigid_frame(config) -> bool:
    """Rigid transverse frames get the DAM analysis: truss-girder buildings
    (rigid bents) and W-girder buildings whose lateral phase moment-connects
    the knees (transverse_moment_frame — see lateral_system.py). Everything
    else is the fully pinned gravity frame."""
    return (getattr(config, "girder_system", None) == TRUSS
            or getattr(config, "transverse_moment_frame", False))


@dataclass(frozen=True)
class MemberDemand:
    """Design actions for one member, enveloped over the strength combos.

    Sign convention: Pu > 0 tension, Pu < 0 compression (opposite of Pynite).
    Deflections are relative to the member-end chord, service-level, positive
    down. Ix_used records the analysis section so the checker can project
    deflections onto other candidate sections (delta ~ 1/I).
    """
    name: str
    group: str
    story: int
    length_in: float
    trib_width_in: float
    shape_used: str
    Ix_used: float
    Pu: float            # kip, signed extreme of larger magnitude (reporting)
    Mux: float           # kip-in, max |major-axis moment| along the member
    Muy: float           # kip-in, max |minor-axis moment| along the member
    Vu: float            # kip, max |web shear| along the member
    defl_total_in: float  # under D+L
    defl_live_in: float   # under L
    # signed axial extremes across combos AND along the member, so members
    # with force reversal (rigid-bent truss chords: midspan tension, end
    # compression) get checked against BOTH capacities. None -> derived from
    # Pu (legacy constructors).
    Pu_min: float | None = None   # most negative (compression extreme), <= 0
    Pu_max: float | None = None   # most positive (tension extreme), >= 0

    @property
    def Pu_compression(self) -> float:
        return min(self.Pu, 0.0) if self.Pu_min is None else self.Pu_min

    @property
    def Pu_tension(self) -> float:
        return max(self.Pu, 0.0) if self.Pu_max is None else self.Pu_max


def _line_load_kip_in(psf: float, trib_width_in: float) -> float:
    """psf x tributary width -> kip/in. (psf * ft = plf; trib is stored in inches.)"""
    return psf * (trib_width_in / 12.0) * PLF_TO_KIP_PER_IN


def _notional_node_loads(geometry: FrameGeometry, assignment: dict[str, WShape],
                         config: FrameConfig) -> list[tuple[str, str, float]]:
    """DAM notional loads: (node, case, Fx_kip) entries.

    Each transverse frame plane (shared z coordinate) gets its own story
    force Ni = NOTIONAL_RATIO * Yi per gravity case, split between the
    frame's two top-corner nodes. Yi sums the frame's gravity load exactly
    as build_model applies it (self-weight + tributary line loads); members
    spanning between frames (purlins) contribute half to each end. One +X
    direction suffices: the frames are symmetric, and the member-force
    envelopes take absolute extremes, so the -X mirror produces the same
    governing demands member-for-member across each design group.
    """
    node_by_name = {n.name: n for n in geometry.nodes}
    frames: dict[float, list] = {}
    for n in geometry.nodes:
        frames.setdefault(round(n.z, 6), []).append(n)

    totals = {z: {"D": 0.0, "L": 0.0} for z in frames}
    for m in geometry.members:
        shape = assignment[m.group]
        w_dead = (shape.weight_plf * PLF_TO_KIP_PER_IN
                  + _line_load_kip_in(config.superimposed_dead_psf, m.trib_width_in))
        w_live = _line_load_kip_in(config.live_psf, m.trib_width_in)
        zi = round(node_by_name[m.i_node].z, 6)
        zj = round(node_by_name[m.j_node].z, 6)
        shares = ((zi, 1.0),) if zi == zj else ((zi, 0.5), (zj, 0.5))
        for z, share in shares:
            totals[z]["D"] += share * w_dead * m.length_in
            totals[z]["L"] += share * w_live * m.length_in

    loads = []
    for z, nodes in frames.items():
        y_top = max(n.y for n in nodes)
        top = [n for n in nodes if abs(n.y - y_top) < 1e-6]
        x_lo = min(n.x for n in top)
        x_hi = max(n.x for n in top)
        corners = [n for n in top
                   if abs(n.x - x_lo) < 1e-6 or abs(n.x - x_hi) < 1e-6]
        for case, total in totals[z].items():
            if total <= 0.0:
                continue
            per_corner = NOTIONAL_RATIO * total / len(corners)
            for n in corners:
                loads.append((n.name, _NOTIONAL_CASE[case], per_corner))
    return loads


def build_model(geometry: FrameGeometry, assignment: dict[str, WShape],
                config: FrameConfig, rigid_strength: bool = False,
                strength_combos: dict[str, dict[str, float]] | None = None) -> FEModel3D:
    """Assemble the Pynite model for one {group: shape} assignment.

    rigid_strength=True builds the DAM strength model for rigid truss bents:
    stiffness reduced to 0.8*E (C2.3) and notional lateral load cases ND/NL
    added, carried into the strength combos with their parent-case factors.
    The default model (nominal stiffness, no notional loads) serves
    serviceability, reaction export, and every pinned-frame analysis.

    strength_combos (None -> the gravity STRENGTH_COMBOS) lets the lateral
    phase register additional combinations, e.g. wind/seismic ones from
    lateral_loads.all_strength_combos(). Factors may reference any load
    case; cases with no loads applied simply contribute nothing.
    """
    combos = STRENGTH_COMBOS if strength_combos is None else strength_combos
    model = FEModel3D()
    E, nu = config.E_ksi, 0.3
    if rigid_strength:
        E *= STIFFNESS_REDUCTION
    model.add_material("steel", E, E / (2.0 * (1.0 + nu)), nu, STEEL_DENSITY_KCI)

    for group, shape in assignment.items():
        # Pynite's add_section signature is (name, A, Iy, Iz, J); local z is the
        # strong axis for a web-vertical member, so Iz takes the shape's Ix.
        model.add_section(group, shape.A, shape.Iy, shape.Ix, shape.J)

    for node in geometry.nodes:
        model.add_node(node.name, node.x, node.y, node.z)
        # Rotations are restrained to remove mechanism DOFs at nodes where
        # every connected member end is moment-released. Nodes on the interior
        # of a continuous physical member (free_rotations=True) must NOT be
        # clamped: the member passing through supplies rotational stiffness,
        # and clamping would falsify its moments and deflections. Likewise,
        # truss-chord nodes and roller bearings (free_dx=True) must keep DX:
        # clamping it would absorb the web members' horizontal components and
        # falsify the chord axial forces.
        rot = not node.free_rotations
        model.def_support(node.name,
                          support_DX=not node.free_dx, support_DY=node.is_base,
                          support_DZ=True,
                          support_RX=rot, support_RY=rot, support_RZ=rot)

    for m in geometry.members:
        model.add_member(m.name, m.i_node, m.j_node, "steel", m.group)
        # bending released at each end unless a moment connection is declared
        # there (fixed_i/fixed_j — portal-frame knees); torsion stays attached
        model.def_releases(m.name,
                           Ryi=not m.fixed_i, Rzi=not m.fixed_i,
                           Ryj=not m.fixed_j, Rzj=not m.fixed_j)

        shape = assignment[m.group]
        w_self = shape.weight_plf * PLF_TO_KIP_PER_IN
        w_dead = w_self + _line_load_kip_in(config.superimposed_dead_psf, m.trib_width_in)
        model.add_member_dist_load(m.name, "FY", -w_dead, -w_dead, case="D")
        w_live = _line_load_kip_in(config.live_psf, m.trib_width_in)
        if w_live > 0.0:
            model.add_member_dist_load(m.name, "FY", -w_live, -w_live, case="L")

    if rigid_strength:
        for node, case, fx in _notional_node_loads(geometry, assignment, config):
            model.add_node_load(node, "FX", fx, case=case)

    for name, factors in combos.items():
        if rigid_strength:
            # notional loads track their parent gravity case's factor (C2.2b)
            factors = {**factors,
                       **{_NOTIONAL_CASE[c]: f for c, f in factors.items()
                          if c in _NOTIONAL_CASE}}
        model.add_load_combo(name, factors)
    if not rigid_strength:
        # service combos live in the nominal-stiffness model only; the DAM
        # strength model is never read for serviceability
        for name, factors in (SERVICE_TOTAL_COMBO, SERVICE_LIVE_COMBO):
            model.add_load_combo(name, factors)

    return model


def _abs_extreme(member, kind: str, direction: str | None, combo: str) -> float:
    """max(|max|, |min|) of a member diagram for one combo."""
    if direction is None:
        hi = getattr(member, f"max_{kind}")(combo)
        lo = getattr(member, f"min_{kind}")(combo)
    else:
        hi = getattr(member, f"max_{kind}")(direction, combo)
        lo = getattr(member, f"min_{kind}")(direction, combo)
    return max(abs(hi), abs(lo))


def _chord_relative_sag(member, length: float, combo: str, n: int = 20) -> float:
    """Max downward deflection relative to the straight line between the two
    member ends (removes support settlement / column shortening).

    Sampled station-by-station against the chord so it stays exact when the
    two supports settle by different amounts (e.g. a purlin spanning between
    an interior girder and a stiffer end girder), which tilts the deflected
    shape and shifts its minimum away from midspan."""
    # never sample past the member's real end: a stored length that exceeds
    # the node-geometry length by any amount would raise inside Pynite
    length = min(length, member.L())
    d0 = member.deflection("dy", 0.0, combo)
    dL = member.deflection("dy", length, combo)
    worst = 0.0
    for k in range(1, n):
        x = length * k / n
        chord = d0 + (dL - d0) * (x / length)
        worst = max(worst, chord - member.deflection("dy", x, combo))
    return worst


def extract_demands(model: FEModel3D, geometry: FrameGeometry,
                    assignment: dict[str, WShape],
                    service_model: FEModel3D | None = None,
                    strength_combos: Iterable[str] | None = None) -> list[MemberDemand]:
    """Envelope each member's design actions over the strength combos.

    `model` supplies the strength actions; deflections come from
    `service_model` when given (rigid bents: nominal stiffness, no notional
    loads) and from `model` otherwise (pinned frames: one model serves both).
    strength_combos names the combos to envelope (None -> the gravity
    STRENGTH_COMBOS; passing the dict used for build_model works — iterating
    it yields its keys).
    """
    combo_names = list(STRENGTH_COMBOS if strength_combos is None
                       else strength_combos)
    node_y = {n.name: n.y for n in geometry.nodes}
    defl_model = model if service_model is None else service_model
    demands = []
    for m in geometry.members:
        member = model.members[m.name]
        # chord-relative vertical sag only makes sense for horizontal members;
        # vertical members (columns) report 0 and are never deflection-checked
        is_horizontal = abs(node_y[m.i_node] - node_y[m.j_node]) < 1e-6

        Pu_min = Pu_max = 0.0
        Mux = Muy = Vu = 0.0
        for combo in combo_names:
            # Pynite: compression positive -> flip to tension-positive and
            # keep BOTH signed extremes (chords in rigid bents reverse sign
            # along their length: midspan tension, end compression)
            Pu_min = min(Pu_min, -member.max_axial(combo))
            Pu_max = max(Pu_max, -member.min_axial(combo))
            Mux = max(Mux, _abs_extreme(member, "moment", "Mz", combo))
            Muy = max(Muy, _abs_extreme(member, "moment", "My", combo))
            Vu = max(Vu, _abs_extreme(member, "shear", "Fy", combo))
        Pu = Pu_min if -Pu_min >= Pu_max else Pu_max

        if is_horizontal:
            sag_member = defl_model.members[m.name]
            defl_total = _chord_relative_sag(sag_member, m.length_in,
                                             SERVICE_TOTAL_COMBO[0])
            defl_live = _chord_relative_sag(sag_member, m.length_in,
                                            SERVICE_LIVE_COMBO[0])
        else:
            defl_total = defl_live = 0.0

        shape = assignment[m.group]
        demands.append(MemberDemand(
            name=m.name, group=m.group, story=m.story, length_in=m.length_in,
            trib_width_in=m.trib_width_in, shape_used=shape.name, Ix_used=shape.Ix,
            Pu=Pu, Mux=Mux, Muy=Muy, Vu=Vu,
            defl_total_in=defl_total, defl_live_in=defl_live,
            Pu_min=Pu_min, Pu_max=Pu_max,
        ))
    return demands


def analyze_frame(geometry: FrameGeometry, assignment: dict[str, WShape],
                  config: FrameConfig,
                  strength_combos: dict[str, dict[str, float]] | None = None) -> list[MemberDemand]:
    """Build, solve, and extract demands for one candidate assignment.

    Pinned frames: one linear model. Rigid truss bents: the DAM strength
    model (0.8E, notional loads) solved second-order via P-Delta, plus a
    nominal-stiffness linear model for serviceability deflections.
    strength_combos (None -> gravity defaults) is enveloped as-is; see
    build_model.
    """
    if _rigid_frame(config):
        strength = build_model(geometry, assignment, config, rigid_strength=True,
                               strength_combos=strength_combos)
        strength.analyze_PDelta(check_stability=True, sparse=True)
        service = build_model(geometry, assignment, config)
        service.analyze(check_stability=True, check_statics=False, sparse=True)
        return extract_demands(strength, geometry, assignment,
                               service_model=service,
                               strength_combos=strength_combos)
    model = build_model(geometry, assignment, config,
                        strength_combos=strength_combos)
    model.analyze(check_stability=True, check_statics=False, sparse=True)
    return extract_demands(model, geometry, assignment,
                           strength_combos=strength_combos)
