"""Pynite FEA of the pinned gravity frame.

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
* Exception: truss nodes are flagged free_dx — a truss cannot develop chord
  or diagonal axial force if its panel points are held horizontally, so each
  truss frame is supported pin (one bearing keeps DX) + roller (the other
  bearing and all panel nodes release it). DZ stays restrained everywhere.
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

from Pynite import FEModel3D

from ..config import KPA_TO_KSI, MPA_TO_KSI, PLF_TO_KIP_PER_IN, FrameConfig
from ..geometry import FrameGeometry
from ..sections import WShape

STRENGTH_COMBOS = {"1.4D": {"D": 1.4}, "1.2D+1.6L": {"D": 1.2, "L": 1.6}}
SERVICE_TOTAL_COMBO = ("D+L", {"D": 1.0, "L": 1.0})
SERVICE_LIVE_COMBO = ("L", {"L": 1.0})

STEEL_DENSITY_KCI = 0.490 / 12.0**3  # kip/in^3, only used for Pynite's material record


@dataclass(frozen=True)
class MemberDemand:
    """Design actions for one member, enveloped over the strength combos.

    Sign convention: Pu > 0 tension, Pu < 0 compression (opposite of Pynite).
    Deflections are relative to the member-end chord, service-level, positive
    down. Ix_used records the analysis section so the checker can project
    deflections onto other candidate sections (delta ~ 1/I); A_used likewise
    for members whose deflection is governed by axial stiffness (truss
    chords: delta ~ 1/A).
    """
    name: str
    group: str
    story: int
    length_in: float
    trib_width_in: float
    shape_used: str
    Ix_used: float
    Pu: float            # kip
    Mux: float           # kip-in, max |major-axis moment| along the member
    Muy: float           # kip-in, max |minor-axis moment| along the member
    Vu: float            # kip, max |web shear| along the member
    defl_total_in: float  # under D+L
    defl_live_in: float   # under L
    A_used: float = 0.0   # analysis-section area; 0 = not recorded (pseudo-
                          # demands that never take the axial deflection scale)


def _line_load_kip_in(kpa: float, trib_width_in: float) -> float:
    """kPa x tributary width -> kip/in. (kPa converts to ksi = kip/in^2, so
    multiplying by the tributary width in inches gives kip/in directly.)"""
    return kpa * KPA_TO_KSI * trib_width_in


def build_model(geometry: FrameGeometry, assignment: dict[str, WShape],
                config: FrameConfig) -> FEModel3D:
    """Assemble the Pynite model for one {group: shape} assignment."""
    model = FEModel3D()
    E, nu = config.E_mpa * MPA_TO_KSI, 0.3
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
        # and clamping would falsify its moments and deflections.
        rot = not node.free_rotations
        model.def_support(node.name,
                          support_DX=not node.free_dx,
                          support_DY=node.is_base, support_DZ=True,
                          support_RX=rot, support_RY=rot, support_RZ=rot)

    for m in geometry.members:
        model.add_member(m.name, m.i_node, m.j_node, "steel", m.group)
        model.def_releases(m.name, Ryi=True, Rzi=True, Ryj=True, Rzj=True)

        shape = assignment[m.group]
        w_self = shape.weight_plf * PLF_TO_KIP_PER_IN
        w_dead = w_self + _line_load_kip_in(config.superimposed_dead_kpa, m.trib_width_in)
        model.add_member_dist_load(m.name, "FY", -w_dead, -w_dead, case="D")
        w_live = _line_load_kip_in(config.live_kpa, m.trib_width_in)
        if w_live > 0.0:
            model.add_member_dist_load(m.name, "FY", -w_live, -w_live, case="L")

    for name, factors in STRENGTH_COMBOS.items():
        model.add_load_combo(name, factors)
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
    d0 = member.deflection("dy", 0.0, combo)
    dL = member.deflection("dy", length, combo)
    worst = 0.0
    for k in range(1, n):
        x = length * k / n
        chord = d0 + (dL - d0) * (x / length)
        worst = max(worst, chord - member.deflection("dy", x, combo))
    return worst


def extract_demands(model: FEModel3D, geometry: FrameGeometry,
                    assignment: dict[str, WShape]) -> list[MemberDemand]:
    """Envelope each member's design actions over the strength combos."""
    node_y = {n.name: n.y for n in geometry.nodes}
    demands = []
    for m in geometry.members:
        member = model.members[m.name]
        # chord-relative vertical sag only makes sense for horizontal members;
        # vertical members (columns) report 0 and are never deflection-checked
        is_horizontal = abs(node_y[m.i_node] - node_y[m.j_node]) < 1e-6

        Pu = 0.0
        Mux = Muy = Vu = 0.0
        for combo in STRENGTH_COMBOS:
            # Pynite: compression positive -> flip to tension-positive, keeping
            # the sign of whichever extreme has the larger magnitude.
            for p in (-member.max_axial(combo), -member.min_axial(combo)):
                if abs(p) > abs(Pu):
                    Pu = p
            Mux = max(Mux, _abs_extreme(member, "moment", "Mz", combo))
            Muy = max(Muy, _abs_extreme(member, "moment", "My", combo))
            Vu = max(Vu, _abs_extreme(member, "shear", "Fy", combo))

        if is_horizontal:
            defl_total = _chord_relative_sag(member, m.length_in, SERVICE_TOTAL_COMBO[0])
            defl_live = _chord_relative_sag(member, m.length_in, SERVICE_LIVE_COMBO[0])
        else:
            defl_total = defl_live = 0.0

        shape = assignment[m.group]
        demands.append(MemberDemand(
            name=m.name, group=m.group, story=m.story, length_in=m.length_in,
            trib_width_in=m.trib_width_in, shape_used=shape.name, Ix_used=shape.Ix,
            A_used=shape.A,
            Pu=Pu, Mux=Mux, Muy=Muy, Vu=Vu,
            defl_total_in=defl_total, defl_live_in=defl_live,
        ))
    return demands


def solve_model(model: FEModel3D, check_stability: bool = True) -> None:
    """First-order solve of every load combination WITHOUT reaction recovery.

    Pynite's public analyze() spends ~40% of its time in _calc_reactions,
    but the optimizer only reads member results (forces, deflections), never
    reactions — so the screening loop skips that step entirely. Member
    results are identical to analyze(): reactions are recovered FROM the
    solved displacements, not the other way around. Callers that do need
    reactions (the baseplate export) use model.analyze() directly.

    Mirrors FEModel3D.analyze(sparse=True); if Pynite's internal API ever
    changes, falls back to the public analyze().
    """
    try:
        from Pynite import Analysis
        Analysis._prepare_model(model)
        combos = Analysis._identify_combos(model, None)
        D1_indices, D2_indices, D2 = Analysis._partition_D(model)
        for combo in combos:
            FER1, _ = Analysis._partition(model, model.FER(combo.name),
                                          D1_indices, D2_indices)
            P1, _ = Analysis._partition(model, model.P(combo.name),
                                        D1_indices, D2_indices)
            Analysis._first_order(model, combo.name, P1, FER1, D1_indices,
                                  D2_indices, D2, log=False, sparse=True,
                                  check_stability=check_stability)
    except (ImportError, AttributeError, TypeError):
        model.analyze(check_stability=check_stability, check_statics=False,
                      sparse=True)


def solve_model_second_order(model: FEModel3D, check_stability: bool = True) -> None:
    """Geometrically nonlinear (P-Delta) solve of every load combination.

    Runs once on a search winner as the second-order verification (AISC 360
    C2), so the ~40% reaction-recovery overhead of Pynite's public entry
    point is irrelevant — no fast path needed. Service combos are solved
    second-order too, which can only increase the checked deflections."""
    try:
        model.analyze_PDelta(check_stability=check_stability, sparse=True)
    except Exception as exc:   # Pynite raises plain Exception on divergence
        raise RuntimeError(
            "P-Delta analysis did not converge — the design could not be "
            f"verified second-order ({exc})"
        ) from exc


def analyze_frame(geometry: FrameGeometry, assignment: dict[str, WShape],
                  config: FrameConfig,
                  check_stability: bool = True,
                  second_order: bool = False) -> list[MemberDemand]:
    """Build, solve, and extract demands for one candidate assignment.

    check_stability=False skips Pynite's singularity scan; safe when the
    same geometry has already solved cleanly once (stability depends on the
    support/release topology, not on which W-shapes are assigned).
    second_order=True runs the iterative P-Delta solve instead of the fast
    first-order path (used to verify a finished design, not to search)."""
    model = build_model(geometry, assignment, config)
    if second_order:
        solve_model_second_order(model, check_stability=check_stability)
    else:
        solve_model(model, check_stability=check_stability)
    return extract_demands(model, geometry, assignment)
