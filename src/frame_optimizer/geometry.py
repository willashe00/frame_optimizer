"""Grid generation: turn a FrameConfig into nodes and tagged members.

Groups:
    'column' - vertical members, one per story per column line
    'beam'   - ALL horizontal floor members; they share one design section.
               Which ones actually carry floor load is a member property, not
               a group: members perpendicular to the deck span get a one-way
               tributary width, members parallel to it get trib = 0 (they see
               only self-weight until infill beams are implemented).

All lengths in inches.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import BEAM, COLUMN, FT, FrameConfig

GROUPS = (COLUMN, BEAM)   # groups of the conventional grid frame


@dataclass(frozen=True)
class NodeInfo:
    name: str
    x: float
    y: float
    z: float
    is_base: bool
    # True for nodes that sit on the interior of a continuous (physical)
    # member, e.g. purlin-to-girder connections: the member passing through
    # provides rotational stiffness, and clamping these rotations with the
    # mechanism-stabilization supports would falsify its bending behavior.
    free_rotations: bool = False


@dataclass(frozen=True)
class MemberInfo:
    name: str
    group: str          # 'column' | 'beam'
    i_node: str
    j_node: str
    length_in: float
    story: int          # story number for columns, floor level for beams (1-based)
    trib_width_in: float  # one-way tributary width; 0 for columns and for
                          # beams running parallel to the deck span


@dataclass(frozen=True)
class FrameGeometry:
    nodes: list[NodeInfo]
    members: list[MemberInfo]

    def members_in_group(self, group: str) -> list[MemberInfo]:
        return [m for m in self.members if m.group == group]

    @property
    def groups(self) -> tuple[str, ...]:
        """Design groups present in this geometry, in first-appearance order."""
        return tuple(dict.fromkeys(m.group for m in self.members))


def _node_name(ix: int, iz: int, level: int) -> str:
    return f"N{ix}.{iz}.{level}"


def build_geometry(config: FrameConfig) -> FrameGeometry:
    sx = config.x_bay_spacing_ft * FT
    sz = config.z_bay_spacing_ft * FT
    heights = [h * FT for h in config.story_heights_ft]
    levels_y = [0.0]
    for h in heights:
        levels_y.append(levels_y[-1] + h)

    nx = config.x_bays + 1   # column lines in x
    nz = config.z_bays + 1   # column lines in z

    nodes = [
        NodeInfo(_node_name(ix, iz, lvl), ix * sx, levels_y[lvl], iz * sz, is_base=(lvl == 0))
        for ix in range(nx)
        for iz in range(nz)
        for lvl in range(config.stories + 1)
    ]

    def one_way_trib(line_idx: int, n_lines: int, spacing: float) -> float:
        """Tributary width of a support line under a one-way deck: half a bay
        from each adjacent side (edge lines have only one side)."""
        interior = 0 < line_idx < n_lines - 1
        return spacing if interior else spacing / 2.0

    members: list[MemberInfo] = []

    # columns: story s spans level s-1 -> s
    for ix in range(nx):
        for iz in range(nz):
            for s in range(1, config.stories + 1):
                members.append(MemberInfo(
                    name=f"C{ix}.{iz}.s{s}",
                    group=COLUMN,
                    i_node=_node_name(ix, iz, s - 1),
                    j_node=_node_name(ix, iz, s),
                    length_in=heights[s - 1],
                    story=s,
                    trib_width_in=0.0,
                ))

    # horizontal members at every floor level 1..stories; all are 'beam',
    # but only the ones perpendicular to the deck span carry tributary load
    x_running_loaded = config.deck_span_direction == "z"

    for lvl in range(1, config.stories + 1):
        # x-running members: one per x-bay on every z line
        for iz in range(nz):
            trib = one_way_trib(iz, nz, sz) if x_running_loaded else 0.0
            for bx in range(config.x_bays):
                members.append(MemberInfo(
                    name=f"Bx{bx}.{iz}.L{lvl}",
                    group=BEAM,
                    i_node=_node_name(bx, iz, lvl),
                    j_node=_node_name(bx + 1, iz, lvl),
                    length_in=sx,
                    story=lvl,
                    trib_width_in=trib,
                ))
        # z-running members: one per z-bay on every x line
        for ix in range(nx):
            trib = 0.0 if x_running_loaded else one_way_trib(ix, nx, sx)
            for bz in range(config.z_bays):
                members.append(MemberInfo(
                    name=f"Bz{ix}.{bz}.L{lvl}",
                    group=BEAM,
                    i_node=_node_name(ix, bz, lvl),
                    j_node=_node_name(ix, bz + 1, lvl),
                    length_in=sz,
                    story=lvl,
                    trib_width_in=trib,
                ))

    return FrameGeometry(nodes=nodes, members=members)
