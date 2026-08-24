"""Interactive 3-D wireframe of an optimized frame (self-contained HTML).

   No part of frame_optimizer depends on this module."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import plotly.graph_objects as go

from frame_optimizer import geometry_for
from frame_optimizer.config import IN_TO_M, IN_TO_MM, KIP_TO_KN
from frame_optimizer.geometry import BEAM, COLUMN
from frame_optimizer.results import OptimizationResult

if TYPE_CHECKING:                      # type-checking only: drawing the
    from baseplate_design import (BaseplateCheck,  # baseplates is optional,
                                  ColumnDemand,    # so this module never
                                  UniformBaseplateDesign)  # imports it at run time

_GROUP_COLOR = {COLUMN: "#2a78d6", BEAM: "#1baf7a",
                "girder": "#1baf7a", "purlin": "#eda100",
                "end_girder": "#008300",
                # Pratt truss roof: the top chord inherits the girder's
                # primary-flexural slot (they never appear together)
                "top_chord": "#1baf7a", "bottom_chord": "#7a4fd0",
                "truss_web": "#3fb6c9"}
_GROUP_WIDTH = {COLUMN: 6, BEAM: 4, "girder": 5, "purlin": 3, "end_girder": 5,
                "top_chord": 5, "bottom_chord": 5, "truss_web": 2}
_DEFAULT_WIDTH = 4

_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_CRITICAL = "#d03b3b"

_PLATE = "#6b7280"        # steel plate fill
_PLATE_EDGE = "#374151"   # its outline and anchor rods

_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# hover targets per member (evenly spaced, endpoints excluded so joints stay
# unambiguous) and the DCR below which a check row is omitted as negligible
_HOVER_SAMPLES = 9
_SHOW_UC = 0.01

# Baseplate plan orientation. The gravity model does not define column web
# direction (all sections are vertical W-shapes), so the drawing has to pick
# one: N (the plate dimension parallel to the column depth d) is laid along
# model X, the clear-span/girder direction, which is how a portal column is
# normally set. The hover card says so. Swap these to draw the other way.
_PLATE_N_ALONG_X = True


def _notna(v) -> bool:
    return v is not None and v == v          # NaN != NaN


def _hover_card(name: str, group: str, section: str, row) -> str:
    """Concise per-member design summary: size, each governing demand vs.
    capacity with its demand-capacity ratio, and the overall verdict."""
    if row is None:
        return f"<b>{name}</b><br>{group} · {section}"

    verdict = "PASS" if row["PASS"] else "<b>FAIL</b>"
    lines = [
        f"<b>{name}</b> — {group} · <b>{row['profile']}</b>",
        f"L = {row['length_m']:.2f} m · story {int(row['story'])}",
        f"governing DCR <b>{row['governing_uc']:.2f}</b> "
        f"[{row['governing_limitstate']}] · {verdict}",
    ]
    if row["UC_axial"] >= _SHOW_UC:
        kind = "compression" if row["Pu_kN"] < 0 else "tension"
        lines.append(f"P ({kind}): {abs(row['Pu_kN']):,.1f} / "
                     f"{row['phiPn_kN']:,.0f} kN · DCR {row['UC_axial']:.2f}")
    if row["UC_Mx"] >= _SHOW_UC:
        lines.append(f"Mx: {row['Mux_kNm']:,.1f} / "
                     f"{row['phiMnx_kNm']:,.0f} kN·m · DCR {row['UC_Mx']:.2f}")
    if row["UC_My"] >= _SHOW_UC:
        lines.append(f"My: {row['Muy_kNm']:,.1f} / "
                     f"{row['phiMny_kNm']:,.0f} kN·m · DCR {row['UC_My']:.2f}")
    if row["UC_V"] >= _SHOW_UC:
        lines.append(f"V: {row['Vu_kN']:,.1f} / "
                     f"{row['phiVn_kN']:,.0f} kN · DCR {row['UC_V']:.2f}")
    if row["UC_H1"] >= _SHOW_UC:
        lines.append(f"P–M interaction (H1): DCR {row['UC_H1']:.2f}")
    slender = row.get("UC_slenderness")
    if _notna(slender):
        lines.append(f"KL/r ≤ 200: DCR {slender:.2f}")
    d_live, d_total = row.get("UC_defl_live"), row.get("UC_defl_total")
    if _notna(d_live):
        lines.append(f"deflection: live DCR {d_live:.2f} · total DCR {d_total:.2f}")
    return "<br>".join(lines)


def _baseplate_hover_card(design: "UniformBaseplateDesign",
                          demand: "ColumnDemand",
                          check: "BaseplateCheck") -> str:
    """Per-baseplate design summary: the plate itself, this column's demands,
    and the design-critical limit states with their DCRs.
    """
    plate = design.plate
    verdict = "PASS" if check.passes else "<b>FAIL</b>"

    lines = [
        f"<b>baseplate @ {demand.column_id}</b> — pinned · {demand.section_name}",
        f"plate <b>{plate.B * IN_TO_MM:,.0f} × {plate.N * IN_TO_MM:,.0f} × "
        f"{plate.tp * IN_TO_MM:,.1f} mm</b> "
        f"({plate.B:g} × {plate.N:g} × {plate.tp:g} in)",
        f"{plate.n_rods} − {plate.d_rod * IN_TO_MM:,.1f} mm anchor rods · "
        f"edge {plate.edge_distance * IN_TO_MM:,.0f} mm",
        f"A1 = {check.A1 * IN_TO_MM ** 2 / 1e6:,.3f} m² · "
        f"plate mass {design.plate_mass_kg:,.1f} kg",
        "",
        f"Pu = {demand.Pu * KIP_TO_KN:,.1f} kN · "
        f"Vu = {demand.Vu * KIP_TO_KN:,.1f} kN",
        f"governing DCR <b>{check.max_dcr:.2f}</b> "
        f"[{check.governing_limit_state}] · {verdict}",
        f"bearing: {demand.Pu * KIP_TO_KN:,.1f} / {check.phiPp * KIP_TO_KN:,.0f} kN"
        f" · DCR {check.bearing_dcr:.2f}",
        f"plate flexure: t req {check.t_req * IN_TO_MM:,.1f} / "
        f"{plate.tp * IN_TO_MM:,.1f} mm · DCR {check.flexure_dcr:.2f}",
    ]
    return "<br>".join(lines)


def _plate_corners_m(demand: "ColumnDemand", plate) -> list[tuple[float, float]]:
    """The plate's four plan corners in meters, as (model x, model z).

    Centred on the column base node; the plate top is the base elevation, so
    it is drawn flat at y = 0 and hangs below out of sight.
    """
    half_N, half_B = plate.N / 2.0 * IN_TO_M, plate.B / 2.0 * IN_TO_M
    dx, dz = (half_N, half_B) if _PLATE_N_ALONG_X else (half_B, half_N)
    cx, cz = demand.location_in[0] * IN_TO_M, demand.location_in[2] * IN_TO_M
    return [(cx - dx, cz - dz), (cx + dx, cz - dz),
            (cx + dx, cz + dz), (cx - dx, cz + dz)]


def _rod_points_m(demand: "ColumnDemand", plate) -> list[tuple[float, float]]:
    """Anchor rod plan positions in meters, as (model x, model z).

    rod_positions() returns (x along B, y along N); the orientation switch
    decides which of those becomes model X.
    """
    cx, cz = demand.location_in[0] * IN_TO_M, demand.location_in[2] * IN_TO_M
    points = []
    for along_B, along_N in plate.rod_positions():
        dx, dz = ((along_N, along_B) if _PLATE_N_ALONG_X
                  else (along_B, along_N))
        points.append((cx + dx * IN_TO_M, cz + dz * IN_TO_M))
    return points


def _append_ring(xs: list, ys: list, zs: list,
                 corners: list[tuple[float, float]], y0: float) -> None:
    """Append a closed plan ring as a None-terminated scene polyline."""
    for cx, cz in corners + [corners[0]]:
        xs.append(cx)
        ys.append(cz)          # model z -> scene y
        zs.append(y0)          # model y (vertical) -> scene z
    xs.append(None)
    ys.append(None)
    zs.append(None)


def _add_baseplates(fig: go.Figure, design: "UniformBaseplateDesign") -> None:
    """Draw every baseplate as a flat plate at its column base, with the
    anchor rods and a hover card carrying the full design and its checks."""
    plate = design.plate
    checks = {c.column_id: c for c in design.checks}

    # one Mesh3d for all plates: 4 vertices and 2 triangles each
    mx, my, mz, i, j, k = [], [], [], [], [], []
    ox, oy, oz = [], [], []          # outlines
    rx, ry, rz = [], [], []          # anchor rods
    hx, hy, hz, hover = [], [], [], []
    fx, fy, fz = [], [], []          # outlines of plates that fail

    for demand in design.demands:
        check = checks[demand.column_id]
        corners = _plate_corners_m(demand, plate)
        y0 = demand.location_in[1] * IN_TO_M      # base elevation

        base = len(mx)
        for cx, cz in corners:
            mx.append(cx)
            my.append(cz)             # model z -> scene y
            mz.append(y0)             # model y (vertical) -> scene z
        i += [base, base]
        j += [base + 1, base + 2]
        k += [base + 2, base + 3]

        # a plate that fails any check goes to the critical-color outline
        outline = (ox, oy, oz) if check.passes else (fx, fy, fz)
        _append_ring(*outline, corners, y0)

        for cx, cz in _rod_points_m(demand, plate):
            rx.append(cx)
            ry.append(cz)
            rz.append(y0)

        hx.append(demand.location_in[0] * IN_TO_M)
        hy.append(demand.location_in[2] * IN_TO_M)
        hz.append(y0)
        hover.append(_baseplate_hover_card(design, demand, check))

    label = (f"baseplate · {plate.B * IN_TO_MM:,.0f}×{plate.N * IN_TO_MM:,.0f}"
             f"×{plate.tp * IN_TO_MM:,.1f} mm")
    fig.add_trace(go.Mesh3d(
        x=mx, y=my, z=mz, i=i, j=j, k=k,
        color=_PLATE, opacity=0.95, flatshading=True,
        name=label, legendgroup="baseplate", showlegend=True,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        x=ox, y=oy, z=oz, mode="lines",
        line=dict(color=_PLATE_EDGE, width=3),
        legendgroup="baseplate", showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        x=rx, y=ry, z=rz, mode="markers",
        marker=dict(size=2.5, color=_PLATE_EDGE),
        name=f"{plate.n_rods} − {plate.d_rod * IN_TO_MM:,.1f} mm anchor rods",
        legendgroup="baseplate", showlegend=False, hoverinfo="skip",
    ))
    if fx:
        fig.add_trace(go.Scatter3d(
            x=fx, y=fy, z=fz, mode="lines",
            line=dict(color=_CRITICAL, width=6),
            name="✕ baseplate fails checks", hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter3d(
        x=hx, y=hy, z=hz, mode="markers",
        marker=dict(size=14, color=_PLATE, opacity=0.0),
        legendgroup="baseplate", showlegend=False,
        hovertemplate="%{customdata}<extra></extra>", customdata=hover,
        hoverlabel=dict(bgcolor=_SURFACE, bordercolor=_PLATE_EDGE, align="left",
                        font=dict(size=12, color=_INK, family=_FONT)),
    ))


def _member_ends_m(geometry) -> dict[str, tuple]:
    """member name -> ((xi, yi, zi), (xj, yj, zj)) in meters."""
    nodes = {n.name: (n.x * IN_TO_M, n.y * IN_TO_M, n.z * IN_TO_M)
             for n in geometry.nodes}
    return {m.name: (nodes[m.i_node], nodes[m.j_node]) for m in geometry.members}


def _polyline(segments):
    """Concatenate 3-D segments into one None-separated plotly polyline."""
    xs, ys, zs = [], [], []
    for (xi, yi, zi), (xj, yj, zj) in segments:
        xs += [xi, xj, None]
        ys += [zi, zj, None]   # model z -> scene y
        zs += [yi, yj, None]   # model y (vertical) -> scene z (up)
    return xs, ys, zs


def visualize_result(result: OptimizationResult, path: str = "structure_wireframe.html",
                     show: bool = True,
                     baseplates: "UniformBaseplateDesign | None" = None) -> Path:
    """Write a standalone interactive HTML wireframe of the final design.

    Pass `baseplates` (a UniformBaseplateDesign) to draw the column baseplates
    as flat plates at the bases, with their anchor rods and a hover card
    carrying the plate dimensions, demands and every DCR. Omitting it just
    leaves them out, so this module keeps working without baseplate_design.

    Returns the path of the written file; opens it in the browser if `show`.
    """
    if result.config is None:
        raise ValueError("OptimizationResult.config is None; visualize_result "
                         "needs the config to rebuild the frame geometry.")

    geometry = geometry_for(result.config)
    ends = _member_ends_m(geometry)
    checks = {row["member"]: row for _, row in result.member_table.iterrows()}

    fig = go.Figure()

    groups = geometry.groups
    for group in groups:
        members = geometry.members_in_group(group)
        if not members:
            continue
        color = _GROUP_COLOR.get(group, _MUTED)
        section = result.sections[group]

        xs, ys, zs = _polyline([ends[m.name] for m in members])
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=color, width=_GROUP_WIDTH.get(group, _DEFAULT_WIDTH)),
            name=f"{group} · {section}",
            legendgroup=group, hoverinfo="skip",
        ))

        # invisible hover targets spaced along every member (hover anywhere,
        # not just at a "perfect" midpoint) + one section label at midspan
        hx, hy, hz, hover = [], [], [], []
        lx, ly, lz, labels = [], [], [], []
        for m in members:
            (xi, yi, zi), (xj, yj, zj) = ends[m.name]
            card = _hover_card(m.name, group, section, checks.get(m.name))
            for k in range(1, _HOVER_SAMPLES + 1):
                t = k / (_HOVER_SAMPLES + 1)
                hx.append(xi + (xj - xi) * t)
                hy.append(zi + (zj - zi) * t)   # model z -> scene y
                hz.append(yi + (yj - yi) * t)   # model y (vertical) -> scene z
                hover.append(card)
            lx.append((xi + xj) / 2.0)
            ly.append((zi + zj) / 2.0)
            lz.append((yi + yj) / 2.0)
            labels.append(section)

        fig.add_trace(go.Scatter3d(
            x=hx, y=hy, z=hz, mode="markers",
            marker=dict(size=16, color=color, opacity=0.0),
            legendgroup=group, showlegend=False,
            hovertemplate="%{customdata}<extra></extra>", customdata=hover,
            hoverlabel=dict(bgcolor=_SURFACE, bordercolor=color, align="left",
                            font=dict(size=12, color=_INK, family=_FONT)),
        ))
        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz, mode="text",
            text=labels, textposition="top center",
            textfont=dict(size=10, color=_INK_2, family=_FONT),
            name="section labels", legendgroup="labels",
            showlegend=(group == groups[0]), hoverinfo="skip",
        ))

    # pinned bases
    base = [n for n in geometry.nodes if n.is_base]
    fig.add_trace(go.Scatter3d(
        x=[n.x * IN_TO_M for n in base], y=[n.z * IN_TO_M for n in base],
        z=[n.y * IN_TO_M for n in base],
        mode="markers",
        marker=dict(size=4, color=_MUTED, symbol="diamond"),
        name="pinned base", hoverinfo="skip",
    ))

    if baseplates is not None:
        _add_baseplates(fig, baseplates)

    # failed members overdrawn in the reserved critical color
    failed = [m for m in geometry.members if not checks[m.name]["PASS"]]
    if failed:
        xs, ys, zs = _polyline([ends[m.name] for m in failed])
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=_CRITICAL, width=8),
            name="✕ fails checks", hoverinfo="skip",
        ))

    status = "feasible" if result.feasible else "INFEASIBLE — best attempt shown"
    parts = " · ".join(f"{g}: {s}" for g, s in result.sections.items())
    if baseplates is not None:
        p = baseplates.plate
        parts += (f" · baseplate: {p.B * IN_TO_MM:,.0f}×{p.N * IN_TO_MM:,.0f}"
                  f"×{p.tp * IN_TO_MM:,.1f} mm")
    title = (
        f"<b>Optimized gravity frame</b> — {result.total_weight_kg:,.0f} kg ({status})"
        f"<br><span style='font-size:13px;color:{_INK_2}'>{parts}</span>"
    )

    axis = dict(
        showbackground=True, backgroundcolor=_SURFACE,
        gridcolor=_GRID, zerolinecolor=_GRID,
        tickfont=dict(size=11, color=_MUTED),
        title_font=dict(size=12, color=_INK_2),
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color=_INK, family=_FONT),
                   x=0.02, xanchor="left"),
        font=dict(family=_FONT, color=_INK),
        paper_bgcolor=_SURFACE,
        scene=dict(
            xaxis={**axis, "title": "X (m)"},
            yaxis={**axis, "title": "Z (m)"},
            zaxis={**axis, "title": "Elevation (m)"},
            aspectmode="data",
            camera=dict(eye=dict(x=1.7, y=1.4, z=0.8)),
        ),
        legend=dict(
            x=0.99, xanchor="right", y=0.95, yanchor="top",
            bgcolor="rgba(252,252,251,0.85)",
            bordercolor="rgba(11,11,11,0.10)", borderwidth=1,
            font=dict(size=12, color=_INK_2),
        ),
        margin=dict(l=0, r=0, t=70, b=0),
    )

    out = Path(path)
    fig.write_html(out, include_plotlyjs=True, auto_open=show)
    return out
