"""Interactive 3-D wireframe of an optimized frame (self-contained HTML).

Rendering choices:
* One polyline trace per member group in a fixed categorical color; the
  legend entry carries the group's section designation.
* Every member gets a visible text label with its section plus a row of
  invisible hover targets spaced along its full length, so hovering anywhere
  on the member raises its design card: section, length, every demand vs.
  capacity with its demand-capacity ratio (DCR), the governing check, and
  PASS/FAIL. Labels are one legend item ("section labels") so they can be
  toggled off in a click.
* Members that fail their checks are overdrawn in the reserved critical color
  with their own legend entry, so a failed design is impossible to misread.
* Model Y is vertical; plotly's scene Z is up, so coordinates map
  (x, y, z) -> (x, z, y) and axes are titled X / Z / Elevation in feet.

Only `visualize_result` is public. Everything here is display-only: no part of
frame_optimizer depends on this module.
"""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from frame_optimizer import geometry_for
from frame_optimizer.config import FT
from frame_optimizer.geometry import BEAM, COLUMN
from frame_optimizer.results import OptimizationResult

# Categorical slots 1-4 of the reference palette in fixed order (validated as
# a set; aqua's and yellow's lower surface contrast is relieved by the direct
# section labels on every member). 'girder' takes slot 2 because it plays the
# same primary-flexural role as 'beam' and the two never appear in one figure.
# Groups without an entry fall back to the muted neutral until they get a slot.
_GROUP_COLOR = {COLUMN: "#2a78d6", BEAM: "#1baf7a",
                "girder": "#1baf7a", "purlin": "#eda100",
                "end_girder": "#008300"}
_GROUP_WIDTH = {COLUMN: 6, BEAM: 4, "girder": 5, "purlin": 3, "end_girder": 5}
_DEFAULT_WIDTH = 4

_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_CRITICAL = "#d03b3b"

_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# hover targets per member (evenly spaced, endpoints excluded so joints stay
# unambiguous) and the DCR below which a check row is omitted as negligible
_HOVER_SAMPLES = 9
_SHOW_UC = 0.01


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
        f"L = {row['length_ft']:.1f} ft · story {int(row['story'])}",
        f"governing DCR <b>{row['governing_uc']:.2f}</b> "
        f"[{row['governing_limitstate']}] · {verdict}",
    ]
    if row["UC_axial"] >= _SHOW_UC:
        kind = "compression" if row["Pu_kip"] < 0 else "tension"
        lines.append(f"P ({kind}): {abs(row['Pu_kip']):,.1f} / "
                     f"{row['phiPn_kip']:,.0f} kip · DCR {row['UC_axial']:.2f}")
    if row["UC_Mx"] >= _SHOW_UC:
        lines.append(f"Mx: {row['Mux_kipft']:,.1f} / "
                     f"{row['phiMnx_kipft']:,.0f} kip·ft · DCR {row['UC_Mx']:.2f}")
    if row["UC_My"] >= _SHOW_UC:
        lines.append(f"My: {row['Muy_kipft']:,.1f} / "
                     f"{row['phiMny_kipft']:,.0f} kip·ft · DCR {row['UC_My']:.2f}")
    if row["UC_V"] >= _SHOW_UC:
        lines.append(f"V: {row['Vu_kip']:,.1f} / "
                     f"{row['phiVn_kip']:,.0f} kip · DCR {row['UC_V']:.2f}")
    if row["UC_H1"] >= _SHOW_UC:
        lines.append(f"P–M interaction (H1): DCR {row['UC_H1']:.2f}")
    slender = row.get("UC_slenderness")
    if _notna(slender):
        lines.append(f"KL/r ≤ 200: DCR {slender:.2f}")
    d_live, d_total = row.get("UC_defl_live"), row.get("UC_defl_total")
    if _notna(d_live):
        lines.append(f"deflection: live DCR {d_live:.2f} · total DCR {d_total:.2f}")
    return "<br>".join(lines)


def _member_ends_ft(geometry) -> dict[str, tuple]:
    """member name -> ((xi, yi, zi), (xj, yj, zj)) in feet."""
    nodes = {n.name: (n.x / FT, n.y / FT, n.z / FT) for n in geometry.nodes}
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
                     show: bool = True) -> Path:
    """Write a standalone interactive HTML wireframe of the final design.

    Returns the path of the written file; opens it in the browser if `show`.
    """
    if result.config is None:
        raise ValueError("OptimizationResult.config is None; visualize_result "
                         "needs the config to rebuild the frame geometry.")

    geometry = geometry_for(result.config)
    ends = _member_ends_ft(geometry)
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
        x=[n.x / FT for n in base], y=[n.z / FT for n in base],
        z=[n.y / FT for n in base],
        mode="markers",
        marker=dict(size=4, color=_MUTED, symbol="diamond"),
        name="pinned base", hoverinfo="skip",
    ))

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
    title = (
        f"<b>Optimized gravity frame</b> — {result.total_weight_lb:,.0f} lb ({status})"
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
            xaxis={**axis, "title": "X (ft)"},
            yaxis={**axis, "title": "Z (ft)"},
            zaxis={**axis, "title": "Elevation (ft)"},
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
