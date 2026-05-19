"""
2D geometry helpers for the Simplex Explorer visualization.

Public API
----------
feasible_vertices(A_ub, b_ub) -> np.ndarray
    CCW-ordered vertices of the feasible region.

objective_line_endpoints(c, obj_val, xlim, ylim) -> tuple | None
    Two endpoints of the objective line clipped to a bounding box.

make_2d_figure(A_ub, b_ub, c_obj, obj_val, current_pt) -> go.Figure
    Plotly figure with traces in a fixed order:
      [0] filled feasible polygon
      [1] dashed objective line
      [2] vertex markers (uniform blue)
      [3] current point marker (gold diamond; empty placeholder when out of range or absent)
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull


def feasible_vertices(A_ub: np.ndarray, b_ub: np.ndarray) -> np.ndarray:
    """
    Return vertices of {x in R^2 : A_ub @ x <= b_ub, x >= 0} in CCW order.

    Returns an empty (0, 2) array when the region is empty or degenerate.
    """
    A = np.vstack([np.asarray(A_ub, dtype=float), [-1.0, 0.0], [0.0, -1.0]])
    b = np.concatenate([np.asarray(b_ub, dtype=float), [0.0, 0.0]])
    n = len(A)

    pts: set[tuple[float, float]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            det = A[i, 0] * A[j, 1] - A[j, 0] * A[i, 1]
            if abs(det) < 1e-10:
                continue
            x = (b[i] * A[j, 1] - b[j] * A[i, 1]) / det
            y = (A[i, 0] * b[j] - A[j, 0] * b[i]) / det
            if all(A[k, 0] * x + A[k, 1] * y <= b[k] + 1e-9 for k in range(n)):
                pts.add((round(x, 8), round(y, 8)))

    if len(pts) < 3:
        return np.zeros((0, 2))

    arr = np.array(sorted(pts))
    try:
        hull = ConvexHull(arr)
    except Exception:
        return np.zeros((0, 2))
    return arr[hull.vertices]


def objective_line_endpoints(
    c: list | np.ndarray,
    obj_val: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """
    Return the two endpoints of c[0]*x + c[1]*y = obj_val clipped to [xlim x ylim].
    Returns None if the line doesn't cross the box.
    """
    c0, c1 = float(c[0]), float(c[1])
    xmin, xmax = xlim
    ymin, ymax = ylim
    candidates: list[tuple[float, float]] = []

    if abs(c1) > 1e-10:
        for x in (xmin, xmax):
            y = (obj_val - c0 * x) / c1
            if ymin - 1e-9 <= y <= ymax + 1e-9:
                candidates.append((x, y))

    if abs(c0) > 1e-10:
        for y in (ymin, ymax):
            x = (obj_val - c1 * y) / c0
            if xmin - 1e-9 <= x <= xmax + 1e-9:
                candidates.append((x, y))

    unique: list[tuple[float, float]] = []
    for pt in candidates:
        if not any(abs(pt[0] - u[0]) < 1e-9 and abs(pt[1] - u[1]) < 1e-9 for u in unique):
            unique.append(pt)
        if len(unique) == 2:
            break

    return (unique[0], unique[1]) if len(unique) >= 2 else None


def make_2d_figure(
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    c_obj: list | np.ndarray,
    obj_val: float,
    current_pt: tuple[float, float] | None = None,
) -> go.Figure:
    """
    Build a Plotly figure with four traces in fixed order:

      [0] Filled feasible region polygon
      [1] Dashed objective line at obj_val (empty placeholder when off-screen)
      [2] Feasible vertex markers (uniform blue; indices [0,1,2] are stable for
          the instructions clientside callback)
      [3] Current point marker — gold diamond at current_pt if it is within the
          diagram range, otherwise an empty placeholder

    Returns a figure with an infeasibility annotation and no data traces when
    the feasible region is empty.
    """
    verts = feasible_vertices(A_ub, b_ub)
    fig = go.Figure()

    if len(verts) == 0:
        fig.add_annotation(
            text='<b>Infeasible: no feasible region</b>',
            x=0.5, y=0.5, xref='paper', yref='paper',
            font=dict(size=16, color='crimson'),
            showarrow=False,
        )
        fig.update_layout(height=420, plot_bgcolor='white', paper_bgcolor='white')
        return fig

    all_pts = list(verts.flatten())
    if current_pt is not None:
        all_pts += [float(current_pt[0]), float(current_pt[1])]
    arr = np.array(all_pts)
    pad = max(float(arr.max() - arr.min()) * 0.2, 1.0)
    lo = float(arr.min() - pad)
    hi = float(arr.max() + pad)
    xlim = ylim = (lo, hi)

    # Trace 0: feasible region fill
    vx = np.append(verts[:, 0], verts[0, 0])
    vy = np.append(verts[:, 1], verts[0, 1])
    fig.add_trace(go.Scatter(
        x=list(vx), y=list(vy),
        fill='toself',
        fillcolor='rgba(100,181,246,0.35)',
        line=dict(color='royalblue', width=2),
        name='Feasible region',
        hoverinfo='skip',
    ))

    # Trace 1: objective line clipped to a large box so it stays visible after panning
    span = hi - lo
    far = lo - 20 * span, hi + 20 * span
    eps = objective_line_endpoints(c_obj, obj_val, far, far)
    c0, c1 = float(c_obj[0]), float(c_obj[1])
    if eps:
        lx, ly = zip(*eps)
        fig.add_trace(go.Scatter(
            x=list(lx), y=list(ly),
            mode='lines',
            line=dict(color='crimson', width=2.5, dash='dash'),
            name=f'{c0:g}·x + {c1:g}·y = {obj_val:.3g}',
        ))
    else:
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', showlegend=False))

    # Trace 2: vertex markers (uniform style; current point is shown separately in trace 3)
    fig.add_trace(go.Scatter(
        x=[float(v[0]) for v in verts],
        y=[float(v[1]) for v in verts],
        mode='markers',
        marker=dict(
            size=9, color='royalblue',
            line=dict(color='royalblue', width=2),
        ),
        showlegend=False,
        hovertemplate='(%{x:.3g}, %{y:.3g})<extra></extra>',
    ))

    # Trace 3: current point — gold diamond, always shown when present
    if current_pt is not None:
        cx, cy = float(current_pt[0]), float(current_pt[1])
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode='markers',
            marker=dict(
                symbol='diamond',
                size=14, color='gold',
                line=dict(color='darkorange', width=2),
            ),
            name='Current point',
            hovertemplate='Current: (%{x:.3g}, %{y:.3g})<extra></extra>',
        ))
    else:
        fig.add_trace(go.Scatter(x=[], y=[], mode='markers', showlegend=False))

    fig.update_layout(
        xaxis=dict(
            range=list(xlim), title='x',
            showgrid=True, gridwidth=1, gridcolor='rgba(180,180,180,0.4)',
            zeroline=True, zerolinewidth=2, zerolinecolor='#888',
        ),
        yaxis=dict(
            range=list(ylim), title='y',
            showgrid=True, gridwidth=1, gridcolor='rgba(180,180,180,0.4)',
            zeroline=True, zerolinewidth=2, zerolinecolor='#888',
            scaleanchor='x',
        ),
        dragmode='pan',
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        height=420,
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    return fig
