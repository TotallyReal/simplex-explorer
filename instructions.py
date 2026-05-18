import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from scipy.spatial import ConvexHull


# ── Feasible region geometry ──────────────────────────────────────────────────
# LP: max x+y  s.t.  x-y<=3,  x<=4,  -x+2y<=5,  x>=0,  y>=0

def _feasible_vertices():
    """Return vertices of the feasible region in CCW order."""
    # Constraints as (a, b, c) meaning a*x + b*y <= c
    constraints = [
        (1, -1, 3),   # x - y <= 3
        (1,  0, 4),   # x <= 4
        (-1, 2, 5),   # -x + 2y <= 5
        (-1, 0, 0),   # x >= 0  →  -x <= 0
        (0, -1, 0),   # y >= 0  →  -y <= 0
    ]

    def intersect(c1, c2):
        a1, b1, r1 = c1
        a2, b2, r2 = c2
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-10:
            return None
        x = (r1 * b2 - r2 * b1) / det
        y = (a1 * r2 - a2 * r1) / det
        return x, y

    def feasible(pt):
        x, y = pt
        for a, b, c in constraints:
            if a * x + b * y > c + 1e-9:
                return False
        return True

    pts = []
    n = len(constraints)
    for i in range(n):
        for j in range(i + 1, n):
            pt = intersect(constraints[i], constraints[j])
            if pt and feasible(pt):
                pts.append(pt)

    pts = list({(round(x, 8), round(y, 8)) for x, y in pts})
    arr = np.array(pts)
    hull = ConvexHull(arr)
    return arr[hull.vertices]


_VERTICES = _feasible_vertices()
_OPT = (4.0, 4.5)   # known optimum: x+y=8.5


def _make_diagram(d: float) -> go.Figure:
    vx = np.append(_VERTICES[:, 0], _VERTICES[0, 0])
    vy = np.append(_VERTICES[:, 1], _VERTICES[0, 1])

    # Objective line x+y=d, endpoints slightly outside the axis range so
    # Plotly clips it at the axis boundary rather than ending inside the plot.
    line_pts = [(d - 7, 7), (7, d - 7)]

    def _vertex_style(v, d):
        s = v[0] + v[1]
        if abs(s - (_OPT[0] + _OPT[1])) < 1e-6:
            return ('gold', 16, 'darkorange') if abs(d - s) < 0.15 else ('royalblue', 9, 'royalblue')
        return ('tomato', 14, 'firebrick') if abs(d - s) < 0.15 else ('royalblue', 9, 'royalblue')

    v_colors, v_sizes, v_lines = zip(*[_vertex_style(v, d) for v in _VERTICES])

    fig = go.Figure()

    # Feasible region fill
    fig.add_trace(go.Scatter(
        x=list(vx), y=list(vy),
        fill='toself',
        fillcolor='rgba(100,181,246,0.35)',
        line=dict(color='royalblue', width=2),
        name='Feasible region',
        hoverinfo='skip',
    ))

    # Objective line
    lx, ly = zip(*line_pts)
    fig.add_trace(go.Scatter(
        x=list(lx), y=list(ly),
        mode='lines',
        line=dict(color='crimson', width=2.5, dash='dash'),
        name=f'x + y = {d:.2f}',
    ))

    # All vertices in one trace with per-vertex colours
    fig.add_trace(go.Scatter(
        x=[v[0] for v in _VERTICES],
        y=[v[1] for v in _VERTICES],
        mode='markers',
        marker=dict(size=list(v_sizes), color=list(v_colors),
                    line=dict(color=list(v_lines), width=2)),
        name='Vertices',
        hovertemplate='(%{x}, %{y})<extra></extra>',
    ))

    at_optimum = abs(d - (_OPT[0] + _OPT[1])) < 0.15
    annotations = [dict(
        x=_OPT[0], y=_OPT[1], text='<b>Optimal!</b>',
        showarrow=True, arrowhead=2, ax=40, ay=-40,
        font=dict(color='darkorange', size=13),
        bgcolor='rgba(255,255,255,0.8)', bordercolor='darkorange', borderwidth=1,
    )] if at_optimum else []

    fig.update_layout(
        xaxis=dict(range=[-0.3, 6], title='x', zeroline=True, zerolinewidth=1, zerolinecolor='#aaa'),
        yaxis=dict(range=[-0.3, 6], title='y', zeroline=True, zerolinewidth=1, zerolinecolor='#aaa', scaleanchor='x'),
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        height=420,
        plot_bgcolor='white',
        paper_bgcolor='white',
        annotations=annotations,
    )
    return fig


# ── Section content ───────────────────────────────────────────────────────────

def _md(text: str) -> dcc.Markdown:
    return dcc.Markdown(text, mathjax=True, style={'lineHeight': '1.7'})


def _section(section_id: str, title: str, *children) -> html.Div:
    return html.Div([
        html.H3(title, id=section_id, style={'marginTop': '2rem', 'paddingTop': '1rem', 'borderTop': '1px solid #e0e0e0'}),
        *children,
    ])


_SEC_INTRO = _section('sec-intro', '1. Introduction to Linear Programming',
    _md("""
A **linear program (LP)** asks you to maximize (or minimize) a linear objective function
subject to linear inequality constraints. Many real-world problems fit this shape: resource
allocation, network flow, production planning, and more.

**Example — diet problem:**
Minimize cost subject to nutritional requirements (calories, protein, …) all being
linear functions of how much of each food you buy.

**Example — production:**
Maximize revenue from products given limited machine time and raw materials.

The simplex algorithm, due to George Dantzig (1947), is the classical method for solving LPs.
It moves along the boundary of the feasible region from vertex to vertex, always
improving the objective, until it finds the optimum.
"""))


_SEC_2D = _section('sec-2d', '2. A 2D Example',
    _md(r"""
Consider the LP

$$\begin{aligned}
\max\;  & x + y \\
\text{s.t.}\; & x - y \le 3 \\
              & x \le 4 \\
              & -x + 2y \le 5 \\
              & x,\, y \ge 0
\end{aligned}$$

The shaded region below is the **feasible region** — all $(x,y)$ that satisfy every constraint.
The dashed red line is the **objective line** $x + y = d$; use the slider to move it.
"""),
    dcc.Graph(id='lp-2d-diagram', figure=_make_diagram(3.0), config={'displayModeBar': False}),
    html.Div([
        html.Label("Objective value d:", style={'fontWeight': 'bold', 'marginBottom': '4px'}),
        dcc.Slider(
            id='objective-slider',
            min=0, max=10, step=0.25, value=3.0,
            marks={0: '0', 2: '2', 4: '4', 6: '6', 8.5: '8.5 (max)', 10: '10'},
            tooltip={'placement': 'bottom', 'always_visible': True},
            updatemode='drag',
        ),
    ], style={'maxWidth': '520px', 'marginLeft': 'auto', 'marginRight': 'auto', 'marginBottom': '1.5rem'}),
    _md(r"""
**Key insight:** the maximum of $x+y$ over the feasible region is achieved at one of its
**extreme points** (vertices). An extreme point is a corner where exactly 2 constraints are
tight (hold as equalities). You can always start at *any* extreme point and hop from
vertex to vertex along the boundary, improving the objective each time, until you reach the global optimum.

For this LP the optimum is $(4,\, 4.5)$ with $x + y = 8.5$.
"""),
)


_SEC_STANDARD = _section('sec-standard', '3. Standard Form and Slack Form',
    _md(r"""
Any LP can be rewritten in **standard form**:

$$\begin{aligned}
\max\;        & \langle c, x\rangle \\
\text{s.t.}\; & Ax \le b \\
              & x \ge 0
\end{aligned}$$

where $x \in \mathbb{R}^n$, $A \in \mathbb{R}^{m\times n}$, $b \in \mathbb{R}^m$, $c \in \mathbb{R}^n$.

We then convert to **slack form** by introducing $m$ slack variables $y_1, \dots, y_m \ge 0$,
turning each inequality into an equality:

$$y_i = b_i - A_i \cdot x, \quad i = 1,\dots,m$$

The objective becomes $\max\; \langle c, x\rangle + 0$.

In this representation:
- The **nonbasic variables** are $x_1,\dots,x_n$ — they appear in the objective and on the right-hand side.
- The **basic variables** are $y_1,\dots,y_m$ — they appear only on the left-hand side (as a linear combination of the nonbasic ones).

The slack form of our 2D example ($n=2$, $m=3$ after adding non-negativity constraints to the original 3):

$$\begin{aligned}
\max\;\; & x + y \\
s.t.\;\;& y_1 = 3 - x + y \\
&y_2   = 4 - x \\
&y_3   = 5 + x - 2y \\
& x,y,y_1,y_2,y_3 \ge 0 \\[4pt]

\end{aligned}$$
"""))


_SEC_PIVOT = _section('sec-pivot', '4. The Pivot Operation',
    _md(r"""
**Why slack form matters:** We start in an $(n+m)$-dimensional space with $m$ independent
equality constraints, so the solution space is $n$-dimensional. Setting $n$ variables to
zero pins us to a single point — an **extreme point** (vertex) of the feasible region.

At each step we choose the nonbasic variables to be $0$, which immediately gives us the
basic variables from the right-hand sides.  The point is **feasible** iff all those
right-hand-side constants are $\ge 0$.

**How to move to an adjacent vertex — the pivot:**

1. Pick a nonbasic variable $x_j$ whose objective coefficient is *positive* (the **entering variable**).
2. Increasing $x_j$ from $0$ will increase the objective. But it also decreases some basic
   variables. Compute for each basic variable $y_i$ how large $x_j$ can grow before
   $y_i$ hits $0$: the bound is $b_i / A_{ij}$ when $A_{ij} > 0$.
3. Take the *smallest* such bound — call the corresponding basic variable the **leaving variable** $y_r$.
4. Solve the equation for $y_r$ to express $x_j$ in terms of the remaining variables,
   then substitute everywhere. The roles of $x_j$ and $y_r$ swap: $x_j$ becomes basic,
   $y_r$ becomes nonbasic.

**2D example — first pivot.** Start at $(0,0)$:

$$\begin{aligned}
\max\; & x + y \\
s.t.\; & y_1   = 3 - x + y \\
&y_2 = 4 - x \\
&y_3 = 5 + x - 2y \\
& x,y,y_1,y_2,y_3 \ge 0 \\[4pt]
\end{aligned}$$

Pick entering $= x$ (positive coefficient $+1$ in objective). Bounds:
$y_1$: $3/1 = 3$, $y_2$: $4/1 = 4$. Minimum is $3$, so $y_1$ leaves.

Solve $y_1 = 3 - x + y$ for $x$: $x = 3 + y - y_1$.  Substitute:

$$\begin{aligned}
\max\; & (3 + y - y_1) + y = 3 + 2y - y_1 \\
s.t.\;\;&x = 3 + y - y_1 \\
&y_2 = 4 - (3 + y - y_1) = 1 - y + y_1 \\
&y_3 = 5 + (3 + y - y_1) - 2y = 8 - y - y_1 \\
& x,y,y_1,y_2,y_3 \ge 0 \\[4pt]
\end{aligned}$$

We moved from $(0,0)$ to $(3,0)$, and the objective went from $0$ to $3$.
"""))


_SEC_OPTIMALITY = _section('sec-optimality', '5. Optimality Conditions',
    _md(r"""
**When do we stop?**

- **Unique optimum:** If every nonbasic variable has a *negative* objective coefficient,
  increasing any of them would only decrease the objective.  The current basic feasible
  solution is the unique maximum.

- **Multiple optima:** If some coefficient is *zero*, there may be infinitely many optima
  (a whole edge of optimal solutions), but the current solution is still optimal.

**Continuing the 2D example.** After two more pivots the slack form becomes:

$$\begin{aligned}
\max\; & 8.5 - \tfrac{3}{2}y_2 - \tfrac{1}{2}y_3 \\
s.t.\;\;&x = 4 - y_2 \\
&y = 4.5 - \tfrac{1}{2}y_2 - \tfrac{1}{2}y_3 \\
&y_1 = 3.5 +\tfrac{1}{2}y_2 - \tfrac{1}{2}y_3 \\
& x,y,y_1,y_2,y_3 \ge 0 \\[4pt]
\end{aligned}$$

Both $y_2$ and $y_3$ have negative coefficients, so we stop.  The optimum is $8.5$,
achieved at $x = 4$, $y = 4.5$ (set the nonbasic $y_2 = y_3 = 0$).

This matches the slider: push $d$ to $8.5$ in the diagram above and the objective line just
touches the vertex $(4,\, 4.5)$.
"""))


_SEC_INFEASIBILITY = _section('sec-infeasibility', '6. Handling Infeasibility: Two-Phase Method',
    _md(r"""
The algorithm above assumes we start with a **feasible** basic solution (all right-hand
sides $\ge 0$).  If the initial slack form is infeasible we need to find a feasible
starting point first — or prove none exists.

**Two-phase approach.** Replace the original problem with the **auxiliary LP**:

$$\begin{aligned}
\max\;        & {-\sum_i z_i} \\
\text{s.t.}\; & y - z = Ax \\
              & x,\, z \ge 0
\end{aligned}$$

where each $z_i \ge 0$ is an artificial variable added to make the system immediately
feasible (set $x = 0$, $z_i = \max(0, -b_i)$, $y_i = b_i + z_i \ge 0$).

The auxiliary LP always has a feasible starting point. Run the simplex method on it:

- If the optimum value is $0$ (i.e. all $z_i = 0$ at optimum), then the corresponding
  $x$ values give a feasible point for the original LP — use it to start Phase 2.
- If the optimum is $< 0$, the original LP is **infeasible** (no solution exists).

Phase 2 runs the simplex algorithm on the original objective starting from the feasible
point found in Phase 1.

This app handles this automatically: if you enter an infeasible slack form (some $b_i < 0$),
it runs the two-phase method for you.
"""))


_SEC_APP = _section('sec-app', '7. Using This App',
    _md(r"""
The **Simplex Explorer** lets you enter any LP in slack form and step through the pivots manually.

**Editor (left panel):**
Enter the slack form as a matrix:
- First row: $A_0, A_1, \dots, A_n$ — the objective coefficients ($A_0$ is the constant offset).
- Each subsequent row: $C_i, B_{i,1}, \dots, B_{i,n}$ — constant and coefficients for basic variable $y_i$.

Click **Generate** to parse the input.

**Pivot table (center panel):**
The table shows the current tableau. Columns are nonbasic variables, rows are basic variables.

- Click a **column header** to select the entering variable (highlighted in blue).
- Click a **row header** to select the leaving variable.
- Click **Pivot** to perform the swap.
"""),
    html.Img(src='/assets/pivot_example.png',
             style={'width': '75%', 'display': 'block',
                    'marginLeft': 'auto', 'marginRight': 'auto',
                    'marginTop': '1rem', 'marginBottom': '1rem',
                    'borderRadius': '4px'}),
    _md(r"""

**Auto-solve:** Choose a pivot rule (Bland's, Largest Coefficient, Lexicographic) and click
**Auto-solve** to run until optimality or detect cycling/unboundedness.

**History (right panel):** Every pivot is recorded. Click any entry to inspect the tableau
at that step. Use **Undo** to revert the last pivot.
"""))


# ── TOC sidebar ───────────────────────────────────────────────────────────────

_TOC_ITEMS = [
    ('sec-intro',        '1. Introduction'),
    ('sec-2d',           '2. A 2D Example'),
    ('sec-standard',     '3. Standard & Slack Form'),
    ('sec-pivot',        '4. The Pivot Operation'),
    ('sec-optimality',   '5. Optimality Conditions'),
    ('sec-infeasibility','6. Handling Infeasibility'),
    ('sec-app',          '7. Using This App'),
]

_toc = html.Div([
    html.H6("Contents", style={'fontWeight': 'bold', 'marginBottom': '0.5rem', 'color': '#555'}),
    html.Ul([
        html.Li(html.A(label, href=f'#{sid}', style={
            'textDecoration': 'none',
            'color': '#1976d2',
            'fontSize': '13px',
            'lineHeight': '2',
            'display': 'block',
        }))
        for sid, label in _TOC_ITEMS
    ], style={'listStyleType': 'none', 'paddingLeft': '0', 'margin': '0'}),
], style={
    'padding': '0.75rem 1rem',
    'background': '#f8f9fa',
    'border': '1px solid #dee2e6',
    'borderRadius': '6px',
})


# ── Public API ────────────────────────────────────────────────────────────────

def get_layout() -> html.Div:
    content = html.Div([
        html.H1("How the Simplex Algorithm Works",
                style={'marginTop': '2rem', 'marginBottom': '0.25rem'}),
        html.P("An interactive tour from linear programs to pivots.",
               style={'color': '#666', 'marginBottom': '0'}),
        _SEC_INTRO,
        _SEC_2D,
        _SEC_STANDARD,
        _SEC_PIVOT,
        _SEC_OPTIMALITY,
        _SEC_INFEASIBILITY,
        _SEC_APP,
        html.Div(style={'height': '4rem'}),
    ], style={'flex': '1', 'minWidth': '0'})

    return html.Div([
        html.Div(
            _toc,
            style={
                'width': '210px',
                'flexShrink': '0',
                'paddingTop': '2rem',
                'paddingRight': '1.5rem',
                'alignSelf': 'flex-start',
                'position': 'sticky',
                'top': '1rem',
            },
        ),
        content,
    ], style={
        'display': 'flex',
        'alignItems': 'flex-start',
        'maxWidth': '960px',
        'margin': '0 auto',
        'padding': '0 1.5rem',
    })


def register_callbacks(app) -> None:
    app.clientside_callback(
        """
        function(d, fig) {
            if (!fig || !fig.data) return window.dash_clientside.no_update;
            const pts = [[d - 7, 7], [7, d - 7]];
            const out = JSON.parse(JSON.stringify(fig));
            if (out.data[1]) {
                out.data[1].x    = pts.map(p => p[0]);
                out.data[1].y    = pts.map(p => p[1]);
                out.data[1].name = 'x + y = ' + (+d).toFixed(2);
            }
            if (out.data[2]) {
                const vx = out.data[2].x;
                const vy = out.data[2].y;
                const OPT_SUM = 8.5;
                const colors = [], sizes = [], lines = [];
                for (let i = 0; i < vx.length; i++) {
                    const s = vx[i] + vy[i];
                    const hit = Math.abs(s - d) < 0.15;
                    const isOpt = Math.abs(s - OPT_SUM) < 0.01;
                    colors.push(hit ? (isOpt ? 'gold'   : 'tomato')    : 'royalblue');
                    sizes.push( hit ? (isOpt ? 16       : 14)          : 9);
                    lines.push( hit ? (isOpt ? 'darkorange' : 'firebrick') : 'royalblue');
                }
                out.data[2].marker = {
                    ...out.data[2].marker,
                    color: colors, size: sizes,
                    line: {color: lines, width: 2},
                };
            }
            const atOpt = Math.abs(d - 8.5) < 0.15;
            out.layout.annotations = atOpt ? [{
                x: 4, y: 4.5, text: '<b>Optimal!</b>',
                showarrow: true, arrowhead: 2, ax: 40, ay: -40,
                font: {color: 'darkorange', size: 13},
                bgcolor: 'rgba(255,255,255,0.8)', bordercolor: 'darkorange', borderwidth: 1,
            }] : [];
            return out;
        }
        """,
        Output('lp-2d-diagram', 'figure'),
        Input('objective-slider', 'value'),
        State('lp-2d-diagram', 'figure'),
    )
