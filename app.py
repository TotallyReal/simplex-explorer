import os
import sys
from fractions import Fraction

import dash
from dash import Dash, html, dcc, Input, Output, State, ctx, ALL, MATCH, no_update
import dash_bootstrap_components as dbc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simplex import LinearProgram
from geometry_2d import make_2d_figure
import instructions

"""
Example to save:

unique maximal solution
0, 1, 1
3, -1, 1
4, -1, 0
5, 1, -2

infinite maximal solutions
0, 1, 1
4, -1, -1
2, -1, 6
3, 2, -5

inifinite
0, 1, 1
1, 1, -1
1, -1, 3
3, 1, -2


"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def lp_to_dict(lp: LinearProgram) -> dict:
    return {
        'B': [[str(v) for v in row] for row in lp.B],
        'C': [str(v) for v in lp.C],
        'A': [str(v) for v in lp.A],
        'basic_vars': lp.basic_vars[:],
        'nonbasic_vars': lp.nonbasic_vars[:],
    }

def dict_to_lp(d: dict) -> LinearProgram:
    B = [[Fraction(v) for v in row] for row in d['B']]
    C = [Fraction(v) for v in d['C']]
    A = [Fraction(v) for v in d['A']]
    lp = LinearProgram(B, C, A)
    lp.basic_vars = d['basic_vars']
    lp.nonbasic_vars = d['nonbasic_vars']
    return lp

def frac_str(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"

def parse_num(x: str) -> Fraction:
    x = x.strip()
    try:
        return Fraction(x)
    except ValueError:
        return Fraction(float(x)).limit_denominator(10 ** 6)

def xsub(v: int):
    return html.Span(["x", html.Sub(str(v))])

def ysub(v: int):
    return html.Span(["y", html.Sub(str(v))])

def _default_editor_values(m: int, n: int) -> dict:
    return {
        'B': [['0'] * n for _ in range(m)],
        'C': ['0'] * m,
        'A': ['0'] * (n + 1),
    }

def _is_valid_fraction(value) -> bool:
    if value is None or not str(value).strip():
        return False
    try:
        parse_num(str(value))
        return True
    except Exception:
        return False

def lp_to_editor(lp_dict: dict) -> tuple[dict, dict]:
    """Convert a serialized LP back to editor meta + values."""
    B = lp_dict['B']
    C = lp_dict['C']
    A = lp_dict['A']
    return {'m': len(C), 'n': len(A) - 1}, {'B': B, 'C': C, 'A': A}

def _values_to_text(meta: dict, values: dict) -> str:
    """Serialize editor values as CSV rows (one row per line)."""
    n = meta['n']
    A = values['A']
    rows = [', '.join(A)]
    for i in range(meta['m']):
        row = [values['C'][i]] + values['B'][i]
        rows.append(', '.join(row))
    return '\n'.join(rows)


def _text_to_values(text: str) -> tuple[dict, dict]:
    """Parse CSV rows back to editor meta + values."""
    import re

    matrix = []

    for line in text.splitlines():
        line = re.sub(r"\s+", "", line)
        if not line:
            continue

        row = [parse_num(x) for x in line.split(",")]
        matrix.append(row)

    if len(matrix) < 2:
        raise ValueError("Need at least 2 rows (objective + 1 constraint)")
    n = len(matrix[0]) - 1
    m = len(matrix) - 1
    if any(len(row) != n + 1 for row in matrix):
        raise ValueError("All rows must have the same number of columns")

    A = [frac_str(v) for v in matrix[0]]
    C = [frac_str(matrix[i + 1][0]) for i in range(m)]
    B = [[frac_str(matrix[i + 1][j + 1]) for j in range(n)] for i in range(m)]

    return {'m': m, 'n': n}, {'B': B, 'C': C, 'A': A}

def _read_editor_state(meta, B_vals, C_vals, A_vals):
    """Extract B, C, A from DOM State values (row-major ordering from ALL pattern match)."""
    m, n = meta['m'], meta['n']
    B = [[B_vals[i * n + j] or '0' for j in range(n)] for i in range(m)]
    C = [v or '0' for v in C_vals]
    A = [v or '0' for v in A_vals]
    return m, n, B, C, A

# ── Editor cell styles ────────────────────────────────────────────────────────

_CELL_INPUT_STYLE = {
    'width': '70px',
    'fontFamily': 'monospace',
    'fontSize': '13px',
    'textAlign': 'center',
    'border': 'none',
    'background': 'transparent',
    'padding': '0',
}

_CELL_INVALID_STYLE = {
    **_CELL_INPUT_STYLE,
    'border': '2px solid #f44336',
    'background': '#ffebee',
    'borderRadius': '2px',
}

_BTN_SMALL = {
    'padding': '1px 7px',
    'fontSize': '16px',
    'lineHeight': '1.2',
    'borderRadius': '3px',
    'cursor': 'pointer',
    'border': '1px solid #ccc',
    'background': '#f5f5f5',
    'marginLeft': '3px',
}

# ── Editor table builder ──────────────────────────────────────────────────────

def build_editor_table(m: int, n: int, values: dict) -> html.Table:
    B = values['B']
    C = values['C']
    A = values['A']

    # Header: empty | const | x₁ … xₙ | [+col] [−col]
    header = html.Thead(html.Tr([
        html.Th("", className="grey sep-r sep-b"),
        html.Th("const", className="grey sep-r sep-b"),
        *[html.Th(xsub(s + 1), className="grey sep-b") for s in range(n)],
        html.Th(
            [
                html.Button("+", id='btn-add-col', n_clicks=0, style=_BTN_SMALL, title="Add variable"),
                html.Button("−", id='btn-rem-col', n_clicks=0,
                            style={**_BTN_SMALL, 'marginLeft': '4px'}, title="Remove variable"),
            ],
            className="grey sep-b",
            style={'whiteSpace': 'nowrap'},
        ),
    ]))

    # Objective row: max | A[0] | A[1] … A[n]
    obj_row = html.Tr([
        html.Th("max", className="grey sep-r sep-b"),
        html.Td(
            dcc.Input(id={'type': 'ed-A', 'col': 0}, value=A[0],
                      debounce=True, type='text', style=_CELL_INPUT_STYLE),
            className="grey sep-r sep-b",
        ),
        *[
            html.Td(
                dcc.Input(id={'type': 'ed-A', 'col': s + 1}, value=A[s + 1],
                          debounce=True, type='text', style=_CELL_INPUT_STYLE),
                className="sep-b",
            )
            for s in range(n)
        ],
        html.Td("", className="grey sep-b"),
    ])

    # Constraint rows: y_{n+1+i} | C[i] | B[i][0] … B[i][n-1]
    con_rows = []
    for i in range(m):
        con_rows.append(html.Tr([
            html.Th(ysub(n + 1 + i), className="grey sep-r"),
            html.Td(
                dcc.Input(id={'type': 'ed-C', 'row': i}, value=C[i],
                          debounce=True, type='text', style=_CELL_INPUT_STYLE),
                className="grey sep-r",
            ),
            *[
                html.Td(
                    dcc.Input(id={'type': 'ed-B', 'row': i, 'col': j}, value=B[i][j],
                              debounce=True, type='text', style=_CELL_INPUT_STYLE),
                )
                for j in range(n)
            ],
            html.Td("", className="grey"),
        ]))

    # Footer row: [+row] [−row]
    footer_row = html.Tr(html.Td(
        [
            html.Button("+", id='btn-add-row', n_clicks=0, style=_BTN_SMALL, title="Add constraint"),
            html.Button("−", id='btn-rem-row', n_clicks=0,
                        style={**_BTN_SMALL, 'marginLeft': '4px'}, title="Remove constraint"),
        ],
        colSpan=n + 3,
        style={'paddingTop': '6px', 'borderTop': '1px solid #d0d0d0'},
    ))

    return html.Table(
        [header, html.Tbody([obj_row, *con_rows, footer_row])],
        id='editor-table',
        style={'borderCollapse': 'collapse', 'fontFamily': 'monospace', 'fontSize': '13px'},
    )

# ── 2D inequality editor builder ─────────────────────────────────────────────

def build_2d_editor_table(m: int, values: dict) -> html.Table:
    a = values['a']   # m x 2 list of strings
    b = values['b']   # m list of strings
    c = values['c']   # [c1, c2] strings

    header = html.Thead(html.Tr([
        html.Th("", className="grey sep-r sep-b"),
        html.Th(xsub(1), className="grey sep-b"),
        html.Th(xsub(2), className="grey sep-b"),
        html.Th("≤", className="grey sep-b"),
        html.Th("b", className="grey sep-b"),
    ]))

    obj_row = html.Tr([
        html.Th("max", className="grey sep-r sep-b"),
        html.Td(
            dcc.Input(id={'type': 'ed-2d-c', 'col': 0}, value=c[0],
                      debounce=True, type='text', style=_CELL_INPUT_STYLE),
            className="sep-b",
        ),
        html.Td(
            dcc.Input(id={'type': 'ed-2d-c', 'col': 1}, value=c[1],
                      debounce=True, type='text', style=_CELL_INPUT_STYLE),
            className="sep-b",
        ),
        html.Td("", className="grey sep-b"),
        html.Td("", className="grey sep-b"),
    ])

    con_rows = []
    for i in range(m):
        con_rows.append(html.Tr([
            html.Th(xsub(3 + i), className="grey sep-r"),
            html.Td(
                dcc.Input(id={'type': 'ed-2d-a', 'row': i, 'col': 0}, value=a[i][0],
                          debounce=True, type='text', style=_CELL_INPUT_STYLE),
            ),
            html.Td(
                dcc.Input(id={'type': 'ed-2d-a', 'row': i, 'col': 1}, value=a[i][1],
                          debounce=True, type='text', style=_CELL_INPUT_STYLE),
            ),
            html.Td("≤", className="grey", style={'textAlign': 'center'}),
            html.Td(
                dcc.Input(id={'type': 'ed-2d-b', 'row': i}, value=b[i],
                          debounce=True, type='text', style=_CELL_INPUT_STYLE),
            ),
        ]))

    footer_row = html.Tr(html.Td(
        [
            html.Button("+", id='btn-2d-add-row', n_clicks=0, style=_BTN_SMALL, title="Add constraint"),
            html.Button("−", id='btn-2d-rem-row', n_clicks=0,
                        style={**_BTN_SMALL, 'marginLeft': '4px'}, title="Remove constraint"),
        ],
        colSpan=5,
        style={'paddingTop': '6px', 'borderTop': '1px solid #d0d0d0'},
    ))

    return html.Table(
        [header, html.Tbody([obj_row, *con_rows, footer_row])],
        id='editor-2d-table',
        style={'borderCollapse': 'collapse', 'fontFamily': 'monospace', 'fontSize': '13px'},
    )

# ── Pivot table builder ───────────────────────────────────────────────────────

def build_pivot_table(lp: LinearProgram, entering_s, leaving_r) -> html.Table:
    m, n = lp.m, lp.n

    def col_cls(s):
        base = "col-hdr sep-b"
        if s == entering_s:
            base += " selected"
        return base

    def row_cls(i):
        base = "row-hdr grey sep-r"
        if i == leaving_r:
            base += " selected"
        return base

    def cell_cls(i, s):
        inC = (s == entering_s)
        inR = (i == leaving_r)
        base = "b-cell"
        if inC and inR:
            base += " pivot"
        elif inC:
            base += " in-col"
        elif inR:
            base += " in-row"
        return base

    def row_bound(i):
        if entering_s is None:
            return ""
        b = lp.B[i][entering_s]
        if b < 0:
            return f"≤ {frac_str(lp.C[i] / (-b))}"
        return "∞"

    header = html.Thead(html.Tr([
        html.Th("", className="grey sep-r sep-b"),
        html.Th("const", className="grey sep-r sep-b"),
        *[
            html.Th(xsub(lp.nonbasic_vars[s]), className=col_cls(s),
                    id={'type': 'col-hdr', 'index': s}, n_clicks=0)
            for s in range(n)
        ],
        html.Th("bound", className="grey sep-b"),
    ]))

    obj_row = html.Tr([
        html.Th("max", className="grey sep-r sep-b"),
        html.Td(frac_str(lp.A[0]), className="grey sep-r sep-b"),
        *[
            html.Td(frac_str(lp.A[s + 1]),
                    className="grey sep-b" + (" in-col" if s == entering_s else ""))
            for s in range(n)
        ],
        html.Td("", className="grey sep-b"),
    ])

    con_rows = []
    for i in range(m):
        bv = lp.basic_vars[i]
        con_rows.append(html.Tr([
            html.Th(xsub(bv), className=row_cls(i),
                    id={'type': 'row-hdr', 'index': i}, n_clicks=0),
            html.Td(frac_str(lp.C[i]),
                    className="grey sep-r" + (" in-row" if i == leaving_r else "")),
            *[
                html.Td(frac_str(lp.B[i][s]), className=cell_cls(i, s),
                        id={'type': 'b-cell', 'row': i, 'col': s}, n_clicks=0)
                for s in range(n)
            ],
            html.Td(row_bound(i), className="grey"),
        ]))

    return html.Table(
        [header, html.Tbody([obj_row, *con_rows])],
        id="pivot-table",
    )

# ── App setup ─────────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Simplex Explorer",
)

# ── Layout ────────────────────────────────────────────────────────────────────

_btn_sidebar = {
    'width': '100%',
    'padding': '7px 10px',
    'marginBottom': '8px',
    'borderRadius': '4px',
    'cursor': 'pointer',
    'border': '1px solid #ccc',
    'background': 'white',
    'textAlign': 'left',
    'fontSize': '13px',
}

_collapse_btn = {
    'background': 'none',
    'border': '1px solid #ccc',
    'borderRadius': '3px',
    'cursor': 'pointer',
    'fontSize': '11px',
    'padding': '2px 6px',
    'color': '#666',
    'lineHeight': '1.4',
}

sidebar = html.Div([
    html.Div(
        html.Button("◀", id='btn-toggle-left', n_clicks=0, title="Collapse sidebar",
                    style=_collapse_btn),
        style={'textAlign': 'right', 'marginBottom': '8px'},
    ),
    html.Div([
        html.H4("📐 Simplex Explorer", style={'marginBottom': '4px'}),
        html.P("Interactive simplex pivot explorer.", style={'fontSize': '12px', 'color': '#666', 'marginBottom': '12px'}),
        html.Hr(),
        html.Button("✏️  Edit original", id='btn-edit-original', n_clicks=0, disabled=True, style=_btn_sidebar),
        html.Button("✏️  Edit current",  id='btn-edit-current',  n_clicks=0, disabled=True, style=_btn_sidebar),
        html.Hr(),
        html.Label("LaTeX font size", style={'fontSize': '12px', 'fontWeight': 'bold'}),
        dcc.Slider(
            id='font-size-slider',
            min=5, max=25, step=1, value=16,
            marks={d: str(d) for d in range(5,30,5)},
            tooltip={'placement': 'bottom', 'always_visible': False},
        ),
        # html.Label("Row height multiplier", style={'fontSize': '12px', 'fontWeight': 'bold', 'marginTop': '12px'}),
        # dcc.Slider(
        #     id='panel-mult-slider',
        #     min=2, max=12, step=1, value=4,
        #     marks={2: '2', 4: '4', 6: '6', 8: '8', 10: '10', 12: '12'},
        #     tooltip={'placement': 'bottom', 'always_visible': False},
        # ),
        # html.Label("Panel extra padding", style={'fontSize': '12px', 'fontWeight': 'bold', 'marginTop': '12px'}),
        # dcc.Slider(
        #     id='panel-pad-slider',
        #     min=0, max=80, step=4, value=20,
        #     marks={0: '0', 20: '20', 40: '40', 60: '60', 80: '80'},
        #     tooltip={'placement': 'bottom', 'always_visible': False},
        # ),
    ], id='left-sidebar-content'),
], id='left-sidebar', style={
    'width': '220px',
    'flexShrink': '0',
    'background': '#f8f9fa',
    'padding': '12px 14px',
    'borderRight': '1px solid #dee2e6',
    'overflowY': 'auto',
    'height': '100vh',
})

main_area = html.Div([
    dcc.Store(id='lp-store'),
    dcc.Store(id='selection-store', data={'entering_s': None, 'leaving_r': None}),
    dcc.Store(id='history-store', data=[]),
    dcc.Store(id='app-mode', data='edit'),
    dcc.Store(id='editor-meta', data={'m': 3, 'n': 2}),
    dcc.Store(id='editor-values', data={
        'B': [['-1', '1'], ['-1', '0'], ['1', '-2']],
        'C': ['3', '4', '5'],
        'A': ['0', '1', '1'],
    }),
    dcc.Store(id='original-lp'),
    dcc.Store(id='constraints-2d'),
    dcc.Store(id='editor-2d-meta', data={'m': 3}),
    dcc.Store(id='editor-2d-values', data={
        'a': [['1', '-1'], ['1', '0'], ['-1', '2']],
        'b': ['3', '4', '5'],
        'c': ['1', '1'],
    }),
    dcc.Store(id='editor-tab-store', data='table'),
    dcc.Store(id='font-size-store', data=12),
    dcc.Store(id='panel-mult-store', data=4),
    dcc.Store(id='panel-pad-store', data=20),
    dcc.Store(id='left-collapsed', data=False),
    dcc.Store(id='right-collapsed', data=False),
    html.Div(id='editor-section'),
    html.Div(id='system-section'),
    html.Div(id='controls-section'),
], style={'flex': '1', 'padding': '24px', 'overflowY': 'auto', 'height': '100vh'})

history_panel = html.Div([
    html.Div(
        html.Button("▶", id='btn-toggle-right', n_clicks=0, title="Collapse history",
                    style=_collapse_btn),
        style={'marginBottom': '8px'},
    ),
    html.Div([
        html.H5("Pivot history", style={'marginBottom': '12px'}),
        html.Div(id='history-panel'),
    ], id='right-sidebar-content'),
], id='right-sidebar', style={
    'width': '300px',
    'flexShrink': '0',
    'background': '#f8f9fa',
    'borderLeft': '1px solid #dee2e6',
    'padding': '12px 14px',
    'overflowY': 'auto',
    'height': '100vh',
})

main_layout = html.Div(
    [sidebar, main_area, history_panel],
    style={'display': 'flex', 'height': '100vh'},
)

_navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Simplex Explorer", href="/", style={'fontWeight': 'bold'}),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Tool", href="/", id='nav-tool')),
            dbc.NavItem(dbc.NavLink("Instructions", href="/instructions", id='nav-instructions')),
        ], navbar=True),
    ], fluid=True),
    color="light",
    dark=False,
    style={'borderBottom': '1px solid #dee2e6', 'padding': '0 1rem'},
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    _navbar,
    html.Div(id='page-content'),
])


@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
)
def render_page(pathname):
    if pathname == '/instructions':
        return instructions.get_layout()
    return main_layout

# ── Callback: Render editor ────────────────────────────────────────────────────

@app.callback(
    Output('editor-section', 'children'),
    Input('app-mode', 'data'),
    Input('editor-meta', 'data'),
    Input('editor-values', 'data'),
    Input('editor-2d-meta', 'data'),
    Input('editor-2d-values', 'data'),
    State('editor-tab-store', 'data'),
)
def render_editor(app_mode, meta, values, meta_2d, values_2d, active_tab):
    if app_mode == 'view':
        return html.Div()
    m, n = meta['m'], meta['n']

    _etab = {'padding': '4px 14px', 'fontSize': '13px', 'lineHeight': '26px',
             'borderBottom': '1px solid #d6d6d6'}
    _etab_sel = {**_etab, 'fontWeight': 'bold', 'borderTop': '2px solid #1976d2',
                 'borderBottom': 'none', 'color': '#1976d2'}

    table_tab = dcc.Tab(label='Table', value='table', style=_etab, selected_style=_etab_sel,
        children=html.Div([
            html.P([
                "Slack form:  ", html.B("y"), " = B·x + C,   max A·x + const. "
                "Rows = basic vars (y), columns = nonbasic vars (x).",
            ], style={'fontSize': '12px', 'color': '#666', 'margin': '10px 0 10px'}),
            build_editor_table(m, n, values),
        ]),
    )

    text_tab = dcc.Tab(label='Text', value='text', style=_etab, selected_style=_etab_sel,
        children=html.Div([
            html.P(
                "Each row: [const, x₁, …, xₙ]. First row = objective (max), rest = constraints.",
                style={'fontSize': '12px', 'color': '#666', 'margin': '10px 0 8px'},
            ),
            dcc.Textarea(
                id='text-editor',
                value=_values_to_text(meta, values),
                style={'width': '100%', 'height': '200px', 'fontFamily': 'monospace',
                       'fontSize': '13px', 'resize': 'vertical'},
            ),
            html.Div([
                html.Button("Apply", id='btn-apply-text', n_clicks=0,
                            style={'marginTop': '8px', 'padding': '6px 16px',
                                   'borderRadius': '4px', 'cursor': 'pointer'}),
                html.Div(id='text-apply-msg', style={'display': 'inline-block', 'marginLeft': '12px'}),
            ]),
        ]),
    )

    d2_tab = dcc.Tab(label='2D (inequalities)', value='2d', style=_etab, selected_style=_etab_sel,
        children=html.Div([
            html.P([
                "Enter constraints as ", html.B("a₁·x₁ + a₂·x₂ ≤ b"),
                " and an objective ", html.B("max c₁·x₁ + c₂·x₂"),
                ". The app converts to slack form automatically.",
            ], style={'fontSize': '12px', 'color': '#666', 'margin': '10px 0 10px'}),
            build_2d_editor_table(meta_2d['m'], values_2d),
        ]),
    )

    return html.Div([
        html.H5("Define system", style={'marginBottom': '8px'}),
        dcc.Tabs(id='editor-tabs', value=active_tab or 'table', children=[table_tab, text_tab, d2_tab]),
        html.Button(
            "Generate →",
            id='btn-generate',
            n_clicks=0,
            style={
                'marginTop': '16px',
                'padding': '8px 20px',
                'background': '#1976d2',
                'color': 'white',
                'border': 'none',
                'borderRadius': '4px',
                'cursor': 'pointer',
                'fontWeight': 'bold',
            },
        ),
        html.Div(id='editor-msg', style={'marginTop': '8px'}),
    ])

# ── Callbacks: resize editor ───────────────────────────────────────────────────

@app.callback(
    Output('editor-meta', 'data', allow_duplicate=True),
    Output('editor-values', 'data', allow_duplicate=True),
    Input('btn-add-col', 'n_clicks'),
    State('editor-meta', 'data'),
    State({'type': 'ed-B', 'row': ALL, 'col': ALL}, 'value'),
    State({'type': 'ed-C', 'row': ALL}, 'value'),
    State({'type': 'ed-A', 'col': ALL}, 'value'),
    prevent_initial_call=True,
)
def add_col(n_clicks, meta, B_vals, C_vals, A_vals):
    if not n_clicks:
        return no_update, no_update
    m, n, B, C, A = _read_editor_state(meta, B_vals, C_vals, A_vals)
    return {'m': m, 'n': n + 1}, {'B': [row + ['0'] for row in B], 'C': C, 'A': A + ['0']}

@app.callback(
    Output('editor-meta', 'data', allow_duplicate=True),
    Output('editor-values', 'data', allow_duplicate=True),
    Input('btn-rem-col', 'n_clicks'),
    State('editor-meta', 'data'),
    State({'type': 'ed-B', 'row': ALL, 'col': ALL}, 'value'),
    State({'type': 'ed-C', 'row': ALL}, 'value'),
    State({'type': 'ed-A', 'col': ALL}, 'value'),
    prevent_initial_call=True,
)
def rem_col(n_clicks, meta, B_vals, C_vals, A_vals):
    if not n_clicks:
        return no_update, no_update
    m, n, B, C, A = _read_editor_state(meta, B_vals, C_vals, A_vals)
    if n <= 1:
        return no_update, no_update
    return {'m': m, 'n': n - 1}, {'B': [row[:-1] for row in B], 'C': C, 'A': A[:-1]}

@app.callback(
    Output('editor-meta', 'data', allow_duplicate=True),
    Output('editor-values', 'data', allow_duplicate=True),
    Input('btn-add-row', 'n_clicks'),
    State('editor-meta', 'data'),
    State({'type': 'ed-B', 'row': ALL, 'col': ALL}, 'value'),
    State({'type': 'ed-C', 'row': ALL}, 'value'),
    State({'type': 'ed-A', 'col': ALL}, 'value'),
    prevent_initial_call=True,
)
def add_row(n_clicks, meta, B_vals, C_vals, A_vals):
    if not n_clicks:
        return no_update, no_update
    m, n, B, C, A = _read_editor_state(meta, B_vals, C_vals, A_vals)
    return {'m': m + 1, 'n': n}, {'B': B + [['0'] * n], 'C': C + ['0'], 'A': A}

@app.callback(
    Output('editor-meta', 'data', allow_duplicate=True),
    Output('editor-values', 'data', allow_duplicate=True),
    Input('btn-rem-row', 'n_clicks'),
    State('editor-meta', 'data'),
    State({'type': 'ed-B', 'row': ALL, 'col': ALL}, 'value'),
    State({'type': 'ed-C', 'row': ALL}, 'value'),
    State({'type': 'ed-A', 'col': ALL}, 'value'),
    prevent_initial_call=True,
)
def rem_row(n_clicks, meta, B_vals, C_vals, A_vals):
    if not n_clicks:
        return no_update, no_update
    m, n, B, C, A = _read_editor_state(meta, B_vals, C_vals, A_vals)
    if m <= 1:
        return no_update, no_update
    return {'m': m - 1, 'n': n}, {'B': B[:-1], 'C': C[:-1], 'A': A}

# ── Callbacks: 2D editor resize ──────────────────────────────────────────────

def _read_2d_state(meta_2d, a_vals, b_vals, c_vals):
    m = meta_2d['m']
    a = [[a_vals[i * 2 + j] or '0' for j in range(2)] for i in range(m)]
    b = [v or '0' for v in b_vals]
    c = [v or '0' for v in c_vals]
    return m, a, b, c

@app.callback(
    Output('editor-2d-meta', 'data', allow_duplicate=True),
    Output('editor-2d-values', 'data', allow_duplicate=True),
    Input('btn-2d-add-row', 'n_clicks'),
    State('editor-2d-meta', 'data'),
    State({'type': 'ed-2d-a', 'row': ALL, 'col': ALL}, 'value'),
    State({'type': 'ed-2d-b', 'row': ALL}, 'value'),
    State({'type': 'ed-2d-c', 'col': ALL}, 'value'),
    prevent_initial_call=True,
)
def add_row_2d(n, meta_2d, a_vals, b_vals, c_vals):
    if not n:
        return no_update, no_update
    m, a, b, c = _read_2d_state(meta_2d, a_vals, b_vals, c_vals)
    return {'m': m + 1}, {'a': a + [['0', '0']], 'b': b + ['0'], 'c': c}

@app.callback(
    Output('editor-2d-meta', 'data', allow_duplicate=True),
    Output('editor-2d-values', 'data', allow_duplicate=True),
    Input('btn-2d-rem-row', 'n_clicks'),
    State('editor-2d-meta', 'data'),
    State({'type': 'ed-2d-a', 'row': ALL, 'col': ALL}, 'value'),
    State({'type': 'ed-2d-b', 'row': ALL}, 'value'),
    State({'type': 'ed-2d-c', 'col': ALL}, 'value'),
    prevent_initial_call=True,
)
def rem_row_2d(n, meta_2d, a_vals, b_vals, c_vals):
    if not n:
        return no_update, no_update
    m, a, b, c = _read_2d_state(meta_2d, a_vals, b_vals, c_vals)
    if m <= 1:
        return no_update, no_update
    return {'m': m - 1}, {'a': a[:-1], 'b': b[:-1], 'c': c}

# ── Callback: persist active editor tab ───────────────────────────────────────

@app.callback(
    Output('editor-tab-store', 'data'),
    Input('editor-tabs', 'value'),
    prevent_initial_call=True,
)
def sync_editor_tab(tab):
    return tab

# ── Callbacks: text tab sync / apply ─────────────────────────────────────────

@app.callback(
    Output('text-editor', 'value'),
    Input('editor-tabs', 'value'),
    State('editor-meta', 'data'),
    State({'type': 'ed-B', 'row': ALL, 'col': ALL}, 'value'),
    State({'type': 'ed-C', 'row': ALL}, 'value'),
    State({'type': 'ed-A', 'col': ALL}, 'value'),
    prevent_initial_call=True,
)
def sync_text_tab(tab, meta, B_vals, C_vals, A_vals):
    if tab != 'text':
        return no_update
    m, n, B, C, A = _read_editor_state(meta, B_vals, C_vals, A_vals)
    return _values_to_text({'m': m, 'n': n}, {'B': B, 'C': C, 'A': A})

@app.callback(
    Output('editor-meta', 'data', allow_duplicate=True),
    Output('editor-values', 'data', allow_duplicate=True),
    Output('text-apply-msg', 'children'),
    Input('btn-apply-text', 'n_clicks'),
    State('text-editor', 'value'),
    prevent_initial_call=True,
)
def apply_text(n_clicks, text):
    if not n_clicks or not text:
        return no_update, no_update, no_update
    try:
        meta, values = _text_to_values(text)
        return meta, values, html.Div("Applied.", className="msg-success")
    except Exception as e:
        return no_update, no_update, html.Div(str(e), className="msg-error")

# ── Callbacks: per-cell validation (MATCH — one instance per cell) ─────────────

@app.callback(
    Output({'type': 'ed-B', 'row': MATCH, 'col': MATCH}, 'style'),
    Input({'type': 'ed-B', 'row': MATCH, 'col': MATCH}, 'value'),
    prevent_initial_call=True,
)
def validate_B(value):
    return _CELL_INPUT_STYLE if _is_valid_fraction(value) else _CELL_INVALID_STYLE

@app.callback(
    Output({'type': 'ed-C', 'row': MATCH}, 'style'),
    Input({'type': 'ed-C', 'row': MATCH}, 'value'),
    prevent_initial_call=True,
)
def validate_C(value):
    return _CELL_INPUT_STYLE if _is_valid_fraction(value) else _CELL_INVALID_STYLE

@app.callback(
    Output({'type': 'ed-A', 'col': MATCH}, 'style'),
    Input({'type': 'ed-A', 'col': MATCH}, 'value'),
    prevent_initial_call=True,
)
def validate_A(value):
    return _CELL_INPUT_STYLE if _is_valid_fraction(value) else _CELL_INVALID_STYLE

# ── Callback: Generate LP ──────────────────────────────────────────────────────

@app.callback(
    Output('lp-store', 'data', allow_duplicate=True),
    Output('original-lp', 'data', allow_duplicate=True),
    Output('app-mode', 'data', allow_duplicate=True),
    Output('history-store', 'data', allow_duplicate=True),
    Output('selection-store', 'data', allow_duplicate=True),
    Output('constraints-2d', 'data', allow_duplicate=True),
    Output('editor-msg', 'children'),
    Input('btn-generate', 'n_clicks'),
    State('editor-tab-store', 'data'),
    # Table/text tab inputs
    State('editor-meta', 'data'),
    State({'type': 'ed-B', 'row': ALL, 'col': ALL}, 'value'),
    State({'type': 'ed-C', 'row': ALL}, 'value'),
    State({'type': 'ed-A', 'col': ALL}, 'value'),
    # 2D tab inputs
    State('editor-2d-meta', 'data'),
    State({'type': 'ed-2d-a', 'row': ALL, 'col': ALL}, 'value'),
    State({'type': 'ed-2d-b', 'row': ALL}, 'value'),
    State({'type': 'ed-2d-c', 'col': ALL}, 'value'),
    prevent_initial_call=True,
)
def generate_lp(n_clicks, tab, meta, B_vals, C_vals, A_vals, meta_2d, a_vals, b_vals, c_vals):
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update

    sel = {'entering_s': None, 'leaving_r': None}

    if tab == '2d':
        try:
            m, a, b, c = _read_2d_state(meta_2d, a_vals, b_vals, c_vals)
            c1 = parse_num(c[0])
            c2 = parse_num(c[1])
            A = [Fraction(0), c1, c2]
            B_lp = [[-parse_num(a[i][0]), -parse_num(a[i][1])] for i in range(m)]
            C_lp = [parse_num(b[i]) for i in range(m)]
            lp = LinearProgram(B_lp, C_lp, A)
            lp_dict = lp_to_dict(lp)
            # Store original inequalities for the geometry diagram (persists across pivots)
            constraints = {
                'A_ub': [[float(parse_num(a[i][j])) for j in range(2)] for i in range(m)],
                'b_ub': [float(parse_num(b[i])) for i in range(m)],
                'c_obj': [float(c1), float(c2)],
            }
            return lp_dict, lp_dict, 'view', [], sel, constraints, html.Div()
        except Exception as e:
            return no_update, no_update, no_update, no_update, no_update, no_update, html.Div(str(e), className="msg-error")

    m, n, B_str, C_str, A_str = _read_editor_state(meta, B_vals, C_vals, A_vals)

    errors = []
    for i in range(m):
        for j in range(n):
            if not _is_valid_fraction(B_str[i][j]):
                errors.append(f"B[{i+1},{j+1}]")
    for i, v in enumerate(C_str):
        if not _is_valid_fraction(v):
            errors.append(f"C[{i+1}]")
    for j, v in enumerate(A_str):
        if not _is_valid_fraction(v):
            errors.append(f"A[{j}]")

    if errors:
        msg = html.Div(f"Invalid values: {', '.join(errors)}", className="msg-error")
        return no_update, no_update, no_update, no_update, no_update, no_update, msg

    try:
        B = [[parse_num(B_str[i][j]) for j in range(n)] for i in range(m)]
        C = [parse_num(v) for v in C_str]
        A = [parse_num(v) for v in A_str]
        lp = LinearProgram(B, C, A)
        lp_dict = lp_to_dict(lp)
        return lp_dict, lp_dict, 'view', [], sel, None, html.Div()
    except Exception as e:
        return no_update, no_update, no_update, no_update, no_update, no_update, html.Div(str(e), className="msg-error")

# ── Callbacks: Edit original / Edit current ────────────────────────────────────

@app.callback(
    Output('editor-meta', 'data', allow_duplicate=True),
    Output('editor-values', 'data', allow_duplicate=True),
    Output('app-mode', 'data', allow_duplicate=True),
    Output('selection-store', 'data', allow_duplicate=True),
    Input('btn-edit-original', 'n_clicks'),
    State('original-lp', 'data'),
    prevent_initial_call=True,
)
def edit_original(n_clicks, original_lp):
    if not n_clicks or not original_lp:
        return no_update, no_update, no_update, no_update
    meta, values = lp_to_editor(original_lp)
    return meta, values, 'edit', {'entering_s': None, 'leaving_r': None}

@app.callback(
    Output('editor-meta', 'data', allow_duplicate=True),
    Output('editor-values', 'data', allow_duplicate=True),
    Output('app-mode', 'data', allow_duplicate=True),
    Output('selection-store', 'data', allow_duplicate=True),
    Input('btn-edit-current', 'n_clicks'),
    State('lp-store', 'data'),
    prevent_initial_call=True,
)
def edit_current(n_clicks, lp_data):
    if not n_clicks or not lp_data:
        return no_update, no_update, no_update, no_update
    meta, values = lp_to_editor(lp_data)
    return meta, values, 'edit', {'entering_s': None, 'leaving_r': None}

# ── Callback: Font size ────────────────────────────────────────────────────────

@app.callback(
    Output('font-size-store', 'data'),
    Input('font-size-slider', 'value'),
)
def update_font_size(value):
    return value

@app.callback(
    Output('panel-mult-store', 'data'),
    Input('panel-mult-slider', 'value'),
)
def update_panel_mult(value):
    return value

@app.callback(
    Output('panel-pad-store', 'data'),
    Input('panel-pad-slider', 'value'),
)
def update_panel_pad(value):
    return value

# ── Callbacks: collapsible sidebars ───────────────────────────────────────────

_LEFT_EXPANDED = {'width': '220px', 'flexShrink': '0', 'background': '#f8f9fa',
                  'padding': '12px 14px', 'borderRight': '1px solid #dee2e6',
                  'overflowY': 'auto', 'height': '100vh'}
_LEFT_COLLAPSED = {'width': '36px', 'flexShrink': '0', 'background': '#f8f9fa',
                   'padding': '8px 4px', 'borderRight': '1px solid #dee2e6',
                   'overflowY': 'hidden', 'height': '100vh'}
_RIGHT_EXPANDED = {'width': '300px', 'flexShrink': '0', 'background': '#f8f9fa',
                   'borderLeft': '1px solid #dee2e6', 'padding': '12px 14px',
                   'overflowY': 'auto', 'height': '100vh'}
_RIGHT_COLLAPSED = {'width': '36px', 'flexShrink': '0', 'background': '#f8f9fa',
                    'borderLeft': '1px solid #dee2e6', 'padding': '8px 4px',
                    'overflowY': 'hidden', 'height': '100vh'}

@app.callback(
    Output('left-collapsed', 'data'),
    Input('btn-toggle-left', 'n_clicks'),
    State('left-collapsed', 'data'),
    prevent_initial_call=True,
)
def toggle_left(_, collapsed):
    return not collapsed

@app.callback(
    Output('left-sidebar', 'style'),
    Output('left-sidebar-content', 'style'),
    Output('btn-toggle-left', 'children'),
    Input('left-collapsed', 'data'),
)
def update_left_sidebar(collapsed):
    if collapsed:
        return _LEFT_COLLAPSED, {'display': 'none'}, "▶"
    return _LEFT_EXPANDED, {'display': 'block'}, "◀"

@app.callback(
    Output('right-collapsed', 'data'),
    Input('btn-toggle-right', 'n_clicks'),
    State('right-collapsed', 'data'),
    prevent_initial_call=True,
)
def toggle_right(_, collapsed):
    return not collapsed

@app.callback(
    Output('right-sidebar', 'style'),
    Output('right-sidebar-content', 'style'),
    Output('btn-toggle-right', 'children'),
    Input('right-collapsed', 'data'),
)
def update_right_sidebar(collapsed):
    if collapsed:
        return _RIGHT_COLLAPSED, {'display': 'none'}, "◀"
    return _RIGHT_EXPANDED, {'display': 'block'}, "▶"

# ── Callback: Sidebar button enable/disable ────────────────────────────────────

@app.callback(
    Output('btn-edit-original', 'disabled'),
    Output('btn-edit-current', 'disabled'),
    Input('original-lp', 'data'),
    Input('lp-store', 'data'),
)
def update_sidebar_btns(original_lp, lp_data):
    return original_lp is None, lp_data is None

# ── Callbacks: pivot table interaction (clientside — no server round-trip) ─────

app.clientside_callback(
    """
    function(n_clicks_list, sel) {
        const t = dash_clientside.callback_context.triggered;
        if (!t || !t.length || !t[0].value) return dash_clientside.no_update;
        const s = JSON.parse(t[0].prop_id.split('.')[0]).index;
        return {...sel, entering_s: sel.entering_s === s ? null : s};
    }
    """,
    Output('selection-store', 'data', allow_duplicate=True),
    Input({'type': 'col-hdr', 'index': ALL}, 'n_clicks'),
    State('selection-store', 'data'),
    prevent_initial_call=True,
)

app.clientside_callback(
    """
    function(n_clicks_list, sel) {
        const t = dash_clientside.callback_context.triggered;
        if (!t || !t.length || !t[0].value) return dash_clientside.no_update;
        const i = JSON.parse(t[0].prop_id.split('.')[0]).index;
        return {...sel, leaving_r: sel.leaving_r === i ? null : i};
    }
    """,
    Output('selection-store', 'data', allow_duplicate=True),
    Input({'type': 'row-hdr', 'index': ALL}, 'n_clicks'),
    State('selection-store', 'data'),
    prevent_initial_call=True,
)

app.clientside_callback(
    """
    function(n_clicks_list) {
        const t = dash_clientside.callback_context.triggered;
        if (!t || !t.length || !t[0].value) return dash_clientside.no_update;
        const id = JSON.parse(t[0].prop_id.split('.')[0]);
        return {entering_s: id.col, leaving_r: id.row};
    }
    """,
    Output('selection-store', 'data', allow_duplicate=True),
    Input({'type': 'b-cell', 'row': ALL, 'col': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)

# ── Callback: Perform pivot ───────────────────────────────────────────────────

@app.callback(
    Output('lp-store', 'data', allow_duplicate=True),
    Output('selection-store', 'data', allow_duplicate=True),
    Output('history-store', 'data', allow_duplicate=True),
    Input('btn-pivot', 'n_clicks'),
    State('lp-store', 'data'),
    State('selection-store', 'data'),
    State('history-store', 'data'),
    prevent_initial_call=True,
)
def perform_pivot(n_clicks, lp_data, sel, history):
    if not n_clicks or not lp_data:
        return no_update, no_update, no_update
    entering_s = sel.get('entering_s')
    leaving_r  = sel.get('leaving_r')
    if entering_s is None or leaving_r is None:
        return no_update, no_update, no_update
    lp = dict_to_lp(lp_data)
    entering_v = lp.nonbasic_vars[entering_s]
    leaving_v  = lp.basic_vars[leaving_r]
    snap = lp_to_dict(lp)
    history = list(history) + [(f"Enter x{entering_v}, leave x{leaving_v}", snap, lp._to_latex_array_mathjax())]
    lp.swap_variables(leaving_v, entering_v)
    return lp_to_dict(lp), {'entering_s': None, 'leaving_r': None}, history

# ── Callback: Apply basis ─────────────────────────────────────────────────────

@app.callback(
    Output('lp-store', 'data', allow_duplicate=True),
    Output('history-store', 'data', allow_duplicate=True),
    Output('basis-msg', 'children'),
    Input('btn-basis', 'n_clicks'),
    State('basis-input', 'value'),
    State('lp-store', 'data'),
    State('history-store', 'data'),
    prevent_initial_call=True,
)
def apply_basis(n_clicks, basis_text, lp_data, history):
    if not n_clicks or not lp_data:
        return no_update, no_update, no_update
    try:
        indices = [int(x.strip()) for x in basis_text.split(',') if x.strip()]
        lp = dict_to_lp(lp_data)
        snap = lp_to_dict(lp)
        latex = lp._to_latex_array_mathjax()
        lp.set_basis(indices)
        label = f"Set basis → {{{', '.join(f'x{i}' for i in indices)}}}"
        history = list(history) + [(label, snap, latex)]
        return lp_to_dict(lp), history, html.Div("Basis applied.", className="msg-success")
    except Exception as e:
        return no_update, no_update, html.Div(str(e), className="msg-error")

# ── Callback: One step / Solve ────────────────────────────────────────────────

@app.callback(
    Output('lp-store', 'data', allow_duplicate=True),
    Output('history-store', 'data', allow_duplicate=True),
    Output('auto-msg', 'children'),
    Input('btn-one-step', 'n_clicks'),
    Input('btn-solve', 'n_clicks'),
    State('lp-store', 'data'),
    State('history-store', 'data'),
    State('rule-radio', 'value'),
    prevent_initial_call=True,
)
def auto_pivot(n_step, n_solve, lp_data, history, rule):
    if not (n_step or n_solve) or not lp_data:
        return no_update, no_update, no_update
    lp = dict_to_lp(lp_data)
    history = list(history)
    try:
        if ctx.triggered_id == 'btn-one-step':
            p = lp.find_pivot(rule=rule)
            if p is None:
                return no_update, no_update, html.Div("Already optimal.", className="msg-info")
            ev, lv = p
            history.append((f"[{rule}] Enter x{ev}, leave x{lv}", lp_to_dict(lp), lp._to_latex_array_mathjax()))
            lp.swap_variables(lv, ev)
        else:
            steps = 0
            while True:
                p = lp.find_pivot(rule=rule)
                if p is None:
                    break
                ev, lv = p
                history.append((f"[{rule}] Enter x{ev}, leave x{lv}", lp_to_dict(lp), lp._to_latex_array_mathjax()))
                lp.swap_variables(lv, ev)
                steps += 1
            return lp_to_dict(lp), history, html.Div(f"Solved in {steps} step(s).", className="msg-success")
    except ValueError as e:
        return no_update, no_update, html.Div(str(e), className="msg-error")
    return lp_to_dict(lp), history, no_update

# ── Callback: Undo ────────────────────────────────────────────────────────────

@app.callback(
    Output('lp-store', 'data', allow_duplicate=True),
    Output('history-store', 'data', allow_duplicate=True),
    Input('btn-undo', 'n_clicks'),
    State('history-store', 'data'),
    prevent_initial_call=True,
)
def undo(n_clicks, history):
    if not n_clicks or not history:
        return no_update, no_update
    history = list(history)
    _, snap, _ = history.pop()
    return snap, history

# ── Callback: System section (MathJax rendering) ──────────────────────────────

@app.callback(
    Output('system-section', 'children'),
    Input('lp-store', 'data'),
    Input('app-mode', 'data'),
    Input('font-size-store', 'data'),
    Input('constraints-2d', 'data'),
)
def render_system(lp_data, app_mode, font_size, constraints):
    if app_mode == 'edit' or not lp_data:
        return html.Div()

    lp = dict_to_lp(lp_data)
    is_feasible = all(c >= 0 for c in lp.C)

    _tab_style = {
        'padding': '4px 14px',
        'fontSize': '13px',
        'lineHeight': '26px',
        'borderBottom': '1px solid #d6d6d6',
    }
    _tab_selected = {**_tab_style, 'fontWeight': 'bold', 'borderTop': '2px solid #1976d2',
                     'borderBottom': 'none', 'color': '#1976d2'}

    feasibility_badge = html.Span(
        "Feasible" if is_feasible else "Infeasible",
        style={
            'fontSize': '13px',
            'fontWeight': 'bold',
            'padding': '3px 10px',
            'borderRadius': '10px',
            'marginLeft': '10px',
            'background': '#e8f5e9' if is_feasible else '#ffebee',
            'color': '#2e7d32' if is_feasible else '#c62828',
            'border': f"1px solid {'#a5d6a7' if is_feasible else '#ef9a9a'}",
        },
    )

    eq_latex = lp._to_latex_array_mathjax()
    mat_latex = '$$\n' + lp.to_latex(matrix_form=True) + '\n$$'

    _md_style = {'paddingTop': '10px', 'paddingLeft': '20px', 'fontSize': f'{font_size}pt'}

    algebra_panel = html.Div([
        html.Div([
            html.H5("Current system", style={'display': 'inline', 'marginBottom': '0'}),
            feasibility_badge,
        ], style={'marginBottom': '8px'}),
        dcc.Tabs([
            dcc.Tab(label='Equations', style=_tab_style, selected_style=_tab_selected, children=[
                dcc.Markdown(eq_latex, mathjax=True, style=_md_style),
            ]),
            dcc.Tab(label='Matrix form', style=_tab_style, selected_style=_tab_selected, children=[
                dcc.Markdown(mat_latex, mathjax=True, style=_md_style),
            ]),
            dcc.Tab(label='Raw LaTeX', style=_tab_style, selected_style=_tab_selected, children=[
                html.Pre(lp.to_latex(matrix_form=False),
                         style={'overflowY': 'auto', 'fontSize': '12px', 'paddingTop': '8px',
                                'background': 'none', 'border': 'none'}),
            ]),
        ]),
    ])

    if constraints:
        import numpy as np
        pt = lp.current_point()
        current_xy = (float(pt.get(1, 0)), float(pt.get(2, 0)))
        obj_val = float(lp.A[0])
        fig = make_2d_figure(
            np.array(constraints['A_ub']),
            np.array(constraints['b_ub']),
            constraints['c_obj'],
            obj_val,
            current_pt=current_xy,
        )
        diagram_panel = html.Div([
            html.H5("Geometry", style={'marginBottom': '8px'}),
            dcc.Graph(figure=fig, config={'displayModeBar': False}),
        ], style={'minWidth': '420px'})

        return html.Div([
            html.Div([algebra_panel, diagram_panel],
                     style={'display': 'flex', 'gap': '32px', 'alignItems': 'flex-start'}),
            html.Hr(),
        ])

    return html.Div([algebra_panel, html.Hr()])

# ── Callback: Controls section (fast — no pdflatex) ───────────────────────────

@app.callback(
    Output('controls-section', 'children'),
    Input('lp-store', 'data'),
    Input('selection-store', 'data'),
    Input('app-mode', 'data'),
)
def render_controls(lp_data, sel, app_mode):
    if app_mode == 'edit' or not lp_data:
        return html.Div()

    lp = dict_to_lp(lp_data)
    entering_s = sel.get('entering_s')
    leaving_r  = sel.get('leaving_r')

    try:
        suggested = lp.find_pivot(rule='bland')
        is_unbounded = False
    except ValueError:
        suggested = None
        is_unbounded = True

    if is_unbounded:
        status = html.Div("LP appears unbounded.", className="msg-error")
    elif suggested is None:
        sol = {f"x_{v}": frac_str(lp.C[i]) for i, v in enumerate(lp.basic_vars)}
        sol.update({f"x_{v}": "0" for v in lp.nonbasic_vars})
        sol_str = ", ".join(f"{k} = {v}" for k, v in sorted(sol.items()))
        status = html.Div([html.B("Optimal! "), f"Objective = {frac_str(lp.A[0])}.  ", sol_str],
                          className="msg-success")
    else:
        status = html.Div()

    table = build_pivot_table(lp, entering_s, leaving_r)

    if entering_s is None or leaving_r is None:
        parts = (["a column"] if entering_s is None else []) + (["a row"] if leaving_r is None else [])
        pivot_msg = html.Div(f"Choose {' and '.join(parts)} in the table.", className="msg-info")
        pivot_btn = html.Button("Perform pivot", id='btn-pivot', n_clicks=0, disabled=True,
                                style={'opacity': '0.5', 'cursor': 'not-allowed',
                                       'padding': '8px 20px', 'borderRadius': '4px'})
    else:
        entering_v = lp.nonbasic_vars[entering_s]
        leaving_v  = lp.basic_vars[leaving_r]
        pivot_val  = lp.B[leaving_r][entering_s]
        if pivot_val == 0:
            pivot_msg = html.Div("Pivot element = 0 — choose a different row.", className="msg-error")
            pivot_btn = html.Button("Perform pivot (zero element)", id='btn-pivot', n_clicks=0,
                                    disabled=True, style={'opacity': '0.5', 'cursor': 'not-allowed',
                                                          'padding': '8px 20px', 'borderRadius': '4px'})
        else:
            pivot_msg = html.Div(["Enter ", html.B(xsub(entering_v)), ", leave ", html.B(xsub(leaving_v)),
                                   f"  (pivot = {frac_str(pivot_val)})"], className="msg-success")
            pivot_btn = html.Button("Perform pivot ↔", id='btn-pivot', n_clicks=0,
                                    style={'background': '#1976d2', 'color': 'white', 'border': 'none',
                                           'padding': '8px 20px', 'cursor': 'pointer',
                                           'borderRadius': '4px', 'fontWeight': 'bold'})

    basis_section = html.Div([
        html.Hr(),
        html.H6("Set basis"),
        html.P([f"Current basis: {{", ", ".join([f"x{v}" for v in sorted(lp.basic_vars)]),
                f"}}. Enter {lp.m} indices:"],
               style={'fontSize': '13px', 'marginBottom': '8px'}),
        dcc.Input(id='basis-input', value=", ".join(str(v) for v in lp.basic_vars),
                  type='text', style={'fontFamily': 'monospace', 'width': '200px', 'marginRight': '8px'}),
        html.Button("Apply basis", id='btn-basis', n_clicks=0,
                    style={'padding': '6px 14px', 'borderRadius': '4px', 'cursor': 'pointer'}),
        html.Div(id='basis-msg', style={'marginTop': '6px'}),
    ])

    auto_section = html.Div([
        html.Hr(),
        html.H6("Auto solve"),
        dcc.RadioItems(
            id='rule-radio',
            options=[{'label': ' Bland', 'value': 'bland'},
                     {'label': ' Largest coeff', 'value': 'largest_coeff'},
                     {'label': ' Lexicographic', 'value': 'lexicographic'}],
            value='bland', inline=True,
            style={'marginBottom': '10px', 'fontSize': '13px'},
        ),
        html.Button("One step", id='btn-one-step', n_clicks=0,
                    style={'padding': '7px 16px', 'marginRight': '8px',
                           'borderRadius': '4px', 'cursor': 'pointer'}),
        html.Button("Solve to optimum", id='btn-solve', n_clicks=0,
                    style={'padding': '7px 16px', 'background': '#1976d2', 'color': 'white',
                           'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer',
                           'fontWeight': 'bold'}),
        html.Div(id='auto-msg', style={'marginTop': '8px'}),
    ])

    return html.Div([
        html.H5("Manual pivot", style={'marginBottom': '12px'}),
        table,
        html.Div([pivot_msg, pivot_btn],
                 style={'marginTop': '12px', 'display': 'flex', 'alignItems': 'center', 'gap': '12px'}),
        status,
        basis_section,
        auto_section,
    ])

# ── Callback: History panel ────────────────────────────────────────────────────

@app.callback(
    Output('history-panel', 'children'),
    Input('history-store', 'data'),
)
def render_history(history):
    if not history:
        return [
            html.P("No pivots yet.", style={'color': '#888', 'fontSize': '13px'}),
            html.Button(id='btn-undo', n_clicks=0, style={'display': 'none'}),
        ]
    undo_btn = html.Button("↩ Undo last pivot", id='btn-undo', n_clicks=0,
                           style={'padding': '6px 14px', 'marginBottom': '10px', 'width': '100%',
                                  'borderRadius': '4px', 'cursor': 'pointer'})
    entries = []
    for idx, entry in enumerate(reversed(history)):
        label, _, latex_before = entry
        step_num = len(history) - idx
        body = dcc.Markdown(latex_before, mathjax=True, style={'fontSize': '12px'})
        entries.append(html.Details([
            html.Summary(f"Step {step_num}: {label}"),
            html.Div(body, className="entry-body"),
        ], className="history-entry", open=(idx == 0)))
    return [
        undo_btn,
        html.P(f"{len(history)} pivot(s) performed.",
               style={'fontSize': '12px', 'color': '#666', 'marginBottom': '10px'}),
        *entries,
    ]


instructions.register_callbacks(app)

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host='0.0.0.0', port=PORT)
