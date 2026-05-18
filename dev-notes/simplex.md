# Simplex Explorer — Developer Notes

## What this project is

An interactive web application for exploring the **Simplex algorithm** step-by-step.
Users enter a linear program (LP) in slack form, then manually perform pivot operations
or let the app auto-solve it. The primary audience is students and educators learning
how the Simplex method works.

The app has two pages:
- **`/` — Tool:** The main pivot explorer (enter an LP, step through pivots, view history).
- **`/instructions` — Instructions:** A 7-section interactive tutorial explaining LP fundamentals,
  slack form, the pivot operation, optimality conditions, and how to use the tool.

## How to run

```bash
pip install -r requirements.txt
python3 app.py          # starts on http://localhost:8050
```

On startup `app.py` kills any existing process on port 8050 before launching.
The port can be overridden with the `PORT` environment variable.

## Tech stack

| Library | Role |
|---|---|
| **Plotly Dash** (`dash`, `dash-bootstrap-components`) | Web framework; all UI is server-rendered Python components |
| **Python `fractions.Fraction`** | Exact rational arithmetic throughout the LP engine |
| **NumPy** | Input conversion only (`_convert_input`) — not used in pivot math |
| **Matplotlib / Pillow** | Fallback LaTeX renderer (when pdflatex is absent) |
| **pdflatex + ImageMagick** | Primary LaTeX-to-PNG renderer (used in the "Current system" panel) |
| **Plotly** | 2D feasible-region diagram on the Instructions page |
| **SciPy** (`ConvexHull`) | Computes feasible-region vertices for the diagram |

## File map

```
app.py              Main Dash app — layout, all callbacks, UI helpers
simplex.py          Core LP engine (LinearProgram class)
instructions.py     Instructions page layout + clientside callback
latex_rendering.py  LatexRenderer: pdflatex -> PNG, with matplotlib fallback
requirements.txt    Python dependencies
assets/style.css    Custom CSS loaded automatically by Dash
components/index.html  Leftover Streamlit custom-component HTML (no longer used)
old code/           Previous Streamlit-based UI (reference only, not run)
```

---

## simplex.py — The LP engine

### LP representation (slack form)

```
max   A[0] + A[1]*x_1 + ... + A[n]*x_n
s.t.  x_{n+i} = C[i] + sum_j B[i][j] * x_j    for i = 1..m
      all x >= 0
```

- `B` — `m × n` matrix (Fraction); coefficient of nonbasic var `j` in constraint `i`.
- `C` — length-`m` list (Fraction); right-hand-side constants (values of basic vars when nonbasics = 0).
- `A` — length-`n+1` list (Fraction); `A[0]` is the objective constant, `A[k]` (k≥1) is the objective coefficient of the k-th nonbasic variable.
- `basic_vars` — list of `m` global 1-based variable indices currently basic.
- `nonbasic_vars` — list of `n` global 1-based variable indices currently nonbasic.

Variables are numbered globally: nonbasics start at `1..n`, basics at `n+1..n+m`.
After pivots the indices in `basic_vars`/`nonbasic_vars` reflect the current basis.
All arithmetic uses `fractions.Fraction` — no floating-point rounding.

### Key methods

| Method | Description |
|---|---|
| `LinearProgram(B, C, A)` | Construct. Accepts lists or numpy arrays; converts to Fraction internally. No feasibility check — `C[i] < 0` is allowed for educational purposes. |
| `swap_variables(basic_idx, nonbasic_idx)` | Core pivot. Swaps one basic and one nonbasic variable, updating B, C, A in-place. Raises `ValueError` if the pivot element is zero. Uses a snapshot of old values to avoid update-order bugs. |
| `find_pivot(rule='bland')` | Returns `(entering_global_idx, leaving_global_idx)` or `None` if optimal. Raises `ValueError` if unbounded. Supported rules: `'bland'`, `'largest_coeff'`, `'lexicographic'`. |
| `set_basis(basic_indices)` | Pivot to an arbitrary basis by repeatedly calling `swap_variables`. Useful for jumping to a specific vertex. |
| `to_latex(matrix_form=False)` | Render the current system as a LaTeX string. `matrix_form=False` produces aligned equations; `True` produces compact matrix/vector notation. |
| `to_float()` | Returns a dict of numpy float64 arrays (for external use / debugging). |

### Pivot rules in `find_pivot`

- **`bland`** — smallest global index for both entering and leaving variable. Guarantees termination.
- **`largest_coeff`** — entering = variable with the largest positive objective coefficient; leaving = standard min-ratio with index tie-break.
- **`lexicographic`** — entering = smallest global index; leaving = lexicographic ratio test. Also guarantees termination.

---

## app.py — Dash application

### State management (dcc.Store)

Dash stores keep all mutable state client-side as JSON. The main stores are:

| Store id | Contents |
|---|---|
| `lp-store` | Serialized current LP (`B`, `C`, `A`, `basic_vars`, `nonbasic_vars` — all as strings for JSON safety) |
| `original-lp` | Snapshot of the LP right after "Generate" — used by "Edit original" |
| `history-store` | List of `(label, lp_snapshot, latex_str, img_b64)` tuples, one per pivot |
| `selection-store` | `{entering_s, leaving_r}` — column/row indices currently selected in the pivot table |
| `app-mode` | `'edit'` or `'view'` — controls which sections render |
| `editor-meta` | `{m, n}` — dimensions of the editor grid |
| `editor-values` | `{B, C, A}` — raw string values from the editor inputs |
| `font-size-store` | LaTeX renderer font size (synced from sidebar slider) |
| `panel-mult-store` | Height multiplier for the LaTeX panel |
| `panel-pad-store` | Extra padding for the LaTeX panel |
| `left-collapsed` / `right-collapsed` | Sidebar collapse state |

### LP serialization helpers

- `lp_to_dict(lp)` — `LinearProgram` → JSON-safe dict (all Fractions become strings).
- `dict_to_lp(d)` — dict → `LinearProgram` (strings parsed back to Fractions).
- `lp_to_editor(lp_dict)` — dict → `(meta, values)` for populating the editor.
- `parse_num(x)` — parse a string as a Fraction; accepts decimals, fractions, integers.

### UI sections and callbacks

**Editor section** (`render_editor`) — visible only in `'edit'` mode.
Two tabs: *Table* (grid of `dcc.Input` cells with pattern-matching IDs) and *Text* (CSV textarea).
Add/remove row and column buttons update `editor-meta` and `editor-values` stores.
Cell validation callbacks (`validate_B/C/A`) use `MATCH` to style individual cells red when input is not a valid fraction.
"Generate →" parses the editor state, creates a `LinearProgram`, and switches to `'view'` mode.

**System section** (`render_system`) — renders the current LP as a LaTeX image.
Triggers on LP/mode/font changes. Shows equation form, matrix form, or raw LaTeX in tabs.
Also shows a feasibility badge (all `C[i] >= 0` ↔ feasible).
The rendered base64 PNG is stored in `current-img-store` for use in pivot history entries.

**Controls section** (`render_controls`) — pivot table and action buttons.
Rebuilt on every LP or selection change (fast — no LaTeX rendering).
Contains: pivot table, "Perform pivot" button, basis input, auto-solve controls.

**Pivot table interaction** — three overlapping callbacks:
- `click_col` / `click_row` — toggle individual column/row selection.
- `click_cell` — select both column and row at once (most convenient).

**Auto-solve** (`auto_pivot`) — handles both "One step" and "Solve to optimum" via `ctx.triggered_id`.

**History panel** (`render_history`) — shows each pivot as a collapsible `<details>` element;
if a base64 image was captured at pivot time it shows the PNG, otherwise falls back to raw LaTeX text.

### Sidebar layout

- **Left sidebar** — app title, "Edit original"/"Edit current" buttons, LaTeX font size slider,
  row-height multiplier, panel padding. Collapsible.
- **Right sidebar** — pivot history list with undo button. Collapsible.

---

## instructions.py — Tutorial page

Seven static sections built from `dcc.Markdown` blocks (MathJax enabled):

1. Introduction to Linear Programming
2. A 2D Example (interactive Plotly diagram + objective-value slider)
3. Standard Form and Slack Form
4. The Pivot Operation
5. Optimality Conditions
6. Handling Infeasibility: Two-Phase Method
7. Using This App

The 2D diagram is updated via a **clientside callback** (JavaScript) to avoid a round-trip to the
server on every slider drag. The callback mutates the Plotly figure's trace data directly in the browser.

`get_layout()` returns the full page layout (TOC sidebar + content column).
`register_callbacks(app)` attaches the clientside callback to the passed Dash app instance.

---

## latex_rendering.py — LaTeX rendering

`LatexRenderer(font_size, font_color)` — renders a LaTeX string to a PIL Image.

**Primary path:** shell out to `pdflatex` (produces a PDF), then `convert` (ImageMagick) to PNG.
Wraps the input in a minimal `article` document with `amsmath` and `xcolor`.

**Fallback path:** `matplotlib` mathtext — strips LaTeX environments (align*, array, multicolumn)
and renders each `\\`-separated line as an inline math expression.
Less faithful but requires no system dependencies.

The `_renderer` singleton in `app.py` is reused across requests; its `font_size` is mutated
by the font-size slider callback (safe because Dash callbacks are synchronous).

---

## Architectural notes

- The project was previously Streamlit-based (`old code/streamlit_app.py`).
  `components/index.html` is a leftover Streamlit custom component that rendered the pivot table
  as an interactive HTML/JS iframe; it is not used in the current Dash app.
- The Dash app uses `suppress_callback_exceptions=True` because the pivot table buttons
  (`btn-pivot`, `btn-undo`, etc.) are created dynamically inside callbacks and don't exist
  at startup.
- `fractions.Fraction` is used end-to-end to avoid floating-point drift across many pivot steps.
  Inputs accept decimals or fraction strings (e.g. `"1/3"`, `"0.5"`) and are converted once on entry.
- The devcontainer (`devcontainer.json`) uses a Node 22 base image with Python added via `apt`.
  It forwards ports 8050 (Dash) and 8501 (Streamlit, unused).
