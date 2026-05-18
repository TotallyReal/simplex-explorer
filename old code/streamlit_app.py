import os
import sys
import pandas as pd
import streamlit as st
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simplex import LinearProgram

# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Simplex Explorer", layout="wide", page_icon="📐")

st.markdown("""
<style>
[data-testid="stDataFrame"] [data-testid="column-header-menu"] { display: none !important; }
[data-testid="stDataFrame"] .gdg-header-button { display: none !important; }
[data-testid="stDataFrame"] button[aria-label="column menu"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def frac_str(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"

def to_katex(s: str) -> str:
    return (s.replace(r'\begin{align*}', r'\begin{aligned}')
             .replace(r'\end{align*}', r'\end{aligned}'))

def parse_num(x: str) -> Fraction:
    x = x.strip()
    try:
        return Fraction(x)
    except ValueError:
        return Fraction(float(x)).limit_denominator(10 ** 6)

def parse_vec(text: str) -> list:
    return [parse_num(x) for x in text.split(',') if x.strip()]

def parse_matrix(text: str) -> list:
    return [parse_vec(row) for row in text.strip().splitlines() if row.strip()]

def snapshot(lp: LinearProgram) -> dict:
    return {
        'B': [row[:] for row in lp.B],
        'C': lp.C[:],
        'A': lp.A[:],
        'basic_vars': lp.basic_vars[:],
        'nonbasic_vars': lp.nonbasic_vars[:],
        'm': lp.m,
        'n': lp.n,
    }

def restore(lp: LinearProgram, snap: dict) -> None:
    lp.B = [row[:] for row in snap['B']]
    lp.C = snap['C'][:]
    lp.A = snap['A'][:]
    lp.basic_vars = snap['basic_vars'][:]
    lp.nonbasic_vars = snap['nonbasic_vars'][:]

def push_history(lp: LinearProgram, label: str) -> None:
    st.session_state.history.append((label, snapshot(lp), lp.to_latex()))

# ── session state ─────────────────────────────────────────────────────────────

if 'lp' not in st.session_state:
    st.session_state.lp = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'pt_entering_s' not in st.session_state:
    st.session_state.pt_entering_s = None
if 'pt_leaving_r' not in st.session_state:
    st.session_state.pt_leaving_r = None

# ── sidebar: system definition ────────────────────────────────────────────────

with st.sidebar:
    st.title("📐 Simplex Explorer")
    st.divider()
    st.subheader("Define system")
    st.caption(
        "**m** = number of inequalities (constraints).  "
        "**n** = number of decision variables.  "
        "Total variables: n + m."
    )

    B_text = st.text_area(
        "**B** — constraint matrix  (m × n, one row per line)",
        value="-6, -4, -2\n-3, -2, -5",
        height=100,
    )
    C_text = st.text_area(
        "**C** — RHS vector  (m values)",
        value="5, 4",
        height=68,
    )
    A_text = st.text_area(
        "**A** — objective  (n+1 values: constant, then x₁…xₙ coefficients)",
        value="0, 5, 4, 3",
        height=68,
    )

    if st.button("Create / Reset", type="primary", use_container_width=True):
        try:
            lp = LinearProgram(parse_matrix(B_text), parse_vec(C_text), parse_vec(A_text))
            st.session_state.lp = lp
            st.session_state.history = []
            st.session_state.pt_entering_s = None
            st.session_state.pt_leaving_r = None
            st.success(f"Created: m={lp.m}, n={lp.n}, {lp.n + lp.m} variables total")
        except Exception as e:
            st.error(f"Parse error: {e}")

    st.divider()
    st.caption("Tip: values can be integers, fractions (1/2), or decimals (0.5).")

# ── gate: no system yet ───────────────────────────────────────────────────────

lp: LinearProgram | None = st.session_state.lp

if lp is None:
    st.info("👈 Define a system in the sidebar and click **Create / Reset**.")
    st.stop()

# ── layout ────────────────────────────────────────────────────────────────────

col_main, col_hist = st.columns([3, 2], gap="large")

# ════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN: current system + pivot controls
# ════════════════════════════════════════════════════════════════════════════

with col_main:

    st.subheader("Current system")

    tab_eq, tab_mat, tab_raw = st.tabs(["Equations", "Matrix form", "Raw LaTeX"])
    with tab_eq:
        st.latex(to_katex(lp.to_latex(matrix_form=False)))
    with tab_mat:
        st.latex(to_katex(lp.to_latex(matrix_form=True)))
    with tab_raw:
        st.code(lp.to_latex(matrix_form=False), language="latex")

    st.divider()

    try:
        suggested = lp.find_pivot(rule='bland')
        is_unbounded = False
    except ValueError:
        suggested = None
        is_unbounded = True

    if is_unbounded:
        st.error("LP appears unbounded (no leaving variable found for Bland's entering choice).")
    elif suggested is None:
        st.success(f"🎉 **Optimal!**  Objective value = **{frac_str(lp.A[0])}**")
        sol = {f"x_{v}": frac_str(lp.C[i]) for i, v in enumerate(lp.basic_vars)}
        sol.update({f"x_{v}": "0" for v in lp.nonbasic_vars})
        st.write("Optimal solution:", sol)
    else:
        t_manual, t_basis, t_auto = st.tabs(["Manual pivot", "Set basis", "Auto solve"])

        with t_manual:
            col_labels = ['const'] + [f'x_{v}' for v in lp.nonbasic_vars]
            row_labels = ['max'] + [f'x_{v}' for v in lp.basic_vars]

            obj_row = [frac_str(lp.A[0])] + [frac_str(lp.A[s + 1]) for s in range(lp.n)]
            con_rows = [
                [frac_str(lp.C[i])] + [frac_str(lp.B[i][j]) for j in range(lp.n)]
                for i in range(lp.m)
            ]
            df = pd.DataFrame([obj_row] + con_rows, columns=col_labels, index=row_labels)

            entering_s = st.session_state.pt_entering_s
            leaving_r  = st.session_state.pt_leaving_r

            col_tbl, col_rng = st.columns([3, 2])

            with col_tbl:
                table_key = f"pt_{id(lp)}_{len(st.session_state.history)}"
                event = st.dataframe(
                    df,
                    on_select="rerun",
                    selection_mode=["single-column", "single-row"],
                    key=table_key,
                    use_container_width=True,
                )

                sel_cols = event.selection.columns
                sel_rows = event.selection.rows

                new_entering_s = None
                if sel_cols and sel_cols[0] != 'const':
                    for s, v in enumerate(lp.nonbasic_vars):
                        if f'x_{v}' == sel_cols[0]:
                            new_entering_s = s
                            break

                new_leaving_r = None
                if sel_rows and sel_rows[0] > 0:
                    new_leaving_r = sel_rows[0] - 1

                st.session_state.pt_entering_s = new_entering_s
                st.session_state.pt_leaving_r  = new_leaving_r
                entering_s = new_entering_s
                leaving_r  = new_leaving_r

            with col_rng:
                if entering_s is None:
                    st.caption("← Click a variable column to see bounds")
                else:
                    ev = lp.nonbasic_vars[entering_s]
                    st.markdown(f"**Bounds for x_{ev}:**")
                    for i in range(lp.m):
                        bv = lp.basic_vars[i]
                        b  = lp.B[i][entering_s]
                        c  = lp.C[i]
                        if b < 0:
                            bound = c / abs(b)
                            st.latex(
                                rf'x_{{{bv}}} = {frac_str(c)} + ({frac_str(b)})\,x_{{{ev}}} \geq 0'
                                rf'\;\Rightarrow\; x_{{{ev}}} \leq {frac_str(bound)}'
                            )
                        else:
                            st.markdown(f"x_{bv}: coeff {frac_str(b)} ≥ 0 → no bound")

            st.divider()
            if entering_s is None or leaving_r is None:
                parts = (["a column"] if entering_s is None else []) + \
                        (["a row"]    if leaving_r  is None else [])
                st.error(f"Choose {' and '.join(parts)} in the table above.")
                st.button("Perform pivot", disabled=True, use_container_width=True)
            else:
                entering_v = lp.nonbasic_vars[entering_s]
                leaving_v  = lp.basic_vars[leaving_r]
                pivot_val  = lp.B[leaving_r][entering_s]
                if pivot_val == 0:
                    st.error(
                        f"Pivot element B[x_{leaving_v}, x_{entering_v}] = 0 — "
                        "choose a different row."
                    )
                    st.button("Perform pivot (zero element)", disabled=True, use_container_width=True)
                else:
                    st.success(
                        f"Enter **x_{entering_v}**, leave **x_{leaving_v}**"
                        f"  (pivot = {frac_str(pivot_val)})"
                    )
                    if st.button("Perform pivot ↔", type="primary", use_container_width=True):
                        push_history(lp, f"Enter x_{entering_v}, leave x_{leaving_v}")
                        lp.swap_variables(leaving_v, entering_v)
                        st.session_state.pt_entering_s = None
                        st.session_state.pt_leaving_r  = None
                        st.rerun()

        with t_basis:
            st.markdown(f"""
Current basis: **{{{', '.join(f'x_{v}' for v in sorted(lp.basic_vars))}}}**
Variables range from **x_1** to **x_{lp.n + lp.m}**.

Enter the indices of the {lp.m} variables you want as the new basis.
""")
            basis_input = st.text_input(
                f"New basis ({lp.m} indices, comma-separated)",
                value=", ".join(str(v) for v in lp.basic_vars),
            )
            if st.button("Apply basis", use_container_width=True):
                try:
                    indices = [int(x.strip()) for x in basis_input.split(',') if x.strip()]
                    push_history(lp, f"Set basis → {{{', '.join(f'x_{i}' for i in indices)}}}")
                    lp.set_basis(indices)
                    st.rerun()
                except Exception as e:
                    st.session_state.history.pop()
                    st.error(str(e))

        with t_auto:
            rule = st.radio(
                "Anti-cycling rule",
                ["bland", "largest_coeff", "lexicographic"],
                horizontal=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("One step", use_container_width=True):
                    try:
                        p = lp.find_pivot(rule=rule)
                        if p:
                            ev, lv = p
                            push_history(lp, f"[{rule}] Enter x_{ev}, leave x_{lv}")
                            lp.swap_variables(lv, ev)
                    except ValueError as e:
                        st.error(str(e))
                    st.rerun()
            with c2:
                if st.button("Solve to optimum", type="primary", use_container_width=True):
                    try:
                        while True:
                            p = lp.find_pivot(rule=rule)
                            if p is None:
                                break
                            ev, lv = p
                            push_history(lp, f"[{rule}] Enter x_{ev}, leave x_{lv}")
                            lp.swap_variables(lv, ev)
                    except ValueError as e:
                        st.error(str(e))
                    st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN: pivot history
# ════════════════════════════════════════════════════════════════════════════

with col_hist:
    st.subheader("Pivot history")

    history = st.session_state.history

    if not history:
        st.caption("No pivots yet — the history of each step will appear here.")
    else:
        if st.button("↩  Undo last pivot", use_container_width=True):
            _, snap_before, _ = history.pop()
            restore(lp, snap_before)
            st.rerun()

        st.caption(f"{len(history)} pivot(s) performed.")

        for i, (label, _, latex_before) in enumerate(reversed(history)):
            step_num = len(history) - i
            with st.expander(f"Step {step_num}: {label}", expanded=(i == 0)):
                st.latex(to_katex(latex_before))
