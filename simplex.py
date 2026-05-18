"""
Simplex algorithm for linear programs in slack form.

LP representation:
    max  A[0] + A[1]*x_1 + ... + A[n]*x_n
    s.t. x_{n+i} = C[i] + sum_j B[i][j] * x_j   for i = 1..m
         x_j, x_{n+i} >= 0

Variables are numbered 1..n (non-basic) and n+1..n+m (basic) globally.
All arithmetic uses fractions.Fraction for exactness.
"""

from fractions import Fraction
from typing import Union

import numpy as np

ArrayLike = Union[list, np.ndarray]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _to_fraction(x) -> Fraction:
    """Convert a scalar to Fraction (floats are approximated to avoid long denominators)."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float):
        return Fraction(x).limit_denominator(10 ** 12)
    if isinstance(x, str):
        return Fraction(x)
    return Fraction(float(x)).limit_denominator(10 ** 12)


def _convert_input(M: ArrayLike) -> list:
    """Recursively convert an array-like to a nested list of Fraction."""
    if isinstance(M, np.ndarray):
        if M.ndim == 1:
            return [_to_fraction(x) for x in M]
        return [[_to_fraction(x) for x in row] for row in M]
    if isinstance(M, list):
        if not M:
            return []
        if isinstance(M[0], (list, np.ndarray)):
            return [[_to_fraction(x) for x in row] for row in M]
        return [_to_fraction(x) for x in M]
    raise TypeError(f"Unsupported input type: {type(M)}")


def _frac_to_latex(f: Fraction) -> str:
    """Format a Fraction as a LaTeX string (integer or \\frac{p}{q})."""
    if f.denominator == 1:
        return str(f.numerator)
    sign = '-' if f < 0 else ''
    return rf'{sign}\frac{{{abs(f.numerator)}}}{{{f.denominator}}}'


def _matrix_to_latex(M: list) -> str:
    """Format a 2-D list of Fraction as a LaTeX pmatrix."""
    rows = [' & '.join(_frac_to_latex(v) for v in row) for row in M]
    body = ' \\\\\n'.join(rows)
    return f'\\begin{{pmatrix}}\n{body}\n\\end{{pmatrix}}'


def _term_latex(coeff: Fraction, var_name: str, is_first: bool) -> str:
    """
    Format a single term 'coeff * var_name' for a LaTeX sum.

    Args:
        coeff: coefficient value (must be non-zero).
        var_name: LaTeX variable name string, e.g. 'x_{3}'.
        is_first: True when this is the leading term of the sum.

    Returns:
        LaTeX fragment including leading sign for non-first terms.
    """
    abs_c = abs(coeff)
    coeff_str = '' if abs_c == 1 else _frac_to_latex(abs_c) + ' '
    if is_first:
        prefix = '-' if coeff < 0 else ''
        return f'{prefix}{coeff_str}{var_name}'
    sign = '+' if coeff > 0 else '-'
    return f' {sign} {coeff_str}{var_name}'


def _col_vec_latex(names: list) -> str:
    """Format a list of LaTeX strings as a column pmatrix."""
    body = ' \\\\\n'.join(names)
    return f'\\begin{{pmatrix}}\n{body}\n\\end{{pmatrix}}'


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LinearProgram:
    """
    A linear program in slack form.

    Internal representation after any sequence of pivots:
        max  A[0] + sum_{s=0}^{n-1} A[s+1] * x_{nonbasic_vars[s]}
        s.t. x_{basic_vars[r]} = C[r] + sum_{s=0}^{n-1} B[r][s] * x_{nonbasic_vars[s]}
             all variables >= 0

    Variables are tracked by their global 1-based index.
    """

    def __init__(self, B: ArrayLike, C: ArrayLike, A: ArrayLike) -> None:
        """
        Construct the LP.

        Args:
            B: m×n matrix; B[i][j] is the coefficient of the j-th non-basic variable
               in the i-th constraint.
            C: m-vector; C[i] is the value of the i-th basic variable when all
               non-basics are zero (the current basic feasible solution value).
            A: (n+1)-vector; A[0] is the objective constant, A[k] (k=1..n) is the
               objective coefficient of the k-th non-basic variable.

        Note:
            No feasibility check is performed — C[i] < 0 is allowed so that
            arbitrary systems can be constructed and inspected for educational use.
        """
        self.B: list[list[Fraction]] = _convert_input(B)
        self.C: list[Fraction] = _convert_input(C)
        self.A: list[Fraction] = _convert_input(A)

        self.m: int = len(self.C)
        self.n: int = len(self.A) - 1

        if len(self.B) != self.m:
            raise ValueError(f"B has {len(self.B)} rows but C has length {self.m}")
        for i, row in enumerate(self.B):
            if len(row) != self.n:
                raise ValueError(f"B row {i} has {len(row)} columns, expected {self.n}")

        # Global 1-based variable indices: non-basics = 1..n, basics = n+1..n+m
        self.nonbasic_vars: list[int] = list(range(1, self.n + 1))
        self.basic_vars: list[int] = list(range(self.n + 1, self.n + self.m + 1))

    # ------------------------------------------------------------------
    # Core pivot
    # ------------------------------------------------------------------

    def swap_variables(self, basic_idx: int, nonbasic_idx: int) -> None:
        """
        Pivot: swap a basic variable with a non-basic variable.

        Applies the standard simplex pivot, updating B, C, and A in-place so
        that nonbasic_idx becomes basic (in the row formerly occupied by
        basic_idx) and basic_idx becomes non-basic (in the column formerly
        occupied by nonbasic_idx).

        Args:
            basic_idx: global 1-based index of the leaving (basic) variable.
            nonbasic_idx: global 1-based index of the entering (non-basic) variable.

        Raises:
            ValueError: if either index is not in the expected set, or if the
                        pivot element B[r][s] is zero.
        """
        try:
            r = self.basic_vars.index(basic_idx)
        except ValueError:
            raise ValueError(f"x_{basic_idx} is not in the current basis")
        try:
            s = self.nonbasic_vars.index(nonbasic_idx)
        except ValueError:
            raise ValueError(f"x_{nonbasic_idx} is not a current non-basic variable")

        pivot = self.B[r][s]
        if pivot == 0:
            raise ValueError(
                f"Degenerate pivot: B[{r}][{s}] = 0 "
                f"(x_{basic_idx} leaving, x_{nonbasic_idx} entering)"
            )

        # Snapshot old values before any mutation to avoid update-order bugs.
        old_B_row_r = self.B[r][:]                          # length n
        old_C_r = self.C[r]
        old_B_col_s = [self.B[i][s] for i in range(self.m)] # length m
        old_As = self.A[s + 1]                              # obj coeff of entering var

        # --- Update row r (new equation for the entering variable) ---
        # Derived from: x_{entering} = -C[r]/pivot + (1/pivot)*x_{leaving} + ...
        self.C[r] = -old_C_r / pivot
        self.B[r][s] = Fraction(1) / pivot          # coeff of x_{leaving} (now non-basic)
        for j in range(self.n):
            if j != s:
                self.B[r][j] = -old_B_row_r[j] / pivot

        # --- Update all other rows (eliminate entering variable) ---
        for i in range(self.m):
            if i == r:
                continue
            Bis = old_B_col_s[i]
            self.C[i] = self.C[i] - Bis * old_C_r / pivot
            self.B[i][s] = Bis / pivot               # coeff of x_{leaving} in row i
            for j in range(self.n):
                if j != s:
                    self.B[i][j] = self.B[i][j] - Bis * old_B_row_r[j] / pivot

        # --- Update objective ---
        self.A[0] = self.A[0] - old_As * old_C_r / pivot
        self.A[s + 1] = old_As / pivot               # coeff of x_{leaving} (now non-basic)
        for k in range(self.n):
            if k != s:
                self.A[k + 1] = self.A[k + 1] - old_As * old_B_row_r[k] / pivot

        # --- Swap the index tracking lists ---
        self.basic_vars[r] = nonbasic_idx
        self.nonbasic_vars[s] = basic_idx

    # ------------------------------------------------------------------
    # Pivot selection
    # ------------------------------------------------------------------

    def find_pivot(self, rule: str = 'bland') -> 'tuple[int, int] | None':
        """
        Find the next (entering, leaving) variable pair for the simplex method.

        Args:
            rule: anti-cycling pivot rule. Supported values:
                - 'bland': Bland's rule — smallest global index for both entering
                  and leaving (guarantees termination).
                - 'largest_coeff': largest positive objective coefficient for
                  entering; standard min-ratio with index tie-break for leaving.
                - 'lexicographic': smallest global index for entering; lexicographic
                  ratio test for leaving (guarantees termination).

        Returns:
            (entering_global_idx, leaving_global_idx), or None if already optimal
            (no positive objective coefficient).

        Raises:
            ValueError: if the LP is unbounded for the chosen entering variable,
                        or if rule is not recognized.
        """
        # Collect non-basic columns with a positive objective coefficient.
        candidates = [
            (self.nonbasic_vars[s], s)
            for s in range(self.n)
            if self.A[s + 1] > 0
        ]
        if not candidates:
            return None  # Current solution is optimal.

        # Select entering variable according to the rule.
        if rule == 'bland':
            _, entering_s = min(candidates, key=lambda t: t[0])
        elif rule == 'largest_coeff':
            _, entering_s = max(candidates, key=lambda t: self.A[t[1] + 1])
        elif rule == 'lexicographic':
            _, entering_s = min(candidates, key=lambda t: t[0])
        else:
            raise ValueError(
                f"Unknown pivot rule '{rule}'. "
                "Choose from 'bland', 'largest_coeff', 'lexicographic'."
            )

        # Ratio test: rows where B[i][entering_s] < 0 give a finite upper bound on x_s.
        valid_rows = [i for i in range(self.m) if self.B[i][entering_s] < 0]
        if not valid_rows:
            raise ValueError(
                f"LP is unbounded: x_{self.nonbasic_vars[entering_s]} can increase "
                "without bound (no binding constraint)."
            )

        # Select leaving variable according to the rule.
        if rule in ('bland', 'largest_coeff'):
            # Min ratio C[i]/|B[i,s]|, tie-break by smallest global index.
            def ratio_key_bland(i: int):
                ratio = self.C[i] / abs(self.B[i][entering_s])
                return (ratio, self.basic_vars[i])
            leaving_r = min(valid_rows, key=ratio_key_bland)

        else:  # lexicographic
            # Lexicographic ratio test: minimise the tuple
            # (C[i]/|B[i,s]|, B[i,0]/|B[i,s]|, ..., B[i,n-1]/|B[i,s]|).
            def lex_key(i: int):
                piv_abs = abs(self.B[i][entering_s])
                return tuple(
                    [self.C[i] / piv_abs]
                    + [self.B[i][j] / piv_abs for j in range(self.n)]
                )
            leaving_r = min(valid_rows, key=lex_key)

        return (self.nonbasic_vars[entering_s], self.basic_vars[leaving_r])

    # ------------------------------------------------------------------
    # Basis change
    # ------------------------------------------------------------------

    def set_basis(self, basic_indices: list[int]) -> bool:
        """
        Pivot the system so that the given variables form the basis.

        The method checks that each required basis change is achievable (non-zero
        pivot element) and performs the necessary swaps.

        Args:
            basic_indices: list of m distinct global 1-based variable indices
                           to make basic.

        Returns:
            True on success.

        Raises:
            ValueError: if basic_indices is invalid (wrong length, duplicates,
                        out-of-range), or if the target basis is not achievable
                        because all eligible pivot elements are zero.
        """
        if len(basic_indices) != self.m:
            raise ValueError(
                f"Expected {self.m} basis indices, got {len(basic_indices)}"
            )
        if len(set(basic_indices)) != self.m:
            raise ValueError("Duplicate entries in basic_indices")
        if any(idx < 1 or idx > self.n + self.m for idx in basic_indices):
            raise ValueError(
                f"All indices must be in range 1..{self.n + self.m}"
            )

        target_set = set(basic_indices)

        # Iteratively pivot one variable at a time into the basis.
        # Restart the scan after each successful pivot (basis lists change).
        progress = True
        while progress:
            progress = False
            for target_var in basic_indices:
                if target_var in self.basic_vars:
                    continue  # Already basic.

                s = self.nonbasic_vars.index(target_var)

                # Eligible leaving row: current basic variable not in target_set,
                # and pivot element is non-zero.
                eligible = [
                    r for r in range(self.m)
                    if self.basic_vars[r] not in target_set
                    and self.B[r][s] != 0
                ]
                if not eligible:
                    raise ValueError(
                        f"Cannot pivot x_{target_var} into the basis: "
                        "all eligible rows have a zero coefficient "
                        "(the target basis may not be achievable)."
                    )
                self.swap_variables(self.basic_vars[eligible[0]], target_var)
                progress = True
                break  # Restart after each pivot — indices have shifted.

        if set(self.basic_vars) != target_set:
            raise ValueError(
                "Failed to reach the target basis (possible linear dependence)."
            )
        return True

    # ------------------------------------------------------------------
    # LaTeX export
    # ------------------------------------------------------------------

    def to_latex(self, matrix_form: bool = False) -> str:
        """
        Return a LaTeX string describing the current LP system.

        Args:
            matrix_form: if False (default), write each equation in full;
                         if True, use compact matrix/vector notation.

        Returns:
            A LaTeX string enclosed in an align* environment.
        """
        if matrix_form:
            return self._to_latex_matrix()
        return self._to_latex_array()

    def _build_sum_latex(
        self,
        constant: Fraction,
        coeffs: list[Fraction],
        var_indices: list[int],
    ) -> str:
        """Build a LaTeX sum 'constant + c1*x_v1 + c2*x_v2 + ...' skipping zeros."""
        terms: list[str] = []

        # Constant term: always show when non-zero; show '0' if everything is zero.
        all_zero_coeffs = all(c == 0 for c in coeffs)
        if constant != 0 or all_zero_coeffs:
            terms.append(_frac_to_latex(constant))

        for c, v in zip(coeffs, var_indices):
            if c == 0:
                continue
            var = f'x_{{{v}}}'
            terms.append(_term_latex(c, var, not terms))

        return ''.join(terms) if terms else '0'

    def _to_latex_full(self) -> str:
        lines: list[str] = []

        # Objective
        obj = self._build_sum_latex(
            self.A[0],
            [self.A[k + 1] for k in range(self.n)],
            self.nonbasic_vars,
        )
        lines.append(r'\max \quad & z = ' + obj)

        # Constraints
        for i in range(self.m):
            bv = self.basic_vars[i]
            rhs = self._build_sum_latex(
                self.C[i],
                [self.B[i][j] for j in range(self.n)],
                self.nonbasic_vars,
            )
            lines.append(f'& x_{{{bv}}} = {rhs}')

        lines.append(r'& x_i \geq 0 \quad \forall i')
        lines[1] = r'\text{s.t.}\quad ' + lines[1]  # Add "s.t." to the first constraint line.

        body = ' \\\\\n'.join(lines)
        return f'\\begin{{align*}}\n{body}\n\\end{{align*}}'

    def _to_latex_array(self) -> str:
        n = self.n
        total_cols = 2 + 2 * n
        col_spec = 'l r r' + ' c r' * n

        def build_row(lhs: str, constant: Fraction, coeffs: list[Fraction]) -> str:
            all_zero = all(c == 0 for c in coeffs)
            const_str = _frac_to_latex(constant) if (constant != 0 or all_zero) else ''
            cells = [lhs, const_str]
            for c, v in zip(coeffs, self.nonbasic_vars):
                if c == 0:
                    cells.extend(['', ''])
                else:
                    sign = '+' if c > 0 else '-'
                    abs_c = abs(c)
                    coeff_str = '' if abs_c == 1 else _frac_to_latex(abs_c)
                    cells.append(sign)
                    cells.append(f'{coeff_str}x_{{{v}}}')
            return ' & '.join(cells)

        rows: list[str] = []

        # Objective
        rows.append(build_row(
            r'\max \quad & z =',
            self.A[0],
            [self.A[k + 1] for k in range(n)],
        ))

        # Constraints
        for i in range(self.m):
            bv = self.basic_vars[i]
            prefix = r'\text{s.t.}\quad &' if i == 0 else '&'
            rows.append(build_row(
                prefix + f'x_{{{bv}}} =',
                self.C[i],
                [self.B[i][j] for j in range(n)],
            ))

        # Non-negativity spanning all columns
        rows.append(rf'&\multicolumn{{{total_cols-1}}}{{l}}{{x_i \geq 0 \quad \forall i}}')

        body = ' \\\\[4pt]\n'.join(rows)
        return (
            f'\\[\n'
            f'{{\\setlength{{\\arraycolsep}}{{2pt}}\n'
            f'\\begin{{array}}{{{col_spec}}}\n'
            f'{body}\n'
            f'\\end{{array}}}}\n'
            f'\\]'
        )

    def _to_latex_matrix(self) -> str:
        # Objective row vector a^T
        a_vals = [self.A[k + 1] for k in range(self.n)]
        a_row_latex = (
            '\\begin{pmatrix}\n'
            + ' & '.join(_frac_to_latex(v) for v in a_vals)
            + '\n\\end{pmatrix}'
        )

        # Column vectors for non-basic (x) and basic (y) variables
        x_vec = _col_vec_latex([f'x_{{{v}}}' for v in self.nonbasic_vars])
        y_vec = _col_vec_latex([f'x_{{{v}}}' for v in self.basic_vars])

        # B matrix and C column vector
        B_latex = _matrix_to_latex(self.B)
        C_vec = _col_vec_latex([_frac_to_latex(c) for c in self.C])

        # Objective constant
        const = _frac_to_latex(self.A[0])
        if self.A[0] == 0:
            obj_line = r'\max \quad & ' + a_row_latex + x_vec
        else:
            obj_line = r'\max \quad & ' + const + ' + ' + a_row_latex + x_vec

        lines = [
            obj_line,
            r'\text{s.t.} \quad & ' + y_vec + ' = ' + C_vec + ' + ' + B_latex + x_vec,
            r'& x_1, \ldots, x_{' + str(self.n + self.m) + r'} \geq 0',
        ]
        body = ' \\\\\n'.join(lines)
        return f'\\begin{{align*}}\n{body}\n\\end{{align*}}'


    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def to_float(self) -> dict:
        """
        Return a dict of numpy float64 arrays for the current system state.

        Returns:
            dict with keys 'B', 'C', 'A' (numpy arrays), 'basic_vars',
            'nonbasic_vars', and 'objective_value'.
        """
        return {
            'B': np.array(
                [[float(self.B[i][j]) for j in range(self.n)]
                 for i in range(self.m)]
            ),
            'C': np.array([float(c) for c in self.C]),
            'A': np.array([float(a) for a in self.A]),
            'basic_vars': list(self.basic_vars),
            'nonbasic_vars': list(self.nonbasic_vars),
            'objective_value': float(self.A[0]),
        }

    def __repr__(self) -> str:
        obj_val = float(self.A[0])
        basic_str = ', '.join(f'x{v}' for v in self.basic_vars)
        nonbasic_str = ', '.join(f'x{v}' for v in self.nonbasic_vars)
        return (
            f'LinearProgram(m={self.m}, n={self.n}, '
            f'obj={obj_val:.6g}, '
            f'basic=[{basic_str}], nonbasic=[{nonbasic_str}])'
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Classic 3-variable LP:
    #
    #   max  5*x1 + 4*x2 + 3*x3
    #   s.t. x4 = 5 - 6*x1 - 4*x2 - 2*x3
    #        x5 = 4 - 3*x1 - 2*x2 - 5*x3
    #        x1, x2, x3, x4, x5 >= 0
    #
    # In the slack form y = Bx + C: the coefficients in B are negative because
    # the original constraints subtract the x terms.

    B = [[-6, -4, -2],
         [-3, -2, -5]]
    C = [5, 4]
    A = [0, 5, 4, 3]   # A[0]=constant=0, A[1..3]=obj coefficients of x1,x2,x3

    lp = LinearProgram(B, C, A)
    print('=== Initial system ===')
    print(repr(lp))
    print()
    print(lp.to_latex())
    print()

    step = 0
    while True:
        pivot = lp.find_pivot(rule='bland')
        if pivot is None:
            print(f'Optimal after {step} pivot(s).')
            break
        entering, leaving = pivot
        print(f'Step {step + 1}: enter x{entering}, leave x{leaving}')
        lp.swap_variables(leaving, entering)
        print(repr(lp))
        step += 1

    print()
    print('=== Final system (full equations) ===')
    full_latex = lp.to_latex(matrix_form=False)
    print(full_latex)
    print()
    print('=== Final system (matrix form) ===')
    print(lp.to_latex(matrix_form=True))

    out = 'lp_optimal.png'
    print(f'\nGenerating {out} ...')
    import latex_rendering
    renderer = latex_rendering.LatexRenderer(font_color='white')
    # lp.generate_image(save_path=out)
    renderer.generate_image(full_latex, view=True, save_path=None)
    # print(f'Saved to {out}')


# To run:
# python3 simplex-algorithm/simplex.py 