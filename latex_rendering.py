import os
import re
import sys
import subprocess
import tempfile
import warnings
import matplotlib.pyplot as plt
from PIL import Image


def _show_image(img: Image.Image) -> None:
    """
    Display a PIL Image inside the current Python environment.

    Priority:
      1. IPython.display  — works in Jupyter notebooks and VS Code interactive window.
      2. VS Code CLI (code) — opens the image as a tab in the editor.
      3. xdg-open / open   — system image viewer (requires a display).
      4. Print the temp file path as a last resort.
    """
    try:
        from IPython.display import display as ipy_display
        ipy_display(img)
        return
    except ImportError:
        pass

    # Write to a temp file so external viewers can open it.
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp.close()
    img.save(tmp.name)

    for cmd in [['code', tmp.name], ['xdg-open', tmp.name], ['open', tmp.name]]:
        try:
            subprocess.Popen(cmd)
            return
        except FileNotFoundError:
            continue

    print(f"Image written to: {tmp.name}")


class LatexRenderer:
    def __init__(self, font_size: int = 13, font_color: str = 'black'):
        self.font_size = font_size
        self.font_color = font_color

    def generate_image(
        self,
        latex_str: str,
        save_path: 'str | None' = None,
        view: bool = False,
    ) -> Image.Image:
        """
        Render a LaTeX string to a PIL Image.

        Attempts pdflatex + ImageMagick convert first; falls back to
        matplotlib mathtext if unavailable.

        Args:
            latex_str: LaTeX string to render.
            save_path: path at which to save the PNG (optional).
            view: if True, display the image inside the current Python
                  environment (IPython/Jupyter or matplotlib).
            font_size: font size in points (used by both rendering paths).
            font_color: font color as a CSS/matplotlib color string, e.g.
                        'black', 'red', '#2a9d8f'.  Named colors are also
                        accepted by LaTeX's xcolor package.

        Returns:
            PIL Image object.
        """
        img = self._try_pdflatex(latex_str, font_size=self.font_size, font_color=self.font_color)
        if img is None:
            img = self._render_matplotlib(latex_str, font_size=self.font_size, font_color=self.font_color)

        if save_path is not None:
            img.save(save_path)

        if view:
            _show_image(img)

        return img

    def _try_pdflatex(
        self,
        latex_str: str,
        font_size: int = 13,
        font_color: str = 'black',
    ) -> 'Image.Image | None':
        """Compile latex_str with pdflatex and convert the PDF to PNG via ImageMagick."""
        # font_size -> LaTeX \fontsize{size}{baselineskip}\selectfont
        baseline = round(font_size * 1.2)
        size_cmd = rf'\fontsize{{{font_size}}}{{{baseline}}}\selectfont'

        doc = (
            '\\documentclass{article}\n'
            '\\usepackage{amsmath}\n'
            '\\usepackage{xcolor}\n'
            '\\pagestyle{empty}\n'
            '\\begin{document}\n'
            + rf'{{\color{{{font_color}}}{size_cmd}'
            + '\n'
            + latex_str
            + '\n}\n'
            '\\end{document}\n'
        )
        try:
            with tempfile.TemporaryDirectory() as td:
                tex_path = os.path.join(td, 'lp.tex')
                pdf_path = os.path.join(td, 'lp.pdf')
                png_path = os.path.join(td, 'lp.png')

                with open(tex_path, 'w') as fh:
                    fh.write(doc)

                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode',
                     '-output-directory', td, tex_path],
                    capture_output=True, timeout=30,
                )
                if result.returncode != 0 or not os.path.exists(pdf_path):
                    return None

                result = subprocess.run(
                    ['convert', '-density', '150', '-trim', '+repage',
                     pdf_path, png_path],
                    capture_output=True, timeout=30,
                )
                if result.returncode != 0 or not os.path.exists(png_path):
                    return None

                return Image.open(png_path).copy()

        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return None

    def _render_matplotlib(
        self,
        latex_str: str,
        font_size: int = 13,
        font_color: str = 'black',
    ) -> Image.Image:
        """Render latex_str using matplotlib mathtext (no system LaTeX required)."""
        text = re.sub(r'\\begin\{align\*\}', '', latex_str)
        text = re.sub(r'\\end\{align\*\}', '', text)
        text = re.sub(r'\\\[', '', text)
        text = re.sub(r'\\\]', '', text)
        text = re.sub(r'\\begin\{array\}\{[^}]*\}', '', text)
        text = re.sub(r'\\end\{array\}', '', text)
        text = re.sub(r'\\multicolumn\{\d+\}\{[^}]*\}\{([^}]*)\}', r'\1', text)
        text = re.sub(r'&', '', text)
        raw_lines = [
            ln.strip().rstrip('\\').strip()
            for ln in text.split('\\\\')
        ]
        lines = [ln for ln in raw_lines if ln]

        display = [f'${ln}$' for ln in lines]

        fig_h = 0.6 + 0.55 * len(display)
        fig, ax = plt.subplots(figsize=(9, fig_h))
        ax.axis('off')

        for idx, line in enumerate(display):
            y_pos = 1.0 - (idx + 0.5) / len(display)
            ax.text(
                0.03, y_pos, line,
                ha='left', va='center',
                fontsize=font_size,
                color=font_color,
                transform=ax.transAxes,
            )

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as fh:
            tmp_path = fh.name

        fig.savefig(tmp_path, dpi=150, bbox_inches='tight', facecolor='none')
        plt.close(fig)

        img = Image.open(tmp_path).copy()
        os.unlink(tmp_path)
        return img
