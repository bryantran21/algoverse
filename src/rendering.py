"""Render a text QA item to a PNG using Typst.

Design choice: a FIXED page box (config.RENDER page_width/height + ppi). Changing
font size changes how much text fits, i.e. the visual "compression", but the
output image dimensions stay constant. That keeps aspect ratio and pixel budget
fixed across the sweep so accuracy-vs-compression isn't confounded by resolution.

Rendering backend: prefer the `typst` Python binding (`pip install typst`), which
exposes `typst.compile(...)` and returns PNG bytes directly — this is what's
available on Kaggle/Colab. The `typst` CLI is used only as a fallback if the
Python package isn't importable but a binary happens to be on PATH.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from config import RENDER


class TypstNotFound(RuntimeError):
    pass


def _escape(text: str) -> str:
    """Escape Typst markup so passage text renders literally."""
    if text is None:
        return ""
    out = []
    for ch in str(text):
        if ch in "\\#$*_`<>@=+-/[]":
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def _build_source(
    passage: str,
    question: str,
    font: str,
    font_size: float,
) -> str:
    r = RENDER
    pw, ph, mg = r["page_width_mm"], r["page_height_mm"], r["margin_mm"]
    return f"""\
#set page(width: {pw}mm, height: {ph}mm, margin: {mg}mm, fill: white)
#set text(font: "{font}", size: {font_size}pt, fill: black)
#set par(justify: false, leading: 0.65em)

*Passage.* {_escape(passage)}

*Question.* {_escape(question)}
"""


def render_qa_to_image(
    passage: str,
    question: str,
    font: str | None = None,
    font_size: float | None = None,
    ppi: int | None = None,
) -> bytes:
    """Render one QA item to PNG bytes.

    font / font_size / ppi default to config.RENDER. Image dimensions are a
    function of page size + ppi only, so they stay constant across font sizes.
    """
    font = font or RENDER["font"]
    font_size = font_size if font_size is not None else RENDER["font_size_pt"]
    ppi = ppi or RENDER["ppi"]

    src = _build_source(passage, question, font, font_size)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src_path = td / "doc.typ"
        src_path.write_text(src, encoding="utf-8")
        return _compile_png(src_path, ppi, td)


def _compile_png(src_path: Path, ppi: int, td: Path) -> bytes:
    """Compile a .typ file to PNG bytes. Python binding first, CLI fallback."""
    # 1) Python binding (the Kaggle/Colab path).
    try:
        import typst  # typst-py
    except ImportError:
        typst = None

    if typst is not None:
        # Single-page doc -> compile returns PNG bytes (a list only if multipage).
        out = typst.compile(str(src_path), format="png", ppi=float(ppi))
        return out[0] if isinstance(out, (list, tuple)) else out

    # 2) CLI fallback (only if a `typst` binary is on PATH).
    exe = shutil.which("typst")
    if exe is None:
        raise TypstNotFound(
            "No Typst backend found. Install the Python binding: `pip install typst` "
            "(exposes typst.compile), or put the typst CLI binary on PATH "
            "(https://github.com/typst/typst/releases)."
        )
    out_path = td / "doc.png"
    subprocess.run(
        [exe, "compile", "--format", "png", "--ppi", str(ppi),
         str(src_path), str(out_path)],
        check=True,
        capture_output=True,
    )
    return out_path.read_bytes()


def is_blank_image(png_bytes: bytes, ink_threshold: float = 0.001) -> bool:
    """True if the PNG has essentially no dark ink (blank/near-white page).

    Guards against a silent Typst font fallback that renders nothing. `ink` is the
    fraction of pixels darker than mid-gray; a page with text is well above 0.1%.
    """
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(png_bytes)).convert("L")
    import numpy as np

    arr = np.asarray(img)
    ink = float((arr < 128).mean())
    return ink < ink_threshold


def image_dims(ppi: int | None = None) -> tuple[int, int]:
    """Expected (width_px, height_px) for the fixed page box at the given ppi.

    Useful for a sanity check / logging the constant pixel budget.
    """
    ppi = ppi or RENDER["ppi"]
    mm_to_in = 1.0 / 25.4
    w = round(RENDER["page_width_mm"] * mm_to_in * ppi)
    h = round(RENDER["page_height_mm"] * mm_to_in * ppi)
    return w, h
