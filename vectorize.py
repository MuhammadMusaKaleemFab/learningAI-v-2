"""Vectorize a diagram crop to make its LINES sharp at any resolution.

WHAT THIS DOES (and doesn't)
---------------------------
Exam diagrams are low-resolution screenshots (~median 560px). Lines and shapes
can be traced into vector paths that are crisp at any zoom. We tested this on
real crops:
  * line art (geometry, number lines, force diagrams) -> genuinely sharp.
  * text labels -> NOT recovered. Vectorization cannot sharpen blurry small
    text; it can only preserve a soft approximation (color mode) or shatter it
    (binary mode). So this is a win for line-heavy diagrams, neutral-to-worse
    for text-heavy ones.

Therefore this module:
  * uses COLOR mode (preserves text rather than shattering it),
  * renders the SVG back to a high-res PNG (default 3x) for display,
  * NEVER discards the original crop — callers keep both and let a human choose.

Optional dependencies: ``vtracer`` and ``cairosvg``. If either is missing,
``is_available()`` returns False and callers fall back to the original crop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import vtracer
    import cairosvg
    from PIL import Image
    _VEC_AVAILABLE = True
    _VEC_ERROR: Optional[str] = None
except Exception as _e:  # pragma: no cover
    _VEC_AVAILABLE = False
    _VEC_ERROR = f"{type(_e).__name__}: {_e}"


# Tuning (validated on this project's crops in color mode).
_SCALE = 3              # render the vector at this multiple of source size
_COLOR_PRECISION = 6
_FILTER_SPECKLE = 2     # drop tiny noise specks
_CORNER_THRESHOLD = 60
_PATH_PRECISION = 3
_MAX_RENDER_PX = 4000   # cap output dimension so huge crops don't blow up


def is_available() -> bool:
    """True if vectorization deps are importable."""
    return _VEC_AVAILABLE


def load_error() -> Optional[str]:
    return _VEC_ERROR


def vectorize_to_png(
    image_path: str | Path,
    *,
    scale: int = _SCALE,
    out_path: Optional[str | Path] = None,
) -> Optional[str]:
    """Vectorize ``image_path`` and render a sharp PNG.

    Returns the path to the rendered PNG (``<name>_vector.png`` next to the
    source by default), or None on any failure. Never raises — callers fall
    back to the original crop if this returns None.
    """
    if not _VEC_AVAILABLE:
        return None
    try:
        src = Path(image_path)
        if not src.exists():
            return None

        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            # vtracer is picky about input encoding; re-save a clean PNG first.
            import tempfile, os as _os
            tmp_in = _os.path.join(
                tempfile.gettempdir(), f"_vec_in_{abs(hash(str(src)))}.png"
            )
            im.save(tmp_in)

        # cap render size
        target_w, target_h = w * scale, h * scale
        longest = max(target_w, target_h)
        if longest > _MAX_RENDER_PX:
            factor = _MAX_RENDER_PX / longest
            target_w = int(target_w * factor)
            target_h = int(target_h * factor)

        svg_path = src.with_suffix(".svg")
        vtracer.convert_image_to_svg_py(
            tmp_in,
            str(svg_path),
            colormode="color",          # preserves text rather than shattering it
            color_precision=_COLOR_PRECISION,
            filter_speckle=_FILTER_SPECKLE,
            corner_threshold=_CORNER_THRESHOLD,
            path_precision=_PATH_PRECISION,
        )
        try:
            _os.remove(tmp_in)
        except OSError:
            pass

        png_path = Path(out_path) if out_path else src.with_name(src.stem + "_vector.png")
        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(png_path),
            output_width=target_w,
            output_height=target_h,
        )

        # tidy up the intermediate SVG
        try:
            svg_path.unlink()
        except OSError:
            pass

        return str(png_path)
    except Exception:
        return None
