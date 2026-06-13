"""Crop a diagram region out of a saved source image.

Claude returns each diagram's ``bbox`` as four fractions [x_min, y_min, x_max,
y_max] (0-1) on a given ``source_image_index``. This module turns that into an
actual cropped PNG, using the REAL pixel dimensions of the saved file.

No new dependencies beyond Pillow (already present via Streamlit).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from PIL import Image

# Fraction of the box size to add as breathing room on each side, so labels
# sitting right at the edge of Claude's box are not clipped.
_PAD_FRAC = 0.04


def crop_bbox_from_file(
    image_path: str | Path,
    bbox: list[float],
    *,
    pad_frac: float = _PAD_FRAC,
) -> Optional[bytes]:
    """Return PNG bytes of the cropped region, or None if it can't be done.

    ``bbox`` is [x_min, y_min, x_max, y_max] as fractions 0-1. Padding is added
    around the box and then clamped to the image edges, so a label near the
    border is not lost. Returns None (never raises) on any failure, so a bad
    crop degrades to "no crop" rather than breaking the UI.
    """
    try:
        path = Path(image_path)
        if not path.exists() or not bbox or len(bbox) != 4:
            return None

        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size

            x0, y0, x1, y1 = bbox
            # add padding (in fractional space) then clamp to [0, 1]
            pad_x = (x1 - x0) * pad_frac
            pad_y = (y1 - y0) * pad_frac
            x0 = max(0.0, x0 - pad_x)
            y0 = max(0.0, y0 - pad_y)
            x1 = min(1.0, x1 + pad_x)
            y1 = min(1.0, y1 + pad_y)

            # to pixels
            left, top = int(x0 * w), int(y0 * h)
            right, bottom = int(x1 * w), int(y1 * h)
            if right <= left or bottom <= top:
                return None

            crop = im.crop((left, top, right, bottom))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        return None


def resolve_image_for_diagram(
    saved_image_paths: list[str],
    source_image_index: Optional[int],
) -> Optional[str]:
    """Pick the saved file a diagram lives on.

    ``saved_image_paths`` is in the same order the images were sent to Claude,
    so ``source_image_index`` indexes straight into it. Falls back to the first
    image when the index is missing or out of range (better to show something
    than nothing — the reviewer can still judge it).
    """
    if not saved_image_paths:
        return None
    idx = source_image_index if source_image_index is not None else 0
    if 0 <= idx < len(saved_image_paths):
        return saved_image_paths[idx]
    return saved_image_paths[0]
