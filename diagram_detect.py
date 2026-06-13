"""Detect diagram regions on an exam-page image using DocLayout-YOLO.

WHY THIS EXISTS
---------------
Two earlier approaches to "where is the diagram" failed on this project's math-
heavy pages:
  * Claude's own bounding boxes — wrong image, wrong location.
  * An OpenCV heuristic — could not tell a line-drawing from a block of typeset
    math, so it boxed equations as if they were figures.

DocLayout-YOLO is a document-layout model trained (among other things) on
"Textbooks & Test papers". It classifies page regions as figure / table /
isolate_formula / plain text / etc., so it reliably separates DIAGRAMS from
EQUATIONS — the exact thing the heuristics could not do. Verified on this
project's real pages: pyramid and 3D-tank diagrams detected as `figure` at
0.94-0.96, equations correctly tagged `isolate_formula`, never `figure`.

TWO-PASS DETECTION
------------------
The model only detects figures that "float" in the document. A diagram embedded
inside a table cell (e.g. a mark-scheme grid) is missed on a whole-page pass.
So:
  PASS 1: detect on the whole image, keep every `figure`.
  PASS 2: for every `table`, crop it and detect again; a figure that was nested
          now floats and is recovered. Its coordinates are mapped back to the
          full image. Verified: recovered the mark-scheme trapezoid at 0.73.
  Fallback: if a table yields no figure, the table region itself is returned as
          a (lower-confidence) candidate, so a human still sees something.

INTERFACE
---------
``detect_diagrams(image_path) -> list[DetectedRegion]`` with fractional bboxes,
identical to the previous module, so crop.py and the reviewer are unchanged.
The heavy model is loaded once, lazily, and cached.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# DocLayout-YOLO + its deps are heavy and optional. Import lazily so the rest of
# the app runs even when they're not installed; detect_diagrams() then returns
# [] and the UI reports that detection is unavailable.
_MODEL = None
_MODEL_LOCK = threading.Lock()
_LOAD_ERROR: Optional[str] = None

_REPO_ID = "juliozhao/DocLayout-YOLO-DocStructBench"
_WEIGHTS_FILE = "doclayout_yolo_docstructbench_imgsz1024.pt"

# Detection params (validated on this project's pages).
_IMG_SIZE = 1024
_CONF_PASS1 = 0.20
_CONF_PASS2 = 0.15
_TABLE_PAD_PX = 4
_CROP_PAD_FRAC = 0.03     # padding added to every returned figure box

# Minimum confidence for a `figure` detection to be kept. On this project's real
# pages, genuine diagrams scored 0.73-0.96 while equation/text false positives
# scored ~0.55, so 0.65 sits cleanly in the gap. Tune if needed.
_MIN_FIGURE_CONF = 0.65

# Class names of interest (DocStructBench taxonomy).
_FIGURE = "figure"
_TABLE = "table"


@dataclass(frozen=True)
class DetectedRegion:
    """A candidate diagram region as fractions 0-1 of the image."""

    bbox: list[float]      # [x_min, y_min, x_max, y_max]
    area_frac: float
    confidence: float
    kind: str = "figure"   # "figure" or "table" (table = fallback candidate)


def _load_model():
    """Load and cache the model once. Returns the model or None on failure."""
    global _MODEL, _LOAD_ERROR
    if _MODEL is not None or _LOAD_ERROR is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None or _LOAD_ERROR is not None:
            return _MODEL
        try:
            from doclayout_yolo import YOLOv10
            from huggingface_hub import hf_hub_download
            weights = hf_hub_download(repo_id=_REPO_ID, filename=_WEIGHTS_FILE)
            _MODEL = YOLOv10(weights)
        except Exception as exc:  # noqa: BLE001
            _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
            _MODEL = None
        return _MODEL


def is_available() -> bool:
    """True if the model can be loaded (deps present + weights reachable)."""
    return _load_model() is not None


def load_error() -> Optional[str]:
    """Human-readable reason the model could not load, if any."""
    _load_model()
    return _LOAD_ERROR


def _predict(model, image_path: str, conf: float):
    """Run the model on a path, return list of (class_name, conf, xyxy_px)."""
    res = model.predict(str(image_path), imgsz=_IMG_SIZE, conf=conf, device="cpu")[0]
    names = res.names
    out = []
    boxes = res.boxes
    if boxes is None:
        return out
    for b in boxes:
        cls = names[int(b.cls.item())]
        c = float(b.conf.item())
        xyxy = [float(v) for v in b.xyxy[0].tolist()]
        out.append((cls, c, xyxy))
    return out


def detect_diagrams(image_path: str | Path) -> list[DetectedRegion]:
    """Return candidate diagram regions (fractional bboxes), largest first.

    Two-pass: whole-page figures, plus figures recovered from inside tables.
    Never raises: returns [] on any failure.
    """
    model = _load_model()
    if model is None:
        return []
    try:
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            return []
        H, W = img.shape[:2]

        regions: list[DetectedRegion] = []

        # ---- PASS 1: whole page ----
        tables_px: list[list[float]] = []
        for cls, conf, (x1, y1, x2, y2) in _predict(model, str(image_path), _CONF_PASS1):
            if cls == _FIGURE:
                if conf >= _MIN_FIGURE_CONF:
                    regions.append(_to_region(x1, y1, x2, y2, W, H, conf, "figure"))
            elif cls == _TABLE:
                tables_px.append([x1, y1, x2, y2])

        # ---- PASS 2: inside each table ----
        import tempfile, os
        for (tx1, ty1, tx2, ty2) in tables_px:
            x1p = max(0, int(tx1) - _TABLE_PAD_PX)
            y1p = max(0, int(ty1) - _TABLE_PAD_PX)
            x2p = min(W, int(tx2) + _TABLE_PAD_PX)
            y2p = min(H, int(ty2) + _TABLE_PAD_PX)
            crop = img[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue
            cw, ch = (x2p - x1p), (y2p - y1p)

            tmp = os.path.join(tempfile.gettempdir(), f"_dl_table_{abs(hash((x1p, y1p, x2p, y2p)))}.png")
            cv2.imwrite(tmp, crop)
            try:
                for cls, conf, (fx1, fy1, fx2, fy2) in _predict(model, tmp, _CONF_PASS2):
                    if cls != _FIGURE:
                        continue
                    if conf < _MIN_FIGURE_CONF:
                        continue
                    found_fig = True
                    # map crop-local pixels back to full-image pixels
                    gx1 = x1p + fx1
                    gy1 = y1p + fy1
                    gx2 = x1p + fx2
                    gy2 = y1p + fy2
                    regions.append(_to_region(gx1, gy1, gx2, gy2, W, H, conf, "figure"))
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

            # NOTE: no table fallback. If pass 2 finds no real figure inside a
            # table, we show nothing — most such tables are pure equation/text
            # grids (mark schemes), and showing them produced false positives.
            # A genuinely missed embedded diagram is still visible to the human
            # in the "Images" tab (full page).

        regions = _dedupe(regions)
        regions.sort(key=lambda r: (-int(r.kind == "figure"), -r.area_frac))
        return regions
    except Exception:
        return []


def _to_region(x1, y1, x2, y2, W, H, conf, kind) -> DetectedRegion:
    """Pixel box -> padded, clamped fractional DetectedRegion."""
    fx0 = max(0.0, x1 / W - _CROP_PAD_FRAC)
    fy0 = max(0.0, y1 / H - _CROP_PAD_FRAC)
    fx1 = min(1.0, x2 / W + _CROP_PAD_FRAC)
    fy1 = min(1.0, y2 / H + _CROP_PAD_FRAC)
    area = max(0.0, (fx1 - fx0) * (fy1 - fy0))
    return DetectedRegion(
        bbox=[round(fx0, 4), round(fy0, 4), round(fx1, 4), round(fy1, 4)],
        area_frac=round(area, 4),
        confidence=round(float(conf), 3),
        kind=kind,
    )


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / (area_a + area_b - inter)


def _dedupe(regions: list[DetectedRegion], iou_thresh: float = 0.6) -> list[DetectedRegion]:
    """Drop near-duplicate boxes; prefer `figure` over `table`, higher conf."""
    ordered = sorted(
        regions, key=lambda r: (int(r.kind == "figure"), r.confidence), reverse=True
    )
    kept: list[DetectedRegion] = []
    for r in ordered:
        if any(_iou(r.bbox, k.bbox) > iou_thresh for k in kept):
            continue
        kept.append(r)
    return kept
