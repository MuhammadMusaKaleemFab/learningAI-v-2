"""Attach detected diagram crops to a question's diagrams.

Ties together the pieces for tasks 1-4:
  * run the layout detector on each saved source image,
  * crop each detected `figure` region (tight, to avoid surrounding text),
  * save the crop as a PNG next to the source images,
  * write the crop's path + detector bbox into the matching diagram entry of
    the OCR JSON (so it's no longer null, and downstream / render can use it).

Matching strategy (data is "mostly one diagram per image"):
  * group detected crops and JSON diagrams by source image index,
  * if an image has the same number of detections as described diagrams ->
    pair them in order (exact, the common case),
  * otherwise attach all of that image's crops to that image's diagram(s) and
    leave extras as standalone diagram entries, so nothing is lost and the
    reviewer can sort it out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from .crop import crop_bbox_from_file
    from .diagram_detect import detect_diagrams, is_available as _detect_available
except ImportError:  # flat layout
    from crop import crop_bbox_from_file  # type: ignore[no-redef]
    from diagram_detect import detect_diagrams, is_available as _detect_available  # type: ignore[no-redef]


def attach_diagram_crops(
    question: dict,
    saved_image_paths: list[str],
    *,
    out_dir: Optional[str] = None,
    question_number: Optional[str] = None,
) -> dict:
    """Return a copy of ``question`` with diagram crops detected and attached.

    For every saved image, detect figure regions, crop+save them, and write
    ``image_path`` + ``detected_bbox`` into the matching diagram entry. Diagrams
    the detector finds but Claude did not describe are appended as new entries
    with an empty description (so the image is never lost).

    Never raises: on any failure the original question is returned unchanged.
    """
    if not question or not saved_image_paths or not _detect_available():
        return question
    try:
        q = dict(question)
        diagrams = [dict(d) for d in (q.get("diagrams") or [])]

        # group existing described diagrams by their source image index
        by_image: dict[int, list[dict]] = {}
        for d in diagrams:
            idx = d.get("source_image_index")
            idx = idx if isinstance(idx, int) and 0 <= idx < len(saved_image_paths) else 0
            by_image.setdefault(idx, []).append(d)

        base = Path(out_dir) if out_dir else Path(saved_image_paths[0]).parent
        base.mkdir(parents=True, exist_ok=True)
        qtag = (question_number or "q").replace("/", "_").replace(" ", "")

        result_diagrams: list[dict] = []
        saved_counter = 0

        for img_idx, img_path in enumerate(saved_image_paths):
            regions = [r for r in detect_diagrams(img_path) if r.kind == "figure"]
            described = by_image.get(img_idx, [])

            # crop + save every detected region for this image
            crops: list[tuple[str, list[float]]] = []
            for reg in regions:
                png = crop_bbox_from_file(img_path, reg.bbox, pad_frac=0.0)
                if not png:
                    continue
                fname = f"diagram_{qtag}_img{img_idx}_{saved_counter}.png"
                fpath = base / fname
                fpath.write_bytes(png)
                crops.append((str(fpath), reg.bbox))
                saved_counter += 1

            if described and crops and len(described) == len(crops):
                # exact 1:1 (the common case) — pair in order
                for d, (path, bbox) in zip(described, crops):
                    d["image_path"] = path
                    d["detected_bbox"] = bbox
                    result_diagrams.append(d)
            elif described and crops:
                # counts differ — attach first crop to each described diagram,
                # then append any leftover crops as standalone diagrams
                for i, d in enumerate(described):
                    if i < len(crops):
                        d["image_path"], d["detected_bbox"] = crops[i]
                    result_diagrams.append(d)
                for path, bbox in crops[len(described):]:
                    result_diagrams.append({
                        "location": described[0].get("location", "unknown"),
                        "kind": described[0].get("kind"),
                        "description": "",
                        "labels": [],
                        "source_image_index": img_idx,
                        "image_path": path,
                        "detected_bbox": bbox,
                    })
            elif described and not crops:
                # detector found nothing on this image — keep descriptions as-is
                result_diagrams.extend(described)
            elif crops and not described:
                # detector found figures Claude didn't describe — keep them
                for path, bbox in crops:
                    result_diagrams.append({
                        "location": "unknown",
                        "kind": None,
                        "description": "",
                        "labels": [],
                        "source_image_index": img_idx,
                        "image_path": path,
                        "detected_bbox": bbox,
                    })

        # carry over any diagrams whose source_image_index was out of range
        seen = {id(d) for d in result_diagrams}
        for d in diagrams:
            if id(d) not in seen and d not in result_diagrams:
                result_diagrams.append(d)

        q["diagrams"] = result_diagrams
        return q
    except Exception:
        return question
