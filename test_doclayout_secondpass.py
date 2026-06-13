"""
STANDALONE TEST #2: can a SECOND PASS recover a diagram embedded inside a table?

Image 2 (the mark-scheme page) failed because the trapezoid is INSIDE a table
cell, and the model only detects "isolate figure FLOATING in the document". This
script tests the fix: for every `table` region the model finds, crop it and run
the model AGAIN on just that crop — where the trapezoid is now the dominant
floating element and should get its own `figure` box.

Does NOT touch your project.

------------------------------------------------------------------------------
HOW TO RUN (Windows PowerShell, inside your venv):

    pip install doclayout-yolo huggingface_hub opencv-python
    python test_doclayout_secondpass.py

Put your images in a "test_images" folder next to this script (especially the
mark-scheme page), OR edit IMAGE_PATHS below.
------------------------------------------------------------------------------

What to look for:
  * For each image it prints PASS 1 detections (whole page).
  * For every `table` found, it prints PASS 2 detections (inside that table)
    and saves an annotated crop:  <name>_table<N>_secondpass.jpg
  * SUCCESS for image 2 = PASS 2 shows a `figure` box around the trapezoid.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE_DIR = HERE / "test_images"
IMAGE_PATHS: list[str] = []
OUTPUT_DIR = HERE / "doclayout_output_pass2"
CONF_PASS1 = 0.20
CONF_PASS2 = 0.15          # slightly lower inside the table — less competing content
IMG_SIZE = 1024
TABLE_PAD = 4              # px padding when cropping a table region
# ---------------------------------------------------------------------------


def _gather_images() -> list[Path]:
    if IMAGE_PATHS:
        return [Path(p) for p in IMAGE_PATHS]
    if not IMAGE_DIR.exists():
        print(f"[!] Folder not found: {IMAGE_DIR} — create it and add images.")
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in exts)


def _detections(res):
    """Yield (class_name, conf, [x1,y1,x2,y2]) for one result object."""
    names = res.names
    boxes = res.boxes
    if boxes is None:
        return
    for b in boxes:
        cls = names[int(b.cls.item())]
        conf = float(b.conf.item())
        xyxy = [int(round(float(v))) for v in b.xyxy[0].tolist()]
        yield cls, conf, xyxy


def main() -> None:
    try:
        import cv2
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print("[!] Missing dependency:", e)
        print("    pip install doclayout-yolo huggingface_hub opencv-python")
        return

    images = _gather_images()
    if not images:
        return

    print("[*] Downloading / loading model...")
    weights = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        filename="doclayout_yolo_docstructbench_imgsz1024.pt",
    )
    model = YOLOv10(weights)
    OUTPUT_DIR.mkdir(exist_ok=True)

    for img_path in images:
        print("\n" + "=" * 70)
        print(f"IMAGE: {img_path.name}")
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]

        # ---- PASS 1: whole page ----
        res1 = model.predict(str(img_path), imgsz=IMG_SIZE, conf=CONF_PASS1, device="cpu")[0]
        print("  PASS 1 (whole page):")
        tables = []
        found_fig_pass1 = False
        for cls, conf, box in _detections(res1):
            mark = ""
            if cls == "figure":
                mark = "  <-- DIAGRAM (pass 1)"
                found_fig_pass1 = True
            if cls == "table":
                tables.append(box)
                mark = "  [will second-pass this]"
            print(f"     {cls:18} conf={conf:.2f} {box}{mark}")

        # ---- PASS 2: inside each table ----
        for ti, (x1, y1, x2, y2) in enumerate(tables):
            x1p = max(0, x1 - TABLE_PAD); y1p = max(0, y1 - TABLE_PAD)
            x2p = min(W, x2 + TABLE_PAD); y2p = min(H, y2 + TABLE_PAD)
            crop = img[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue
            crop_file = OUTPUT_DIR / f"{img_path.stem}_table{ti}_crop.png"
            cv2.imwrite(str(crop_file), crop)

            res2 = model.predict(str(crop_file), imgsz=IMG_SIZE, conf=CONF_PASS2, device="cpu")[0]
            print(f"  PASS 2 (inside table #{ti}, region={[x1,y1,x2,y2]}):")
            fig_in_table = False
            for cls, conf, box in _detections(res2):
                mark = ""
                if cls == "figure":
                    mark = "  <-- DIAGRAM RECOVERED!"
                    fig_in_table = True
                print(f"     {cls:18} conf={conf:.2f} {box}{mark}")
            if not fig_in_table:
                print("     (no figure recovered inside this table)")

            annotated = res2.plot(pil=True, line_width=4, font_size=18)
            out = OUTPUT_DIR / f"{img_path.stem}_table{ti}_secondpass.jpg"
            cv2.imwrite(str(out), annotated)
            print(f"     -> saved {out.name}")

        if not tables and not found_fig_pass1:
            print("  (no tables and no figures — nothing to recover)")

    print("\n" + "=" * 70)
    print(f"Done. Inspect annotated crops in: {OUTPUT_DIR}")
    print("For the mark-scheme page: did PASS 2 recover a 'figure' (the trapezoid)?")


if __name__ == "__main__":
    main()
