"""
STANDALONE TEST: does DocLayout-YOLO correctly find the diagrams (and NOT the
equations) on your exam pages?

This does NOT touch your project. It just downloads the model once, runs it on
the images you point it at, prints what it detected, and saves annotated copies
so you can see the boxes with your own eyes.

------------------------------------------------------------------------------
HOW TO RUN (Windows PowerShell, inside your venv):

    pip install doclayout-yolo huggingface_hub
    python test_doclayout.py

By default it scans a folder called "test_images" next to this script. Put a few
of your exam page images in there first (copy them out of saved_images/), OR
edit IMAGE_PATHS below to point at specific files.
------------------------------------------------------------------------------

What to look for in the output:
  * Each detected region prints as:  <class_name>  conf=0.xx  box=[...]
  * The classes come from DocStructBench. The ones that matter:
        - "figure"          <- this is what we WANT for diagrams
        - "isolate_formula" <- standalone equations (should NOT be "figure")
        - "plain text", "title", "table", ...
  * SUCCESS = the trapezoid / pyramid / tank get a "figure" box, and the big
    equation blocks get "isolate_formula" (or text), NOT "figure".

Then look at the saved  *_annotated.jpg  files to confirm visually.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — edit if you want to point at specific files instead of a folder.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
IMAGE_DIR = HERE / "test_images"          # folder to scan for images
IMAGE_PATHS: list[str] = []               # OR list explicit paths here
OUTPUT_DIR = HERE / "doclayout_output"    # annotated images go here
CONF_THRESHOLD = 0.20                     # detection score cutoff
IMG_SIZE = 1024
# ---------------------------------------------------------------------------


def _gather_images() -> list[Path]:
    if IMAGE_PATHS:
        return [Path(p) for p in IMAGE_PATHS]
    if not IMAGE_DIR.exists():
        print(f"[!] No image list set and folder not found: {IMAGE_DIR}")
        print(f"    Create it and drop a few exam page images inside, then re-run.")
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in exts)


def main() -> None:
    try:
        import cv2  # noqa: F401
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print("[!] Missing dependency:", e)
        print("    Run:  pip install doclayout-yolo huggingface_hub opencv-python")
        return

    images = _gather_images()
    if not images:
        return

    print("[*] Downloading model weights (first run only, ~hundreds of MB)...")
    weights = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        filename="doclayout_yolo_docstructbench_imgsz1024.pt",
    )
    print(f"[*] Model: {weights}")
    model = YOLOv10(weights)

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"[*] Running on {len(images)} image(s). conf>={CONF_THRESHOLD}\n")

    import cv2

    for img_path in images:
        print("=" * 70)
        print(f"IMAGE: {img_path.name}")
        det = model.predict(
            str(img_path),
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            device="cpu",   # change to "0" or "cuda:0" if you have a GPU
        )
        res = det[0]
        names = res.names  # class id -> class name

        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            print("   (nothing detected)")
        else:
            # print each detection
            for b in boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                xyxy = [round(float(v), 1) for v in b.xyxy[0].tolist()]
                flag = "  <-- DIAGRAM" if names[cls_id] == "figure" else ""
                print(f"   {names[cls_id]:18} conf={conf:.2f}  box={xyxy}{flag}")

        # save annotated image
        annotated = res.plot(pil=True, line_width=4, font_size=18)
        out = OUTPUT_DIR / f"{img_path.stem}_annotated.jpg"
        cv2.imwrite(str(out), annotated)
        print(f"   -> saved {out.name}")

    print("=" * 70)
    print(f"\nDone. Open the annotated images in: {OUTPUT_DIR}")
    print("Check: did the actual drawings get a 'figure' box, and did the")
    print("equation blocks get 'isolate_formula'/text instead of 'figure'?")


if __name__ == "__main__":
    main()
