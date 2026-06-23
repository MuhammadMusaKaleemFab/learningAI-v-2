"""Diagnose diagram blur: measure the resolution of saved source images and
the diagram crops, so we know whether blur is source-limited or fixable.

Run from your project folder (with venv active):
    python check_resolution.py

It scans saved_images/ and reports each image's dimensions, plus a verdict.
"""

from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow not installed. Run: pip install Pillow")

ROOT = Path("saved_images")

# Rough guidance for diagram readability.
LOW_WIDTH = 700     # below this, fine text/labels often look soft
GOOD_WIDTH = 1200   # comfortable for crisp labels


def verdict(w: int, h: int) -> str:
    long_side = max(w, h)
    if long_side < LOW_WIDTH:
        return "LOW  -> source-limited; crop will look soft no matter what"
    if long_side < GOOD_WIDTH:
        return "OK   -> usable; display scaling matters most"
    return "GOOD -> plenty of detail; any blur is from display, not source"


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"No '{ROOT}' folder here. Run from your project root.")

    imgs = sorted(
        p for p in ROOT.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not imgs:
        raise SystemExit(f"No images found under {ROOT}/")

    print(f"{'file':45} {'WxH':>13}  verdict")
    print("-" * 100)
    widths = []
    for p in imgs:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception as e:
            print(f"{p.name[:45]:45} {'?':>13}  (unreadable: {e})")
            continue
        widths.append(max(w, h))
        tag = "[crop]" if p.name.startswith("diagram_") else ""
        print(f"{p.name[:45]:45} {f'{w}x{h}':>13}  {verdict(w, h)} {tag}")

    if widths:
        widths.sort()
        mid = widths[len(widths) // 2]
        print("-" * 100)
        print(f"median long-side: {mid}px")
        if mid < LOW_WIDTH:
            print(">> Most sources are LOW-RES. The blur is largely in the source "
                  "images themselves; re-capturing diagrams at higher quality "
                  "(original PDFs, not WhatsApp/compressed screenshots) is the only "
                  "real fix for those.")
        else:
            print(">> Sources have enough detail. The blur is mostly a DISPLAY "
                  "scaling issue we can fix in the app (show crops at natural size).")


if __name__ == "__main__":
    main()
