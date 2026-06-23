"""
STANDALONE TEST: can Gemini "redraw" a low-res diagram crop into a sharp,
high-resolution version WITHOUT changing any numbers, labels, or geometry?

This does NOT touch your project. It sends a few of your real diagram crops to
the Gemini image API with a faithful-redraw instruction, saves the results, and
prints what to check. Run it on a handful of crops first so you and the client
can judge quality + correctness BEFORE paying for a full batch.

------------------------------------------------------------------------------
SETUP (Windows PowerShell, in your venv):

    pip install google-genai pillow
    # put the client's key in your .env  ->  GEMINI_API_KEY=xxxx
    # then:
    python test_gemini_redraw.py

By default it reads images from a folder "redraw_in" next to this script.
Put 4-5 real diagram crops in there (copy from saved_images/, include both a
line-heavy one and a text-heavy one). Results go to "redraw_out".
------------------------------------------------------------------------------

WHAT TO CHECK in redraw_out/:
  * Is the redraw genuinely sharper / more legible than the original?
  * CRITICAL: are ALL numbers, angles, labels, and geometry EXACTLY the same?
    (A generative model can silently change a 6 to an 8, move a label, or
    redraw a chemistry bond wrong. For exam content, any change = reject.)
  * Compare <name>_original.png vs <name>_redraw.png side by side.
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
IN_DIR = HERE / "redraw_in"
OUT_DIR = HERE / "redraw_out"

# Model choice:
#   "gemini-3-pro-image-preview"      -> best text fidelity, ~$0.13/image
#   "gemini-3.1-flash-image"          -> cheaper/faster, ~$0.04/image
MODEL = "gemini-3-pro-image-preview"

# The instruction. Deliberately conservative: redraw faithfully, change nothing.
REDRAW_PROMPT = (
    "Redraw this exam diagram as a clean, high-resolution, sharp black-and-white "
    "line drawing. Reproduce it EXACTLY: keep every label, letter, number, angle, "
    "symbol, axis, and geometric relationship identical to the original. Do NOT "
    "add, remove, relabel, or reposition anything. Do NOT change any numbers or "
    "values. Only improve clarity and sharpness. Preserve the exact layout. "
    "If any text is unreadable in the original, leave it as-is rather than guessing."
)


def _load_env():
    """Load GEMINI_API_KEY from .env if present (no external dep)."""
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    _load_env()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY in your .env or environment.")

    try:
        from google import genai
        from google.genai import types
        from PIL import Image
    except ImportError as e:
        raise SystemExit(f"Missing dep: {e}\n  pip install google-genai pillow")

    if not IN_DIR.exists():
        raise SystemExit(
            f"Create a '{IN_DIR.name}' folder and put a few diagram crops in it."
        )
    imgs = sorted(
        p for p in IN_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not imgs:
        raise SystemExit(f"No images in {IN_DIR}/")

    OUT_DIR.mkdir(exist_ok=True)
    client = genai.Client(api_key=key)

    print(f"Model: {MODEL}\nProcessing {len(imgs)} image(s)...\n")
    for p in imgs:
        print(f"--- {p.name}")
        img_bytes = p.read_bytes()
        # keep a copy of the original for side-by-side comparison
        Image.open(p).convert("RGB").save(OUT_DIR / f"{p.stem}_original.png")

        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=[
                    REDRAW_PROMPT,
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                ),
            )
        except Exception as e:
            print(f"    API error: {e}")
            continue

        # Gemini can return an empty/blocked response (no parts) — e.g. its
        # safety filter declined to redraw a sensitive symbol. Handle that
        # gracefully instead of crashing.
        cand = (resp.candidates or [None])[0]
        finish = getattr(cand, "finish_reason", None)
        parts = getattr(getattr(cand, "content", None), "parts", None)
        if not parts:
            reason = finish or "no content returned"
            print(f"    !! no image returned (reason: {reason}) — likely blocked/declined; skipping")
            continue

        saved = False
        for part in parts:
            if getattr(part, "inline_data", None):
                from io import BytesIO
                out = OUT_DIR / f"{p.stem}_redraw.png"
                Image.open(BytesIO(part.inline_data.data)).save(out)
                print(f"    -> saved {out.name}")
                saved = True
            elif getattr(part, "text", None):
                print(f"    (model note: {part.text[:100]})")
        if not saved:
            print("    !! no image returned")

    print(f"\nDone. Compare *_original.png vs *_redraw.png in {OUT_DIR}/")
    print("CHECK CAREFULLY: every number/label/angle must be unchanged.")


if __name__ == "__main__":
    main()
