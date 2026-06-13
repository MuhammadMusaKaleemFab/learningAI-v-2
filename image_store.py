"""Local image persistence for the OCR pipeline.

WHY THIS EXISTS
---------------
Airtable attachment URLs expire after ~2 hours. The OCR pipeline downloads
every image to send to Claude, then discards the bytes. That means once the
URL dies, the original diagram is gone forever — and you cannot crop or display
a diagram you no longer have.

This module saves the bytes to local disk during the run you are already doing,
keyed by row, so you own a permanent copy. It is intentionally tiny and has no
dependencies beyond the standard library.

Swap ``LocalImageStore`` for an S3/GCS-backed version later without touching the
rest of the pipeline — the interface is just ``save()``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

# Map a media type to the file extension we write on disk.
_EXT_FOR_MEDIA_TYPE: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


@dataclass(frozen=True)
class SavedImage:
    """Record of one image written to disk."""

    path: Path          # absolute path to the saved file
    row_index: int      # which CSV row it belongs to
    url: str            # the (now possibly expired) source URL
    media_type: str     # image/png, image/jpeg, ...
    is_answer: bool     # True if this was a solution/answer image
    size_bytes: int     # how many bytes we wrote


class LocalImageStore:
    """Writes image bytes under ``root/row_<n>/<key>.<ext>``.

    The key is a short hash of the URL, so:
      * multiple images in one row never collide, and
      * the same image re-downloaded later maps to the same filename
        (idempotent — re-running won't create duplicates).
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, row_index: int, url: str, media_type: str) -> Path:
        url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
        ext = _EXT_FOR_MEDIA_TYPE.get(media_type, ".bin")
        row_dir = self.root / f"row_{row_index:04d}"
        row_dir.mkdir(parents=True, exist_ok=True)
        return row_dir / f"{url_hash}{ext}"

    def save(
        self,
        body: bytes,
        *,
        row_index: int,
        url: str,
        media_type: str,
        is_answer: bool = False,
    ) -> SavedImage:
        """Write ``body`` to disk and return a ``SavedImage`` record.

        Idempotent: if the target file already exists with the same size, we
        skip the write and just return the record. Re-running the batch will
        not duplicate or corrupt anything.
        """
        path = self._path_for(row_index, url, media_type)
        if not (path.exists() and path.stat().st_size == len(body)):
            path.write_bytes(body)
        return SavedImage(
            path=path,
            row_index=row_index,
            url=url,
            media_type=media_type,
            is_answer=is_answer,
            size_bytes=len(body),
        )
