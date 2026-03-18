import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def persist_image_sample(
    *,
    base_dir: str,
    image_bytes: bytes,
    filename_hint: str,
    metadata: Dict[str, Any],
    subdir: str,
    image_ext: str = ".jpg",
) -> Dict[str, Any]:
    """
    Persist an uploaded image + metadata to disk for later labeling/training.

    Returns a dict with paths and ids. No-op responsibility (caller must ensure
    base_dir is configured and safe for their environment).
    """
    root = Path(base_dir).expanduser().resolve()
    out_dir = root / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    digest = _sha256(image_bytes)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    safe_hint = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in filename_hint)[:40]
    stem = f"{ts}_{safe_hint}_{digest[:12]}"

    img_path = out_dir / f"{stem}{image_ext}"
    meta_path = out_dir / f"{stem}.json"

    img_path.write_bytes(image_bytes)
    meta = {
        "id": digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filename_hint": filename_hint,
        "image_path": str(img_path),
        "metadata": metadata,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "id": digest,
        "image_path": str(img_path),
        "meta_path": str(meta_path),
    }

