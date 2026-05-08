"""Vision helpers shared by MedGemma preprocessing."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_rgb_image(path: str | Path) -> Image.Image:
    resolved = Path(path)
    img = Image.open(resolved).convert("RGB")
    return img


def infer_image_suffix(filename: str) -> str | None:
    lower = filename.lower()
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"):
        if lower.endswith(ext):
            return ext
    return None


def maybe_image_path(upload_name: str | None, temp_saved_path: str) -> tuple[str | None, bool]:
    """
    Filter non-image uploads; returns (path-or-None, is_image).
    """
    if upload_name is None:
        suffix = Path(temp_saved_path).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}:
            return None, False
        return temp_saved_path, True
    suf = infer_image_suffix(upload_name)
    if suf is None:
        return None, False
    return temp_saved_path, True
