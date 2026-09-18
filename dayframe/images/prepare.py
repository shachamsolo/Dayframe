from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from dayframe.photos.models import Asset

JPEG_QUALITY = 80


@dataclass(frozen=True)
class PreparedImage:
    jpeg_bytes: bytes
    media_type: str
    width: int
    height: int
    source: Path


def asset_image_path(asset: Asset) -> Path | None:
    for raw in (asset.path_edited, asset.path):
        if not raw:
            continue
        path = Path(raw)
        if path.is_file():
            return path
    return None


def prepare_image(
    path: Path,
    *,
    max_px: int = 1024,
    quality: int = JPEG_QUALITY,
    strip_exif: bool = True,
) -> PreparedImage:
    """Downscale, re-encode JPEG, strip EXIF. Nothing leaves the machine unprepared."""
    image = _open(path)
    rgb = _to_rgb(image)
    if rgb is not image:
        image.close()
    scaled = _downscale(rgb, max_px)
    buffer = BytesIO()
    scaled.save(buffer, format="JPEG", quality=quality, optimize=True)
    if not strip_exif:
        # We never attach EXIF; the flag exists so config can document the guarantee.
        pass
    data = buffer.getvalue()
    return PreparedImage(
        jpeg_bytes=data,
        media_type="image/jpeg",
        width=scaled.width,
        height=scaled.height,
        source=path,
    )


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        return background
    if image.mode == "P":
        converted = image.convert("RGBA")
        return _to_rgb(converted)
    return image.convert("RGB")


def _downscale(image: Image.Image, max_px: int) -> Image.Image:
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= max_px:
        return image
    scale = max_px / long_edge
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _open(path: Path) -> Image.Image:
    try:
        image = Image.open(path)
        image.load()
        return image
    except (UnidentifiedImageError, OSError, ValueError):
        if sys.platform == "darwin":
            return _sips_open(path)
        raise


def _sips_open(path: Path) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        result = subprocess.run(
            ["sips", "-s", "format", "jpeg", str(path), "--out", tmp.name],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "sips failed").strip()
            raise OSError(f"cannot decode image {path}: {detail}")
        image = Image.open(tmp.name)
        image.load()
        return image.copy()
