from io import BytesIO
from pathlib import Path

from PIL import Image

from dayframe.images.prepare import asset_image_path, prepare_image
from tests.factories import make_asset


def _write_jpeg(path: Path, size: tuple[int, int] = (2000, 800), color: str = "red") -> Path:
    image = Image.new("RGB", size, color)
    image.save(path, format="JPEG", quality=95)
    return path


def test_prepare_downscales_and_strips_exif(tmp_path: Path) -> None:
    source = tmp_path / "wide.jpg"
    image = Image.new("RGB", (2000, 800), "blue")
    exif = Image.Exif()
    exif[256] = 2000
    image.save(source, format="JPEG", quality=95, exif=exif)

    prepared = prepare_image(source, max_px=1024)
    assert prepared.media_type == "image/jpeg"
    assert max(prepared.width, prepared.height) == 1024
    assert prepared.width == 1024
    assert prepared.height == 410
    out = Image.open(BytesIO(prepared.jpeg_bytes))
    assert out.format == "JPEG"
    assert dict(out.getexif()) == {}


def test_prepare_does_not_upscale(tmp_path: Path) -> None:
    source = _write_jpeg(tmp_path / "small.jpg", size=(200, 100))
    prepared = prepare_image(source, max_px=1024)
    assert prepared.width == 200
    assert prepared.height == 100


def test_asset_image_path_prefers_edited(tmp_path: Path) -> None:
    original = _write_jpeg(tmp_path / "orig.jpg")
    edited = _write_jpeg(tmp_path / "edit.jpg", color="green")
    asset = make_asset("x", 12, path=str(original), path_edited=str(edited))
    assert asset_image_path(asset) == edited


def test_asset_image_path_missing() -> None:
    asset = make_asset("x", 12, path="/nope/missing.jpg")
    assert asset_image_path(asset) is None
