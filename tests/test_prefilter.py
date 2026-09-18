from dayframe.prefilter.rules import prefilter
from tests.factories import make_asset


def test_drops_screenshots() -> None:
    kept = make_asset("keep", 10)
    shot = make_asset("shot", 10, minute=1, screenshot=True, uti="public.png")
    result = prefilter([kept, shot])
    assert [asset.uuid for asset in result.kept] == ["keep"]
    assert result.dropped[0].reason == "screenshot"


def test_drops_pdf_and_png_without_camera() -> None:
    pdf = make_asset("pdf", 10, uti="com.adobe.pdf", has_camera_exif=False)
    png = make_asset("png", 10, minute=1, uti="public.png", has_camera_exif=False)
    camera_png = make_asset("png-cam", 10, minute=2, uti="public.png", has_camera_exif=True)
    result = prefilter([pdf, png, camera_png])
    assert [asset.uuid for asset in result.kept] == ["png-cam"]
    assert {item.reason for item in result.dropped} == {"pdf", "png_no_camera"}


def test_drops_already_seen() -> None:
    asset = make_asset("seen", 10)
    result = prefilter([asset], seen={"seen"})
    assert result.kept == []
    assert result.dropped[0].reason == "already_seen"


def test_keeps_no_gps_and_singletons() -> None:
    lonely = make_asset("one", 12, latitude=None, longitude=None, place_name=None)
    result = prefilter([lonely])
    assert [asset.uuid for asset in result.kept] == ["one"]


def test_burst_keeps_highest_score() -> None:
    low = make_asset("b1", 18, burst=True, burst_id="burst-a", score_overall=0.2)
    high = make_asset("b2", 18, minute=1, burst=True, burst_id="burst-a", score_overall=0.9)
    mid = make_asset("b3", 18, minute=2, burst=True, burst_id="burst-a", score_overall=0.5)
    other = make_asset("c1", 19)
    result = prefilter([low, high, mid, other])
    assert [asset.uuid for asset in result.kept] == ["b2", "c1"]
    assert [item.reason for item in result.dropped] == ["burst_collapse", "burst_collapse"]
