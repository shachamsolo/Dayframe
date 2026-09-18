from dayframe.cluster.sessions import cluster_assets, haversine_km
from dayframe.config import ClusterConfig
from tests.factories import make_asset

HOME = (32.113, 34.801)
BEACH = (31.933, 34.706)


def test_time_gap_splits_after_90_minutes() -> None:
    a = make_asset("a", 10, 0, latitude=HOME[0], longitude=HOME[1])
    on_the_line = make_asset("b", 11, 30, latitude=HOME[0], longitude=HOME[1])
    after = make_asset("c", 11, 31, latitude=HOME[0], longitude=HOME[1])
    together = cluster_assets([a, on_the_line])
    assert len(together) == 1
    split = cluster_assets([a, after])
    assert [cluster.id for cluster in split] == ["2026-09-17-001", "2026-09-17-002"]
    assert [asset.uuid for asset in split[0].assets] == ["a"]
    assert [asset.uuid for asset in split[1].assets] == ["c"]


def test_geo_split_requires_distance_and_min_gap() -> None:
    first = make_asset("a", 16, 0, latitude=HOME[0], longitude=HOME[1], place_name="home")
    too_soon = make_asset(
        "b", 16, 15, latitude=BEACH[0], longitude=BEACH[1], place_name="Palmachim Beach"
    )
    later = make_asset(
        "c", 16, 25, latitude=BEACH[0], longitude=BEACH[1], place_name="Palmachim Beach"
    )
    close = cluster_assets([first, too_soon])
    assert len(close) == 1
    split = cluster_assets([first, later])
    assert [cluster.place for cluster in split] == ["home", "Palmachim Beach"]


def test_missing_gps_does_not_geo_split() -> None:
    first = make_asset("a", 16, 0, latitude=HOME[0], longitude=HOME[1])
    second = make_asset("b", 16, 30)
    clusters = cluster_assets([first, second])
    assert len(clusters) == 1


def test_short_move_does_not_split() -> None:
    first = make_asset("a", 9, 0, latitude=32.074, longitude=34.792)
    second = make_asset("b", 9, 30, latitude=32.075, longitude=34.793)
    assert haversine_km(first.coords or (0, 0), second.coords or (0, 0)) < 1.5
    assert len(cluster_assets([first, second])) == 1


def test_cluster_metadata() -> None:
    a = make_asset(
        "a",
        16,
        12,
        latitude=BEACH[0],
        longitude=BEACH[1],
        place_name="Palmachim Beach",
        persons=["Maya"],
    )
    b = make_asset(
        "b",
        16,
        40,
        latitude=31.934,
        longitude=34.705,
        place_name="Palmachim Beach",
        persons=["Maya", "Noa"],
    )
    c = make_asset(
        "c",
        17,
        10,
        latitude=BEACH[0],
        longitude=BEACH[1],
        place_name="parking lot",
    )
    [cluster] = cluster_assets([c, a, b])
    assert cluster.id == "2026-09-17-001"
    assert cluster.place == "Palmachim Beach"
    assert cluster.people == ["Maya", "Noa"]
    assert cluster.start == a.date
    assert cluster.end == c.date
    assert cluster.centroid is not None
    assert abs(cluster.centroid[0] - (BEACH[0] + 31.934 + BEACH[0]) / 3) < 1e-9


def test_custom_thresholds() -> None:
    cfg = ClusterConfig(time_gap_minutes=10, geo_gap_km=0.1, min_gap_for_geo_split_minutes=1)
    a = make_asset("a", 10, 0, latitude=HOME[0], longitude=HOME[1])
    b = make_asset("b", 10, 11, latitude=HOME[0], longitude=HOME[1])
    assert len(cluster_assets([a, b], cfg)) == 2
