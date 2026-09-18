from pathlib import Path

from dayframe.photos.reader import FixturePhotosReader
from dayframe.pipeline import run_for_date

FIXTURE = Path(__file__).parent / "fixtures" / "days" / "beach.json"


def test_beach_day_clusters_match_spoken_shape() -> None:
    result = run_for_date(FixturePhotosReader.from_path(FIXTURE), "2026-09-17")
    assert len(result.raw) == 18
    assert len(result.dropped) == 4
    assert [cluster.id for cluster in result.clusters] == [
        "2026-09-17-001",
        "2026-09-17-002",
        "2026-09-17-003",
        "2026-09-17-004",
        "2026-09-17-005",
        "2026-09-17-006",
    ]
    assert [cluster.place for cluster in result.clusters] == [
        "home",
        "Azrieli Center, Tel Aviv",
        None,
        "Palmachim Beach",
        None,
        "home",
    ]
    assert result.clusters[3].people == ["Maya", "Noa"]
    assert [len(cluster.assets) for cluster in result.clusters] == [3, 2, 1, 4, 2, 2]
