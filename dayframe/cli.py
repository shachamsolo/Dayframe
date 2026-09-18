from __future__ import annotations

from collections import Counter
from datetime import datetime

import typer

from dayframe import __version__
from dayframe.baseline.anthropic import BaselineProviderError
from dayframe.baseline.digest import format_digest_line, people_label
from dayframe.baseline.models import BaselineResult
from dayframe.baseline.run import require_anthropic, run_baseline
from dayframe.doctor import failed, format_report, run_checks
from dayframe.paths import api_key, load_env
from dayframe.photos.models import Asset
from dayframe.photos.reader import get_reader
from dayframe.pipeline import PipelineResult, load_config_optional, run_for_date
from dayframe.store.db import (
    connect,
    fetch_clusters,
    fetch_run,
    init_db,
    persist_baseline_run,
    replace_inspect_run,
    seen_uuids,
)
from dayframe.window import calendar_day_window, parse_since, parse_target_date, since_window

app = typer.Typer(
    help="AI memories from your photos.",
    no_args_is_help=True,
    add_completion=False,
)
photos_app = typer.Typer(help="Inspect what the Photos reader sees.")
clusters_app = typer.Typer(help="Inspect clustered sessions.")
eval_app = typer.Typer(help="Score runs against labelled fixtures.")

app.add_typer(photos_app, name="photos")
app.add_typer(clusters_app, name="clusters")
app.add_typer(eval_app, name="eval")


def _not_yet(milestone: str) -> None:
    typer.echo(f"Not implemented until {milestone}.", err=True)
    raise typer.Exit(code=1)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """AI memories from your photos."""
    load_env()


@app.command()
def doctor() -> None:
    """Check permissions, library, API key, and calendar."""
    results = run_checks()
    typer.echo(format_report(results))
    if failed(results):
        raise typer.Exit(code=1)


@app.command()
def auth() -> None:
    """One-time Google OAuth consent flow."""
    _not_yet("M4")


@app.command()
def run(
    date: str = typer.Option("yesterday", "--date", help="Target date (yesterday or YYYY-MM-DD)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print memories; write nothing."),
    no_calendar: bool = typer.Option(
        False, "--no-calendar", help="Run everything except the write."
    ),
) -> None:
    """Run Dayframe for a date."""
    key = api_key()
    if not key:
        _die("DAYFRAME_API_KEY is not set; add it to .env or export it (never config.toml)")
    cfg = load_config_optional()
    try:
        require_anthropic(cfg)
    except BaselineProviderError as exc:
        _die(str(exc))

    started = datetime.now().astimezone()
    conn = init_db(connect())
    try:
        try:
            day = parse_target_date(date)
        except ValueError as exc:
            _die(str(exc))
        run_id = f"run_{day.isoformat()}"
        seen = seen_uuids(conn, exclude_run_id=run_id)
        try:
            photos = run_for_date(get_reader(), date, cfg=cfg, seen=seen)
        except (FileNotFoundError, PermissionError, RuntimeError, OSError, ValueError) as exc:
            _die(str(exc))

        _print_pipeline_summary(photos)
        if not photos.clusters:
            typer.echo("Nothing to send.")
            return

        try:
            baseline = run_baseline(photos, cfg, api_key=key)
        except BaselineProviderError as exc:
            _die(str(exc))
        finished = datetime.now().astimezone()
        _print_baseline(baseline, title_prefix=cfg.calendar.title_prefix)

        if dry_run:
            typer.echo("dry-run: wrote nothing")
            return

        persist_baseline_run(
            conn,
            run_id=photos.run_id,
            target_date=photos.target_date.isoformat(),
            started_at=started,
            finished_at=finished,
            labeled=baseline.labeled,
            memories=baseline.memories,
            discarded=baseline.discarded,
            assets=photos.raw,
            provider=cfg.provider.name,
            model=baseline.model,
            prompt_version=baseline.prompt_version,
            images_sent=baseline.images_sent,
            input_tokens=baseline.input_tokens,
            output_tokens=baseline.output_tokens,
            cost_usd=baseline.cost_usd,
        )
        typer.echo(f"Wrote run {photos.run_id} to the local ledger.")
        if not no_calendar:
            typer.echo("calendar write not implemented until M4")
    finally:
        conn.close()


@app.command("install-agent")
def install_agent() -> None:
    """Write the launchd plist for the daily job."""
    _not_yet("M5")


@app.command()
def replay(run_id: str) -> None:
    """Walk a JSONL trace in human-readable form."""
    del run_id
    _not_yet("M3")


@app.command()
def undo(run_id: str) -> None:
    """Delete calendar events written by a run."""
    del run_id
    _not_yet("M4")


def _fmt_dt(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M")


def format_asset_line(asset: Asset) -> str:
    place = asset.place_name or "unknown"
    people = people_label(asset.persons)
    flags: list[str] = []
    if asset.screenshot:
        flags.append("screenshot")
    if asset.burst:
        flags.append("burst")
    extra = f"  [{', '.join(flags)}]" if flags else ""
    return (
        f"{asset.uuid}  {_fmt_dt(asset.date)}  added {_fmt_dt(asset.date_added)}  "
        f"{place}  {people}{extra}"
    )


def format_stored_cluster_line(seq: int, row: object) -> str:
    start = datetime.fromisoformat(str(row["start_ts"]))
    end = datetime.fromisoformat(str(row["end_ts"]))
    n = int(row["asset_count"])
    photo_word = "photo" if n == 1 else "photos"
    place = row["place"] or "unknown"
    return (
        f"[c{seq:03d}] {start.strftime('%H:%M')}–{end.strftime('%H:%M')} · "
        f"{n} {photo_word} · {place} · {row['id']}"
    )


def _die(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(code=1)


def _print_pipeline_summary(result: PipelineResult) -> None:
    drop_counts = Counter(item.reason for item in result.dropped)
    session_word = "session" if len(result.clusters) == 1 else "sessions"
    typer.echo(
        f"{result.target_date.isoformat()}. {len(result.raw)} photos across "
        f"{len(result.clusters)} candidate {session_word}."
    )
    typer.echo(
        f"Run {result.run_id}  "
        f"{len(result.raw)} read → {len(result.dropped)} dropped → {len(result.kept)} clustered"
    )
    if drop_counts:
        summary = ", ".join(f"{reason} {count}" for reason, count in sorted(drop_counts.items()))
        typer.echo(f"Dropped: {summary}")
    if not result.clusters:
        return
    typer.echo("")
    for index, cluster in enumerate(result.clusters, start=1):
        typer.echo(format_digest_line(index, cluster))


def _print_baseline(baseline: BaselineResult, *, title_prefix: str) -> None:
    typer.echo("")
    typer.echo(
        f"Sent {baseline.images_sent} images in 1 call. "
        f"{baseline.wall_seconds:.1f}s  ${baseline.cost_usd:.4f}  "
        f"{baseline.input_tokens} in / {baseline.output_tokens} out"
    )
    if baseline.memories:
        typer.echo("")
        for memory in baseline.memories:
            title = f"{title_prefix}{memory.title}" if title_prefix else memory.title
            typer.echo(
                f"KEEP  [{memory.cluster_id}] {title}  ({memory.confidence:.2f}, {memory.category})"
            )
            typer.echo(f"      {memory.body}")
    if baseline.discarded:
        typer.echo("")
        for discard in baseline.discarded:
            typer.echo(f"SKIP  [{discard.cluster_id}]  {discard.reason}")


@photos_app.command("list")
def photos_list(
    since: str | None = typer.Option(None, "--since", help="Lookback window, e.g. 24h."),
    date: str | None = typer.Option(None, "--date", help="Calendar day (yesterday or YYYY-MM-DD)."),
) -> None:
    """List assets the reader would see."""
    if since and date:
        _die("pass --since or --date, not both")
    try:
        if date:
            day = parse_target_date(date)
            start, end = calendar_day_window(day)
        else:
            start, end = since_window(parse_since(since or "24h"))
    except ValueError as exc:
        _die(str(exc))

    try:
        assets = get_reader().assets_added_between(start, end)
    except (FileNotFoundError, PermissionError, RuntimeError, OSError) as exc:
        _die(str(exc))

    typer.echo(f"{len(assets)} assets added {_fmt_dt(start)} → {_fmt_dt(end)}")
    if not assets:
        return
    typer.echo("")
    for asset in assets:
        typer.echo(format_asset_line(asset))


@clusters_app.command("show")
def clusters_show(
    run_id: str | None = typer.Argument(None, help="Persisted run id, e.g. run_2026-09-17."),
    date: str = typer.Option("yesterday", "--date", help="Target date (yesterday or YYYY-MM-DD)."),
) -> None:
    """Show clusters produced for a date or a persisted run."""
    if run_id:
        _show_stored_run(run_id)
        return
    _show_live_clusters(date)


def _show_stored_run(run_id: str) -> None:
    conn = init_db(connect())
    try:
        run = fetch_run(conn, run_id)
        if run is None:
            _die(f"no run {run_id}; try: dayframe clusters show --date yesterday")
        rows = fetch_clusters(conn, run_id)
    finally:
        conn.close()
    typer.echo(f"Run {run['id']}  date={run['target_date']}  {len(rows)} clusters")
    if not rows:
        return
    typer.echo("")
    for index, row in enumerate(rows, start=1):
        typer.echo(format_stored_cluster_line(index, row))


def _show_live_clusters(date: str) -> None:
    cfg = load_config_optional()
    conn = init_db(connect())
    started = datetime.now().astimezone()
    try:
        seen = seen_uuids(conn)
        try:
            result = run_for_date(get_reader(), date, cfg=cfg, seen=seen)
        except (FileNotFoundError, PermissionError, RuntimeError, OSError, ValueError) as exc:
            _die(str(exc))
        finished = datetime.now().astimezone()
        replace_inspect_run(
            conn,
            run_id=result.run_id,
            target_date=result.target_date.isoformat(),
            started_at=started,
            finished_at=finished,
            clusters=result.clusters,
            provider=cfg.provider.name,
            model=cfg.provider.model,
        )
    finally:
        conn.close()

    _print_pipeline_summary(result)


@eval_app.command("run")
def eval_run() -> None:
    """Score against the fixture set."""
    _not_yet("M6")
