"""
Bulk backfill of the GFS analysis cache: download f000 slices, extract every cell,
drop the GRIB.

The training cache is built from GFS 0.25° analysis (f000) on S3, one ~100 MB
byte-range slice per (date, hour). A slice carries *global* GRIB messages, so its
size is independent of how many cells we extract — widening the region is free in
bandwidth, and only the date range costs anything. That is why this runner is
organised per slice, not per cell.

What happens to the raw GRIB is the one decision worth making up front. Dropped
after extraction, peak disk stays at `workers` x ~110 MB — but the cache then holds
only the cells extracted that day, and widening the region later means pulling all
~434 GB again. Kept via `--archive-dir` (~434 GB for March-October 2021-2026, so a
spinning disk), new cells re-extract off the archive with `data gfs` and download
nothing.

Usage:
    paraglideml data backfill --start 2021-03-01 --end 2026-08-02 \
        --archive-dir /mnt/backup/paraglideml/gfs/anl
"""

import datetime as dt
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..config import GFS_CACHE_DIR, PROCESSED_DATA_DIR
from .gfs_processor import get_cache_path, process_grib_pygrib

HOURS = (6, 12, 18)


def _parse_cell(cell_id: str) -> Tuple[int, int]:
    lat, lon = cell_id.split("_")
    return int(lat), int(lon)


def load_extract_cells(path: Optional[Path] = None) -> List[Tuple[int, int]]:
    """Cell list to extract, as (lat, lon) ints. Defaults to extract_cells.json."""
    path = Path(path or PROCESSED_DATA_DIR / "extract_cells.json")
    return [_parse_cell(c) for c in json.loads(path.read_text())]


def slices_between(start: dt.date, end: dt.date, months: Sequence[int]) -> List[dt.datetime]:
    """Every (date, hour) slice in [start, end] whose month is in `months`."""
    out = []
    d = start
    while d <= end:
        if d.month in months:
            out.extend(dt.datetime(d.year, d.month, d.day, h) for h in HOURS)
        d += dt.timedelta(days=1)
    return out


def _missing_cells(
    when: dt.datetime, cells: Sequence[Tuple[int, int]], cache_root: Path
) -> List[Tuple[int, int]]:
    return [c for c in cells if not get_cache_path(cache_root, c[0], c[1], when).exists()]


def archive_path(archive_root: Path, when: dt.datetime) -> Path:
    """Where a slice lives in the raw-GRIB archive.

    Deliberately the layout `gfs_processor.run_gfs_cache_creation` reads, so a later
    region expansion can re-extract new cells straight off the archive with
    `paraglideml data gfs --dates ... --bbox ...` and download nothing.
    """
    return Path(archive_root) / when.strftime("%Y-%m") / when.strftime(
        "gfsanl_3_%Y%m%d_%H00_000.grb2"
    )


def _do_slice(job) -> Tuple[str, int, int, bool]:
    """Fetch one slice, extract the missing cells, keep or drop the GRIB. Runs in a worker.

    Returns (tag, npz written, cells wanted, ok). `ok` is False only on a real failure —
    a slice fetched purely to fill the archive writes no npz and still succeeds.
    """
    when, cells, cache_root, grib_root, archive_root = job
    # Imported here so the parent process never pulls the (heavy) predict module.
    from ..predict import download_gfs_slice

    cache_root, grib_root = Path(cache_root), Path(grib_root)
    tag = f"{when:%Y%m%d}_{when.hour:02d}"
    keep = archive_root is not None
    dest = archive_path(archive_root, when) if keep else grib_root / f"bf_{tag}.grb2"
    reused = keep and dest.exists() and dest.stat().st_size > 0
    try:
        if not reused and not download_gfs_slice(when.date(), when.hour, dest, fxx=0):
            return tag, 0, 0, False
        if not cells:  # slice was only wanted for the archive
            return tag, 0, 0, True
        data_map = process_grib_pygrib(dest, list(cells))
        if not data_map:
            return tag, 0, 0, False
        written = 0
        for (lat, lon), content in data_map.items():
            if not content["values"]:
                continue
            out = get_cache_path(cache_root, lat, lon, when)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out,
                values=np.array(content["values"], dtype=np.float32),
                keys=np.array(content["keys"]),
                lat=lat,
                lon=lon,
                timestamp=when.isoformat(),
            )
            written += 1
        return tag, written, len(cells), True
    except Exception as e:  # one bad slice must not abort a multi-hour run
        print(f"  [{tag}] failed: {e}")
        return tag, 0, 0, False
    finally:
        if not keep:
            dest.unlink(missing_ok=True)


def run_backfill(
    start: str,
    end: str,
    cells: Optional[Iterable[str]] = None,
    months: Sequence[int] = tuple(range(3, 11)),
    cache_root: Optional[Path] = None,
    grib_root: Optional[Path] = None,
    archive_root: Optional[Path] = None,
    workers: int = 3,
) -> dict:
    """
    Fill the analysis cache for every (date, hour) slice in the window.

    Slices whose cells are all cached are skipped, so the run is resumable: kill it
    and restart with the same arguments.

    `archive_root` keeps the raw GRIB instead of deleting it — ~105 MB per slice, so
    a full March-October 2021-2026 window is ~434 GB and wants a spinning disk. Worth
    it if the region may grow again: extracting cells from the archive costs no
    download, whereas the cache alone only holds the cells extracted at the time. An
    archived slice is reused in place, so a re-run over an existing archive is offline.
    """
    cache_root = Path(cache_root or GFS_CACHE_DIR)
    grib_root = Path(grib_root or (Path(cache_root).parent / "backfill_tmp"))
    grib_root.mkdir(parents=True, exist_ok=True)
    if archive_root is not None:
        archive_root = Path(archive_root)
        archive_root.mkdir(parents=True, exist_ok=True)

    cell_list: List[Tuple[int, int]] = (
        [_parse_cell(c) for c in cells] if cells else load_extract_cells()
    )
    start_d = dt.datetime.strptime(start, "%Y-%m-%d").date()
    end_d = dt.datetime.strptime(end, "%Y-%m-%d").date()
    all_slices = slices_between(start_d, end_d, months)

    jobs = []
    to_download = 0
    for when in all_slices:
        missing = _missing_cells(when, cell_list, cache_root)
        archived = archive_root is not None and archive_path(archive_root, when).exists()
        # With an archive, a slice still needs fetching for its GRIB even when every
        # cell is already extracted — that is the copy a future expansion re-reads.
        if not missing and (archive_root is None or archived):
            continue
        jobs.append((when, missing, str(cache_root), str(grib_root), str(archive_root) if archive_root else None))
        if not archived:
            to_download += 1

    print(
        f"Срезов в окне: {len(all_slices)}; к обработке: {len(jobs)}, "
        f"из них качать {to_download} (~{to_download * 0.105:.0f} ГБ), "
        f"ячеек: {len(cell_list)}"
        + (f"; архив GRIB: {archive_root}" if archive_root else "; GRIB удаляются")
    )

    stats = {"slices": 0, "npz": 0, "failed": 0}
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_do_slice, j): j[0] for j in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                tag, written, wanted, ok = fut.result()
                stats["slices"] += 1
                stats["npz"] += written
                if not ok:
                    stats["failed"] += 1
                if i % 25 == 0 or i == len(jobs):
                    print(
                        f"  [{i}/{len(jobs)}] последний {tag}: {written}/{wanted} ячеек; "
                        f"всего npz {stats['npz']}, сбоев {stats['failed']}",
                        flush=True,
                    )
    finally:
        shutil.rmtree(grib_root, ignore_errors=True)

    print(f"Готово: срезов {stats['slices']}, npz {stats['npz']}, сбоев {stats['failed']}")
    return stats
