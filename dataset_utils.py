"""
Shared helpers for reading/writing apparel_dataset/metadata.json safely
across multiple long-running scripts.

Why this exists: every scraping/enrichment/captioning/segmentation script in
this pipeline loads the full record list into memory once, then periodically
checkpoints by writing that in-memory list back to disk. If two such scripts
run concurrently — e.g. a several-hour segment_apparel.py run started before
a separate pacsun_scraper.py run began and finished — the longer-running
script's stale in-memory snapshot (taken before the other script's new
records existed) gets written back on every checkpoint, silently erasing
whatever the other script appended. This happened for real: a ~10-hour
segment_apparel.py run clobbered 176 freshly-scraped PacSun records on every
checkpoint because its own copy of the database predated them.

save_records_safe() fixes this by re-reading the current on-disk state at
save time and merging this run's changes into it (keyed by product_code),
rather than blindly overwriting with an old in-memory copy. It does not
eliminate the race entirely (two processes could still both re-read-merge-
write in a tight enough window to lose one one of the two updates), but it
turns "guaranteed to erase anything added since I started" into "only loses
data in an actual same-instant collision," which is what actually happened
in practice.
"""

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

DB_FILE = Path("apparel_dataset/metadata.json")

# The real-outfit dataset (photos of real people wearing real outfits) is a
# SEPARATE tree with its own metadata file -- see SCRAPING_PROCESS.md,
# "Real-outfit scraping". It must never be merged into apparel_dataset, which
# is the labelled retrieval gallery every eval number depends on.
OUTFIT_DB_FILE = Path("outfit_dataset/metadata.json")


def load_records() -> list[dict]:
    if DB_FILE.exists():
        return json.loads(DB_FILE.read_text())
    return []


def save_records_safe(touched: dict) -> list[dict]:
    """Merge `touched` (product_code -> record) into whatever is currently
    on disk and write the result. Returns the merged full list."""
    current = load_records()
    by_code = {r["product_code"]: r for r in current}
    by_code.update(touched)
    merged = list(by_code.values())
    DB_FILE.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    return merged


def outfit_key(record: dict) -> str:
    """Stable identity for an outfit record: source + per-source id."""
    return f"{record['source']}:{record['source_id']}"


# Outfit sources are scraped CONCURRENTLY (Reddit is API-bound, the browser
# scraper is render-bound, so running both at once is nearly free) and every
# one of them does a read-modify-write of the SAME metadata.json. Two things
# break without the lock + atomic replace below, and both break silently:
#   - lost updates: two processes each read N records, each add their own,
#     each write N+k -- whoever writes last erases the other's batch;
#   - torn reads: `write_text` truncates before it writes, so a reader that
#     lands mid-write gets a half file and json.load() raises.
# The lock is advisory and process-wide, held only across the merge+write.
OUTFIT_LOCK_FILE = Path("outfit_dataset/.metadata.lock")


@contextmanager
def _outfit_lock():
    OUTFIT_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTFIT_LOCK_FILE, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_outfit_records() -> list[dict]:
    if OUTFIT_DB_FILE.exists():
        return json.loads(OUTFIT_DB_FILE.read_text())
    return []


def save_outfit_records_safe(touched: dict) -> list[dict]:
    """Same merge-on-save discipline as save_records_safe(), for
    outfit_dataset/metadata.json. `touched` maps outfit_key -> record.

    Concurrency-safe: the read, merge and write happen under an exclusive
    file lock, and the write itself is a temp-file + os.replace, which is
    atomic on POSIX -- a concurrent reader sees either the old file or the
    new one, never a truncated one.
    """
    with _outfit_lock():
        current = load_outfit_records()
        by_key = {outfit_key(r): r for r in current}
        by_key.update(touched)
        merged = list(by_key.values())
        OUTFIT_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUTFIT_DB_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        os.replace(tmp, OUTFIT_DB_FILE)
    return merged
