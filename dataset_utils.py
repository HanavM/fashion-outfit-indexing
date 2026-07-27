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

import json
from pathlib import Path

DB_FILE = Path("apparel_dataset/metadata.json")


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
