"""Merge detections computed on Modal back into the local outfit corpus.

The GPU job writes its results into `outfit_dataset/metadata.json` on the
`outfit-index` Volume. That file is a full copy of the corpus as it stood
when it was uploaded, so copying it over the local one would silently
revert anything the scrapers appended in the meantime -- which is exactly
the failure `dataset_utils.save_records_safe` exists to prevent, and which
cost this project 176 records once.

So this merges instead of copying, and it merges ONLY the detector's own
namespaced keys:

    detected_items, detection_meta

Scraped provenance (post_url, author, title, images, phash, scraped_at,
source_tags ...) is never touched, in either direction. That separation is
the 2026-08-03 decision recorded in SCRAPING_PROCESS.md: outfit photos are
collected UNLABELED, and model output must stay distinguishable from
scraped fact rather than being blended into it.

Records present remotely but absent locally are NOT resurrected -- if a
scraper or the deduper removed one, it stays removed. Records present
locally but absent remotely are left alone.

Usage:
    modal volume get outfit-index outfit_dataset/metadata.json /tmp/remote_metadata.json
    python3 merge_outfit_detections.py --remote /tmp/remote_metadata.json
    python3 merge_outfit_detections.py --remote /tmp/remote_metadata.json --dry-run
"""

import argparse
import json
import shutil
import time
from pathlib import Path

LOCAL_METADATA = Path("outfit_dataset/metadata.json")

# The only keys this script is allowed to move. Anything else in the
# remote file is ignored on purpose.
DETECTION_KEYS = ("detected_items", "detection_meta")


def record_key(record):
    return f"{record.get('source')}:{record.get('source_id')}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--remote", required=True,
                        help="metadata.json fetched from the outfit-index Volume.")
    parser.add_argument("--local", default=str(LOCAL_METADATA))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    local_path = Path(args.local)
    local = json.loads(local_path.read_text(encoding="utf-8"))
    remote = json.loads(Path(args.remote).read_text(encoding="utf-8"))

    remote_by_key = {record_key(r): r for r in remote if r.get("detection_meta")}
    print(f"local records : {len(local):,}")
    print(f"remote records carrying detections: {len(remote_by_key):,}")

    updated = 0
    unchanged = 0
    for record in local:
        source = remote_by_key.get(record_key(record))
        if not source:
            continue
        fields = {key: source[key] for key in DETECTION_KEYS if key in source}
        if all(record.get(key) == value for key, value in fields.items()):
            unchanged += 1
            continue
        record.update(fields)
        updated += 1

    missing_locally = len(remote_by_key) - updated - unchanged
    print(f"  updated            : {updated:,}")
    print(f"  already up to date : {unchanged:,}")
    print(f"  in remote but not local (left alone): {missing_locally:,}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    if not args.no_backup:
        backup = local_path.with_suffix(f".backup-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
        shutil.copy2(local_path, backup)
        print(f"\nbackup: {backup}")

    local_path.write_text(json.dumps(local, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {local_path}")
    print("\nThe merged fields are UNVALIDATED model output. They live only under "
          "detected_items / detection_meta and assert nothing about the scraped "
          "record they attach to.")


if __name__ == "__main__":
    main()
