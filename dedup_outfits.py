"""Perceptual-dedup sweep over `outfit_dataset/` (NOT apparel_dataset).

Why this exists: the scrapers dedup as they go, but each *process* holds its
own in-memory phash list loaded at startup. That is exact within a process
and blind across them, and the reddit scrape is deliberately fanned out to
one process per subreddit (run_reddit_wide.sh) because it is network-bound.
Reposts across fashion subreddits are endemic -- the same fit picture shows
up under a different post id in r/streetwear and r/malefashion the same
week -- so a cross-process sweep is the counterpart to that fan-out, not an
optional cleanup.

It also catches the same photo arriving from two different *sources*, e.g.
a Reddit post and a wear.jp coordinate of the same image.

Dedup is at IMAGE level, not record level: a gallery post can share one
photo with another post while its other images are unique, and dropping the
whole record would throw away good data. A record left with zero images is
removed entirely, along with its directory.

Keeps the FIRST occurrence in metadata.json order, which is scrape order,
so the earliest-collected copy wins and repeat runs are stable.

    python dedup_outfits.py            # report only
    python dedup_outfits.py --apply    # rewrite metadata + delete files
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import imagehash
import numpy as np

from dataset_utils import (
    OUTFIT_DB_FILE, _outfit_lock, load_outfit_records, outfit_key,
)
from outfit_scrape_common import PHASH_DISTANCE


def hash_bits(records):
    """Flatten to (record_index, image_index, 64-bit array) for every image."""
    index, bits = [], []
    for r_index, record in enumerate(records):
        for i_index, value in enumerate(record.get("phash", [])):
            try:
                array = imagehash.hex_to_hash(value).hash.flatten()
            except Exception:
                continue
            index.append((r_index, i_index))
            bits.append(array.astype(np.uint8))
    if not bits:
        return index, np.zeros((0, 64), dtype=np.uint8)
    return index, np.vstack(bits)


def find_duplicates(index, bits):
    """Indices into `index` that duplicate an earlier entry.

    O(n^2) in principle, but vectorised one row at a time: 64 uint8 columns
    means the whole comparison for a row is a single numpy op, which is
    plenty for a dataset in the low tens of thousands of images.
    """
    dropped = set()
    kept = np.zeros_like(bits)  # preallocated: rebuilding the array each
    kept_count = 0              # iteration would dominate the runtime
    for position in range(len(index)):
        row = bits[position]
        if kept_count:
            distances = np.count_nonzero(kept[:kept_count] != row, axis=1)
            if distances.min() <= PHASH_DISTANCE:
                dropped.add(position)
                continue
        kept[kept_count] = row
        kept_count += 1
    return dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="actually rewrite metadata and delete files "
                             "(default is a report-only dry run)")
    args = parser.parse_args()

    records = load_outfit_records()
    index, bits = hash_bits(records)
    print(f"{len(records)} records, {len(index)} hashed images "
          f"(phash distance <= {PHASH_DISTANCE})")
    if not len(index):
        return

    dropped = find_duplicates(index, bits)
    if not dropped:
        print("no cross-source duplicates found")
        return

    # Group by record so each record's image lists are rebuilt once, and
    # drop from the highest image index down so earlier indices stay valid.
    per_record = {}
    for position in sorted(dropped):
        r_index, i_index = index[position]
        per_record.setdefault(r_index, []).append(i_index)

    by_source, files, emptied = {}, [], []
    for r_index, image_indices in per_record.items():
        record = records[r_index]
        by_source[record["source"]] = by_source.get(record["source"], 0) + len(image_indices)
        for i_index in sorted(image_indices, reverse=True):
            for field in ("images", "image_urls", "phash"):
                if i_index < len(record.get(field, [])):
                    value = record[field].pop(i_index)
                    if field == "images":
                        files.append(value)
        record["image_count"] = len(record["images"])
        if not record["images"]:
            emptied.append(r_index)

    print(f"duplicate images: {len(dropped)}"
          + "".join(f"\n  {s:<10} {n}" for s, n in sorted(by_source.items())))
    print(f"records left empty (would be removed): {len(emptied)}")

    if not args.apply:
        print("\ndry run -- nothing written. Re-run with --apply.")
        return

    for path in files:
        Path(path).unlink(missing_ok=True)
    for r_index in emptied:
        images_dir = (Path("outfit_dataset") / records[r_index]["source"]
                      / records[r_index]["source_id"])
        shutil.rmtree(images_dir, ignore_errors=True)

    edited = {outfit_key(records[i]): records[i]
              for i in per_record if i not in set(emptied)}
    removed_keys = {outfit_key(records[i]) for i in emptied}

    # This is the clobber bug this repo has already been bitten by once:
    # `records` was read minutes ago and scrapers are very likely still
    # appending. Writing that stale list back would erase everything they
    # added. So re-read INSIDE the lock and apply only this run's edits and
    # deletions by key, leaving anything new untouched. save_outfit_records_safe
    # can't be reused here because it only merges and never deletes.
    with _outfit_lock():
        current = load_outfit_records()
        merged = []
        for record in current:
            key = outfit_key(record)
            if key in removed_keys:
                continue
            merged.append(edited.get(key, record))
        tmp = OUTFIT_DB_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        os.replace(tmp, OUTFIT_DB_FILE)
    print(f"wrote {len(merged)} records, deleted {len(files)} image files")


if __name__ == "__main__":
    main()
