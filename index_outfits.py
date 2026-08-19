"""
Batch multi-item detection over outfit_dataset/ (spec section 4.2, Phase 3).

WHAT THIS IS FOR
----------------
Every accuracy number this project has ever produced is catalog-photo-to-
catalog-photo: one clean studio photo, one garment, known category. The
real deployment condition is a photo of a person wearing several garments
at once. `segment_outfit.py` can already find the items in ONE such photo,
but only as a single-image CLI tool -- there was no way to run it across a
corpus. This script is that runner: it turns a folder of raw scraped
outfit photos into per-garment crops plus structured detections, which is
the precondition for both the spec's catalog-to-consumer evaluation and
for showing users real worn outfits.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not claim its labels are correct. Per the 2026-08-03 decision
recorded in SCRAPING_PROCESS.md, outfit images are collected UNLABELED and
there is no ground truth for what garments are actually in them. Every
field this script writes is model output, so all of it goes under its own
namespaced keys (`detected_items`, `detection_meta`) and never mixes with
scraped fact. `detection_meta` records exactly which models and thresholds
produced the labels, because those labels WILL be wrong some fraction of
the time and whoever reads them later needs to know what produced them.

CONCURRENCY
-----------
Reads and writes outfit_dataset/metadata.json with the same read-merge-
write discipline as dataset_utils.save_records_safe (see that module for
the real data-loss incident that motivated it), but keyed on
(source, source_id) rather than product_code, and merging ONLY this
script's own namespaced fields into each record -- so a scraper appending
new records concurrently can't be clobbered by this script's checkpoints,
and vice versa. The merge helper is kept local rather than added to
dataset_utils because scrapers were actively writing that file when this
was authored; consolidate later if it earns it.

DEVICE
------
CUDA if available, else CPU. Deliberately no MPS branch -- see
segment_outfit.py's module docstring for the measured reason (SAM2 on
Apple Silicon MPS was a real, documented performance trap). On a Mac this
means CPU and it will be slow; this script is meant to run on Colab/Modal
for any real corpus size, and --limit exists to smoke-test locally.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# sam2 is imported LAZILY, at the one call site that needs it (the
# --proposer sam2 path). At module scope it made `dominant_color` --
# which is pure PIL and numpy -- unimportable anywhere sam2 is not
# installed, including the Modal GPU image, where installing it means
# building a CUDA extension for a proposer that is no longer the default.
import segment_outfit
from segment_outfit import (
    detect_outfit_items,
    SAM2_CONFIG,
    SAM2_CHECKPOINT,
    MASK_GENERATOR_KWARGS,
    MIN_CONFIDENCE,
    MIN_AREA_FRACTION,
    MAX_AREA_FRACTION,
)

OUTFIT_DB_FILE = Path("outfit_dataset/metadata.json")
OUTFIT_ROOT = Path("outfit_dataset")
FASHION_CLIP_MODEL = "patrickjohncyh/fashion-clip"
CHECKPOINT_EVERY = 10

# Bumped whenever a change would make previously-written detections
# non-comparable to new ones (different model, different thresholds,
# different selection logic). Records carrying an older version are
# re-detected instead of being silently mixed with newer ones -- the same
# staleness discipline the retrieval index's cache fingerprint uses.
DETECTION_VERSION = 1


# ------------------------------------------------------------------
# Crude dominant-colour naming.
#
# Added because the co-occurrence index this run feeds (see
# build_cooccurrence.py) wants "black jacket + blue jeans", not just
# "jacket + pants" -- and colour is the one extra attribute that can be
# read straight off the pixels without a second model pass over 8k images.
#
# It is deliberately CRUDE and it is NOT ground truth:
#   - it reads the BBOX crop, not the mask (segment_outfit.detect_outfit_items
#     returns only the crop, and that file is off-limits for edits), so
#     background bleeds in at the corners. Mitigated by sampling the
#     centre 50% box only, where the garment actually is.
#   - nearest-neighbour in plain RGB against a 16-entry palette is not a
#     perceptual colour space. Navy/black and beige/white will confuse.
# Both facts are recorded in the co-occurrence index's schema so nobody
# downstream reads these as measured colour labels.
# ------------------------------------------------------------------
COLOR_PALETTE = {
    "black": (25, 25, 25),
    "charcoal": (70, 70, 72),
    "gray": (128, 128, 130),
    "light gray": (190, 190, 192),
    "white": (240, 240, 238),
    "cream": (232, 222, 198),
    "beige": (205, 183, 148),
    "brown": (110, 78, 52),
    "tan": (170, 132, 88),
    "olive": (110, 112, 62),
    "green": (60, 130, 70),
    "navy": (36, 46, 82),
    "blue": (60, 100, 180),
    "light blue": (145, 180, 220),
    "purple": (110, 70, 150),
    "red": (170, 45, 45),
    "pink": (225, 150, 165),
    "orange": (215, 125, 50),
    "yellow": (225, 200, 70),
}
_PALETTE_NAMES = list(COLOR_PALETTE)
_PALETTE_RGB = np.array([COLOR_PALETTE[name] for name in _PALETTE_NAMES], dtype=np.float32)


def dominant_color(crop):
    """Nearest palette name for the centre of a crop, plus the mean RGB it
    came from so a reader can re-judge the naming without the image."""
    try:
        width, height = crop.size
        if width < 4 or height < 4:
            return None
        box = (width // 4, height // 4, width - width // 4, height - height // 4)
        patch = crop.convert("RGB").crop(box)
        patch.thumbnail((32, 32))
        pixels = np.asarray(patch, dtype=np.float32).reshape(-1, 3)
        if pixels.size == 0:
            return None
        # Per-pixel vote rather than mean-then-name: the mean of a
        # black-and-white striped shirt is gray, which is a colour that is
        # not in the photo. A vote returns the actual majority colour.
        distances = ((pixels[:, None, :] - _PALETTE_RGB[None, :, :]) ** 2).sum(axis=2)
        assignments = distances.argmin(axis=1)
        counts = np.bincount(assignments, minlength=len(_PALETTE_NAMES))
        winner = int(counts.argmax())
        return {
            "name": _PALETTE_NAMES[winner],
            "share": float(counts[winner] / len(assignments)),
            "mean_rgb": [round(float(v), 1) for v in pixels.mean(axis=0)],
            "method": "centre-box palette vote (heuristic, not ground truth)",
        }
    except Exception:
        # A colour guess is never worth failing a detection over.
        return None


def record_key(record):
    return f"{record.get('source')}:{record.get('source_id')}"


def load_outfit_records():
    if OUTFIT_DB_FILE.exists():
        return json.loads(OUTFIT_DB_FILE.read_text(encoding="utf-8"))
    return []


def save_detections_safe(touched):
    """Merge ONLY this script's namespaced fields into whatever is on disk
    right now, keyed by (source, source_id). Re-reads at save time so a
    scraper that appended records after this run started doesn't get
    erased by a stale in-memory snapshot -- the exact failure
    dataset_utils.py exists to prevent, which cost this project 176
    records once."""
    current = load_outfit_records()
    by_key = {record_key(r): r for r in current}
    for key, fields in touched.items():
        if key in by_key:
            by_key[key].update(fields)
        # A record that vanished from disk is not resurrected here -- if a
        # scraper removed it, it stays removed.
    OUTFIT_DB_FILE.write_text(json.dumps(list(by_key.values()), indent=2, ensure_ascii=False), encoding="utf-8")
    return by_key


def detection_params(proposer="sam2"):
    """The knobs that decide what counts as a detected item. Stored on
    every record so a future reader can tell whether two records'
    detections are comparable without re-reading this file's git history.

    `proposer` is part of this on purpose. needs_detection() compares the
    whole dict, so without it a human-parsing re-run over sam2-labelled
    records reads as "already detected with the same params" and is
    skipped unless --force -- i.e. the better proposer would silently
    no-op. It is also the field that tells a later reader which half of a
    partially re-run corpus they are looking at.
    """
    return {
        "version": DETECTION_VERSION,
        "proposer": proposer,
        "sam2_checkpoint": str(SAM2_CHECKPOINT),
        "fashion_clip": FASHION_CLIP_MODEL,
        "mask_generator_kwargs": dict(MASK_GENERATOR_KWARGS),
        "min_confidence": MIN_CONFIDENCE,
        "min_area_fraction": MIN_AREA_FRACTION,
        "max_area_fraction": MAX_AREA_FRACTION,
    }


def needs_detection(record, params, force):
    if force:
        return True
    meta = record.get("detection_meta")
    if not meta:
        return True
    # Re-run when the params that decide what counts as an item changed.
    # Comparing the whole dict (not just version) catches a threshold
    # edited without a version bump.
    return meta.get("params") != params


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=None,
                        help="Only process records from this source (e.g. 'reddit'). Default: all.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N records. Use for smoke tests -- this is slow on CPU.")
    parser.add_argument("--force", action="store_true",
                        help="Re-detect records that already have detections from the same params.")
    parser.add_argument("--proposer", choices=["sam2", "human-parsing"], default="sam2",
                        help="Mask proposer. 'human-parsing' measured 2.48 items/photo at "
                             "91%% real-garment crops vs sam2's 1.53 at ~48%% (40 photos, all "
                             "127 crops inspected), and is ~53x faster because it drops SAM2's "
                             "mask generator. Default stays sam2 because outfit_cooccurrence.json "
                             "was built from it -- switching is a deliberate corpus re-run.")
    parser.add_argument("--max-images-per-record", type=int, default=2,
                        help="Cap images processed per post (default 2). Multi-image posts are often "
                             "the same outfit from several angles, so the marginal image is worth "
                             "much less than the first.")
    args = parser.parse_args()

    records = load_outfit_records()
    if not records:
        print(f"No records in {OUTFIT_DB_FILE} -- nothing to do. "
              f"(Scrapers write there; see SCRAPING_PROCESS.md's real-outfit section.)")
        return

    params = detection_params(args.proposer)
    pending = [r for r in records
               if (args.source is None or r.get("source") == args.source)
               and needs_detection(r, params, args.force)]
    if args.limit:
        pending = pending[:args.limit]

    print(f"{len(records):,} outfit records on disk; {len(pending):,} need detection"
          f"{f' (limited to {args.limit})' if args.limit else ''}.")
    if not pending:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"  # no MPS: see segment_outfit.py's docstring
    if device == "cpu":
        print("WARNING: running on CPU. SAM2 mask generation is slow here -- expect minutes per image. "
              "This is meant for Colab/Modal at real corpus sizes; use --limit locally.")
    print(f"Device: {device}")

    print("Loading FashionCLIP...")
    processor = AutoProcessor.from_pretrained(FASHION_CLIP_MODEL)
    clip_model = AutoModelForZeroShotImageClassification.from_pretrained(FASHION_CLIP_MODEL).to(device)

    print("Loading SAM2...")
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    sam2 = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(sam2, **MASK_GENERATOR_KWARGS)

    # Loaded once for the whole corpus, like the SAM2 generator above.
    parser_processor = parser_model = None
    if args.proposer == "human-parsing":
        from garment_proposer import load_human_parser, propose_garment_items
        print("Loading human parser...")
        parser_processor, parser_model = load_human_parser(device)

    touched = {}
    stats = {"records": 0, "images": 0, "items": 0, "empty_images": 0, "failed_images": 0}
    started = time.time()

    for position, record in enumerate(pending, start=1):
        key = record_key(record)
        crops_dir = OUTFIT_ROOT / str(record.get("source")) / str(record.get("source_id")) / "crops"
        detected_items = []

        for image_index, image_path in enumerate(record.get("images", [])[:args.max_images_per_record]):
            if not Path(image_path).is_file():
                print(f"  missing file, skipped: {image_path}")
                stats["failed_images"] += 1
                continue
            try:
                if args.proposer == "human-parsing":
                    items = propose_garment_items(
                        image_path, parser_processor, parser_model,
                        processor, clip_model, device)
                else:
                    items = detect_outfit_items(image_path, mask_generator, processor, clip_model, device)
            except Exception as error:
                # One bad image must not kill a multi-hour corpus run.
                print(f"  detection failed on {image_path}: {error}")
                stats["failed_images"] += 1
                continue

            stats["images"] += 1
            if not items:
                stats["empty_images"] += 1

            if items:
                crops_dir.mkdir(parents=True, exist_ok=True)
            for rank, item in enumerate(items, start=1):
                crop_path = crops_dir / f"img{image_index}_item{rank}_{item['category'].replace(' ', '-')}.jpg"
                try:
                    item["crop"].convert("RGB").save(crop_path, quality=90)
                except Exception as error:
                    print(f"  crop save failed ({crop_path}): {error}")
                    continue
                detected_items.append({
                    "source_image": str(image_path),
                    "rank": rank,
                    "category_group": item["category_group"],
                    "category": item["category"],
                    "label": item["label"],
                    "confidence": item["confidence"],
                    "bbox": list(item["bbox"]),
                    "area_fraction": item["area_fraction"],
                    "crop_path": str(crop_path),
                    "color": dominant_color(item["crop"]),
                })

        stats["items"] += len(detected_items)
        stats["records"] += 1
        # Written even when empty: "this record was processed and produced
        # nothing" is a real, useful result (it means the photo had no
        # findable garments), and without it every run would retry the
        # same hopeless images forever.
        touched[key] = {
            "detected_items": detected_items,
            "detection_meta": {
                "params": params,
                "detected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "num_items": len(detected_items),
                "device": device,
            },
        }

        rate = stats["records"] / max(time.time() - started, 1e-6)
        print(f"[{position}/{len(pending)}] {key}: {len(detected_items)} item(s)  ({rate*60:.1f} records/min)")

        if len(touched) >= CHECKPOINT_EVERY:
            save_detections_safe(touched)
            touched = {}
            print(f"  checkpointed ({stats['records']} records so far)")

    if touched:
        save_detections_safe(touched)

    elapsed = time.time() - started
    print(f"\nDone in {elapsed/60:.1f} min.")
    print(f"  records processed : {stats['records']:,}")
    print(f"  images processed  : {stats['images']:,}")
    print(f"  items detected    : {stats['items']:,}")
    # Percentage computed on its own line rather than inline: the nested
    # same-quote f-string this replaced only parses on Python 3.12+, and
    # the Modal image is 3.11 -- so the summary at the END of a multi-hour
    # GPU run was a SyntaxError at IMPORT time there. Caught by compiling
    # this file under an older interpreter before shipping it, not by
    # burning a GPU run to find out.
    empty_share = (f" ({stats['empty_images'] / stats['images'] * 100:.1f}%)"
                   if stats["images"] else "")
    print(f"  images with 0 items: {stats['empty_images']:,}{empty_share}")
    print(f"  images failed     : {stats['failed_images']:,}")
    if stats["images"]:
        print(f"  mean items/image  : {stats['items']/stats['images']:.2f}")
    print("\nReminder: these labels are UNVALIDATED model output -- there is no ground truth for "
          "what garments are actually in these photos (see SCRAPING_PROCESS.md's real-outfit section). "
          "Treat the numbers above as coverage, not accuracy.")


if __name__ == "__main__":
    main()
