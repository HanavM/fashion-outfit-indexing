"""Export image-text pairs for a SigLIP2 fine-tune on OUTFIT photos.

    .venv/bin/python prepare_outfit_finetune_data.py --dry-run
    .venv/bin/python prepare_outfit_finetune_data.py
    nohup .venv/bin/python -u prepare_outfit_finetune_data.py \
        --materialise-crops --resume > logs/finetune_export.log 2>&1 & disown

Reads local files only. Writes a JSONL manifest. **No GPU, no paid API, no
Modal.** The optional `--materialise-crops` pass runs `garment_proposer`
on this machine's CPU/MPS, which is free and slow; everything else is
seconds.

## Why this exists

The project concluded the ENCODER is the binding constraint on outfit
search, with numbers rather than a hunch: mean pairwise similarity between
two random outfit photos is **0.743**, a real query's top-1 scores 0.203
against top-50 at 0.187, and "a photo of a bicycle" still scores 0.155.
Most of each vector encodes "a person standing in a room". The deployed
checkpoint (`finetuned_siglip2_hierarchical_v3`) was fine-tuned on CATALOG
PRODUCT PHOTOS -- one garment, white sweep -- and is out of distribution on
a phone photo of a person in a bedroom. Fine-tuning it on the outfit
corpus is the next real lever and it needs labelled image-text pairs.

## Which labels are usable, decided by measurement not by count

Three label files exist over the corpus. Only one of them is a usable
training TEXT, and saying so is the point of this docstring.

| file | records | what the text would be | verdict |
|---|---:|---|---|
| `outfit_eval/vlm_labels.json` | 4,431 photos | gpt-4o-mini's (colour, garment) list | **use this** |
| `outfit_dataset/product_labels.json` | 15,869 crops | FashionCLIP's 13-way category; `detected_color` is **None on all 15,869** | weak, opt-in |
| `outfit_dataset/product_labels.bbox.json` | 20,681 crops | same, plus the heuristic palette colour | **do not use** |

- The two `product_labels` files are **not VLM labels**. They are the
  deployed identifier's ranked CATALOG PRODUCTS for each crop, carried
  alongside the detector's own category/group. Their richest text -- the
  top-1 product name -- is wrong more often than not on consumer crops:
  best measured R@1 is 47.65% and that is catalog-photo-to-catalog-product,
  a strictly easier condition. Open-set rejection never fires (0 of 36,550
  records have `rejected_open_set`), so nothing filters the guesses.
  Training a contrastive objective on "this crop IS a Gap 10.5 inch Baggy
  Denim Short" when it is a 50/50 coin flip teaches the encoder noise.
  `--use-detector-categories` will emit the 13-way category text anyway,
  off by default, because a 13-token vocabulary carries very little for a
  contrastive loss and it is FashionCLIP's opinion, not a label.
- `bbox`'s `detected_color` is the heuristic palette vote, measured at
  **38.5% agreement** with the VLM on 1,632 garments, with one-directional
  errors (brown/navy/blue -> black). That is the exact bug the colour head
  was built to fix. It is not a training label.
- The VLM labels are independent of the retrieval stack -- no shared
  weights, no shared training data -- and they answer the question a user
  is asking ("the colour of the GARMENT", not "the colour of the PIXELS").
  That independence is what makes them worth training on.

## Two views, and the photo-level one is the more valuable

`--view photo` pairs the **whole photo** with the whole outfit
("beige pants, brown loafer, white t-shirt"). This attacks the measured
failure directly: whole-photo embeddings are the ones sitting at 0.743
mean pairwise similarity. The captions are also genuinely discriminative
-- 2,812 distinct strings over 3,948 photos, 2,345 of them singletons --
which is what a contrastive loss needs.

`--view crop` pairs one masked garment crop with "a black jacket", using
the same **exactly-one** alignment rule `train_colour_head.py` uses: a
(garment, colour) is attached to a crop only when the photo has exactly
one crop of that category and the VLM named exactly one colour for it.
Ambiguity is dropped rather than guessed. This view is larger in rows but
far poorer in text: **189 distinct strings**, the top ten of which are
50.6% of the data. Every batch will contain several rows whose texts are
byte-identical, which an in-batch contrastive loss will treat as negatives
of each other. `text_id` is emitted on every row so the trainer can mask
those collisions (or switch to a SupCon-style multi-positive loss, which
is what `dino_identity_finetune.py` already does and what beat ArcFace by
13.4pt at this data scale).

Both views are exported, and **they share one photo-level split**, so a
photo's crops never sit opposite that same photo's full frame.

## Masked crops, not bbox crops

Measured paired on 237 garments (`crop_style_ab.py`): group agreement
**40.9% bbox -> 72.2% masked**, footwear **15.8% -> 87.7%**. A bbox drags
the floor, the wall and the other leg into the encoder. So the crop view
uses the proposer's masked crop -- `detected_items[].crop_path`, which
blanks every pixel outside the garment mask to white inside its own bbox.

**10,571 of 43,712 detections have that file on disk right now** (the
rest were written on Modal and never synced). Every source photo is
present, so the missing ones are *regenerable*, free, by re-running the
proposer locally; that is `--materialise-crops`, about 4 hours of CPU for
the corpus. Rows whose mask is not on disk are exported with
`image_kind: "pending_mask"` and are excluded from `split` unless
`--allow-bbox-crops` is passed, which downgrades them to a plain bbox crop
and marks them `image_kind: "bbox"` so the trainer can weight or drop
them. Nothing silently mixes the two crop styles.

## Class imbalance is handled explicitly, because it bit once already

`train_colour_head.py` hit "black" at ~30% and an unweighted head drifted
straight into predicting it -- inheriting the incumbent heuristic's bug.
Same shape here: "black" is 28.7% of aligned colours and ("black","pants")
alone is 9.3% of the crop view. Three levers, all reported in the summary:
`--cap-per-class` truncates over-represented label strings (per split, by
a seeded shuffle, so capping never changes the split), `--min-per-class`
drops strings too rare to learn or evaluate, and every row carries
`sample_weight = (N / count) ** 0.5` -- the same square-root inverse
frequency the colour head uses.

## Incremental, and checkpointed on ELAPSED TIME

Long local jobs on this machine get killed: a 40-minute crop build once
took 10.5 hours and died four times with macOS background services
competing for the silicon. Batch-count checkpoint intervals stretch past
the kill interval exactly when load is high, which is when you need them,
so this checkpoints on `--checkpoint-seconds` of wall clock. `--resume`
reads the manifest already on disk and skips the rows in it.
"""

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

OUTFIT_METADATA = REPO_ROOT / "outfit_dataset" / "metadata.json"
VLM_LABELS = REPO_ROOT / "outfit_eval" / "vlm_labels.json"
PRODUCT_LABELS_MASKED = REPO_ROOT / "outfit_dataset" / "product_labels.json"
OUT_DIR = REPO_ROOT / "outfit_finetune"
MANIFEST = OUT_DIR / "pairs.jsonl"
SUMMARY = OUT_DIR / "summary.json"
CROP_CACHE = OUT_DIR / "crops"

# Square root inverse frequency, matching train_colour_head.py. Full
# inverse frequency over-corrects a 19-class head into predicting the tail.
WEIGHT_EXPONENT = 0.5


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load_detections():
    """One dict per detected garment, keyed so labels can join onto it.

    `crop_path` is the proposer's MASKED crop. It is recorded whether or
    not the file is local -- 31,239 of 43,712 detections carry a path and
    only 10,571 of those were ever written to this disk.
    """
    records = json.loads(OUTFIT_METADATA.read_text())
    detections = []
    for record in records:
        post_id = f"{record.get('source')}:{record.get('source_id')}"
        for order, item in enumerate(record.get("detected_items") or []):
            rel = item.get("source_image")
            bbox = item.get("bbox")
            if not rel or not bbox:
                continue
            crop_path = item.get("crop_path")
            detections.append({
                "rel": rel,
                "bbox": [int(v) for v in bbox],
                "post_id": post_id,
                "source": record.get("source"),
                "post_url": record.get("post_url"),
                "category": item.get("category"),
                "category_group": item.get("category_group"),
                "detection_index": order,
                "crop_path": crop_path,
                "crop_on_disk": bool(crop_path) and (REPO_ROOT / crop_path).exists(),
                "photo_on_disk": (REPO_ROOT / rel).exists(),
            })
    return detections


def load_vlm():
    if not VLM_LABELS.exists():
        raise SystemExit(f"no VLM labels at {VLM_LABELS} -- "
                         "run `outfit_search_eval.py label` first")
    return json.loads(VLM_LABELS.read_text())


# ------------------------------------------------------------------
# Pair construction
# ------------------------------------------------------------------

def photo_pairs(vlm, detections, args):
    """Whole photo <-> whole-outfit caption.

    Garments are sorted before joining so two photos wearing the same
    outfit get the same caption string rather than two orderings of it --
    otherwise `text_id` overcounts distinct texts and the in-batch
    collision masking under-fires.
    """
    have_detection = {d["rel"] for d in detections}
    rows, skipped = [], collections.Counter()
    for rel, entry in vlm.items():
        if not entry.get("is_outfit_photo"):
            skipped["not_an_outfit_photo"] += 1
            continue
        if not (REPO_ROOT / rel).exists():
            skipped["photo_missing_on_disk"] += 1
            continue
        parts = sorted({f"{g['colour']} {g['garment']}"
                        for g in (entry.get("garments") or [])
                        if g.get("colour") and g.get("garment")})
        if len(parts) < args.min_garments:
            skipped["too_few_garments"] += 1
            continue
        rows.append({
            "view": "photo",
            "image": rel,
            "image_kind": "photo",
            "photo": rel,
            "text": ", ".join(parts),
            "label": ", ".join(parts),
            "n_garments": len(parts),
            "has_detections": rel in have_detection,
        })
    return rows, skipped


def crop_pairs(vlm, detections, args):
    """One masked garment crop <-> "<colour> <garment>".

    Alignment discipline copied from train_colour_head.py, deliberately:
    a (garment, colour) attaches to a crop only when the photo has
    EXACTLY ONE crop of that category and the VLM named EXACTLY ONE
    colour for it. A wrong pair teaches the encoder the wrong thing; a
    missing one only costs sample size.
    """
    by_key = collections.defaultdict(list)
    for detection in detections:
        by_key[(detection["rel"], detection["category"])].append(detection)

    rows, skipped = [], collections.Counter()
    for rel, entry in vlm.items():
        if not entry.get("is_outfit_photo"):
            skipped["not_an_outfit_photo"] += 1
            continue
        if not (REPO_ROOT / rel).exists():
            skipped["photo_missing_on_disk"] += 1
            continue
        by_garment = collections.defaultdict(set)
        for garment in entry.get("garments") or []:
            if garment.get("garment") and garment.get("colour"):
                by_garment[garment["garment"]].add(garment["colour"])
        for garment, colours in by_garment.items():
            candidates = by_key.get((rel, garment), [])
            if not candidates:
                skipped["no_matching_crop"] += 1
                continue
            if len(candidates) != 1:
                skipped["ambiguous_crop"] += 1
                continue
            if len(colours) != 1:
                skipped["ambiguous_colour"] += 1
                continue
            detection = candidates[0]
            colour = next(iter(colours))
            if detection["crop_on_disk"]:
                image, kind = detection["crop_path"], "masked"
            else:
                cached = CROP_CACHE / f"{_crop_stem(detection)}.jpg"
                if cached.exists():
                    image, kind = str(cached.relative_to(REPO_ROOT)), "masked"
                elif args.allow_bbox_crops:
                    image, kind = rel, "bbox"
                else:
                    image, kind = rel, "pending_mask"
            rows.append({
                "view": "crop",
                "image": image,
                "image_kind": kind,
                "bbox": detection["bbox"],
                "photo": rel,
                "text": f"a {colour} {garment}",
                "label": f"{colour} {garment}",
                "colour": colour,
                "garment": garment,
                "group": detection["category_group"],
                "detection_index": detection["detection_index"],
            })
    return rows, skipped


def detector_category_pairs(detections, args):
    """Opt-in, weak: FashionCLIP's 13-way category as the text.

    Off by default. 13 tokens is almost no contrastive signal, 26% of them
    are "pants", and it is a model's opinion rather than a label. Exposed
    so the claim can be tested rather than argued about.
    """
    if not PRODUCT_LABELS_MASKED.exists():
        return [], collections.Counter(missing_file=1)
    labels = json.loads(PRODUCT_LABELS_MASKED.read_text())
    by_bbox = {(d["rel"], tuple(d["bbox"])): d for d in detections}
    rows, skipped = [], collections.Counter()
    for entry in labels.values():
        category = entry.get("detected_category")
        rel = entry.get("rel")
        if not category or not rel:
            skipped["no_category"] += 1
            continue
        if not (REPO_ROOT / rel).exists():
            skipped["photo_missing_on_disk"] += 1
            continue
        detection = by_bbox.get((rel, tuple(entry.get("bbox") or [])))
        if detection is None:
            skipped["no_matching_detection"] += 1
            continue
        if detection["crop_on_disk"]:
            image, kind = detection["crop_path"], "masked"
        else:
            cached = CROP_CACHE / f"{_crop_stem(detection)}.jpg"
            if cached.exists():
                image, kind = str(cached.relative_to(REPO_ROOT)), "masked"
            elif args.allow_bbox_crops:
                image, kind = rel, "bbox"
            else:
                image, kind = rel, "pending_mask"
        rows.append({
            "view": "detector_category",
            "image": image,
            "image_kind": kind,
            "bbox": detection["bbox"],
            "photo": rel,
            "text": f"a photo of a {category}",
            "label": category,
            "garment": category,
            "group": detection["category_group"],
            "detection_index": detection["detection_index"],
        })
    return rows, skipped


def _crop_stem(obj):
    """Cache filename for a regenerated mask.

    Takes a detection (`rel`) or an already-built row (`photo`) -- both
    name the same photo, and the pair must agree or `--resume` would
    regenerate every mask it already has.
    """
    photo = obj.get("rel") or obj.get("photo")
    safe = photo.replace("/", "_").replace(".jpg", "")
    return f"{safe}__{obj.get('detection_index')}"


# ------------------------------------------------------------------
# Split, balance, weights
# ------------------------------------------------------------------

def split_by_photo(rows, args):
    """ONE split over photos, shared by every view.

    Split by crop and a photo's crops land on both sides of the boundary,
    which is leakage of the least visible kind: the val number looks fine
    and the encoder has already seen the background, the person, the
    lighting and usually the garment itself.
    """
    photos = sorted({row["photo"] for row in rows})
    random.Random(args.seed).shuffle(photos)
    n_train = int(len(photos) * args.train_fraction)
    n_val = int(len(photos) * args.val_fraction)
    assignment = {}
    for index, photo in enumerate(photos):
        if index < n_train:
            assignment[photo] = "train"
        elif index < n_train + n_val:
            assignment[photo] = "val"
        else:
            assignment[photo] = "test"
    for row in rows:
        row["split"] = assignment[row["photo"]]
    return assignment


# Views whose `label` is a CLOSED CLASS ("black pants", "sneaker") and so
# can be rebalanced. The `photo` view's label is a free-text outfit
# description -- 2,812 distinct strings over 3,948 photos, 2,345 of them
# singletons -- and its rarity is the whole point of it. Applying a
# min-per-class rule there deletes 97% of the view and leaves six generic
# captions behind, which is a bug, not balancing.
CLASS_VIEWS = {"crop", "detector_category"}


def balance(rows, args):
    """Drop labels too rare to learn, cap labels frequent enough to swamp.

    Applied PER VIEW and only to the closed-class views. Capping happens
    within a split so it cannot move a photo across the boundary, and it
    only ever caps `train` -- a capped val set would be measuring a
    distribution that does not exist.
    """
    dropped = collections.Counter()
    rare = set()
    kept = []
    for view, group in _by_view(rows).items():
        if view not in CLASS_VIEWS:
            kept.extend(group)
            continue
        counts = collections.Counter(row["label"] for row in group)
        view_rare = {label for label, count in counts.items()
                     if count < args.min_per_class}
        rare |= {(view, label) for label in view_rare}
        survivors = []
        for row in group:
            if row["label"] in view_rare:
                dropped[f"{view}/rare_label"] += 1
                continue
            survivors.append(row)

        if args.cap_per_class:
            rng = random.Random(args.seed + 1)
            buckets = collections.defaultdict(list)
            passthrough = []
            for row in survivors:
                (buckets[row["label"]] if row["split"] == "train"
                 else passthrough).append(row)
            capped = []
            for bucket in buckets.values():
                if len(bucket) > args.cap_per_class:
                    rng.shuffle(bucket)
                    dropped[f"{view}/over_cap"] += len(bucket) - args.cap_per_class
                    bucket = bucket[:args.cap_per_class]
                capped.extend(bucket)
            survivors = capped + passthrough
        kept.extend(survivors)

    # Square-root inverse frequency, computed per view on the TRAIN split
    # AFTER capping -- weighting by a distribution the trainer will not
    # see is a quiet way to double-correct. Free-text views get 1.0:
    # there is no class to be imbalanced about.
    text_ids = {text: index for index, text
                in enumerate(sorted({row["text"] for row in kept}))}
    for view, group in _by_view(kept).items():
        train_counts = collections.Counter(row["label"] for row in group
                                           if row["split"] == "train")
        total = sum(train_counts.values()) or 1
        for row in group:
            if view in CLASS_VIEWS:
                count = train_counts.get(row["label"], 1) or 1
                row["sample_weight"] = round((total / count) ** WEIGHT_EXPONENT, 4)
            else:
                row["sample_weight"] = 1.0
            row["text_id"] = text_ids[row["text"]]
            row["label_count_train"] = train_counts.get(row["label"], 0)
    return kept, dropped, rare


def _by_view(rows):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["view"]].append(row)
    return grouped


# ------------------------------------------------------------------
# Optional: regenerate the masked crops that are not on this disk
# ------------------------------------------------------------------

def materialise_crops(rows, args):
    """Re-run the proposer locally to recover masks that live on Modal.

    Free (local CPU/MPS), slow (~4 hours for the corpus), and the only
    part of this script that needs the incremental/elapsed-time
    discipline. Proposals are matched back to stored detections by bbox
    IoU rather than by order, because the proposer sorts by region area
    and a rerun can reorder two regions of near-equal size.
    """
    import numpy as np
    from PIL import Image
    from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

    import garment_proposer

    CROP_CACHE.mkdir(parents=True, exist_ok=True)
    pending = collections.defaultdict(list)
    for row in rows:
        if row.get("image_kind") == "pending_mask":
            pending[row["photo"]].append(row)
    if not pending:
        print("  nothing pending -- every crop already has a mask on disk")
        return 0
    print(f"  {len(pending):,} photos to re-propose ({sum(len(v) for v in pending.values()):,} crops)")
    print(f"  loading proposer + FashionCLIP on {args.device}…")
    processor, model = garment_proposer.load_human_parser(args.device)
    clip_processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    clip_model = AutoModelForZeroShotImageClassification.from_pretrained(
        "patrickjohncyh/fashion-clip").to(args.device)

    def iou(a, b):
        x0, y0 = max(a[0], b[0]), max(a[1], b[1])
        x1, y1 = min(a[2], b[2]), min(a[3], b[3])
        if x1 <= x0 or y1 <= y0:
            return 0.0
        inter = (x1 - x0) * (y1 - y0)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / float(area_a + area_b - inter)

    started = last_checkpoint = time.time()
    written = failed = 0
    for index, (rel, group) in enumerate(sorted(pending.items()), 1):
        try:
            proposals = garment_proposer.propose_garment_items(
                str(REPO_ROOT / rel), processor, model,
                clip_processor, clip_model, args.device)
        except Exception as error:  # noqa: BLE001
            failed += 1
            if failed <= 3:
                print(f"  [warn] proposer {rel}: {error}")
            continue
        for row in group:
            best, best_iou = None, 0.0
            for proposal in proposals:
                box = proposal.get("bbox")
                if not box:
                    continue
                score = iou(row["bbox"], list(box))
                if score > best_iou:
                    best, best_iou = proposal, score
            if best is None or best_iou < args.min_iou:
                continue
            out = CROP_CACHE / f"{_crop_stem(row)}.jpg"
            best["crop"].convert("RGB").save(out, "JPEG", quality=90)
            row["image"] = str(out.relative_to(REPO_ROOT))
            row["image_kind"] = "masked"
            row["mask_iou"] = round(best_iou, 4)
            written += 1
        # ELAPSED TIME, not batch count. Under load a batch-count interval
        # stretches past the interval that kills the job, which is exactly
        # when the checkpoint was needed.
        if time.time() - last_checkpoint >= args.checkpoint_seconds:
            last_checkpoint = time.time()
            elapsed = time.time() - started
            rate = index / elapsed
            print(f"  [{index:,}/{len(pending):,}] {written:,} crops  "
                  f"{elapsed/60:.1f} min  {rate:.2f} photos/s  "
                  f"~{(len(pending)-index)/max(rate, 1e-6)/60:.0f} min left", flush=True)
    print(f"  materialised {written:,} masked crops ({failed:,} proposer failures) "
          f"in {(time.time()-started)/60:.1f} min")
    return written


# ------------------------------------------------------------------
# Reporting / writing
# ------------------------------------------------------------------

def report(rows, dropped, rare, skipped, assignment, args):
    print("\n  --- what would be exported ---" if args.dry_run
          else "\n  --- exported ---")
    by_view = collections.Counter(row["view"] for row in rows)
    by_kind = collections.Counter(row["image_kind"] for row in rows)
    by_split = collections.Counter(row["split"] for row in rows)
    print(f"  rows {len(rows):,}  ·  photos {len({r['photo'] for r in rows}):,}  "
          f"·  distinct texts {len({r['text'] for r in rows}):,}")
    for view, count in by_view.most_common():
        sub = [r for r in rows if r["view"] == view]
        print(f"    view {view:18} {count:7,}  "
              f"{len({r['photo'] for r in sub}):6,} photos  "
              f"{len({r['text'] for r in sub}):6,} texts")
    print(f"  image kind: " + "  ".join(f"{k} {v:,}" for k, v in by_kind.most_common()))
    print(f"  split (by PHOTO, {len(assignment):,} photos): " +
          "  ".join(f"{k} {v:,}" for k, v in sorted(by_split.items())))

    trainable = [r for r in rows if r["image_kind"] in ("photo", "masked")]
    print(f"  TRAINABLE NOW (photo frames + masks on disk): {len(trainable):,}")
    pending = sum(1 for r in rows if r["image_kind"] == "pending_mask")
    if pending:
        print(f"  pending a mask (regenerable free, --materialise-crops): {pending:,}")

    for view, group in sorted(_by_view(rows).items()):
        train = [r for r in group if r["split"] == "train"]
        if not train:
            continue
        counts = collections.Counter(r["label"] for r in train)
        total = sum(counts.values())
        kind = "class" if view in CLASS_VIEWS else "free text"
        print(f"\n  label skew — view '{view}' ({kind}), train split "
              f"({total:,} rows, {len(counts):,} labels):")
        for label, count in counts.most_common(6):
            print(f"    {label[:44]:46} {count:6,}  {count/total:5.1%}")
        print(f"    top-1 {counts.most_common(1)[0][1]/total:5.1%}   "
              f"top-10 {sum(c for _, c in counts.most_common(10))/total:5.1%}   "
              f"singleton labels {sum(1 for c in counts.values() if c == 1):,}")
        weights = [r["sample_weight"] for r in train]
        print(f"    sample_weight {min(weights):.2f} – {max(weights):.2f} "
              f"(inverse frequency ** {WEIGHT_EXPONENT})" if view in CLASS_VIEWS
              else "    sample_weight 1.0 (free text — no class to rebalance)")
        collisions = collections.Counter(r["text_id"] for r in train)
        duplicated = sum(c for c in collisions.values() if c > 1)
        print(f"    rows sharing their text with another train row: {duplicated:,} "
              f"({duplicated/len(train):.1%}) — mask these as in-batch false "
              f"negatives, or use a multi-positive loss")

    if skipped:
        print("\n  skipped while building pairs:")
        for reason, count in skipped.most_common():
            print(f"    {reason:28} {count:7,}")
    if dropped:
        print("  dropped while balancing:")
        for reason, count in dropped.most_common():
            print(f"    {reason:28} {count:7,}")
    if rare:
        print(f"    ({len(rare)} labels under --min-per-class {args.min_per_class})")


def write(rows, args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = set()
    if args.resume and MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                existing.add((row["view"], row["image"], row["text"]))
        print(f"  resuming: {len(existing):,} rows already in {MANIFEST.name}")

    started = last_checkpoint = time.time()
    appended = 0
    mode = "a" if (args.resume and MANIFEST.exists()) else "w"
    with MANIFEST.open(mode) as handle:
        for row in rows:
            if (row["view"], row["image"], row["text"]) in existing:
                continue
            handle.write(json.dumps(row) + "\n")
            appended += 1
            if time.time() - last_checkpoint >= args.checkpoint_seconds:
                handle.flush()
                last_checkpoint = time.time()
                print(f"  [{appended:,}/{len(rows):,}] "
                      f"{(time.time()-started)/60:.1f} min", flush=True)
    return appended


def summarise(rows, args, skipped, dropped):
    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": args.seed,
        "rows": len(rows),
        "photos": len({r["photo"] for r in rows}),
        "distinct_texts": len({r["text"] for r in rows}),
        "by_view": dict(collections.Counter(r["view"] for r in rows)),
        "by_image_kind": dict(collections.Counter(r["image_kind"] for r in rows)),
        "by_split": dict(collections.Counter(r["split"] for r in rows)),
        "trainable_now": sum(1 for r in rows
                             if r["image_kind"] in ("photo", "masked")),
        "pending_mask": sum(1 for r in rows if r["image_kind"] == "pending_mask"),
        "skipped": dict(skipped),
        "dropped": dict(dropped),
        "label_counts_train": dict(collections.Counter(
            r["label"] for r in rows if r["split"] == "train").most_common()),
        "args": {k: v for k, v in vars(args).items()},
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    return summary


# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", choices=("photo", "crop", "both"), default="both",
                        help="photo = full frame <-> outfit caption (the stronger "
                             "signal); crop = masked garment <-> '<colour> <garment>'")
    parser.add_argument("--use-detector-categories", action="store_true",
                        help="also emit FashionCLIP's 13-way category as text. "
                             "Off by default: 13 tokens, 26%% of them 'pants', and "
                             "it is a model's opinion, not a label.")
    parser.add_argument("--allow-bbox-crops", action="store_true",
                        help="fall back to a plain bbox crop when the mask is not "
                             "on disk. Measured 40.9%% vs 72.2%% group agreement — "
                             "rows are tagged image_kind=bbox, never silently mixed.")
    parser.add_argument("--materialise-crops", action="store_true",
                        help="re-run garment_proposer locally (free, ~4h CPU) to "
                             "recover the masks that were written on Modal")
    parser.add_argument("--device", default="cpu",
                        help="proposer device for --materialise-crops (cpu/mps/cuda)")
    parser.add_argument("--min-iou", type=float, default=0.5,
                        help="bbox IoU required to accept a re-proposed mask")
    parser.add_argument("--min-garments", type=int, default=2,
                        help="photo view: drop photos the VLM saw fewer garments in")
    parser.add_argument("--min-per-class", type=int, default=12,
                        help="drop label strings with fewer examples than this "
                             "(matches train_colour_head.py's MIN_PER_CLASS)")
    parser.add_argument("--cap-per-class", type=int, default=0,
                        help="truncate over-represented labels in TRAIN only "
                             "(0 = no cap; sample_weight still corrects)")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-seconds", type=float, default=30.0,
                        help="checkpoint on ELAPSED TIME, not batch count — under "
                             "load a batch interval stretches past the kill interval")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be exported; write nothing")
    args = parser.parse_args()

    print(f"  reading {OUTFIT_METADATA.name}, {VLM_LABELS.name}…")
    detections = load_detections()
    vlm = load_vlm()
    on_disk = sum(1 for d in detections if d["crop_on_disk"])
    print(f"  {len(detections):,} detections  ·  "
          f"{sum(1 for d in detections if d['photo_on_disk']):,} with the photo on disk  ·  "
          f"{on_disk:,} with a MASKED CROP on disk")
    print(f"  {len(vlm):,} VLM-labelled photos, "
          f"{sum(1 for v in vlm.values() if v.get('is_outfit_photo')):,} judged real outfits")

    rows, skipped = [], collections.Counter()
    if args.view in ("photo", "both"):
        new, skip = photo_pairs(vlm, detections, args)
        rows += new
        skipped.update({f"photo/{k}": v for k, v in skip.items()})
    if args.view in ("crop", "both"):
        new, skip = crop_pairs(vlm, detections, args)
        rows += new
        skipped.update({f"crop/{k}": v for k, v in skip.items()})
    if args.use_detector_categories:
        new, skip = detector_category_pairs(detections, args)
        rows += new
        skipped.update({f"category/{k}": v for k, v in skip.items()})

    if not rows:
        raise SystemExit("no pairs built -- nothing to export")

    if args.materialise_crops:
        if args.dry_run:
            pending = sum(1 for r in rows if r["image_kind"] == "pending_mask")
            print(f"  --dry-run: would re-propose masks for {pending:,} crops "
                  f"over {len({r['photo'] for r in rows if r['image_kind']=='pending_mask'}):,} photos")
        else:
            materialise_crops(rows, args)

    assignment = split_by_photo(rows, args)
    rows, dropped, rare = balance(rows, args)
    report(rows, dropped, rare, skipped, assignment, args)

    if args.dry_run:
        print(f"\n  --dry-run: nothing written. Would write {MANIFEST} "
              f"and {SUMMARY}.")
        return
    appended = write(rows, args)
    summarise(rows, args, skipped, dropped)
    print(f"\n  wrote {appended:,} rows -> {MANIFEST}")
    print(f"  summary -> {SUMMARY}")


if __name__ == "__main__":
    main()
