"""Search real people's outfit photos with a picture and/or a text query.

    .venv/bin/python outfit_search.py build   # embed the 9,999 photos (once)
    .venv/bin/python outfit_search.py serve   # browse at http://localhost:7880
    .venv/bin/python outfit_search.py search --text "baggy jeans" --image jacket.jpg

**Use `.venv/bin/python`, not `python3`.** This machine's system Python
3.10 carries torch 2.0.0, which current `transformers` refuses to use --
it disables the PyTorch backend and then fails deep inside
`AutoModel.from_pretrained` with "requires the PyTorch library but it was
not found", which does not look like a wrong-interpreter error at all.
`.venv` (Python 3.14) has the working torch. `check_environment()` below
turns that into a one-line message.

## What this is, and how it differs from everything else in this repo

Every retrieval surface built here so far answers **"what IS this item?"** --
`/identify` returns catalog products, `/compose` returns catalog products,
`/search` returns catalog products. This one answers a different question:
**"show me outfits like this."** The result set is real worn photos from
`outfit_dataset` (6,860 posts / 9,999 images), not catalog SKUs.

That difference is not cosmetic; it changes which encoder is correct.
`docs/unified_query_design.md` sets out the tension:

| query | wants a metric where... |
|---|---|
| "show me blue jeans" | all blue jeans are NEAR each other |
| "show me *this exact* sneaker" | two colorways of one shoe are FAR apart |

The identify path needs the second geometry, which is why DINOv3's
identity fine-tune bought +31.3pt there. **Outfit search needs the
first.** So this module uses SigLIP2 -- the semantic encoder -- and does
not touch DINOv3 at all.

## Why blending image and text here is NOT the -6.2pt score fusion

Score fusion lost 6.22pt R@1 by averaging a SigLIP2 score with a DINOv3
score: two different models, two different geometries, one of them
identity-level so every colorway sibling got an identical score. That
mechanism does not apply here. SigLIP2 is natively a joint image-text
model: its image tower and text tower are trained contrastively into
**one shared space**. Adding a text vector to an image vector inside that
space is the operation the model was built for, not a blend of two
incompatible rankings.

It is still a baseline, not a solved problem -- composed image retrieval
is an open research area and a weighted sum is the simplest thing that
can work. It is exposed as a slider rather than a constant precisely
because the right weight is a matter of what the user meant, and that is
not something to guess.

## The honest limits, up front

- **The encoder is out of its training distribution.** SigLIP2 v3 was
  fine-tuned on catalog product photos paired with product captions. These
  are real photos of people in rooms and on streets. It transfers, but
  nothing here has measured how well.
- **Nothing about this is evaluated yet.** There is no ground truth for
  "is this a good outfit match," and no eval row. Judge it by looking.
- **The detected-garment filter uses UNVALIDATED labels.** 20,681
  detections from a human parser plus zero-shot FashionCLIP over an
  unlabelled corpus. Precision was eyeballed at ~91% on 40 photos, on the
  same 40 the threshold was picked on. Evidence of what the model saw, not
  measured fact -- so the filter is off unless the text actually names a
  category the detector knows.
- **These are photographs of real people**, collected under varying terms,
  and `docs/licensing_review.md` is explicit that we have provenance but
  no permission. Fine for local development. Displaying them in a shipped
  product is a different question, and 1,342 Pinterest records have no
  author recorded at all.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OUTFIT_METADATA = REPO_ROOT / "outfit_dataset" / "metadata.json"
# Two indexes, because the query has two halves that want different
# granularity. See `rank` for how they combine.
PHOTO_INDEX_PATH = REPO_ROOT / "outfit_dataset" / "outfit_search_index.pt"
CROP_INDEX_PATH = REPO_ROOT / "outfit_dataset" / "outfit_crop_index.pt"

# `free_text_visual_search` (and the pipeline) resolve DATASET_ROOT at
# IMPORT time and default to a Colab Drive path, then mkdir it -- which on
# this machine dies with "Read-only file system: '/content'" before a
# single line of this module runs. Set it before any such import, unless
# the caller already has.
os.environ.setdefault("APPAREL_DATASET_ROOT", str(REPO_ROOT / "apparel_dataset"))


# The palette index_outfits.dominant_color votes against. Reused here so a
# named colour can be matched perceptually (CIELAB) as well as by name.
COLOUR_PALETTE_RGB = {
    "black": (25, 25, 27), "charcoal": (65, 65, 68), "gray": (128, 128, 130),
    "light gray": (190, 190, 192), "white": (240, 240, 240),
    "cream": (238, 226, 200), "beige": (214, 196, 166), "brown": (110, 78, 52),
    "tan": (170, 132, 88), "olive": (110, 112, 62), "green": (60, 130, 70),
    "navy": (36, 46, 82), "blue": (60, 100, 180), "light blue": (145, 180, 220),
    "purple": (110, 70, 150), "red": (170, 45, 45), "pink": (225, 150, 165),
    "orange": (215, 125, 50), "yellow": (225, 200, 70),
}


# ----------------------------------------------------------------------
# index
# ----------------------------------------------------------------------

def load_image_records():
    """One record per IMAGE (not per post), carrying its post's detections.

    Indexing per image because two photos of one outfit are genuinely
    different views and either may be the better match. Results are
    collapsed back to one row per post at query time so the grid does not
    fill up with four shots of the same person."""
    records = json.loads(OUTFIT_METADATA.read_text())
    out = []
    for record in records:
        # Detections are stored per post with a `source_image` pointer, so
        # group them by image rather than attaching all of a post's items
        # to every one of its photos.
        by_image = {}
        for item in record.get("detected_items") or []:
            by_image.setdefault(item.get("source_image"), []).append(item)

        for path in record.get("images") or []:
            full = REPO_ROOT / path
            if not full.exists():
                continue
            items = by_image.get(path, [])
            out.append({
                "path": str(full),
                "rel": path,
                "post_id": f"{record.get('source')}:{record.get('source_id')}",
                "source": record.get("source"),
                "section": record.get("section"),
                "post_url": record.get("post_url"),
                "title": record.get("title"),
                "author": record.get("author"),
                "categories": sorted({i.get("category") for i in items if i.get("category")}),
                "groups": sorted({i.get("category_group") for i in items if i.get("category_group")}),
                "colors": sorted({(i.get("color") or {}).get("name")
                                  for i in items if (i.get("color") or {}).get("name")}),
            })
    return out


def load_crop_records():
    """One record per DETECTED GARMENT, cut from its source photo's bbox.

    This is the index the item half of the query actually needs. Matching a
    product photo of a jacket against a full-frame photo of a person is a
    scale and content mismatch -- most of the target frame is face, legs,
    street and sky. Scoring against the jacket REGION is the comparison the
    question is asking for.

    Uses the stored bbox rather than the proposer's own masked crop,
    because `crop_path` points at files that were written on Modal during
    the corpus re-detection and only 13 of 20,681 exist locally.
    Re-running `garment_proposer` over the whole corpus to regenerate them
    is ~4 hours on this machine's CPU; the bboxes are already here and
    cost nothing.

    The honest difference: the proposer blanks everything outside the
    garment INSIDE its bbox, and a plain bbox crop does not, so these
    carry some background the shipped crops would not. That is a real
    (unmeasured) difference between this index and what `/identify`
    consumes, and it is why this is recorded here rather than glossed.
    """
    records = json.loads(OUTFIT_METADATA.read_text())
    out = []
    for record in records:
        post_id = f"{record.get('source')}:{record.get('source_id')}"
        for order, item in enumerate(record.get("detected_items") or []):
            source_image = item.get("source_image")
            bbox = item.get("bbox")
            if not source_image or not bbox:
                continue
            full = REPO_ROOT / source_image
            if not full.exists():
                continue
            out.append({
                "path": str(full),
                "rel": source_image,
                "bbox": bbox,
                "post_id": post_id,
                "source": record.get("source"),
                "post_url": record.get("post_url"),
                "category": item.get("category"),
                "category_group": item.get("category_group"),
                "color": (item.get("color") or {}).get("name"),
                "detection_index": order,
            })
    # Group by source photo before encoding. A post's detections interleave
    # its images (img0_item1, img1_item1, img0_item2, ...), so in stored
    # order the source photo changes between 36.8% of consecutive crops and
    # the single-entry decode cache misses on every one of those. Sorting
    # makes the cache hit for every crop after the first of each photo --
    # 31,239 decodes become 16,330.
    out.sort(key=lambda r: r["path"])
    return out


def encode_crops(model, processor, records, fts, torch, checkpoint_path=None,
                 checkpoint_seconds=120):
    """Mirror of `free_text_visual_search.encode_images`, but cropping to
    each record's bbox first. Kept local rather than generalising that
    function, because it is imported by the serving path and this is not
    the moment to change something `/search` depends on."""
    import time

    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from tqdm import tqdm

    from concurrent.futures import ThreadPoolExecutor

    batch_size = fts.IMAGE_BATCH_SIZE
    embeddings, kept = [], []
    # Open each source photo once even though several crops share it --
    # decoding a 1536px JPEG per crop would dominate the runtime. Records
    # arrive sorted by path (see load_crop_records) so this single entry is
    # enough on LOCAL disk.
    #
    # On a NETWORK filesystem (Modal volume) the bottleneck is not decoding
    # but the per-file round trip: measured 0.46 s/image there, with the
    # GPU idle. So each batch's distinct source photos are opened in
    # parallel first, then cropped from memory. IMAGE_LOADER_WORKERS is 1
    # locally, 32 on Modal.
    workers = fts.IMAGE_LOADER_WORKERS
    cache_path, cache_image = None, None

    def _open(path):
        try:
            with Image.open(path) as handle:
                return path, ImageOps.exif_transpose(handle).convert("RGB")
        except Exception:  # noqa: BLE001
            return path, None

    # Resume support. The first full-corpus rebuild ran 9h36m and was killed
    # at 82% with nothing saved, losing all of it. A partial index is worth
    # far more than a pristine one that may never finish on a laptop.
    done = 0
    last_save = time.time()
    if checkpoint_path and Path(checkpoint_path).exists():
        partial = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if partial.get("total") == len(records):
            embeddings = [partial["embeddings"].float()]
            kept = partial["records"]
            done = partial["done"]
            print(f"  resuming from checkpoint at {done:,}/{len(records):,}")
        else:
            print("  checkpoint is for a different record set — starting over")

    for start in tqdm(range(done, len(records), batch_size), desc="Encoding garment crops"):
        batch = records[start:start + batch_size]
        images, batch_kept = [], []
        opened = {}
        if workers > 1:
            wanted = {r["path"] for r in batch if r["path"] != cache_path}
            if wanted:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    opened = {p: im for p, im in pool.map(_open, sorted(wanted))
                              if im is not None}
        for record in batch:
            try:
                if record["path"] != cache_path:
                    if record["path"] in opened:
                        cache_image = opened[record["path"]]
                    else:
                        with Image.open(record["path"]) as handle:
                            cache_image = ImageOps.exif_transpose(handle).convert("RGB")
                    cache_path = record["path"]
                left, top, right, bottom = record["bbox"]
                if right - left < 8 or bottom - top < 8:
                    continue
                images.append(cache_image.crop((left, top, right, bottom)))
                batch_kept.append(record)
            except Exception:
                continue
        if not images:
            continue
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(fts.DEVICE) for k, v in inputs.items()}
        with fts.autocast_context():
            output = fts.extract_embeddings(model.get_image_features(**inputs)).float()
        embeddings.append(F.normalize(output, dim=-1).cpu())
        kept.extend(batch_kept)

        # Checkpoint on ELAPSED TIME, not batch count. Batch time here
        # varies by more than an order of magnitude with what else the
        # machine is doing (6s/batch idle, 23-86s/batch against the ANE
        # compiler and Spotlight), so a fixed batch interval was 25 minutes
        # apart under load -- longer than the ~10 minutes between the kills
        # it was meant to survive, so nothing ever accumulated.
        if checkpoint_path and embeddings and (time.time() - last_save) > checkpoint_seconds:
            merged = torch.cat(embeddings, dim=0)
            torch.save({"embeddings": merged.half(), "records": kept,
                        "done": start + batch_size, "total": len(records)},
                       checkpoint_path)
            # Collapse to one tensor so the list does not grow unbounded
            # and every checkpoint does not re-concatenate hundreds of
            # small tensors.
            embeddings = [merged]
            last_save = time.time()

    return torch.cat(embeddings, dim=0), kept


def build(args):
    import torch

    sys.path.insert(0, str(REPO_ROOT))
    import free_text_visual_search as fts
    from transformers import AutoModel, AutoProcessor

    checkpoint = fts.pick_checkpoint()
    load_from = str(checkpoint) if checkpoint else fts.BASE_MODEL_ID
    print(f"  SigLIP2: {load_from}"
          f"{'' if checkpoint else '  (no fine-tune found — base model)'}")
    print(f"  device : {fts.DEVICE}")
    processor = AutoProcessor.from_pretrained(load_from)
    model = AutoModel.from_pretrained(load_from).to(fts.DEVICE).eval()

    if args.index in ("photos", "both"):
        records = load_image_records()
        if args.limit:
            records = records[:args.limit]
        print(f"\n  {len(records):,} full photos to encode")
        # Reuses free_text_visual_search's encoder verbatim rather than a
        # second copy: it already handles the EXIF transpose, the RGB
        # convert, the per-version embedding unwrap and the L2 normalise.
        with torch.inference_mode():
            embeddings, kept = fts.encode_images(model, processor, records)
        torch.save({"embeddings": embeddings.half(), "records": kept,
                    "checkpoint": load_from}, PHOTO_INDEX_PATH)
        print(f"  wrote {len(kept):,} photo vectors "
              f"({PHOTO_INDEX_PATH.stat().st_size / 1e6:.0f} MB)")

    if args.index in ("crops", "both"):
        records = load_crop_records()
        if args.limit:
            records = records[:args.limit]

        # INCREMENTAL: reuse vectors for crops already in the index and
        # encode only what is new. A crop's embedding depends on its source
        # image and its bbox, neither of which changes once detected, so a
        # cached vector stays valid -- the same reasoning the catalog
        # identity index already uses.
        #
        # This matters because the machine is not always free: a
        # full-corpus rebuild ran 9h36m and was killed at 82%, and a retry
        # crawled at 40-70s/batch against an ANE compiler and Spotlight
        # using the same silicon. After tonight's detection pass, 20,681 of
        # 31,239 crops were already indexed, so this turns a ~2h job into
        # a ~40min one.
        reused_embeddings, reused_records = None, []
        if not args.rebuild and CROP_INDEX_PATH.exists():
            cached = torch.load(CROP_INDEX_PATH, map_location="cpu", weights_only=False)
            if cached.get("checkpoint") == load_from:
                index = {(r["rel"], r["detection_index"]): row
                         for row, r in enumerate(cached["records"])}
                keep_rows = [index[(r["rel"], r["detection_index"])]
                             for r in records
                             if (r["rel"], r["detection_index"]) in index]
                if keep_rows:
                    reused_embeddings = cached["embeddings"][keep_rows].float()
                    reused_records = [cached["records"][row] for row in keep_rows]
                    have = {(r["rel"], r["detection_index"]) for r in reused_records}
                    records = [r for r in records
                               if (r["rel"], r["detection_index"]) not in have]
                    print(f"\n  reusing {len(reused_records):,} cached crop vectors")
            else:
                print("\n  checkpoint changed — rebuilding every crop vector")

        print(f"  {len(records):,} garment crops to encode")
        if records:
            with torch.inference_mode():
                embeddings, kept = encode_crops(
                    model, processor, records, fts, torch,
                    checkpoint_path=CROP_INDEX_PATH.with_suffix(".partial.pt"))
        else:
            embeddings, kept = None, []

        # Concatenate reused and freshly-encoded vectors. Order does not
        # matter -- every lookup is by row index into `records`.
        if reused_embeddings is not None and embeddings is not None:
            embeddings = torch.cat([reused_embeddings, embeddings], dim=0)
            kept = reused_records + kept
        elif reused_embeddings is not None:
            embeddings, kept = reused_embeddings, reused_records

        torch.save({"embeddings": embeddings.half(), "records": kept,
                    "checkpoint": load_from}, CROP_INDEX_PATH)
        CROP_INDEX_PATH.with_suffix(".partial.pt").unlink(missing_ok=True)
        print(f"  wrote {len(kept):,} crop vectors "
              f"({CROP_INDEX_PATH.stat().st_size / 1e6:.0f} MB)")


# ----------------------------------------------------------------------
# search
# ----------------------------------------------------------------------

class OutfitSearch:
    def __init__(self):
        import numpy as np
        import torch

        sys.path.insert(0, str(REPO_ROOT))
        import free_text_visual_search as fts
        from transformers import AutoModel, AutoProcessor

        if not PHOTO_INDEX_PATH.exists():
            raise SystemExit(f"{PHOTO_INDEX_PATH} not found — run `outfit_search.py build` first")
        self.torch, self.np, self.fts = torch, np, fts

        photos = torch.load(PHOTO_INDEX_PATH, map_location="cpu", weights_only=False)
        self.photo_embeddings = photos["embeddings"].float()
        self.photo_records = photos["records"]
        self.checkpoint = photos["checkpoint"]

        # Crops are optional: without them the item half degrades to
        # whole-frame similarity, which still returns something sane. A
        # hard requirement here would mean a 50-minute build before the
        # first query could run at all.
        self.crop_embeddings, self.crop_records = None, []
        if CROP_INDEX_PATH.exists():
            crops = torch.load(CROP_INDEX_PATH, map_location="cpu", weights_only=False)
            self.crop_embeddings = crops["embeddings"].float()
            self.crop_records = crops["records"]

        self.processor = AutoProcessor.from_pretrained(self.checkpoint)
        self.model = AutoModel.from_pretrained(self.checkpoint).to(fts.DEVICE).eval()

        # `section` (subreddit / query URL) is joined from metadata at load
        # time rather than read from the index, because indexes built
        # before the field existed carry it as None -- and a source filter
        # that silently matches nothing is worse than one that is absent.
        # The join is a dict build over 6,860 records; it costs nothing and
        # it means the filter works without a 35-minute re-encode.
        try:
            sections = {}
            for record in json.loads(OUTFIT_METADATA.read_text()):
                sections[f"{record.get('source')}:{record.get('source_id')}"] = \
                    record.get("section")
            missing = 0
            for record in self.photo_records:
                if not record.get("section"):
                    record["section"] = sections.get(record["post_id"])
                    missing += record["section"] is None
            if missing:
                print(f"  ({missing:,} photos have no section — source filters "
                      f"cannot exclude them)")
        except (OSError, json.JSONDecodeError) as error:
            print(f"  (section join skipped: {error}; source filters degraded)")

        # Garment group per crop row, as a numpy array so the type
        # preference is one vectorised comparison rather than a per-row
        # dict lookup inside the scoring loop.
        self._crop_groups = None
        if self.crop_records:
            self._crop_groups = np.array(
                [self.DETECTOR_GROUP.get(r.get("category"), "") for r in self.crop_records])
            self._crop_colors = np.array(
                [r.get("color") or "" for r in self.crop_records])
            self._crop_categories = np.array(
                [r.get("category") or "" for r in self.crop_records])

            # Learned colours, if a distilled head has been trained. It
            # replaces the palette vote's name for matching purposes --
            # measured +7.2pt over the heuristic on held-out crops, at zero
            # marginal cost, because it is one matmul over embeddings that
            # already exist. Falls back silently to the heuristic name when
            # no head is present.
            self._crop_colors_learned = None
            head_path = REPO_ROOT / "outfit_dataset" / "colour_head.pt"
            if head_path.exists():
                try:
                    payload = self.torch.load(head_path, map_location="cpu",
                                              weights_only=False)
                    model = self.torch.nn.Sequential(
                        self.torch.nn.Linear(payload["dim"], payload["hidden"]),
                        self.torch.nn.ReLU(),
                        self.torch.nn.Dropout(0.0),
                        self.torch.nn.Linear(payload["hidden"], len(payload["classes"])))
                    model.load_state_dict(payload["state_dict"])
                    model.eval()
                    with self.torch.no_grad():
                        predicted = model(self.crop_embeddings).argmax(1).numpy()
                    names = payload["classes"]
                    self._crop_colors_learned = np.array([names[i] for i in predicted])
                    metrics = payload.get("metrics", {})
                    print(f"  colour head: {len(names)} colours, held-out "
                          f"{metrics.get('head', 0):.1%} vs heuristic "
                          f"{metrics.get('heuristic', 0):.1%}")
                except Exception as error:  # noqa: BLE001
                    print(f"  (colour head not loaded: {error})")
            # Per-crop mean RGB, joined from metadata like `section`: the
            # crop index predates the field, and rebuilding 31,239 vectors
            # to add a colour column would be hours of GPU for data that
            # is already on disk.
            self._crop_lab = None
            try:
                rgb_by_key = {}
                for record in json.loads(OUTFIT_METADATA.read_text()):
                    for order, item in enumerate(record.get("detected_items") or []):
                        mean_rgb = (item.get("color") or {}).get("mean_rgb")
                        if mean_rgb:
                            rgb_by_key[(item.get("source_image"), order)] = mean_rgb
                rgb = [rgb_by_key.get((r["rel"], r["detection_index"])) for r in self.crop_records]
                found = sum(v is not None for v in rgb)
                if found:
                    filled = np.array([v if v is not None else [0, 0, 0] for v in rgb],
                                      dtype=float)
                    self._crop_lab = self._srgb_to_lab(filled)
                    self._crop_has_rgb = np.array([v is not None for v in rgb])
                    if found < len(rgb):
                        print(f"  ({len(rgb) - found:,} crops have no mean_rgb)")
            except (OSError, json.JSONDecodeError) as error:
                print(f"  (crop colour join skipped: {error})")

        # Skin tone per POST, joined from metadata. Relative only -- see
        # extract_skin_tone.py: absolute Monk binning is measurably broken
        # on photographed skin (the light half of the scale is unreachable
        # at these exposures), so the filter compares a photo against a
        # REFERENCE photo, where the shared lighting bias largely cancels.
        self._skin_lab, self._skin_conf = {}, {}
        try:
            for record in json.loads(OUTFIT_METADATA.read_text()):
                tone = record.get("skin_tone") or {}
                if tone.get("lab"):
                    key = f"{record.get('source')}:{record.get('source_id')}"
                    self._skin_lab[key] = tone["lab"]
                    self._skin_conf[key] = tone.get("confidence", 0.0)
        except (OSError, json.JSONDecodeError):
            pass

        # post_id -> row indexes, so a photo's score can be computed from
        # its own garments without scanning the whole crop table per photo.
        self.crops_by_post = {}
        for row, record in enumerate(self.crop_records):
            self.crops_by_post.setdefault(record["post_id"], []).append(row)
        # post_id -> the best (first) photo row to display for that post.
        self.photo_rows_by_post = {}
        for row, record in enumerate(self.photo_records):
            self.photo_rows_by_post.setdefault(record["post_id"], []).append(row)

        # Vocabulary the DETECTOR actually emits. The filter can only be as
        # good as these labels, so it is keyed off them rather than off the
        # catalog taxonomy -- offering a filter for a term no detection
        # carries would return an empty grid and look like a bug.
        self.category_vocab = sorted({r["category"] for r in self.crop_records if r.get("category")}
                                     or {c for r in self.photo_records for c in r["categories"]})
        self.color_vocab = sorted({r["color"] for r in self.crop_records if r.get("color")}
                                  or {c for r in self.photo_records for c in r["colors"]})

    def encode_text(self, text):
        with self.torch.inference_mode():
            return self.fts.encode_text(self.model, self.processor, [text])[0]

    def encode_image(self, path):
        from PIL import Image, ImageOps

        with self.torch.inference_mode():
            with Image.open(path) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
            inputs = self.processor(images=[rgb], return_tensors="pt")
            inputs = {k: v.to(self.fts.DEVICE) for k, v in inputs.items()}
            with self.fts.autocast_context():
                output = self.fts.extract_embeddings(
                    self.model.get_image_features(**inputs)).float()
            return self.torch.nn.functional.normalize(output, dim=-1)[0].cpu()

    # Sections that cannot match the catalog, measured 2026-08-06.
    #
    # The catalog is 18 US brands and **100% men's clothing** -- every
    # scraper targeted men's categories. The outfit corpus is not: 22% is
    # Japan/Korea-sourced (wear.jp, "korean street fashion") and 20% is
    # women's fashion. That ~42% cannot match a men's US catalog no matter
    # how good the encoder gets, so it is worth being able to exclude.
    #
    # Matched on `section` (subreddit or query URL), which every record
    # carries. Deliberately a display filter rather than a deletion: these
    # are real outfit photos and still useful for a womenswear or JP
    # catalog later. Off by default -- narrowing a corpus is the caller's
    # decision, not a silent one.
    NON_US_SECTIONS = ("wear.jp", "korean")
    WOMENS_SECTIONS = ("femalefashion", "femalefashionadvice",
                       "petitefashionadvice", "womens%20outfit", "womens outfit")

    def _section_excluded(self, record, drop_non_us, drop_womens):
        section = str(record.get("section") or "").lower()
        source = str(record.get("source") or "").lower()
        if drop_non_us and (source == "wear"
                            or any(k in section for k in self.NON_US_SECTIONS)):
            return True
        if drop_womens and any(k in section for k in self.WOMENS_SECTIONS):
            return True
        return False

    # Zero-shot garment-group prototypes for the QUERY image. Averaged over
    # several phrasings per group because a single prompt is noisy.
    GROUP_PROMPTS = {
        "tops": ["a t-shirt", "a shirt", "a sweater", "a hoodie", "a sweatshirt",
                 "a tank top"],
        "bottoms": ["a pair of pants", "a pair of jeans", "a pair of shorts",
                    "trousers"],
        "outerwear": ["a jacket", "a coat", "an overshirt", "a puffer jacket"],
        "footwear": ["a sneaker", "a shoe", "a boot", "a loafer"],
        "accessories": ["a hat", "a cap", "a bag", "a belt", "a pair of socks"],
    }
    DETECTOR_GROUP = {
        "sneaker": "footwear", "loafer": "footwear", "pants": "bottoms",
        "shorts": "bottoms", "jacket": "outerwear", "hat": "accessories",
        "socks": "accessories", "shirt": "tops", "t-shirt": "tops",
        "sweater": "tops", "hoodie": "tops", "sweatshirt": "tops",
        "tank top": "tops",
    }

    def _group_prototypes(self):
        """Built once, lazily — five groups x ~5 prompts is 25 text encodes."""
        if getattr(self, "_protos", None) is None:
            names, vectors = [], []
            for group, prompts in self.GROUP_PROMPTS.items():
                stacked = self.torch.stack(
                    [self.encode_text(f"a photo of {p}") for p in prompts])
                vectors.append(self.torch.nn.functional.normalize(
                    stacked.mean(0), dim=-1))
                names.append(group)
            self._protos = (names, self.torch.stack(vectors))
        return self._protos

    def classify_group(self, vector):
        """Which garment group is this query image? Zero-shot, SigLIP2.

        Measured 88.0% on 150 catalog images with a known canonical group
        (outerwear 100%, accessories 94%, bottoms 85%, tops 84%).

        88% is why this is a PREFERENCE and not a filter. Restricting the
        candidate crops to the predicted group would give the 12% of
        misclassified queries nothing but wrong-type crops -- a total
        failure on those, in exchange for fixing partial failures
        elsewhere. That is also the shape of the mistake `--category-gate`
        already made on the identification task, where it measured
        net-negative on seven independent runs."""
        names, protos = self._group_prototypes()
        scores = (protos @ vector)
        best = int(scores.argmax())
        return names[best], float(scores[best])

    # Words people type that name a garment GROUP but are not taxonomy
    # leaves, so `parse_text_fragment` finds nothing for them. "shoes" is
    # the one that started this: `parse_filters("yellow shoes")` returned
    # no category at all, so nothing constrained the garment type and the
    # top-20 crops came back 8 footwear / 12 jackets, hats, pants and tees.
    GROUP_WORDS = {
        "shoe": "footwear", "shoes": "footwear", "sneaker": "footwear",
        "sneakers": "footwear", "trainers": "footwear", "kicks": "footwear",
        "boot": "footwear", "boots": "footwear", "loafer": "footwear",
        "loafers": "footwear", "sandal": "footwear", "sandals": "footwear",
        "top": "tops", "tops": "tops", "shirt": "tops", "shirts": "tops",
        "tee": "tops", "tees": "tops", "t-shirt": "tops", "sweater": "tops",
        "hoodie": "tops", "sweatshirt": "tops", "jumper": "tops",
        "pants": "bottoms", "trousers": "bottoms", "jeans": "bottoms",
        "denim": "bottoms", "shorts": "bottoms", "bottoms": "bottoms",
        "joggers": "bottoms", "sweatpants": "bottoms", "chinos": "bottoms",
        "jacket": "outerwear", "coat": "outerwear", "outerwear": "outerwear",
        "puffer": "outerwear", "parka": "outerwear", "windbreaker": "outerwear",
        "hat": "accessories", "cap": "accessories", "beanie": "accessories",
        "bag": "accessories", "belt": "accessories", "socks": "accessories",
    }

    @staticmethod
    def _srgb_to_lab(rgb):
        """sRGB (0-255, shape (...,3)) -> CIELAB. Pure numpy, D65.

        Colour distance has to be perceptual. In RGB, navy and black are
        further apart than two visibly different mid-greys, so an RGB
        threshold either floods the results or misses obvious matches.
        The catalog colour pipeline already made this call for the same
        reason (`build_color_index.py` uses CIELAB + CIEDE2000)."""
        import numpy as np

        srgb = np.asarray(rgb, dtype=float) / 255.0
        linear = np.where(srgb <= 0.04045, srgb / 12.92,
                          ((srgb + 0.055) / 1.055) ** 2.4)
        matrix = np.array([[0.4124, 0.3576, 0.1805],
                           [0.2126, 0.7152, 0.0722],
                           [0.0193, 0.1192, 0.9505]])
        xyz = linear @ matrix.T
        white = np.array([0.95047, 1.0, 1.08883])
        scaled = xyz / white
        epsilon = 216 / 24389
        kappa = 24389 / 27
        f = np.where(scaled > epsilon, np.cbrt(scaled),
                     (kappa * scaled + 16) / 116)
        return np.stack([116 * f[..., 1] - 16,
                         500 * (f[..., 0] - f[..., 1]),
                         200 * (f[..., 1] - f[..., 2])], axis=-1)

    def colour_similarity(self, rgb, falloff=25.0):
        """Per-crop closeness to a target colour, 1.0 at exact, ->0 far away.

        Continuous rather than a hard radius, so "that exact colour" ranks
        the closest crops first instead of returning an unordered bucket of
        everything inside a threshold. `falloff` is in CIELAB delta-E,
        where ~2.3 is the just-noticeable difference and 25 is comfortably
        'the same colour family'."""
        import numpy as np

        if self._crop_lab is None:
            return None
        target = self._srgb_to_lab(np.asarray(rgb, dtype=float))
        delta = np.linalg.norm(self._crop_lab - target, axis=-1)
        return np.exp(-(delta / falloff) ** 2)

    # Words that separate one garment from the next in a typed sentence.
    CLAUSE_SPLIT = re.compile(
        r"\b(?:with|and|plus|over|under|paired with|worn with)\b|[,+&/]")

    def split_constraints(self, text):
        """One sentence -> one constraint per garment mentioned.

        "jorts with red shoes" names TWO garments, and parsing it as a
        single (group, colour) pair silently dropped one of them: the
        parse returned ('footwear', 'red') and nothing whatsoever required
        the photo to contain jorts, so the results were red-shoe photos
        whose other red items were incidental. That is exactly what the
        owner saw -- "a lot of stuff are definitely red, but they are the
        shirts mostly".

        Each clause becomes its own query part, gets its own text
        embedding, and claims its own crop -- so the photo has to satisfy
        all of them at once, which is what "A with B" means.

        Returns [(phrase, group, colour), ...] with at least one entry.
        """
        raw = [fragment.strip() for fragment in self.CLAUSE_SPLIT.split(text or "")]
        fragments = [f for f in raw if f]
        parsed = []
        for fragment in fragments:
            group, colour, category = self.parse_text_attributes(fragment)
            if group or colour:
                parsed.append((fragment, group, colour, category))
        # Only split when the clauses genuinely name different garments;
        # "a black and white jacket" must stay one constraint.
        groups = [g for _, g, _, _ in parsed if g]
        if len(parsed) > 1 and len(set(groups)) > 1:
            return parsed
        group, colour, category = self.parse_text_attributes(text)
        return [(text, group, colour, category)]

    # Word -> the DETECTOR'S OWN category, one level finer than group.
    #
    # Group-level binding cannot separate "red sweater" from "red hoodie":
    # sweater, hoodie, t-shirt, shirt, sweatshirt and tank top are all the
    # group `tops`, so every one of them earned the same bonus. This is
    # the level at which those differ.
    CATEGORY_WORDS = {
        "sweater": "sweater", "jumper": "sweater", "knit": "sweater",
        "cardigan": "sweater",
        "hoodie": "hoodie", "hooded": "hoodie",
        "sweatshirt": "sweatshirt", "crewneck": "sweatshirt",
        "t-shirt": "t-shirt", "tshirt": "t-shirt", "tee": "t-shirt",
        "shirt": "shirt", "button-up": "shirt", "button-down": "shirt",
        "oxford": "shirt", "flannel": "shirt",
        "tank": "tank top", "tank top": "tank top", "vest": "tank top",
        "jacket": "jacket", "coat": "jacket", "puffer": "jacket",
        "parka": "jacket", "windbreaker": "jacket", "bomber": "jacket",
        "pants": "pants", "trousers": "pants", "jeans": "pants",
        "chinos": "pants", "joggers": "pants", "sweatpants": "pants",
        "shorts": "shorts", "jorts": "shorts",
        "sneaker": "sneaker", "sneakers": "sneaker", "trainers": "sneaker",
        "kicks": "sneaker",
        "loafer": "loafer", "loafers": "loafer",
        "hat": "hat", "cap": "hat", "beanie": "hat",
        "socks": "socks", "sock": "socks",
    }

    def parse_text_attributes(self, text):
        """Pull a garment GROUP and a COLOUR out of one text part.

        This is what makes "yellow shoes" mean yellow shoes. SigLIP2 does
        not bind attributes to objects reliably -- a well-known
        CLIP-family weakness, and measured here: the top-20 crops for
        "yellow shoes" were 8 footwear and 3 actually yellow, with yellow
        jackets and yellow hats scoring above black sneakers. The text
        embedding encodes "yellow" and "shoe-ish" without tying them
        together.

        Binding them structurally -- reward the crop that is BOTH -- is
        cheaper and more reliable than hoping the encoder does it."""
        lowered = (text or "").lower()
        # In READING order. This iterated a set, so which garment won was
        # decided by hash order rather than by the sentence.
        words = re.findall(r"[a-z][a-z-]*", lowered)

        group = None
        for word in words:
            if word in self.GROUP_WORDS:
                group = self.GROUP_WORDS[word]
                break
        if group is None:
            # Fall back to the real taxonomy, which knows leaves and slang
            # ("jorts") that the flat table above does not.
            categories, _ = self.parse_filters(text)
            if categories:
                group = self.DETECTOR_GROUP.get(categories[0])

        # The finer category, when the text names one. "shirt" is a
        # substring of "t-shirt", so scan longest-first.
        category = None
        for word in sorted(self.CATEGORY_WORDS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                category = self.CATEGORY_WORDS[word]
                break

        colour = None
        for candidate in sorted(self.color_vocab, key=len, reverse=True):
            if candidate and candidate in lowered:
                colour = candidate
                break
        return group, colour, category

    def parse_filters(self, text):
        """Pull detector-known category/color terms out of the text.

        Routed through `composed_query_search.parse_text_fragment` rather
        than raw substring matching, because the two vocabularies do not
        line up. The detector emits hierarchy CATEGORIES -- pants,
        sneaker, jacket, t-shirt -- while people type leaves and slang:
        "baggy jeans", "jorts", "tee". Substring matching finds no
        category for any of those, so the filter would silently never
        fire on exactly the phrasings it exists to serve.

        `parse_text_fragment` already climbs the real taxonomy and carries
        the slang table ("jorts" -> denim shorts), and it is the same
        parser `/compose` uses, so the two surfaces agree about what a
        word means. Its `category` field is hierarchy-category level,
        which is precisely the granularity the detections are labelled at
        -- that is the join.

        Colours stay substring: the detector's colour names are already
        the plain words people type ("black", "navy", "beige")."""
        lowered = (text or "").lower()

        categories = []
        try:
            import composed_query_search

            parsed = composed_query_search.parse_text_fragment(
                text, str(REPO_ROOT / "apparel_dataset" / "metadata.json"))
            category = (parsed or {}).get("category") or {}
            # Only accept a term the DETECTOR can actually satisfy --
            # filtering on a category no crop carries returns an empty
            # grid, which reads as a bug rather than as "no matches".
            if category.get("category") in self.category_vocab:
                categories = [category["category"]]
        except Exception:
            # Fail open to no filter, never to a wrong one.
            categories = []

        if not categories:
            categories = [c for c in sorted(self.category_vocab, key=len, reverse=True)
                          if c and c in lowered][:1]

        colors = [c for c in sorted(self.color_vocab, key=len, reverse=True)
                  if c and c in lowered][:2]
        return categories, colors

    def search(self, parts, top_k=24, use_filters=True,
               drop_non_us=False, drop_womens=False, type_preference=0.05,
               colour_name=None, colour_rgb=None, colour_weight=0.10,
               skin_lab=None, skin_weight=0.08, skin_min_confidence=0.10,
               colour_match="both", use_colour_head=True):
        """Multimodal outfit retrieval over an arbitrary set of query parts.

        `parts` is a list of {"kind": "image"|"text", "value": ..., "weight": float}.
        Any number, any mix, in any order. One image; three images; two
        images and a phrase; a phrase alone; two phrases. Nothing is
        privileged -- there is no distinguished "the image" or "the text".

        ## The rule that makes a multi-part query mean what it says

        Each part is matched to a DIFFERENT garment in the photo.

        A query is a set of things you want the outfit to contain
        simultaneously, and "simultaneously" is the whole content of the
        request. If every part scored against the photo's best-matching
        region, the winner would be a single garment that is a bit like
        all of them -- one jacket that is somewhat jeans-ish -- which
        satisfies the query on paper and not at all in fact.

        So parts are assigned to distinct crops, greedily by confidence:
        the part with the strongest single match claims its garment first,
        the next part takes the best garment still unclaimed, and so on.
        Greedy rather than optimal (Hungarian) assignment because the
        arrays are tiny (a few parts against a median of 3 garments) and
        the difference has never been the limiting factor here -- the
        encoder is, by a wide margin. See the module docstring.

        A photo's score is the MEAN of its parts' assigned scores, so
        adding a part cannot inflate a total, and a photo that satisfies
        two of three parts brilliantly does not beat one that satisfies
        all three well.

        Parts left unassigned -- more parts than garments, no detections,
        or no crop index yet -- fall back to whole-frame similarity, so a
        photo is never silently unreachable.
        """
        import numpy as np

        parts = [p for p in (parts or []) if p.get("value") not in (None, "")]
        if not parts:
            raise ValueError("supply at least one image or text part")

        # Expand a multi-garment sentence into one part per garment, so
        # each claims a different crop and the photo must satisfy them all.
        expanded = []
        for part in parts:
            if part["kind"] != "text":
                expanded.append(part)
                continue
            clauses = self.split_constraints(str(part["value"]))
            if len(clauses) <= 1:
                expanded.append(part)
                continue
            for phrase, _, _, _ in clauses:
                expanded.append({**part, "value": phrase, "label": phrase})
        parts = expanded

        encoded = []
        for part in parts:
            if part["kind"] == "image":
                vector = self.encode_image(part["value"])
            else:
                vector = self.encode_text(str(part["value"]))
            # An IMAGE part gets a soft same-garment-type preference.
            # Without it, 27.3% of item matches land on the wrong garment
            # group entirely -- measured on 150 catalog images: a top
            # matching a pants crop, accessories worst at 27.8% correct.
            # A crop of the wrong type cannot be what "find outfits with
            # something like this" is asking for.
            #
            # Added to the similarity rather than filtering on it, because
            # the query classifier is 88% accurate and a hard filter would
            # hand the other 12% nothing but wrong-type crops.
            crop_sim = (self.crop_embeddings @ vector).numpy() \
                if self.crop_embeddings is not None else None
            group = colour = category = None
            if crop_sim is not None and type_preference and self._crop_groups is not None:
                if part["kind"] == "image":
                    group, _ = self.classify_group(vector)
                else:
                    group, colour, category = self.parse_text_attributes(str(part["value"]))

                if group:
                    crop_sim = crop_sim + type_preference * (self._crop_groups == group)
                if category is not None and self._crop_categories is not None:
                    # On TOP of the group bonus, so a red sweater outranks a
                    # red hoodie (both are `tops`) while a red hoodie still
                    # outranks red trousers.
                    crop_sim = crop_sim + type_preference * (
                        self._crop_categories == category)
                if colour:
                    # The SAME crop earns both bonuses, which is the whole
                    # point: a yellow shoe gets type+colour, a yellow
                    # jacket gets colour only, a black shoe gets type only.
                    #
                    # The stored colour NAME comes from a nearest-palette
                    # vote in RGB Euclidean distance (index_outfits.
                    # dominant_color), which places boundaries badly:
                    # black/charcoal/navy and gray/white/light-gray sit
                    # close together in RGB but not perceptually. Matching
                    # the stored mean_rgb in CIELAB instead is continuous
                    # and perceptually spaced. Neither dominates -- the
                    # name survives patterns (a striped shirt's MEAN is a
                    # grey that is nowhere in the photo, which is why the
                    # vote exists), so "both" takes whichever is stronger.
                    bonus = None
                    if colour_match in ("name", "both"):
                        # Prefer the learned colour where available; it beat
                        # the palette vote 43.5% to 36.3% on held-out crops.
                        learned = (self._crop_colors_learned
                                   if use_colour_head else None)
                        source = learned if learned is not None else self._crop_colors
                        bonus = (source == colour).astype(float)
                        if learned is not None:
                            # Keep a smaller credit for the heuristic name:
                            # the two disagree on genuinely hard crops and
                            # a union recalls more than either alone.
                            bonus = np.maximum(
                                bonus, 0.5 * (self._crop_colors == colour))
                    if colour_match in ("lab", "both"):
                        target = COLOUR_PALETTE_RGB.get(colour)
                        if target is not None:
                            lab_bonus = self.colour_similarity(target)
                            if lab_bonus is not None:
                                bonus = lab_bonus if bonus is None else \
                                    __import__("numpy").maximum(bonus, lab_bonus)
                    if bonus is not None:
                        crop_sim = crop_sim + type_preference * bonus

            encoded.append({
                "kind": part["kind"],
                "label": part.get("label") or (part["value"] if part["kind"] == "text" else "image"),
                "weight": float(part.get("weight", 1.0)),
                "group": group,
                "colour": colour,
                "category": category,
                "photo_sim": (self.photo_embeddings @ vector).numpy(),
                "crop_sim": crop_sim,
            })

        # An EXPLICIT colour, from the dropdown or the eyedropper, applies
        # to every part's crop scores. Unlike a colour word inside a text
        # part it is unambiguous, so it gets a larger weight.
        colour_bonus = None
        if colour_rgb is not None:
            colour_bonus = self.colour_similarity(colour_rgb)
        elif colour_name and self._crop_colors is not None:
            colour_bonus = (self._crop_colors == colour_name).astype(float)
        if colour_bonus is not None:
            for part in encoded:
                if part["crop_sim"] is not None:
                    part["crop_sim"] = part["crop_sim"] + colour_weight * colour_bonus

        total_weight = sum(p["weight"] for p in encoded) or 1.0

        # Filters come from the text parts only; an image names nothing.
        categories, colors = [], []
        if use_filters:
            for part in parts:
                if part["kind"] == "text":
                    part_categories, part_colors = self.parse_filters(str(part["value"]))
                    categories += part_categories
                    colors += part_colors
        categories, colors = sorted(set(categories)), sorted(set(colors))

        # Skin-tone closeness per post, if a reference was given.
        skin_bonus = {}
        if skin_lab is not None and self._skin_lab:
            import numpy as np

            reference = np.asarray(skin_lab, dtype=float)
            for key, value in self._skin_lab.items():
                if self._skin_conf.get(key, 0.0) < skin_min_confidence:
                    continue
                # Hue and chroma carry this; lightness is the axis most
                # corrupted by exposure, so it is down-weighted the same
                # way the extractor does.
                delta = (np.asarray(value) - reference) * np.array([0.4, 1.0, 1.0])
                skin_bonus[key] = float(np.exp(-(np.linalg.norm(delta) / 12.0) ** 2))

        scored = []
        excluded_posts = 0
        for post_id, photo_rows in self.photo_rows_by_post.items():
            crop_rows = self.crops_by_post.get(post_id, [])

            if (drop_non_us or drop_womens) and self._section_excluded(
                    self.photo_records[photo_rows[0]], drop_non_us, drop_womens):
                excluded_posts += 1
                continue

            if categories or colors:
                # Fall back to the PHOTO records' own labels when this post
                # has no crop rows -- either because the crop index is not
                # built yet, or because the detector found nothing here.
                # Deriving the filter only from crops meant an unbuilt crop
                # index excluded every post and returned an empty grid.
                if crop_rows:
                    post_categories = {self.crop_records[r].get("category") for r in crop_rows}
                    post_colors = {self.crop_records[r].get("color") for r in crop_rows}
                else:
                    post_categories = {c for r in photo_rows
                                       for c in self.photo_records[r]["categories"]}
                    post_colors = {c for r in photo_rows
                                   for c in self.photo_records[r]["colors"]}
                # OR within a facet, AND across facets: several categories
                # in one query are different garments the outfit should
                # contain, so requiring all of them is right, but a photo
                # only has to match one colour term to be in scope.
                if categories and not set(categories).issubset(post_categories):
                    if not (set(categories) & post_categories):
                        continue
                if colors and not (set(colors) & post_colors):
                    continue

            whole_frame_best = None
            assignments, available = [], list(crop_rows)

            # Greedy: strongest single match claims its garment first.
            order = sorted(
                range(len(encoded)),
                key=lambda i: max((encoded[i]["crop_sim"][r] for r in crop_rows), default=-1.0),
                reverse=True)
            for part_index in order:
                part = encoded[part_index]
                if available and part["crop_sim"] is not None:
                    best = max(available, key=lambda r: part["crop_sim"][r])
                    available.remove(best)
                    assignments.append((part_index, float(part["crop_sim"][best]), best))
                else:
                    if whole_frame_best is None:
                        whole_frame_best = {}
                    score = max(part["photo_sim"][r] for r in photo_rows)
                    assignments.append((part_index, float(score), None))

            total = sum(encoded[i]["weight"] * score for i, score, _ in assignments) / total_weight
            if skin_bonus:
                total += skin_weight * skin_bonus.get(post_id, 0.0)

            # Display the photo the FIRST part's garment actually came from.
            display_row = photo_rows[0]
            first = next((a for a in assignments if a[0] == 0 and a[2] is not None), None)
            if first is not None:
                matched_rel = self.crop_records[first[2]]["rel"]
                for row in photo_rows:
                    if self.photo_records[row]["rel"] == matched_rel:
                        display_row = row
                        break

            scored.append((total, assignments, display_row))

        scored.sort(key=lambda entry: entry[0], reverse=True)

        hits = []
        for total, assignments, display_row in scored[:top_k]:
            record = self.photo_records[display_row]
            breakdown = []
            for part_index, score, crop_row in sorted(assignments):
                crop = self.crop_records[crop_row] if crop_row is not None else None
                breakdown.append({
                    "part": encoded[part_index]["label"],
                    "kind": encoded[part_index]["kind"],
                    "score": score,
                    "matched_garment": (crop or {}).get("category"),
                    "whole_frame": crop is None,
                })
            hits.append({**record, "score": total, "parts": breakdown})

        return {"results": hits,
                "filters_applied": {"categories": categories, "colors": colors},
                "inferred_types": [p["group"] for p in encoded if p["group"]],
                "inferred_colours": [p["colour"] for p in encoded if p.get("colour")],
                "inferred_categories": [p["category"] for p in encoded if p.get("category")],
                # Per-part parse, so the UI can show one control group per
                # garment. The flat lists above lose the association: with
                # "baggy jeans" + "blue hoodie" they render as
                # [colour blue][item pants][item hoodie], and nothing says
                # the blue belongs to the hoodie.
                "parts_parsed": [{"label": p["label"], "kind": p["kind"],
                                  "colour": p.get("colour"),
                                  "category": p.get("category")}
                                 for p in encoded],
                "category_vocab": sorted({c for c in self.CATEGORY_WORDS.values()}),
                "colour_applied": ("rgb" if colour_rgb is not None
                                   else (colour_name or None)),
                "colour_vocab": self.color_vocab,
                "colour_source": ("learned head" if self._crop_colors_learned is not None
                                  else "palette heuristic"),
                "skin_reference_used": skin_lab is not None,
                "skin_coverage": len(self._skin_lab),
                "used_crop_index": self.crop_embeddings is not None,
                "num_parts": len(encoded),
                "excluded_posts": excluded_posts,
                "corpus_posts": len(self.photo_rows_by_post) - excluded_posts}


def search_cli(args):
    engine = OutfitSearch()
    parts = ([{"kind": "image", "value": path, "label": Path(path).name}
              for path in (args.image or [])] +
             [{"kind": "text", "value": text} for text in (args.text or [])])
    result = engine.search(parts, args.top_k)
    print(f"\n  {result['corpus_posts']:,} posts · {result['num_parts']} query part(s) "
          f"· crop index "
          f"{'in use' if result['used_crop_index'] else 'NOT BUILT (whole-frame fallback)'}")
    print(f"  filters: {result['filters_applied']}\n")
    for rank, hit in enumerate(result["results"], 1):
        print(f"  {rank:2d}. {hit['score']:.3f}  {hit['rel']}")
        for part in hit["parts"]:
            where = part["matched_garment"] or ("whole frame" if part["whole_frame"] else "?")
            print(f"        {part['score']:.3f}  {part['part'][:34]:36} -> {where}")


# ----------------------------------------------------------------------
# serve
# ----------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>outfit search</title>
<style>
 :root{color-scheme:light dark;--bg:#fff;--fg:#16161a;--muted:#6b6b76;--line:#e3e3e8;
       --card:#fafafb;--accent:#2f6f4f;--warn:#8a5a00}
 @media(prefers-color-scheme:dark){:root{--bg:#131316;--fg:#ececf1;--muted:#9a9aa6;
       --line:#2c2c33;--card:#1b1b20;--accent:#7fd1a6;--warn:#e0b25f}}
 *{box-sizing:border-box}
 body{margin:0;padding:22px 16px 70px;background:var(--bg);color:var(--fg);
      font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
 main{max-width:1080px;margin:0 auto}
 h1{font-size:19px;margin:0 0 3px}
 .sub{color:var(--muted);font-size:13px;margin-bottom:18px}
 .controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
 .drop{flex:0 0 190px;border:1.5px dashed var(--line);border-radius:10px;padding:12px;
       text-align:center;cursor:pointer;background:var(--card);font-size:13px;
       color:var(--muted)}
 .drop.over{border-color:var(--accent)}
 .drop input{display:none}
 .drop img{max-width:100%;max-height:110px;border-radius:6px;display:block;margin:0 auto}
 input[type=text]{flex:1 1 300px;min-width:0;padding:11px 13px;font-size:15px;
       border:1px solid var(--line);border-radius:9px;background:var(--bg);color:var(--fg)}
 button{padding:11px 22px;font-size:15px;font-weight:600;border:0;border-radius:9px;
        background:var(--accent);color:var(--bg);cursor:pointer}
 button:disabled{opacity:.5;cursor:default}
 .slider{display:flex;align-items:center;gap:10px;font-size:12px;color:var(--muted);
         margin-top:12px;flex-wrap:wrap}
 .parts{display:flex;gap:7px;flex-wrap:wrap;margin-top:12px;align-items:center}
 .chip{display:inline-flex;align-items:center;gap:6px;padding:4px 8px;font-size:12px;
       border:1px solid var(--line);border-radius:99px;background:var(--card)}
 .chip img{width:22px;height:22px;object-fit:cover;border-radius:50%}
 .chip button{background:none;border:0;color:var(--muted);cursor:pointer;font-size:15px;
              padding:0 0 0 2px;line-height:1}
 .parts .hint{font-size:11px;color:var(--muted)}
 .card{display:inline-flex;flex-direction:column;gap:6px;padding:8px 10px;
       border:1px solid var(--line);border-radius:10px;background:var(--card)}
 .card .hdr{display:flex;align-items:center;gap:8px;font-size:13px;font-weight:600}
 .card .hdr button{background:none;border:0;color:var(--muted);cursor:pointer;
       font-size:15px;line-height:1;padding:0;margin-left:auto}
 .card .ctl{display:flex;gap:6px;align-items:center}
 .card select{border:1px solid var(--line);border-radius:6px;background:var(--bg);
       color:var(--fg);font-size:12px;padding:3px 5px}
 .card img{width:34px;height:34px;object-fit:cover;border-radius:6px}
 select{padding:5px 7px;border:1px solid var(--line);border-radius:7px;
        background:var(--bg);color:var(--fg);font-size:12px}
 .eyedrop{display:inline-flex;align-items:center;gap:7px}
 .eyedrop input[type=file]{display:none}
 .filebtn{padding:5px 9px;border:1px solid var(--line);border-radius:7px;
          background:var(--card);cursor:pointer;font-size:12px}
 .filebtn:hover{border-color:var(--accent)}
 #cv{border:1px solid var(--line);border-radius:6px;cursor:crosshair;display:none}
 .swatch{width:20px;height:20px;border-radius:50%;border:1px solid var(--line);
         display:inline-block}
 .status{font-size:12px;color:var(--muted);margin:16px 0 10px;font-family:ui-monospace,Menlo,monospace}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}
 .cell{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--card)}
 .cell img{width:100%;display:block;aspect-ratio:3/4;object-fit:cover;background:var(--bg)}
 .cell .m{padding:7px 9px;font-size:11px;color:var(--muted);line-height:1.4}
 .cell .m b{color:var(--fg);font-variant-numeric:tabular-nums}
 .caveat{margin-top:26px;padding:12px 14px;border-left:3px solid var(--warn);
         background:var(--card);font-size:13px;color:var(--muted);border-radius:0 8px 8px 0}
 .caveat b{color:var(--fg)}
</style></head><body><main>
 <h1>outfit search</h1>
 <div class="sub">A photo of an <b>item</b> plus words for what it should be worn
   <b>with</b> — real outfit photos out. Searching <span id="count">…</span>
   photos of <span id="posts">…</span> outfits.</div>

 <div class="controls">
   <label class="drop" id="drop">
     <input type="file" id="file" accept="image/*" multiple>
     <span id="droptext">items<br>drop · tap · paste<br><small>as many as you like</small></span>
   </label>
   <input type="text" id="text" placeholder='words (optional) — "baggy jeans", or leave empty'>
   <button id="go">Search</button>
 </div>

 <div id="parts" class="parts"></div>

 <div class="slider">
   <label><input type="checkbox" id="filters" checked>
     require the named garment</label>
   <label><input type="checkbox" id="us"> US only <span class="hint">(drops
     wear.jp / Korean, 22%)</span></label>
   <label><input type="checkbox" id="mens"> men's only <span class="hint">(drops
     women's subs, 20% — the catalog is 100% men's)</span></label>
   <label>colour
     <select id="colour"><option value="">any</option></select></label>
   <span class="eyedrop">
     <!-- The canvas must NOT live inside the label: a click anywhere in a
          label activates its form control, so putting the canvas in here
          re-opened the file picker instead of picking a pixel. -->
     <label class="filebtn">pick colour from a photo
       <input type="file" id="colourfile" accept="image/*"></label>
     <canvas id="cv" width="150" height="150"
             title="click any pixel to use that exact colour"></canvas>
     <span id="eyehint" class="hint" hidden>← click a pixel</span>
     <span id="swatch" class="swatch" hidden></span>
     <button type="button" id="clearcol" hidden>clear</button>
   </span>
   <span class="eyedrop">
     <label class="filebtn">similar skin tone
       <input type="file" id="skinfile" accept="image/*"></label>
     <span id="skinstate" class="hint"></span>
     <button type="button" id="clearskin" hidden>clear</button>
   </span>
   <label><input type="checkbox" id="typepref" checked> match garment type
     <span class="hint">(soft; without it 27% of item matches land on the
     wrong garment type)</span></label>
 </div>

 <div class="status" id="status"></div>
 <div class="grid" id="grid"></div>

 <div class="caveat">
   <b>What you are looking at.</b> These are photographs of real people from
   Reddit, Pinterest and wear.jp, matched by a SigLIP2 encoder that was
   fine-tuned on <i>catalog product photos</i> — so it is working outside the
   distribution it was trained on, and how well it transfers has not been
   measured. The garment filter uses detector labels that are unvalidated
   model output. Nothing here is evaluated yet; judge it by looking at it.
   The skin-tone reference is <b>relative</b> — it finds photos whose skin
   reads similarly to your reference under similar lighting. Absolute tone
   binning measured badly here (photographed skin is far darker than the
   reference swatches), so it is deliberately not offered, and extraction
   is still running so coverage is partial.
 </div>
</main>
<script>
// The query is a LIST of parts, not one image and one string. Images and
// phrases are peers -- any number of either, in any combination, and a
// query of only images is as valid as a query of only words.
let images=[], texts=[];
const $=id=>document.getElementById(id);

let pickedRGB=null, lastColourVocab=[], lastCategoryVocab=[];

fetch('/info').then(r=>r.json()).then(d=>{
  for(const c of (d.colours||[]))
    $('colour').insertAdjacentHTML('beforeend','<option>'+c+'</option>');
  $('count').textContent=d.count.toLocaleString();
  $('posts').textContent=d.posts.toLocaleString();
  if(!d.crops) $('status').textContent =
    'note: the garment-crop index is still building — parts are falling back '+
    'to whole-frame similarity until it finishes.';
});

// One CARD per query part. Each card owns its own colour and item
// dropdowns, so it is visually unambiguous which colour belongs to which
// garment -- the previous flat row rendered "blue hoodie" + "baggy jeans"
// as [colour blue][item pants][item hoodie] with nothing tying them.
let lastParsed=null;
function renderParts(){
  const cards=images.map((src,i)=>
      '<span class="card"><span class="hdr"><img src="'+src+'">image '+(i+1)+
      '<button onclick="dropImage('+i+')" title="remove">×</button></span></span>');

  texts.forEach((t,i)=>{
    const parsed=(lastParsed||[]).find(p=>p.kind==='text'&&p.label===t)||{};
    const cols=(lastColourVocab||[]), cats=(lastCategoryVocab||[]);
    const sel=(id,val,opts,any)=>
      '<select onchange="editPart('+i+',\''+id+'\',this.value)">'+
      '<option value=""'+(val?'':' selected')+'>'+any+'</option>'+
      opts.map(o=>'<option'+(o===val?' selected':'')+'>'+o+'</option>').join('')+
      '</select>';
    cards.push('<span class="card"><span class="hdr">“'+esc(t)+'”'+
      '<button onclick="dropText('+i+')" title="remove">×</button></span>'+
      (lastParsed ? '<span class="ctl">'+
        sel('colour',parsed.colour||'',cols,'any colour')+
        sel('category',parsed.category||'',cats,'any item')+'</span>' : '')+
      '</span>');
  });

  $('parts').innerHTML = cards.length
    ? cards.join('')+'<span class="hint">'+cards.length+
      ' part'+(cards.length>1?'s':'')+' — each matched to a different garment</span>'
    : '';
}

// Rewrite one part's phrase from its dropdowns, preserving any words the
// parser did not claim ("baggy" in "baggy jeans" survives a jeans->shorts
// change) by replacing the matched word in place rather than rebuilding.
function editPart(i,field,value){
  const parsed=(lastParsed||[]).find(p=>p.kind==='text'&&p.label===texts[i])||{};
  const old=parsed[field];
  let phrase=texts[i];
  if(old && value){
    phrase=phrase.replace(new RegExp('\\b'+old.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+'\\b','i'),value);
  } else if(old && !value){
    phrase=phrase.replace(new RegExp('\\b'+old.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+'\\b','i'),'').replace(/\s+/g,' ').trim();
  } else if(!old && value){
    phrase=(field==='colour') ? value+' '+phrase : phrase+' '+value;
  }
  if(!phrase.trim()){ dropText(i); run(); return; }
  texts[i]=phrase.trim();
  renderParts(); run();
}
function dropImage(i){ images.splice(i,1); renderParts(); }
function dropText(i){ texts.splice(i,1); renderParts(); }

function loadFiles(list){
  for(const f of list){
    if(!f||!f.type.startsWith('image/'))continue;
    const r=new FileReader();
    r.onload=e=>{ images.push(e.target.result); renderParts(); };
    r.readAsDataURL(f);
  }
}
$('file').addEventListener('change',e=>loadFiles(e.target.files));
$('drop').addEventListener('dragover',e=>{e.preventDefault();$('drop').classList.add('over')});
$('drop').addEventListener('dragleave',()=>$('drop').classList.remove('over'));
$('drop').addEventListener('drop',e=>{e.preventDefault();$('drop').classList.remove('over');
  loadFiles(e.dataTransfer.files)});
document.addEventListener('paste',e=>{
  const files=[...e.clipboardData.items].filter(i=>i.type.startsWith('image/'))
    .map(i=>i.getAsFile());
  if(files.length)loadFiles(files);
});

function commitText(){
  const v=$('text').value.trim();
  if(v){ texts.push(v); $('text').value=''; renderParts(); }
}

// Eyedropper: draw the uploaded photo into a canvas and read the pixel the
// user clicks. Exact colour beats a 19-name vocabulary for "this shade".
$('colourfile').addEventListener('change',e=>{
  const f=e.target.files[0]; if(!f) return;
  const img=new Image();
  img.onload=()=>{
    const cv=$('cv'), ctx=cv.getContext('2d');
    const scale=Math.min(cv.width/img.width, cv.height/img.height);
    const w=img.width*scale, h=img.height*scale;
    ctx.clearRect(0,0,cv.width,cv.height);
    ctx.drawImage(img,(cv.width-w)/2,(cv.height-h)/2,w,h);
    cv.style.display='block'; $('eyehint').hidden=false;
  };
  img.src=URL.createObjectURL(f);
});
$('cv').addEventListener('click',ev=>{
  ev.preventDefault(); ev.stopPropagation();
  const cv=$('cv'), r=cv.getBoundingClientRect();
  const x=Math.round((ev.clientX-r.left)*cv.width/r.width);
  const y=Math.round((ev.clientY-r.top)*cv.height/r.height);
  // Average a small patch, not one pixel: JPEG noise and compression make
  // a single sample unreliable, and people aim at a region anyway.
  const ctx=cv.getContext('2d',{willReadFrequently:true});
  const half=3, sx=Math.max(0,x-half), sy=Math.max(0,y-half);
  const d=ctx.getImageData(sx,sy,half*2+1,half*2+1).data;
  let rr=0,gg=0,bb=0,n=0;
  for(let i=0;i<d.length;i+=4){ if(d[i+3]===0) continue; rr+=d[i];gg+=d[i+1];bb+=d[i+2];n++; }
  if(!n) return;
  pickedRGB=[Math.round(rr/n),Math.round(gg/n),Math.round(bb/n)];
  $('swatch').style.background='rgb('+pickedRGB.join(',')+')';
  $('swatch').hidden=false; $('clearcol').hidden=false;
  $('eyehint').hidden=true;
  $('colour').value='';   // an exact pick overrides the dropdown
  run();                  // searching immediately is the point of picking
});
$('clearcol').addEventListener('click',()=>{
  pickedRGB=null; $('swatch').hidden=true; $('clearcol').hidden=true;
  $('eyehint').hidden=true; $('cv').style.display='none'; $('colourfile').value='';
});

let skinImage=null;
$('skinfile').addEventListener('change',e=>{
  const f=e.target.files[0]; if(!f) return;
  const r=new FileReader();
  r.onload=ev=>{ skinImage=ev.target.result;
    $('skinstate').textContent='reference set'; $('clearskin').hidden=false; run(); };
  r.readAsDataURL(f);
});
$('clearskin').addEventListener('click',()=>{
  skinImage=null; $('skinstate').textContent=''; $('clearskin').hidden=true;
  $('skinfile').value=''; run();
});

async function run(){
  commitText();
  if(!images.length&&!texts.length){
    $('status').textContent='add a photo or some words (either alone is fine)';return}
  $('go').disabled=true; $('status').textContent='searching…';
  try{
    const res=await fetch('/search',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({images, texts, use_filters:$('filters').checked,
        drop_non_us:$('us').checked, drop_womens:$('mens').checked,
        type_preference:$('typepref').checked?0.05:0,
        colour_name:$('colour').value||null, colour_rgb:pickedRGB,
        skin_image:skinImage, top_k:24})});
    const d=await res.json();
    if(!res.ok)throw new Error(d.detail||'failed');
    const f=d.filters_applied, bits=[];
    if(f.categories.length)bits.push('category '+f.categories.join(' + '));
    if(f.colors.length)bits.push('color '+f.colors.join('/'));
    $('status').textContent=d.results.length+' outfits · '+d.num_parts+' part'+
      (d.num_parts>1?'s':'')+
      (d.excluded_posts?'  ·  '+d.excluded_posts.toLocaleString()+
        ' posts excluded, searching '+d.corpus_posts.toLocaleString():'')+
      (d.inferred_types&&d.inferred_types.length?'  ·  read as '+
        d.inferred_types.join('/'):'')+
      (d.colour_applied?'  ·  colour '+
        (d.colour_applied==='rgb'?'(exact pick)':d.colour_applied):'')+
      (d.skin_reference_used?'  ·  skin tone ref ('+
        (d.skin_coverage||0).toLocaleString()+' photos measured so far)':'')+
      (bits.length?'  ·  must contain '+bits.join(', '):'')+
      (d.used_crop_index?'':'  ·  whole-frame fallback (crop index not built)')+
      '  ·  '+d.ms+' ms';
    lastParsed=d.parts_parsed||null;
    lastColourVocab=d.colour_vocab||lastColourVocab;
    lastCategoryVocab=d.category_vocab||lastCategoryVocab;
    renderParts();
    $('grid').innerHTML=d.results.map(r=>
      '<div class="cell"><a href="'+(r.post_url||'#')+'" target="_blank" rel="noopener">'+
      '<img loading="lazy" src="/photo?path='+encodeURIComponent(r.rel)+'"></a>'+
      '<div class="m"><b>'+r.score.toFixed(3)+'</b> · '+r.source+'<br>'+
      r.parts.map(p=>esc(String(p.part).slice(0,18))+' '+p.score.toFixed(2)+
        ' → '+esc(p.matched_garment||'whole frame')).join('<br>')+
      '</div></div>').join('');
  }catch(err){ $('status').textContent=err.message; }
  $('go').disabled=false;
}
$('go').addEventListener('click',run);
// Enter adds another phrase rather than searching, so a multi-part text
// query is possible without leaving the keyboard.
$('text').addEventListener('keydown',e=>{
  if(e.key==='Enter'){ e.preventDefault(); if($('text').value.trim())commitText(); else run(); }});

function esc(s){ return String(s==null?'':s).replace(/[&<>"']/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
</script></body></html>
"""


def serve(args):
    import base64
    import tempfile
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, unquote, urlparse

    print("  loading index and encoder…")
    engine = OutfitSearch()
    print(f"  {len(engine.photo_records):,} photos / "
          f"{len(engine.photo_rows_by_post):,} outfits · "
          f"{len(engine.crop_records):,} garment crops")
    if engine.crop_embeddings is None:
        print("  crop index NOT built — the item half falls back to whole frames")
    outfit_root = (REPO_ROOT / "outfit_dataset").resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, code, body, ctype="application/json"):
            payload = body if isinstance(body, bytes) else str(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if self.path.startswith("/info"):
                return self._send(200, json.dumps({
                    "count": len(engine.photo_records),
                    "posts": len(engine.photo_rows_by_post),
                    "crops": engine.crop_embeddings is not None,
                    "colours": engine.color_vocab,
                }).encode())
            if self.path.startswith("/photo"):
                raw = parse_qs(urlparse(self.path).query).get("path", [""])[0]
                target = (REPO_ROOT / unquote(raw)).resolve()
                if not str(target).startswith(str(outfit_root)):
                    return self._send(403, b'{"detail":"forbidden"}')
                if not target.is_file():
                    return self._send(404, b'{"detail":"missing"}')
                return self._send(200, target.read_bytes(), "image/jpeg")
            self._send(404, b'{"detail":"not found"}')

        def do_POST(self):
            if self.path != "/search":
                return self._send(404, b'{"detail":"not found"}')
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            started = time.time()

            # Any number of images, any number of phrases, any mix.
            parts, temps = [], []
            for index, encoded in enumerate(body.get("images") or []):
                if encoded.startswith("data:"):
                    encoded = encoded.split(",", 1)[-1]
                handle = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                handle.write(base64.b64decode(encoded))
                handle.close()
                temps.append(handle.name)
                parts.append({"kind": "image", "value": handle.name,
                              "label": f"image {index + 1}"})
            for phrase in body.get("texts") or []:
                if str(phrase).strip():
                    parts.append({"kind": "text", "value": str(phrase).strip()})

            # A skin reference photo: parse it once here, so the client
            # sends an image and the server does the extraction.
            skin_lab = body.get("skin_lab")
            skin_image = body.get("skin_image")
            if skin_lab is None and skin_image:
                try:
                    import extract_skin_tone as skin

                    encoded_skin = skin_image.split(",", 1)[-1]
                    handle = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    handle.write(base64.b64decode(encoded_skin))
                    handle.close()
                    temps.append(handle.name)
                    from PIL import Image as PILImage, ImageOps as PILOps

                    with PILImage.open(handle.name) as raw:
                        reference_image = PILOps.exif_transpose(raw).convert("RGB")
                    import numpy as np

                    import garment_proposer

                    if getattr(engine, "_skin_parser", None) is None:
                        engine._skin_parser = garment_proposer.load_human_parser("cpu")
                    proc, mdl = engine._skin_parser
                    label_map, probs = garment_proposer.parse_person(
                        reference_image, proc, mdl, "cpu")
                    estimate = skin.estimate(reference_image, np.asarray(label_map),
                                             np.asarray(probs), skin.monk_lab())
                    skin_lab = (estimate or {}).get("lab")
                except Exception as error:  # noqa: BLE001
                    print(f"[skin] reference failed: {error}")
                    skin_lab = None

            try:
                result = engine.search(parts, int(body.get("top_k") or 24),
                                       bool(body.get("use_filters", True)),
                                       bool(body.get("drop_non_us", False)),
                                       bool(body.get("drop_womens", False)),
                                       float(body.get("type_preference", 0.05)),
                                       body.get("colour_name") or None,
                                       body.get("colour_rgb") or None,
                                       skin_lab=skin_lab)
            except ValueError as error:
                return self._send(400, json.dumps({"detail": str(error)}).encode())
            finally:
                for path in temps:
                    os.unlink(path)

            result["ms"] = round((time.time() - started) * 1000)
            self._send(200, json.dumps(result).encode())

    server = ThreadingHTTPServer(("0.0.0.0" if args.lan else "127.0.0.1", args.port), Handler)
    print(f"\n  open: http://localhost:{args.port}\n  ctrl-c to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")


def check_environment():
    """Fail with the actual problem instead of an ImportError 300 frames in.

    Running this under system python3 produces "AutoModel requires the
    PyTorch library but it was not found" raised from inside transformers,
    long after a model download has already started -- which reads as a
    broken install rather than as the wrong interpreter. torch IS
    installed there; it is 2.0.0, and transformers silently disables its
    PyTorch backend for anything below 2.4."""
    try:
        import torch
    except ImportError:
        raise SystemExit(
            "\n  torch is not available in this interpreter.\n"
            f"  You are running: {sys.executable}\n"
            "  Use the project venv instead:\n\n"
            "      .venv/bin/python outfit_search.py ...\n")

    major, minor = (int(part) for part in torch.__version__.split(".")[:2])
    if (major, minor) < (2, 4):
        raise SystemExit(
            f"\n  torch {torch.__version__} is too old — transformers needs >= 2.4\n"
            f"  and silently disables its PyTorch backend below that, so the\n"
            f"  real error surfaces much later as a confusing ImportError.\n"
            f"  You are running: {sys.executable}\n"
            "  Use the project venv instead:\n\n"
            "      .venv/bin/python outfit_search.py ...\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="embed the outfit corpus (run once)")
    b.add_argument("--index", choices=("photos", "crops", "both"), default="both",
                   help="photos = full frames; crops = detected garments "
                        "(what the item half of a query scores against)")
    b.add_argument("--limit", type=int, help="smoke-test on the first N images")
    b.add_argument("--rebuild", action="store_true",
                   help="ignore cached crop vectors and re-encode everything")
    b.set_defaults(func=build)

    s = sub.add_parser("search", help="one query from the CLI")
    s.add_argument("--image", action="append",
                   help="repeatable — each image is a separate query part")
    s.add_argument("--text", action="append",
                   help="repeatable — each phrase is a separate query part")
    s.add_argument("--top-k", type=int, default=24)
    s.set_defaults(func=search_cli)

    v = sub.add_parser("serve", help="browse it")
    v.add_argument("--port", type=int, default=7880)
    v.add_argument("--lan", action="store_true")
    v.set_defaults(func=serve)

    args = ap.parse_args()
    check_environment()
    args.func(args)


if __name__ == "__main__":
    main()
