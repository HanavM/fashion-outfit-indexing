"""Search real people's outfit photos with a picture and/or a text query.

    python outfit_search.py build          # embed the 9,999 outfit photos (once)
    python outfit_search.py serve          # browse it at http://localhost:7880
    python outfit_search.py search --text "baggy jeans" --image jacket.jpg

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
    return out


def encode_crops(model, processor, records, fts, torch):
    """Mirror of `free_text_visual_search.encode_images`, but cropping to
    each record's bbox first. Kept local rather than generalising that
    function, because it is imported by the serving path and this is not
    the moment to change something `/search` depends on."""
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from tqdm import tqdm

    batch_size = fts.IMAGE_BATCH_SIZE
    embeddings, kept = [], []
    # Open each source photo once even though several crops share it --
    # decoding a 1536px JPEG per crop would dominate the runtime at ~2.1
    # crops per photo.
    cache_path, cache_image = None, None

    for start in tqdm(range(0, len(records), batch_size), desc="Encoding garment crops"):
        batch = records[start:start + batch_size]
        images, batch_kept = [], []
        for record in batch:
            try:
                if record["path"] != cache_path:
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
        print(f"\n  {len(records):,} garment crops to encode")
        with torch.inference_mode():
            embeddings, kept = encode_crops(model, processor, records, fts, torch)
        torch.save({"embeddings": embeddings.half(), "records": kept,
                    "checkpoint": load_from}, CROP_INDEX_PATH)
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

    def search(self, image_path=None, text=None, top_k=24,
               image_weight=0.5, use_filters=True):
        """Item-anchored outfit retrieval.

        The query is "an outfit containing something like THIS, plus
        THAT". Those two halves are about *different garments in the same
        photo*, so they are scored against different garments:

            item_score = max over the photo's crops of sim(query_image, crop)
            text_score = max over its OTHER crops    of sim(query_text, crop)

        Excluding the item-matched crop from the text half is the part
        that makes "this jacket with baggy jeans" mean what it says. Score
        both halves against the same region and the top result is a photo
        of a jacket that is somewhat jeans-like -- one garment satisfying
        both terms, which is not what was asked.

        Photos with no detections (or with no crop index built yet) fall
        back to whole-frame similarity for both halves, so they stay
        reachable rather than silently dropping out of the corpus.
        """
        text = (text or "").strip()
        if not image_path and not text:
            raise ValueError("supply an image, text, or both")

        query_image = self.encode_image(image_path) if image_path else None
        query_text = self.encode_text(text) if text else None

        photo_image_sim = (self.photo_embeddings @ query_image).numpy() if query_image is not None else None
        photo_text_sim = (self.photo_embeddings @ query_text).numpy() if query_text is not None else None
        crop_image_sim = (self.crop_embeddings @ query_image).numpy() \
            if (query_image is not None and self.crop_embeddings is not None) else None
        crop_text_sim = (self.crop_embeddings @ query_text).numpy() \
            if (query_text is not None and self.crop_embeddings is not None) else None

        categories, colors = self.parse_filters(text) if (use_filters and text) else ([], [])

        # Weights: when only one half is present it takes the whole score,
        # so the slider cannot silently zero out a query the user did give.
        weight_image = image_weight if (query_image is not None and query_text is not None) \
            else (1.0 if query_image is not None else 0.0)
        weight_text = 1.0 - weight_image if (query_image is not None and query_text is not None) \
            else (1.0 if query_text is not None else 0.0)

        scored = []
        for post_id, photo_rows in self.photo_rows_by_post.items():
            crop_rows = self.crops_by_post.get(post_id, [])

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
                if categories and not (set(categories) & post_categories):
                    continue
                if colors and not (set(colors) & post_colors):
                    continue

            item_score = matched_row = None
            if query_image is not None:
                if crop_rows and crop_image_sim is not None:
                    best = max(crop_rows, key=lambda r: crop_image_sim[r])
                    item_score, matched_row = float(crop_image_sim[best]), best
                else:
                    item_score = float(max(photo_image_sim[r] for r in photo_rows))

            text_score = None
            if query_text is not None:
                others = [r for r in crop_rows if r != matched_row]
                if others and crop_text_sim is not None:
                    text_score = float(max(crop_text_sim[r] for r in others))
                elif crop_rows and crop_text_sim is not None:
                    # Only one garment detected: scoring the text against
                    # the same region is wrong, but dropping the photo is
                    # worse. Use the whole frame instead.
                    text_score = float(max(photo_text_sim[r] for r in photo_rows))
                else:
                    text_score = float(max(photo_text_sim[r] for r in photo_rows))

            total = (weight_image * (item_score or 0.0)) + (weight_text * (text_score or 0.0))
            display_row = photo_rows[0]
            if matched_row is not None:
                # Show the photo the matched garment actually came from.
                matched_rel = self.crop_records[matched_row]["rel"]
                for row in photo_rows:
                    if self.photo_records[row]["rel"] == matched_rel:
                        display_row = row
                        break

            scored.append((total, item_score, text_score, display_row, matched_row))

        scored.sort(key=lambda entry: entry[0], reverse=True)

        hits = []
        for total, item_score, text_score, display_row, matched_row in scored[:top_k]:
            record = self.photo_records[display_row]
            matched = self.crop_records[matched_row] if matched_row is not None else None
            hits.append({
                **record,
                "score": total,
                "item_score": item_score,
                "text_score": text_score,
                "matched_garment": (matched or {}).get("category"),
                "matched_bbox": (matched or {}).get("bbox"),
            })

        return {"results": hits,
                "filters_applied": {"categories": categories, "colors": colors},
                "used_crop_index": self.crop_embeddings is not None,
                "corpus_posts": len(self.photo_rows_by_post)}


def search_cli(args):
    engine = OutfitSearch()
    result = engine.search(args.image, args.text, args.top_k, args.image_weight)
    print(f"\n  {result['corpus_posts']:,} posts · crop index "
          f"{'in use' if result['used_crop_index'] else 'NOT BUILT (whole-frame fallback)'}")
    print(f"  filters: {result['filters_applied']}\n")
    for rank, hit in enumerate(result["results"], 1):
        item = f"item {hit['item_score']:.3f}" if hit["item_score"] is not None else ""
        text = f"text {hit['text_score']:.3f}" if hit["text_score"] is not None else ""
        print(f"  {rank:2d}. {hit['score']:.3f}  {item:12} {text:12}  "
              f"matched={hit['matched_garment'] or '-'}")
        print(f"      {hit['rel']}")


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
 .slider input{flex:1 1 220px;max-width:340px}
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
     <input type="file" id="file" accept="image/*">
     <span id="droptext">the ITEM<br>drop · tap · paste</span>
   </label>
   <input type="text" id="text" placeholder='worn with… e.g. "baggy jeans"'>
   <button id="go">Search</button>
 </div>

 <div class="slider">
   <span>matters most: the words</span>
   <input type="range" id="w" min="0" max="100" value="50">
   <span>the item</span>
   <span id="wlabel">50 / 50</span>
   <label style="margin-left:auto"><input type="checkbox" id="filters" checked>
     require the named garment</label>
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
 </div>
</main>
<script>
let imageData=null;
const $=id=>document.getElementById(id);

fetch('/info').then(r=>r.json()).then(d=>{
  $('count').textContent=d.count.toLocaleString();
  $('posts').textContent=d.posts.toLocaleString();
  if(!d.crops) $('status').textContent =
    'note: the garment-crop index is still building — the item half is '+
    'falling back to whole-frame similarity until it finishes.';
});

function loadFile(f){
  if(!f||!f.type.startsWith('image/'))return;
  const r=new FileReader();
  r.onload=e=>{ imageData=e.target.result;
    $('droptext').innerHTML='<img src="'+imageData+'">'; };
  r.readAsDataURL(f);
}
$('file').addEventListener('change',e=>loadFile(e.target.files[0]));
$('drop').addEventListener('dragover',e=>{e.preventDefault();$('drop').classList.add('over')});
$('drop').addEventListener('dragleave',()=>$('drop').classList.remove('over'));
$('drop').addEventListener('drop',e=>{e.preventDefault();$('drop').classList.remove('over');
  loadFile(e.dataTransfer.files[0])});
document.addEventListener('paste',e=>{for(const it of e.clipboardData.items)
  if(it.type.startsWith('image/'))loadFile(it.getAsFile())});
$('w').addEventListener('input',()=>{const v=+$('w').value;
  $('wlabel').textContent=(100-v)+' / '+v});

async function run(){
  const text=$('text').value.trim();
  if(!imageData&&!text){$('status').textContent='give me a photo or some words';return}
  $('go').disabled=true; $('status').textContent='searching…';
  try{
    const res=await fetch('/search',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({image_base64:imageData,text,
        image_weight:+$('w').value/100, use_filters:$('filters').checked, top_k:24})});
    const d=await res.json();
    if(!res.ok)throw new Error(d.detail||'failed');
    const f=d.filters_applied;
    const bits=[];
    if(f.categories.length)bits.push('category '+f.categories.join('/'));
    if(f.colors.length)bits.push('color '+f.colors.join('/'));
    $('status').textContent=d.results.length+' outfits'+
      (bits.length?'  ·  must contain '+bits.join(' + '):'')+
      (d.used_crop_index?'':'  ·  whole-frame fallback (crop index not built)')+
      '  ·  '+d.ms+' ms';
    $('grid').innerHTML=d.results.map(r=>{
      const parts=[];
      if(r.item_score!=null)parts.push('item '+r.item_score.toFixed(2)+
        (r.matched_garment?' ('+r.matched_garment+')':''));
      if(r.text_score!=null)parts.push('with '+r.text_score.toFixed(2));
      return '<div class="cell"><a href="'+(r.post_url||'#')+'" target="_blank" rel="noopener">'+
      '<img loading="lazy" src="/photo?path='+encodeURIComponent(r.rel)+'"></a>'+
      '<div class="m"><b>'+r.score.toFixed(3)+'</b> · '+r.source+'<br>'+
      parts.join(' · ')+'</div></div>';}).join('');
  }catch(err){ $('status').textContent=err.message; }
  $('go').disabled=false;
}
$('go').addEventListener('click',run);
$('text').addEventListener('keydown',e=>{if(e.key==='Enter')run()});
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

            temp = None
            encoded = body.get("image_base64")
            if encoded:
                if encoded.startswith("data:"):
                    encoded = encoded.split(",", 1)[-1]
                handle = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                handle.write(base64.b64decode(encoded))
                handle.close()
                temp = handle.name
            try:
                result = engine.search(
                    temp, body.get("text"), int(body.get("top_k") or 24),
                    float(body.get("image_weight", 0.5)),
                    bool(body.get("use_filters", True)))
            except ValueError as error:
                return self._send(400, json.dumps({"detail": str(error)}).encode())
            finally:
                if temp:
                    os.unlink(temp)

            result["ms"] = round((time.time() - started) * 1000)
            self._send(200, json.dumps(result).encode())

    server = ThreadingHTTPServer(("0.0.0.0" if args.lan else "127.0.0.1", args.port), Handler)
    print(f"\n  open: http://localhost:{args.port}\n  ctrl-c to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="embed the outfit corpus (run once)")
    b.add_argument("--index", choices=("photos", "crops", "both"), default="both",
                   help="photos = full frames; crops = detected garments "
                        "(what the item half of a query scores against)")
    b.add_argument("--limit", type=int, help="smoke-test on the first N images")
    b.set_defaults(func=build)

    s = sub.add_parser("search", help="one query from the CLI")
    s.add_argument("--image")
    s.add_argument("--text")
    s.add_argument("--top-k", type=int, default=24)
    s.add_argument("--image-weight", type=float, default=0.5)
    s.set_defaults(func=search_cli)

    v = sub.add_parser("serve", help="browse it")
    v.add_argument("--port", type=int, default=7880)
    v.add_argument("--lan", action="store_true")
    v.set_defaults(func=serve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
