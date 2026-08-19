"""Multi-item outfit segmentation -- Phase 3's real starting point (spec
section 4.2, docs/roadmap.md's Tier 3 gap: "the biggest single finding"
of the 2026-08-01 spec audit was that nothing in this pipeline had ever
detected more than one item per photo). Every other script in this repo
(segment_apparel.py included) assumes the input is a single-product
catalog photo and the record's own category is already known -- this
script drops both assumptions: given an arbitrary photo (a real outfit,
category unknown, possibly several visible items), find and crop EVERY
plausible garment/accessory in it, not just one.

Reuses segment_apparel.py's validated SAM2 + FashionCLIP building blocks
(CPU device -- MPS is measured 15-20x slower for SAM2's mask generator,
see segment_apparel.py's own comment; downsize to MAX_IMAGE_DIM before
generate(); sam2.1_hiera_small checkpoint) but the selection logic is
genuinely different, not a copy:

  segment_apparel.py: ONE known target category -> pick the SINGLE best-
  scoring mask matching that category's label phrasings, among masks in
  the tuned area/confidence band.

  segment_outfit.py: category UNKNOWN and potentially MULTIPLE real items
  -> score every candidate mask against every category in the newly
  restructured 5-group taxonomy (footwear/tops/bottoms/outerwear/
  accessories, docs/hierarchy.json), suppress masks that heavily overlap
  a higher-scoring mask (SAM2's automatic generator proposes many
  redundant/nested masks for the same physical region), then keep at most
  one mask per category (a real outfit photo realistically shows one
  instance of each garment type -- multi-instance-per-category, e.g. two
  visible rings, is out of scope for this first version).

Honesty check on where this stands (per this project's own norm of
reporting real numbers, not assumed ones): the area-band/confidence
thresholds below are carried over VERBATIM from segment_apparel.py,
which tuned them against single-product catalog photos, not real
multi-item outfit photos -- they are a reasonable starting point, not a
validated one for this harder case. This script has only been smoke-
tested against real lifestyle/model photos already in this dataset
(Champion/Gap catalog images that happen to show a model wearing more
than one visible garment, the closest real proxy available locally to an
actual "outfit photo") -- it has NOT been benchmarked with real
detection metrics (mAP, recall by visibility/occlusion per spec section
8.2) against a labeled multi-item test set, because no such labeled set
exists in this project yet. That's the honest next step, not implied
complete by this script existing.

Usage:
    python3 segment_outfit.py --image path/to/outfit_photo.jpg
    python3 segment_outfit.py --image path/to/outfit_photo.jpg --out-dir /tmp/outfit_items
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# sam2 imported LAZILY at its one call site: this module also holds
# CATEGORY_LABELS and crop helpers that garment_proposer (the default
# proposer, which uses no SAM2 at all) imports, and a module-scope import
# made those unreachable wherever sam2 is not installed.

# Same device/checkpoint/resize choices as segment_apparel.py -- see that
# script's comment for why (MPS measured >2min/image vs. ~7s on CPU for
# SAM2's automatic mask generator on this hardware).
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_small.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
# points_per_side raised 16->32, 2026-08-02, after a real miss: a Carhartt
# outfit photo (model small in frame, large empty background) proposed
# only 4 raw SAM2 candidate masks at 16, and the one region that should
# have matched "jeans" never got proposed as its own mask at all -- 0/1
# items detected on an image with 2 clearly visible garments. At 32,
# the same image proposed 6 masks including a real, correct, high-
# confidence "jeans" candidate (0.826). Re-validated the two ALREADY-
# working test cases (Champion hoodie, Vans sweatshirt+pants) at the new
# setting before committing to it -- both still correctly detected,
# same categories, no new false positives, no regression. Real,
# 3-image-validated improvement, not a blind parameter guess -- see
# docs/roadmap.md's 2026-08-02 entry for the full before/after.
MASK_GENERATOR_KWARGS = dict(points_per_side=32, points_per_batch=64)
MAX_IMAGE_DIM = 1024

MIN_MASK_AREA = 1000
MIN_CONFIDENCE = 0.4
# Lowered 0.03->0.02 alongside the points_per_side change above, same
# validation -- the real "jeans" candidate found at points_per_side=32
# had area_fraction=0.028, which the old 0.03 floor excluded by a razor-
# thin margin despite it being a correct, high-confidence detection.
# Carried over from segment_apparel.py originally -- see this module's
# own docstring: tuned against single-product catalog photos, not fully
# re-validated against real multi-item outfit photos, though this specific
# adjustment now has 3 real test images behind it, not zero.
MIN_AREA_FRACTION = 0.02  # lower than segment_apparel.py's 0.08 -- one item
                           # among several visible ones occupies a smaller
                           # fraction of the frame than a product photo's
                           # single subject does.
MAX_AREA_FRACTION = 0.55

# Masks whose IoU with an already-kept, higher-scoring mask exceeds this
# are suppressed -- SAM2's automatic generator proposes many overlapping/
# nested candidates for the same physical region (e.g. "the whole
# hoodie" and "just the hoodie pocket" both score reasonably for "a
# hoodie"); without this, the same garment would be reported multiple
# times as if they were different items.
NMS_IOU_THRESHOLD = 0.5

DISTRACTOR_LABELS = ["a person", "a face", "human skin", "background", "the ground", "a wall"]

# How decisively a distractor must beat the best garment label before the
# mask is discarded.
#
# Measured 2026-08-04 by instrumenting the funnel over 8 real outfit
# photos: of 174 proposed masks, 137 cleared the area floor and only 16
# survived everything -- and **90 of those 137 (66%) died solely because
# their top label was a distractor**. Confidence dropped 12 and the area
# band dropped 19, so this single rule discarded more than four times as
# much as the other two filters combined. That is the reason the detector
# recovers ~1.85 garments from photos containing 4-5.
#
# The rule was inherited from segment_apparel.py, where it is correct: a
# catalog photo is one garment on white, so a mask scoring highest for "a
# person" really is background. A WORN OUTFIT photo inverts that -- every
# garment mask also contains a person, so "a person" wins on regions that
# are predominantly clothing.
#
# Argmax is the wrong comparison for that; a margin is the right one, the
# same fix that worked for the garment gate and find_subject_bbox.
#
# ---- AND THEN IT TURNED OUT TO BUY NOTHING. Kept, at 0.0, as a record. ----
#
# Sweeping 0.0 / 0.10 / 0.20 / 0.30 over 6 real outfit photos produced
# BIT-IDENTICAL output: 7 items, same per-image counts, same categories.
# The reason is that the funnel attribution above was an artifact of
# filter ORDER, not a real bottleneck. Measuring the 63 masks a distractor
# actually won:
#
#     best-garment score   min 0.000   median 0.031   max 0.241
#     would clear MIN_CONFIDENCE=0.4:  0 of 63
#     would clear MIN_CONFIDENCE=0.2:  3 of 63
#
# So those masks die at MIN_CONFIDENCE regardless; the distractor check
# merely got credit for kills that came first in the code. A median
# garment score of 0.031 against 0.034 for uniform over 29 labels means
# FashionCLIP sees no clothing in them AT ALL -- the rule is correct, and
# there is nothing there to rescue.
#
# The real bottleneck is UPSTREAM of every filter: SAM2's class-agnostic
# point grid is not proposing masks that isolate individual garments. Of
# 174 proposals across 8 photos only ~16 were garment-shaped regions, and
# no threshold can recover a garment from a proposal that does not contain
# one. Fixing this needs a garment-aware proposer -- human parsing (this
# repo already vendors Self-Correction-Human-Parsing, currently unused by
# any script) or SAM2 prompted with person keypoints instead of a uniform
# grid. Parameter tuning is the wrong tool.
DISTRACTOR_MARGIN = float(os.environ.get("DISTRACTOR_MARGIN", "0.0"))

# Category label phrasings per the 5-group/13-category taxonomy finalized
# in docs/hierarchy.json (2026-08-01 restructuring) -- kept in sync with
# that file's groups by hand since this taxonomy changes rarely and a
# detector's label phrasings need actual English text, not just the raw
# category token. Re-check this dict against docs/hierarchy.json if that
# file's top-level groups change again.
CATEGORY_LABELS = {
    ("footwear", "sneaker"): ["a sneaker", "a shoe"],
    ("footwear", "loafer"): ["a loafer"],
    ("tops", "t-shirt"): ["a t-shirt"],
    ("tops", "shirt"): ["a button-down shirt", "a collared shirt"],
    ("tops", "sweatshirt"): ["a sweatshirt", "a crewneck"],
    ("tops", "hoodie"): ["a hoodie", "a pullover hoodie"],
    ("tops", "tank top"): ["a tank top"],
    ("tops", "sweater"): ["a sweater", "a knit sweater"],
    ("bottoms", "shorts"): ["shorts"],
    ("bottoms", "pants"): ["pants", "jeans", "trousers"],
    ("outerwear", "jacket"): ["a jacket", "a coat"],
    ("accessories", "hat"): ["a hat", "a baseball cap", "a beanie"],
    ("accessories", "socks"): ["a pair of socks"],
}

ALL_LABELS = [label for labels in CATEGORY_LABELS.values() for label in labels]
LABEL_TO_CATEGORY = {label: key for key, labels in CATEGORY_LABELS.items() for label in labels}
CANDIDATE_LABELS = ALL_LABELS + DISTRACTOR_LABELS


def load_rgb_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def crop_masked_region(image_pil, mask):
    mask = mask.astype(bool)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None, None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return image_pil.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


# Prompts for locating the photographic SUBJECT inside a screenshot, used
# only by --subject-crop. Kept separate from CANDIDATE_LABELS: this stage
# is deliberately looking for the whole person, which the garment stage
# spends its effort excluding.
SUBJECT_LABELS = [
    "a photo of a person wearing clothes",
    "a full body photo of a person",
    "a fashion model posing",
]
SUBJECT_DISTRACTOR_LABELS = [
    "a screenshot of a website",
    "a block of text",
    "a user interface with buttons",
    "a plain background",
    "a logo",
    "a price tag",
]
# Padding around the subject bbox before re-detecting, as a fraction of the
# bbox. A tight crop can clip shoes or a hat at the boundary.
SUBJECT_CROP_PAD = 0.05
# A subject must be at least this fraction of the canvas to be believed --
# below it, we are probably locking onto a thumbnail or an icon.
MIN_SUBJECT_AREA_FRACTION = 0.03


def find_subject_bbox(image, mask_generator, processor, clip_model, device):
    """Locate the photographic subject in a screenshot. -> bbox or None.

    Why this exists: `area_fraction` is computed against the WHOLE canvas,
    and MAX_AREA_FRACTION=0.55 was calibrated on catalog photos where the
    garment fills the frame -- there, a "whole person" mask lands above
    0.55 and is correctly rejected. In a phone screenshot the person
    occupies maybe 25-30% of the canvas (the rest is nav bars, price text,
    other products), so the whole-person mask sails through the very
    filter meant to exclude it, and FashionCLIP happily scores it for a
    generic garment label. Measured symptom: a real product-page
    screenshot returned exactly 1 item, the whole person, wrong garment.

    Cropping to the subject first re-normalizes every area fraction to the
    thing that actually matters, and incidentally recovers resolution --
    thumbnail() caps the LONG edge, so a 1170x2532 screenshot shrinks to
    473x1024 and its hero image to roughly 473x358 before SAM2 ever sees
    it.
    """
    image_np = np.array(image)
    full_area = image_np.shape[0] * image_np.shape[1]
    masks = mask_generator.generate(image_np)

    crops, boxes, areas = [], [], []
    for mask_data in masks:
        if mask_data["area"] < MIN_MASK_AREA:
            continue
        crop, box = crop_masked_region(image, mask_data["segmentation"])
        if crop is None:
            continue
        if mask_data["area"] / full_area < MIN_SUBJECT_AREA_FRACTION:
            continue
        crops.append(crop)
        boxes.append(box)
        areas.append(mask_data["area"])

    if not crops:
        return None

    labels = SUBJECT_LABELS + SUBJECT_DISTRACTOR_LABELS
    inputs = processor(images=crops, text=labels, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        probs = clip_model(**inputs).logits_per_image.softmax(dim=-1)

    best_box, best_score = None, 0.0
    for idx, box in enumerate(boxes):
        subject_score = probs[idx, :len(SUBJECT_LABELS)].max().item()
        distractor_score = probs[idx, len(SUBJECT_LABELS):].max().item()
        # Must beat every distractor, not merely clear a threshold -- page
        # chrome is abundant and some of it scores respectably on "person".
        if subject_score > distractor_score and subject_score > best_score:
            best_box, best_score = box, subject_score

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    pad_x = int((x2 - x1) * SUBJECT_CROP_PAD)
    pad_y = int((y2 - y1) * SUBJECT_CROP_PAD)
    return (max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(image.width, x2 + pad_x), min(image.height, y2 + pad_y))


def detect_outfit_items(image_path, mask_generator, processor, clip_model, device,
                        subject_crop=False):
    """Returns a list of detected items, most-confident first, each:
    {"category_group": ..., "category": ..., "label": ..., "confidence": ...,
     "bbox": (x1,y1,x2,y2), "area_fraction": ..., "crop": PIL.Image}.

    `subject_crop` is OFF by default on purpose. It changes what this
    function detects, and index_outfits.py is mid-corpus-run over
    outfit_dataset/ -- flipping the default underneath it would make half
    that corpus incomparable to the other half. It is also unnecessary for
    the photos it is processing: a Reddit outfit photo IS the subject,
    already filling its frame. Turn it on for SCREENSHOTS, where the
    garment is a minority of the canvas.
    """
    image = load_rgb_image(image_path)
    image.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM))

    if subject_crop:
        bbox = find_subject_bbox(image, mask_generator, processor, clip_model, device)
        if bbox is not None:
            image = image.crop(bbox)
            # Re-expand: the crop is smaller than the canvas, so SAM2 gets
            # the garment region back at closer to full working resolution
            # rather than at whatever the screenshot's aspect ratio left it.
            image.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM))
    image_np = np.array(image)
    full_area = image_np.shape[0] * image_np.shape[1]

    masks = mask_generator.generate(image_np)

    crops, boxes, areas = [], [], []
    for mask_data in masks:
        if mask_data["area"] < MIN_MASK_AREA:
            continue
        crop, box = crop_masked_region(image, mask_data["segmentation"])
        if crop is None:
            continue
        crops.append(crop)
        boxes.append(box)
        areas.append(mask_data["area"])

    if not crops:
        return []

    inputs = processor(images=crops, text=CANDIDATE_LABELS, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=-1)  # [num_crops, num_labels]

    candidates = []
    for idx, (crop, box, area) in enumerate(zip(crops, boxes, areas)):
        top_label_idx = probs[idx].argmax().item()
        top_score = probs[idx, top_label_idx].item()
        top_label = CANDIDATE_LABELS[top_label_idx]
        area_fraction = area / full_area

        # Garment-vs-distractor by MARGIN, not argmax. See DISTRACTOR_MARGIN:
        # at 0.0 this is bit-identical to the old `top_label not in
        # LABEL_TO_CATEGORY: continue`, because requiring the distractor to
        # merely exceed the garment is the same as it winning the argmax.
        garment_scores = [(probs[idx, j].item(), CANDIDATE_LABELS[j])
                          for j in range(len(CANDIDATE_LABELS))
                          if CANDIDATE_LABELS[j] in LABEL_TO_CATEGORY]
        distractor_best = max((probs[idx, j].item()
                               for j in range(len(CANDIDATE_LABELS))
                               if CANDIDATE_LABELS[j] not in LABEL_TO_CATEGORY),
                              default=0.0)
        if not garment_scores:
            continue
        garment_best, garment_label = max(garment_scores)
        if distractor_best > garment_best + DISTRACTOR_MARGIN:
            continue  # a distractor wins decisively -- background, not clothing

        # From here the garment interpretation is the one under test, so
        # confidence and area are judged on IT rather than on whatever
        # happened to win the raw argmax.
        top_label, top_score = garment_label, garment_best
        if top_score < MIN_CONFIDENCE:
            continue
        if not (MIN_AREA_FRACTION <= area_fraction <= MAX_AREA_FRACTION):
            continue
        group, category = LABEL_TO_CATEGORY[top_label]
        candidates.append({
            "category_group": group, "category": category, "label": top_label,
            "confidence": top_score, "bbox": box, "area_fraction": area_fraction, "crop": crop,
        })

    # Highest confidence first, then greedy NMS across all candidates
    # (not per-category -- two different categories' masks can also
    # heavily overlap if FashionCLIP scored the same region for both,
    # and only the more confident interpretation should survive).
    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    kept = []
    for candidate in candidates:
        if any(box_iou(candidate["bbox"], k["bbox"]) > NMS_IOU_THRESHOLD for k in kept):
            continue
        kept.append(candidate)

    # At most one item per (group, category) -- see module docstring's
    # "out of scope for this first version" note on multi-instance items.
    best_per_category = {}
    for candidate in kept:
        key = (candidate["category_group"], candidate["category"])
        if key not in best_per_category or candidate["confidence"] > best_per_category[key]["confidence"]:
            best_per_category[key] = candidate

    return sorted(best_per_category.values(), key=lambda c: c["confidence"], reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an outfit photo.")
    parser.add_argument("--out-dir", default=None, help="Directory to save per-item crops (optional).")
    parser.add_argument("--proposer", choices=["sam2", "human-parsing"], default="sam2",
                        help="Which module proposes candidate regions. 'sam2' (default) is "
                             "the class-agnostic point grid this script has always used. "
                             "'human-parsing' uses garment_proposer.py, which segments the "
                             "person into clothing classes first -- see that module's "
                             "docstring and the DISTRACTOR_MARGIN block above for the "
                             "measurement that motivated it. OPT-IN on purpose: "
                             "outfit_cooccurrence.json was built from the sam2 path, and "
                             "silently changing detection would make that corpus only "
                             "half-comparable to itself.")
    parser.add_argument("--subject-crop", action="store_true",
                        help="Locate the photographic subject first and detect garments "
                             "within it. Use for SCREENSHOTS, where the garment is a "
                             "minority of the canvas and the whole-person mask otherwise "
                             "passes the area filter meant to exclude it. Off by default: "
                             "it changes detections, and index_outfits.py is mid-corpus-run.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"  # deliberately no MPS branch, see module docstring
    print(f"Using device: {device}")

    print("Loading FashionCLIP...")
    processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    clip_model = AutoModelForZeroShotImageClassification.from_pretrained("patrickjohncyh/fashion-clip")
    clip_model.to(device)

    if args.proposer == "human-parsing":
        # Imported here, not at module scope: garment_proposer imports back
        # from this module for the taxonomy, and it must stay possible to
        # run the default sam2 path without the parser's weights present.
        from garment_proposer import load_human_parser, propose_garment_items
        if args.subject_crop:
            # --subject-crop's whole mechanism is SAM2-based (find_subject_bbox
            # runs the mask generator), and the parser needs no help finding
            # the person anyway -- it segments one directly.
            print("Note: --subject-crop is ignored by the human-parsing proposer.")
        print("Loading human parser...")
        parser_processor, parser_model = load_human_parser(device)
        items = propose_garment_items(args.image, parser_processor, parser_model,
                                      processor, clip_model, device)
    else:
        print("Loading SAM2...")
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        sam2 = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
        mask_generator = SAM2AutomaticMaskGenerator(sam2, **MASK_GENERATOR_KWARGS)

        items = detect_outfit_items(args.image, mask_generator, processor, clip_model, device,
                                    subject_crop=args.subject_crop)

    print(f"\n{len(items)} item(s) detected in {args.image}:")
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    results_json = []
    for rank, item in enumerate(items, start=1):
        print(f"  #{rank}  {item['category_group']}/{item['category']}  "
              f"(label='{item['label']}', confidence={item['confidence']:.3f}, "
              f"area_fraction={item['area_fraction']:.3f}, bbox={item['bbox']})")
        crop_path = None
        if out_dir:
            crop_path = out_dir / f"item_{rank}_{item['category_group']}_{item['category'].replace(' ', '-')}.jpg"
            item["crop"].convert("RGB").save(crop_path, quality=90)
        results_json.append({
            "rank": rank, "category_group": item["category_group"], "category": item["category"],
            "label": item["label"], "confidence": item["confidence"], "bbox": item["bbox"],
            "area_fraction": item["area_fraction"], "crop_path": str(crop_path) if crop_path else None,
        })

    if out_dir:
        (out_dir / "detections.json").write_text(json.dumps(results_json, indent=2), encoding="utf-8")
        print(f"\nSaved crops + detections.json to {out_dir}")


if __name__ == "__main__":
    main()
