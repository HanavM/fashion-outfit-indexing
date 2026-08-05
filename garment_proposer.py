"""Garment-aware mask proposer -- the fix for the bottleneck recorded in
`segment_outfit.py`'s DISTRACTOR_MARGIN comment block.

WHY THIS EXISTS (the measurement, not a hunch). Instrumenting the existing
SAM2 path over 8 real outfit photos: 174 masks proposed, 137 cleared the
area floor, 16 survived. The distractor rule looked responsible (90 of
137) but sweeping a margin over it gave BIT-IDENTICAL output at
0.0/0.10/0.20/0.30, because the 63 masks a distractor won have a median
best-GARMENT score of 0.031 (uniform over 29 labels is 0.034) and zero of
them would ever clear MIN_CONFIDENCE=0.4. The funnel attribution was an
artifact of filter order. The real problem is upstream: SAM2's
class-agnostic point grid proposes regions that are not garments --
half-limbs, background patches, texture blobs, the whole person -- and no
threshold recovers a garment from a proposal that does not contain one.

So this module replaces the PROPOSER, not the thresholds. It segments the
person semantically into clothing regions first, and only then asks
FashionCLIP which category each region is. Skin and background are
excluded structurally (the parser has its own Face/Left-arm/Left-leg/
Background classes) rather than by a score comparison that was never
going to work.

APPROACH CHOSEN: human parsing, over the alternative of prompting SAM2
with person keypoints. Prompted SAM2 would still be class-agnostic -- a
keypoint on the torso yields *a* region near the torso, and you are back
to asking FashionCLIP to decide whether it is a shirt, a jacket, or a
chest-shaped piece of background. A human parser answers "this pixel is
Upper-clothes" directly, which is exactly the question the funnel was
failing on, and it gives instance boundaries that follow the garment
rather than the point grid.

WHICH human parser, and why not the vendored one. This repo vendors
`Self-Correction-Human-Parsing/` (unused by any script). Two blockers,
both checked rather than assumed:
  1. No weights are present anywhere in the tree (`find -name '*.pth'`
     returns only a setuptools shim). The official checkpoints are Google
     Drive links; mirrors exist on the Hub.
  2. More decisively, its network does not build on this machine at all:
     `networks.init_model` pulls in `modules/bn.py`'s InPlaceABN, which
     JIT-compiles a CUDA C++ extension and dies with `CUDA_HOME
     environment variable is not set` -- CUDA-only by construction, on a
     CPU/MPS Mac. Installing ninja gets you exactly one error further.
So this uses a SegFormer trained on the SAME dataset and the SAME label
space that SCHP's ATR variant uses (ATR-18: Upper-clothes, Skirt, Pants,
Dress, Left/Right-shoe, Hat, Belt, Bag, Scarf, plus Hair/Face/arms/legs/
Background). Same task, same label semantics, obtainable weights, plain
`transformers` classes -- no vendored CUDA extension and no
`trust_remote_code`.

WHAT IT RETURNS: the same candidate dicts `segment_outfit.detect_outfit_items`
returns, so it is a drop-in alternative behind `--proposer human-parsing`
and nothing downstream (crop saving, index_outfits.py's record shape)
needs to change.

Standalone usage:
    .venv/bin/python garment_proposer.py --image photo.jpg --out-dir /tmp/items
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy import ndimage

HUMAN_PARSER_MODEL = "mattmdjaga/segformer_b2_clothes"

# ATR-18, in the order the checkpoint's config declares it.
ATR_LABELS = [
    "Background", "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt",
    "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face",
    "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf",
]

# Which parser classes are garments we have a taxonomy slot for, and which
# of segment_outfit.CATEGORY_LABELS' groups each one may resolve to.
#
# Upper-clothes maps to BOTH tops and outerwear on purpose: ATR has no
# Coat class (LIP does), so a jacket and a t-shirt land in the same parser
# class and FashionCLIP has to make that call. Skirt/Dress are deliberately
# absent -- docs/hierarchy.json has no category for either, and inventing
# one here would put items in the co-occurrence index that no catalog
# category can ever match.
PARSER_CLASS_TO_GROUPS = {
    "Upper-clothes": ("tops", "outerwear"),
    "Pants": ("bottoms",),
    "Left-shoe": ("footwear",),
    "Right-shoe": ("footwear",),
    "Hat": ("accessories",),
}

# A parsed region below this fraction of the frame is noise -- a few
# stray pixels of "Hat" on a hairline, a shoe-coloured speck. Chosen to
# sit BELOW segment_outfit.MIN_AREA_FRACTION (0.02) rather than at it: a
# real shoe in a full-body photo is genuinely small (measured 0.004-0.015
# on this corpus), and the 0.02 floor was calibrated for SAM2 proposals
# that were mostly not garments. The parser has already asserted this
# region IS a garment, so the floor's job here is only despeckling.
MIN_REGION_AREA_FRACTION = 0.002
MIN_REGION_PIXELS = 400

# Connected components smaller than this fraction of their class's largest
# component are dropped: one garment occluded into two blobs is common
# (a shirt split by a crossed arm), but the second blob should be a real
# piece of it, not a 20-pixel fragment.
MIN_COMPONENT_RATIO = 0.15

# Minimum mean parser posterior for a region to be believed.
#
# MEASURED, on the 40-image sample whose 105 crops I inspected one by one
# (see the commit message / docs entry for the run). Of those 105, 15 were
# not real garments -- mostly a bare foot or a sandal strap read as
# Left-shoe, plus one collage image whose four regions came out shredded.
# Their mean parser posterior is median 0.653 against 0.944 for the 90
# real ones, so this single number separates them well:
#
#     thr 0.0  kept 105  real 85.7%  items/photo 2.62
#     thr 0.5  kept  99  real 90.9%  items/photo 2.48   <- default
#     thr 0.7  kept  86  real 96.5%  items/photo 2.15
#     thr 0.9  kept  57  real 98.2%  items/photo 1.43
#
# 0.5 is the strictly-free point on this sample: it removes 6 false
# positives and zero true ones. 0.7 buys another 5.6pp of precision for
# 0.33 items/photo and is one argument away. Honest caveat: these numbers
# come from the same 40 images the threshold was picked on -- it is a
# defensible default, not a held-out result.
MIN_PARSER_SCORE = 0.5

MAX_IMAGE_DIM = 1024


def load_human_parser(device="cpu"):
    """-> (processor, model). Separate from the caller so a batch run pays
    the load cost once."""
    from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
    processor = SegformerImageProcessor.from_pretrained(HUMAN_PARSER_MODEL)
    model = AutoModelForSemanticSegmentation.from_pretrained(HUMAN_PARSER_MODEL)
    model.to(device)
    model.eval()
    return processor, model


def parse_person(image, processor, model, device="cpu"):
    """Semantic parse of `image` (PIL RGB) -> (label_map HxW int, probs CxHxW).

    The model emits logits at 1/4 resolution; upsampled bilinearly back to
    the image size before argmax so region boundaries land on the image's
    own pixel grid and the crops line up with what a human sees.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=(image.height, image.width), mode="bilinear", align_corners=False
    )[0]
    probs = upsampled.softmax(dim=0)
    return probs.argmax(dim=0).cpu().numpy(), probs.cpu().numpy()


def garment_regions(label_map, probs, min_area_fraction=MIN_REGION_AREA_FRACTION):
    """-> list of {parser_class, mask, area, parser_score}, largest first.

    One entry per connected component of each garment class, so two shoes
    or a shirt split by an occluding arm come out as separate proposals
    instead of one bounding box spanning the gap between them (which is
    what makes SAM2's whole-person masks useless for retrieval).
    """
    height, width = label_map.shape
    full_area = height * width
    regions = []
    for class_idx, class_name in enumerate(ATR_LABELS):
        if class_name not in PARSER_CLASS_TO_GROUPS:
            continue
        class_mask = label_map == class_idx
        if not class_mask.any():
            continue
        components, n_components = ndimage.label(class_mask)
        sizes = ndimage.sum(class_mask, components, range(1, n_components + 1))
        if len(sizes) == 0:
            continue
        largest = sizes.max()
        for comp_idx, size in enumerate(sizes, start=1):
            if size < MIN_REGION_PIXELS or size / full_area < min_area_fraction:
                continue
            if size < largest * MIN_COMPONENT_RATIO:
                continue
            mask = components == comp_idx
            regions.append({
                "parser_class": class_name,
                "mask": mask,
                "area": int(size),
                # Mean posterior for the winning class over the region --
                # how sure the parser is this really is that garment,
                # reported alongside (never merged into) FashionCLIP's
                # category confidence, because they answer different
                # questions.
                "parser_score": float(probs[class_idx][mask].mean()),
            })
    regions.sort(key=lambda r: r["area"], reverse=True)
    return regions


def propose_garment_items(image_path, parser_processor, parser_model,
                          clip_processor, clip_model, device="cpu",
                          min_confidence=0.0, mask_background=True,
                          min_parser_score=MIN_PARSER_SCORE):
    """Garment-aware replacement for segment_outfit.detect_outfit_items.

    Returns the same dicts: category_group, category, label, confidence,
    bbox, area_fraction, crop -- plus parser_class/parser_score, which the
    SAM2 path has no equivalent of.

    `min_confidence` defaults to 0.0, NOT to segment_outfit's 0.4. That
    threshold's job was rejecting non-garment SAM2 proposals; here the
    parser has already done that job structurally, and FashionCLIP's
    remaining job is only picking a category *within* the group the parser
    named -- a distribution over 2-8 labels, where 0.4 would throw away
    genuine ambiguity between, say, a sweatshirt and a hoodie. Pass a
    value if you want the stricter behaviour.
    """
    from segment_outfit import CATEGORY_LABELS, LABEL_TO_CATEGORY, crop_masked_region

    image = load_rgb_image(image_path)
    image.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM))
    full_area = image.width * image.height

    label_map, probs = parse_person(image, parser_processor, parser_model, device)
    regions = [r for r in garment_regions(label_map, probs)
               if r["parser_score"] >= min_parser_score]
    if not regions:
        return []

    crops, boxes = [], []
    for region in regions:
        crop, box = crop_masked_region(image, region["mask"])
        if crop is None:
            crops.append(None)
            boxes.append(None)
            continue
        if mask_background:
            # Blank everything outside the garment inside its own bbox.
            # A shoe's bbox is mostly ground; a pants bbox contains the
            # other leg's background. docs/product_gap_analysis.md item
            # 11.1 already measured this exact failure -- a correctly
            # labelled "loafer" whose COLOUR came off the wall behind it.
            # Slice by the crop's own size, not by box+1: crop_masked_region
            # builds the box from inclusive max indices but PIL's crop()
            # treats right/bottom as exclusive, so the returned crop is one
            # pixel smaller than the box in each axis.
            sub = region["mask"][box[1]:box[1] + crop.height, box[0]:box[0] + crop.width]
            arr = np.array(crop)
            arr[~sub] = 255
            crop = Image.fromarray(arr)
        crops.append(crop)
        boxes.append(box)

    valid = [i for i, c in enumerate(crops) if c is not None]
    if not valid:
        return []

    all_labels = [lbl for labels in CATEGORY_LABELS.values() for lbl in labels]
    inputs = clip_processor(images=[crops[i] for i in valid], text=all_labels,
                            return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        clip_probs = clip_model(**inputs).logits_per_image.softmax(dim=-1)

    items = []
    for row, region_idx in enumerate(valid):
        region = regions[region_idx]
        allowed_groups = PARSER_CLASS_TO_GROUPS[region["parser_class"]]
        # Restrict to labels the parser's class permits, then RENORMALISE
        # over just those. The resulting number is an honest posterior for
        # "which top is it, given it is a top" -- not comparable to the
        # SAM2 path's confidence, which is a posterior over 29 labels
        # including distractors. Recorded as such.
        allowed = [j for j, lbl in enumerate(all_labels)
                   if LABEL_TO_CATEGORY[lbl][0] in allowed_groups]
        if not allowed:
            continue
        subset = clip_probs[row, allowed]
        best = int(subset.argmax().item())
        confidence = float((subset / subset.sum())[best].item())
        label = all_labels[allowed[best]]
        if confidence < min_confidence:
            continue
        group, category = LABEL_TO_CATEGORY[label]
        items.append({
            "category_group": group, "category": category, "label": label,
            "confidence": confidence, "bbox": boxes[region_idx],
            "area_fraction": region["area"] / float(full_area),
            "crop": crops[region_idx],
            "parser_class": region["parser_class"],
            "parser_score": region["parser_score"],
        })

    # One item per (group, category), same rule as the SAM2 path -- two
    # shoes are one footwear item for co-occurrence purposes. No NMS: the
    # parse is a partition, so two regions cannot overlap by construction.
    best_per_category = {}
    for item in items:
        key = (item["category_group"], item["category"])
        if key not in best_per_category or item["area_fraction"] > best_per_category[key]["area_fraction"]:
            best_per_category[key] = item
    return sorted(best_per_category.values(), key=lambda i: i["confidence"], reverse=True)


def load_rgb_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-parser-score", type=float, default=MIN_PARSER_SCORE,
                        help="Minimum mean parser posterior per region. See MIN_PARSER_SCORE "
                             "for the measured precision/recall trade at 0.5 / 0.7 / 0.9.")
    args = parser.parse_args()

    from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser_processor, parser_model = load_human_parser(device)
    clip_processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    clip_model = AutoModelForZeroShotImageClassification.from_pretrained(
        "patrickjohncyh/fashion-clip").to(device)

    items = propose_garment_items(args.image, parser_processor, parser_model,
                                  clip_processor, clip_model, device,
                                  min_confidence=args.min_confidence,
                                  min_parser_score=args.min_parser_score)
    print(f"{len(items)} item(s) in {args.image}:")
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for rank, item in enumerate(items, start=1):
        print(f"  #{rank}  {item['category_group']}/{item['category']}  "
              f"parser={item['parser_class']} ({item['parser_score']:.2f})  "
              f"clip={item['confidence']:.3f}  area={item['area_fraction']:.3f}")
        crop_path = None
        if out_dir:
            crop_path = out_dir / f"item_{rank}_{item['category_group']}_{item['category'].replace(' ', '-')}.jpg"
            item["crop"].convert("RGB").save(crop_path, quality=90)
        records.append({k: v for k, v in item.items() if k != "crop"} | {
            "crop_path": str(crop_path) if crop_path else None})
    if out_dir:
        (out_dir / "detections.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
