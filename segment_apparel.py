"""
Crop apparel product photos down to just the garment, removing model
face/body/other-clothing background noise, using the SAM2 + FashionCLIP
pattern prototyped in fashionCLIP_SAM2.py — generalized to run over every
image of every record in a given category instead of one hardcoded image.

Collapsed single-step design (vs. fashionCLIP_SAM2.py's generic
candidate_labels list): SAM produces masks with no semantics at all — some
image-aware model has to look at each crop's pixels and decide what garment
it shows, which is why FashionCLIP (an image<->text model) is used here, not
a plain text encoder (which never sees the image). Rather than classifying
against a big generic label list and then separately text-matching that
label to the category name, each crop is classified DIRECTLY against a small
per-record candidate set: the record's own category phrased a couple of ways,
plus a fixed set of distractor labels representing exactly the noise being
cropped out (person/face/skin/background/shoe/bag). Whichever crop's
top-scoring label is one of the category's own phrasings wins.

Only ever applied to the new Nike clothing records (brand=="nike",
category in CATEGORY_LABELS) — the existing 563 shoe records don't have the
same background-noise problem and are left untouched.

Non-destructive: writes new `*_cropped.jpg` files next to each original
image, and a `cropped_images` field on the record — never overwrites or
deletes `images`.

Usage:
    pip install -e /tmp/sam2_src   # already done; needs checkpoints/sam2.1_hiera_large.pt
    python segment_apparel.py --limit 10   # test batch
    python segment_apparel.py              # full run over Nike clothing records
"""

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from dataset_utils import DB_FILE, load_records, save_records_safe

CHECKPOINT_EVERY = 5

# Using the "small" checkpoint (184MB) instead of "large" — lighter, and
# device is forced to CPU (see main()), never MPS: SAM2 on Apple's MPS
# backend hits slow/unsupported-op fallbacks that made even the small
# checkpoint take >2min for a single image with no progress; plain CPU
# does the same image in ~7s. This matches fashionCLIP_SAM2.py's original
# working device selection (cuda-or-cpu, MPS was never in that script).
# Images are also downsized to max 1024px before mask generation — SAM2
# upsamples every mask back to the input resolution, and these product
# photos are ~2880x3600 natively, which was the single biggest cost driver
# (resizing alone took a single image from 42s down to ~3-7s).
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_small.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
MASK_GENERATOR_KWARGS = dict(points_per_side=16, points_per_batch=64)
MAX_IMAGE_DIM = 1024

MIN_MASK_AREA = 1000
MIN_CONFIDENCE = 0.4
# Bounds on mask area as a fraction of the (resized) image. Excludes both
# extremes that a plain highest-score-wins or smallest-wins rule falls into:
# too large ~= "the whole person" (CLIP scores that highest for generic
# category labels since it's the "typical" framing of someone wearing the
# garment, but it's exactly the background/body noise we're trying to crop
# out); too small ~= a homogeneous fabric fragment (a sleeve, a waistband)
# that coincidentally clears MIN_CONFIDENCE just by having plausible color/
# texture, without actually being a usable crop of the garment.
MIN_AREA_FRACTION = 0.08
MAX_AREA_FRACTION = 0.55

DISTRACTOR_LABELS = ["a person", "a face", "human skin", "background", "a shoe", "a bag"]

# Category name -> candidate label phrasings to match crops against.
CATEGORY_LABELS = {
    "Tops and T-Shirts": ["a t-shirt", "a top"],
    "Shorts": ["shorts"],
    "Hoodies and Pullovers": ["a hoodie", "a pullover", "a sweatshirt"],
    "Pants and Tights": ["pants", "tights", "trousers"],
    "Pants": ["pants", "jeans", "trousers"],
    "Graphic Tees": ["a t-shirt", "a graphic tee"],
    "Sweaters": ["a sweater", "a knit sweater"],
    "T-Shirts": ["a t-shirt", "a top"],
    "Jackets": ["a jacket", "a coat"],
    "Hats": ["a hat", "a baseball cap", "a beanie"],
    "Socks": ["a sock", "a pair of socks"],
    # Champion expansion -- reuses the same label phrasings as the
    # equivalent existing categories above, just under Champion's own
    # category-name strings.
    "Hoodies and Sweatshirts": ["a hoodie", "a pullover", "a sweatshirt"],
    "T-Shirts and Tops": ["a t-shirt", "a top"],
    "Pants and Joggers": ["pants", "joggers", "sweatpants", "trousers"],
    # Levi's expansion.
    "Jeans": ["jeans", "pants"],
    "Jean Jackets": ["a denim jacket", "a jacket"],
    "Shirts": ["a shirt", "a button-down shirt", "a top"],
    # "Accessories" is genuinely heterogeneous at the Levi's category
    # level (backpacks, hats, belts, wallets, bandanas, even underwear --
    # verified against real scraped records, not assumed) -- unlike every
    # other category here, best_crop_for_category doesn't need the top
    # match to be one SPECIFIC phrasing, just any phrasing in this list,
    # so listing every real accessory type found lets FashionCLIP
    # classify whichever specific item is actually in a given photo
    # rather than forcing one label onto a mixed bucket.
    "Accessories": ["a hat", "a cap", "a belt", "a backpack", "a bag", "a wallet",
                     "a bandana", "underwear"],
    # Vans expansion -- "Hoodies and Jackets" bundles both garment types
    # under one category name (site's own grouping, not a scraper
    # miscategorization), so it needs both label families combined,
    # same reasoning as "Accessories" above: best_crop_for_category only
    # needs the top match to be ANY phrasing in the list.
    "Hoodies and Jackets": ["a hoodie", "a pullover", "a sweatshirt", "a jacket", "a coat"],
    # Dickies expansion.
    "Coats and Jackets": ["a jacket", "a coat"],
}


def crop_masked_region(image_pil: Image.Image, mask: np.ndarray) -> Image.Image | None:
    mask = mask.astype(bool)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return image_pil.crop((x1, y1, x2, y2))


def best_crop_for_category(
    image_path: Path,
    category: str,
    mask_generator: SAM2AutomaticMaskGenerator,
    processor,
    clip_model,
    device: str,
) -> Image.Image | None:
    """Returns the crop whose top FashionCLIP label is a phrasing of `category`,
    or None if no mask clears the area/confidence bar (caller should fall back
    to the original image)."""
    category_labels = CATEGORY_LABELS[category]
    candidate_labels = category_labels + DISTRACTOR_LABELS

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM))
    image_np = np.array(image)
    full_area = image_np.shape[0] * image_np.shape[1]

    masks = mask_generator.generate(image_np)
    crops, valid_masks = [], []
    for mask_data in masks:
        if mask_data["area"] < MIN_MASK_AREA:
            continue
        crop = crop_masked_region(image, mask_data["segmentation"])
        if crop is not None:
            crops.append(crop)
            valid_masks.append(mask_data)

    if not crops:
        return None

    inputs = processor(images=crops, text=candidate_labels, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=-1)  # [num_crops, num_labels]

    # Among crops whose top label is one of this category's phrasings, clears
    # the confidence bar, AND falls within the mid-sized area band (excludes
    # both "basically the whole photo" and "a tiny fabric fragment" — see
    # MIN/MAX_AREA_FRACTION comment above), pick the highest-scoring one.
    best_score, best_crop = -1.0, None
    for crop_idx, (crop, mask_data) in enumerate(zip(crops, valid_masks)):
        top_label_idx = probs[crop_idx].argmax().item()
        top_score = probs[crop_idx, top_label_idx].item()
        top_label = candidate_labels[top_label_idx]
        area_fraction = mask_data["area"] / full_area
        if (
            top_label in category_labels
            and top_score >= MIN_CONFIDENCE
            and MIN_AREA_FRACTION <= area_fraction <= MAX_AREA_FRACTION
            and top_score > best_score
        ):
            best_score = top_score
            best_crop = crop

    return best_crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="limit number of RECORDS processed")
    parser.add_argument("--brand", default="nike")
    args = parser.parse_args()

    # Deliberately no MPS branch — see MASK_GENERATOR_KWARGS comment above.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading FashionCLIP...")
    processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    clip_model = AutoModelForZeroShotImageClassification.from_pretrained("patrickjohncyh/fashion-clip")
    clip_model.to(device)

    print("Loading SAM2...")
    sam2 = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(sam2, **MASK_GENERATOR_KWARGS)

    database = load_records()
    todo = [
        r for r in database
        if r.get("brand") == args.brand
        and r.get("category") in CATEGORY_LABELS
        and "cropped_images" not in r
    ]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} records to segment (of {len(database)} total).\n")

    touched = {}
    processed = 0
    for idx, record in enumerate(todo, 1):
        category = record["category"]
        cropped_paths = []
        for img_path_str in record.get("images", []):
            img_path = Path(img_path_str)
            if not img_path.exists():
                continue
            try:
                crop = best_crop_for_category(img_path, category, mask_generator, processor, clip_model, device)
            except Exception as e:
                print(f"  [warn] {img_path}: {e}")
                crop = None

            if crop is None:
                cropped_paths.append(img_path_str)  # fall back to original, don't drop the image
                continue

            dest = img_path.with_name(img_path.stem + "_cropped.jpg")
            crop.convert("RGB").save(dest, quality=90)
            cropped_paths.append(str(dest))

        record["cropped_images"] = cropped_paths
        touched[record["product_code"]] = record
        processed += 1
        print(f"[{idx}/{len(todo)}] {record['product_code']} ({category}): "
              f"{len(cropped_paths)} images processed")

        if processed % CHECKPOINT_EVERY == 0:
            save_records_safe(touched)
            print(f"  [checkpoint] saved")

    save_records_safe(touched)
    print(f"\nDone. {processed} records segmented.")


if __name__ == "__main__":
    main()
