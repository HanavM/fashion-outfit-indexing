"""A/B: does a masked garment crop beat a plain bbox crop for identification?

    .venv/bin/python crop_style_ab.py --n 120

## Why this exists

`label_outfit_products.py` cuts crops from the stored bboxes, because the
proposer's own masked crops were written on Modal during the corpus
re-detection and only 13 of 20,681 survive locally. The difference is not
cosmetic: `garment_proposer` blanks every pixel outside the garment
*inside* its bounding box, and that structural background removal is
precisely what made its precision beat SAM2's (91% vs ~48%).

The product labels those bbox crops produced agree with the detector's own
garment group only **38.6%** of the time — below the 47.0% a constant
"tops" guess would score. Before concluding that consumer-crop
identification does not work, the shortcut I introduced has to be ruled
out, because "the crop carried a bedroom wall into the encoder" and "the
model cannot bridge the domain gap" predict the same bad number.

So: same garments, same pipeline, two crop styles, one paired comparison.

## What is being measured

Garment-GROUP agreement between the detector's category and the retrieved
product's canonical taxonomy group. This is a **proxy**, and a weak one in
a specific direction: neither side is ground truth (the detector's labels
are unvalidated model output, and identification is what is under test).
It cannot say a label is correct. It can say the two disagree, and if a
sneaker crop retrieves a t-shirt then at least one of them is wrong.

Its value is that it costs nothing and needs no human. `pair_eval.py` is
the honest measurement; this is the one that can be run every time
something changes.
"""

import argparse
import base64
import collections
import io
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("APPAREL_DATASET_ROOT", str(REPO_ROOT / "apparel_dataset"))

OUTFIT_METADATA = REPO_ROOT / "outfit_dataset" / "metadata.json"
DEFAULT_URL = "https://hanavm--fashion-serve-fashionservice-api.modal.run"

DET2GROUP = {"sneaker": "footwear", "loafer": "footwear", "pants": "bottoms",
             "shorts": "bottoms", "jacket": "outerwear", "hat": "accessories",
             "socks": "accessories", "shirt": "tops", "t-shirt": "tops",
             "sweater": "tops", "hoodie": "tops", "sweatshirt": "tops",
             "tank top": "tops"}

# Canonical terms that appear in metadata's taxonomy paths but not in
# docs/hierarchy.json. Kept explicit rather than guessed by substring, so a
# mapping error is visible as a line here rather than as a silent miscount.
EXTRA_GROUPS = {
    "tee": "tops", "polo": "tops", "polo shirt": "tops", "button-up shirt": "tops",
    "crewneck sweatshirt": "tops",
    "low-top shoe": "footwear", "high-top shoe": "footwear", "slip-on shoe": "footwear",
    "skate shoe": "footwear", "mule": "footwear", "sandal": "footwear",
    "boot": "footwear", "slipper": "footwear", "boat shoe": "footwear",
    "belt": "accessories", "backpack": "accessories", "crossbody bag": "accessories",
    "bandana": "accessories", "tote bag": "accessories", "wallet": "accessories",
    "bag": "accessories",
    "underwear": "bottoms", "boxers": "bottoms", "boxer brief": "bottoms",
}


def group_lookup():
    hierarchy = json.loads((REPO_ROOT / "docs" / "hierarchy.json").read_text())
    mapping = {}
    for group, categories in hierarchy.items():
        for category, leaves in categories.items():
            mapping[category.lower()] = group
            for leaf in leaves:
                mapping[leaf.lower()] = group
    mapping.update(EXTRA_GROUPS)
    return mapping


def load_api_key():
    key = os.environ.get("FASHION_API_KEY")
    if key:
        return key
    for line in (REPO_ROOT / ".env").read_text().splitlines():
        if line.strip().startswith("FASHION_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("FASHION_API_KEY not found")


def identify(session, url, key, payload, top_k=5):
    response = session.post(
        f"{url.rstrip('/')}/identify",
        json={"image_base64": base64.b64encode(payload).decode(), "top_k": top_k},
        headers={"Authorization": f"Bearer {key}"}, timeout=300)
    response.raise_for_status()
    return response.json()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=120, help="photos to run (both arms each)")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--url", default=os.environ.get("FASHION_API_URL", DEFAULT_URL))
    args = ap.parse_args()

    import requests
    from PIL import Image, ImageOps
    from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

    import garment_proposer

    to_group = group_lookup()
    key = load_api_key()
    session = requests.Session()

    records = json.loads(OUTFIT_METADATA.read_text())
    usable = [r for r in records
              if (r.get("images") and (REPO_ROOT / r["images"][0]).exists()
                  and r.get("detected_items"))]
    rng = random.Random(args.seed)
    rng.shuffle(usable)
    photos = usable[:args.n]
    print(f"  {len(photos)} photos, both crop styles each\n")

    print("  loading proposer + FashionCLIP…")
    processor, model = garment_proposer.load_human_parser("cpu")
    clip_processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    clip_model = AutoModelForZeroShotImageClassification.from_pretrained(
        "patrickjohncyh/fashion-clip")

    stats = {"bbox": collections.Counter(), "masked": collections.Counter()}
    per_group = {"bbox": collections.defaultdict(collections.Counter),
                 "masked": collections.defaultdict(collections.Counter)}
    started = time.time()
    pairs = 0

    for index, record in enumerate(photos, 1):
        source = REPO_ROOT / record["images"][0]
        try:
            proposals = garment_proposer.propose_garment_items(
                str(source), processor, model, clip_processor, clip_model, "cpu")
        except Exception as error:  # noqa: BLE001
            print(f"  [{index}] proposer failed: {error}")
            continue
        if not proposals:
            continue
        with Image.open(source) as handle:
            full = ImageOps.exif_transpose(handle).convert("RGB")

        for proposal in proposals[:3]:
            detected = proposal.get("category")
            want = DET2GROUP.get(detected)
            if not want or not proposal.get("bbox"):
                continue
            left, top, right, bottom = proposal["bbox"]
            if right - left < 8 or bottom - top < 8:
                continue

            arms = {}
            buffer = io.BytesIO()
            full.crop((left, top, right, bottom)).save(buffer, "JPEG", quality=90)
            arms["bbox"] = buffer.getvalue()
            buffer = io.BytesIO()
            proposal["crop"].convert("RGB").save(buffer, "JPEG", quality=90)
            arms["masked"] = buffer.getvalue()

            try:
                for arm, payload in arms.items():
                    result = identify(session, args.url, key, payload)
                    hits = result.get("results") or []
                    if not hits:
                        continue
                    got = to_group.get(str(hits[0].get("category")).lower())
                    stats[arm]["n"] += 1
                    stats[arm]["agree"] += int(got == want)
                    stats[arm]["gate"] += int(bool(
                        (result.get("garment_gate") or {}).get("looks_like_clothing")))
                    per_group[arm][want]["n"] += 1
                    per_group[arm][want]["agree"] += int(got == want)
            except Exception as error:  # noqa: BLE001
                print(f"  [{index}] identify failed: {error}")
                continue
            pairs += 1

        if index % 20 == 0:
            elapsed = time.time() - started
            print(f"  [{index}/{len(photos)}] {pairs} paired crops  "
                  f"{elapsed/60:.1f} min")

    print(f"\n  PAIRED COMPARISON — {pairs} garments, both arms\n")
    print(f"  {'arm':8} {'n':>5} {'group agrees':>13} {'gate passes':>13}")
    for arm in ("bbox", "masked"):
        counts = stats[arm]
        if not counts["n"]:
            continue
        print(f"  {arm:8} {counts['n']:5d} {counts['agree']/counts['n']:12.1%} "
              f"{counts['gate']/counts['n']:12.1%}")

    if stats["bbox"]["n"] and stats["masked"]["n"]:
        delta = (stats["masked"]["agree"] / stats["masked"]["n"]
                 - stats["bbox"]["agree"] / stats["bbox"]["n"])
        print(f"\n  masked − bbox: {delta:+.1%}")
        print("\n  " + ("Masking is a real effect -- the full corpus should be "
                        "re-labelled from proposer crops." if delta > 0.05 else
                        "Masking does NOT explain the failure. The bbox shortcut is "
                        "not\n  the cause, so the gap is the model on consumer crops."))

    print(f"\n  {'group':12} {'bbox':>8} {'masked':>8}")
    for group in sorted(set(per_group["bbox"]) | set(per_group["masked"])):
        row = []
        for arm in ("bbox", "masked"):
            counts = per_group[arm][group]
            row.append(f"{counts['agree']/counts['n']:7.1%}" if counts["n"] else "      -")
        print(f"  {group:12} {row[0]:>8} {row[1]:>8}")

    print("\n  Proxy only: neither side is ground truth. It cannot say a label is\n"
          "  correct, only that the two disagree. pair_eval.py is the honest one.")


if __name__ == "__main__":
    main()
