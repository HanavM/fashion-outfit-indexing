"""Tests the actual claim this project has been making about scaling the
catalog: "new products can be added via embedding alone, no retraining
needed" (spec section 8.1's "unseen-product enrollment" split: "entire
identities are excluded from training and added to the gallery only at
evaluation").

**Why this needed its own script, not just rerunning dino_identity_
finetune.py's existing eval**: checked `make_view_split()` in
dino_identity_finetune.py directly -- it splits by held-out IMAGE
(VAL_IMAGES_PER_PRODUCT / TEST_IMAGES_PER_PRODUCT per product), not by
held-out PRODUCT. Every product_code appears in train, val, AND test,
just with different photos. That's the right methodology for spec
8.1's *separate* "closed-set product recognition" split ("products
appear in training, query images are held out") -- a real, valid,
already-measured result (56.55% R@1) -- but it does NOT test the
enrollment claim at all, since the model saw every identity's OTHER
images during training. This was stated as already-proven earlier in
this project's own conversation history; it wasn't -- this script is
the correction, not a restatement.

**The genuinely free way to test this**: the current production DINOv3
checkpoint (`finetuned_dinov3_identity_v1_supcon/stage2_lastblock_best`)
was trained BEFORE Champion and Levi's existed in this catalog (added
2026-08-01/02, ~378 records, this session). Those are real, naturally
unseen identities -- no retraining needed, just embed them with the
existing frozen checkpoint and measure retrieval accuracy among them,
compared against a same-size sample of brands the checkpoint WAS
trained on. If there's no meaningful gap, that's real evidence for the
"enrollment" claim. If there is one, that's honest, useful evidence
against it -- report whichever comes back, don't assume the favorable
answer.

Methodology: for each of the "unseen" brands (Champion, Levi's) and a
size-matched control sample from an "already-seen" brand, build a
gallery (all-but-1 image per product) + 1 held-out query image per
product, embed both with the frozen checkpoint (no gradient, no
training), and compute standard exact-SKU Recall@1/5/10/MRR/median rank
within each group's own gallery separately (not mixed -- this isolates
"can it discriminate among unseen identities" from any catalog-size
confound).

Requires the real DINOv3 checkpoint reachable under DATASET_ROOT/
APPAREL_DATASET_ROOT (same convention as hierarchical_retrieval_
pipeline.py) -- import from there directly rather than duplicating the
load/embed code, since this project already reuses across scripts when
a full working implementation exists elsewhere (e.g. catalog_query_
search.py importing FreeTextVisualSearch from free_text_visual_
search.py).

Usage:
    python3 unseen_product_enrollment_eval.py
    python3 unseen_product_enrollment_eval.py --unseen-brands champion,levis --control-brand nike
"""

import argparse
import random
from collections import defaultdict

import numpy as np
import torch

from hierarchical_retrieval_pipeline import (
    CATALOG, DEVICE, load_dinov3, embed_images_dino, load_rgb_image,
)

SEED = 42
TEST_IMAGES_PER_PRODUCT = 1


def make_gallery_query_split(product_codes, rng):
    """Same view-level held-out-image split as dino_identity_finetune.py's
    make_view_split, scoped to a specific subset of products -- 1 query
    image per product, the rest form that product's gallery entry."""
    gallery_paths_by_product, query_by_product = {}, {}
    for code in product_codes:
        images = list(CATALOG[code]["images"])
        if len(images) < 2:
            continue  # need at least 1 gallery image + 1 query image
        rng.shuffle(images)
        query_by_product[code] = images[0]
        gallery_paths_by_product[code] = images[1:]
    return gallery_paths_by_product, query_by_product


@torch.inference_mode()
def evaluate_group(label, product_codes, backbone, head, use_projection, processor, rng):
    gallery_by_product, query_by_product = make_gallery_query_split(product_codes, rng)
    eligible_codes = list(query_by_product.keys())
    print(f"\n[{label}] {len(eligible_codes)} products eligible (>=2 images) out of {len(product_codes)} total")
    if len(eligible_codes) < 2:
        print(f"[{label}] too few eligible products to evaluate, skipping")
        return None

    gallery_embeddings, gallery_codes = [], []
    for code in eligible_codes:
        images = [load_rgb_image(p) for p in gallery_by_product[code][:2]]  # cap at 2 for speed, matches hierarchical_retrieval_pipeline.py's own identity-index convention
        embeddings = embed_images_dino(backbone, head, use_projection, processor, images)
        gallery_embeddings.append(torch.nn.functional.normalize(embeddings.mean(dim=0, keepdim=True), dim=-1))
        gallery_codes.append(code)
    gallery_matrix = torch.cat(gallery_embeddings, dim=0)

    ranks = []
    for code in eligible_codes:
        query_image = load_rgb_image(query_by_product[code])
        query_embedding = embed_images_dino(backbone, head, use_projection, processor, [query_image])
        similarity = (query_embedding @ gallery_matrix.T).squeeze(0)
        order = torch.argsort(similarity, descending=True).tolist()
        ranked_codes = [gallery_codes[i] for i in order]
        rank = ranked_codes.index(code) + 1
        ranks.append(rank)

    ranks = np.asarray(ranks, dtype=float)
    metrics = {
        "label": label, "num_products": len(eligible_codes),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }
    print(f"[{label}] R@1={metrics['recall_at_1']*100:.2f}%  R@5={metrics['recall_at_5']*100:.2f}%  "
          f"R@10={metrics['recall_at_10']*100:.2f}%  MRR={metrics['mrr']*100:.2f}%  "
          f"median_rank={metrics['median_rank']:.1f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unseen-brands", type=str, default="champion,levis",
                         help="Comma-separated brands added AFTER the current DINOv3 checkpoint was trained -- "
                              "genuinely never seen during training, the real enrollment test.")
    parser.add_argument("--control-brand", type=str, default="nike",
                         help="A brand the checkpoint WAS trained on, size-matched, for comparison.")
    args = parser.parse_args()

    unseen_brands = set(b.strip() for b in args.unseen_brands.split(","))
    products_by_brand = defaultdict(list)
    for code, entry in CATALOG.items():
        products_by_brand[entry["brand"]].append(code)

    unseen_codes = [c for brand in unseen_brands for c in products_by_brand.get(brand, [])]
    control_pool = products_by_brand.get(args.control_brand, [])

    rng = random.Random(SEED)
    control_codes = rng.sample(control_pool, min(len(unseen_codes), len(control_pool))) if control_pool else []

    print(f"Unseen brands ({', '.join(sorted(unseen_brands))}): {len(unseen_codes)} products")
    print(f"Control brand ({args.control_brand}), size-matched sample: {len(control_codes)} products")

    backbone, head, use_projection, checkpoint = load_dinov3()
    from transformers import AutoImageProcessor
    import os
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov3-vitb16-pretrain-lvd1689m", token=os.environ.get("HF_TOKEN"))

    unseen_metrics = evaluate_group("UNSEEN (never trained on)", unseen_codes, backbone, head, use_projection, processor, rng)
    control_metrics = evaluate_group(f"CONTROL ({args.control_brand}, trained on)", control_codes, backbone, head, use_projection, processor, rng)

    if unseen_metrics and control_metrics:
        gap = control_metrics["recall_at_1"] - unseen_metrics["recall_at_1"]
        print(f"\nR@1 gap (control - unseen): {gap*100:+.2f}pt")
        if abs(gap) < 0.05:
            print("Small gap (<5pt) -- real evidence the 'embed new products without retraining' claim holds.")
        else:
            print("Real gap (>=5pt) -- the enrollment claim does NOT hold as cleanly as assumed; "
                  "report this honestly rather than the favorable assumption.")
