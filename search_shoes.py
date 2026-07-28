"""
Search the apparel catalog for a query image.

Runs in Colab against Google Drive, not local disk -- mount Drive first,
then run embed_catalog.py to build the index before using this.

Usage:
    # Basic — embed the whole image (good if the image is already a cropped item)
    python search_shoes.py --query path/to/shoe.jpg --top 5

    # With SAM2 segmentation — segments the image first, classifies each segment,
    # then runs retrieval on the shoe crop. Requires a SAM2 checkpoint on Drive.
    python search_shoes.py --query path/to/outfit.jpg --segment --top 5

    # Write results to a file
    python search_shoes.py --query path/to/shoe.jpg --out results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DATASET_ROOT = Path("/content/drive/MyDrive/apparel_dataset")
EMBEDDINGS_PATH = DATASET_ROOT / "catalog_embeddings.npy"
METADATA_PATH = DATASET_ROOT / "catalog_metadata.json"
MODEL_NAME = "hf-hub:Marqo/marqo-fashionSigLIP"
SAM2_CHECKPOINT = Path("/content/drive/MyDrive/apparel_dataset/checkpoints/sam2.1_hiera_large.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SHOE_LABELS = ["sneakers", "shoes", "boots", "running shoes", "basketball shoes", "sandals"]
MIN_SEGMENT_AREA = 2000


# ──────────────────────────────────────────────
# Model loading (cached across calls in same process)
# ──────────────────────────────────────────────

_clip_model = None
_clip_preprocess = None
_clip_device = None


def get_clip():
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is None:
        import open_clip
        _clip_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
        _clip_model = _clip_model.to(_clip_device).eval()
        print(f"FashionSigLIP loaded on {_clip_device}")
    return _clip_model, _clip_preprocess, _clip_device


# ──────────────────────────────────────────────
# SAM2 segmentation + shoe crop extraction
# ──────────────────────────────────────────────

def segment_and_crop_shoes(image_pil: Image.Image) -> list[Image.Image]:
    """Use SAM2 + FashionSigLIP zero-shot to find shoe segments in an outfit image."""
    import open_clip
    import numpy as np
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    sam2 = build_sam2(str(SAM2_CONFIG), str(SAM2_CHECKPOINT), device=device)
    mask_gen = SAM2AutomaticMaskGenerator(sam2)

    image_np = np.array(image_pil)
    masks = mask_gen.generate(image_np)
    print(f"SAM2 found {len(masks)} segments")

    model, preprocess, clip_device = get_clip()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    all_labels = SHOE_LABELS + ["shirt", "pants", "jeans", "jacket", "background"]
    text_tokens = tokenizer(all_labels).to(clip_device)
    with torch.no_grad():
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    shoe_crops = []
    for mask_data in masks:
        if mask_data["area"] < MIN_SEGMENT_AREA:
            continue

        seg = mask_data["segmentation"].astype(bool)
        ys, xs = np.where(seg)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        crop = image_pil.crop((x1, y1, x2, y2))
        tensor = preprocess(crop).unsqueeze(0).to(clip_device)

        with torch.no_grad():
            img_feat = model.encode_image(tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        sims = (img_feat @ text_feats.T).squeeze(0)
        probs = sims.softmax(dim=-1)
        top_idx = probs.argmax().item()
        top_label = all_labels[top_idx]
        top_score = probs[top_idx].item()

        if top_label in SHOE_LABELS and top_score > 0.3:
            print(f"  Shoe segment: '{top_label}' ({top_score:.2f}), bbox=({x1},{y1},{x2},{y2})")
            shoe_crops.append(crop)

    return shoe_crops


# ──────────────────────────────────────────────
# Embedding + search
# ──────────────────────────────────────────────

def embed_query(image_pil: Image.Image) -> np.ndarray:
    model, preprocess, device = get_clip()
    tensor = preprocess(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy()[0]


def search(query_vec: np.ndarray, embeddings: np.ndarray, metadata: list[dict], top_k: int) -> list[dict]:
    sims = embeddings @ query_vec  # cosine similarity (both L2-normed)
    ranked = np.argsort(sims)[::-1]

    # Deduplicate by product_code — keep best-matching image per product
    seen = {}
    for idx in ranked:
        rec = metadata[idx]
        code = rec["product_code"]
        if code not in seen:
            seen[code] = {"score": float(sims[idx]), **rec}
        if len(seen) >= top_k:
            break

    return list(seen.values())[:top_k]


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Path to query image")
    parser.add_argument("--top", type=int, default=5, help="Number of results to return")
    parser.add_argument("--segment", action="store_true", help="Use SAM2 to extract shoe crops first")
    parser.add_argument("--out", help="Write results to this JSON file")
    args = parser.parse_args()

    if not EMBEDDINGS_PATH.exists():
        print(f"ERROR: {EMBEDDINGS_PATH} not found. Run embed_catalog.py first.")
        return

    print(f"Loading index ({EMBEDDINGS_PATH})...")
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    print(f"Index: {len(metadata)} images, embeddings shape={embeddings.shape}")

    image = Image.open(args.query).convert("RGB")

    if args.segment:
        crops = segment_and_crop_shoes(image)
        if not crops:
            print("No shoe segments found — falling back to full image.")
            crops = [image]
    else:
        crops = [image]

    all_results = []
    for i, crop in enumerate(crops):
        print(f"\nSearching with crop {i + 1}/{len(crops)}...")
        qvec = embed_query(crop)
        results = search(qvec, embeddings, metadata, top_k=args.top)
        all_results.extend(results)

    # If multiple crops, re-rank and deduplicate by product_code
    if len(crops) > 1:
        seen = {}
        for r in sorted(all_results, key=lambda x: -x["score"]):
            if r["product_code"] not in seen:
                seen[r["product_code"]] = r
        all_results = list(seen.values())[: args.top]

    print(f"\n{'─'*60}")
    print(f"Top {len(all_results)} matches for: {args.query}")
    print(f"{'─'*60}")
    for rank, r in enumerate(all_results, 1):
        print(f"#{rank}  {r['product_code']}  score={r['score']:.4f}")
        print(f"     brand: {r.get('brand', '')}")
        print(f"     label: {r.get('display_label', '')}")
        print(f"     url:  {r['product_url']}")
        print(f"     img:  {r['image_path']}")
        print()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
