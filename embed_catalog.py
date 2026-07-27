"""
Embed all adidas_catalog images with Marqo FashionSigLIP and save a searchable index.

Install deps first:
    pip install open-clip-torch

Outputs:
    catalog_embeddings.npy  — float32 array [N_images, 768]
    catalog_metadata.json   — list of dicts, index-aligned with embeddings
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CATALOG_DIRS = [Path("adidas_catalog"), Path("nike_catalog"), Path("newbalance_catalog")]
PRODUCTS_JSON = {
    "adidas_catalog": Path("adidas_products.json"),
    "nike_catalog": Path("nike_products.json"),
    "newbalance_catalog": Path("newbalance_products.json"),
}
EMBEDDINGS_OUT = Path("catalog_embeddings.npy")
METADATA_OUT = Path("catalog_metadata.json")
MODEL_NAME = "hf-hub:Marqo/marqo-fashionSigLIP"
BATCH_SIZE = 32


def load_model():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}")
    return model, preprocess, device


def load_url_map(catalog_dir: Path) -> dict[str, str]:
    """product_code -> product_url, sourced from the scraper's own JSON output."""
    products_file = PRODUCTS_JSON.get(catalog_dir.name)
    if not products_file or not products_file.exists():
        return {}
    products = json.loads(products_file.read_text())
    return {p["product_code"]: p["product_url"] for p in products}


def collect_images(catalog_dir: Path) -> list[dict]:
    """Walk catalog_dir/{slug}/{product_code}/image_N.jpg and return metadata records."""
    if not catalog_dir.exists():
        return []
    url_map = load_url_map(catalog_dir)
    records = []
    for slug_dir in sorted(catalog_dir.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name
        for code_dir in sorted(slug_dir.iterdir()):
            if not code_dir.is_dir():
                continue
            product_code = code_dir.name
            images = sorted(
                code_dir.glob("image_*.jpg"),
                key=lambda p: int(p.stem.split("_")[1]),
            )
            product_url = url_map.get(
                product_code,
                f"https://www.adidas.com/us/{slug}/{product_code}.html",
            )
            for img_path in images:
                records.append(
                    {
                        "image_path": str(img_path),
                        "slug": slug,
                        "product_code": product_code,
                        "image_index": int(img_path.stem.split("_")[1]),
                        "product_url": product_url,
                    }
                )
    return records


def embed_images(records: list[dict], model, preprocess, device: str) -> np.ndarray:
    all_embeddings = []
    total = len(records)

    for batch_start in range(0, total, BATCH_SIZE):
        batch = records[batch_start : batch_start + BATCH_SIZE]
        tensors = []
        valid_idx = []

        for i, rec in enumerate(batch):
            try:
                img = Image.open(rec["image_path"]).convert("RGB")
                tensors.append(preprocess(img))
                valid_idx.append(i)
            except Exception as e:
                print(f"  Skipping {rec['image_path']}: {e}")

        if not tensors:
            # Fill with zeros for skipped images so indices stay aligned
            all_embeddings.extend([np.zeros(768, dtype=np.float32)] * len(batch))
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize

        feats_np = feats.cpu().float().numpy()

        # Re-insert zeros for any skipped positions
        result = np.zeros((len(batch), feats_np.shape[1]), dtype=np.float32)
        for out_pos, orig_pos in enumerate(valid_idx):
            result[orig_pos] = feats_np[out_pos]

        all_embeddings.append(result)

        done = min(batch_start + BATCH_SIZE, total)
        print(f"  [{done}/{total}] embedded")

    return np.vstack(all_embeddings)


def main():
    print("Scanning catalogs...")
    records = []
    for catalog_dir in CATALOG_DIRS:
        dir_records = collect_images(catalog_dir)
        print(f"  {catalog_dir}: {len(dir_records)} images")
        records.extend(dir_records)
    print(f"Found {len(records)} images across all catalogs")

    model, preprocess, device = load_model()

    print("Embedding images...")
    embeddings = embed_images(records, model, preprocess, device)

    print(f"Saving {EMBEDDINGS_OUT} ({embeddings.shape}) ...")
    np.save(EMBEDDINGS_OUT, embeddings)

    print(f"Saving {METADATA_OUT} ...")
    with open(METADATA_OUT, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Done. {len(records)} images embedded.")
    print(f"  Embeddings: {EMBEDDINGS_OUT}  shape={embeddings.shape}")
    print(f"  Metadata:   {METADATA_OUT}")


if __name__ == "__main__":
    main()
