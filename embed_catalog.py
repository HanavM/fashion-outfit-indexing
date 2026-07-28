"""
Embed all apparel_dataset images with a SigLIP-family model and save a searchable index.

Runs in Colab against Google Drive, not local disk -- mount Drive first:

    from google.colab import drive
    drive.mount('/content/drive')

Then either `!pip install -q -U open-clip-torch` and run this file directly
(`!python embed_catalog.py`), or `sys.path.append('/content')` after
uploading this file and `from embed_catalog import main; main()`.

Outputs (written under DATASET_ROOT, alongside metadata.json):
    catalog_embeddings.npy  -- float32 array [N_images, embedding_dim]
    catalog_metadata.json   -- list of dicts, index-aligned with embeddings

NOTE: this embeds with the frozen pretrained Marqo FashionSigLIP checkpoint,
same as classify_views.py/search_shoes.py currently use. Once the SigLIP2
HSC fine-tune (notebooks/fashionsiglip2_hsc_finetune.ipynb) has a real
trained checkpoint at DATASET_ROOT/finetuned_siglip2_hierarchical/best_model,
repoint MODEL_NAME there -- see project roadmap Phase 1.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

DATASET_ROOT = Path("/content/drive/MyDrive/apparel_dataset")
METADATA_PATH = DATASET_ROOT / "metadata.json"

EMBEDDINGS_OUT = DATASET_ROOT / "catalog_embeddings.npy"
METADATA_OUT = DATASET_ROOT / "catalog_metadata.json"

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


def resolve_image_path(raw_path: str) -> Path | None:
    """
    metadata.json's `images` entries carry inconsistent prefixes -- some
    products still have a stale "shoe_dataset/..." prefix baked in from
    before the apparel_dataset rename (~563/1115 products), others have
    the correct "apparel_dataset/..." prefix. Real files always live at
    DATASET_ROOT/<brand>/<slug>/<product_code>/image_N.jpg, so reconstruct
    from the last 4 path components regardless of what prefix is present.
    """
    raw_path = Path(raw_path)

    candidates = [raw_path, DATASET_ROOT / raw_path]

    if raw_path.parts and raw_path.parts[0] == DATASET_ROOT.name:
        candidates.append(DATASET_ROOT.joinpath(*raw_path.parts[1:]))

    if len(raw_path.parts) >= 4:
        candidates.append(DATASET_ROOT.joinpath(*raw_path.parts[-4:]))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return None


def collect_images() -> list[dict]:
    """Read every product's images out of apparel_dataset/metadata.json, all 6 brands."""
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    records = []
    missing_paths = []

    for product in metadata:
        product_code = str(product.get("product_code", "")).strip()
        brand = str(product.get("brand", "")).strip()
        structured_caption = product.get("structured_caption") or {}

        for image_index, raw_path in enumerate(product.get("images") or []):
            image_path = resolve_image_path(raw_path)

            if image_path is None:
                missing_paths.append(raw_path)
                continue

            records.append(
                {
                    "image_path": str(image_path),
                    "brand": brand,
                    "product_code": product_code,
                    "image_index": image_index,
                    "product_url": product.get("product_url", ""),
                    "taxonomy_path": structured_caption.get("taxonomy_path", []),
                    "display_label": (structured_caption.get("positive_texts") or [None])[-1],
                }
            )

    if missing_paths:
        print(f"  {len(missing_paths)} image paths could not be resolved (sample: {missing_paths[:3]})")

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
                img = Image.open(rec["image_path"])
                img = ImageOps.exif_transpose(img).convert("RGB")
                tensors.append(preprocess(img))
                valid_idx.append(i)
            except Exception as e:
                print(f"  Skipping {rec['image_path']}: {e}")

        if not tensors:
            all_embeddings.append(np.zeros((len(batch), 768), dtype=np.float32))
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize

        feats_np = feats.cpu().float().numpy()

        result = np.zeros((len(batch), feats_np.shape[1]), dtype=np.float32)
        for out_pos, orig_pos in enumerate(valid_idx):
            result[orig_pos] = feats_np[out_pos]

        all_embeddings.append(result)

        done = min(batch_start + BATCH_SIZE, total)
        print(f"  [{done}/{total}] embedded")

    return np.vstack(all_embeddings)


def main():
    print("Scanning apparel_dataset...")
    records = collect_images()
    print(f"Found {len(records)} images across all brands")

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
