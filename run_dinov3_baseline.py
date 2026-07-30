"""Phase 3 step 1: frozen-DINOv3 held-out Recall@K baseline across all 6
brands, extracted from notebooks/base_dino_visual_search.ipynb sections 1-5
and 7 (skips the single-query visualization in sections 5-6, not needed for
a metrics run).

Drive-backed caching (already the notebook's own design, kept as-is here):
embeddings for the full catalog are saved once to
DATASET_ROOT/embedding_indexes/base_DINO/base_DINO_embeddings.pt (+ a
records.json alongside it for index alignment) and reloaded on every
subsequent run instead of re-encoding, as long as the image count matches
what's currently in metadata.json. Re-run after a scrape adds new images
and it rebuilds from scratch (no incremental delta yet -- unlike
embedding_cache.py's per-image cache used for the SigLIP2 script, this
matches the notebook's original coarser whole-index caching so Phase 3
stays consistent with the notebook it's cloned from).
"""

from pathlib import Path
from contextlib import nullcontext
import gc
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

# facebook/dinov3-vitb16-pretrain-lvd1689m is a gated HF repo -- you must
# accept Meta's license on the model page (huggingface.co/facebook/
# dinov3-vitb16-pretrain-lvd1689m) with your own HF account first, then
# either export HF_TOKEN before running this script, or run
# `huggingface_hub.login()` in a cell/shell once beforehand. No token ->
# this raises GatedRepoError (401) at the AutoImageProcessor/AutoModel
# .from_pretrained() calls below, nothing silent.
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    print("Logged into Hugging Face Hub via HF_TOKEN.")


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = Path("/content/drive/MyDrive/apparel_dataset")
METADATA_PATH = DATASET_ROOT / "metadata.json"

INDEX_LABEL = "base_DINO"
INDEX_DIR = DATASET_ROOT / "embedding_indexes" / INDEX_LABEL
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_PATH = INDEX_DIR / f"{INDEX_LABEL}_embeddings.pt"
RECORDS_PATH = INDEX_DIR / f"{INDEX_LABEL}_records.json"
CONFIG_PATH = INDEX_DIR / f"{INDEX_LABEL}_config.json"

MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"

IMAGE_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 512
SPLIT_SEED = 42
GALLERY_FRACTION = 0.80

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
AMP_DTYPE = torch.float16

print("Device:", DEVICE)
print("Model:", MODEL_ID)
print("Index directory:", INDEX_DIR)


# ============================================================
# Load metadata and resolve paths
# ============================================================

with METADATA_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)


def resolve_image_path(raw_path):
    raw_path = Path(raw_path)
    candidates = [raw_path, DATASET_ROOT / raw_path, DATASET_ROOT.parent / raw_path]
    if raw_path.parts and raw_path.parts[0] == DATASET_ROOT.name:
        candidates.append(DATASET_ROOT.joinpath(*raw_path.parts[1:]))
    if len(raw_path.parts) >= 4:
        candidates.append(DATASET_ROOT.joinpath(*raw_path.parts[-4:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


records = []
missing_paths = []

for product in metadata:
    product_code = str(product.get("product_code", "")).strip()
    image_paths = product.get("images") or []
    if not image_paths:
        continue
    for raw_path in image_paths:
        resolved_path = resolve_image_path(raw_path)
        if resolved_path is None:
            missing_paths.append(str(raw_path))
            continue
        records.append({
            "image_path": str(resolved_path),
            "product_code": product_code,
            "brand": product.get("brand", ""),
            "name": product.get("name", ""),
        })

print(f"Products in metadata: {len(metadata):,}")
print(f"Evaluation images:    {len(records):,}")
print(f"Missing image paths:  {len(missing_paths):,}")


# ============================================================
# Encoder + embedding helpers (only loaded if we need to (re)build)
# ============================================================

def load_rgb_image(image_path):
    with Image.open(image_path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def build_index():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=AMP_DTYPE if USE_AMP else torch.float32)
    model = model.to(DEVICE)
    model.eval()
    print("Loaded:", MODEL_ID)

    def extract_global_embedding(outputs):
        pooler_output = getattr(outputs, "pooler_output", None)
        features = pooler_output if pooler_output is not None else outputs.last_hidden_state[:, 0]
        return F.normalize(features.float(), dim=-1)

    @torch.inference_mode()
    def embed_pil_images(images):
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        amp_context = torch.autocast(device_type="cuda", dtype=AMP_DTYPE) if USE_AMP else nullcontext()
        with amp_context:
            outputs = model(**inputs)
        return extract_global_embedding(outputs).cpu()

    all_embeddings = []
    for start in tqdm(range(0, len(records), IMAGE_BATCH_SIZE), desc="Creating DINOv3 embeddings"):
        batch_records = records[start:start + IMAGE_BATCH_SIZE]
        images = [load_rgb_image(r["image_path"]) for r in batch_records]
        all_embeddings.append(embed_pil_images(images))

    embeddings = torch.cat(all_embeddings, dim=0).contiguous()
    if embeddings.shape[0] != len(records):
        raise RuntimeError(f"Embedding count {embeddings.shape[0]} != record count {len(records)}.")

    index_payload = {
        "label": INDEX_LABEL,
        "model_id": MODEL_ID,
        "normalized": True,
        "embedding_dimension": int(embeddings.shape[1]),
        "embeddings": embeddings,
    }
    torch.save(index_payload, EMBEDDINGS_PATH)

    with RECORDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    config = {
        "label": INDEX_LABEL, "model_id": MODEL_ID, "dataset_root": str(DATASET_ROOT),
        "metadata_path": str(METADATA_PATH), "number_of_images": len(records),
        "embedding_dimension": int(embeddings.shape[1]), "normalized": True,
        "similarity": "cosine via normalized dot product",
    }
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Saved embeddings:", EMBEDDINGS_PATH)
    print("Embedding shape: ", tuple(embeddings.shape))

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings, records


def load_or_build_index():
    if EMBEDDINGS_PATH.exists() and RECORDS_PATH.exists():
        payload = torch.load(EMBEDDINGS_PATH, map_location="cpu", weights_only=False)
        with RECORDS_PATH.open("r", encoding="utf-8") as f:
            cached_records = json.load(f)

        if payload["label"] == INDEX_LABEL and len(cached_records) == len(records):
            print(f"Cache hit: loading existing index ({len(cached_records):,} images) from Drive, no re-encoding.")
            return payload["embeddings"].float(), cached_records

        print(f"Cache stale (cached {len(cached_records):,} vs current {len(records):,} images) -- rebuilding.")

    print("No usable cache found -- encoding full catalog with DINOv3.")
    return build_index()


full_database_embeddings, full_database_records = load_or_build_index()


# ============================================================
# Held-out 80/20 split + evaluation
# ============================================================

import random
from collections import defaultdict

indices_by_product = defaultdict(list)
for index, record in enumerate(full_database_records):
    indices_by_product[str(record.get("product_code", "")).strip()].append(index)

rng = random.Random(SPLIT_SEED)
gallery_indices, test_indices = [], []

for product_code, product_indices in indices_by_product.items():
    product_indices = product_indices.copy()
    rng.shuffle(product_indices)
    num_images = len(product_indices)
    if num_images < 2:
        gallery_indices.extend(product_indices)
        continue
    num_test = max(1, round(num_images * (1 - GALLERY_FRACTION)))
    num_test = min(num_test, num_images - 1)
    test_indices.extend(product_indices[:num_test])
    gallery_indices.extend(product_indices[num_test:])

gallery_indices = sorted(gallery_indices)
test_indices = sorted(test_indices)

gallery_index_tensor = torch.tensor(gallery_indices, dtype=torch.long)
test_index_tensor = torch.tensor(test_indices, dtype=torch.long)

gallery_embeddings = full_database_embeddings[gallery_index_tensor]
gallery_records = [full_database_records[i] for i in gallery_indices]
heldout_query_embeddings = full_database_embeddings[test_index_tensor]
heldout_query_records = [full_database_records[i] for i in test_indices]

print("Full index:   ", len(full_database_records))
print("Gallery 80%:  ", len(gallery_records))
print("Test 20%:     ", len(heldout_query_records))

gallery_product_codes = np.array([str(r["product_code"]).strip() for r in gallery_records])
query_product_codes = np.array([str(r["product_code"]).strip() for r in heldout_query_records])

ranks = []
for start in tqdm(range(0, len(heldout_query_embeddings), EVAL_BATCH_SIZE), desc="Scoring held-out queries"):
    end = min(start + EVAL_BATCH_SIZE, len(heldout_query_embeddings))
    query_batch = F.normalize(heldout_query_embeddings[start:end].float(), dim=-1)
    similarity_matrix = query_batch @ gallery_embeddings.T
    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True).cpu().numpy()

    for row_index, ranked_gallery_indices in enumerate(sorted_indices):
        query_code = query_product_codes[start + row_index]
        ranked_codes = gallery_product_codes[ranked_gallery_indices]
        matching_positions = np.flatnonzero(ranked_codes == query_code)
        rank = int(matching_positions[0]) + 1 if len(matching_positions) else len(gallery_records) + 1
        ranks.append(rank)

ranks = np.asarray(ranks)

metrics = {
    "model_id": MODEL_ID,
    "num_queries": int(len(heldout_query_records)),
    "num_gallery": int(len(gallery_records)),
    "num_gallery_products": int(len(set(gallery_product_codes))),
    "recall_at_1": float(np.mean(ranks <= 1)),
    "recall_at_5": float(np.mean(ranks <= 5)),
    "recall_at_10": float(np.mean(ranks <= 10)),
    "mrr": float(np.mean(1.0 / ranks)),
    "median_rank": float(np.median(ranks)),
    "mean_rank": float(np.mean(ranks)),
}

print("\nbase_DINO: held-out 20% exact-product retrieval, full 6-brand catalog")
print(f"Queries:     {metrics['num_queries']:,}")
print(f"Gallery:     {metrics['num_gallery']:,}")
print(f"Candidates:  {metrics['num_gallery_products']:,} products")
print(f"R@1:         {metrics['recall_at_1'] * 100:.2f}%")
print(f"R@5:         {metrics['recall_at_5'] * 100:.2f}%")
print(f"R@10:        {metrics['recall_at_10'] * 100:.2f}%")
print(f"MRR:         {metrics['mrr'] * 100:.2f}%")
print(f"Median rank: {metrics['median_rank']:.1f}")
print(f"Mean rank:   {metrics['mean_rank']:.2f}")

metrics_path = INDEX_DIR / "held_out_eval_metrics.json"
with metrics_path.open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print("\nSaved metrics to:", metrics_path)
