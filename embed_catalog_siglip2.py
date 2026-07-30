"""SigLIP2 frozen-baseline caption-retrieval benchmark, with Drive-backed
embedding caching (see embedding_cache.py).

Only images/captions not already in the cache get encoded -- re-running this
after a new scrape batch, or just to re-evaluate, does not re-touch Drive or
the GPU for anything already cached. Cache lives at
DATASET_ROOT/embeddings_cache/{model_name}/, keyed by resolved image path /
exact caption string.
"""

from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

import embedding_cache as ec


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = Path("/content/drive/MyDrive/apparel_dataset")
METADATA_PATH = DATASET_ROOT / "metadata.json"

OUTPUT_DIR = DATASET_ROOT / "siglip2_caption_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_IDS = {
    "finetuned": "srpone/zooclaw-fashionsiglip2",

    # Fair baseline because ZooClaw was fine-tuned from this model.
    "base": "google/siglip2-base-patch16-384",
}

IMAGE_BATCH_SIZE = 32
TEXT_BATCH_SIZE = 128
SIMILARITY_BATCH_SIZE = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print("Device:", DEVICE)
print("Dtype:", DTYPE)


# ============================================================
# Load metadata
# ============================================================

with METADATA_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)


def resolve_image_path(raw_path):
    raw_path = Path(raw_path)

    candidates = [
        raw_path,
        DATASET_ROOT / raw_path,
        DATASET_ROOT.parent / raw_path,
    ]

    if raw_path.parts and raw_path.parts[0] == DATASET_ROOT.name:
        candidates.append(DATASET_ROOT.joinpath(*raw_path.parts[1:]))

    if len(raw_path.parts) >= 4:
        candidates.append(DATASET_ROOT.joinpath(*raw_path.parts[-4:]))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return None


# ============================================================
# Build image records and candidate captions
# ============================================================

captions = []
caption_to_index = {}

image_records = []
missing_paths = []
skipped_products = []

for product in metadata:
    caption = str(product.get("caption", "")).strip()
    image_paths = product.get("images") or []

    if not caption or not image_paths:
        skipped_products.append(product.get("product_code"))
        continue

    if caption not in caption_to_index:
        caption_to_index[caption] = len(captions)
        captions.append(caption)

    caption_index = caption_to_index[caption]

    for raw_path in image_paths:
        resolved_path = resolve_image_path(raw_path)

        if resolved_path is None:
            missing_paths.append(raw_path)
            continue

        image_records.append({
            "image_path": str(resolved_path),
            "caption_index": caption_index,
            "caption": caption,
            "product_code": product.get("product_code", ""),
            "brand": product.get("brand", ""),
            "name": product.get("name", ""),
        })


if not image_records:
    raise RuntimeError("No valid images were found. Check DATASET_ROOT and metadata paths.")
if not captions:
    raise RuntimeError("No valid captions were found.")


print(f"Products in metadata:      {len(metadata):,}")
print(f"Candidate captions:        {len(captions):,}")
print(f"Evaluation images:         {len(image_records):,}")
print(f"Missing image paths:       {len(missing_paths):,}")
print(f"Skipped products:          {len(skipped_products):,}")

if missing_paths:
    missing_report = OUTPUT_DIR / "missing_image_paths.txt"
    missing_report.write_text("\n".join(map(str, missing_paths)), encoding="utf-8")
    print("Missing-path report:", missing_report)


# ============================================================
# Embedding functions
# ============================================================

def extract_embeddings(output):
    if torch.is_tensor(output):
        return output
    if getattr(output, "text_embeds", None) is not None:
        return output.text_embeds
    if getattr(output, "image_embeds", None) is not None:
        return output.image_embeds
    if getattr(output, "pooler_output", None) is not None:
        return output.pooler_output
    if isinstance(output, (tuple, list)) and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Could not extract embeddings from output type: {type(output)}")


def move_to_device(inputs):
    return {
        key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


@torch.inference_mode()
def encode_texts_raw(model, processor, texts):
    embeddings = []
    for start in tqdm(range(0, len(texts), TEXT_BATCH_SIZE), desc="Encoding captions (cache miss)"):
        batch_texts = texts[start:start + TEXT_BATCH_SIZE]
        inputs = processor(text=batch_texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        inputs = move_to_device(inputs)
        outputs = model.get_text_features(**inputs)
        batch_embeddings = extract_embeddings(outputs)
        batch_embeddings = F.normalize(batch_embeddings.float(), p=2, dim=-1)
        embeddings.append(batch_embeddings.cpu())
    return torch.cat(embeddings, dim=0)


@torch.inference_mode()
def encode_images_raw(model, processor, records):
    embeddings = []
    valid_records = []
    failed_images = []

    for start in tqdm(range(0, len(records), IMAGE_BATCH_SIZE), desc="Encoding images (cache miss)"):
        batch_records = records[start:start + IMAGE_BATCH_SIZE]
        images = []
        kept_records = []

        for record in batch_records:
            try:
                with Image.open(record["image_path"]) as image:
                    image = ImageOps.exif_transpose(image)
                    image = image.convert("RGB")
                    images.append(image)
                kept_records.append(record)
            except Exception as error:
                failed_images.append({"image_path": record["image_path"], "error": repr(error)})

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt")
        inputs = move_to_device(inputs)
        outputs = model.get_image_features(**inputs)
        batch_embeddings = extract_embeddings(outputs)
        batch_embeddings = F.normalize(batch_embeddings.float(), p=2, dim=-1)

        embeddings.append(batch_embeddings.cpu())
        valid_records.extend(kept_records)

    if not embeddings:
        raise RuntimeError("Every image failed to load or encode.")

    return torch.cat(embeddings, dim=0), valid_records, failed_images


def encode_texts_cached(model, processor, texts, model_name):
    cached_embeddings, cached_keys = ec.load_cache(DATASET_ROOT, model_name, "text")
    hit_pos, hit_rows, missing = ec.split_cached(texts, cached_keys)

    print(f"Text cache: {len(hit_pos)}/{len(texts)} hits, {len(missing)} to encode")

    if missing:
        new_embeddings = encode_texts_raw(model, processor, missing)
        merged_embeddings, merged_keys = ec.save_cache(
            DATASET_ROOT, model_name, "text",
            new_embeddings.numpy(), missing,
        )
    else:
        merged_embeddings, merged_keys = cached_embeddings, cached_keys

    row_of = {key: i for i, key in enumerate(merged_keys)}
    ordered = np.stack([merged_embeddings[row_of[t]] for t in texts])
    return torch.from_numpy(ordered)


def encode_images_cached(model, processor, records, model_name):
    keys = [r["image_path"] for r in records]
    cached_embeddings, cached_keys = ec.load_cache(DATASET_ROOT, model_name, "image")
    hit_pos, hit_rows, missing_keys = ec.split_cached(keys, cached_keys)

    print(f"Image cache: {len(hit_pos)}/{len(records)} hits, {len(missing_keys)} to encode")

    missing_key_set = set(missing_keys)
    missing_records = [r for r in records if r["image_path"] in missing_key_set]

    failed_images = []
    if missing_records:
        new_embeddings, valid_new_records, failed_images = encode_images_raw(model, processor, missing_records)
        new_keys = [r["image_path"] for r in valid_new_records]
        merged_embeddings, merged_keys = ec.save_cache(
            DATASET_ROOT, model_name, "image",
            new_embeddings.numpy(), new_keys,
        )
    else:
        merged_embeddings, merged_keys = cached_embeddings, cached_keys

    row_of = {key: i for i, key in enumerate(merged_keys)}
    failed_key_set = {f["image_path"] for f in failed_images}

    valid_records = [r for r in records if r["image_path"] in row_of]
    ordered = np.stack([merged_embeddings[row_of[r["image_path"]]] for r in valid_records])

    return torch.from_numpy(ordered), valid_records, failed_images


# ============================================================
# Image-to-caption evaluation
# ============================================================

def evaluate_image_to_caption(image_embeddings, text_embeddings, records, candidate_captions):
    if len(image_embeddings) != len(records):
        raise ValueError("Image embedding count does not match record count.")

    target_indices = torch.tensor([r["caption_index"] for r in records], dtype=torch.long)
    maximum_top_k = min(10, len(candidate_captions))

    all_ranks = []
    prediction_rows = []

    for start in tqdm(range(0, len(records), SIMILARITY_BATCH_SIZE), desc="Scoring images"):
        end = min(start + SIMILARITY_BATCH_SIZE, len(records))
        scores = image_embeddings[start:end] @ text_embeddings.T
        targets = target_indices[start:end]
        target_scores = scores[torch.arange(end - start), targets]
        ranks = (scores > target_scores[:, None]).sum(dim=1) + 1
        all_ranks.append(ranks)

        top_scores, top_indices = scores.topk(k=maximum_top_k, dim=1)

        for local_index, global_index in enumerate(range(start, end)):
            record = records[global_index]
            predicted_index = int(top_indices[local_index, 0])
            target_index = int(targets[local_index])

            row = {
                **record,
                "rank": int(ranks[local_index]),
                "target_score": float(target_scores[local_index]),
                "predicted_caption_index": predicted_index,
                "predicted_caption": candidate_captions[predicted_index],
                "predicted_score": float(top_scores[local_index, 0]),
                "correct_top1": predicted_index == target_index,
            }

            for k in (1, 5, 10):
                effective_k = min(k, maximum_top_k)
                row[f"correct_top{k}"] = bool((top_indices[local_index, :effective_k] == target_index).any())

            prediction_rows.append(row)

    ranks = torch.cat(all_ranks).float()

    metrics = {
        "num_images": int(len(records)),
        "num_candidate_captions": int(len(candidate_captions)),
        "recall_at_1": float((ranks <= 1).float().mean()),
        "recall_at_5": float((ranks <= min(5, len(candidate_captions))).float().mean()),
        "recall_at_10": float((ranks <= min(10, len(candidate_captions))).float().mean()),
        "mrr": float((1.0 / ranks).mean()),
        "median_rank": float(ranks.median()),
        "mean_rank": float(ranks.mean()),
    }

    return metrics, pd.DataFrame(prediction_rows)


# ============================================================
# Evaluate both models
# ============================================================

all_metrics = []

for model_name, model_id in MODEL_IDS.items():
    print("\n" + "=" * 80)
    print(f"Evaluating {model_name}: {model_id}")
    print("=" * 80)

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=DTYPE)
    model = model.to(DEVICE)
    model.eval()

    text_embeddings = encode_texts_cached(model, processor, captions, model_name)
    image_embeddings, valid_records, failed_images = encode_images_cached(model, processor, image_records, model_name)

    metrics, predictions = evaluate_image_to_caption(image_embeddings, text_embeddings, valid_records, captions)

    metrics["model_name"] = model_name
    metrics["model_id"] = model_id
    metrics["failed_images"] = len(failed_images)
    all_metrics.append(metrics)

    predictions.insert(0, "model_id", model_id)
    predictions.to_csv(OUTPUT_DIR / f"{model_name}_per_image_predictions.csv", index=False)

    if failed_images:
        pd.DataFrame(failed_images).to_csv(OUTPUT_DIR / f"{model_name}_failed_images.csv", index=False)

    with (OUTPUT_DIR / f"{model_name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nMetrics:")
    print(pd.Series(metrics).to_string())

    del model, processor, text_embeddings, image_embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Final comparison
# ============================================================

summary = pd.DataFrame(all_metrics)[[
    "model_name", "model_id", "num_images", "num_candidate_captions",
    "recall_at_1", "recall_at_5", "recall_at_10", "mrr", "median_rank", "mean_rank", "failed_images",
]]

for column in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr"):
    summary[column] *= 100.0

summary = summary.rename(columns={
    "recall_at_1": "R@1 (%)", "recall_at_5": "R@5 (%)", "recall_at_10": "R@10 (%)", "mrr": "MRR (%)",
})

summary.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

print("\nFinal comparison:")
print(summary)
print("\nResults saved to:")
print(OUTPUT_DIR)
