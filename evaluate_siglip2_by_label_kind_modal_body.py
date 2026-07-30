"""Breaks the v3 SigLIP2 checkpoint's exact-label R@1 (18.83% category-
scoped) down by label kind (generic/attribute/brand/model), per direct
user request: "captions are simple and very feature oriented" -- is the
model actually weak everywhere, or is it strong on some kinds and dragged
down by others? evaluate_exact_retrieval in finetune_siglip2_v3.py only
ever scores against `exact_label` (the "model" kind's target string), so
this number didn't exist anywhere yet.

Reuses build_training_labels / make_view_split verbatim from
finetune_siglip2_v3.py (same SEED=42, same TEST_IMAGES_PER_PRODUCT=2) so
the test split here is the *exact same held-out images* the 18.83% number
came from -- this is a breakdown of that same result, not a new eval on
different data.

Methodology per kind (generic/attribute/brand/model): candidate texts =
every unique label of that kind across the catalog (e.g. every unique
attribute-kind string like "black sneaker", every unique brand-kind string
like "Nike sneaker"). A test image can have multiple valid candidates of a
given kind (e.g. several attribute strings all describe the same product);
rank = position of the *best-scoring* valid candidate, matching standard
multi-label retrieval evaluation and the same "count higher-scoring
candidates than the target" logic evaluate_exact_retrieval already uses,
generalized from one target to a valid-set. A product with zero entries of
a given kind (e.g. no `defining_features` -> no attribute-kind entries) is
skipped for that kind's eval, not counted as a miss.

Runs on Modal against the fashion-dataset Volume -- same checkpoint the
training run wrote to /data/apparel_dataset/finetuned_siglip2_hierarchical_v3.
"""

from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext

import json
import re

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

DATASET_ROOT = Path("/data/apparel_dataset")
METADATA_PATH = DATASET_ROOT / "metadata.json"

BASE_MODEL_ID = "google/siglip2-base-patch16-384"
CHECKPOINT_CANDIDATES = [
    DATASET_ROOT / "finetuned_siglip2_hierarchical_v3" / "stage2_lastnblocks_best",
    DATASET_ROOT / "finetuned_siglip2_hierarchical_v3" / "stage1_heads_best",
]

SEED = 42
VAL_IMAGES_PER_PRODUCT = 2
TEST_IMAGES_PER_PRODUCT = 2

TEXT_BATCH_SIZE = 128
IMAGE_EVAL_BATCH_SIZE = 32
SIMILARITY_BATCH_SIZE = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

LABEL_KINDS = ("generic", "attribute", "brand", "model")


def normalize_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,.")


def normalize_label_key(text):
    return normalize_text(text).lower()


def display_brand(brand):
    brand = normalize_text(brand)
    if not brand:
        return ""
    known = {
        "adidas": "Adidas", "nike": "Nike", "puma": "Puma", "reebok": "Reebok",
        "asics": "ASICS", "vans": "Vans", "converse": "Converse", "salomon": "Salomon",
        "saucony": "Saucony", "new balance": "New Balance", "gap": "Gap",
        "pacsun": "PacSun", "skechers": "Skechers",
    }
    return known.get(brand.lower(), brand.title())


SKU_TEXT_PATTERN = re.compile(r"\bsku\b", re.IGNORECASE)


def build_training_labels(product):
    """Verbatim from finetune_siglip2_v3.py -- must match exactly, since
    this determines what each label "kind" contains and thus what this
    breakdown is actually measuring."""
    structured = product.get("structured_caption") or {}

    positive_texts = [normalize_text(t) for t in structured.get("positive_texts", []) or [] if normalize_text(t)]
    positive_texts = [t for t in positive_texts if not SKU_TEXT_PATTERN.search(t)]
    taxonomy_path = [normalize_text(t) for t in structured.get("taxonomy_path", []) or [] if normalize_text(t)]
    attributes = structured.get("attributes", {}) or {}

    brand = display_brand(product.get("brand", ""))
    leaf_category = taxonomy_path[-1] if taxonomy_path else "apparel item"

    entries = []
    seen = set()

    def add(text, kind):
        text = normalize_text(text)
        if not text:
            return
        key = normalize_label_key(text)
        if key in seen:
            return
        seen.add(key)
        entries.append({"text": text, "key": key, "kind": kind})

    for node in taxonomy_path:
        add(node, "generic")

    for facet in ("color", "material", "fit", "pattern", "closure", "silhouette", "length"):
        for value in attributes.get(facet, []) or []:
            add(f"{value} {leaf_category}", "attribute")

    for feature_entry in attributes.get("defining_features", []) or []:
        feature_text = normalize_text(feature_entry.get("feature") or "")
        if not feature_text:
            continue
        add(f"{feature_text} {leaf_category}", "attribute")

    if brand:
        add(f"{brand} {leaf_category}", "brand")

    text_count = len(positive_texts)
    for position, text_item in enumerate(positive_texts):
        if text_count <= 1:
            kind = "model"
        else:
            fraction = position / (text_count - 1)
            if fraction < 0.2:
                kind = "generic"
            elif fraction < 0.45:
                kind = "attribute"
            elif fraction < 0.7:
                kind = "brand"
            else:
                kind = "model"
        add(text_item, kind)

    if len(positive_texts) >= 2:
        identity = positive_texts[-2]
    elif positive_texts:
        identity = positive_texts[-1]
    elif brand:
        identity = f"{brand} {leaf_category}"
    else:
        identity = leaf_category
    identity = normalize_text(identity)
    exact_label = identity

    return {
        "entries": entries,
        "identity": identity,
        "exact_label": exact_label,
        "category": leaf_category,
        "colors": attributes.get("color", []) or [],
        "materials": attributes.get("material", []) or [],
    }


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


with METADATA_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

records = []
missing_paths = []

for product in metadata:
    product_code = normalize_text(product.get("product_code", ""))
    if not product.get("structured_caption") or not product_code:
        continue
    label_data = build_training_labels(product)
    for raw_path in product.get("images", []):
        image_path = resolve_image_path(raw_path)
        if image_path is None:
            missing_paths.append(str(raw_path))
            continue
        try:
            with Image.open(image_path) as check_image:
                check_image.verify()
        except Exception:
            missing_paths.append(str(image_path))
            continue
        records.append({
            "image_path": str(image_path),
            "product_code": product_code,
            "training_labels": label_data["entries"],
        })

print(f"Images: {len(records):,}  Missing/corrupted: {len(missing_paths):,}")


def make_view_split(image_records):
    grouped = defaultdict(list)
    for record in image_records:
        grouped[record["product_code"]].append(record)

    train, validation, test = [], [], []
    import random
    rng = random.Random(SEED)

    for product_code, product_records in grouped.items():
        product_records = product_records.copy()
        rng.shuffle(product_records)
        max_holdout = max(0, len(product_records) - 1)
        num_test = min(TEST_IMAGES_PER_PRODUCT, max_holdout)
        remaining_after_test = len(product_records) - num_test
        num_val = min(VAL_IMAGES_PER_PRODUCT, max(0, remaining_after_test - 1))

        test.extend(product_records[:num_test])
        validation.extend(product_records[num_test:num_test + num_val])
        train.extend(product_records[num_test + num_val:])

    return train, validation, test


_, _, test_records = make_view_split(records)
print(f"Test images (held out, same split as training): {len(test_records):,}")


checkpoint = None
for candidate in CHECKPOINT_CANDIDATES:
    if (candidate / "model.safetensors").is_file():
        checkpoint = candidate
        break
load_from = str(checkpoint) if checkpoint else BASE_MODEL_ID
print(f"Checkpoint: {load_from}" + ("" if checkpoint else " (WARNING: no v3 checkpoint found, using base model)"))

model = AutoModel.from_pretrained(load_from, torch_dtype=torch.float32).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)


def extract_embeddings(output):
    if torch.is_tensor(output):
        return output
    for attribute in ("text_embeds", "image_embeds", "pooler_output"):
        value = getattr(output, attribute, None)
        if value is not None:
            return value
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Cannot extract embeddings from {type(output)}")


def autocast_context():
    return torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else nullcontext()


def move_to_device(inputs):
    return {k: v.to(DEVICE, non_blocking=True) for k, v in inputs.items()}


@torch.inference_mode()
def encode_texts(texts):
    embeddings = []
    for start in tqdm(range(0, len(texts), TEXT_BATCH_SIZE), desc="Encoding texts"):
        batch_texts = texts[start:start + TEXT_BATCH_SIZE]
        inputs = processor(text=batch_texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        inputs = move_to_device(inputs)
        with autocast_context():
            outputs = model.get_text_features(**inputs)
            batch_embeddings = extract_embeddings(outputs)
        embeddings.append(F.normalize(batch_embeddings.float(), p=2, dim=-1).cpu())
    return torch.cat(embeddings, dim=0)


@torch.inference_mode()
def encode_images(image_records):
    embeddings, valid_records = [], []
    for start in tqdm(range(0, len(image_records), IMAGE_EVAL_BATCH_SIZE), desc="Encoding images"):
        record_batch = image_records[start:start + IMAGE_EVAL_BATCH_SIZE]
        images, kept_records = [], []
        for record in record_batch:
            try:
                with Image.open(record["image_path"]) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    images.append(image.copy())
                kept_records.append(record)
            except Exception as error:
                print("Skipped:", record["image_path"], error)
        if not images:
            continue
        inputs = processor(images=images, return_tensors="pt")
        inputs = move_to_device(inputs)
        with autocast_context():
            outputs = model.get_image_features(**inputs)
            batch_embeddings = extract_embeddings(outputs)
        embeddings.append(F.normalize(batch_embeddings.float(), p=2, dim=-1).cpu())
        valid_records.extend(kept_records)
    return torch.cat(embeddings, dim=0), valid_records


# ============================================================
# Per-label-kind evaluation
# ============================================================

print("\nEncoding all held-out test images once (reused across all 4 kinds)...")
image_embeddings, valid_test_records = encode_images(test_records)


def evaluate_kind(kind):
    candidate_by_key = {}
    for record in test_records:
        for entry in record["training_labels"]:
            if entry["kind"] == kind:
                candidate_by_key[entry["key"]] = entry["text"]
    # Also pull candidates from the full catalog (not just test-split
    # records) so the candidate pool matches "every unique label of this
    # kind in the catalog," not just ones that happen to appear on a
    # held-out image -- otherwise a kind's candidate pool would be
    # artificially small and R@1 inflated.
    for record in records:
        for entry in record["training_labels"]:
            if entry["kind"] == kind and entry["key"] not in candidate_by_key:
                candidate_by_key[entry["key"]] = entry["text"]

    candidate_keys = list(candidate_by_key.keys())
    candidate_texts = [candidate_by_key[k] for k in candidate_keys]
    key_to_index = {k: i for i, k in enumerate(candidate_keys)}

    text_embeddings = encode_texts(candidate_texts)

    ranks = []
    num_skipped = 0
    for image_embedding, record in zip(image_embeddings, valid_test_records):
        valid_indices = [key_to_index[e["key"]] for e in record["training_labels"] if e["kind"] == kind and e["key"] in key_to_index]
        if not valid_indices:
            num_skipped += 1
            continue
        similarities = image_embedding @ text_embeddings.T
        target_score = similarities[valid_indices].max()
        rank = int((similarities > target_score).sum().item()) + 1
        ranks.append(rank)

    if not ranks:
        return {"num_images": 0, "num_skipped": num_skipped, "num_candidates": len(candidate_texts)}

    ranks_tensor = torch.tensor(ranks, dtype=torch.float32)
    return {
        "num_images": len(ranks),
        "num_skipped": num_skipped,
        "num_candidates": len(candidate_texts),
        "recall_at_1": float((ranks_tensor <= 1).float().mean()),
        "recall_at_5": float((ranks_tensor <= 5).float().mean()),
        "recall_at_10": float((ranks_tensor <= 10).float().mean()),
        "mrr": float((1.0 / ranks_tensor).mean()),
        "median_rank": float(ranks_tensor.median()),
        "mean_rank": float(ranks_tensor.mean()),
    }


results = {}
for kind in LABEL_KINDS:
    print(f"\n=== Evaluating kind: {kind} ===")
    metrics = evaluate_kind(kind)
    results[kind] = metrics
    if metrics["num_images"] == 0:
        print(f"No test images had a '{kind}' label -- skipped entirely ({metrics['num_skipped']} skipped).")
        continue
    print(f"Images evaluated: {metrics['num_images']:,}  (skipped, no '{kind}' label: {metrics['num_skipped']:,})")
    print(f"Candidates: {metrics['num_candidates']:,}")
    print(f"R@1:  {metrics['recall_at_1'] * 100:.2f}%")
    print(f"R@5:  {metrics['recall_at_5'] * 100:.2f}%")
    print(f"R@10: {metrics['recall_at_10'] * 100:.2f}%")
    print(f"MRR:  {metrics['mrr'] * 100:.2f}%")
    print(f"Median rank: {metrics['median_rank']:.1f}")
    print(f"Mean rank: {metrics['mean_rank']:.2f}")

output_path = DATASET_ROOT / "finetuned_siglip2_hierarchical_v3" / "eval_by_label_kind.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    json.dump({"checkpoint": load_from, "results": results}, f, indent=2)
print(f"\nSaved: {output_path}")
