"""Phase 4: the combined SigLIP2 + DINOv3 retrieval pipeline, per
docs/project_spec_v1.md sections 4.7-4.8 ("separate metadata/semantic/
identity indexes", "query facet parsing and multi-index fusion") and the
role split spelled out in finetune_siglip2_v3.py / dino_identity_finetune.py's
own docstrings: SigLIP2 narrows by category then by model-level semantic
identity (brand + product name + attributes, deliberately NOT colorway-
unique); DINOv3 picks the exact SKU/colorway within whatever SigLIP2
shortlisted. Neither encoder alone answers "which exact product is this,"
only both stages together do -- this script is that missing "together."

Three stages, in order:
1. Category classification (canonical_taxonomy_path leaf, e.g. "sneaker",
   "hoodie") -- SigLIP2 image embedding vs. the 13 canonical category text
   embeddings from docs/hierarchy.json. Hard-gates the next stage's search
   space to that category, which the roadmap's own design principle
   supports (categories were deliberately chosen to be visually
   non-overlapping -- see docs/roadmap.md's canonical-hierarchy section --
   so a category misclassification should be rare enough to gate on rather
   than just softly weight). evaluate_pipeline measures the real
   gate-exclusion rate rather than assuming this holds.
2. Semantic identity shortlist -- SigLIP2 image embedding vs. every
   model-level "identity" text embedding *within* the gated category (same
   identity string construction as finetune_siglip2_v3.py's
   build_training_labels, since that's the exact text space the fine-tuned
   text tower was trained against). Each matched identity string expands
   to every product_code sharing it (colorway siblings all read identically
   at this level by design -- see finetune_siglip2_v3.py item 8's module
   note), forming the DINOv3 candidate pool.
3. Exact-identity rerank -- DINOv3 image embedding vs. the candidate pool's
   product-level DINOv3 embeddings only (not the whole catalog), final
   ranking by cosine similarity. A same-model/different-colorway ambiguity
   flag is set when the top-2 result shares model_identity with the top-1
   and their DINOv3 scores are within AMBIGUITY_MARGIN -- the same
   diagnostic dino_identity_finetune.py's own eval reports, surfaced here
   per-query instead of only in aggregate.

Checkpoint selection is auto-detected (newest-trained-stage-first, per
model) rather than hardcoded, since both encoders are still mid-iteration
as of this writing -- rerun this script's index-build step after either
model's checkpoint improves and it picks the new one up automatically
(caches are invalidated by checkpoint path + catalog size, not by hand).

One real cross-storage wrinkle: DINOv3 has been training on Colab (Drive-
backed, DATASET_ROOT below), but SigLIP2 v3 was moved to Modal (a separate
Volume) after Colab's idle-disconnect problems. This script assumes both
checkpoints are reachable under one DATASET_ROOT -- pull the Modal one down
first if it's not already on Drive:
    modal volume get fashion-dataset \\
        apparel_dataset/finetuned_siglip2_hierarchical_v3 \\
        /content/drive/MyDrive/apparel_dataset/finetuned_siglip2_hierarchical_v3 -r

Usage:
    python3 hierarchical_retrieval_pipeline.py --image path/to/query.jpg
    python3 hierarchical_retrieval_pipeline.py --evaluate
"""

from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext

import argparse
import json
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

# facebook/dinov3-vitb16-pretrain-lvd1689m is a gated HF repo -- same
# HF_TOKEN convention as dino_identity_finetune.py / run_dinov3_baseline.py.
# Missed on this script's first Modal run (401 GatedRepoError) since none
# of the training scripts needed it locally the same way -- this eval is
# the first thing here that actually loads DINOv3 itself.
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    print("Logged into Hugging Face Hub via HF_TOKEN.")


# ============================================================
# Configuration
# ============================================================

# APPAREL_DATASET_ROOT override exists for local/non-Colab test runs (e.g.
# against the full local rsync copy) without touching the Colab default.
DATASET_ROOT = Path(os.environ.get("APPAREL_DATASET_ROOT", "/content/drive/MyDrive/apparel_dataset"))
METADATA_PATH = DATASET_ROOT / "metadata.json"
# Checks the script's own directory first (e.g. Colab drag-and-drop
# upload, where only single files land, not the docs/ subfolder), then
# falls back to the repo's real docs/ layout.
_HIERARCHY_CANDIDATES = [
    Path(__file__).parent / "hierarchy.json",
    Path(__file__).parent / "docs" / "hierarchy.json",
]
HIERARCHY_PATH = next((p for p in _HIERARCHY_CANDIDATES if p.is_file()), _HIERARCHY_CANDIDATES[0])

INDEX_DIR = DATASET_ROOT / "retrieval_indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

SIGLIP2_BASE_MODEL_ID = "google/siglip2-base-patch16-384"
DINOV3_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINOV3_PROJECTION_DIM = 256

# Preference order, most- to least-trained. First existing path wins.
SIGLIP2_CHECKPOINT_CANDIDATES = [
    DATASET_ROOT / "finetuned_siglip2_hierarchical_v3" / "stage2_lastnblocks_best",
    DATASET_ROOT / "finetuned_siglip2_hierarchical_v3" / "stage1_heads_best",
    DATASET_ROOT / "finetuned_siglip2_hierarchical_v2" / "stage2_lastblock_best",
    DATASET_ROOT / "finetuned_siglip2_hierarchical_v2" / "stage1_heads_best",
]
DINOV3_CHECKPOINT_CANDIDATES = [
    DATASET_ROOT / "finetuned_dinov3_identity_v1_supcon" / "stage2_lastblock_best",
    DATASET_ROOT / "finetuned_dinov3_identity_v1_supcon" / "stage1_head_best",
    DATASET_ROOT / "finetuned_dinov3_identity_v1_arcface" / "stage2_lastblock_best",
    DATASET_ROOT / "finetuned_dinov3_identity_v1_arcface" / "stage1_head_best",
]

TOP_IDENTITY_CANDIDATES = 10   # SigLIP2 stage: how many identity strings to expand into DINOv3 candidates
FINAL_TOP_K = 5
AMBIGUITY_MARGIN = 0.03        # DINOv3 cosine-similarity gap under which top-1/top-2 count as "too close to call"
# Bare category name, not a "a photo of a {category}" template -- matches
# what finetune_siglip2_v3.py's build_training_labels actually trained the
# text tower on for taxonomy nodes (raw strings like "sneaker", "hoodie"
# under the "generic" label kind, no caption-style prefix).
CATEGORY_PROMPT_TEMPLATE = "{category}"

VAL_IMAGES_PER_PRODUCT = 1
TEST_IMAGES_PER_PRODUCT = 1
SPLIT_SEED = 42  # matches dino_identity_finetune.py's split exactly, so eval here is apples-to-apples

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
USE_AMP = DEVICE == "cuda"  # torch.autocast(device_type="mps") is still flaky across ops; MPS just runs fp32


def autocast_context():
    return torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else nullcontext()


# ============================================================
# Shared helpers (duplicated from finetune_siglip2_v3.py /
# dino_identity_finetune.py on purpose -- every script in this repo is a
# standalone Colab-runnable unit, no local package imports between them)
# ============================================================

def normalize_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,.")


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


def load_rgb_image(image_path):
    with Image.open(image_path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def product_identity_and_category(product):
    """Model-level identity text (colorway-collapsed) -- identical
    construction to finetune_siglip2_v3.py's build_training_labels, since
    the SigLIP2 text tower was trained against exactly this string space --
    plus the canonical leaf category (structured_caption.canonical_taxonomy_path,
    falling back to the raw taxonomy_path leaf if a product predates
    build_hierarchy.py's canonicalization pass)."""
    structured = product.get("structured_caption") or {}
    positive_texts = [normalize_text(t) for t in structured.get("positive_texts", []) or [] if normalize_text(t)]
    positive_texts = [t for t in positive_texts if not SKU_TEXT_PATTERN.search(t)]
    taxonomy_path = [normalize_text(t) for t in structured.get("taxonomy_path", []) or [] if normalize_text(t)]
    canonical_path = structured.get("canonical_taxonomy_path") or []
    leaf_category = normalize_text(canonical_path[-1]) if canonical_path else (taxonomy_path[-1] if taxonomy_path else "apparel item")
    brand = display_brand(product.get("brand", ""))

    if len(positive_texts) >= 2:
        identity = positive_texts[-2]
    elif positive_texts:
        identity = positive_texts[-1]
    elif brand:
        identity = f"{brand} {leaf_category}"
    else:
        identity = leaf_category
    return normalize_text(identity), leaf_category, brand


# ============================================================
# Load metadata + catalog-wide product records
# ============================================================

with METADATA_PATH.open("r", encoding="utf-8") as f:
    _metadata = json.load(f)

CATALOG = {}          # product_code -> {brand, name, category, model_identity, images: [resolved paths]}
IMAGES_BY_PRODUCT = defaultdict(list)

for _product in _metadata:
    _product_code = normalize_text(_product.get("product_code", ""))
    if not _product_code:
        continue
    _identity, _category, _brand = product_identity_and_category(_product)
    _resolved_images = []
    for _raw_path in _product.get("images", []):
        _path = resolve_image_path(_raw_path)
        if _path is None:
            continue
        # Corrupted/truncated-file skip, same convention as
        # finetune_siglip2_v3.py / dino_identity_finetune.py -- without
        # this a single bad JPEG crashes the whole identity-index build
        # deep inside HierarchicalRetriever.__init__ instead of being
        # filtered out here at catalog-build time.
        try:
            with Image.open(_path) as _check_image:
                _check_image.verify()
        except Exception:
            continue
        _resolved_images.append(str(_path))
    if not _resolved_images:
        continue
    CATALOG[_product_code] = {
        "brand": _brand,
        "name": _product.get("name", ""),
        "category": _category,
        "model_identity": _identity,
        "images": _resolved_images,
    }
    IMAGES_BY_PRODUCT[_product_code] = _resolved_images

print(f"Catalog: {len(CATALOG):,} products with resolvable images")

IDENTITY_TO_PRODUCT_CODES = defaultdict(set)
for _code, _entry in CATALOG.items():
    IDENTITY_TO_PRODUCT_CODES[_entry["model_identity"]].add(_code)

with HIERARCHY_PATH.open("r", encoding="utf-8") as f:
    _hierarchy = json.load(f)
CANONICAL_CATEGORIES = sorted({category for categories in _hierarchy.values() for category in categories})
print(f"Canonical categories: {len(CANONICAL_CATEGORIES)}")


def make_view_split():
    """Identical logic/seed to dino_identity_finetune.py's make_view_split,
    reapplied per product_code here (rather than per image record) since
    that's the granularity this pipeline's gallery/query split needs --
    reproduces the exact same held-out test image per product so
    evaluate_pipeline's numbers are directly comparable to that script's
    own isolated eval."""
    rng = random.Random(SPLIT_SEED)
    gallery_images_by_product, test_image_by_product = {}, {}
    for code, images in IMAGES_BY_PRODUCT.items():
        images = images.copy()
        rng.shuffle(images)
        if len(images) < 2:
            gallery_images_by_product[code] = images
            continue
        max_holdout = len(images) - 1
        num_test = min(TEST_IMAGES_PER_PRODUCT, max_holdout)
        test_image_by_product[code] = images[:num_test]
        gallery_images_by_product[code] = images[num_test:]
    return gallery_images_by_product, test_image_by_product


# ============================================================
# Checkpoint auto-detection
# ============================================================

def pick_first_existing(candidates):
    for path in candidates:
        if (path / "model.safetensors").is_file() or (path / "backbone" / "model.safetensors").is_file():
            return path
    return None


def load_siglip2():
    checkpoint = pick_first_existing(SIGLIP2_CHECKPOINT_CANDIDATES)
    load_from = str(checkpoint) if checkpoint else SIGLIP2_BASE_MODEL_ID
    print(f"SigLIP2 checkpoint: {load_from}" + ("" if checkpoint else "  (no fine-tune found -- using base model)"))
    model = AutoModel.from_pretrained(load_from, torch_dtype=torch.float32).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(SIGLIP2_BASE_MODEL_ID)
    return model, processor, load_from


class DinoProjectionHead(nn.Module):
    """Must match dino_identity_finetune.py's ProjectionHead exactly --
    duplicated rather than imported per this repo's standalone-script
    convention (see module docstring)."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_dim = max(output_dim, input_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, pooled_features):
        return F.normalize(self.net(pooled_features), dim=-1)


def load_dinov3():
    # Explicit token= rather than relying solely on the module-level
    # login() call -- on the first Modal run, login() plus an already-set
    # HF_TOKEN env var interacted in a way that still left this specific
    # from_pretrained() call sending unauthenticated requests (401
    # GatedRepoError), even though login() appeared to run. Passing the
    # token directly to every gated-repo call removes the ambiguity.
    hf_token = os.environ.get("HF_TOKEN")
    checkpoint = pick_first_existing(DINOV3_CHECKPOINT_CANDIDATES)
    processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL_ID, token=hf_token)
    if checkpoint is not None:
        backbone = AutoModel.from_pretrained(checkpoint / "backbone", torch_dtype=torch.float32).to(DEVICE).eval()
        projection_head = DinoProjectionHead(backbone.config.hidden_size, DINOV3_PROJECTION_DIM).to(DEVICE)
        projection_head.load_state_dict(torch.load(checkpoint / "projection_head.pt", map_location=DEVICE))
        projection_head.eval()
        print(f"DINOv3 checkpoint: {checkpoint}")
        return backbone, projection_head, True, str(checkpoint)
    print("DINOv3 checkpoint: none found -- using frozen base model (raw pooled features, no projection)")
    backbone = AutoModel.from_pretrained(DINOV3_MODEL_ID, torch_dtype=torch.float32, token=hf_token).to(DEVICE).eval()
    return backbone, None, False, DINOV3_MODEL_ID


def dino_pooled_features(outputs):
    pooler_output = getattr(outputs, "pooler_output", None)
    return pooler_output if pooler_output is not None else outputs.last_hidden_state[:, 0]


@torch.inference_mode()
def embed_images_dino(backbone, projection_head, use_projection, processor, pil_images, batch_size=32):
    embeddings = []
    for start in range(0, len(pil_images), batch_size):
        batch = pil_images[start:start + batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with autocast_context():
            outputs = backbone(**inputs)
            raw = dino_pooled_features(outputs).float()
            batch_embeddings = projection_head(raw) if (use_projection and projection_head is not None) else F.normalize(raw, dim=-1)
        embeddings.append(batch_embeddings.cpu())
    return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, DINOV3_PROJECTION_DIM)


def extract_siglip_embeddings(output):
    """transformers-version-dependent: get_text_features/get_image_features
    return a bare tensor on some versions, a wrapped output object (e.g.
    BaseModelOutputWithPooling) on others -- same defensive unwrap as
    finetune_siglip2_v3.py's extract_embeddings, duplicated per this repo's
    standalone-script convention."""
    if torch.is_tensor(output):
        return output
    for attribute in ("text_embeds", "image_embeds", "pooler_output"):
        value = getattr(output, attribute, None)
        if value is not None:
            return value
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Cannot extract embeddings from {type(output)}")


@torch.inference_mode()
def embed_texts_siglip(model, processor, texts, batch_size=128, max_length=64):
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = processor(text=batch, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with autocast_context():
            batch_embeddings = extract_siglip_embeddings(model.get_text_features(**inputs)).float()
        embeddings.append(F.normalize(batch_embeddings, dim=-1).cpu())
    return torch.cat(embeddings, dim=0)


@torch.inference_mode()
def embed_images_siglip(model, processor, pil_images, batch_size=32):
    embeddings = []
    for start in range(0, len(pil_images), batch_size):
        batch = pil_images[start:start + batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with autocast_context():
            batch_embeddings = extract_siglip_embeddings(model.get_image_features(**inputs)).float()
        embeddings.append(F.normalize(batch_embeddings, dim=-1).cpu())
    return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, model.config.text_config.hidden_size)


# ============================================================
# Index building + caching (invalidated by checkpoint path + catalog size,
# not by hand -- rerun this script after either checkpoint improves and it
# picks up the change automatically)
# ============================================================

def _index_is_fresh(config_path, expected):
    if not config_path.is_file():
        return False
    try:
        current = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return all(current.get(k) == v for k, v in expected.items())


def build_or_load_semantic_index(model, processor, siglip2_checkpoint):
    config_path = INDEX_DIR / "semantic_index_config.json"
    identities = sorted(IDENTITY_TO_PRODUCT_CODES)
    expected = {"checkpoint": siglip2_checkpoint, "num_identities": len(identities), "num_categories": len(CANONICAL_CATEGORIES)}

    if _index_is_fresh(config_path, expected):
        payload = torch.load(INDEX_DIR / "semantic_identity_embeddings.pt", map_location="cpu", weights_only=False)
        category_payload = torch.load(INDEX_DIR / "semantic_category_embeddings.pt", map_location="cpu", weights_only=False)
        print("Semantic index: cache hit.")
        return payload["identities"], payload["embeddings"], category_payload["categories"], category_payload["embeddings"]

    print("Semantic index: (re)building...")
    identity_embeddings = embed_texts_siglip(model, processor, identities)
    torch.save({"identities": identities, "embeddings": identity_embeddings}, INDEX_DIR / "semantic_identity_embeddings.pt")

    category_prompts = [CATEGORY_PROMPT_TEMPLATE.format(category=c) for c in CANONICAL_CATEGORIES]
    category_embeddings = embed_texts_siglip(model, processor, category_prompts)
    torch.save({"categories": CANONICAL_CATEGORIES, "embeddings": category_embeddings}, INDEX_DIR / "semantic_category_embeddings.pt")

    config_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return identities, identity_embeddings, CANONICAL_CATEGORIES, category_embeddings


def build_or_load_identity_index(backbone, projection_head, use_projection, processor, dino_checkpoint, gallery_images_by_product):
    config_path = INDEX_DIR / "identity_index_config.json"
    product_codes = sorted(gallery_images_by_product)
    expected = {"checkpoint": dino_checkpoint, "use_projection": use_projection, "num_products": len(product_codes)}

    if _index_is_fresh(config_path, expected):
        payload = torch.load(INDEX_DIR / "identity_embeddings.pt", map_location="cpu", weights_only=False)
        print("Identity index: cache hit.")
        return payload["product_codes"], payload["embeddings"]

    print("Identity index: (re)building (this re-encodes up to 2 gallery images per product)...")
    flat_paths, owners = [], []
    for code in product_codes:
        for path in gallery_images_by_product[code][:2]:
            flat_paths.append(path)
            owners.append(code)

    all_embeddings = []
    valid_owners = []
    batch_size = 32
    for start in range(0, len(flat_paths), batch_size):
        batch_paths = flat_paths[start:start + batch_size]
        batch_owners = owners[start:start + batch_size]
        images, kept_owners = [], []
        for path, owner in zip(batch_paths, batch_owners):
            try:
                images.append(load_rgb_image(path))
                kept_owners.append(owner)
            except Exception as error:
                # Belt-and-suspenders: the catalog build already verify()s
                # every image, but that doesn't catch every corruption
                # mode -- skip rather than crash the whole index build over
                # one bad file.
                print(f"Skipped (unreadable): {path} ({error})")
        if not images:
            continue
        all_embeddings.append(embed_images_dino(backbone, projection_head, use_projection, processor, images, batch_size=batch_size))
        valid_owners.extend(kept_owners)
    all_embeddings = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, DINOV3_PROJECTION_DIM)

    embeddings_by_product = defaultdict(list)
    for embedding, code in zip(all_embeddings, valid_owners):
        embeddings_by_product[code].append(embedding)
    # Drop any product whose every gallery image failed to load (should be
    # rare given the catalog-build verify() pass, but a product left with
    # zero embeddings would otherwise crash the stack() below).
    product_codes = [code for code in product_codes if embeddings_by_product.get(code)]
    product_embeddings = torch.stack([
        F.normalize(torch.stack(embeddings_by_product[code]).mean(dim=0), dim=-1) for code in product_codes
    ])

    torch.save({"product_codes": product_codes, "embeddings": product_embeddings}, INDEX_DIR / "identity_embeddings.pt")
    config_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return product_codes, product_embeddings


# ============================================================
# The pipeline itself
# ============================================================

class HierarchicalRetriever:
    def __init__(self):
        self.siglip2_model, self.siglip2_processor, siglip2_checkpoint = load_siglip2()
        self.dino_backbone, self.dino_head, self.dino_use_projection, dino_checkpoint = load_dinov3()
        self.dino_processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL_ID, token=os.environ.get("HF_TOKEN"))

        self.gallery_images_by_product, self.test_image_by_product = make_view_split()

        self.identities, self.identity_embeddings, self.categories, self.category_embeddings = build_or_load_semantic_index(
            self.siglip2_model, self.siglip2_processor, siglip2_checkpoint,
        )
        self.identity_category = {}
        for identity in self.identities:
            codes = IDENTITY_TO_PRODUCT_CODES[identity]
            if codes:
                self.identity_category[identity] = CATALOG[next(iter(codes))]["category"]

        self.gallery_product_codes, self.gallery_embeddings = build_or_load_identity_index(
            self.dino_backbone, self.dino_head, self.dino_use_projection, self.dino_processor,
            dino_checkpoint, self.gallery_images_by_product,
        )
        self.gallery_index_by_code = {code: i for i, code in enumerate(self.gallery_product_codes)}

    def predict_category(self, siglip_image_embedding):
        similarity = (siglip_image_embedding @ self.category_embeddings.T).squeeze(0)
        ranked = torch.argsort(similarity, descending=True)
        top_category = self.categories[ranked[0].item()]
        margin = float(similarity[ranked[0]] - similarity[ranked[1]]) if len(ranked) > 1 else float("inf")
        return top_category, margin

    def shortlist_identities(self, siglip_image_embedding, category, top_k):
        if category is not None:
            candidate_indices = [i for i, identity in enumerate(self.identities) if self.identity_category.get(identity) == category]
        else:
            candidate_indices = list(range(len(self.identities)))
        if not candidate_indices:
            candidate_indices = list(range(len(self.identities)))

        index_tensor = torch.tensor(candidate_indices, dtype=torch.long)
        sub_embeddings = self.identity_embeddings[index_tensor]
        similarity = (siglip_image_embedding @ sub_embeddings.T).squeeze(0)
        k = min(top_k, len(candidate_indices))
        top_local = torch.topk(similarity, k).indices.tolist()

        candidate_product_codes = set()
        for local_index in top_local:
            global_index = candidate_indices[local_index]
            candidate_product_codes.update(IDENTITY_TO_PRODUCT_CODES[self.identities[global_index]])
        return candidate_product_codes

    def rerank_by_identity(self, dino_image_embedding, candidate_product_codes, final_top_k):
        available = [code for code in candidate_product_codes if code in self.gallery_index_by_code]
        if not available:
            return []
        indices = torch.tensor([self.gallery_index_by_code[code] for code in available], dtype=torch.long)
        similarity = (dino_image_embedding @ self.gallery_embeddings[indices].T).squeeze(0)
        order = torch.argsort(similarity, descending=True)
        ranked = [(available[i], float(similarity[i])) for i in order.tolist()]
        return ranked[:final_top_k]

    def retrieve(self, image_path, use_category_gate=True, top_identity_candidates=TOP_IDENTITY_CANDIDATES, final_top_k=FINAL_TOP_K):
        image = load_rgb_image(image_path)

        siglip_embedding = embed_images_siglip(self.siglip2_model, self.siglip2_processor, [image])
        dino_embedding = embed_images_dino(self.dino_backbone, self.dino_head, self.dino_use_projection, self.dino_processor, [image])

        predicted_category, category_margin = self.predict_category(siglip_embedding)
        gate_category = predicted_category if use_category_gate else None

        candidates = self.shortlist_identities(siglip_embedding, gate_category, top_identity_candidates)
        ranked = self.rerank_by_identity(dino_embedding, candidates, final_top_k)

        results = []
        for rank, (code, score) in enumerate(ranked, start=1):
            entry = CATALOG[code]
            results.append({
                "rank": rank, "product_code": code, "brand": entry["brand"], "name": entry["name"],
                "category": entry["category"], "model_identity": entry["model_identity"],
                "dino_identity_score": score,
            })

        ambiguous = False
        if len(results) >= 2:
            same_model = results[0]["model_identity"] == results[1]["model_identity"]
            close_scores = (results[0]["dino_identity_score"] - results[1]["dino_identity_score"]) < AMBIGUITY_MARGIN
            ambiguous = same_model and close_scores

        return {
            "predicted_category": predicted_category, "category_margin": category_margin,
            "num_identity_candidates": len(candidates), "results": results,
            "same_model_different_colorway_ambiguous": ambiguous,
        }

    # --------------------------------------------------------
    # End-to-end held-out evaluation
    # --------------------------------------------------------

    def evaluate(self, use_category_gate=True, top_identity_candidates=TOP_IDENTITY_CANDIDATES):
        queries = [(code, path) for code, paths in self.test_image_by_product.items() for path in paths]
        print(f"Evaluating {len(queries):,} held-out queries (category gate: {use_category_gate})...")

        ranks = []
        gate_exclusions = 0
        identity_shortlist_misses = 0

        for true_code, image_path in queries:
            image = load_rgb_image(image_path)
            siglip_embedding = embed_images_siglip(self.siglip2_model, self.siglip2_processor, [image])
            dino_embedding = embed_images_dino(self.dino_backbone, self.dino_head, self.dino_use_projection, self.dino_processor, [image])

            predicted_category, _ = self.predict_category(siglip_embedding)
            true_category = CATALOG[true_code]["category"]
            gate_category = predicted_category if use_category_gate else None
            if use_category_gate and predicted_category != true_category:
                gate_exclusions += 1

            candidates = self.shortlist_identities(siglip_embedding, gate_category, top_identity_candidates)
            if true_code not in candidates:
                identity_shortlist_misses += 1

            available = [code for code in candidates if code in self.gallery_index_by_code]
            if not available:
                ranks.append(len(self.gallery_product_codes) + 1)
                continue
            indices = torch.tensor([self.gallery_index_by_code[code] for code in available], dtype=torch.long)
            similarity = (dino_embedding @ self.gallery_embeddings[indices].T).squeeze(0)
            order = torch.argsort(similarity, descending=True).tolist()
            ranked_codes = [available[i] for i in order]
            rank = ranked_codes.index(true_code) + 1 if true_code in ranked_codes else len(self.gallery_product_codes) + 1
            ranks.append(rank)

        ranks = np.asarray(ranks, dtype=float)
        metrics = {
            "num_queries": len(queries),
            "category_gate_exclusion_rate": gate_exclusions / len(queries) if use_category_gate else None,
            "identity_shortlist_miss_rate": identity_shortlist_misses / len(queries),
            "recall_at_1": float(np.mean(ranks <= 1)),
            "recall_at_5": float(np.mean(ranks <= 5)),
            "recall_at_10": float(np.mean(ranks <= 10)),
            "mrr": float(np.mean(1.0 / ranks)),
            "median_rank": float(np.median(ranks)),
            "mean_rank": float(np.mean(ranks)),
        }
        return metrics


def print_result(query_path, result):
    print(f"\nQuery: {query_path}")
    print(f"Predicted category: {result['predicted_category']} (margin {result['category_margin']:.3f})")
    print(f"Identity candidates considered: {result['num_identity_candidates']}")
    for entry in result["results"]:
        print(f"  #{entry['rank']}  {entry['brand']} {entry['name']}  [{entry['product_code']}]  "
              f"score={entry['dino_identity_score']:.4f}")
    if result["same_model_different_colorway_ambiguous"]:
        print("  ! Top-2 results are the same model, different colorway, within the ambiguity margin.")


def print_metrics(title, metrics):
    print(f"\n{title}")
    if metrics["category_gate_exclusion_rate"] is not None:
        print(f"Category-gate exclusion rate: {metrics['category_gate_exclusion_rate'] * 100:.2f}%")
    print(f"Identity-shortlist miss rate: {metrics['identity_shortlist_miss_rate'] * 100:.2f}%")
    print(f"R@1:  {metrics['recall_at_1'] * 100:.2f}%")
    print(f"R@5:  {metrics['recall_at_5'] * 100:.2f}%")
    print(f"R@10: {metrics['recall_at_10'] * 100:.2f}%")
    print(f"MRR:  {metrics['mrr'] * 100:.2f}%")
    print(f"Median rank: {metrics['median_rank']:.1f}")
    print(f"Mean rank: {metrics['mean_rank']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to a single query image.")
    parser.add_argument("--evaluate", action="store_true", help="Run end-to-end held-out evaluation.")
    parser.add_argument("--no-category-gate", action="store_true", help="Disable stage-1 category gating (fallback comparison).")
    args = parser.parse_args()

    retriever = HierarchicalRetriever()

    if args.image:
        result = retriever.retrieve(args.image, use_category_gate=not args.no_category_gate)
        print_result(args.image, result)

    if args.evaluate:
        gated_metrics = retriever.evaluate(use_category_gate=True)
        print_metrics("End-to-end held-out eval -- WITH category gate", gated_metrics)

        ungated_metrics = retriever.evaluate(use_category_gate=False)
        print_metrics("End-to-end held-out eval -- WITHOUT category gate (fallback comparison)", ungated_metrics)

        with (INDEX_DIR / "pipeline_eval_metrics.json").open("w", encoding="utf-8") as f:
            json.dump({"with_category_gate": gated_metrics, "without_category_gate": ungated_metrics}, f, indent=2)

    if not args.image and not args.evaluate:
        print("Nothing to do -- pass --image PATH and/or --evaluate.")
