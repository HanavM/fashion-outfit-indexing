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
1. **Real HSC (Hierarchical Softmax Classification) climbing** (added
   2026-08-01, per notebooks/fashionsiglip2_hsc_finetune.ipynb's own
   "Algorithm 1 from the HSC paper" -- this project had already
   implemented genuine HSC once before, this pipeline just wasn't using
   it). SigLIP2 image embedding scored against every LEAF of the full
   docs/hierarchy.json tree (group -> category -> fine leaf, ~42 leaves,
   not just the 13 categories), softmaxed into a real probability
   distribution, summed up through every ancestor, then climbed from the
   most probable leaf toward the root until an ancestor's aggregated
   probability clears HSC_THRESHOLD. This backs off to a broader, more-
   confident prediction (category -> group -> no restriction at all)
   instead of blindly committing to a specific leaf/category regardless
   of margin, which a flat top-1 classification can't do (a real ~0.05-
   margin leaf call was observed in production output before this
   change -- confidently wrong specificity, not backed off). When the
   category gate is on, this climbed node's `allowed_categories` (itself
   if it's already a category, all categories under it if it's a group or
   root) restricts stage 2's search space, instead of a single hard
   category. evaluate() measures the real gate-exclusion rate and how
   often climbing lands at each tree level, rather than assuming either.
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
from tqdm.auto import tqdm
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

# SigLIP2 stage: how many identity strings to expand into DINOv3
# candidates. Was 10; widened after the real-checkpoint Phase 4 eval
# (docs/eval_log.md, 2026-07-30) found the identity-shortlist miss rate
# (43-52%) is the dominant bottleneck -- DINOv3's rerank is already ~65%
# accurate conditional on getting a fair candidate set, so giving it more
# candidates to work with should matter more than anything else right now.
TOP_IDENTITY_CANDIDATES = 25
FINAL_TOP_K = 5
AMBIGUITY_MARGIN = 0.03        # DINOv3 cosine-similarity gap under which top-1/top-2 count as "too close to call"

# Open-set rejection (spec section 7: the system must be able to say
# "unknown" instead of always forcing a top-1). None = disabled by default
# -- deliberately NOT shipped with an assumed "calibrated" number, since
# real calibration needs a held-out split with genuinely off-catalog
# query images (spec section 8.1's "open-set rejection" split), which
# this project doesn't have yet. evaluate()'s reject_threshold_sweep
# reports the one thing that CAN be measured honestly right now without
# that data: how often a given threshold would falsely reject a query
# DINOv3 actually got right, using only the existing known-product split.
# Pick a threshold from that false-reject-rate table (accept some rate you
# consider tolerable) and pass it via --reject-threshold once you do.
REJECT_SIMILARITY_THRESHOLD = None
# Bare category name, not a "a photo of a {category}" template -- matches
# what finetune_siglip2_v3.py's build_training_labels actually trained the
# text tower on for taxonomy nodes (raw strings like "sneaker", "hoodie"
# under the "generic" label kind, no caption-style prefix).
CATEGORY_PROMPT_TEMPLATE = "{category}"

# Real HSC (Hierarchical Softmax Classification) climbing, per
# notebooks/fashionsiglip2_hsc_finetune.ipynb's own "Algorithm 1 from the
# HSC paper" cell -- this project already implemented genuine HSC once
# before, this pipeline just wasn't using it. Score every LEAF of the full
# docs/hierarchy.json tree (group -> category -> fine leaf, ~42 leaves,
# not just the 13 categories), softmax into a real probability
# distribution, sum probability mass up through every ancestor, then climb
# from the most probable leaf toward the root until an ancestor's summed
# probability clears HSC_THRESHOLD. This is the specificity-backoff
# behavior a flat "always commit to the top-1 category regardless of
# margin" classification can't do -- e.g. a genuinely low-margin leaf call
# (~0.05, seen in real Phase 4 output) should back off to a broader,
# more-confident ancestor instead of confidently asserting a specific
# leaf/category it isn't actually confident about.
HSC_ROOT_LABEL = "apparel item"
HSC_TEMPERATURE = 0.05  # matches the notebook's own softmax temperature for the finetuned-SigLIP2 config
HSC_THRESHOLD = 0.5     # default confidence threshold to climb to; higher = broader/safer predictions

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

print("Building catalog (verifying every image -- slow on Drive FUSE, this is expected to take a while, not hung)...")
for _product in tqdm(_metadata, desc="Verifying catalog images"):
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


# ============================================================
# HSC tree construction (docs/hierarchy.json -> root/group/category/leaf
# parent-child maps), per notebooks/fashionsiglip2_hsc_finetune.ipynb's
# own HSC cell. A category with no fine leaves listed (e.g. "sweatshirt",
# "socks") is itself a leaf -- there's nothing more specific below it.
# ============================================================

def _build_hsc_tree(hierarchy_dict, root_label):
    parent = {root_label: None}
    children = defaultdict(list)
    category_of_leaf = {}

    for group, categories in hierarchy_dict.items():
        parent[group] = root_label
        children[root_label].append(group)
        for category, leaves in categories.items():
            parent[category] = group
            children[group].append(category)
            if leaves:
                for leaf in leaves:
                    parent[leaf] = category
                    children[category].append(leaf)
                    category_of_leaf[leaf] = category
            else:
                category_of_leaf[category] = category

    children = dict(children)
    leaf_ids = [node for node in parent if not children.get(node)]
    return parent, children, leaf_ids, category_of_leaf


HSC_PARENT, HSC_CHILDREN, HSC_LEAF_IDS, HSC_CATEGORY_OF_LEAF = _build_hsc_tree(_hierarchy, HSC_ROOT_LABEL)
HSC_GROUP_NAMES = set(_hierarchy.keys())
print(f"HSC tree: {len(HSC_LEAF_IDS)} leaves under {len(CANONICAL_CATEGORIES)} categories under {len(HSC_GROUP_NAMES)} groups")

_HSC_DESCENDANT_LEAVES_CACHE = {}


def hsc_descendant_leaves(node):
    if node in _HSC_DESCENDANT_LEAVES_CACHE:
        return _HSC_DESCENDANT_LEAVES_CACHE[node]
    if not HSC_CHILDREN.get(node):
        result = (node,)
    else:
        result = tuple(leaf for child in HSC_CHILDREN[node] for leaf in hsc_descendant_leaves(child))
    _HSC_DESCENDANT_LEAVES_CACHE[node] = result
    return result


def hsc_categories_under(node):
    """Every CANONICAL_CATEGORIES-level category reachable under this node
    (itself, if it already is one) -- what the category-gate filter
    actually restricts to when HSC climbs to this node."""
    return {HSC_CATEGORY_OF_LEAF[leaf] for leaf in hsc_descendant_leaves(node)}


def hsc_node_level(node):
    if node == HSC_ROOT_LABEL:
        return "root"
    if node in HSC_GROUP_NAMES:
        return "group"
    if node in CANONICAL_CATEGORIES:
        return "category"
    return "leaf"


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
    expected = {"checkpoint": siglip2_checkpoint, "num_identities": len(identities)}

    if _index_is_fresh(config_path, expected):
        payload = torch.load(INDEX_DIR / "semantic_identity_embeddings.pt", map_location="cpu", weights_only=False)
        print("Semantic index: cache hit.")
        return payload["identities"], payload["embeddings"]

    print("Semantic index: (re)building...")
    identity_embeddings = embed_texts_siglip(model, processor, identities)
    torch.save({"identities": identities, "embeddings": identity_embeddings}, INDEX_DIR / "semantic_identity_embeddings.pt")

    config_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return identities, identity_embeddings


def build_or_load_hsc_leaf_index(model, processor, siglip2_checkpoint):
    """Text embeddings for every leaf of the full HSC tree (~42 leaves,
    e.g. "golf sneaker", "loafer", "sweatshirt"), not just the 13
    categories -- this is what hsc_climb scores against. Bare leaf names
    as prompts, same CATEGORY_PROMPT_TEMPLATE convention/rationale as the
    old flat category classifier (matches what build_training_labels
    actually trained the text tower on for taxonomy nodes)."""
    config_path = INDEX_DIR / "hsc_leaf_index_config.json"
    expected = {"checkpoint": siglip2_checkpoint, "num_leaves": len(HSC_LEAF_IDS)}

    if _index_is_fresh(config_path, expected):
        payload = torch.load(INDEX_DIR / "hsc_leaf_embeddings.pt", map_location="cpu", weights_only=False)
        print("HSC leaf index: cache hit.")
        return payload["leaf_ids"], payload["embeddings"]

    print("HSC leaf index: (re)building...")
    leaf_prompts = [CATEGORY_PROMPT_TEMPLATE.format(category=leaf) for leaf in HSC_LEAF_IDS]
    leaf_embeddings = embed_texts_siglip(model, processor, leaf_prompts)
    torch.save({"leaf_ids": HSC_LEAF_IDS, "embeddings": leaf_embeddings}, INDEX_DIR / "hsc_leaf_embeddings.pt")

    config_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return HSC_LEAF_IDS, leaf_embeddings


def hsc_climb(leaf_probabilities, leaf_ids, threshold):
    """Algorithm 1 from the HSC paper (this project's own earlier
    implementation, notebooks/fashionsiglip2_hsc_finetune.ipynb): start at
    the most probable leaf, climb toward the root while the current node's
    aggregated probability (sum of all its descendant leaves) is below
    threshold, stop at the first node that clears it."""
    leaf_to_index = {leaf: i for i, leaf in enumerate(leaf_ids)}
    node_probabilities = {}
    for node in HSC_PARENT:
        indices = [leaf_to_index[leaf] for leaf in hsc_descendant_leaves(node)]
        node_probabilities[node] = float(leaf_probabilities[indices].sum())

    best_leaf_index = int(torch.argmax(leaf_probabilities))
    best_leaf = leaf_ids[best_leaf_index]
    current_node = best_leaf
    climbing_path = [current_node]

    while node_probabilities[current_node] < threshold and HSC_PARENT[current_node] is not None:
        current_node = HSC_PARENT[current_node]
        climbing_path.append(current_node)

    return {
        "predicted_node": current_node,
        "predicted_level": hsc_node_level(current_node),
        "confidence": node_probabilities[current_node],
        "best_leaf": best_leaf,
        "best_leaf_probability": float(leaf_probabilities[best_leaf_index]),
        "climbing_path": climbing_path,
        "allowed_categories": hsc_categories_under(current_node),
    }


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

        self.identities, self.identity_embeddings = build_or_load_semantic_index(
            self.siglip2_model, self.siglip2_processor, siglip2_checkpoint,
        )
        self.identity_category = {}
        for identity in self.identities:
            codes = IDENTITY_TO_PRODUCT_CODES[identity]
            if codes:
                self.identity_category[identity] = CATALOG[next(iter(codes))]["category"]

        self.hsc_leaf_ids, self.hsc_leaf_embeddings = build_or_load_hsc_leaf_index(
            self.siglip2_model, self.siglip2_processor, siglip2_checkpoint,
        )

        self.gallery_product_codes, self.gallery_embeddings = build_or_load_identity_index(
            self.dino_backbone, self.dino_head, self.dino_use_projection, self.dino_processor,
            dino_checkpoint, self.gallery_images_by_product,
        )
        self.gallery_index_by_code = {code: i for i, code in enumerate(self.gallery_product_codes)}

    def hsc_predict(self, siglip_image_embedding, threshold=HSC_THRESHOLD):
        similarity = (siglip_image_embedding @ self.hsc_leaf_embeddings.T).squeeze(0)
        leaf_probabilities = F.softmax(similarity / HSC_TEMPERATURE, dim=0)
        return hsc_climb(leaf_probabilities, self.hsc_leaf_ids, threshold)

    def shortlist_identities(self, siglip_image_embedding, allowed_categories, top_k):
        if allowed_categories is not None:
            candidate_indices = [i for i, identity in enumerate(self.identities) if self.identity_category.get(identity) in allowed_categories]
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

    def retrieve(self, image_path, use_category_gate=False, hsc_threshold=HSC_THRESHOLD, top_identity_candidates=TOP_IDENTITY_CANDIDATES, final_top_k=FINAL_TOP_K, reject_threshold=REJECT_SIMILARITY_THRESHOLD):
        # Default flipped to off: the old flat (non-HSC) gate was confirmed
        # net-negative in two independent Phase 4 runs (docs/eval_log.md,
        # 2026-07-30) -- it excluded the true category ~30% of the time,
        # unrecoverable, for no rerank-quality benefit. Real HSC climbing
        # (2026-08-01) backs off to a broader, more-confident ancestor
        # instead of blindly committing to a low-margin leaf, which should
        # reduce that exclusion rate -- re-benchmark both arms once this
        # has real eval numbers rather than assuming HSC fixes it.
        image = load_rgb_image(image_path)

        siglip_embedding = embed_images_siglip(self.siglip2_model, self.siglip2_processor, [image])
        dino_embedding = embed_images_dino(self.dino_backbone, self.dino_head, self.dino_use_projection, self.dino_processor, [image])

        hsc_result = self.hsc_predict(siglip_embedding, threshold=hsc_threshold)
        allowed_categories = hsc_result["allowed_categories"] if use_category_gate else None

        candidates = self.shortlist_identities(siglip_embedding, allowed_categories, top_identity_candidates)
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

        # Open-set rejection (spec section 7): if the best candidate's own
        # score doesn't clear reject_threshold, don't force an exact-product
        # claim -- fall back to the broadest label the pipeline still has
        # real evidence for, which is exactly what HSC climbing already
        # computed upstream (hsc_predicted_node/allowed_categories), not a
        # new signal. Doesn't remove `results` -- a rejected top-1 is still
        # shown as a "closest visual match" candidate, just not asserted as
        # an identified product.
        rejected = reject_threshold is not None and (not results or results[0]["dino_identity_score"] < reject_threshold)

        return {
            "hsc_predicted_node": hsc_result["predicted_node"], "hsc_predicted_level": hsc_result["predicted_level"],
            "hsc_confidence": hsc_result["confidence"], "hsc_climbing_path": hsc_result["climbing_path"],
            "hsc_best_leaf": hsc_result["best_leaf"], "hsc_best_leaf_probability": hsc_result["best_leaf_probability"],
            "allowed_categories": sorted(hsc_result["allowed_categories"]),
            "num_identity_candidates": len(candidates), "results": results,
            "same_model_different_colorway_ambiguous": ambiguous,
            "rejected_open_set": rejected, "reject_threshold": reject_threshold,
        }

    # --------------------------------------------------------
    # End-to-end held-out evaluation
    # --------------------------------------------------------

    def evaluate(self, use_category_gate=True, hsc_threshold=HSC_THRESHOLD, top_identity_candidates=TOP_IDENTITY_CANDIDATES):
        queries = [(code, path) for code, paths in self.test_image_by_product.items() for path in paths]
        print(f"Evaluating {len(queries):,} held-out queries (category gate: {use_category_gate}, HSC threshold: {hsc_threshold})...")

        ranks = []
        gate_exclusions = 0
        identity_shortlist_misses = 0
        climb_level_counts = {"leaf": 0, "category": 0, "group": 0, "root": 0}
        # Top-1 DINOv3 score on every query where top-1 was actually
        # CORRECT -- used below to compute, for a range of candidate reject
        # thresholds, what fraction of real correct answers a threshold
        # would falsely throw away. This is a known-product-only query set
        # (every true_code is in the catalog by construction), so it can
        # only measure false rejections, never a true open-set
        # accept/reject rate -- that needs queries with no true catalog
        # match at all, which this eval doesn't have (see
        # REJECT_SIMILARITY_THRESHOLD's module comment).
        correct_top1_scores = []

        for true_code, image_path in queries:
            image = load_rgb_image(image_path)
            siglip_embedding = embed_images_siglip(self.siglip2_model, self.siglip2_processor, [image])
            dino_embedding = embed_images_dino(self.dino_backbone, self.dino_head, self.dino_use_projection, self.dino_processor, [image])

            hsc_result = self.hsc_predict(siglip_embedding, threshold=hsc_threshold)
            climb_level_counts[hsc_result["predicted_level"]] += 1
            true_category = CATALOG[true_code]["category"]
            allowed_categories = hsc_result["allowed_categories"] if use_category_gate else None
            if use_category_gate and true_category not in allowed_categories:
                gate_exclusions += 1

            candidates = self.shortlist_identities(siglip_embedding, allowed_categories, top_identity_candidates)
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
            if rank == 1:
                correct_top1_scores.append(float(similarity[order[0]]))

        ranks = np.asarray(ranks, dtype=float)
        correct_scores = np.asarray(correct_top1_scores, dtype=float)
        # Sweep candidate thresholds and report what fraction of REAL
        # correct top-1 answers each would falsely reject -- the only
        # thing measurable without a genuinely-unknown-product eval split
        # (see REJECT_SIMILARITY_THRESHOLD's module comment). Percentile-
        # based so the sweep is meaningful regardless of this checkpoint's
        # absolute score scale.
        reject_threshold_sweep = {}
        if len(correct_scores) > 0:
            for percentile in (1, 2, 5, 10, 20):
                threshold = float(np.percentile(correct_scores, percentile))
                false_reject_rate = float(np.mean(correct_scores < threshold))
                reject_threshold_sweep[f"p{percentile}"] = {"threshold": threshold, "false_reject_rate": false_reject_rate}

        metrics = {
            "num_queries": len(queries),
            "category_gate_exclusion_rate": gate_exclusions / len(queries) if use_category_gate else None,
            "identity_shortlist_miss_rate": identity_shortlist_misses / len(queries),
            "hsc_climb_level_fractions": {level: count / len(queries) for level, count in climb_level_counts.items()},
            "recall_at_1": float(np.mean(ranks <= 1)),
            "recall_at_5": float(np.mean(ranks <= 5)),
            "recall_at_10": float(np.mean(ranks <= 10)),
            "mrr": float(np.mean(1.0 / ranks)),
            "median_rank": float(np.median(ranks)),
            "mean_rank": float(np.mean(ranks)),
            "reject_threshold_sweep": reject_threshold_sweep,
        }
        return metrics


def print_result(query_path, result):
    print(f"\nQuery: {query_path}")
    path_str = " -> ".join(result["hsc_climbing_path"])
    print(f"HSC prediction: {result['hsc_predicted_node']} ({result['hsc_predicted_level']}, "
          f"confidence={result['hsc_confidence']:.3f})")
    print(f"  Climbing path (best leaf -> root): {path_str}")
    print(f"  Best single leaf guess: {result['hsc_best_leaf']} (p={result['hsc_best_leaf_probability']:.3f})")
    print(f"  Categories in play: {result['allowed_categories']}")
    print(f"Identity candidates considered: {result['num_identity_candidates']}")
    if result["rejected_open_set"]:
        print(f"  ! Best match scores below reject_threshold={result['reject_threshold']:.4f} -- "
              f"NOT asserting an exact product. Falling back to: {result['hsc_predicted_node']} "
              f"({result['hsc_predicted_level']}-level, confidence={result['hsc_confidence']:.3f}). "
              f"Closest visual candidates shown below are NOT a confirmed identification.")
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
    levels = metrics["hsc_climb_level_fractions"]
    print(f"HSC climb landed at: leaf {levels['leaf']*100:.1f}%, category {levels['category']*100:.1f}%, "
          f"group {levels['group']*100:.1f}%, root (no confidence anywhere) {levels['root']*100:.1f}%")
    print(f"R@1:  {metrics['recall_at_1'] * 100:.2f}%")
    print(f"R@5:  {metrics['recall_at_5'] * 100:.2f}%")
    print(f"R@10: {metrics['recall_at_10'] * 100:.2f}%")
    print(f"MRR:  {metrics['mrr'] * 100:.2f}%")
    print(f"Median rank: {metrics['median_rank']:.1f}")
    print(f"Mean rank: {metrics['mean_rank']:.2f}")
    if metrics["reject_threshold_sweep"]:
        print("Open-set reject-threshold sweep (false-reject rate against REAL correct top-1 answers only --")
        print("  not a true accept/reject rate, this eval set has no genuinely-unknown-product queries yet):")
        for label, row in metrics["reject_threshold_sweep"].items():
            print(f"    {label}: threshold={row['threshold']:.4f}  false_reject_rate={row['false_reject_rate']*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to a single query image.")
    parser.add_argument("--evaluate", action="store_true", help="Run end-to-end held-out evaluation.")
    parser.add_argument("--category-gate", action="store_true",
                         help="Enable HSC-based stage-1 category gating (off by default -- the old flat/non-HSC gate was confirmed net-negative, see docs/eval_log.md 2026-07-30; real HSC climbing as of 2026-08-01 hasn't been re-benchmarked yet).")
    parser.add_argument("--hsc-threshold", type=float, default=HSC_THRESHOLD,
                         help=f"HSC climbing confidence threshold, 0-1 (default {HSC_THRESHOLD}). Higher = broader/safer predictions.")
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K, help="Number of results to print for --image mode.")
    parser.add_argument("--reject-threshold", type=float, default=REJECT_SIMILARITY_THRESHOLD,
                         help="Open-set rejection: DINOv3 cosine-similarity floor below which --image mode won't "
                              "assert an exact product (falls back to the HSC category-level label instead). "
                              "Unset by default -- not calibrated yet. Run --evaluate first to see the "
                              "reject_threshold_sweep's false-reject-rate table, pick a threshold from there.")
    args = parser.parse_args()

    retriever = HierarchicalRetriever()

    if args.image:
        result = retriever.retrieve(args.image, use_category_gate=args.category_gate, hsc_threshold=args.hsc_threshold, final_top_k=args.top_k, reject_threshold=args.reject_threshold)
        print_result(args.image, result)

    if args.evaluate:
        gated_metrics = retriever.evaluate(use_category_gate=True, hsc_threshold=args.hsc_threshold)
        print_metrics("End-to-end held-out eval -- WITH HSC-based category gate", gated_metrics)

        ungated_metrics = retriever.evaluate(use_category_gate=False, hsc_threshold=args.hsc_threshold)
        print_metrics("End-to-end held-out eval -- WITHOUT category gate (fallback comparison)", ungated_metrics)

        with (INDEX_DIR / "pipeline_eval_metrics.json").open("w", encoding="utf-8") as f:
            json.dump({"with_category_gate": gated_metrics, "without_category_gate": ungated_metrics}, f, indent=2)

    if not args.image and not args.evaluate:
        print("Nothing to do -- pass --image PATH and/or --evaluate.")
