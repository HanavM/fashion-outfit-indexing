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
from concurrent.futures import ThreadPoolExecutor
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

# Loads HF_TOKEN (and anything else in .env, gitignored) from the repo root
# so os.environ.get("HF_TOKEN") below actually finds something locally --
# previously only caption_apparel.py/caption_shoes.py called this, so
# HF_TOKEN in .env was invisible to this script and everything that
# imports from it (composed_query_search.py, unseen_product_enrollment_
# eval.py) unless exported manually in the shell first. No-op on Colab/
# Modal, where HF_TOKEN is normally set directly in the environment.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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

# Overridable so an EXPERIMENT cannot clobber the index a live service is
# serving from. This is not hypothetical: `open_set_holdout_fraction` is
# part of the identity index's invalidating fingerprint, so running
# `--evaluate --open-set-holdout-fraction 0.1` against the same dataset
# root rebuilds retrieval_indexes/ IN PLACE with whole identities removed
# from the gallery. modal_app_serve.py loads that directory at container
# start, so the next cold start would come up serving a silently smaller
# catalog -- no error, just quietly worse answers. Point experiments at
# their own directory instead:
#     RETRIEVAL_INDEX_DIR=/data/apparel_dataset/retrieval_indexes_openset
INDEX_DIR = Path(os.environ.get("RETRIEVAL_INDEX_DIR", str(DATASET_ROOT / "retrieval_indexes")))
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

# Open-set rejection EVAL SPLIT (spec section 8.1's named "open-set
# rejection" split, never built until 2026-08-03). Fraction of catalog
# IDENTITIES held out of the gallery entirely -- not just their test
# images held out (that's SPLIT_SEED's job, and those products are still
# retrievable), but every image of every colorway sibling removed from
# the DINOv3 gallery index AND from the SigLIP2 identity shortlist, so a
# query from one of them has NO correct answer available at any rank.
#
# Held out at IDENTITY granularity, not product-code granularity, on
# purpose: dropping a single product code would leave its colorway
# siblings ("same shoe, different colorway") in the gallery, so the
# "correct" answer would still be sitting there in near-identical form
# and the query wouldn't be genuinely off-catalog. Identity-level holdout
# is also what makes this directly comparable to
# unseen_product_enrollment_eval.py's framing.
#
# 0.0 = disabled (default). This deliberately does NOT default on: an
# open-set run shrinks the gallery, so its R@K numbers are NOT comparable
# to the main benchmark rows in docs/eval_log.md and must never be logged
# as if they were. See evaluate()'s open_set block.
OPEN_SET_HOLDOUT_FRACTION = 0.0
# Separate from SPLIT_SEED so that changing the image-level view split
# doesn't silently reshuffle which identities are considered off-catalog
# (and vice versa) -- these two splits answer different questions.
OPEN_SET_SPLIT_SEED = 20260803

# Score fusion (spec sections 2.1/6: candidates should be reranked with
# more than one signal, not DINOv3-rerank-only). Off by default, same
# caution pattern as the category gate -- an unvalidated new arm to
# benchmark against the real 47.65% ungated baseline (docs/eval_log.md,
# 2026-07-31), not a silent default change. IMPORTANT limitation: the
# SigLIP2 identity score is identical across every colorway sibling of
# the same identity string (by design -- see finetune_siglip2_v3.py's own
# note that SigLIP2 is deliberately not trained to distinguish
# colorways), so fusion can only move the ranking of candidates that
# belong to DIFFERENT identities in the shortlist. It contributes zero
# discriminative signal between same-identity colorway siblings -- DINOv3
# alone still has to make that call, fusion doesn't change that.
USE_SCORE_FUSION = False
DINO_FUSION_WEIGHT = 0.8
SIGLIP_FUSION_WEIGHT = 0.2

# Patch-level DINOv3 reranking (spec section 6). Only re-scores the
# already-narrowed top PATCH_RERANK_CANDIDATES from the pooled DINOv3
# rerank -- per-patch similarity is O(N_patches^2) per pair, real money
# more expensive than one pooled-vector comparison, so it only pays for
# itself as a final polish step over a small window, same shape as every
# other shortlist-then-expensive-step stage in this pipeline.
#
# **CONFIRMED SEVERELY HARMFUL, real checkpoint, 2026-08-03 (see
# docs/eval_log.md) -- do not enable.** Not "unvalidated" anymore: a
# real Colab run at K=50 showed R@1 50.76%->20.59% (-30.17pt) with this
# flag on, while R@10 stayed EXACTLY identical (85.46% both times) --
# patch_rerank() only re-sorts the top PATCH_RERANK_CANDIDATES pooled
# items without changing set membership, so the true product stays in
# the top-10 either way, but gets scrambled from rank 1 down to rank
# 5-10 within that window. Root cause: DINOv3's projection_head was
# only ever trained on pooled features, never individual patch tokens
# (see embed_image_dino_patches's docstring) -- applying it to raw
# patch tokens produces near-noise per-patch embeddings that actively
# destroy an already-good pooled ranking when used to reorder it.
USE_PATCH_RERANK = False
PATCH_RERANK_CANDIDATES = 10

# Brand evidence (spec section 4.5's "Brand evidence" path, built
# 2026-08-04 in brand_evidence.py -- OCR the query image, fuzzy-match the
# text against the catalog's own brand vocabulary). Off by default.
#
# **This is a BOOST, never a filter, and that is deliberate.** A brand
# filter is the same shape as the category gate, which has been measured
# net-negative six independent times (docs/eval_log.md): excluding
# candidates is unrecoverable, so a single wrong brand read costs a query
# outright. A bonus added to same-brand candidates can only reorder --
# every candidate the shortlist supplied is still reachable at some rank,
# so the worst case is a ranking regression, the same failure shape as
# score fusion rather than the gate's.
#
# The bonus is added to the DINOv3 cosine similarity, whose top-of-list
# margins are small (AMBIGUITY_MARGIN is 0.03 for exactly that reason), so
# the default weight is of that order: large enough to break near-ties in
# favour of the OCR-confirmed brand, too small to overturn a confident
# DINOv3 decision.
USE_BRAND_BOOST = False
BRAND_BOOST_WEIGHT = float(os.environ.get("BRAND_BOOST_WEIGHT", "0.03"))
# Minimum brand_evidence score to act on at all. Anything below is treated
# as no evidence. Set from the measured precision/recall curve in
# docs/eval_log.md, not guessed.
BRAND_EVIDENCE_MIN_SCORE = float(os.environ.get("BRAND_EVIDENCE_MIN_SCORE", "0.30"))
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

# How many gallery views are averaged into each product's DINOv3 prototype.
#
# Was a hardcoded 2 with no rationale recorded. That leaves most of the
# gallery unused: the median catalog product has 5 images and the mean is
# 5.6, so a cap of 2 embeds 4,729 of 13,419 available images -- 65% of the
# imagery is discarded, and 2,139 products are truncated.
#
# Worth testing rather than just raising, because more is not obviously
# better: product galleries mix front views with detail crops, flat lays
# and back shots, so averaging more views could sharpen the prototype
# (more angles, less per-shot noise) or blur it (unrelated close-ups
# pulling the mean off the garment). This is now the cheapest open lever
# on rerank quality -- conditional R@1 sits at 54.6% (docs/eval_log.md,
# 2026-08-03) and it needs no retraining, just a re-encode.
#
# Changing this correctly invalidates cached embeddings per product via
# `gallery_signature`, which records the exact image paths behind each
# vector -- so a product with only 2 images is reused untouched while a
# product that gains views gets re-embedded. That is why this is NOT in
# the index fingerprint's `core`: it does not change what an embedding
# means, only which images back it.
# RESOLVED 2026-08-04 -- default raised 2 -> 6 on measured results.
# Ungated R@1 at K=150: cap 2 = 53.95%, 4 = 58.32%, 6 = 59.50%, all = 59.92%.
# Shortlist miss was IDENTICAL (1.18%) at every setting, which proves the
# gain is entirely rerank quality: this is a DINOv3-side change and cannot
# touch SigLIP2's shortlist. Conditional R@1 went 54.6% -> 60.6%, moving
# the exact metric that stayed flat through the whole K sweep.
# 6 captures 93% of the available gain; "all" adds 0.42pt for a fuller
# re-encode. The concern that detail crops and flat lays would blur the
# prototype did not materialize at any setting.
GALLERY_IMAGES_PER_PRODUCT = int(os.environ.get("GALLERY_IMAGES_PER_PRODUCT", "6"))
SPLIT_SEED = 42  # matches dino_identity_finetune.py's split exactly, so eval here is apples-to-apples

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
USE_AMP = DEVICE == "cuda"  # torch.autocast(device_type="mps") is still flaky across ops; MPS just runs fp32


def autocast_context():
    return torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else nullcontext()


# ---- GPU throughput knobs (tuned for a Colab T4) --------------------
#
# The T4 is Turing (sm_75): it HAS fp16 tensor cores but no bf16 and no
# TF32, so float16 autocast above is the right and only fast path -- do
# not "upgrade" it to bfloat16, which Turing emulates in software.
#
# Why these exist at all: profiling on 2026-08-03 found the GPU nearly
# idle during evaluation. Two causes, both fixed below rather than here:
#   1. Every index (semantic, HSC leaf, DINOv3 gallery) was loaded with
#      map_location="cpu", and every embed_* helper returned .cpu(), so
#      the entire per-query scoring path -- softmax over leaves, identity
#      shortlist matmul, topk, gallery rerank matmul, argsort -- ran on
#      the CPU. On Colab's 2-vCPU runtime that is the whole bottleneck.
#      Indexes are now moved to DEVICE once, in HierarchicalRetriever.
#   2. Image decode + processor resize is single-threaded CPU work that
#      the GPU sits waiting on. Now overlapped via load_images_parallel.
if DEVICE == "cuda":
    # Fixed input resolution across every batch, so autotuned kernels are
    # picked once and reused rather than re-searched.
    torch.backends.cudnn.benchmark = True

# A T4's 16GB fits far more than the old hardcoded 32 at 224px for these
# ViT-B-class encoders; batch size is the single biggest lever on encode
# throughput. Env-overridable so a smaller GPU can dial it back without
# editing code.
ENCODE_BATCH_SIZE = int(os.environ.get("ENCODE_BATCH_SIZE", "64" if DEVICE == "cuda" else "16"))
# Decode/EXIF-transpose threads feeding the GPU. PIL releases the GIL in
# its C decoders, so threads (not processes) are enough and cost no IPC.
#
# Held at 8 on CUDA, NOT raised further, because the two deployment
# targets disagree about what is safe.
#
# On a Modal Volume the encode loop measured ~3.2 images/s (2026-08-03) --
# far below a T4's ViT-B fp16 rate -- because each file is a network
# round trip the GPU waits on. That is pure latency, so more threads help,
# and modal_app_phase4_eval.py sets IMAGE_LOADER_WORKERS=32 for exactly
# that reason.
#
# On Colab the same mount is Google Drive FUSE, which is far less tolerant
# of concurrent access and is the prime suspect in a run that died with an
# unprompted "^C" right after this loader went threaded. Drive is the more
# fragile of the two, so it sets the default; raise it per-environment
# where the filesystem can take it. IMAGE_LOADER_WORKERS=1 restores fully
# serial reads, which is the thing to try first if Drive misbehaves.
IMAGE_LOADER_WORKERS = int(os.environ.get("IMAGE_LOADER_WORKERS", "8" if DEVICE == "cuda" else "4"))

# How many images are decoded -- and therefore how many files are read
# CONCURRENTLY -- at once. Deliberately NOT tied to ENCODE_BATCH_SIZE.
#
# Context: a Colab run died mid-gallery-encode on 2026-08-03 with a "^C"
# the user did not type. The first theory was RAM (encode batch had just
# gone 32 -> 64, decoded whole-batch), but that was measured and does not
# hold: this catalog's photos are ~1600x798, so a decoded RGB frame is
# ~3.8MB and 64 of them is ~0.25GB, nowhere near a Colab OOM.
#
# The likelier culprit is the other half of that change: images live on a
# Google Drive FUSE mount, and the threaded loader turned a serial read
# into N concurrent ones against a network-API-backed filesystem that is
# well known to hang or fault under exactly that. So the number that
# needs bounding is concurrency, not bytes.
#
# Chunking also keeps peak memory flat regardless of encode batch, which
# is worth having on its own: decode a chunk, convert it straight to
# preprocessed pixel tensors (224px DINOv3 / 384px SigLIP2, ~0.6MB and
# ~1.8MB each), release the full-resolution frames, and only then run the
# model on a full-size batch. GPU keeps the big batch; RAM and the
# filesystem never see it.
IMAGE_DECODE_CHUNK = int(os.environ.get("IMAGE_DECODE_CHUNK", "8"))


def load_images_parallel(paths, workers=None):
    """Decode images concurrently, preserving input order.

    Returns (images, kept_positions) where kept_positions holds the INDEX
    into `paths` of each surviving image. Positions rather than paths on
    purpose: callers align these against a parallel list (owners, query
    records), and a path-keyed mapping would silently collapse if the same
    file ever backed two entries. Unreadable files are skipped, not fatal
    -- the same policy the identity index build has always used.
    """
    workers = workers or IMAGE_LOADER_WORKERS
    if not paths:
        return [], []

    def safe_load(path):
        try:
            return load_rgb_image(path), None
        except Exception as error:  # unreadable/corrupt file
            return None, error

    with ThreadPoolExecutor(max_workers=workers) as pool:
        loaded = list(pool.map(safe_load, paths))

    images, kept_positions = [], []
    for position, (path, (image, error)) in enumerate(zip(paths, loaded)):
        if image is None:
            print(f"Skipped (unreadable): {path} ({error})")
            continue
        images.append(image)
        kept_positions.append(position)
    return images, kept_positions


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


def _brand_key(brand):
    """Comparable form of a brand string.

    CATALOG stores the display form ("New Balance", "PacSun") while
    brand_evidence.py works in metadata.json's raw keys ("newbalance",
    "pacsun"), so both sides get squashed to lowercase alphanumerics
    before comparison. This is the same case/spacing mismatch that
    silently produced zero products in unseen_product_enrollment_eval.py
    (docs/eval_log.md, 2026-08-02) -- worth doing once, in one place.
    """
    return re.sub(r"[^a-z0-9]", "", (brand or "").lower())


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

# Catalog verification is the slowest part of startup on Colab, and it
# runs on EVERY invocation before any model loads. Per image it costs up
# to five path probes (resolve_image_path's candidate list) plus a full
# file read (Image.verify() reads the whole file), and Google Drive FUSE
# charges round-trip latency for each one -- ~14k images, serially.
#
# Two fixes, both here rather than by weakening the check itself: results
# are cached to disk keyed on metadata.json's own fingerprint, and the
# cold path is threaded. The work is pure I/O latency, so threads help
# far past core count and PIL releases the GIL while reading.
CATALOG_CACHE_PATH = INDEX_DIR / "catalog_image_cache.json"
CATALOG_VERIFY_WORKERS = int(os.environ.get("CATALOG_VERIFY_WORKERS", "32"))
# Escape hatch: skip verification entirely and trust whatever resolves.
# Faster still, but a corrupt JPEG then surfaces as a crash deep inside
# the identity-index build instead of being filtered out here.
SKIP_CATALOG_VERIFY = os.environ.get("SKIP_CATALOG_VERIFY", "").lower() in {"1", "true", "yes"}


def _catalog_cache_fingerprint():
    """Cache is valid only for the exact metadata.json that produced it.

    Keyed on the metadata file rather than per-image stats: negative
    results (an image that doesn't resolve at all) can't be revalidated
    cheaply, and metadata.json changes whenever images are added or
    recropped anyway -- so tying the two together keeps negatives safe to
    cache without a per-image stat.
    """
    try:
        stat = METADATA_PATH.stat()
    except OSError:
        return None
    return {"metadata_size": stat.st_size, "metadata_mtime": int(stat.st_mtime),
            "dataset_root": str(DATASET_ROOT), "verified": not SKIP_CATALOG_VERIFY}


def _verify_catalog_image(raw_path):
    """-> (raw_path, resolved path str) or (raw_path, None) if unusable."""
    path = resolve_image_path(raw_path)
    if path is None:
        return raw_path, None
    if SKIP_CATALOG_VERIFY:
        return raw_path, str(path)
    # Corrupted/truncated-file skip, same convention as
    # finetune_siglip2_v3.py / dino_identity_finetune.py -- without this a
    # single bad JPEG crashes the whole identity-index build deep inside
    # HierarchicalRetriever.__init__ instead of being filtered out here at
    # catalog-build time.
    try:
        with Image.open(path) as _check_image:
            _check_image.verify()
    except Exception:
        return raw_path, None
    return raw_path, str(path)


_fingerprint = _catalog_cache_fingerprint()
_resolved_by_raw = None
if _fingerprint is not None and CATALOG_CACHE_PATH.is_file():
    try:
        _cached = json.loads(CATALOG_CACHE_PATH.read_text(encoding="utf-8"))
        if _cached.get("fingerprint") == _fingerprint:
            _resolved_by_raw = _cached["resolved"]
            print(f"Catalog image cache: hit ({len(_resolved_by_raw):,} paths, skipping verification).")
    except (json.JSONDecodeError, OSError, KeyError):
        _resolved_by_raw = None

if _resolved_by_raw is None:
    _all_raw_paths = sorted({raw for product in _metadata for raw in product.get("images", [])})
    print(f"Building catalog ({'resolving' if SKIP_CATALOG_VERIFY else 'verifying'} "
          f"{len(_all_raw_paths):,} images, {CATALOG_VERIFY_WORKERS} threads -- "
          f"cached for next run)...")
    _resolved_by_raw = {}
    with ThreadPoolExecutor(max_workers=CATALOG_VERIFY_WORKERS) as _pool:
        for _raw, _resolved in tqdm(_pool.map(_verify_catalog_image, _all_raw_paths),
                                    total=len(_all_raw_paths), desc="Verifying catalog images"):
            _resolved_by_raw[_raw] = _resolved
    if _fingerprint is not None:
        try:
            CATALOG_CACHE_PATH.write_text(
                json.dumps({"fingerprint": _fingerprint, "resolved": _resolved_by_raw}),
                encoding="utf-8")
        except OSError as _error:
            print(f"Catalog image cache: not saved ({_error}) -- startup will re-verify next run.")

for _product in _metadata:
    _product_code = normalize_text(_product.get("product_code", ""))
    if not _product_code:
        continue
    _identity, _category, _brand = product_identity_and_category(_product)
    _resolved_images = [resolved for _raw_path in _product.get("images", [])
                        if (resolved := _resolved_by_raw.get(_raw_path)) is not None]
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


def pick_open_set_identities(holdout_fraction):
    """Deterministically choose which identities are treated as entirely
    off-catalog (spec section 8.1's open-set rejection split). Returns an
    empty set when disabled, so every caller's non-open-set behavior is
    bit-identical to before this existed."""
    if not holdout_fraction:
        return set()
    identities = sorted(IDENTITY_TO_PRODUCT_CODES)
    num_holdout = int(round(len(identities) * holdout_fraction))
    # At least 1 (a 0-identity "open-set" run would silently report
    # open-set metrics over an empty query set), at most len-1 (never
    # empty the gallery entirely).
    num_holdout = max(1, min(num_holdout, len(identities) - 1))
    rng = random.Random(OPEN_SET_SPLIT_SEED)
    return set(rng.sample(identities, num_holdout))


def make_view_split(open_set_identities=frozenset()):
    """Identical logic/seed to dino_identity_finetune.py's make_view_split,
    reapplied per product_code here (rather than per image record) since
    that's the granularity this pipeline's gallery/query split needs --
    reproduces the exact same held-out test image per product so
    evaluate_pipeline's numbers are directly comparable to that script's
    own isolated eval.

    open_set_identities (default empty = original behavior exactly): any
    product whose identity is in this set is removed from the gallery
    completely and ALL of its images become open-set queries instead of
    the usual 1-image-held-out-per-product treatment -- there is no
    correct gallery answer for them by construction, which is the whole
    point of that split."""
    open_set_codes = set()
    for identity in open_set_identities:
        open_set_codes.update(IDENTITY_TO_PRODUCT_CODES.get(identity, ()))

    rng = random.Random(SPLIT_SEED)
    gallery_images_by_product, test_image_by_product = {}, {}
    open_set_queries = []
    for code, images in IMAGES_BY_PRODUCT.items():
        images = images.copy()
        # Shuffle BEFORE the open-set branch so the RNG consumption order
        # is unchanged for the products that do stay in the gallery --
        # otherwise enabling open-set holdout would also silently perturb
        # which test image every OTHER product gets, making the known-query
        # arm non-comparable to a normal run for reasons that have nothing
        # to do with open-set.
        rng.shuffle(images)
        if code in open_set_codes:
            open_set_queries.extend((code, path) for path in images)
            continue
        if len(images) < 2:
            gallery_images_by_product[code] = images
            continue
        max_holdout = len(images) - 1
        num_test = min(TEST_IMAGES_PER_PRODUCT, max_holdout)
        test_image_by_product[code] = images[:num_test]
        gallery_images_by_product[code] = images[num_test:]
    return gallery_images_by_product, test_image_by_product, open_set_queries


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


def pixel_batches_from_paths(processor, paths, gpu_batch, decode_chunk=None):
    """Yield (pixel_values, kept_positions) preprocessed batches from paths.

    Bounds peak RAM to `decode_chunk` full-resolution images regardless of
    how large `gpu_batch` is -- see IMAGE_DECODE_CHUNK. `kept_positions`
    are indices into `paths`, so callers can keep parallel lists (owners,
    query records) aligned when an unreadable file is dropped.
    """
    decode_chunk = decode_chunk or IMAGE_DECODE_CHUNK
    # More loader threads than images in a chunk buys nothing and only
    # multiplies the transient decode copies.
    workers = min(IMAGE_LOADER_WORKERS, max(decode_chunk, 1))

    for batch_start in range(0, len(paths), gpu_batch):
        batch_paths = paths[batch_start:batch_start + gpu_batch]
        tensors, kept_positions = [], []
        for offset in range(0, len(batch_paths), decode_chunk):
            sub_paths = batch_paths[offset:offset + decode_chunk]
            images, sub_kept = load_images_parallel(sub_paths, workers=workers)
            if not images:
                continue
            processed = processor(images=images, return_tensors="pt")
            tensors.append(processed["pixel_values"])
            kept_positions.extend(batch_start + offset + k for k in sub_kept)
            # Drop the full-resolution frames before decoding the next
            # chunk; only the (much smaller) pixel tensors survive.
            del images, processed
        if tensors:
            yield torch.cat(tensors, dim=0), kept_positions


@torch.inference_mode()
def embed_paths_dino(backbone, projection_head, use_projection, processor, paths,
                     batch_size=None, keep_on_device=False, progress=None):
    """DINOv3 embeddings for images given BY PATH. -> (embeddings, kept_positions)"""
    batch_size = batch_size or ENCODE_BATCH_SIZE
    embeddings, kept_all = [], []
    batches = pixel_batches_from_paths(processor, paths, batch_size)
    if progress:
        batches = tqdm(batches, total=(len(paths) + batch_size - 1) // batch_size,
                       desc=progress, unit="batch")
    for pixel_values, kept in batches:
        pixel_values = pixel_values.to(DEVICE, non_blocking=True)
        with autocast_context():
            outputs = backbone(pixel_values=pixel_values)
            raw = dino_pooled_features(outputs).float()
            batch_embeddings = projection_head(raw) if (use_projection and projection_head is not None) else F.normalize(raw, dim=-1)
        embeddings.append(batch_embeddings if keep_on_device else batch_embeddings.cpu())
        kept_all.extend(kept)
    if not embeddings:
        empty = torch.empty(0, DINOV3_PROJECTION_DIM)
        return (empty.to(DEVICE) if keep_on_device else empty), []
    return torch.cat(embeddings, dim=0), kept_all


@torch.inference_mode()
def embed_paths_siglip(model, processor, paths, batch_size=None, keep_on_device=False,
                       progress=None):
    """SigLIP2 embeddings for images given BY PATH. -> (embeddings, kept_positions)"""
    batch_size = batch_size or ENCODE_BATCH_SIZE
    embeddings, kept_all = [], []
    batches = pixel_batches_from_paths(processor, paths, batch_size)
    if progress:
        batches = tqdm(batches, total=(len(paths) + batch_size - 1) // batch_size,
                       desc=progress, unit="batch")
    for pixel_values, kept in batches:
        pixel_values = pixel_values.to(DEVICE, non_blocking=True)
        with autocast_context():
            batch_embeddings = extract_siglip_embeddings(
                model.get_image_features(pixel_values=pixel_values)).float()
        normalized = F.normalize(batch_embeddings, dim=-1)
        embeddings.append(normalized if keep_on_device else normalized.cpu())
        kept_all.extend(kept)
    if not embeddings:
        empty = torch.empty(0, model.config.text_config.hidden_size)
        return (empty.to(DEVICE) if keep_on_device else empty), []
    return torch.cat(embeddings, dim=0), kept_all


def dino_pooled_features(outputs):
    pooler_output = getattr(outputs, "pooler_output", None)
    return pooler_output if pooler_output is not None else outputs.last_hidden_state[:, 0]


@torch.inference_mode()
def embed_images_dino(backbone, projection_head, use_projection, processor, pil_images,
                      batch_size=None, keep_on_device=False):
    """keep_on_device=True skips the per-batch .cpu() copy.

    That copy is a hard GPU->CPU sync every batch, and for the query path
    the embeddings are only ever used in GPU matmuls against a
    device-resident index -- so rounddtripping them through host memory
    both stalls the pipeline and forces the scoring that follows onto the
    CPU. Index BUILDING still wants CPU tensors (they get torch.save'd),
    hence the default stays False.
    """
    batch_size = batch_size or ENCODE_BATCH_SIZE
    embeddings = []
    for start in range(0, len(pil_images), batch_size):
        batch = pil_images[start:start + batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in inputs.items()}
        with autocast_context():
            outputs = backbone(**inputs)
            raw = dino_pooled_features(outputs).float()
            batch_embeddings = projection_head(raw) if (use_projection and projection_head is not None) else F.normalize(raw, dim=-1)
        embeddings.append(batch_embeddings if keep_on_device else batch_embeddings.cpu())
    if not embeddings:
        empty = torch.empty(0, DINOV3_PROJECTION_DIM)
        return empty.to(DEVICE) if keep_on_device else empty
    return torch.cat(embeddings, dim=0)


@torch.inference_mode()
def embed_image_dino_patches(backbone, projection_head, use_projection, processor, pil_image):
    """Per-patch identity-space features for ONE image -- for patch-level
    reranking (spec section 6's "local patch comparison," never
    implemented before this: DINOv3 was only ever read via its pooled
    vector everywhere else in this pipeline, same architectural ceiling
    already identified and fixed for SigLIP2's free-text search).

    Applies the trained projection_head to every patch token individually,
    not just the pooled/CLS feature it was actually trained on -- the same
    kind of generalization used for SigLIP2's MaskCLIP-style dense
    matching (free_text_visual_search.py), but a weaker assumption here:
    that script's projection (value-proj + out-proj + residual MLP) is
    architecturally IDENTICAL for the pooled probe and every patch inside
    the attention-pooling head, whereas this projection_head is a generic
    MLP trained only on pooled features, so patch-token inputs sit outside
    its training distribution.

    **This concern is now CONFIRMED, not hypothetical, 2026-08-03**: real
    Colab run against the real checkpoints, --patch-rerank at K=50, R@1
    50.76%->20.59% (-30.17pt) while R@10 stayed EXACTLY unchanged
    (85.46% both times) -- proof the out-of-distribution patch
    embeddings are closer to noise than signal, scrambling an already-
    correct top-1 down to rank 5-10 without even being bad enough to
    push it out of the top 10 entirely. See docs/eval_log.md's 2026-08-03
    row for the full numbers. Do not enable --patch-rerank.
    """
    inputs = processor(images=[pil_image], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with autocast_context():
        outputs = backbone(**inputs)
        patch_tokens = outputs.last_hidden_state[:, 1:].float().squeeze(0)  # drop CLS/register-0 token
        if use_projection and projection_head is not None:
            patches = projection_head(patch_tokens)
        else:
            patches = F.normalize(patch_tokens, dim=-1)
    return patches.cpu()


def patch_similarity_score(query_patches, candidate_patches):
    """ColBERT-style late-interaction score: for every query patch, take
    its best-matching candidate patch, then average across query patches.
    Asymmetric on purpose -- rewards the candidate containing a strong
    local match for each part of the query, rather than requiring every
    candidate patch to also matter (a candidate photo can show more of
    the product/background than the query crop does)."""
    similarity_matrix = query_patches @ candidate_patches.T
    return float(similarity_matrix.max(dim=1).values.mean())


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
def embed_images_siglip(model, processor, pil_images, batch_size=None, keep_on_device=False):
    """keep_on_device=True skips the per-batch .cpu() sync -- see
    embed_images_dino for why that matters on the query path."""
    batch_size = batch_size or ENCODE_BATCH_SIZE
    embeddings = []
    for start in range(0, len(pil_images), batch_size):
        batch = pil_images[start:start + batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in inputs.items()}
        with autocast_context():
            batch_embeddings = extract_siglip_embeddings(model.get_image_features(**inputs)).float()
        normalized = F.normalize(batch_embeddings, dim=-1)
        embeddings.append(normalized if keep_on_device else normalized.cpu())
    if not embeddings:
        empty = torch.empty(0, model.config.text_config.hidden_size)
        return empty.to(DEVICE) if keep_on_device else empty
    return torch.cat(embeddings, dim=0)


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


def build_or_load_identity_index(backbone, projection_head, use_projection, processor, dino_checkpoint, gallery_images_by_product, open_set_holdout_fraction=0.0):
    config_path = INDEX_DIR / "identity_index_config.json"
    product_codes = sorted(gallery_images_by_product)
    # SPLIT_SEED/TEST_IMAGES_PER_PRODUCT/VAL_IMAGES_PER_PRODUCT included
    # even though they're module constants, not CLI flags today -- found
    # via code review, 2026-08-02: gallery_images_by_product's actual
    # CONTENTS (which images end up in the gallery vs. held out) depend on
    # all three, but the old fingerprint only tracked product COUNT, which
    # stays ~constant if any of these were ever edited to try a different
    # split -- the stale cache would silently look "fresh" and every
    # downstream eval/retrieve call would score against gallery embeddings
    # built from a different held-out split than the one actually in use,
    # with no warning. Matches the same caching-invalidation care already
    # taken for --top-identity-candidates (a real CLI flag, unlike these).
    # open_set_* included for exactly the same reason SPLIT_SEED et al.
    # are: an open-set run REMOVES whole identities from the gallery, so a
    # cached full-gallery index would look "fresh" by product count alone
    # only if the count happened to match -- and worse, a cached OPEN-SET
    # index (missing identities) could be silently reused by a subsequent
    # normal run, quietly deflating every headline number with no warning.
    #
    # INCREMENTAL ENROLLMENT (2026-08-03): these invalidating fields are
    # split out from num_products deliberately. Everything in `core`
    # changes what an embedding MEANS, so any change to it invalidates
    # every cached vector. num_products does not -- adding a brand leaves
    # every existing product's embedding perfectly valid. Keeping the two
    # in one fingerprint is what forced a full re-encode of the entire
    # catalog every time a single new brand landed, which is why the local
    # index sat at 872 products while the catalog grew past 2,300.
    #
    # This is also the exact mechanism behind this project's "you can add
    # new products without retraining" claim (docs/roadmap.md's enrollment
    # argument, spec section 8.1): embed the new items with the unchanged
    # encoder and append them to the gallery. Making it real here means
    # the claim is now something the code actually does, not just an
    # argument about metric learning.
    core = {
        "checkpoint": dino_checkpoint, "use_projection": use_projection,
        "split_seed": SPLIT_SEED, "test_images_per_product": TEST_IMAGES_PER_PRODUCT,
        "val_images_per_product": VAL_IMAGES_PER_PRODUCT,
        "open_set_holdout_fraction": open_set_holdout_fraction,
        "open_set_split_seed": OPEN_SET_SPLIT_SEED if open_set_holdout_fraction else None,
    }
    expected = {**core, "num_products": len(product_codes)}

    # Which exact images back each product's embedding. Compared per
    # product on load so a product whose gallery images CHANGED (recrop,
    # rescrape, different split) gets re-embedded even though its code is
    # unchanged -- a code-only check would silently keep a stale vector.
    # Sliced identically to what actually gets embedded below -- if these
    # two ever disagree, cached vectors silently stop matching the images
    # they claim to represent.
    signature = {code: list(gallery_images_by_product[code][:GALLERY_IMAGES_PER_PRODUCT])
                 for code in product_codes}

    reusable = {}
    if config_path.is_file():
        try:
            cached_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached_config = {}
        if all(cached_config.get(key) == value for key, value in core.items()):
            try:
                payload = torch.load(INDEX_DIR / "identity_embeddings.pt", map_location="cpu", weights_only=False)
                cached_signature = payload.get("gallery_signature") or {}
                cached_embeddings = payload["embeddings"]
                for position, code in enumerate(payload["product_codes"]):
                    # No signature at all = an index written before this
                    # feature existed. Reuse nothing rather than trust a
                    # vector whose backing images can't be verified.
                    if code in signature and cached_signature.get(code) == signature[code]:
                        reusable[code] = cached_embeddings[position]
            except (OSError, KeyError, RuntimeError) as error:
                print(f"Identity index: cached embeddings unusable ({error}) -- rebuilding from scratch.")
                reusable = {}

    if len(reusable) == len(product_codes):
        print(f"Identity index: cache hit ({len(product_codes):,} products).")
        product_embeddings = torch.stack([reusable[code] for code in product_codes])
        # Rewrite the config so num_products reflects reality even when
        # every vector came from cache (e.g. products were REMOVED).
        config_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
        torch.save({"product_codes": product_codes, "embeddings": product_embeddings, "gallery_signature": signature},
                   INDEX_DIR / "identity_embeddings.pt")
        return product_codes, product_embeddings

    to_embed = [code for code in product_codes if code not in reusable]
    if reusable:
        print(f"Identity index: ENROLLING {len(to_embed):,} new/changed products, reusing {len(reusable):,} cached embeddings "
              f"(no re-encode of the existing catalog).")
    else:
        print(f"Identity index: (re)building (up to {GALLERY_IMAGES_PER_PRODUCT} "
              f"gallery images per product)...")
    flat_paths, owners = [], []
    for code in to_embed:
        for path in gallery_images_by_product[code][:GALLERY_IMAGES_PER_PRODUCT]:
            flat_paths.append(path)
            owners.append(code)

    # Encoded BY PATH so peak RAM is bounded by IMAGE_DECODE_CHUNK rather
    # than by the encode batch -- decoding a full 64-image batch of
    # ~2880x3600 product photos at once is what OOM-killed a Colab
    # session here. Unreadable files are still skipped rather than
    # crashing the build (the catalog build verify()s images, but that
    # doesn't catch every corruption mode); kept_positions keeps `owners`
    # aligned across those drops.
    all_embeddings, kept_positions = embed_paths_dino(
        backbone, projection_head, use_projection, processor, flat_paths,
        progress="Encoding gallery",
    )
    valid_owners = [owners[position] for position in kept_positions]

    embeddings_by_product = defaultdict(list)
    for embedding, code in zip(all_embeddings, valid_owners):
        embeddings_by_product[code].append(embedding)
    newly_embedded = {
        code: F.normalize(torch.stack(vectors).mean(dim=0), dim=-1)
        for code, vectors in embeddings_by_product.items()
    }

    # Merge cached + newly-enrolled, preserving the sorted product_codes
    # order. Drop any product with no embedding from either source -- i.e.
    # one whose every gallery image failed to load (rare given the
    # catalog-build verify() pass, but a product left with zero vectors
    # would otherwise crash the stack() below).
    merged = {**reusable, **newly_embedded}
    product_codes = [code for code in product_codes if code in merged]
    product_embeddings = torch.stack([merged[code] for code in product_codes])
    # Signature is rewritten from the surviving codes only, so a product
    # dropped for unreadable images doesn't leave a signature entry that
    # would make a later run think it's cached.
    surviving_signature = {code: signature[code] for code in product_codes}

    expected["num_products"] = len(product_codes)
    torch.save({"product_codes": product_codes, "embeddings": product_embeddings, "gallery_signature": surviving_signature},
               INDEX_DIR / "identity_embeddings.pt")
    config_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return product_codes, product_embeddings


# ============================================================
# The pipeline itself
# ============================================================

def roc_auc(positive_scores, negative_scores):
    """Rank-based (Mann-Whitney U) AUROC, with proper average-rank tie
    handling. Written out rather than pulled from sklearn/scipy to avoid
    adding a dependency to a script that already runs in three different
    environments (local/Colab/Modal). Interpretation here: the
    probability that a randomly chosen in-catalog correct answer scores
    higher than a randomly chosen off-catalog query."""
    positives = np.asarray(positive_scores, dtype=float)
    negatives = np.asarray(negative_scores, dtype=float)
    if len(positives) == 0 or len(negatives) == 0:
        return None

    combined = np.concatenate([positives, negatives])
    order = combined.argsort(kind="mergesort")
    ranks = np.empty(len(combined), dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=float)

    sorted_values = combined[order]
    start = 0
    while start < len(sorted_values):
        stop = start
        while stop + 1 < len(sorted_values) and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        if stop > start:
            ranks[order[start:stop + 1]] = (start + 1 + stop + 1) / 2.0
        start = stop + 1

    positive_rank_sum = ranks[:len(positives)].sum()
    return float((positive_rank_sum - len(positives) * (len(positives) + 1) / 2.0) / (len(positives) * len(negatives)))


def _threshold_row(threshold, correct_scores, open_set_scores):
    return {
        "threshold": float(threshold),
        # Correct in-catalog answers this threshold would throw away.
        "false_reject_rate": float(np.mean(correct_scores < threshold)),
        # Off-catalog queries this threshold would still confidently
        # assert some (necessarily wrong) product for.
        "false_accept_rate": float(np.mean(open_set_scores >= threshold)),
    }


def open_set_threshold_table(correct_scores, open_set_scores):
    """The operating table --reject-threshold has never had: for each
    candidate threshold, BOTH error rates it buys. Percentile rows are
    anchored on the correct-answer distribution (so they stay meaningful
    whatever absolute score scale a checkpoint happens to produce);
    best_balanced is the threshold minimizing the sum of both error rates
    (equivalently, maximizing Youden's J) over every observed score."""
    rows = {}
    for percentile in (1, 2, 5, 10, 20, 30, 50):
        rows[f"p{percentile}"] = _threshold_row(np.percentile(correct_scores, percentile), correct_scores, open_set_scores)

    grid = np.unique(np.concatenate([correct_scores, open_set_scores]))
    best = min(grid, key=lambda t: np.mean(correct_scores < t) + np.mean(open_set_scores >= t))
    rows["best_balanced"] = _threshold_row(best, correct_scores, open_set_scores)
    return rows


class HierarchicalRetriever:
    def __init__(self, open_set_holdout_fraction=OPEN_SET_HOLDOUT_FRACTION):
        # Lazily constructed on first use so the OCR model is only paid for
        # when --brand-boost is actually on.
        self._brand_detector = None
        self._brand_evidence_cache = {}
        self.siglip2_model, self.siglip2_processor, siglip2_checkpoint = load_siglip2()
        self.dino_backbone, self.dino_head, self.dino_use_projection, dino_checkpoint = load_dinov3()
        self.dino_processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL_ID, token=os.environ.get("HF_TOKEN"))

        self.open_set_holdout_fraction = open_set_holdout_fraction
        self.open_set_identities = pick_open_set_identities(open_set_holdout_fraction)
        self.gallery_images_by_product, self.test_image_by_product, self.open_set_queries = make_view_split(self.open_set_identities)
        if self.open_set_identities:
            print(f"Open-set split: {len(self.open_set_identities):,} of {len(IDENTITY_TO_PRODUCT_CODES):,} identities held "
                  f"OUT of the gallery entirely ({len(self.open_set_queries):,} off-catalog queries). "
                  f"R@K from this run is NOT comparable to normal-run numbers -- the gallery is smaller.")

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
            dino_checkpoint, self.gallery_images_by_product, open_set_holdout_fraction,
        )
        self.gallery_index_by_code = {code: i for i, code in enumerate(self.gallery_product_codes)}

        # ---- Move every index onto the GPU, once -------------------
        # These are built/loaded as CPU tensors (they get torch.save'd, and
        # a CPU artifact stays portable across machines), but every read of
        # them afterwards is a matmul against a query embedding. Leaving
        # them on the host meant the whole scoring path ran on Colab's 2
        # vCPUs while the T4 sat idle -- the actual reason evaluation was
        # slow while "barely using compute units". They are small (a few
        # thousand rows), so this costs a few MB of VRAM.
        self.identity_embeddings = self.identity_embeddings.to(DEVICE)
        self.hsc_leaf_embeddings = self.hsc_leaf_embeddings.to(DEVICE)
        self.gallery_embeddings = self.gallery_embeddings.to(DEVICE)

        # ---- Precomputed shortlist masks ---------------------------
        # shortlist_identities used to rebuild its candidate list with a
        # Python comprehension over every identity on EVERY query, then
        # gather a fresh sub-matrix from identity_embeddings -- an O(N*D)
        # host copy per query (~4MB x 1,190 queries per eval arm). The set
        # of distinct `allowed_categories` values is tiny (bounded by the
        # HSC tree's nodes), so both are cached by that set instead.
        self._shortlist_cache = {}
        self._identity_index_by_name = {name: i for i, name in enumerate(self.identities)}
        self._open_set_identity_positions = torch.tensor(
            [i for i, identity in enumerate(self.identities) if identity in self.open_set_identities],
            dtype=torch.long,
        )
        # Cached embedding row -> product codes, so the shortlist expansion
        # below doesn't re-hash IDENTITY_TO_PRODUCT_CODES per query.
        self._codes_by_identity_index = [
            list(IDENTITY_TO_PRODUCT_CODES[identity]) for identity in self.identities
        ]

    def _candidate_set(self, allowed_categories):
        """(indices, sub_embeddings) for a given category gate, cached.

        Keyed on the frozenset of allowed categories -- `None` (no gate)
        is its own key and is by far the hottest one.
        """
        key = None if allowed_categories is None else frozenset(allowed_categories)
        cached = self._shortlist_cache.get(key)
        if cached is not None:
            return cached

        excluded = self.open_set_identities
        if key is None:
            candidate_indices = [i for i, identity in enumerate(self.identities) if identity not in excluded]
        else:
            candidate_indices = [i for i, identity in enumerate(self.identities)
                                 if self.identity_category.get(identity) in key and identity not in excluded]
        if not candidate_indices:
            # Same fallback as before: an empty gate degrades to ungated
            # rather than returning nothing.
            return self._candidate_set(None)

        # Device taken from the tensor being indexed, not the global
        # DEVICE: torch requires the index to sit on the same device as
        # the indexed tensor, and hardcoding DEVICE breaks the moment the
        # two disagree.
        index_tensor = torch.tensor(candidate_indices, dtype=torch.long,
                                    device=self.identity_embeddings.device)
        sub_embeddings = self.identity_embeddings[index_tensor].contiguous()
        entry = (candidate_indices, sub_embeddings)
        self._shortlist_cache[key] = entry
        return entry

    def hsc_predict(self, siglip_image_embedding, threshold=HSC_THRESHOLD):
        similarity = (siglip_image_embedding.to(self.hsc_leaf_embeddings.device)
                      @ self.hsc_leaf_embeddings.T).squeeze(0)
        leaf_probabilities = F.softmax(similarity / HSC_TEMPERATURE, dim=0)
        # Brought back to the host as one transfer: hsc_climb walks the
        # tree in Python and reads individual leaf probabilities, so
        # leaving this on the GPU turns each read into its own sync. The
        # tensor is ~42 floats, so the copy is free next to that.
        return hsc_climb(leaf_probabilities.cpu(), self.hsc_leaf_ids, threshold)

    def shortlist_identities(self, siglip_image_embedding, allowed_categories, top_k):
        # Open-set holdout has to be applied HERE too, not just in the
        # DINOv3 gallery: rerank_by_identity already drops codes missing
        # from the gallery, so a held-out identity could never be returned
        # either way -- but leaving it in the shortlist would let it
        # consume top_k slots that then silently evaporate, shrinking the
        # effective shortlist size and confounding the very parameter
        # (--top-identity-candidates) this project has tuned most.
        candidate_indices, sub_embeddings = self._candidate_set(allowed_categories)
        similarity = (siglip_image_embedding.to(sub_embeddings.device) @ sub_embeddings.T).squeeze(0)
        k = min(top_k, len(candidate_indices))
        top_scores, top_indices = torch.topk(similarity, k)
        # One device->host transfer for the whole top-k, instead of a
        # separate float(similarity[i]) sync inside the loop below.
        top_local = top_indices.tolist()
        top_score_values = top_scores.tolist()

        # product_code -> SigLIP2 identity-level score, for optional score
        # fusion in rerank_by_identity. Every colorway sibling of the same
        # identity gets the SAME score here (see USE_SCORE_FUSION's module
        # comment) -- that's inherent to what SigLIP2 was trained to
        # distinguish, not a bug in this expansion step.
        siglip_score_by_code = {}
        for local_index, identity_score in zip(top_local, top_score_values):
            global_index = candidate_indices[local_index]
            for code in self._codes_by_identity_index[global_index]:
                siglip_score_by_code[code] = identity_score
        return siglip_score_by_code

    def brand_evidence_for(self, image_path, min_score=BRAND_EVIDENCE_MIN_SCORE):
        """OCR-derived brand for a query image, or None (spec 4.5).

        Cached per path: an --evaluate sweep re-runs the same 1,190 query
        images for every arm, and OCR is far more expensive per image than
        the matmuls it feeds.
        """
        key = str(image_path)
        if key not in self._brand_evidence_cache:
            if self._brand_detector is None:
                from brand_evidence import BrandDetector
                self._brand_detector = BrandDetector()
                print(f"Brand evidence: easyocr on {self._brand_detector.device}")
            try:
                evidence = self._brand_detector.detect(image_path)
            except Exception as error:      # never let OCR break retrieval
                print(f"Brand evidence: skipped {image_path} ({error})")
                self._brand_evidence_cache[key] = None
                return None
            self._brand_evidence_cache[key] = (
                _brand_key(evidence.brand) if evidence.brand and evidence.score >= min_score else None
            )
        return self._brand_evidence_cache[key]

    def rerank_by_identity(self, dino_image_embedding, siglip_score_by_code, final_top_k,
                           use_score_fusion=USE_SCORE_FUSION, evidence_brand=None,
                           brand_boost_weight=BRAND_BOOST_WEIGHT):
        available = [code for code in siglip_score_by_code if code in self.gallery_index_by_code]
        if not available:
            return []
        indices = torch.tensor([self.gallery_index_by_code[code] for code in available],
                               dtype=torch.long, device=self.gallery_embeddings.device)
        dino_similarity = (dino_image_embedding.to(self.gallery_embeddings.device)
                           @ self.gallery_embeddings[indices].T).squeeze(0)

        if use_score_fusion and len(available) > 1:
            siglip_similarity = torch.tensor([siglip_score_by_code[code] for code in available],
                                             dtype=torch.float32, device=dino_similarity.device)
            # z-score normalize each signal within this candidate set before
            # combining -- SigLIP2 and DINOv3 cosine similarities don't
            # share a comparable scale/spread, so mixing raw values would
            # arbitrarily over- or under-weight whichever happens to have
            # the larger numeric range for this particular query.
            def z_normalize(values):
                std = values.std()
                return (values - values.mean()) / std if std > 1e-6 else torch.zeros_like(values)

            fused = DINO_FUSION_WEIGHT * z_normalize(dino_similarity) + SIGLIP_FUSION_WEIGHT * z_normalize(siglip_similarity)
            ranking_score = fused
        else:
            ranking_score = dino_similarity

        # Brand evidence (spec 4.5): an ADDITIVE bonus on candidates whose
        # catalog brand matches the brand OCR read off the query image.
        # Additive-only by design -- see USE_BRAND_BOOST's comment. Nothing
        # is removed from `available`, so a wrong brand read can at worst
        # reorder, never make the true product unreachable the way the
        # category gate does.
        if evidence_brand and brand_boost_weight:
            bonus = torch.tensor(
                [brand_boost_weight if _brand_key(CATALOG[code]["brand"]) == evidence_brand else 0.0
                 for code in available],
                dtype=ranking_score.dtype, device=ranking_score.device)
            ranking_score = ranking_score + bonus

        order = torch.argsort(ranking_score, descending=True)

        # Pull the scores across ONCE. The old form indexed the similarity
        # tensor inside the comprehension -- `float(dino_similarity[i])`
        # per element -- which on GPU is a separate device->host sync per
        # candidate. evaluate() passes final_top_k = len(gallery), so that
        # was up to ~1,272 syncs per query, on top of another per element
        # from order.tolist(). Both are now single transfers.
        dino_scores = dino_similarity.tolist()
        ranked = [(available[i], dino_scores[i]) for i in order.tolist()]
        return ranked[:final_top_k]

    def patch_rerank(self, query_image, pooled_ranked, num_candidates=PATCH_RERANK_CANDIDATES):
        """Re-scores the top num_candidates of an already pooled-DINOv3-
        ranked list using patch-level late-interaction similarity, then
        re-sorts just that window (anything beyond it keeps its pooled
        order, since this is a polish step over the pooled shortlist, not
        an independent full rerank)."""
        window = pooled_ranked[:num_candidates]
        rest = pooled_ranked[num_candidates:]
        if len(window) <= 1:
            return pooled_ranked

        query_patches = embed_image_dino_patches(self.dino_backbone, self.dino_head, self.dino_use_projection, self.dino_processor, query_image)

        rescored = []
        for code, pooled_score in window:
            candidate_paths = self.gallery_images_by_product.get(code, [])
            if not candidate_paths:
                rescored.append((code, pooled_score, pooled_score))
                continue
            try:
                candidate_image = load_rgb_image(candidate_paths[0])
            except Exception:
                rescored.append((code, pooled_score, pooled_score))
                continue
            candidate_patches = embed_image_dino_patches(self.dino_backbone, self.dino_head, self.dino_use_projection, self.dino_processor, candidate_image)
            patch_score = patch_similarity_score(query_patches, candidate_patches)
            rescored.append((code, pooled_score, patch_score))

        rescored.sort(key=lambda entry: entry[2], reverse=True)
        return [(code, pooled_score) for code, pooled_score, _ in rescored] + rest

    def retrieve(self, image_path, use_category_gate=False, hsc_threshold=HSC_THRESHOLD, top_identity_candidates=TOP_IDENTITY_CANDIDATES, final_top_k=FINAL_TOP_K, reject_threshold=REJECT_SIMILARITY_THRESHOLD, use_score_fusion=USE_SCORE_FUSION, use_patch_rerank=USE_PATCH_RERANK, use_brand_boost=USE_BRAND_BOOST, brand_boost_weight=BRAND_BOOST_WEIGHT):
        # Default flipped to off: the old flat (non-HSC) gate was confirmed
        # net-negative in two independent Phase 4 runs (docs/eval_log.md,
        # 2026-07-30) -- it excluded the true category ~30% of the time,
        # unrecoverable, for no rerank-quality benefit. Real HSC climbing
        # (2026-08-01) backs off to a broader, more-confident ancestor
        # instead of blindly committing to a low-margin leaf, which should
        # reduce that exclusion rate -- re-benchmark both arms once this
        # has real eval numbers rather than assuming HSC fixes it.
        image = load_rgb_image(image_path)

        siglip_embedding = embed_images_siglip(self.siglip2_model, self.siglip2_processor,
                                               [image], keep_on_device=True)
        dino_embedding = embed_images_dino(self.dino_backbone, self.dino_head, self.dino_use_projection,
                                           self.dino_processor, [image], keep_on_device=True)

        hsc_result = self.hsc_predict(siglip_embedding, threshold=hsc_threshold)
        allowed_categories = hsc_result["allowed_categories"] if use_category_gate else None

        candidates = self.shortlist_identities(siglip_embedding, allowed_categories, top_identity_candidates)
        # Pull a wider pooled-ranked window when patch reranking is on --
        # patch_rerank needs candidates to re-sort AMONG, so truncating to
        # final_top_k before it runs would leave it nothing to work with.
        pooled_top_k = max(final_top_k, PATCH_RERANK_CANDIDATES) if use_patch_rerank else final_top_k
        evidence_brand = self.brand_evidence_for(image_path) if use_brand_boost else None
        ranked = self.rerank_by_identity(dino_embedding, candidates, pooled_top_k,
                                         use_score_fusion=use_score_fusion,
                                         evidence_brand=evidence_brand,
                                         brand_boost_weight=brand_boost_weight)
        if use_patch_rerank:
            ranked = self.patch_rerank(image, ranked)[:final_top_k]

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
            # abs(), not a bare subtraction: `results` is sorted by whatever
            # score actually determined the ranking (fused score under
            # --score-fusion, patch score under --patch-rerank), but
            # `dino_identity_score` always holds the raw pooled DINOv3
            # score regardless of which method ranked it -- so results[0]
            # can legitimately have a LOWER dino_identity_score than
            # results[1] once either flag reorders the top-2 relative to
            # pooled-DINO order. A bare subtraction then goes negative,
            # which is unconditionally < AMBIGUITY_MARGIN, so the
            # ambiguity flag fired spuriously on nearly every query where
            # fusion/patch-rerank actually changed anything -- found via
            # code review, 2026-08-02, verified against both callers'
            # actual tuple contents (rerank_by_identity/patch_rerank both
            # only ever return the pooled DINO score, never the score that
            # sorted them). abs() is correct either way: "how far apart
            # are these two results' own DINOv3 confidence" is meaningful
            # regardless of which method chose them as the top 2.
            close_scores = abs(results[0]["dino_identity_score"] - results[1]["dino_identity_score"]) < AMBIGUITY_MARGIN
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
            "brand_evidence": evidence_brand,
            "same_model_different_colorway_ambiguous": ambiguous,
            "rejected_open_set": rejected, "reject_threshold": reject_threshold,
        }

    # --------------------------------------------------------
    # End-to-end held-out evaluation
    # --------------------------------------------------------

    @torch.inference_mode()
    def _query_embeddings(self, queries):
        """SigLIP2 + DINOv3 embeddings for every query image, GPU-resident.

        Cached on the instance -- see the call site in evaluate() for why
        that is safe. Images are decoded CHUNK BY CHUNK, never all at
        once: holding 1,190+ full-resolution photos in memory at the same
        time is real pressure on a Colab runtime. Only the embeddings (a
        few MB) are kept. Decoding inside each chunk is threaded so the
        CPU stages the next batch while the GPU is still working on the
        current one.
        """
        cache_key = tuple(path for _, path in queries)
        if getattr(self, "_query_embedding_cache", (None,))[0] == cache_key:
            return self._query_embedding_cache[1], self._query_embedding_cache[2]

        paths = [path for _, path in queries]
        all_siglip, siglip_kept = embed_paths_siglip(
            self.siglip2_model, self.siglip2_processor, paths,
            keep_on_device=True, progress="Embedding queries (SigLIP2)")
        all_dino, dino_kept = embed_paths_dino(
            self.dino_backbone, self.dino_head, self.dino_use_projection,
            self.dino_processor, paths,
            keep_on_device=True, progress="Embedding queries (DINOv3)")

        # evaluate() indexes these positionally against `queries`, so a
        # silently dropped image would pair every later query with the
        # wrong vector -- a corrupted R@K rather than an error. Loud.
        expected = list(range(len(paths)))
        if siglip_kept != expected or dino_kept != expected:
            dropped = sorted(set(expected) - (set(siglip_kept) & set(dino_kept)))
            raise RuntimeError("Unreadable query image(s) would misalign the eval: "
                               f"{[paths[i] for i in dropped]}")
        self._query_embedding_cache = (cache_key, all_siglip, all_dino)
        return all_siglip, all_dino

    def evaluate(self, use_category_gate=True, hsc_threshold=HSC_THRESHOLD, top_identity_candidates=TOP_IDENTITY_CANDIDATES, use_score_fusion=USE_SCORE_FUSION, use_patch_rerank=USE_PATCH_RERANK, use_brand_boost=USE_BRAND_BOOST, brand_boost_weight=BRAND_BOOST_WEIGHT):
        queries = [(code, path) for code, paths in self.test_image_by_product.items() for path in paths]
        print(f"Evaluating {len(queries):,} held-out queries (category gate: {use_category_gate}, HSC threshold: {hsc_threshold}, score fusion: {use_score_fusion}, patch rerank: {use_patch_rerank}, brand boost: {use_brand_boost})...")

        ranks = []
        gate_exclusions = 0
        identity_shortlist_misses = 0
        # Brand evidence, measured in-run against ground truth rather than
        # assumed from the standalone brand_evidence_eval.py numbers -- the
        # query images here are the held-out view split, not that eval's
        # sample, so the fire/correct rates have to be re-counted on the
        # actual queries any reported R@1 delta is computed over.
        brand_evidence_fired = 0
        brand_evidence_correct = 0
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

        # Batch-embed every query image ONCE upfront (in chunks, not all
        # 1,190+ full-res images loaded into memory simultaneously -- see
        # below), instead of the old per-query loop calling
        # embed_images_siglip/embed_images_dino with a single-image list
        # each time (effectively batch_size=1 for 1,190+ individual model
        # forward passes per encoder per eval call -- confirmed as the
        # real bottleneck behind Colab runs reported taking 4+ hours,
        # 2026-08-02: GPU forward passes are far more efficient batched
        # than issued one at a time, and this cost was being paid twice
        # per --evaluate call (gated + ungated) times however many
        # --top-identity-candidates values get swept). The per-query
        # control flow below (hsc_predict/shortlist_identities/
        # rerank_by_identity) is unchanged -- those methods assume a
        # single [1, D] embedding row (see hsc_predict's `.squeeze(0)`),
        # so this only batches the actual expensive step (model.forward())
        # and still loops per-query for the cheap downstream logic, rather
        # than risking a behavior change by trying to vectorize HSC
        # climbing/shortlisting/reranking themselves.
        #
        # Images are loaded and embedded CHUNK BY CHUNK, not all at once --
        # 1,190+ full-resolution product photos held in memory
        # simultaneously (not yet resized/tensorized) risks real memory
        # pressure on a constrained instance (e.g. Colab). Raw images are
        # discarded after each chunk is embedded; only the embeddings
        # (tiny -- a few MB total for the whole query set) are kept.
        # patch_rerank (off by default) needs the raw image back later --
        # reloaded from disk on demand in the loop below rather than kept
        # in memory the whole time, a cheap disk read next to the model
        # forward passes this fix is actually targeting.
        # Cached across evaluate() calls, keyed by the query set itself.
        # The embeddings depend ONLY on the images and the (frozen)
        # encoders -- not on use_category_gate, hsc_threshold,
        # top_identity_candidates, score fusion or patch rerank. Every one
        # of those is a pure post-processing choice over these vectors.
        # Re-encoding per call meant a --top-identity-candidates sweep of
        # S values paid the full encode cost 2*S times (gated + ungated
        # per value) for bit-identical results. This makes a sweep cost
        # one encode pass total.
        all_siglip_embeddings, all_dino_embeddings = self._query_embeddings(queries)

        for query_index, (true_code, image_path) in enumerate(queries):
            siglip_embedding = all_siglip_embeddings[query_index:query_index + 1]
            dino_embedding = all_dino_embeddings[query_index:query_index + 1]
            image = load_rgb_image(image_path) if use_patch_rerank else None

            hsc_result = self.hsc_predict(siglip_embedding, threshold=hsc_threshold)
            climb_level_counts[hsc_result["predicted_level"]] += 1
            true_category = CATALOG[true_code]["category"]
            allowed_categories = hsc_result["allowed_categories"] if use_category_gate else None
            if use_category_gate and true_category not in allowed_categories:
                gate_exclusions += 1

            candidates = self.shortlist_identities(siglip_embedding, allowed_categories, top_identity_candidates)
            if true_code not in candidates:
                identity_shortlist_misses += 1

            evidence_brand = self.brand_evidence_for(image_path) if use_brand_boost else None
            if evidence_brand:
                brand_evidence_fired += 1
                if evidence_brand == _brand_key(CATALOG[true_code]["brand"]):
                    brand_evidence_correct += 1
            ranked = self.rerank_by_identity(dino_embedding, candidates, len(self.gallery_product_codes),
                                             use_score_fusion=use_score_fusion,
                                             evidence_brand=evidence_brand,
                                             brand_boost_weight=brand_boost_weight)
            if use_patch_rerank:
                # patch_rerank() only re-sorts its own top PATCH_RERANK_CANDIDATES
                # window and leaves the rest of `ranked` untouched by design (see
                # its own docstring) -- correct here too: this is a polish step
                # over the pooled ranking, not an independent full rerank, so a
                # query whose true product falls outside that window can't be
                # rescued by it, matching --image mode's behavior exactly.
                ranked = self.patch_rerank(image, ranked)
            ranked_codes = [code for code, _ in ranked]
            rank = ranked_codes.index(true_code) + 1 if true_code in ranked_codes else len(self.gallery_product_codes) + 1
            ranks.append(rank)
            if rank == 1:
                correct_top1_scores.append(ranked[0][1])

        ranks = np.asarray(ranks, dtype=float)
        correct_scores = np.asarray(correct_top1_scores, dtype=float)

        # ---- Open-set arm (spec section 8.1) -------------------------
        # Queries whose true identity was removed from the gallery
        # entirely, so NO answer at any rank is correct. Their top-1
        # scores are the negatives the reject threshold has to separate
        # from `correct_scores` (the positives). Empty unless
        # --open-set-holdout-fraction was passed, in which case every
        # number below is None and nothing about a normal run changes.
        open_set_scores = np.asarray(
            self._score_open_set_queries(use_category_gate, hsc_threshold, top_identity_candidates, use_score_fusion),
            dtype=float,
        )
        open_set_metrics = None
        if len(open_set_scores) > 0 and len(correct_scores) > 0:
            open_set_metrics = {
                "num_open_set_queries": int(len(open_set_scores)),
                "num_held_out_identities": len(self.open_set_identities),
                "open_set_score_mean": float(open_set_scores.mean()),
                "correct_score_mean": float(correct_scores.mean()),
                # Threshold-free separability: probability that a random
                # in-catalog correct answer outscores a random off-catalog
                # query. 0.5 = the DINOv3 score carries NO information
                # about whether the product is in the catalog at all, and
                # no choice of --reject-threshold can ever work; that
                # would be a real (negative) finding, not a bug.
                "auroc": roc_auc(correct_scores, open_set_scores),
                # The real operating table: each row is a threshold and
                # BOTH error rates it buys. false_reject = correct answers
                # thrown away; false_accept = off-catalog queries the
                # system still confidently asserts a product for.
                "threshold_table": open_set_threshold_table(correct_scores, open_set_scores),
            }
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
            "open_set": open_set_metrics,
            "brand_evidence": {
                "enabled": use_brand_boost,
                "boost_weight": brand_boost_weight if use_brand_boost else None,
                # Of all queries, how often OCR asserted any brand ...
                "fire_rate": brand_evidence_fired / len(queries),
                # ... and how often that assertion was the right brand.
                # precision_when_fired is the number that decides whether
                # this signal is safe to act on at all.
                "correct_rate": brand_evidence_correct / len(queries),
                "precision_when_fired": (brand_evidence_correct / brand_evidence_fired
                                         if brand_evidence_fired else None),
            } if use_brand_boost else None,
        }
        return metrics

    def _score_open_set_queries(self, use_category_gate, hsc_threshold, top_identity_candidates, use_score_fusion):
        """Top-1 DINOv3 score for every off-catalog query. Same shortlist
        -> rerank path as a real query (deliberately -- these have to be
        scored exactly the way a genuine unknown product would be at
        serving time, not through some shortcut). Returns [] when the
        open-set split is disabled. patch_rerank is intentionally not
        applied: it's confirmed harmful (docs/eval_log.md 2026-08-03) and
        re-sorting the top-10 window can only change WHICH wrong product
        wins here, not whether one is asserted."""
        if not self.open_set_queries:
            return []
        print(f"Scoring {len(self.open_set_queries):,} off-catalog (open-set) queries...")

        scores = []
        chunk_size = max(ENCODE_BATCH_SIZE, 32)
        for start in tqdm(range(0, len(self.open_set_queries), chunk_size),
                          desc="Open-set queries", unit="chunk"):
            chunk = self.open_set_queries[start:start + chunk_size]
            images, kept_positions = load_images_parallel([path for _, path in chunk])
            # Unlike the closed-set path this only appends scores (nothing
            # is indexed positionally against `chunk` afterwards), so a
            # dropped unreadable image is safe to skip rather than fatal.
            chunk = [chunk[position] for position in kept_positions]
            if not images:
                continue
            siglip_chunk = embed_images_siglip(self.siglip2_model, self.siglip2_processor,
                                               images, keep_on_device=True)
            dino_chunk = embed_images_dino(self.dino_backbone, self.dino_head, self.dino_use_projection,
                                           self.dino_processor, images, keep_on_device=True)
            for i in range(len(chunk)):
                siglip_embedding = siglip_chunk[i:i + 1]
                dino_embedding = dino_chunk[i:i + 1]
                allowed = self.hsc_predict(siglip_embedding, threshold=hsc_threshold)["allowed_categories"] if use_category_gate else None
                candidates = self.shortlist_identities(siglip_embedding, allowed, top_identity_candidates)
                ranked = self.rerank_by_identity(dino_embedding, candidates, len(self.gallery_product_codes), use_score_fusion=use_score_fusion)
                if ranked:
                    scores.append(ranked[0][1])
        return scores


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
    brand = metrics.get("brand_evidence")
    if brand:
        precision = brand["precision_when_fired"]
        if precision is None:
            print("Brand evidence (spec 4.5): never fired on any query")
        else:
            print(f"Brand evidence (spec 4.5, boost weight {brand['boost_weight']}): fired on "
                  f"{brand['fire_rate']*100:.2f}% of queries, correct on "
                  f"{brand['correct_rate']*100:.2f}% of all queries, "
                  f"precision when fired {precision*100:.2f}%")
    if metrics["reject_threshold_sweep"]:
        print("Open-set reject-threshold sweep (false-reject rate against REAL correct top-1 answers only --")
        print("  not a true accept/reject rate, this eval set has no genuinely-unknown-product queries yet):")
        for label, row in metrics["reject_threshold_sweep"].items():
            print(f"    {label}: threshold={row['threshold']:.4f}  false_reject_rate={row['false_reject_rate']*100:.2f}%")

    open_set = metrics.get("open_set")
    if open_set:
        print(f"\nOPEN-SET REJECTION (spec 8.1) -- {open_set['num_held_out_identities']:,} identities held fully out of "
              f"the gallery, {open_set['num_open_set_queries']:,} off-catalog queries with NO correct answer at any rank.")
        print(f"  NOTE: R@K above is NOT comparable to normal-run numbers -- this run's gallery is smaller.")
        print(f"  Mean top-1 score: correct in-catalog {open_set['correct_score_mean']:.4f} vs off-catalog {open_set['open_set_score_mean']:.4f}")
        auroc = open_set["auroc"]
        print(f"  AUROC (separability of in-catalog vs off-catalog): {auroc:.4f}"
              f"{'  <-- ~0.5 means NO threshold can work' if auroc is not None and auroc < 0.6 else ''}")
        print("  Threshold operating table (pick a row, pass it as --reject-threshold):")
        for label, row in open_set["threshold_table"].items():
            print(f"    {label:>13}: threshold={row['threshold']:.4f}  "
                  f"false_reject={row['false_reject_rate']*100:5.2f}%  false_accept={row['false_accept_rate']*100:5.2f}%")


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
    parser.add_argument("--score-fusion", action="store_true",
                         help="Rerank by a weighted fusion of DINOv3 + SigLIP2 identity scores instead of "
                              "DINOv3-only (spec sections 2.1/6). Off by default -- unvalidated new arm, "
                              "benchmark against the real 47.65%% ungated baseline before trusting it. Only "
                              "affects ranking ACROSS different identities in the shortlist -- SigLIP2's score "
                              "is identical across colorway siblings of the same identity by design, so this "
                              "can't help DINOv3 pick between those.")
    parser.add_argument("--patch-rerank", action="store_true",
                         help="CONFIRMED SEVERELY HARMFUL, real checkpoint, 2026-08-03 -- R@1 50.76%%->20.59%% "
                              "at K=50 (docs/eval_log.md). Kept as a flag for research/debugging only, "
                              "do not enable for any real use. See USE_PATCH_RERANK's module comment for why.")
    parser.add_argument("--top-identity-candidates", type=int, default=TOP_IDENTITY_CANDIDATES,
                         help=f"SigLIP2 identity-shortlist size fed to DINOv3's rerank (default {TOP_IDENTITY_CANDIDATES}). "
                              "The single most validated lever in this project's history (10->25 drove the "
                              "36.72%%->47.65%% jump, docs/eval_log.md 2026-07-31) -- sweeping past 25 (e.g. 35/50/75/100) "
                              "is the top-priority untested experiment per docs/roadmap.md's 2026-08-02 analysis. "
                              "Exposed as a flag so this can be swept without editing the module constant each time.")
    parser.add_argument("--brand-boost", action="store_true",
                         help="Spec 4.5 brand evidence: OCR the query image (brand_evidence.py), fuzzy-match "
                              "the text against the catalog's brand vocabulary, and ADD --brand-boost-weight to "
                              "the rerank score of every candidate with that brand. A boost, never a filter -- "
                              "nothing is excluded, so a wrong brand read can only reorder, unlike the category "
                              "gate (net-negative six times, docs/eval_log.md). Off by default; see the measured "
                              "precision/recall and no-text rate before enabling.")
    parser.add_argument("--brand-boost-weight", type=float, default=BRAND_BOOST_WEIGHT,
                         help=f"Size of that bonus on the DINOv3 cosine scale (default {BRAND_BOOST_WEIGHT}). "
                              f"AMBIGUITY_MARGIN is {AMBIGUITY_MARGIN}, so this order of magnitude breaks "
                              "near-ties without overturning a confident DINOv3 decision.")
    parser.add_argument("--open-set-holdout-fraction", type=float, default=OPEN_SET_HOLDOUT_FRACTION,
                         help="Fraction of catalog IDENTITIES (0-1) to remove from the gallery entirely, turning all "
                              "their images into off-catalog queries with no correct answer -- spec section 8.1's "
                              "open-set rejection split, the missing ingredient for calibrating --reject-threshold "
                              "(which has shipped uncalibrated because the existing eval can only measure FALSE "
                              "rejects, never false ACCEPTS). Try 0.1. Off by default. IMPORTANT: an open-set run's "
                              "R@K is NOT comparable to the normal benchmark rows in docs/eval_log.md -- the gallery "
                              "is deliberately smaller -- so log it as its own row, never alongside them.")
    args = parser.parse_args()

    retriever = HierarchicalRetriever(open_set_holdout_fraction=args.open_set_holdout_fraction)

    if args.image:
        result = retriever.retrieve(args.image, use_category_gate=args.category_gate, hsc_threshold=args.hsc_threshold, final_top_k=args.top_k, reject_threshold=args.reject_threshold, use_score_fusion=args.score_fusion, use_patch_rerank=args.patch_rerank, top_identity_candidates=args.top_identity_candidates, use_brand_boost=args.brand_boost, brand_boost_weight=args.brand_boost_weight)
        print_result(args.image, result)

    if args.evaluate:
        gated_metrics = retriever.evaluate(use_category_gate=True, hsc_threshold=args.hsc_threshold, use_score_fusion=args.score_fusion, use_patch_rerank=args.patch_rerank, top_identity_candidates=args.top_identity_candidates, use_brand_boost=args.brand_boost, brand_boost_weight=args.brand_boost_weight)
        print_metrics("End-to-end held-out eval -- WITH HSC-based category gate", gated_metrics)

        ungated_metrics = retriever.evaluate(use_category_gate=False, hsc_threshold=args.hsc_threshold, use_score_fusion=args.score_fusion, use_patch_rerank=args.patch_rerank, top_identity_candidates=args.top_identity_candidates, use_brand_boost=args.brand_boost, brand_boost_weight=args.brand_boost_weight)
        print_metrics("End-to-end held-out eval -- WITHOUT category gate (fallback comparison)", ungated_metrics)

        with (INDEX_DIR / "pipeline_eval_metrics.json").open("w", encoding="utf-8") as f:
            json.dump({"with_category_gate": gated_metrics, "without_category_gate": ungated_metrics}, f, indent=2)

    if not args.image and not args.evaluate:
        print("Nothing to do -- pass --image PATH and/or --evaluate.")
