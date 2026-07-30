"""SigLIP2 hierarchical multi-positive fine-tune, on the full 6-brand /
13-category catalog (structured_caption only -- no flat `caption` field
anywhere in training or eval, per user direction: the old single-line
"hardcoded" caption generator is no longer needed now that structured
captions/positive_texts are good enough on their own).

Adapted from notebooks/fashionsiglip2_hsc_finetune.ipynb Part 4, with these
changes on top of the notebook's already-solid design:

1. Flat `caption` field dropped entirely -- no "long" label-kind bucket, no
   caption-based validation/test target. Every training label and every
   eval candidate comes from structured_caption (taxonomy_path, attributes,
   positive_texts). LABEL_KIND_WEIGHTS re-normalized without "long".

2. Data augmentation added (nothing existed before -- the notebook's
   ShoeDataset just loaded and RGB-converted). Per spec section 5.4
   ("preserve identity-critical evidence"): mild RandomResizedCrop
   (scale 0.75-1.0, not aggressive), mild ColorJitter (brightness/contrast
   only, no hue -- spec explicitly warns off strong hue shifts and
   grayscale for identity-critical retrieval), and light Gaussian blur to
   simulate compression/resize artifacts. No horizontal flip (spec: avoid
   flips where side-specific details matter -- true for shoes and
   asymmetric logos/graphics here). Applied train-side only, eval stays on
   clean images.

3. Larger in-batch negative count: P_PRODUCTS 8 -> 16 (batch size 16 -> 32
   images). Contrastive/multi-positive losses benefit directly from more
   in-batch negatives (standard CLIP/SigLIP scaling result) -- this is the
   single highest-leverage, lowest-risk lever available on a dataset this
   size without changing the loss itself. Still comfortably fits a T4 in
   fp16 with the backbone frozen.

4. Two-stage progressive unfreezing (spec section 5.3's own recommended
   fine-tuning order: frozen first, evaluate, only then unfreeze further).
   Stage 1 trains heads only (lr 1e-4, matches the notebook's original
   default). Stage 2 continues from stage 1's best checkpoint with the
   last transformer block also unfrozen (lr 1e-5) -- catches representation
   gains a heads-only linear probe can't reach, without the catastrophic-
   forgetting risk of unfreezing everything on ~1200 products. Whichever
   stage's checkpoint scores higher on validation R@1 is what final test
   metrics get reported against.

5. Selection score simplified to validation exact-label R@1 alone (used to
   be a 70/30 blend with caption R@1, which no longer exists).

6. Category-aware hard-negative batch sampling. Cross-category confusion
   (a shoe scored against a jacket's label) is already close to solved by
   SigLIP's pretraining -- the actual R@1 loss comes from within-category
   confusion (many similar sneaker colorways, many similar pairs of jeans).
   Uniform random in-batch negatives mostly waste batch slots on "easy"
   cross-category pairs that were never going to be confused. Each batch
   now spends CATEGORY_BIAS_FRACTION of its product slots on products from
   one randomly chosen category (seeded per batch) -- concentrating
   in-batch negatives on the actually-hard same-category comparisons -- and
   the rest on uniformly random products, so rare categories and
   cross-category general structure still get seen every epoch.

7. Category-scoped retrieval evaluation alongside the existing global one.
   Global R@1 scores every validation image against all ~1200+ exact
   labels in the whole catalog; category-scoped R@1 restricts the
   candidate pool to only same-category products (matching how a real
   query would actually be served once the spec's hierarchical/multi-index
   query execution, section 4.7-4.8, narrows by category before ranking).
   Reporting both quantifies how much of the global R@1 gap is genuine
   same-category confusion (category-scoped R@1) vs. dilution from
   unrelated cross-category distractors in the candidate pool.
   Result from the first full run: category-scoped R@1 beats global R@1 by
   ~1.5-2pt at every epoch (e.g. epoch 6: 9.62% vs 8.02%) -- real but
   modest; most of the remaining gap is genuine same-category confusion,
   not cross-category dilution.

8. SKU text removed entirely from both training and evaluation. A bare
   SKU/product code has no semantic content a text encoder can learn from
   -- exact per-colorway identity matching is DINOv3's job (the
   visual-identity encoder), not SigLIP2's (semantic encoder). The old
   "exact" label kind (identity + "(SKU)") is gone; the retrieval target
   is now `identity` (model-level: brand + product name + attributes),
   capped one level below full SKU-uniqueness. Two colorways that read
   identically at this level are expected to collide in this eval --
   that's the intended scope boundary with DINOv3, not a bug. "model"
   absorbs the old "exact" bucket's sampling weight since it's now the
   most specific kind.

9. Attribute coverage expanded well past color. The schema
   (newLLMprompt.py) supports color/material/fit/pattern/closure/
   silhouette/length as independent facets, plus free-text
   `defining_features` (present on ~87% of products -- e.g. "premium
   suede upper", "elasticized waist with inner drawcords", "triple
   Gazelle branding") -- all of it previously ignored except color/
   material/fit/pattern. This is the richest per-product visual-grounding
   signal in the schema and was sitting unused.

v2 real full-corpus test result (category-scoped candidate pool, the
retrieval-serving-realistic number): R@1 14.04%, R@5 36.83%, R@10 53.86%,
MRR 26.40%, median rank 9.0 -- run outside Modal, directly against the
Colab/Drive setup. This is the baseline v3 below is trying to beat.

v3 changes on top of v2, per explicit user direction (five requested
levers plus one added during implementation):

10. Stage 2 unfreezes the last NUM_UNFROZEN_BLOCKS=4 transformer blocks
    instead of just 1, with layer-wise LR decay (LAYERWISE_LR_DECAY):
    the block closest to the output trains at the full stage-2 LR, each
    block further back gets LR * decay^distance. Plain single-LR
    unfreezing of 4 blocks risks the lower blocks (which still carry a
    lot of ImageNet/WebLI-general representation) getting yanked around
    by the same LR that's appropriate for the output-adjacent block --
    layer-wise decay is the standard mitigation (ULMFiT/BERT
    discriminative fine-tuning), applied here via per-parameter-group
    optimizer LRs (build_optimizer_param_groups).

11. Offline nearest-neighbor hard-negative mining (mine_hard_negatives),
    replacing pure same-category-random negatives with same-category
    negatives that are specifically close in the *current* model's
    embedding space -- the actual confusable pairs, not just
    same-taxonomy-leaf ones. Mined from one-or-two representative image
    embeddings per product using the model snapshot at the start of each
    stage and re-mined every REMINE_EVERY_N_EPOCHS epochs (a one-shot
    mining from the frozen base model would go stale as soon as training
    moves the embedding space). ProductBatchSampler now draws
    HARD_NEGATIVE_BIAS_FRACTION of each batch's product slots from the
    seed product's mined neighbor pool, before falling back to the v2
    category-random and uniform-random pools for the remaining slots.

12. All positive captions used instead of sampling one per image per
    step. This turned out to need no structural change to the loss or
    positive_mask logic -- sample_training_label -> sample_training_labels
    (plural) draws TEXTS_PER_IMAGE distinct entries per image per batch
    instead of 1 and the collator flattens them into one wider text
    batch; positive_mask is already computed by checking each sampled
    text's key against each image's *full* valid-label set regardless of
    how many texts get sampled, and SigLIP2's image/text batches were
    already independently sized (logits_per_image is [num_images,
    num_texts], never required num_texts == num_images) so nothing else
    moves.

13. Multi-positive InfoNCE added as a selectable alternative to the v2
    sigmoid-pairwise loss (LOSS_TYPE = "sigmoid" default or "infonce").
    The v2 loss is already SigLIP's native per-pair sigmoid BCE
    generalized to multiple positives; softmax-normalized multi-positive
    InfoNCE (multi_positive_infonce_loss) is the SupCon-style
    alternative the user asked to benchmark it against. Kept as a flag
    rather than a hard swap since SigLIP2's pretrained logit_scale/
    logit_bias were tuned for the sigmoid formulation -- worth an actual
    ablation (run this script twice, LOSS_TYPE flipped) rather than
    assuming InfoNCE wins.

14. Batch shape increased P_PRODUCTS x K_IMAGES_PER_PRODUCT from 16x2=32
    to 16x4=64 (K to 4, per direct request). If this OOMs on your GPU
    with 4 unfrozen blocks (item 10) and TEXTS_PER_IMAGE=3 (item 12)
    inflating the text side too, drop P_PRODUCTS to 10-12 (batch 40-48)
    before touching resolution or augmentation -- those two levers are
    the ones actually driving memory here, not a global batch-size knob.
    Note gradient accumulation (unchanged from v2) does NOT increase
    in-batch negative count -- it only smooths the optimizer step across
    micro-batches -- so it does not substitute for raising P/K if what
    you want is more negatives; a true decoupled-negatives approach
    would need something like GradCache, not implemented here.
"""

from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from torch.utils.data import Dataset, DataLoader, Sampler

import gc
import json
import math
import os
import random
import re
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor, get_cosine_schedule_with_warmup


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = Path("/content/drive/MyDrive/apparel_dataset")
METADATA_PATH = DATASET_ROOT / "metadata.json"

BASE_MODEL_ID = "google/siglip2-base-patch16-384"

OUTPUT_DIR = DATASET_ROOT / "finetuned_siglip2_hierarchical_v3"
STAGE1_DIR = OUTPUT_DIR / "stage1_heads_best"
STAGE2_DIR = OUTPUT_DIR / "stage2_lastnblocks_best"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

SPLIT_MODE = "view"  # every product seen in training; held-out views test known-SKU retrieval
VAL_IMAGES_PER_PRODUCT = 2
TEST_IMAGES_PER_PRODUCT = 2

P_PRODUCTS = 16           # see v3 docstring item 14 re: OOM fallback
K_IMAGES_PER_PRODUCT = 4  # was 2 -- batch size 32 -> 64

# Each batch's product slots are filled in this priority order (v3 item 11):
# 1) HARD_NEGATIVE_BIAS_FRACTION from the seed product's mined nearest
#    neighbors, 2) CATEGORY_BIAS_FRACTION from the rest of the seed's
#    category, 3) uniform random across all products for whatever's left.
# Fractions intentionally don't need to sum to 1.0 -- leftover slots always
# fall through to uniform random.
HARD_NEGATIVE_BIAS_FRACTION = 0.35
CATEGORY_BIAS_FRACTION = 0.30
HARD_NEGATIVE_TOP_K = 15
REMINE_EVERY_N_EPOCHS = 2

# v3 item 12: distinct positive-label entries sampled per image per batch
# (was 1). Deduped against each other the same way single-sample already
# deduped against seen keys within one product.
TEXTS_PER_IMAGE = 3

# v3 item 13: "sigmoid" is the v2 per-pair SigLIP-native multi-positive
# loss; "infonce" is the SupCon-style softmax multi-positive alternative
# (multi_positive_infonce_loss). Flip and rerun to benchmark.
LOSS_TYPE = "sigmoid"
INFONCE_TEMPERATURE = 0.07

GRADIENT_ACCUMULATION_STEPS = 2
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0

STAGE1_EPOCHS = 8
STAGE1_LR = 1e-4

STAGE2_EPOCHS = 6
STAGE2_LR = 1e-5
NUM_UNFROZEN_BLOCKS = 4      # was 1 (v3 item 10)
LAYERWISE_LR_DECAY = 0.65    # per-block-distance-from-output LR decay

TEXT_BATCH_SIZE = 128
IMAGE_EVAL_BATCH_SIZE = 32
SIMILARITY_BATCH_SIZE = 512

NUM_WORKERS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

# "long" and "exact" both removed -- no flat caption, no SKU-suffixed
# label (see build_training_labels' module note). "model" absorbs what
# used to be the "exact" bucket's weight since it's now the most specific
# kind available. Weights renormalize automatically since random.choices
# doesn't require them to sum to 1.
LABEL_KIND_WEIGHTS = {
    "generic": 0.10,
    "attribute": 0.28,
    "brand": 0.20,
    "model": 0.42,
}

print("Device:", DEVICE)
print("Split mode:", SPLIT_MODE)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Text parsing (structured_caption only -- no flat `caption` anywhere)
# ============================================================

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
    """Note: SKU-bearing text is deliberately excluded from every label used
    here (training AND eval target). A bare SKU code has no semantic
    content a text encoder can learn from -- exact per-colorway identity
    matching is DINOv3's job (visual-identity encoder), not SigLIP2's
    (semantic encoder). SigLIP2's target here is capped at "model" level:
    brand + product name + descriptive attributes, without the
    product_code suffix that used to make every label artificially unique.
    """
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

    # Every independent attribute facet the schema supports (see
    # newLLMprompt.py) -- not just color. "closure"/"silhouette"/"length"
    # cover things like cropped-vs-full-length, low-top-vs-high-top,
    # zip-vs-button, that color alone never captures.
    for facet in ("color", "material", "fit", "pattern", "closure", "silhouette", "length"):
        for value in attributes.get(facet, []) or []:
            add(f"{value} {leaf_category}", "attribute")

    # Free-text construction/decorative details -- present on ~87% of
    # products (1075/1234) and, before this change, entirely unused for
    # training despite being the richest per-product visual-grounding
    # signal in the schema (e.g. "premium suede upper", "elasticized waist
    # with inner drawcords", "triple Gazelle branding").
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

    # Retrieval target, capped at model level -- deliberately NOT unique
    # per colorway/SKU (see module note above). Two colorways that read
    # identically at this level of description are expected to collide;
    # that's DINOv3's distinction to make, not SigLIP2's.
    exact_label = identity

    return {
        "entries": entries,
        "identity": identity,
        "exact_label": exact_label,
        "category": leaf_category,
        "colors": attributes.get("color", []) or [],
        "materials": attributes.get("material", []) or [],
    }


# ============================================================
# Load metadata
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
product_catalog = {}
missing_paths = []

for product in metadata:
    product_code = normalize_text(product.get("product_code", ""))
    if not product.get("structured_caption") or not product_code:
        continue

    label_data = build_training_labels(product)
    product_catalog[product_code] = label_data
    valid_label_keys = {entry["key"] for entry in label_data["entries"]}

    for raw_path in product.get("images", []):
        image_path = resolve_image_path(raw_path)
        if image_path is None:
            missing_paths.append(str(raw_path))
            continue
        try:
            with Image.open(image_path) as check_image:
                check_image.verify()
        except Exception:
            # Truncated/corrupted file (e.g. a partial download from the
            # Drive zip-export) -- treat the same as a missing path rather
            # than crashing training mid-epoch.
            missing_paths.append(str(image_path))
            continue
        records.append({
            "image_path": str(image_path),
            "product_code": product_code,
            "brand": product.get("brand", ""),
            "name": product.get("name", ""),
            "training_labels": label_data["entries"],
            "valid_label_keys": valid_label_keys,
            "identity": label_data["identity"],
            "exact_label": label_data["exact_label"],
            "category": label_data["category"],
        })

if not records:
    raise RuntimeError("No valid image records were found.")

all_exact_labels = list(dict.fromkeys(record["exact_label"] for record in records))

# Category of each product, for category-aware hard-negative batch sampling
# and category-scoped retrieval evaluation (see run notes at top of file).
category_by_product = {code: data["category"] for code, data in product_catalog.items()}
category_of_exact_label = {data["exact_label"]: data["category"] for data in product_catalog.values()}

print(f"Images: {len(records):,}")
print(f"Products: {len(product_catalog):,}")
print(f"Exact labels: {len(all_exact_labels):,}")
print(f"Missing paths: {len(missing_paths):,}")

serializable_catalog = {
    code: {
        "identity": data["identity"], "exact_label": data["exact_label"], "category": data["category"],
        "colors": data["colors"], "materials": data["materials"],
        "labels": [{"text": e["text"], "kind": e["kind"]} for e in data["entries"]],
    }
    for code, data in product_catalog.items()
}
with (OUTPUT_DIR / "generated_training_labels.json").open("w", encoding="utf-8") as f:
    json.dump(serializable_catalog, f, indent=2, ensure_ascii=False)


# ============================================================
# Train/validation/test split (view-based: every product seen in training)
# ============================================================

def make_view_split(image_records):
    grouped = defaultdict(list)
    for record in image_records:
        grouped[record["product_code"]].append(record)

    train, validation, test = [], [], []
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


train_records, val_records, test_records = make_view_split(records)

print("\nSplit sizes:")
print(f"Train images: {len(train_records):,}")
print(f"Validation images: {len(val_records):,}")
print(f"Test images: {len(test_records):,}")


# ============================================================
# Augmentation (train-side only; spec sec 5.4: preserve identity evidence,
# avoid hue shifts/grayscale/aggressive crops/flips)
# ============================================================

train_augment = T.Compose([
    T.RandomResizedCrop(384, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
    T.ColorJitter(brightness=0.15, contrast=0.15),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
])


# ============================================================
# Dataset and sampler
# ============================================================

class ApparelDataset(Dataset):
    def __init__(self, image_records, augment=False):
        self.records = image_records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image_path"]) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image = image.copy()
        if self.augment:
            image = train_augment(image)
        return {"image": image, "record": record}


class ProductBatchSampler(Sampler):
    """v3 (item 11): each batch fills its product slots in priority order --
    1) HARD_NEGATIVE_BIAS_FRACTION from the seed product's mined nearest
    neighbors (set_hard_negatives, refreshed periodically during training),
    2) CATEGORY_BIAS_FRACTION more from the seed's category, 3) uniform
    random across the full catalog for whatever's left (so cross-category
    structure and rare categories/products with no mined neighbors yet
    still get seen every epoch)."""

    def __init__(self, dataset, products_per_batch, images_per_product, seed,
                 category_by_product=None, category_bias_fraction=0.0,
                 hard_negative_bias_fraction=0.0):
        self.dataset = dataset
        self.products_per_batch = products_per_batch
        self.images_per_product = images_per_product
        self.seed = seed
        self.epoch = 0
        self.category_by_product = category_by_product or {}
        self.category_bias_fraction = category_bias_fraction
        self.hard_negative_bias_fraction = hard_negative_bias_fraction
        self.hard_negatives_by_product = {}
        self.indices_by_product = defaultdict(list)
        for index, record in enumerate(dataset.records):
            self.indices_by_product[record["product_code"]].append(index)
        self.product_codes = sorted(self.indices_by_product)

        self.products_by_category = defaultdict(list)
        for code in self.product_codes:
            category = self.category_by_product.get(code, "")
            self.products_by_category[category].append(code)
        self.categories = sorted(self.products_by_category)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_hard_negatives(self, hard_negatives_by_product):
        # Called periodically (REMINE_EVERY_N_EPOCHS) from run_stage with a
        # freshly-mined map from the current model snapshot -- see
        # mine_hard_negatives. Products absent from the map (e.g. singleton
        # categories) simply never get picked as a hard-negative seed and
        # fall through to the category/random tiers below.
        self.hard_negatives_by_product = hard_negatives_by_product or {}

    def __len__(self):
        return math.ceil(len(self.product_codes) / self.products_per_batch)

    def _select_batch_products(self, rng, already_used):
        selected = []
        seed_product = None

        if self.hard_negative_bias_fraction > 0 and self.hard_negatives_by_product:
            hard_negative_seeds = [c for c in self.product_codes if self.hard_negatives_by_product.get(c)]
            if hard_negative_seeds:
                seed_product = rng.choice(hard_negative_seeds)
                selected.append(seed_product)
                neighbor_pool = [
                    c for c in self.hard_negatives_by_product.get(seed_product, [])
                    if c != seed_product and c in self.indices_by_product
                ]
                rng.shuffle(neighbor_pool)
                num_hard_slots = min(
                    round(self.products_per_batch * self.hard_negative_bias_fraction),
                    len(neighbor_pool),
                )
                selected.extend(neighbor_pool[:num_hard_slots])

        if self.categories and self.category_bias_fraction > 0:
            seed_category = self.category_by_product.get(seed_product) if seed_product else None
            if not seed_category or seed_category not in self.products_by_category:
                seed_category = rng.choice(self.categories)
            category_pool = [c for c in self.products_by_category[seed_category] if c not in selected]
            rng.shuffle(category_pool)
            num_from_category = min(
                round(self.products_per_batch * self.category_bias_fraction),
                len(category_pool),
            )
            selected.extend(category_pool[:num_from_category])

        num_random_needed = self.products_per_batch - len(selected)
        random_pool = [c for c in self.product_codes if c not in selected]
        if len(random_pool) >= num_random_needed:
            selected.extend(rng.sample(random_pool, num_random_needed))
        else:
            selected.extend(rng.choices(random_pool or self.product_codes, k=num_random_needed))

        return selected[:self.products_per_batch]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        used_this_epoch = set()

        for _ in range(len(self)):
            selected = self._select_batch_products(rng, used_this_epoch)
            used_this_epoch.update(selected)

            batch_indices = []
            for product_code in selected:
                candidates = self.indices_by_product[product_code]
                if len(candidates) >= self.images_per_product:
                    chosen = rng.sample(candidates, self.images_per_product)
                else:
                    chosen = rng.choices(candidates, k=self.images_per_product)
                batch_indices.extend(chosen)

            rng.shuffle(batch_indices)
            yield batch_indices


processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)


def sample_training_label(entries):
    buckets = defaultdict(list)
    for entry in entries:
        buckets[entry["kind"]].append(entry)
    available_kinds = list(buckets.keys())
    weights = [LABEL_KIND_WEIGHTS.get(kind, 0.01) for kind in available_kinds]
    selected_kind = random.choices(available_kinds, weights=weights, k=1)[0]
    return random.choice(buckets[selected_kind])


def sample_training_labels(entries, count):
    """v3 item 12: draw up to `count` distinct entries instead of 1,
    weighted-by-kind each draw (without replacement by key) so a product
    with few entries doesn't get the same text duplicated in one batch."""
    remaining = list(entries)
    picked = []
    for _ in range(count):
        if not remaining:
            break
        entry = sample_training_label(remaining)
        picked.append(entry)
        remaining = [e for e in remaining if e["key"] != entry["key"]]
    return picked


def training_collator(batch):
    images = [item["image"] for item in batch]
    records_in_batch = [item["record"] for item in batch]

    sampled_texts, sampled_keys = [], []
    for record in records_in_batch:
        for entry in sample_training_labels(record["training_labels"], TEXTS_PER_IMAGE):
            sampled_texts.append(entry["text"])
            sampled_keys.append(entry["key"])

    processed = processor(
        images=images, text=sampled_texts, padding="max_length",
        truncation=True, max_length=64, return_tensors="pt",
    )

    num_images = len(records_in_batch)
    num_texts = len(sampled_texts)
    positive_mask = torch.zeros(num_images, num_texts, dtype=torch.bool)
    for image_index, record in enumerate(records_in_batch):
        valid_keys = record["valid_label_keys"]
        for text_index, text_key in enumerate(sampled_keys):
            positive_mask[image_index, text_index] = text_key in valid_keys

    processed["positive_mask"] = positive_mask
    return processed


train_dataset = ApparelDataset(train_records, augment=True)
train_batch_sampler = ProductBatchSampler(
    train_dataset, P_PRODUCTS, K_IMAGES_PER_PRODUCT, SEED,
    category_by_product=category_by_product, category_bias_fraction=CATEGORY_BIAS_FRACTION,
    hard_negative_bias_fraction=HARD_NEGATIVE_BIAS_FRACTION,
)
train_loader = DataLoader(
    train_dataset, batch_sampler=train_batch_sampler, collate_fn=training_collator,
    num_workers=NUM_WORKERS, pin_memory=DEVICE == "cuda",
)


# ============================================================
# Model setup
# ============================================================

def unfreeze_module(module):
    for parameter in module.parameters():
        parameter.requires_grad = True


def configure_trainable_parameters(model, mode, num_unfrozen_blocks=1):
    for parameter in model.parameters():
        parameter.requires_grad = False

    if mode in {"heads", "last_n_blocks"}:
        if hasattr(model.text_model, "head"):
            unfreeze_module(model.text_model.head)
        if hasattr(model.text_model, "final_layer_norm"):
            unfreeze_module(model.text_model.final_layer_norm)
        if hasattr(model.vision_model, "head"):
            unfreeze_module(model.vision_model.head)
        if hasattr(model.vision_model, "post_layernorm"):
            unfreeze_module(model.vision_model.post_layernorm)
        if hasattr(model, "logit_scale"):
            model.logit_scale.requires_grad_(True)
        if hasattr(model, "logit_bias"):
            model.logit_bias.requires_grad_(True)

    if mode == "last_n_blocks":
        # v3 item 10: was hardcoded to exactly the last block ("last_block").
        for layer in model.text_model.encoder.layers[-num_unfrozen_blocks:]:
            unfreeze_module(layer)
        for layer in model.vision_model.encoder.layers[-num_unfrozen_blocks:]:
            unfreeze_module(layer)
    elif mode != "heads":
        raise ValueError("mode must be 'heads' or 'last_n_blocks'.")


def build_optimizer_param_groups(model, mode, base_lr, num_unfrozen_blocks=1, layerwise_decay=1.0):
    """v3 item 10: heads/logit-scale params train at base_lr; when mode is
    last_n_blocks, each unfrozen transformer block gets its own LR group,
    decayed by layerwise_decay^distance-from-output so the block right
    next to the output head trains at base_lr and blocks further back
    (still carrying a lot of general pretrained representation) move more
    conservatively -- standard discriminative fine-tuning practice for
    unfreezing more than one block at once."""
    non_block_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and "encoder.layers" not in name
    ]
    param_groups = []
    if non_block_params:
        param_groups.append({"params": non_block_params, "lr": base_lr})

    if mode == "last_n_blocks":
        for encoder in (model.text_model.encoder, model.vision_model.encoder):
            num_layers = len(encoder.layers)
            for layer_index in range(max(0, num_layers - num_unfrozen_blocks), num_layers):
                distance_from_output = (num_layers - 1) - layer_index
                layer_lr = base_lr * (layerwise_decay ** distance_from_output)
                layer_params = [p for p in encoder.layers[layer_index].parameters() if p.requires_grad]
                if layer_params:
                    param_groups.append({"params": layer_params, "lr": layer_lr})

    return param_groups


# ============================================================
# Multi-positive loss
# ============================================================

def directional_multi_positive_loss(logits, positive_mask):
    positive_mask = positive_mask.bool()
    negative_mask = ~positive_mask
    positive_counts = positive_mask.sum(dim=1).clamp_min(1)
    negative_counts = negative_mask.sum(dim=1).clamp_min(1)
    positive_losses = (-F.logsigmoid(logits) * positive_mask).sum(dim=1) / positive_counts
    negative_losses = (-F.logsigmoid(-logits) * negative_mask).sum(dim=1) / negative_counts
    return (0.5 * (positive_losses + negative_losses)).mean()


def hierarchical_multi_positive_loss(logits_per_image, positive_mask):
    image_to_text = directional_multi_positive_loss(logits_per_image, positive_mask)
    text_to_image = directional_multi_positive_loss(logits_per_image.T, positive_mask.T)
    return 0.5 * (image_to_text + text_to_image)


def directional_multi_positive_infonce(similarity, positive_mask, temperature):
    """v3 item 13: softmax (SupCon-style) multi-positive alternative to the
    sigmoid loss above -- for each anchor row, -log(sum(exp(sim/T)) over
    positives / sum(exp(sim/T)) over everything). Anchors with zero
    positives in this direction (can happen text->image when TEXTS_PER_IMAGE
    duplicates collapse) are excluded from the mean rather than propagating
    a -inf/nan loss term."""
    positive_mask = positive_mask.bool()
    logits = similarity / temperature
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_counts = positive_mask.sum(dim=1)
    safe_counts = positive_counts.clamp_min(1)
    per_anchor_loss = -(log_prob * positive_mask).sum(dim=1) / safe_counts
    has_positive = positive_counts > 0
    if has_positive.sum() == 0:
        return per_anchor_loss.sum() * 0.0
    return per_anchor_loss[has_positive].mean()


def hierarchical_multi_positive_infonce_loss(logits_per_image, positive_mask, temperature=INFONCE_TEMPERATURE):
    image_to_text = directional_multi_positive_infonce(logits_per_image, positive_mask, temperature)
    text_to_image = directional_multi_positive_infonce(logits_per_image.T, positive_mask.T, temperature)
    return 0.5 * (image_to_text + text_to_image)


def compute_training_loss(logits_per_image, positive_mask):
    if LOSS_TYPE == "infonce":
        return hierarchical_multi_positive_infonce_loss(logits_per_image, positive_mask)
    if LOSS_TYPE == "sigmoid":
        return hierarchical_multi_positive_loss(logits_per_image, positive_mask)
    raise ValueError("LOSS_TYPE must be 'sigmoid' or 'infonce'.")


# ============================================================
# Evaluation helpers
# ============================================================

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
def encode_texts(model, texts):
    model.eval()
    embeddings = []
    for start in tqdm(range(0, len(texts), TEXT_BATCH_SIZE), desc="Encoding texts", leave=False):
        batch_texts = texts[start:start + TEXT_BATCH_SIZE]
        inputs = processor(text=batch_texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        inputs = move_to_device(inputs)
        with autocast_context():
            outputs = model.get_text_features(**inputs)
            batch_embeddings = extract_embeddings(outputs)
        embeddings.append(F.normalize(batch_embeddings.float(), p=2, dim=-1).cpu())
    return torch.cat(embeddings, dim=0)


@torch.inference_mode()
def encode_images(model, image_records):
    model.eval()
    embeddings, valid_records = [], []
    for start in tqdm(range(0, len(image_records), IMAGE_EVAL_BATCH_SIZE), desc="Encoding images", leave=False):
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
        del inputs, outputs, batch_embeddings
    return torch.cat(embeddings, dim=0), valid_records


# ============================================================
# Offline hard-negative mining (v3 item 11)
# ============================================================

# Up to 2 representative training images per product -- enough signal to
# place a product in embedding space for mining without re-encoding the
# whole train split every remine.
PRODUCT_REPRESENTATIVE_RECORDS = defaultdict(list)
for _record in train_records:
    if len(PRODUCT_REPRESENTATIVE_RECORDS[_record["product_code"]]) < 2:
        PRODUCT_REPRESENTATIVE_RECORDS[_record["product_code"]].append(_record)


@torch.inference_mode()
def mine_hard_negatives(model, top_k=HARD_NEGATIVE_TOP_K):
    """Embed each product's representative image(s) with the *current*
    model snapshot, then for each product return its top_k nearest other
    products by cosine similarity, scoped within the same category (a
    same-category neighbor is the actually-confusable case; see the v2
    CATEGORY_BIAS_FRACTION rationale -- mining just sharpens "same
    category" down to "same category AND currently embeds close")."""
    flat_records = [r for records in PRODUCT_REPRESENTATIVE_RECORDS.values() for r in records]
    embeddings, valid_records = encode_images(model, flat_records)

    embeddings_by_product = defaultdict(list)
    for embedding, record in zip(embeddings, valid_records):
        embeddings_by_product[record["product_code"]].append(embedding)

    product_codes = sorted(embeddings_by_product)
    if len(product_codes) < 2:
        return {}
    product_embeddings = F.normalize(
        torch.stack([torch.stack(embeddings_by_product[code]).mean(dim=0) for code in product_codes]), dim=-1,
    )

    by_category = defaultdict(list)
    for position, code in enumerate(product_codes):
        by_category[category_by_product.get(code, "")].append(position)

    hard_negatives = {}
    for positions in by_category.values():
        if len(positions) < 2:
            continue
        index_tensor = torch.tensor(positions, dtype=torch.long)
        sub_embeddings = product_embeddings[index_tensor]
        similarity = sub_embeddings @ sub_embeddings.T
        similarity.fill_diagonal_(-1.0)
        k = min(top_k, len(positions) - 1)
        top_local_indices = similarity.topk(k, dim=1).indices
        for local_row, global_position in enumerate(positions):
            neighbors = [product_codes[positions[local_col]] for local_col in top_local_indices[local_row].tolist()]
            hard_negatives[product_codes[global_position]] = neighbors

    return hard_negatives


@torch.inference_mode()
def evaluate_exact_retrieval(model, evaluation_records, candidate_labels):
    candidate_labels = list(dict.fromkeys(candidate_labels))
    candidate_to_index = {label: i for i, label in enumerate(candidate_labels)}

    text_embeddings = encode_texts(model, candidate_labels)
    image_embeddings, valid_records = encode_images(model, evaluation_records)
    target_indices = torch.tensor([candidate_to_index[r["exact_label"]] for r in valid_records])

    ranks = []
    for start in range(0, len(valid_records), SIMILARITY_BATCH_SIZE):
        end = min(start + SIMILARITY_BATCH_SIZE, len(valid_records))
        similarities = image_embeddings[start:end] @ text_embeddings.T
        targets = target_indices[start:end]
        target_scores = similarities[torch.arange(end - start), targets]
        batch_ranks = (similarities > target_scores[:, None]).sum(dim=1) + 1
        ranks.append(batch_ranks)
    ranks = torch.cat(ranks).float()

    return {
        "num_images": len(valid_records),
        "num_candidates": len(candidate_labels),
        "recall_at_1": float((ranks <= 1).float().mean()),
        "recall_at_5": float((ranks <= 5).float().mean()),
        "recall_at_10": float((ranks <= 10).float().mean()),
        "mrr": float((1.0 / ranks).mean()),
        "median_rank": float(ranks.median()),
        "mean_rank": float(ranks.mean()),
    }


def evaluate_exact_retrieval_category_scoped(model, evaluation_records, candidate_labels, category_of_exact_label):
    """Same as evaluate_exact_retrieval, but each image is only ranked
    against same-category candidates (reuses one encoding pass over the
    full candidate/image sets, just scores per category subset)."""

    candidate_labels = list(dict.fromkeys(candidate_labels))
    candidate_to_index = {label: i for i, label in enumerate(candidate_labels)}

    text_embeddings = encode_texts(model, candidate_labels)
    image_embeddings, valid_records = encode_images(model, evaluation_records)

    indices_by_category = defaultdict(list)
    for label, idx in candidate_to_index.items():
        indices_by_category[category_of_exact_label.get(label, "")].append(idx)

    records_by_category = defaultdict(list)
    for i, record in enumerate(valid_records):
        records_by_category[record["category"]].append(i)

    all_ranks = []
    scored_records = 0
    skipped_records = 0

    for category, image_positions in records_by_category.items():
        cat_candidate_indices = indices_by_category.get(category, [])
        if len(cat_candidate_indices) < 2:
            skipped_records += len(image_positions)
            continue

        cat_index_tensor = torch.tensor(cat_candidate_indices, dtype=torch.long)
        cat_text_embeddings = text_embeddings[cat_index_tensor]
        local_index_of_global = {g: l for l, g in enumerate(cat_candidate_indices)}

        valid_image_positions = []
        target_local_list = []
        for p in image_positions:
            global_idx = candidate_to_index.get(valid_records[p]["exact_label"])
            local_idx = local_index_of_global.get(global_idx) if global_idx is not None else None
            if local_idx is None:
                # exact_label's recorded category (from category_of_exact_label,
                # which can only keep one category per label) disagrees with
                # this image's own category -- a possible collision now that
                # exact_label no longer includes the SKU suffix. Skip rather
                # than crash.
                skipped_records += 1
                continue
            valid_image_positions.append(p)
            target_local_list.append(local_idx)

        if not valid_image_positions:
            continue

        image_position_tensor = torch.tensor(valid_image_positions, dtype=torch.long)
        cat_image_embeddings = image_embeddings[image_position_tensor]
        target_local_indices = torch.tensor(target_local_list, dtype=torch.long)

        similarities = cat_image_embeddings @ cat_text_embeddings.T
        target_scores = similarities[torch.arange(len(valid_image_positions)), target_local_indices]
        ranks = (similarities > target_scores[:, None]).sum(dim=1) + 1
        all_ranks.append(ranks.float())
        scored_records += len(valid_image_positions)

    if not all_ranks:
        return {
            "num_images": 0, "num_skipped_singleton_category": skipped_records,
            "recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0,
            "mrr": 0.0, "median_rank": float("nan"), "mean_rank": float("nan"),
        }

    ranks = torch.cat(all_ranks)
    return {
        "num_images": scored_records,
        "num_skipped_singleton_category": skipped_records,
        "recall_at_1": float((ranks <= 1).float().mean()),
        "recall_at_5": float((ranks <= 5).float().mean()),
        "recall_at_10": float((ranks <= 10).float().mean()),
        "mrr": float((1.0 / ranks).mean()),
        "median_rank": float(ranks.median()),
        "mean_rank": float(ranks.mean()),
    }


def print_metrics(title, metrics):
    print(f"\n{title}")
    print(f"R@1:  {metrics['recall_at_1'] * 100:.2f}%")
    print(f"R@5:  {metrics['recall_at_5'] * 100:.2f}%")
    print(f"R@10: {metrics['recall_at_10'] * 100:.2f}%")
    print(f"MRR:  {metrics['mrr'] * 100:.2f}%")
    print(f"Median rank: {metrics['median_rank']:.1f}")
    print(f"Mean rank: {metrics['mean_rank']:.2f}")


# ============================================================
# One training stage (shared by stage 1 and stage 2)
# ============================================================

RESUME_STATE_PATH = OUTPUT_DIR / "resume_state.json"


def atomic_write_json(path, data):
    # Write-then-rename so an interrupt (Colab disconnect, Modal preemption)
    # mid-write can never leave a truncated/corrupt file behind -- a plain
    # write_text() left half-written JSON that then crashed every subsequent
    # resume attempt identically, forever.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def load_json_safely(path, default):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Truncated by an earlier interrupt -- fall back rather than crash
        # forever on the same unreadable file.
        print(f"WARNING: could not parse {path}, ignoring and using default")
        return default


def load_resume_state():
    return load_json_safely(RESUME_STATE_PATH, {
        "stage1_complete": False, "stage1_epochs_done": 0, "stage1_best_score": -1.0,
        "stage2_complete": False, "stage2_epochs_done": 0, "stage2_best_score": -1.0,
    })


def save_resume_state(state):
    atomic_write_json(RESUME_STATE_PATH, state)


def save_model_checkpoint(model, processor, save_dir):
    # Save to a fresh temp dir and only swap it into place once fully
    # written, so an interrupt mid-save can't corrupt the last-known-good
    # checkpoint that resume logic depends on.
    tmp_dir = save_dir.with_name(save_dir.name + "_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp_dir, safe_serialization=True)
    processor.save_pretrained(tmp_dir)
    if save_dir.exists():
        shutil.rmtree(save_dir)
    tmp_dir.rename(save_dir)


def run_stage(model, mode, epochs, lr, save_dir, stage_name, num_unfrozen_blocks=1,
              start_epoch=1, history=None, best_score=-1.0, resume_state=None, resume_state_prefix=None):
    configure_trainable_parameters(model, mode, num_unfrozen_blocks=num_unfrozen_blocks)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_parameters)
    total_count = sum(p.numel() for p in model.parameters())
    if trainable_count == 0:
        raise RuntimeError("No model parameters were unfrozen.")
    print(f"\n[{stage_name}] Trainable parameters: {trainable_count:,} / {total_count:,} "
          f"({100 * trainable_count / total_count:.3f}%)")

    if mode != "heads":
        # use_reentrant=False, not the default -- with the earlier blocks
        # frozen and only the last num_unfrozen_blocks trainable, the first
        # trainable block's *input* tensor (coming out of a frozen block)
        # doesn't itself require grad. Reentrant checkpointing needs a
        # requires_grad input to build its backward graph and silently
        # fails to backprop into that block's params without one; the
        # non-reentrant (autograd-hook-based) implementation doesn't have
        # this restriction. Same failure mode already hit and documented
        # in dino_identity_finetune.py's last_block stage -- that script
        # worked around it by disabling checkpointing entirely since only
        # one block was ever trainable there, but with 4 blocks unfrozen
        # here the memory savings are worth keeping, so fix it properly
        # instead.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    remaining_epochs = epochs - (start_epoch - 1)
    param_groups = build_optimizer_param_groups(
        model, mode, lr, num_unfrozen_blocks=num_unfrozen_blocks, layerwise_decay=LAYERWISE_LR_DECAY,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=WEIGHT_DECAY)
    updates_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
    total_training_steps = updates_per_epoch * remaining_epochs
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    history = list(history) if history else []

    # v3 item 11: mine once before the stage's first epoch (skipped on
    # resume mid-stage -- the loaded checkpoint has already been mining off
    # a recent snapshot from before the interrupt) and then every
    # REMINE_EVERY_N_EPOCHS during training, using whatever the model looks
    # like at that point.
    if start_epoch == 1:
        print(f"[{stage_name}] Mining hard negatives from current model snapshot...")
        train_batch_sampler.set_hard_negatives(mine_hard_negatives(model))

    for epoch in range(start_epoch, epochs + 1):
        if epoch > start_epoch and (epoch - 1) % REMINE_EVERY_N_EPOCHS == 0:
            print(f"[{stage_name}] Re-mining hard negatives at epoch {epoch}...")
            train_batch_sampler.set_hard_negatives(mine_hard_negatives(model))

        train_batch_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        batches_seen = 0

        progress = tqdm(train_loader, desc=f"[{stage_name}] Epoch {epoch}/{epochs}")
        for batch_index, batch in enumerate(progress):
            positive_mask = batch.pop("positive_mask").to(DEVICE)
            inputs = move_to_device(batch)

            with autocast_context():
                outputs = model(**inputs)
                loss = compute_training_loss(outputs.logits_per_image, positive_mask)
                scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(scaled_loss).backward()

            should_update = (batch_index + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or batch_index + 1 == len(train_loader)
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += float(loss.detach())
            batches_seen += 1
            progress.set_postfix({"loss": running_loss / batches_seen, "lr": scheduler.get_last_lr()[0]})

        average_train_loss = running_loss / max(batches_seen, 1)
        validation_exact = evaluate_exact_retrieval(model, val_records, all_exact_labels)
        print_metrics(f"[{stage_name}] Epoch {epoch}: exact-label validation (global candidate pool)", validation_exact)

        validation_exact_scoped = evaluate_exact_retrieval_category_scoped(model, val_records, all_exact_labels, category_of_exact_label)
        print_metrics(f"[{stage_name}] Epoch {epoch}: exact-label validation (category-scoped candidate pool)", validation_exact_scoped)

        selection_score = validation_exact["recall_at_1"]
        history.append({
            "epoch": epoch, "train_loss": average_train_loss, "selection_score": selection_score,
            "exact_metrics": validation_exact, "exact_metrics_category_scoped": validation_exact_scoped,
        })
        atomic_write_json(OUTPUT_DIR / f"{stage_name}_training_history.json", history)

        if selection_score > best_score:
            best_score = selection_score
            save_dir.mkdir(parents=True, exist_ok=True)
            save_model_checkpoint(model, processor, save_dir)
            atomic_write_json(save_dir / "validation_metrics.json", {
                "selection_score": selection_score, "exact_metrics": validation_exact,
                "exact_metrics_category_scoped": validation_exact_scoped,
            })
            print(f"[{stage_name}] Saved new best model:", save_dir)

        if resume_state is not None and resume_state_prefix is not None:
            resume_state[f"{resume_state_prefix}_epochs_done"] = epoch
            resume_state[f"{resume_state_prefix}_best_score"] = best_score
            save_resume_state(resume_state)

    if resume_state is not None and resume_state_prefix is not None:
        resume_state[f"{resume_state_prefix}_complete"] = True
        save_resume_state(resume_state)

    return best_score


# ============================================================
# Baseline (untrained) evaluation -- skipped on resume, since Modal
# preemption restarts this whole process with the same input and we don't
# want to burn GPU minutes re-computing it every time.
# ============================================================

resume_state = load_resume_state()
resuming = resume_state["stage1_epochs_done"] > 0 or resume_state["stage1_complete"]

if resuming:
    print(f"\nResuming from saved state: {resume_state}")
    cached_baseline = load_json_safely(OUTPUT_DIR / "baseline_metrics.json", None)
    if cached_baseline is not None:
        baseline_exact = cached_baseline["baseline_exact"]
        baseline_exact_scoped = cached_baseline["baseline_exact_scoped"]
    else:
        baseline_exact = {"recall_at_1": float("nan")}
        baseline_exact_scoped = {"recall_at_1": float("nan")}
    load_from = STAGE1_DIR if (STAGE1_DIR / "model.safetensors").is_file() else BASE_MODEL_ID
    model = AutoModel.from_pretrained(load_from, torch_dtype=torch.float32).to(DEVICE)
else:
    model = AutoModel.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32).to(DEVICE)
    print("Model loaded. Starting baseline eval (global candidate pool)...", flush=True)

    baseline_exact = evaluate_exact_retrieval(model, val_records, all_exact_labels)
    print_metrics("Validation exact-label R@1 before any training (global candidate pool)", baseline_exact)

    print("Starting baseline eval (category-scoped candidate pool)...", flush=True)
    baseline_exact_scoped = evaluate_exact_retrieval_category_scoped(model, val_records, all_exact_labels, category_of_exact_label)
    print_metrics("Validation exact-label R@1 before any training (category-scoped candidate pool)", baseline_exact_scoped)

    atomic_write_json(OUTPUT_DIR / "baseline_metrics.json", {"baseline_exact": baseline_exact, "baseline_exact_scoped": baseline_exact_scoped})


# ============================================================
# Stage 1: heads only
# ============================================================

if resume_state["stage1_complete"]:
    print("\n[stage1_heads] Already complete per resume state, skipping.")
    stage1_best = resume_state["stage1_best_score"]
else:
    stage1_history = load_json_safely(OUTPUT_DIR / "stage1_heads_training_history.json", [])
    stage1_best = run_stage(
        model, "heads", STAGE1_EPOCHS, STAGE1_LR, STAGE1_DIR, "stage1_heads",
        start_epoch=resume_state["stage1_epochs_done"] + 1, history=stage1_history,
        best_score=resume_state["stage1_best_score"], resume_state=resume_state, resume_state_prefix="stage1",
    )


# ============================================================
# Stage 2: continue from stage 1's best checkpoint, unfreeze last block
# ============================================================

del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

stage2_load_from = STAGE2_DIR if (STAGE2_DIR / "model.safetensors").is_file() else STAGE1_DIR
model = AutoModel.from_pretrained(stage2_load_from, torch_dtype=torch.float32).to(DEVICE)

if resume_state["stage2_complete"]:
    print("\n[stage2_lastnblocks] Already complete per resume state, skipping.")
    stage2_best = resume_state["stage2_best_score"]
else:
    stage2_history = load_json_safely(OUTPUT_DIR / "stage2_lastnblocks_training_history.json", [])
    stage2_best = run_stage(
        model, "last_n_blocks", STAGE2_EPOCHS, STAGE2_LR, STAGE2_DIR, "stage2_lastnblocks",
        num_unfrozen_blocks=NUM_UNFROZEN_BLOCKS,
        start_epoch=resume_state["stage2_epochs_done"] + 1, history=stage2_history,
        best_score=resume_state["stage2_best_score"], resume_state=resume_state, resume_state_prefix="stage2",
    )


# ============================================================
# Final held-out evaluation -- whichever stage scored higher on validation
# ============================================================

del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

final_dir = STAGE2_DIR if stage2_best >= stage1_best else STAGE1_DIR
print(f"\nBest stage: {'stage2_lastnblocks' if stage2_best >= stage1_best else 'stage1_heads'} "
      f"(val R@1 stage1={stage1_best * 100:.2f}%, stage2={stage2_best * 100:.2f}%)")

best_model = AutoModel.from_pretrained(final_dir, torch_dtype=torch.float32).to(DEVICE)
best_model.eval()

test_exact = evaluate_exact_retrieval(best_model, test_records, all_exact_labels)
print_metrics("Final test: exact-label retrieval (global candidate pool)", test_exact)

test_exact_scoped = evaluate_exact_retrieval_category_scoped(best_model, test_records, all_exact_labels, category_of_exact_label)
print_metrics("Final test: exact-label retrieval (category-scoped candidate pool)", test_exact_scoped)

atomic_write_json(OUTPUT_DIR / "test_metrics.json", {
    "final_checkpoint": str(final_dir),
    "baseline_val_recall_at_1": baseline_exact["recall_at_1"],
    "baseline_val_recall_at_1_category_scoped": baseline_exact_scoped["recall_at_1"],
    "stage1_best_val_recall_at_1": stage1_best, "stage2_best_val_recall_at_1": stage2_best,
    "test_exact_metrics": test_exact,
    "test_exact_metrics_category_scoped": test_exact_scoped,
})

print("\nBest checkpoint:", final_dir)
