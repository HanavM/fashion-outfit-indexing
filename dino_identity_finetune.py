"""DINOv3 visual-identity adaptation, per docs/project_spec_v1.md section 5.2.

Role split (section 2.2): SigLIP2 (finetune_siglip2.py) now targets semantic/
"model-level" description only -- SKU-bearing exact identity was
deliberately stripped from its labels. This script is the other half:
DINOv3 as the exact-SKU / exact-colorway visual-identity encoder, trained
via a lightweight trainable projection head on top of the (initially
frozen) backbone.

Implements, in order, everything section 5.2 specifies:

1. Positive pairs = different images of the *same product_code* (exact
   SKU/colorway) -- not the model-level "identity" text SigLIP now uses.
2. Hard negatives, prioritized via batch construction (see
   IdentityBatchSampler below): same model/different colorway first (the
   single hardest and most diagnostic negative for an identity encoder),
   then same-category/same-brand, then uniform random fill.
3. Identity-balanced P x K batches (P identities x K images/identity,
   K >= 2 so every identity has genuine in-batch positives).
4. Two benchmarkable losses (LOSS_TYPE below): supervised contrastive
   (SupCon, default -- natural fit for the P x K batch shape, no extra
   classification head to maintain) and ArcFace-style additive angular
   margin (alternative, benchmarked against SupCon before committing).
5. Fine-tuning order: Stage A freezes the backbone and trains only the
   projection head; evaluate; Stage B unfreezes the last transformer
   block(s) at a much lower LR; evaluate; Stage B's checkpoint is only
   used as the final model if it actually beats Stage A on held-out
   retrieval (see the stage2_best >= stage1_best selection at the bottom,
   same pattern as finetune_siglip2.py).
6. Data augmentation (section 5.4): modest resize/crop, mild
   brightness/contrast, blur, simulated JPEG compression, small random
   erasing (occlusion), mild perspective. No hue shift, no grayscale, no
   aggressive crop, no horizontal flip (many shoes/apparel have
   side-specific detail -- logos, lace direction, asymmetric closures).

Eval metrics (section 8, "Same-model/different-colorway error rate"):
alongside standard exact-SKU R@1/R@5/R@10/MRR, every wrong top-1 is
classified as either a same-model-different-colorway confusion (visually
reasonable mistake -- DINO's hardest case by design) or an unrelated-item
confusion (a real failure), reported as their own rates.

Same run-safety infrastructure as finetune_siglip2.py, ported over after
the earlier Modal preemption/corruption incidents:
- corrupted-image skip (Image.verify() at record-build time)
- atomic (temp-file + os.replace) writes for all JSON state
- atomic (temp-dir + rename) writes for model checkpoints
- self-healing loads (a corrupt/truncated file is skipped, not fatal)
- full epoch-level resume via resume_state.json, so an interrupted run
  can be restarted with the exact same command instead of losing progress

Colab usage: mount Drive, `pip install torch torchvision transformers
pillow tqdm numpy accelerate safetensors`, accept the DINOv3 license at
huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m once, then either
export HF_TOKEN or run huggingface_hub.login() once before this script.
Then: `python3 dino_identity_finetune.py`.
"""

from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext
from io import BytesIO

import gc
import json
import math
import os
import random
import re
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel, get_cosine_schedule_with_warmup

if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    print("Logged into Hugging Face Hub via HF_TOKEN.")


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = Path("/content/drive/MyDrive/apparel_dataset")
METADATA_PATH = DATASET_ROOT / "metadata.json"

MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"

LOSS_TYPE = "supcon"          # "supcon" (default) or "arcface" -- see module docstring point 4
SUPCON_TEMPERATURE = 0.07
ARCFACE_MARGIN = 0.30
ARCFACE_SCALE = 30.0

# Namespaced by LOSS_TYPE: the spec explicitly calls for benchmarking
# supcon vs arcface by rerunning this script with LOSS_TYPE flipped, but
# resume_state.json/checkpoints are loss-agnostic paths -- without this,
# a second run with a different LOSS_TYPE would see stage1_complete/
# stage2_complete already True from the first run, skip training
# entirely, and silently report the FIRST loss type's metrics mislabeled
# under the second one.
OUTPUT_DIR = DATASET_ROOT / f"finetuned_dinov3_identity_v1_{LOSS_TYPE}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE1_DIR = OUTPUT_DIR / "stage1_head_best"
STAGE2_DIR = OUTPUT_DIR / "stage2_lastblock_best"

PROJECTION_DIM = 256

# P x K identity-balanced batch (section 5.2 "Batch construction")
IDENTITIES_PER_BATCH = 8      # P
IMAGES_PER_IDENTITY = 4       # K (>= 2 required so every identity has a genuine in-batch positive)
BATCH_SIZE = IDENTITIES_PER_BATCH * IMAGES_PER_IDENTITY

# Hard-negative batch-composition bias (section 5.2 "Hard negatives"):
# fraction of P drawn from colorway siblings of a seed model_identity
# (hardest case: same model, different colorway), then a further fraction
# from the seed's category (same-category/-brand/-silhouette negatives),
# remainder uniform at random across the catalog.
COLORWAY_SIBLING_BIAS_FRACTION = 0.375   # ~3/8 of P when siblings exist
CATEGORY_BIAS_FRACTION = 0.375           # further ~3/8 of P

STAGE1_EPOCHS = 10
STAGE1_LR = 1e-3
STAGE2_EPOCHS = 6
STAGE2_LR = 2e-6              # "unfreeze only final blocks with a lower learning rate"
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
MAX_GRAD_NORM = 1.0
GRADIENT_ACCUMULATION_STEPS = 1

VAL_IMAGES_PER_PRODUCT = 1
TEST_IMAGES_PER_PRODUCT = 1
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
AMP_DTYPE = torch.float16
NUM_WORKERS = 4 if DEVICE == "cuda" else 0

print("Device:", DEVICE)
print("Model:", MODEL_ID)
print("Loss:", LOSS_TYPE)
print("Output directory:", OUTPUT_DIR)


def autocast_context():
    if USE_AMP:
        return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)
    return nullcontext()


# ============================================================
# Run-safety helpers (atomic writes, self-healing loads) -- ported from
# finetune_siglip2.py after the Modal preemption/corrupt-checkpoint
# incidents. A plain write_text()/save_pretrained() left a truncated file
# behind if interrupted mid-write, which then crashed every subsequent
# resume attempt identically, forever. All four write sites here go
# through these instead.
# ============================================================

def atomic_write_json(path, data):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def load_json_safely(path, default):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"WARNING: could not parse {path}, ignoring and using default")
        return default


def save_checkpoint(backbone, projection_head, save_dir, extra_state=None):
    tmp_dir = save_dir.with_name(save_dir.name + "_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    backbone.save_pretrained(tmp_dir / "backbone", safe_serialization=True)
    torch.save(projection_head.state_dict(), tmp_dir / "projection_head.pt")
    if extra_state is not None:
        torch.save(extra_state, tmp_dir / "extra_state.pt")
    if save_dir.exists():
        shutil.rmtree(save_dir)
    tmp_dir.rename(save_dir)


def load_checkpoint_backbone_path(save_dir):
    backbone_dir = save_dir / "backbone"
    return backbone_dir if (backbone_dir / "model.safetensors").is_file() else None


# ============================================================
# Load metadata, resolve + validate image paths (corrupted-file skip)
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


def model_identity_and_category(product):
    """Model-level identity text (colorway-collapsed) and leaf category,
    used only to bias hard-negative batch composition -- the same
    "identity" field finetune_siglip2.py trains SigLIP2 toward. NOT used
    as the contrastive class; product_code (exact SKU/colorway) is."""
    structured = product.get("structured_caption") or {}
    positive_texts = [normalize_text(t) for t in structured.get("positive_texts", []) or [] if normalize_text(t)]
    positive_texts = [t for t in positive_texts if not SKU_TEXT_PATTERN.search(t)]
    taxonomy_path = [normalize_text(t) for t in structured.get("taxonomy_path", []) or [] if normalize_text(t)]
    leaf_category = taxonomy_path[-1] if taxonomy_path else "apparel item"
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


records = []
missing_paths = []

for product in metadata:
    product_code = normalize_text(product.get("product_code", ""))
    if not product_code:
        continue
    model_identity, category, brand = model_identity_and_category(product)

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
            "model_identity": model_identity,
            "category": category,
            "brand": brand,
        })

if not records:
    raise RuntimeError("No valid image records were found.")

print(f"Images: {len(records):,}")
print(f"Products (exact-identity classes): {len(set(r['product_code'] for r in records)):,}")
print(f"Missing/corrupted paths: {len(missing_paths):,}")


# ============================================================
# Colorway-sibling map: model_identity -> set of product_codes sharing it.
# This is exactly the "same model, different colorway" hard-negative
# relationship section 5.2 asks to prioritize.
# ============================================================

product_codes_by_model_identity = defaultdict(set)
category_of_product = {}
model_identity_of_product = {}
for r in records:
    product_codes_by_model_identity[r["model_identity"]].add(r["product_code"])
    category_of_product[r["product_code"]] = r["category"]
    model_identity_of_product[r["product_code"]] = r["model_identity"]

products_with_colorway_siblings = sum(1 for v in product_codes_by_model_identity.values() if len(v) >= 2)
print(f"Model-identity groups with >=2 colorway variants: {products_with_colorway_siblings:,} "
      f"(of {len(product_codes_by_model_identity):,} groups)")


# ============================================================
# Train/validation/test split -- view-based, every product seen in
# training (same convention as finetune_siglip2.py / run_dinov3_baseline.py)
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
# Data augmentation -- section 5.4: preserve identity-critical evidence.
# Modest crop, mild brightness/contrast, blur, simulated JPEG compression,
# small random erasing (occlusion), mild perspective. No hue shift, no
# grayscale, no aggressive crop, no horizontal flip.
# ============================================================

class SimulateJPEGCompression:
    def __init__(self, quality_range=(40, 85), p=0.3):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image
        quality = random.randint(*self.quality_range)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


IMAGE_SIZE = 224  # DINOv3 vitb16 native training resolution

train_augment = T.Compose([
    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.11)),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
    SimulateJPEGCompression(quality_range=(40, 85), p=0.25),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.2),
    T.RandomPerspective(distortion_scale=0.12, p=0.15),
    # RandomErasing only operates on tensors (it indexes img.shape), but
    # everything else in this pipeline stays PIL because embed_batch's
    # processor(images=pil_images, ...) call expects PIL input -- bracket
    # just this one op with PILToTensor/ToPILImage rather than converting
    # the whole pipeline (uint8 round-trip, no float rescale side effects).
    T.PILToTensor(),
    T.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
    T.ToPILImage(),
])

eval_resize = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE))])


# ============================================================
# Dataset + identity-balanced P x K batch sampler
# ============================================================

class IdentityDataset(Dataset):
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
        image = (train_augment if self.augment else eval_resize)(image)
        return {"image": image, "record": record}


class IdentityBatchSampler(Sampler):
    """P identities x K images/identity per batch, with hard-negative-
    biased composition (section 5.2): prioritize colorway siblings of a
    seed identity, then same-category fill, then uniform random."""

    def __init__(self, dataset, identities_per_batch, images_per_identity, seed):
        self.dataset = dataset
        self.P = identities_per_batch
        self.K = images_per_identity
        self.seed = seed
        self.epoch = 0

        self.indices_by_product = defaultdict(list)
        for index, record in enumerate(dataset.records):
            self.indices_by_product[record["product_code"]].append(index)
        # Only products with >= 2 images can serve as anchors -- K >= 2
        # requires genuine in-batch positives, not augmented duplicates
        # of a single image standing in for "different views".
        self.eligible_products = [p for p, idxs in self.indices_by_product.items() if len(idxs) >= 2]
        self.products_by_category = defaultdict(list)
        for p in self.eligible_products:
            self.products_by_category[category_of_product.get(p, "")].append(p)

        if len(self.eligible_products) < identities_per_batch:
            raise RuntimeError(
                f"Only {len(self.eligible_products)} products have >=2 images; "
                f"need at least {identities_per_batch} for one P x K batch."
            )

        images_per_epoch = len(self.eligible_products) * self.K
        self.batches_per_epoch = max(1, images_per_epoch // (self.P * self.K))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.batches_per_epoch

    def _select_products(self, rng):
        seed_product = rng.choice(self.eligible_products)
        seed_identity = model_identity_of_product.get(seed_product, "")
        seed_category = category_of_product.get(seed_product, "")

        chosen = [seed_product]
        chosen_set = {seed_product}

        siblings = [p for p in product_codes_by_model_identity.get(seed_identity, ())
                    if p != seed_product and p in self.indices_by_product and len(self.indices_by_product[p]) >= 2]
        rng.shuffle(siblings)
        num_sibling_slots = max(0, round(self.P * COLORWAY_SIBLING_BIAS_FRACTION))
        for p in siblings:
            if len(chosen) >= 1 + num_sibling_slots:
                break
            if p not in chosen_set:
                chosen.append(p)
                chosen_set.add(p)

        category_pool = [p for p in self.products_by_category.get(seed_category, []) if p not in chosen_set]
        rng.shuffle(category_pool)
        num_category_slots = max(0, round(self.P * CATEGORY_BIAS_FRACTION))
        for p in category_pool:
            if len(chosen) >= 1 + num_sibling_slots + num_category_slots:
                break
            chosen.append(p)
            chosen_set.add(p)

        remaining_pool = [p for p in self.eligible_products if p not in chosen_set]
        rng.shuffle(remaining_pool)
        for p in remaining_pool:
            if len(chosen) >= self.P:
                break
            chosen.append(p)
            chosen_set.add(p)

        return chosen[:self.P]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            products = self._select_products(rng)
            batch_indices = []
            for p in products:
                candidates = self.indices_by_product[p]
                if len(candidates) >= self.K:
                    chosen = rng.sample(candidates, self.K)
                else:
                    chosen = [rng.choice(candidates) for _ in range(self.K)]
                batch_indices.extend(chosen)
            yield batch_indices


def collate_identity_batch(batch_items):
    images = [item["image"] for item in batch_items]
    records_out = [item["record"] for item in batch_items]
    return {"images": images, "records": records_out}


train_dataset = IdentityDataset(train_records, augment=True)
train_batch_sampler = IdentityBatchSampler(train_dataset, IDENTITIES_PER_BATCH, IMAGES_PER_IDENTITY, SEED)
train_loader = DataLoader(
    train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_identity_batch,
    num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
)

print(f"\nEligible anchor products (>=2 images): {len(train_batch_sampler.eligible_products):,}")
print(f"Batches/epoch: {len(train_batch_sampler):,}  (batch size {BATCH_SIZE} = {IDENTITIES_PER_BATCH}P x {IMAGES_PER_IDENTITY}K)")


# ============================================================
# Model: DINOv3 backbone + trainable projection head
# ============================================================

processor = AutoImageProcessor.from_pretrained(MODEL_ID)


class ProjectionHead(nn.Module):
    """pooled visual features -> trainable projection head -> L2-normalized
    identity vector (section 4.4)."""

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


def pooled_features(backbone_outputs):
    pooler_output = getattr(backbone_outputs, "pooler_output", None)
    return pooler_output if pooler_output is not None else backbone_outputs.last_hidden_state[:, 0]


def embed_batch(backbone, projection_head, pil_images, use_projection=True):
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with autocast_context():
        outputs = backbone(**inputs)
        raw = pooled_features(outputs).float()
        if use_projection:
            embeddings = projection_head(raw)
        else:
            embeddings = F.normalize(raw, dim=-1)
    return embeddings


def configure_trainable_parameters(backbone, projection_head, mode):
    for p in backbone.parameters():
        p.requires_grad_(False)
    for p in projection_head.parameters():
        p.requires_grad_(True)
    if mode == "last_block":
        # ViT backbones expose the transformer block stack under various
        # attribute paths depending on the HF model class. DINOv3ViTModel
        # specifically nests it at `.model.layer` (a DINOv3ViTEncoder
        # wrapped as `.model`, not `.encoder` as the name might suggest) --
        # confirmed by inspecting the loaded model directly after this
        # path list's first three guesses all missed and crashed stage 2
        # on the real run. `.model.layer` listed first since it's now the
        # confirmed real path for this repo's target model.
        layers = None
        for attr_path in ("model.layer", "encoder.layer", "encoder.layers", "layers"):
            obj = backbone
            ok = True
            for part in attr_path.split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if ok:
                layers = obj
                break
        if layers is None:
            raise RuntimeError("Could not locate transformer block list on DINOv3 backbone for unfreezing.")
        for p in layers[-1].parameters():
            p.requires_grad_(True)


# ============================================================
# Losses (section 5.2 "Candidate losses" -- benchmarked via LOSS_TYPE)
# ============================================================

def supcon_loss(embeddings, product_codes, temperature=SUPCON_TEMPERATURE):
    """Multi-positive supervised contrastive loss (Khosla et al.), the
    natural fit for P x K identity-balanced batches: every other image of
    the same product_code in the batch is a positive, everything else a
    negative."""
    device = embeddings.device
    n = embeddings.shape[0]
    same_identity = torch.zeros((n, n), dtype=torch.bool, device=device)
    for i in range(n):
        for j in range(n):
            same_identity[i, j] = product_codes[i] == product_codes[j]
    self_mask = torch.eye(n, dtype=torch.bool, device=device)
    positive_mask = same_identity & ~self_mask

    similarity = embeddings @ embeddings.T / temperature
    similarity = similarity.masked_fill(self_mask, float("-inf"))
    log_prob = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    # The self-similarity diagonal is -inf and positive_mask is False there,
    # but 0 * -inf = nan in IEEE float arithmetic -- zero it out explicitly
    # before multiplying by the mask rather than relying on 0 * -inf == 0.
    log_prob = log_prob.masked_fill(self_mask, 0.0)

    positive_counts = positive_mask.sum(dim=1).clamp(min=1)
    per_anchor_loss = -(log_prob * positive_mask).sum(dim=1) / positive_counts
    has_positive = positive_mask.sum(dim=1) > 0
    if has_positive.sum() == 0:
        return per_anchor_loss.sum() * 0.0
    return per_anchor_loss[has_positive].mean()


class ArcFaceHead(nn.Module):
    """Additive angular margin classification head, benchmarked against
    SupCon per section 5.2. Requires a fixed identity->index mapping
    (closed-set over the training products), so it's only meaningful
    while all identities seen at train time are also present at eval
    time -- true here since we use a view-based split (same product_codes
    in train and val/test, just different images)."""

    def __init__(self, embedding_dim, num_identities, margin=ARCFACE_MARGIN, scale=ARCFACE_SCALE):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_identities, embedding_dim) * 0.01)
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings, label_indices):
        weight_normalized = F.normalize(self.weight, dim=-1)
        cosine = embeddings @ weight_normalized.T
        target_cosine = cosine.gather(1, label_indices[:, None]).squeeze(1).clamp(-1 + 1e-7, 1 - 1e-7)
        target_angle = torch.acos(target_cosine)
        margin_cosine = torch.cos(target_angle + self.margin)
        logits = cosine.clone()
        logits.scatter_(1, label_indices[:, None], margin_cosine[:, None])
        return logits * self.scale


all_product_codes_train = sorted(set(r["product_code"] for r in train_records))
product_code_to_index = {code: i for i, code in enumerate(all_product_codes_train)}


# ============================================================
# Evaluation: exact-SKU retrieval + same-model/different-colorway
# confusion rate (section 8)
# ============================================================

@torch.inference_mode()
def encode_records(backbone, projection_head, image_records, use_projection=True, batch_size=64):
    backbone.eval()
    projection_head.eval()
    all_embeddings = []
    for start in range(0, len(image_records), batch_size):
        batch = image_records[start:start + batch_size]
        images = []
        for r in batch:
            with Image.open(r["image_path"]) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                images.append(eval_resize(image))
        embeddings = embed_batch(backbone, projection_head, images, use_projection=use_projection)
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, PROJECTION_DIM)


def evaluate_identity_retrieval(backbone, projection_head, query_records, gallery_records, use_projection=True):
    if not query_records or not gallery_records:
        return {
            "num_queries": 0, "recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0,
            "mrr": 0.0, "median_rank": float("nan"), "mean_rank": float("nan"),
            "same_model_confusion_rate": float("nan"), "unrelated_confusion_rate": float("nan"),
        }

    gallery_embeddings = encode_records(backbone, projection_head, gallery_records, use_projection=use_projection)
    query_embeddings = encode_records(backbone, projection_head, query_records, use_projection=use_projection)

    gallery_codes = np.array([r["product_code"] for r in gallery_records])
    gallery_identities = np.array([r["model_identity"] for r in gallery_records])
    query_codes = np.array([r["product_code"] for r in query_records])
    query_identities = np.array([r["model_identity"] for r in query_records])

    ranks = []
    same_model_confusions = 0
    unrelated_confusions = 0
    num_wrong_top1 = 0

    eval_batch = 256
    for start in range(0, len(query_embeddings), eval_batch):
        end = min(start + eval_batch, len(query_embeddings))
        similarity = query_embeddings[start:end] @ gallery_embeddings.T
        sorted_indices = torch.argsort(similarity, dim=1, descending=True).numpy()

        for row, gallery_order in enumerate(sorted_indices):
            qi = start + row
            q_code = query_codes[qi]
            ranked_codes = gallery_codes[gallery_order]
            matches = np.flatnonzero(ranked_codes == q_code)
            rank = int(matches[0]) + 1 if len(matches) else len(gallery_records) + 1
            ranks.append(rank)

            top1_index = gallery_order[0]
            if gallery_codes[top1_index] != q_code:
                num_wrong_top1 += 1
                if gallery_identities[top1_index] == query_identities[qi]:
                    same_model_confusions += 1
                else:
                    unrelated_confusions += 1

    ranks = np.asarray(ranks)
    return {
        "num_queries": int(len(query_records)),
        "num_gallery": int(len(gallery_records)),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "same_model_confusion_rate": float(same_model_confusions / num_wrong_top1) if num_wrong_top1 else 0.0,
        "unrelated_confusion_rate": float(unrelated_confusions / num_wrong_top1) if num_wrong_top1 else 0.0,
    }


def print_metrics(title, metrics):
    print(f"\n{title}")
    print(f"R@1:  {metrics['recall_at_1'] * 100:.2f}%")
    print(f"R@5:  {metrics['recall_at_5'] * 100:.2f}%")
    print(f"R@10: {metrics['recall_at_10'] * 100:.2f}%")
    print(f"MRR:  {metrics['mrr'] * 100:.2f}%")
    print(f"Median rank: {metrics['median_rank']:.1f}")
    print(f"Mean rank: {metrics['mean_rank']:.2f}")
    if not math.isnan(metrics.get("same_model_confusion_rate", float("nan"))):
        print(f"Of wrong top-1s: {metrics['same_model_confusion_rate'] * 100:.1f}% same-model/different-colorway, "
              f"{metrics['unrelated_confusion_rate'] * 100:.1f}% unrelated item")


# ============================================================
# One training stage (shared by stage 1 and stage 2)
# ============================================================

RESUME_STATE_PATH = OUTPUT_DIR / "resume_state.json"


def load_resume_state():
    return load_json_safely(RESUME_STATE_PATH, {
        "stage1_complete": False, "stage1_epochs_done": 0, "stage1_best_score": -1.0,
        "stage2_complete": False, "stage2_epochs_done": 0, "stage2_best_score": -1.0,
    })


def save_resume_state(state):
    atomic_write_json(RESUME_STATE_PATH, state)


def run_stage(backbone, projection_head, mode, epochs, lr, save_dir, stage_name,
              arcface_head=None, start_epoch=1, history=None, best_score=-1.0,
              resume_state=None, resume_state_prefix=None):
    configure_trainable_parameters(backbone, projection_head, mode)
    trainable_parameters = [p for p in backbone.parameters() if p.requires_grad]
    trainable_parameters += [p for p in projection_head.parameters() if p.requires_grad]
    if arcface_head is not None:
        trainable_parameters += list(arcface_head.parameters())
    trainable_count = sum(p.numel() for p in trainable_parameters)
    total_count = sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for p in projection_head.parameters())
    print(f"\n[{stage_name}] Trainable parameters: {trainable_count:,} / {total_count:,} "
          f"({100 * trainable_count / total_count:.3f}%)")

    # No gradient checkpointing here: with everything except the last
    # transformer block frozen, checkpointed segments upstream of it
    # receive an input that never has requires_grad=True, and HF's
    # (reentrant) checkpoint implementation then either throws
    # "element 0 of tensors does not require grad and does not have a
    # grad_fn" on the first backward pass, or in some version
    # combinations silently produces no gradient at all for the
    # supposedly-trainable block. Only one block is ever trainable here,
    # so the memory savings wouldn't have been worth that risk anyway.

    remaining_epochs = epochs - (start_epoch - 1)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=WEIGHT_DECAY)
    updates_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
    total_training_steps = max(1, updates_per_epoch * remaining_epochs)
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    history = list(history) if history else []

    for epoch in range(start_epoch, epochs + 1):
        train_batch_sampler.set_epoch(epoch)
        backbone.train()
        projection_head.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        batches_seen = 0

        progress = tqdm(train_loader, desc=f"[{stage_name}] Epoch {epoch}/{epochs}")
        for batch_index, batch in enumerate(progress):
            product_codes = [r["product_code"] for r in batch["records"]]
            embeddings = embed_batch(backbone, projection_head, batch["images"], use_projection=True)

            with autocast_context():
                if LOSS_TYPE == "arcface":
                    label_indices = torch.tensor(
                        [product_code_to_index[c] for c in product_codes], device=DEVICE,
                    )
                    logits = arcface_head(embeddings, label_indices)
                    loss = F.cross_entropy(logits, label_indices)
                else:
                    loss = supcon_loss(embeddings, product_codes)
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
        validation_metrics = evaluate_identity_retrieval(backbone, projection_head, val_records, train_records, use_projection=True)
        print_metrics(f"[{stage_name}] Epoch {epoch}: exact-SKU validation retrieval", validation_metrics)

        selection_score = validation_metrics["recall_at_1"]
        history.append({"epoch": epoch, "train_loss": average_train_loss, "selection_score": selection_score,
                         "validation_metrics": validation_metrics})
        atomic_write_json(OUTPUT_DIR / f"{stage_name}_training_history.json", history)

        if selection_score > best_score:
            best_score = selection_score
            save_dir.mkdir(parents=True, exist_ok=True)
            extra_state = {"arcface_state_dict": arcface_head.state_dict()} if arcface_head is not None else None
            save_checkpoint(backbone, projection_head, save_dir, extra_state=extra_state)
            atomic_write_json(save_dir / "validation_metrics.json",
                               {"selection_score": selection_score, "validation_metrics": validation_metrics})
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
# Baseline (frozen, untrained-head) evaluation -- section 5.1: "do not
# modify pretrained weights until baselines are recorded." Uses raw
# backbone pooled features (no projection), matching run_dinov3_baseline.py.
# ============================================================

resume_state = load_resume_state()
resuming = resume_state["stage1_epochs_done"] > 0 or resume_state["stage1_complete"]

backbone_load_path = MODEL_ID
if resuming:
    # Only trust a leftover STAGE1_DIR checkpoint when resume_state.json
    # itself says we're mid-stage1 -- otherwise a stale checkpoint from an
    # unrelated earlier attempt would silently load into what's supposed
    # to be a fresh run, corrupting the "frozen baseline" measurement.
    backbone_load_path = load_checkpoint_backbone_path(STAGE1_DIR) or MODEL_ID
backbone = AutoModel.from_pretrained(backbone_load_path, torch_dtype=torch.float32).to(DEVICE)
projection_head = ProjectionHead(backbone.config.hidden_size, PROJECTION_DIM).to(DEVICE)

stage1_head_ckpt = STAGE1_DIR / "projection_head.pt"
if resuming and stage1_head_ckpt.is_file():
    projection_head.load_state_dict(torch.load(stage1_head_ckpt, map_location=DEVICE))

arcface_head = None
if LOSS_TYPE == "arcface":
    arcface_head = ArcFaceHead(PROJECTION_DIM, len(all_product_codes_train)).to(DEVICE)
    stage1_extra = STAGE1_DIR / "extra_state.pt"
    if resuming and stage1_extra.is_file():
        arcface_head.load_state_dict(torch.load(stage1_extra, map_location=DEVICE)["arcface_state_dict"])

if resuming:
    print(f"\nResuming from saved state: {resume_state}")
    baseline_metrics = load_json_safely(OUTPUT_DIR / "baseline_metrics.json", {"recall_at_1": float("nan")})
else:
    baseline_metrics = evaluate_identity_retrieval(backbone, projection_head, val_records, train_records, use_projection=False)
    print_metrics("Frozen-backbone baseline: exact-SKU validation retrieval (raw pooled features, no head)", baseline_metrics)
    atomic_write_json(OUTPUT_DIR / "baseline_metrics.json", baseline_metrics)


# ============================================================
# Stage 1: frozen backbone, train projection head only
# ============================================================

if resume_state["stage1_complete"]:
    print("\n[stage1_head] Already complete per resume state, skipping.")
    stage1_best = resume_state["stage1_best_score"]
else:
    stage1_history = load_json_safely(OUTPUT_DIR / "stage1_head_training_history.json", [])
    stage1_best = run_stage(
        backbone, projection_head, "head_only", STAGE1_EPOCHS, STAGE1_LR, STAGE1_DIR, "stage1_head",
        arcface_head=arcface_head, start_epoch=resume_state["stage1_epochs_done"] + 1, history=stage1_history,
        best_score=resume_state["stage1_best_score"], resume_state=resume_state, resume_state_prefix="stage1",
    )


# ============================================================
# Stage 2: unfreeze last block, continue from stage 1's best checkpoint --
# only kept as final if it actually beats stage 1 (section 5.2 step 5:
# "continue only if held-out retrieval improves")
# ============================================================

del backbone
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Same rule as the stage1/baseline load above: only trust an on-disk
# STAGE2_DIR checkpoint when resume_state.json says stage 2 actually has
# progress. Otherwise a leftover STAGE2_DIR from an unrelated earlier
# attempt would silently become this "fresh" stage 2's starting point
# instead of stage 1's just-trained checkpoint.
resuming_stage2 = resume_state["stage2_epochs_done"] > 0 or resume_state["stage2_complete"]
stage2_backbone_path = None
if resuming_stage2:
    stage2_backbone_path = load_checkpoint_backbone_path(STAGE2_DIR)
if stage2_backbone_path is None:
    stage2_backbone_path = load_checkpoint_backbone_path(STAGE1_DIR)
backbone = AutoModel.from_pretrained(stage2_backbone_path, torch_dtype=torch.float32).to(DEVICE)

stage2_head_ckpt = STAGE2_DIR / "projection_head.pt"
if resuming_stage2 and stage2_head_ckpt.is_file():
    projection_head.load_state_dict(torch.load(stage2_head_ckpt, map_location=DEVICE))
elif (STAGE1_DIR / "projection_head.pt").is_file():
    projection_head.load_state_dict(torch.load(STAGE1_DIR / "projection_head.pt", map_location=DEVICE))

if arcface_head is not None and resuming_stage2 and (STAGE2_DIR / "extra_state.pt").is_file():
    arcface_head.load_state_dict(torch.load(STAGE2_DIR / "extra_state.pt", map_location=DEVICE)["arcface_state_dict"])

if resume_state["stage2_complete"]:
    print("\n[stage2_lastblock] Already complete per resume state, skipping.")
    stage2_best = resume_state["stage2_best_score"]
else:
    stage2_history = load_json_safely(OUTPUT_DIR / "stage2_lastblock_training_history.json", [])
    stage2_best = run_stage(
        backbone, projection_head, "last_block", STAGE2_EPOCHS, STAGE2_LR, STAGE2_DIR, "stage2_lastblock",
        arcface_head=arcface_head, start_epoch=resume_state["stage2_epochs_done"] + 1, history=stage2_history,
        best_score=resume_state["stage2_best_score"], resume_state=resume_state, resume_state_prefix="stage2",
    )


# ============================================================
# Final held-out evaluation -- whichever stage scored higher on validation
# (section 5.2 step 5: stage 2 must actually beat stage 1 to be used)
# ============================================================

del backbone
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

final_dir = STAGE2_DIR if stage2_best >= stage1_best else STAGE1_DIR
print(f"\nBest stage: {'stage2_lastblock' if stage2_best >= stage1_best else 'stage1_head'} "
      f"(val R@1 stage1={stage1_best * 100:.2f}%, stage2={stage2_best * 100:.2f}%)")

best_backbone = AutoModel.from_pretrained(final_dir / "backbone", torch_dtype=torch.float32).to(DEVICE)
best_projection_head = ProjectionHead(best_backbone.config.hidden_size, PROJECTION_DIM).to(DEVICE)
best_projection_head.load_state_dict(torch.load(final_dir / "projection_head.pt", map_location=DEVICE))

test_metrics = evaluate_identity_retrieval(best_backbone, best_projection_head, test_records, train_records, use_projection=True)
print_metrics("Final test: exact-SKU retrieval", test_metrics)

atomic_write_json(OUTPUT_DIR / "test_metrics.json", {
    "final_checkpoint": str(final_dir),
    "loss_type": LOSS_TYPE,
    "baseline_val_recall_at_1": baseline_metrics["recall_at_1"],
    "stage1_best_val_recall_at_1": stage1_best,
    "stage2_best_val_recall_at_1": stage2_best,
    "test_metrics": test_metrics,
})

print("\nBest checkpoint:", final_dir)
