"""Open-vocabulary free-text visual search: a user's own words, never seen
verbatim in training (e.g. "show me clothes with stitching across the
back"), embedded via SigLIP2's text tower and ranked against cached
SigLIP2 image embeddings for the whole catalog. No relabeling, no
retraining -- this is the standard CLIP/SigLIP zero-shot retrieval
pattern, deliberately kept separate from hierarchical_retrieval_pipeline.py
(which answers "which exact product is this" from a photo) since this
answers a different question ("which products match this description").

Validated before building, not assumed:
1. Researched whether fine-tuning risks catastrophic forgetting of
   general zero-shot capability (a documented real risk -- published
   work found an average 16-17% zero-shot degradation after CLIP
   fine-tuning). Then actually tested it against our own v3 checkpoint
   with 3 real catalog products and hand-written query paraphrases
   (never copied from training text): fine-tuning *improved* 2 of 3
   ranks (37->29, 175->104 out of 1,146) rather than degrading them --
   opposite of the general-literature average. Plausible reason: our
   training already includes `defining_features` (free-text localized-
   detail descriptions) as a real target, so fine-tuning reinforced this
   kind of query instead of narrowing away from it. This is a project-
   specific empirical result, not a general claim -- don't assume it
   transfers to a differently-trained checkpoint without retesting.
2. Rank ~30-175 out of ~1,150 (not top-5) confirms the deeper, separate
   limitation also found in research: SigLIP/CLIP pool the whole image
   into one global vector trained via contrastive loss, which rewards
   whatever distinguishes images most efficiently across a batch --
   usually category/color/silhouette, not a small localized detail. This
   is an architectural bottleneck, not a data problem -- richer
   defining_features labels would apply more training pressure toward
   packing such detail into that one vector (real, but with a ceiling),
   not remove the bottleneck. The actual fix is dense/patch-level
   matching (e.g. the MaskCLIP technique: discard the last attention
   layer's Q/K, turn V + the output projection into a 1x1 conv to get
   text-aligned per-patch features, then match a query against the SET
   of patches instead of one pooled vector) -- a real, established,
   training-free technique, NOT implemented here. More involved and more
   fragile (needs to reach into the vision transformer's attention
   internals) than this global-embedding version, so shipped separately
   as a validated next step rather than rushed into v1 untested.

Usage:
    python3 free_text_visual_search.py --query "stitching across the back" --top-k 15
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

# APPAREL_DATASET_ROOT override exists for local/non-Colab test runs,
# same convention as hierarchical_retrieval_pipeline.py.
DATASET_ROOT = Path(os.environ.get("APPAREL_DATASET_ROOT", "/content/drive/MyDrive/apparel_dataset"))
METADATA_PATH = DATASET_ROOT / "metadata.json"
INDEX_DIR = DATASET_ROOT / "retrieval_indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_EMBEDDINGS_PATH = INDEX_DIR / "free_text_search_image_embeddings.pt"

BASE_MODEL_ID = "google/siglip2-base-patch16-384"
# Same preference order as hierarchical_retrieval_pipeline.py -- v3 is the
# proven, validated checkpoint. v4 regressed on the exact-label task it
# was measured against, and hasn't been separately validated for this
# free-text-query use case (plausible it could actually help here, since
# it's an attribute/facet-reweighting change, but that's a hypothesis to
# test, not an assumption to build in) -- not included as a candidate
# until there's real evidence one way or the other.
SIGLIP2_CHECKPOINT_CANDIDATES = [
    DATASET_ROOT / "finetuned_siglip2_hierarchical_v3" / "stage2_lastnblocks_best",
    DATASET_ROOT / "finetuned_siglip2_hierarchical_v3" / "stage1_heads_best",
    DATASET_ROOT / "finetuned_siglip2_hierarchical_v2" / "stage2_lastblock_best",
    DATASET_ROOT / "finetuned_siglip2_hierarchical_v2" / "stage1_heads_best",
]

IMAGES_PER_PRODUCT = 1  # one representative image per product, for index build speed
IMAGE_BATCH_SIZE = 32
TEXT_MAX_LENGTH = 64

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
USE_AMP = DEVICE == "cuda"


def autocast_context():
    import contextlib
    return torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else contextlib.nullcontext()


def pick_checkpoint():
    for path in SIGLIP2_CHECKPOINT_CANDIDATES:
        if (path / "model.safetensors").is_file():
            return path
    return None


def extract_embeddings(output):
    """Version-dependent: get_text_features/get_image_features return a
    bare tensor on some transformers versions, a wrapped output object on
    others -- same defensive unwrap as finetune_siglip2_v3.py /
    hierarchical_retrieval_pipeline.py."""
    if torch.is_tensor(output):
        return output
    for attribute in ("text_embeds", "image_embeds", "pooler_output"):
        value = getattr(output, attribute, None)
        if value is not None:
            return value
    raise TypeError(f"Cannot extract embeddings from {type(output)}")


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


def load_catalog_records():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    records = []
    for product in metadata:
        code = str(product.get("product_code", "")).strip()
        images = product.get("images") or []
        if not code or not images:
            continue
        for raw_path in images[:IMAGES_PER_PRODUCT]:
            path = resolve_image_path(raw_path)
            if path is not None:
                records.append({"code": code, "path": str(path), "brand": product.get("brand", ""), "name": product.get("name", "")})
                break
    return records


@torch.inference_mode()
def encode_images(model, processor, records):
    embeddings, kept = [], []
    for start in tqdm(range(0, len(records), IMAGE_BATCH_SIZE), desc="Encoding catalog images"):
        batch = records[start:start + IMAGE_BATCH_SIZE]
        images, batch_kept = [], []
        for r in batch:
            try:
                with Image.open(r["path"]) as image:
                    images.append(ImageOps.exif_transpose(image).convert("RGB"))
                batch_kept.append(r)
            except Exception:
                continue
        if not images:
            continue
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with autocast_context():
            output = extract_embeddings(model.get_image_features(**inputs)).float()
        embeddings.append(F.normalize(output, dim=-1).cpu())
        kept.extend(batch_kept)
    return torch.cat(embeddings, dim=0), kept


@torch.inference_mode()
def encode_text(model, processor, texts):
    inputs = processor(text=texts, padding="max_length", truncation=True, max_length=TEXT_MAX_LENGTH, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with autocast_context():
        output = extract_embeddings(model.get_text_features(**inputs)).float()
    return F.normalize(output, dim=-1).cpu()


def build_or_load_image_index(model, processor, checkpoint_label):
    config_path = INDEX_DIR / "free_text_search_config.json"
    records = load_catalog_records()
    expected = {"checkpoint": checkpoint_label, "num_products": len(records)}

    if config_path.is_file():
        try:
            current = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            current = None
        if current == expected and IMAGE_EMBEDDINGS_PATH.is_file():
            payload = torch.load(IMAGE_EMBEDDINGS_PATH, map_location="cpu", weights_only=False)
            print("Image index: cache hit.")
            return payload["embeddings"], payload["records"]

    print("Image index: (re)building...")
    embeddings, kept_records = encode_images(model, processor, records)
    torch.save({"embeddings": embeddings, "records": kept_records}, IMAGE_EMBEDDINGS_PATH)
    config_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return embeddings, kept_records


class FreeTextVisualSearch:
    def __init__(self):
        checkpoint = pick_checkpoint()
        load_from = str(checkpoint) if checkpoint else BASE_MODEL_ID
        print(f"SigLIP2 checkpoint: {load_from}" + ("" if checkpoint else "  (no fine-tune found -- using base model)"))
        self.model = AutoModel.from_pretrained(load_from, torch_dtype=torch.float32).to(DEVICE).eval()
        self.processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
        self.image_embeddings, self.records = build_or_load_image_index(self.model, self.processor, load_from)

    def search(self, query_text, top_k=15):
        text_embedding = encode_text(self.model, self.processor, [query_text])[0]
        similarities = self.image_embeddings @ text_embedding
        order = torch.argsort(similarities, descending=True)[:top_k]
        results = []
        for rank, index in enumerate(order.tolist(), start=1):
            record = self.records[index]
            results.append({
                "rank": rank, "product_code": record["code"], "brand": record["brand"],
                "name": record["name"], "score": float(similarities[index]),
            })
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    engine = FreeTextVisualSearch()
    results = engine.search(args.query, top_k=args.top_k)

    print(f"\nQuery: '{args.query}'")
    for r in results:
        print(f"  #{r['rank']}  {r['brand']} {r['name']}  [{r['product_code']}]  score={r['score']:.4f}")
