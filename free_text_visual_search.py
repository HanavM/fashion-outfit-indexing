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


IMAGE_LOADER_WORKERS = int(os.environ.get("IMAGE_LOADER_WORKERS", "1"))


def _load_one(record):
    """-> (record, PIL image) or (record, None). Never raises: one unreadable
    file must not take down a batch of 32."""
    try:
        with Image.open(record["path"]) as image:
            return record, ImageOps.exif_transpose(image).convert("RGB")
    except Exception:  # noqa: BLE001
        return record, None


@torch.inference_mode()
def encode_images(model, processor, records):
    """Batch-encode, loading each batch's files in PARALLEL.

    Serial loading is fine on local disk and catastrophic on a network
    filesystem, which is where this runs on Modal. Measured there: 14.9 s
    per 32-image batch, i.e. ~0.46 s/image, all of it per-file round trip
    while the A10G idled -- slower than the same job on a laptop's MPS.
    The GPU was never the constraint.

    IMAGE_LOADER_WORKERS defaults to 1 so local behaviour is unchanged;
    Modal sets 32, matching what modal_app_serve.py already does for the
    catalog for exactly this reason.
    """
    from concurrent.futures import ThreadPoolExecutor

    embeddings, kept = [], []
    pool = ThreadPoolExecutor(max_workers=IMAGE_LOADER_WORKERS) \
        if IMAGE_LOADER_WORKERS > 1 else None
    for start in tqdm(range(0, len(records), IMAGE_BATCH_SIZE), desc="Encoding catalog images"):
        batch = records[start:start + IMAGE_BATCH_SIZE]
        images, batch_kept = [], []
        # map() preserves order, so embeddings stay aligned with `kept`.
        loaded = pool.map(_load_one, batch) if pool else map(_load_one, batch)
        for r, image in loaded:
            if image is None:
                continue
            images.append(image)
            batch_kept.append(r)
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


def extract_dense_patch_projection(vision_model):
    """MaskCLIP-style dense/patch-level readout, adapted to SigLIP2's actual
    pooling-head architecture rather than CLIP's CLS-token last-attention-
    layer. SigLIP2 doesn't pool via a CLS token attending to itself in the
    last self-attention block (the case MaskCLIP's original "drop Q/K, keep
    V + out_proj" recipe targets) -- it pools via a separate
    SiglipMultiheadAttentionPoolingHead: a learned probe token attends to
    every patch token via standard multi-head attention (probe=query,
    patches=key+value), then a residual LayerNorm+MLP block. The same core
    trick still applies, generalized to this head shape: skip the query/key
    softmax entirely (which is what collapses every patch into one
    attention-weighted pooled vector) and instead run every patch through
    the same value-projection -> out_proj -> residual LayerNorm+MLP path the
    probe's pooled output goes through. This keeps each patch's resulting
    vector in the exact same space the text tower was aligned against
    (nothing about the text side or the earlier vision layers changes),
    while producing one vector *per patch* instead of one pooled vector for
    the whole image.

    Returns a function: last_hidden_state [B, N, D] -> dense_features [B, N, D]
    (not yet normalized).
    """
    head = vision_model.head
    mha = head.attention
    embed_dim = mha.embed_dim
    v_weight = mha.in_proj_weight[2 * embed_dim:3 * embed_dim]  # packed [q; k; v], each embed_dim rows
    v_bias = mha.in_proj_bias[2 * embed_dim:3 * embed_dim]

    def project(last_hidden_state):
        value = F.linear(last_hidden_state, v_weight, v_bias)
        value = mha.out_proj(value)
        residual = value
        value = head.layernorm(value)
        value = residual + head.mlp(value)
        return value

    return project


@torch.inference_mode()
def encode_images_dense(model, processor, records):
    """Per-patch text-aligned features for a small candidate list -- meant
    to be called on an already-narrowed shortlist (see search_dense_rerank),
    not the full catalog: patch-level similarity is N_patches times more
    comparisons than the pooled-vector version, so running it catalog-wide
    would be needlessly expensive for the same reason
    hierarchical_retrieval_pipeline.py only runs its expensive DINOv3 rerank
    stage against a SigLIP2-narrowed shortlist, not the whole gallery."""
    project = extract_dense_patch_projection(model.vision_model)
    dense_by_index = []
    for r in records:
        with Image.open(r["path"]) as image:
            pil_image = ImageOps.exif_transpose(image).convert("RGB")
        inputs = processor(images=[pil_image], return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with autocast_context():
            vision_out = model.vision_model(**inputs)
            last_hidden_state = vision_out.last_hidden_state.float()
            patches = project(last_hidden_state).squeeze(0)
        dense_by_index.append(F.normalize(patches, dim=-1).cpu())
    return dense_by_index


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

    def search_dense_rerank(self, query_text, top_k=15, shortlist_k=50, patch_agg="max"):
        """Two-stage: cheap global-embedding pre-filter (existing v1 search,
        full catalog) narrows to shortlist_k candidates, then dense
        patch-level matching reranks just that shortlist. A localized query
        ("stitching across the back") can now win on whichever single patch
        matches best, instead of being diluted into one pooled vector --
        but only within images the pooled-vector stage already thought were
        plausible, matching the architecture note in docs/roadmap.md's
        2026-08-01 free-text-search update: dense matching is more
        expensive per-image (a similarity per patch, not per image), so pay
        that cost only on a pre-narrowed shortlist, not the full catalog.

        **Validated against the real v3 checkpoint, 2026-08-02
        (dense_rerank_real_checkpoint_validation.py, 10 real localized
        queries): result is a genuine coin flip, NOT a default-on
        improvement -- 4 improved, 4 worse, 1 unchanged, 1 unrescuable.
        One case actively made an already-#1-ranked result worse (28th).
        Do not enable this by default as currently implemented; see
        docs/eval_log.md's 2026-08-02 entries for full numbers and a
        concrete idea for a threshold-gated version (only rerank when the
        pooled rank is already poor) that wasn't tested here.**"""
        text_embedding = encode_text(self.model, self.processor, [query_text])[0]

        pooled_similarities = self.image_embeddings @ text_embedding
        shortlist_order = torch.argsort(pooled_similarities, descending=True)[:shortlist_k].tolist()
        shortlist_records = [self.records[i] for i in shortlist_order]

        dense_features = encode_images_dense(self.model, self.processor, shortlist_records)

        scored = []
        for local_index, patches in enumerate(dense_features):
            patch_similarities = patches @ text_embedding
            if patch_agg == "max":
                score = float(patch_similarities.max())
            elif patch_agg == "top5_mean":
                k = min(5, patch_similarities.shape[0])
                score = float(torch.topk(patch_similarities, k).values.mean())
            else:
                raise ValueError(f"Unknown patch_agg: {patch_agg}")
            scored.append((local_index, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        results = []
        for rank, (local_index, score) in enumerate(scored[:top_k], start=1):
            record = shortlist_records[local_index]
            results.append({
                "rank": rank, "product_code": record["code"], "brand": record["brand"],
                "name": record["name"], "score": score,
            })
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--dense-rerank", action="store_true",
                         help="Rerank a pooled-embedding shortlist with MaskCLIP-style dense/patch-level matching "
                              "(slower, targets localized queries the pooled-vector search misses -- see "
                              "docs/roadmap.md's 2026-08-01 free-text-search update).")
    parser.add_argument("--shortlist-k", type=int, default=50,
                         help="Candidate pool size for --dense-rerank's pooled pre-filter stage.")
    parser.add_argument("--patch-agg", type=str, default="max", choices=["max", "top5_mean"],
                         help="How to aggregate per-patch similarities into one dense-match score.")
    args = parser.parse_args()

    engine = FreeTextVisualSearch()
    if args.dense_rerank:
        results = engine.search_dense_rerank(args.query, top_k=args.top_k, shortlist_k=args.shortlist_k, patch_agg=args.patch_agg)
    else:
        results = engine.search(args.query, top_k=args.top_k)

    print(f"\nQuery: '{args.query}'" + ("  [dense rerank]" if args.dense_rerank else ""))
    for r in results:
        print(f"  #{r['rank']}  {r['brand']} {r['name']}  [{r['product_code']}]  score={r['score']:.4f}")
