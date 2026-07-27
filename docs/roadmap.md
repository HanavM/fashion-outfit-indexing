# Development roadmap

Status snapshot and phased plan following the audit of the codebase against
`docs/project_spec_v1.md`. Written 2026-07-27.

## Where things actually stand

**Mature / spec-aligned:**
- Scraping + enrichment across 6 brands (nike, skechers, gap, pacsun,
  newbalance, adidas) → `apparel_dataset/metadata.json`, 1115 products,
  images on disk under `apparel_dataset/<brand>/...`.
- `caption_apparel.py` + `newLLMprompt.py`: LLM-generated `structured_caption`
  on 100% of records — taxonomy path, independent attribute facets
  (color/material/pattern/fit/...), 5-10 positive texts spanning specificity
  levels. This is the closest thing in the repo to spec §2.4/§4.5/§4.6 and
  should be the foundation everything else builds from.
- `dataset_utils.py`'s merge-on-save pattern for `metadata.json` — keep using
  it, it exists specifically because a naive overwrite once destroyed 176
  freshly-scraped records.

**Exists but stale/broken:**
- `notebooks/fashionsiglip2_hsc_finetune.ipynb` (cleaned copy of the
  original `fashionsiglip2_v_siglip2.ipynb`): a real SigLIP2 fine-tune —
  frozen baseline benchmark, zero-shot HSC hierarchy-climbing demo,
  hierarchical multi-positive fine-tuning with identity-balanced (P×K)
  batches and product-level splits, fine-tuned-vs-base evaluation by label
  granularity. **But** it still points at the old `shoe_dataset` layout with
  a flat `caption` field and a hand-written, shoe-only, regex-based
  hierarchy (`HIERARCHY` dict in the HSC demo cell) — it predates the
  `apparel_dataset` expansion to 6 brands and the richer `structured_caption`
  schema entirely.
- `embed_catalog.py` / `search_shoes.py` / `classify_views.py`: all three
  reference directories that no longer exist (`adidas_catalog/`,
  `nike_catalog/`, `newbalance_catalog/`, `shoe_dataset/` — superseded by
  `apparel_dataset/`). `catalog_embeddings.npy`/`catalog_metadata.json`
  (2361×768, dated Jul 14) only cover nike+adidas and are gitignored as
  stale — will be regenerated, not fixed in place.
- `segment_apparel.py`: SAM2 automatic-mask + zero-shot FashionCLIP crop
  selection, hardcoded to `brand=="nike"` only. 552/1115 records have crops.
  No bounding boxes, no per-item confidence/visibility/occlusion fields.
- A vendored, unused `Self-Correction-Human-Parsing/` clone — real per-pixel
  garment parsing, a much better fit for spec §4.2 detection than the
  current SAM2-guess approach, but nothing calls it yet.

- `notebooks/base_dino_visual_search.ipynb` (cleaned copy of the
  `base_DINO_visual_search*` notebooks — 4 redundant, monotonically-growing
  copies existed in Downloads, deduplicated into one): frozen DINOv3
  (`facebook/dinov3-vitb16-pretrain-lvd1689m`) global-pooled, L2-normalized
  embedding index with save/reload, nearest-neighbor search, product-level
  aggregation, and held-out 80/20 evaluation (R@1/R@5/R@10/MRR). Fixed two
  real bugs while deduplicating: a variable-name collision that silently let
  the held-out eval gallery include the test queries themselves (inflating
  Recall@K), and a cell calling a function before its definition. Same
  `shoe_dataset`-path staleness as the SigLIP notebook — needs repointing at
  `apparel_dataset` before the next real run (Phase 0 below covers both).
- **The hybrid DINOv3+SigLIP fusion/reranking pipeline
  (`hybrid_dinov3_siglip_apparel_matching.py` + its runner notebook) was
  deleted, per user direction** — it measurably decreased retrieval
  performance vs. the plain approaches, and is being restarted from scratch
  rather than carried forward or debugged. It had real substance (hard-negative
  fusion reranker training, patch-level reranking, optional OCR brand
  evidence) that mapped well onto spec §4.4/§6/§7, so the *shape* of that
  work is worth reconstructing later — just not its current implementation.

**Absent entirely (no code anywhere):** confidence calibration / open-set
rejection (§7), product-candidate ranking (§4.5), separate
metadata/semantic/identity/lexical indexes (§4.7), query facet parsing and
multi-index fusion (§4.8), patch-level/OCR/logo reranking (§6), a
detection schema with bounding boxes and per-item records (§4.2), dataset/
model/index versioning (§11), and any evaluation harness (§8: Recall@K, mAP,
calibration error, unsupported-fact rate).

## Phased plan

### Phase 0 — Repoint the pipeline at current data (do first, small, unblocks everything)
1. Fix `embed_catalog.py`/`search_shoes.py`/`classify_views.py` to read
   `apparel_dataset/metadata.json` and its real image paths instead of the
   dead `*_catalog`/`shoe_dataset` directories.
2. Update the fine-tuning notebook's `DATASET_ROOT`/`METADATA_PATH` to
   `apparel_dataset/metadata.json`, and swap its label-building step from
   regex-based extraction over a flat `caption` string to reading
   `structured_caption.taxonomy_path` / `.attributes` / `.positive_texts`
   directly — this removes a whole layer of fragile parsing since the LLM
   captioner already emits structured fields.
3. Replace the notebook's hand-written shoe-only `HIERARCHY` dict with one
   generated from the actual `taxonomy_path` values observed across all 1115
   records (union of paths, deduped) so the hierarchy reflects 6 brands and
   non-shoe categories (jeans, shorts, tops, jackets from gap/pacsun/nike
   clothing) instead of a hardcoded example.

### Phase 1 — Re-run frozen baselines, then re-train SigLIP2 on the full dataset
Per spec §5.1, re-establish frozen baselines first now that the eval set is
6 brands instead of 2: FashionCLIP, base SigLIP2, and the existing
`srpone/zooclaw-fashionsiglip2` checkpoint, using the **current** structured
captions as retrieval targets. Then re-run the hierarchical multi-positive
fine-tune (the notebook's Part 4) against the full corpus. Recommended
starting point per the spec: `TRAIN_MODE="heads"` first, evaluate, only then
consider unfreezing further blocks — this matches what the notebook already
does, just needs current data.

### Phase 2 — Extend detection/segmentation to all brands
`segment_apparel.py`'s Nike-only scope is the biggest coverage gap for
downstream training (identity-balanced batching and DINOv3 training both
need clean per-item crops for every brand, not just Nike). Two options,
worth benchmarking against each other rather than picking blind:
- Extend the existing SAM2 + zero-shot-CLIP approach's `CATEGORY_LABELS` to
  cover gap/pacsun/skechers/newbalance/adidas categories.
- Wire up the vendored `Self-Correction-Human-Parsing` model instead —
  purpose-built per-pixel garment parsing, likely both faster and more
  accurate than mask-guessing + zero-shot classification, and it's already
  sitting in the repo unused.

Either way, extend the output schema toward the spec's §4.2 shape
(`item_id`, `bounding_box`, `detection_confidence`, `visibility`,
`occlusion`) instead of the current single-best-crop-per-record approach —
outfit images with multiple visible items need multi-item records eventually
(Phase 3 of the spec's own MVP plan), and retrofitting the schema later is
more expensive than building it in now.

### Phase 3 — Build the DINOv3 identity pipeline forward from the base index
`base_dino_visual_search.ipynb` currently does frozen-backbone embedding +
exact search only — no projection head, no training loop, no hard negatives.
That's actually the correct Phase-1-equivalent starting point per spec
§5.2's fine-tuning order (frozen first, evaluate, then adapt). Build up from
there rather than reintroducing the deleted hybrid pipeline's complexity in
one jump:
1. Repoint at `apparel_dataset`, re-run the held-out eval to get a real
   frozen-DINOv3 Recall@K baseline across all 6 brands (currently only ever
   evaluated against the old 2-brand shoe set).
2. Train a projection head only (`heads` mode, same staged-unfreezing
   pattern as the SigLIP notebook) on identity-balanced P×K batches —
   `ProductBatchSampler` from the SigLIP notebook is directly reusable here.
3. Add hard-negative mining once head-only training shows a held-out
   improvement, prioritized per spec §5.2: same model/different colorway →
   same brand/similar silhouette → visually similar competing models → same
   category+color.
4. Only revisit SigLIP+DINO score fusion and patch-level reranking (what the
   deleted hybrid pipeline attempted) after both encoders have their own
   validated frozen-and-adapted baselines — evaluate fusion against each
   encoder alone before trusting it, since that's likely why the previous
   attempt regressed (fusion was probably layered on before either side was
   validated in isolation).

### Phase 4 — Real dual-encoder index + retrieval
- Separate metadata (structured facts), semantic (SigLIP2), and identity
  (DINOv3) indexes per spec §4.7, instead of the current single flat
  `.npy` + JSON pair.
- `search_shoes.py`'s single-vector cosine top-k becomes real query
  execution per §4.8: parse query into facets, search all three indexes,
  union/dedupe candidates, rerank.
- A reasonable local starting point for the vector side is on-disk
  FAISS/HNSW rather than committing to a hosted vector DB yet — the spec
  explicitly says not to permanently couple to one vector DB before
  benchmarking (§10), and the dataset (1115 products, thousands of images)
  is small enough that a managed service isn't needed yet.

### Phase 5 — Confidence calibration, open-set rejection, evaluation harness
Nothing here exists yet and it's required before any "exact product" claim
can be trusted (§7). Needs held-out known/unknown product splits, a
calibration procedure (margin between top-2 candidates, cross-view
agreement, etc.), and the eval metrics from §8 (Recall@K, mAP, calibration
error, unsupported-fact rate) so model/threshold choices are evidence-based
rather than assumed.

## Immediate next step

Phase 0 is small, unblocks re-training and re-indexing, and doesn't require
any new modeling decisions — recommend starting there.
