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

### Phase 0 — Repoint the pipeline at current data — DONE (2026-07-27)
All data/model work now runs against Google Drive via Colab, not local
disk — verified `apparel_dataset` (1115 products, 6 brands) is fully backed
up to Drive before clearing ~8.3GB of local image data (kept a 2-products/
brand local sample, gitignored, for spot-checks).

1. ✅ `embed_catalog.py`/`search_shoes.py`/`classify_views.py` now read
   `apparel_dataset/metadata.json` on Drive instead of the dead
   `*_catalog`/`shoe_dataset` directories.
2. ✅ Both notebooks' `DATASET_ROOT`/`METADATA_PATH` now point at
   `/content/drive/MyDrive/apparel_dataset`. The fine-tuning notebook's
   label-building step was rewritten to read `structured_caption.taxonomy_path`
   / `.attributes` / `.positive_texts` directly instead of regex-parsing a
   flat `caption` string — removed the old shoe-only
   COLOR_ALIASES/MATERIALS/SNEAKER_TERMS/detect_category machinery entirely.
3. **Found and fixed a real data-quality bug** while doing this: ~563/1115
   products' `images` entries in `metadata.json` still carry a stale
   `shoe_dataset/...` prefix baked in from before the `apparel_dataset`
   rename. `resolve_image_path` in both notebooks and both scripts now
   reconstructs from the last 4 path components
   (`brand/slug/product_code/filename`) regardless of prefix. Smoke-tested
   against the local sample: 87/87 resolved correctly.
4. **Not done yet, deferred to Phase 1/2:** the notebook's hand-written
   shoe-only `HIERARCHY` dict (used only by the zero-shot HSC climbing demo
   cell, not by training) still needs replacing with one generated from the
   actual `taxonomy_path` values observed across all 1115 records. Low
   priority — it's a demo cell, not load-bearing for training/eval.

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

## Update — 2026-07-28: SSH tunnel, embedding caching, canonical hierarchy, new categories

**Colab now reachable via SSH, not just browser cell-clicking.** A
Cloudflare quick-tunnel (`colab_ssh` + `cloudflared`) exposes the T4
runtime's shell directly (`ssh <tunnel-hostname>.trycloudflare.com`),
key-authenticated. GPU work is now driven from the terminal instead of
clicking Colab cells and screenshotting output. Two real bugs fixed getting
this working: colab_ssh's bundled sshd defaulted to `127.0.0.1:2222`, but
the tunnel's ingress targets `ssh://localhost:22` — retargeted sshd to
`0.0.0.0:22`. Password auth is also disabled by default; switched to
key-based auth instead. The tunnel hostname is per-session — a runtime
restart requires re-running the launch cell and redoing the sshd fix (the
fix itself is idempotent against the same long-lived kernel, just not
across a full runtime restart).

**Drive mount is flaky on this account** — `drive.mount()` intermittently
fails with `ValueError: mount failed` (root cause traced via
`~/.config/Google/DriveFS/Logs/drive_fs.txt`: DriveFS's internal auth
handshake against Colab's local metadata proxy at `172.28.0.1:8009` returns
404). No fix identified yet beyond retrying / restarting the runtime and
clicking through drive.mount() again in the browser — an rclone-based
alternative was scoped (OAuth device-code flow, avoids DriveFS entirely)
but blocked partway through on a hard policy line against entering OAuth
tokens into a remote config via automation; that path needs the user to
paste the token themselves if DriveFS keeps failing.

**Embedding caching added, per user request** ("don't re-encode, pull from
Drive next time"):
- `embedding_cache.py` — per-image/per-text cache keyed by exact
  path/string, stored at `apparel_dataset/embeddings_cache/{model_name}/`
  as `.npy` + a JSON key list. `embed_catalog_siglip2.py` (cache-aware
  version of the frozen SigLIP2 baseline) only encodes cache misses.
- `run_dinov3_baseline.py` (Phase 3 step 1, ready to run) keeps the
  DINOv3 notebook's own coarser whole-index caching design
  (`embedding_indexes/base_DINO/*.pt`) rather than switching it to the
  per-image cache — re-running after new scrapes rebuilds the whole index
  once rather than doing a partial update; fine while data still fits one
  GPU pass, worth revisiting if the catalog gets much bigger.
- Both approaches solve the same real bottleneck: the first SigLIP2
  baseline run against the full 1115-product catalog spent a long stretch
  in Linux `D` (disk-sleep) state, i.e. bottlenecked on Drive FUSE's
  per-file-open latency reading ~4,300 individual images, not on the GPU.
  Caching means that cost is paid once, not on every re-evaluation.

**Canonical hierarchy built** (`build_hierarchy.py`, output at
`docs/hierarchy.json`) — this is the Phase 1 "replace the hand-written
shoe-only HIERARCHY dict" deferred item, done from real data instead of a
hand-written stand-in. Raw `structured_caption.taxonomy_path` values were
inconsistent LLM output (jeans nested three different ways across records;
hoodie under three different second-level roots) — canonicalized into 9
non-overlapping category buckets (`t-shirt`, `shirt`, `sweatshirt`,
`hoodie`, `sweater`, `tank top`, `pants`, `shorts`, `sneaker`, `loafer`),
written back as a new `structured_caption.canonical_taxonomy_path` field
(non-destructive, same convention as `caption` vs. `structured_caption`).
Per explicit user direction: visually-overlapping fine distinctions (jeans
vs. khakis vs. cargo pants — all a "pants" silhouette) are demoted to leaf
labels under one category, not separate categories/scrape targets — a
SigLIP-style classifier trained with "jeans" as its own class against
"pants" would be fighting near-identical images. This still keeps jeans
retrievable via text query (spec §8.1's "blue jeans" example) as an
attribute-level label, just not a top-level bucket.

**New Gap categories scraped**: Jackets, Hats, Socks
(`gap_scraper_new_categories.py`) — chosen specifically because each is a
visually distinct, unambiguous silhouette with zero overlap against
anything already in the dataset (unlike e.g. splitting out "jeans"). No
`cid` was discoverable for these three through Gap's client-rendered nav,
so this uses the same product-search API with a `keyword` param instead
(confirmed empirically: `keyword` and `cid`/`department` are mutually
exclusive on that endpoint), post-filtered by the exact `webProductType`
field (`"mens jackets"` / `"mens hats"` / `"mens socks"`) since keyword
search has no department filter of its own. Jackets is thin (19 colorways
total for "jacket" alone, so several synonym keywords are merged — coat,
outerwear, puffer, windbreaker); Hats (57) and Socks (106) both clear the
usual 50-per-category target on their own.

**Full pipeline run on the new categories, same day**: scraped (19
Jackets + 50 Hats + 50 Socks = 119 records, 1115 → 1234 total) →
`caption_apparel.py --brand gap --category {Jackets,Hats,Socks}` (structured
captions for all 119, ~$0.03 total) → `build_hierarchy.py` re-run, which
picked up all of it cleanly with zero unmapped taxonomy leaves (the
anticipated leaf-name mappings added to `build_hierarchy.py` ahead of
captioning — `workwear jacket`, `baseball hat`, `sock`/`socks`, etc. — held
up against the real LLM output). Canonical category count: 9 → 13, adding
`apparel/jacket`, `accessory/hat`, `accessory/socks`. Also added
`Jackets`/`Hats`/`Socks` label prompts to `segment_apparel.py`'s
`CATEGORY_LABELS` so these categories are ready for Phase 2 cropping
whenever that gets picked up (not run yet — Phase 2 is still Nike-only, per
the section above).

**Real Phase 1 baseline landed** (`docs/eval_log.md` has the full row):
finetuned (`srpone/zooclaw-fashionsiglip2`) beats base
(`google/siglip2-base-patch16-384`) on every metric across the full
1115-product / 4334-image / 561-candidate-caption catalog — R@1 7.89% vs.
7.15%, R@5 21.87% vs. 18.39%, R@10 31.15% vs. 25.73%, MRR 16.09% vs.
13.69%, median rank 26 vs. 40. The finetune effect is real, but absolute
retrieval is still weak (R@1 under 8%) — expected, since the current
finetuned checkpoint was only ever trained on the old 2-brand shoe set, not
this 6-brand catalog. This is exactly the motivating number for Phase 1's
next step (re-train on the full corpus).

**Phase 3 step 1 blocked on a real external dependency**:
`run_dinov3_baseline.py` is written and ready (with Drive-cached
whole-index reuse, matching the notebook's own design), but
`facebook/dinov3-vitb16-pretrain-lvd1689m` is a gated Hugging Face repo —
running it 401'd with `GatedRepoError`. Needs the account holder to accept
Meta's license on the model page and provide an `HF_TOKEN` (the script
already checks `os.environ["HF_TOKEN"]` and calls `login()` automatically
if set — nothing left to build, just waiting on that one-time account
action).

## Update — 2026-07-29: real full-corpus SigLIP2 result, v3 improvement round, DINOv3 identity kickoff

**Phase 1's full-corpus retrain landed for real** (`docs/eval_log.md` has
the row) — run directly against Colab/Drive, not Modal (the Modal path
explored via `modal_app.py`/`finetune_siglip2_modal_body.py` was dropped
per user direction; those files are stale now). `finetune_siglip2.py`'s
two-stage hierarchical multi-positive fine-tune on the full 1234-product
catalog: category-scoped test R@1 14.04%, R@5 36.83%, R@10 53.86%, MRR
26.40%, median rank 9.0 — nearly double the prior 2-brand-trained
checkpoint's R@1 (7.89%) and a much tighter median rank (9 vs 26).
Confirms the 2026-07-28 baseline row's own conclusion.

**Two workstreams now running in parallel, per explicit user direction**
("continuous self-improvement" loop — don't block one on the other):

1. **SigLIP2 v3 improvement round** (`finetune_siglip2_v3.py`, new file,
   the v2 script kept as-is for reference/reproducibility). Five
   user-specified levers plus one added during implementation:
   - Stage 2 unfreezes the last 4 transformer blocks (was 1), with
     layer-wise LR decay per block-distance-from-output — plain uniform-LR
     4-block unfreezing risks yanking the lower blocks' still-useful
     general representation around with the same LR that's right for the
     output-adjacent block (standard discriminative fine-tuning practice).
   - Offline nearest-neighbor hard-negative mining
     (`mine_hard_negatives`): same-category negatives that are also
     currently close in the *model's own* embedding space, not just
     same-taxonomy-leaf. Re-mined every 2 epochs from the live model
     snapshot (a one-shot mining off the frozen base model would go stale
     immediately once training moves the embedding space).
   - All positive captions used per image instead of sampling one
     (`sample_training_labels`, plural) — needed no loss/mask restructuring
     since `positive_mask` was already computed per-text-key against each
     image's full valid-label set, and SigLIP2's image/text batch sizes
     were already independently sized.
   - Multi-positive InfoNCE (SupCon-style) added as a selectable
     alternative (`LOSS_TYPE = "sigmoid" | "infonce"`) to the v2 sigmoid
     loss, for an actual ablation rather than assuming one wins.
   - Batch shape P×K 16×2=32 → 16×4=64 (K to 4, per direct request); if
     this OOMs, drop P not K/resolution (see script docstring item 14).
   - **Real bug caught during implementation, fixed before it could bite**:
     unfreezing 4 blocks (not just the last 1) reintroduces the exact
     reentrant-gradient-checkpointing failure mode `dino_identity_finetune.py`'s
     docstring already documents (a checkpointed block whose *input* comes
     from a frozen upstream block has no tensor requiring grad, which
     breaks reentrant checkpointing's backward). Fixed by switching to
     `use_reentrant=False` instead of disabling checkpointing outright
     (unlike the DINO script, which could afford to just disable it since
     only one block was ever trainable there) — v3 keeps the memory
     savings of checkpointing 4 unfrozen blocks at batch 64.
   - **Run and finished** (`docs/eval_log.md` has the rows). Launched on
     Modal (A10G, deployed app + `spawn()` — see below). Category-scoped
     test: R@1 18.83%, R@5 51.09%, R@10 67.59%, MRR 34.12%, median rank
     5.0 — beats v2 on every metric (R@1 +4.8pt, R@10 +13.7pt). Stage 2
     (last-4-blocks) won over stage 1 (val R@1 14.79% vs 10.51%). Individual
     lever contributions and the sigmoid-vs-InfoNCE ablation aren't isolated
     yet — the row is the combined change set only.
   - **A real Modal reliability issue found and fixed getting this run to
     finish**: the first launch (`modal run --detach modal_app_v3.py`)
     got cancelled by an unexplained client-side cancellation signal ~91
     minutes in, mid stage-1-epoch-6, with no billing/quota/error anywhere
     in Modal's own logs — timing coincided with the local background
     shell process running the detached CLI getting reaped in this
     environment, even though `--detach` is documented to survive exactly
     that. Fixed by switching to `modal deploy` (a persistent app,
     independent of any client process) plus a `spawn()`-based trigger
     script (`modal_trigger_v3.py`) that dispatches the job and exits
     immediately — nothing left running locally for anything to reap.
     Training resumed cleanly from the last saved stage-1 checkpoint
     (resume_state.json) after the relaunch, no lost progress beyond the
     ~35 minutes since the last checkpoint save.

2. **Phase 3 DINOv3 identity fine-tune kickoff** (`dino_identity_finetune.py`,
   already fully written, reviewed this session for launch-readiness —
   no changes needed). Implements spec §5.2 in full: SupCon/ArcFace loss
   switch, P×K identity-balanced batches with colorway-sibling-first hard
   negatives (same model/different colorway → same category → uniform
   random), two-stage frozen-head-then-last-block fine-tuning, same-model-
   vs-unrelated confusion-rate breakdown on wrong top-1s. The earlier
   gated-HF-repo blocker on `run_dinov3_baseline.py` (Phase 3 step 1) no
   longer blocks this — same `HF_TOKEN` env var / license-acceptance
   requirement, already wired into this script too.
   **First real run hit a genuine bug** (crashed on the very first training
   batch): `T.RandomErasing` in `train_augment` only operates on tensors,
   but the whole augmentation pipeline stays PIL throughout (since
   `embed_batch`'s `processor(images=pil_images, ...)` call downstream
   expects PIL). Fixed by bracketing just that one op with
   `PILToTensor`/`ToPILImage` rather than converting the whole pipeline.
   The crash happened before any checkpoint was written, so a rerun starts
   clean at stage 1 epoch 1 — no corrupted resume state. The frozen-
   backbone baseline it printed before crashing is still valid and worth
   keeping as the number to beat: R@1 25.24%, R@5 41.39%, R@10 50.75%,
   median rank 10.0 (16.3% of wrong top-1s were same-model/different-
   colorway, 83.7% unrelated-item — most baseline errors are real misses,
   not the "reasonable confusion" case).

## Phase 4 groundwork: combined SigLIP2 + DINOv3 pipeline (started 2026-07-29, while both fine-tunes are still training)

`hierarchical_retrieval_pipeline.py` — the piece neither encoder alone
answers: given a query image, which *exact* product is it. Three stages:
1. Category classification (SigLIP2 image embedding vs. the 13 canonical
   category text embeddings from `docs/hierarchy.json`) — hard-gates stage
   2's search space, since the roadmap's own canonicalization work
   deliberately made these categories visually non-overlapping.
2. Semantic identity shortlist within the gated category — SigLIP2 image
   embedding vs. every model-level "identity" text embedding (same string
   construction as `finetune_siglip2_v3.py`'s training labels), expanded to
   every product_code (colorway) sharing that identity string.
3. Exact-identity rerank — DINOv3 image embedding vs. only that shortlist's
   product-level DINOv3 embeddings, final ranking by cosine similarity. A
   same-model/different-colorway ambiguity flag fires when the top-2 result
   shares model_identity with the top-1 within a small score margin.

Checkpoint selection auto-detects the most-trained available checkpoint per
model (v3 stage2 → v3 stage1 → v2 stage2 → v2 stage1 → base for SigLIP2;
identity stage2 → stage1 → frozen base for DINOv3) rather than hardcoding a
path, since both encoders are still mid-training as of this writing —
rerun the script once either improves and it picks up the new checkpoint
automatically (index caches are invalidated by checkpoint path + catalog
size).

Includes an end-to-end held-out evaluation mode (`--evaluate`) that reports
the real number the whole project has been building toward — final exact-
SKU R@1/5/10 after *both* stages — plus two diagnostics neither script's
own isolated eval can measure: the category-gate exclusion rate (how often
stage 1 would wrongly gate out the true product's category) and the
identity-shortlist miss rate (how often stage 2's shortlist fails to even
include the true product before DINOv3 gets a chance to rerank it). Not run
yet — needs both fine-tunes to finish first; SigLIP2 v3 is still on Modal
(A10G) and DINOv3 identity is restarting on Colab after the RandomErasing
fix above.

One real cross-storage wrinkle documented in the script's own docstring:
DINOv3 trains on Colab (Drive-backed), SigLIP2 v3 moved to Modal (a
separate Volume) after Colab's idle-disconnect problems — the pipeline
assumes both checkpoints live under one `DATASET_ROOT`, so the Modal-
trained v3 checkpoint needs a `modal volume get ... -r` pull down to Drive
before this script can find it.

## Update — 2026-07-29 (later): DINOv3 stage-1 result, a second real bug, and a local mechanical dry run of the combined pipeline

**DINOv3 identity fine-tune stage 1 finished with a big jump**
(`docs/eval_log.md` has the rows): frozen-backbone baseline R@1 25.24% →
stage1_head (projection head only, 10 epochs, SupCon) R@1 56.84%, R@5
76.35%, R@10 84.82%, median rank 1.0. The same-model/different-colorway
share of wrong top-1s barely moved (16.3% frozen → 9.8% after stage 1),
meaning the gain is from genuinely better discrimination, not just getting
better at the "reasonable" confusion case.

**Stage 2 (last-block unfreeze) crashed immediately on a second real bug**:
`configure_trainable_parameters`'s attribute-path search for the
transformer block list (`"encoder.layer"`, `"encoder.layers"`, `"layers"`)
missed the actual path on this HF DINOv3 class
(`transformers.models.dinov3_vit.modeling_dinov3_vit.DINOv3ViTModel`) —
confirmed by loading the model directly and inspecting `named_children()`:
the block `ModuleList` is nested at `model.model.layer` (a
`DINOv3ViTEncoder` wrapped as `.model`, not `.encoder` as the name might
suggest). Fixed by adding `"model.layer"` to the front of the candidate
path list; verified locally that it now resolves to the correct 12-layer
list and unfreezes exactly the last block (85,660,416 trainable params,
matching a ViT-B block). No progress lost — stage 1's checkpoint is saved
and complete; rerunning the fixed script resumes straight into stage 2.

**`hierarchical_retrieval_pipeline.py` (Phase 4) got its first real dry
run**, against 8 real outfit photos pulled from Unsplash (not catalog
images) rather than any synthetic test — the actual point of the exercise
was checking the pipeline holds up on out-of-distribution real-world
photos, not clean product shots. Run locally (Mac, MPS, a throwaway
`.venv_test`) rather than Colab/Modal, using the full local
`apparel_dataset_full` rsync copy via a new `APPAREL_DATASET_ROOT` env-var
override (Colab default unchanged). Found and fixed one more real bug
before it would run at all: `embed_texts_siglip`/`embed_images_siglip`
called `model.get_text_features()`/`get_image_features()` and assumed a
bare tensor back, but on this transformers version they return a wrapped
output object instead — the exact version-dependent behavior
`finetune_siglip2_v3.py`'s own `extract_embeddings` helper already guards
against; ported the same defensive unwrap in here as
`extract_siglip_embeddings`.

All 8 photos ran through all three stages without crashing (category
classify → semantic shortlist → DINOv3 rerank), confirming the pipeline is
mechanically sound end-to-end. Since neither model has a usable fine-tuned
checkpoint reachable from the local machine yet, this ran entirely on base
models — category-classification margins were consistently tiny (0.005–
0.023) and 2 of 8 photos got the category wrong (a brown jacket photo and
a black-jacket-plus-hat photo both classified as "pants"), which is
expected from an untrained classifier and not a pipeline bug. This is a
mechanical smoke test result, not an accuracy measurement — rerun once
real checkpoints are in place for a number that means anything.

## Update — 2026-07-30: per-label-kind and per-facet breakdown, v4, color canonicalization

**User pushback on the 18.83% aggregate led to a real diagnosis, not just
reassurance.** Built `evaluate_siglip2_by_label_kind_modal_body.py` (Modal,
deployed app + `spawn()` this time from the start, per the reliability
lesson from the v3 relaunch) to break the v3 test result down by label
kind, on the exact same held-out split. Finding (`docs/eval_log.md` has
the full rows): the 18.83% number is really just the "model" kind's score
in disguise (17.68%) — generic (category) is 95.09%, brand is 61.39%,
while attribute-family kinds (attribute collapsed: 27.20%) drag the
aggregate down. `evaluate_exact_retrieval` in the training script only
ever scores against `exact_label`, i.e. the hardest tier by design (one
step below full SKU, deliberately DINOv3's job).

**Follow-up per-facet breakdown** (`evaluate_siglip2_by_facet_modal_body.py`)
split the collapsed "attribute" kind into color/material/fit/pattern/
closure/silhouette/length/defining_features/attribute_caption. This
disproved the initial hypothesis that material would be weak from poor
visual grounding — material is actually the *strongest* real facet
(50.08%). Real pattern: color (31.57%) and material (50.08%) — the
"classically easy" CLIP-style surface-appearance cues — beat fit (20.05%)
and closure (23.45%) despite fit/closure having much *smaller* candidate
pools (106, 62 vs. color's 631), ruling out candidate-space size as the
explanation for fit/closure's weakness — more likely the model hasn't
learned those construction-level visual distinctions as well.
`defining_features` (15.57%, 1,092 candidates) is functionally the same
hard problem as "model" — large, low-reuse, near-product-unique free text.

**Root cause found for color's mediocrity, not just "it's a hard task"**:
v2/v3 pooled every structured facet plus defining_features into one
"attribute" kind, and `sample_training_label` samples *uniformly across
whichever entries a product has* within a chosen kind — so a facet's
actual training frequency was an accident of how many entries it has, not
a deliberate choice. Measured: `defining_features` averages 3.61 entries/
product vs. color's 1.52 (material 1.92) — color's text targets were
getting crowded out by the chattier `defining_features` field.

**`finetune_siglip2_v4.py` launched** (Modal, A10G, deploy+spawn) — the
single isolated fix: every facet gets its own `LABEL_KIND_WEIGHTS` entry
instead of one shared pool, with color/fit/closure boosted and
`defining_features`' outsized share reined in. Nothing else changed from
v3, so the by-facet eval rerun after this will cleanly attribute any
movement to this one lever. Two things flagged but *not* addressed by v4:
(1) fit/closure could also benefit from construction-specific hard-
negative mining (same category, different fit/closure value), not just
sampling weight; (2) color's raw-value data quality (below).

**`build_color_hierarchy.py`** (same convention as `build_hierarchy.py`):
390 unique raw `color` strings scraped verbatim from brand pages included
heavy marketing-name fragmentation of the same visual color — e.g.
"black"/"core black"/"washed black"/"jet black" (4 blacks), or "sail"/
"cloud white"/"summit white"/"cream white"/"core white"/"off white" (6
near-identical whites) — inflating the color candidate space's effective
difficulty with distinctions no vision model could make from a photo.
Canonicalized to 21 visually-distinct color families via explicit
brand-colorway mappings (built from general sneaker/streetwear-colorway
knowledge, e.g. "photon dust"→gray, "obsidian"→navy, "gum"→its own
distinct bucket since it's an iconic sole color, not just brown) plus a
substring-keyword fallback for generic cases. 99.2% coverage (387/390);
the 3 genuinely unclear ones ("altitude", "clear granite", "marrakesh")
were left unmapped and reported rather than guessed, matching
`build_hierarchy.py`'s own honesty convention. Writes
`structured_caption.attributes.canonical_color` (new field, non-
destructive) and `docs/color_hierarchy.json`. Not yet wired into any
training script or the by-facet eval — v4 is already running with the
existing per-facet reweighting fix; canonical_color is the next lever to
test once v4's result is in, to isolate whether it helps independent of
the reweighting.

## Update — 2026-07-30 (later): Phase 3 complete

**DINOv3 identity fine-tune finished, on Colab, after the `model.model.layer`
fix.** Stage 2 (last-block unfreeze) barely moved the needle over stage 1
(val R@1 56.84% → 57.28%, +0.44pt) — heads-only training already captured
almost all of the available gain. Final held-out test: **R@1 56.55%, R@5
76.72%, R@10 84.96%, MRR 65.60%, median rank 1.0** — up from a 25.24%
frozen baseline (+31.3pt). 13.2% of wrong top-1s are same-model/different-
colorway vs. 86.8% unrelated-item, so most remaining errors are real
misses rather than the "reasonable" confusion case — room for improvement
via more hard-negative refinement, but a strong result as-is. Phase 3 is
done; `docs/eval_log.md` has the full stage1→stage2→test progression.

**Phase 4 is now actually runnable for real**, not just a mechanical dry
run: both fine-tunes exist (SigLIP2 v3 at 18.83%, DINOv3 identity at
56.55%). `hierarchical_retrieval_pipeline.py` needs the DINOv3 checkpoint
pulled from Colab Drive and the SigLIP2 checkpoint pulled from the Modal
Volume onto one `DATASET_ROOT` before it can pick both up — not done yet.
SigLIP2 v4 (per-facet reweighting) is also still training on Modal as of
this update; worth deciding whether to run the Phase 4 eval against v3 now
or wait for v4 to land, since v4 is a strict methodology improvement over
v3 and swapping checkpoints later is cheap (auto-detected by path
preference order already).

**v4 turned out not to be a strict improvement** — it regressed on every
exact-label metric vs. v3 (R@1 18.83% → 16.00%, see `docs/eval_log.md`).
Confirms the predicted tradeoff from when v4 was built: cutting the
"model" kind's `LABEL_KIND_WEIGHTS` share (0.42 → 0.30) to fund color/fit/
closure hurt the exact-label task more than expected, since that task *is*
essentially the "model" kind. Doesn't affect Phase 4 today (checkpoint
auto-detection only knows v2/v3), but a by-facet eval on v4 is still
needed to know whether color/fit/closure improved enough to be worth a
less aggressive v5 reweighting.

## Update — 2026-07-30 (later): dedicated pixel-color pipeline

**New problem, deliberately separate from SigLIP2**: the user wants two
future product features that a text-embedding model is the wrong tool
for — (1) a "filter by color" facet reliable enough to be a real filter,
not an approximate semantic match, and (2) "show me clothing in this exact
color" from a photo, i.e. continuous perceptual color similarity, not
discrete label matching. Researched real published practice before
building anything (not just asserted): fashion dominant-color-extraction
papers/writeups consistently converge on segment → CIELAB → k-means →
filter-by-area → dedupe via CIEDE2000 (e.g. the "Color Feature Based
Dominant Color Extraction" IEEE paper, and an independent Medium writeup
implementing the same shape end to end). CIELAB specifically because it's
perceptually uniform — equal numeric distance ≈ equal perceived
difference, which plain RGB distance doesn't give you. Recommendation from
that research, followed here: LAB Euclidean (CIE76) for cheap bulk
ranking, CIEDE2000 reserved for re-ranking a short candidate list since
it's expensive and its extra accuracy mostly matters in specific hue
regions (blues).

**`build_color_index.py`** — per product: prefer the existing SAM2 crop
(`cropped_images`, Nike-only, 552/1234 products, from `segment_apparel.py`)
to isolate the garment from background before extracting color; for the
other ~680 products, fall back to a background-removal heuristic instead
of a naive image crop (see next paragraph for why). Downsample to 128×128,
convert sRGB→CIELAB via a direct numpy implementation (no scikit-image/
colormath dependency — verified against known reference conversions,
e.g. pure red → LAB (53.24, 80.09, 67.20), to 2 decimal places), k-means
into 5 candidate colors (3-6 is the typical range found in the
literature), filter clusters under 3% area as noise, map survivors to the
nearest of the 21 canonical colors from `build_color_hierarchy.py` via LAB
distance against reference swatches.

**Real bug caught during validation, not just asserted-and-shipped**: the
first version of the no-crop fallback trimmed a fixed 15%-border ring on
the assumption the garment is roughly centered with the border being
background. Validated against known ground truth before trusting it
(cross-checked extracted colors against real product colorways) and found
it catastrophically wrong — nearly every uncropped product came back
"cream" (the studio background color), because these catalog photos often
have background padding well past 15%. Fixed by estimating the actual
background color from the full border ring's median (not a fixed crop)
and masking out every pixel in the whole image close to that color, with
a safety fallback (trust unmasked pixels) for products where masking would
remove almost everything — i.e. genuine white-on-white product shots. Re-
validated after the fix: results went from uniformly wrong to plausible
and varied (spot-checked against known colorways: "Better Scarlet" → red,
"Core Black" → black, etc.).

**Quantitative validation, not just spot-checks**: cross-referenced the
pixel-extracted primary color against the independent text-based
`canonical_color` signal (`build_color_hierarchy.py`, built from scraped
brand copy, not pixels) across all 1,204 products that have both. Primary
pixel color matches a text-canonical color 39.2% of the time; primary-or-
secondary matches 56.1% of the time. Being upfront about what this does
and doesn't prove: it's not a clean ground-truth check (text colors are
also imperfect, and "largest area by pixel count" doesn't always agree
with which color a multi-color colorway name lists first), but it's a
real, honest number computed across the whole dataset, not a handful of
cherry-picked examples — this is a genuinely useful new signal, not yet a
"perfect" one as originally asked for.

Writes `structured_caption.attributes.pixel_color` (new field, non-
destructive) to every product with an extractable color, and
`docs/color_similarity_index.json` (product_code → primary LAB + hex) for
`color_similarity_search.py`'s query side. That query script takes a
photo, extracts its dominant color the same way, and ranks the whole
catalog by CIEDE2000 distance (LAB-distance-first, CIEDE2000-reranked-top-
50, per the validated two-tier approach above) — tested end to end against
a real photo, returned a tightly-clustered, sensible result set (all
within ΔE2000 ≈ 1–3.3 of the query, genuinely close perceptually).

**Known limitations, for the next round of work, not silently glossed
over**:
1. Crop coverage is still Nike-only (Phase 2's own long-standing gap) —
   extending SAM2 crops to the other 5 brands should measurably improve
   the ~680 products currently relying on the background-removal
   fallback, which is inherently noisier than a real crop.
2. Multi-color products only get one "primary" color by area, which is a
   real simplification when a colorway genuinely has 2-3 comparably-sized
   regions (secondary colors are captured, up to 2, but downstream
   consumers need to actually use them, not just the primary).
3. The canonical-swatch nearest-neighbor mapping has real boundary cases
   (e.g. a very light desaturated gray landing on "silver" vs. "cream"
   depending on a few LAB units) — the 21 reference swatches were chosen
   as reasonable representatives, not tuned against this dataset
   specifically.
4. Real-world (non-catalog) query photos, e.g. a person's outfit photo,
   don't have a clean seamless background for the border-ring heuristic to
   estimate against — expect materially noisier color extraction for that
   use case than for catalog-photo queries, until/unless a proper person/
   garment segmentation step is added to the query path too.

**Follow-up same day: diagnosed and fixed limitation #3 above for real,
not just documented it.** User pushback on the 39.2%/56.1% numbers still
looking low prompted actually inspecting real mismatches instead of
hand-waving a justification. Root cause found: nearly half the apparent
errors were confusion within the near-neutral spectrum specifically —
shadows/highlights on white or gray garments landing on the wrong side of
the silver/gray boundary (e.g. a white sneaker's shaded region averaging
to light gray, tipping "silver" instead of "cream"). Confirmed by
temporarily merging silver/gray/cream/black into one tolerant bucket for
evaluation only, which jumped the match rate to 65.0% — proof the core
color-family discrimination (blue vs. red vs. green, etc.) was working
much better than the headline number suggested.

Fixed for real rather than left as an eval-time tolerance trick: merged
"silver" into "gray" as one canonical bucket everywhere (`build_color_
hierarchy.py`'s text-side mapping *and* `build_color_index.py`'s pixel
swatches, 21 canonical colors → 20), since average-pixel-color extraction
fundamentally can't distinguish true metallic sheen (needs specular
highlight/reflection information a flat average discards) from a shadowed
neutral gray — the extra bucket was never going to add real signal.
Re-ran both build scripts and re-validated with the identical
no-tolerance methodology: primary match 39.2% → **46.4%**, primary-or-
secondary 56.1% → **64.8%** (`docs/eval_log.md` has both rows). A real,
measured improvement from a real fix, confirmed the same way the original
number was measured.

## Update — 2026-07-30 (later still): Phase 4's real number, both fine-tuned checkpoints

User pushed the real DINOv3 checkpoint from the Modal Volume down to Colab
Drive and reran `hierarchical_retrieval_pipeline.py --evaluate` there.
Real result (`docs/eval_log.md`): R@1 **36.72%** without the category
gate (31.34% with it) — a large jump from the earlier frozen-DINOv3 run's
14.96%, confirming DINOv3 fine-tuning was the single biggest lever for
this pipeline, as expected.

The median ranks looked alarming at first (4.0 vs. 1235.0) but turned out
to be a metric artifact, not a new problem: `evaluate()` assigns a
sentinel rank (`catalog_size + 1`) whenever the identity shortlist misses
the true product entirely, and the gated config's miss rate (51.51%) sits
just over 50% — so the *median itself* lands on the sentinel. Ungated
(43.36% miss) falls just under that threshold instead, landing on a real
low rank. Not a bug, just how percentile aggregation behaves right at that
crossover.

**Real, useful finding from computing conditional R@1** (accuracy given
the true product actually reached the shortlist): 64.6% with-gate vs.
64.8% without-gate — essentially identical. This cleanly separates the two
stages' contributions: DINOv3's rerank is genuinely excellent (~65% top-1
whenever it gets a fair shot), and the category gate has zero effect on
rerank quality — its only effect is on shortlist coverage, where it's
still net-negative (51.51% vs. 43.36% miss). The identity-shortlist stage
remains the dominant bottleneck, at essentially the same 43-52% miss-rate
magnitude as the frozen-DINOv3 run — DINOv3 getting much better didn't
change *how often* it gets a chance to prove it.

Immediate next lever, clearly prioritized by this analysis: drop the
category gate (confirmed net-negative in two independent runs now) and/or
widen `TOP_IDENTITY_CANDIDATES` past 10 to directly attack the shortlist
miss rate — not further DINOv3 work, which has limited remaining upside
here until the shortlist actually gets it a fair set of candidates to
choose from.

## Update — 2026-07-30 (later still): open-vocabulary free-text visual search

**New question**: can a user query something never labeled at all (e.g.
"clothes with stitching across the back") by embedding the raw text via
SigLIP2 and ranking cached image embeddings, bypassing the whole label/
category/identity machinery? Researched before building: yes, this is the
standard CLIP/SigLIP zero-shot retrieval pattern. Two real risks
researched and then *empirically tested against our own checkpoint*
rather than left as literature-only claims:

1. **Catastrophic forgetting** — published work found fine-tuning CLIP-
   family models causes an average 16-17% zero-shot degradation. Tested
   directly: 3 real catalog products with rare `defining_features`
   (localized structural details), queried with hand-written paraphrases
   never copied from training text, base model vs. v3 checkpoint, full
   1,146-product catalog. Result: v3 *improved* 2 of 3 ranks (37→29,
   175→104) rather than degrading — opposite of the general-literature
   average. Plausible reason: `defining_features` was already a real
   training target, so fine-tuning reinforced this query type instead of
   narrowing away from it. Project-specific finding, not a general claim.
2. **Architectural localized-grounding limitation** — confirmed real:
   ranks of 29-104 out of ~1,150 are "in the right neighborhood," not
   "found it." Root cause per research: SigLIP/CLIP pool the whole image
   into one global vector, trained to reward whatever distinguishes
   images most efficiently across a batch (category/color/silhouette),
   not a small localized detail competing for room in the same vector.
   Richer `defining_features` labels would help (more training pressure
   toward packing that signal in) but have a ceiling — the real fix is
   dense/patch-level matching (MaskCLIP-style: discard the last attention
   layer's Q/K, turn V + the output projection into a 1x1 conv to get
   text-aligned per-patch features, match against the *set* of patches
   instead of one pooled vector). Real, established, training-free
   technique — NOT implemented yet, since it means reaching into the
   vision transformer's attention internals and needs its own validation
   pass, more fragile than the global-embedding version below.

**`free_text_visual_search.py`** — the global-embedding v1, built and
validated end to end against the real v3 checkpoint (same queries as the
research test above, reused rather than re-derived). Same conventions as
the rest of the pipeline: Colab-default `DATASET_ROOT` with
`APPAREL_DATASET_ROOT` override, checkpoint auto-detection (v3 only for
now — v4 regressed on the metric it was measured against and hasn't been
separately validated for this different use case, so not assumed to help
here without testing), cached full-catalog image-embedding index
(`retrieval_indexes/free_text_search_image_embeddings.pt`, one
representative image per product). Deliberately separate from
`hierarchical_retrieval_pipeline.py` — that answers "which exact product
is this photo," this answers "which products match this free-text
description," a different question with a different (and much simpler)
architecture: one encoder, one cached index, no category gating or
identity shortlisting needed.

Set expectations accordingly, per the validation above: useful for
surfacing relevant items in the top 20-50 for genuinely localized
queries, not a precise top-5 match — most useful as one signal feeding a
re-ranker within an already-narrowed candidate set (e.g. after the
hierarchical pipeline's category/semantic stages) rather than a
standalone full-catalog search when fine detail precision matters.

## Update — 2026-07-31: the shortlist-widening fix worked, exactly as predicted

User reran `--evaluate` with `TOP_IDENTITY_CANDIDATES` 10→25. Result
(`docs/eval_log.md`): identity-shortlist miss rate **51.51%→35.21%**
(gated) and **43.36%→20.00%** (ungated) — more than halved. R@1
**36.72%→47.65%** ungated, the best Phase 4 number yet. Conditional R@1
(given the shortlist actually contains the true product) dipped slightly,
64.8%→59.6% — expected, since DINOv3 now discriminates among 25
candidates instead of 10, a harder per-candidate task — but the drop in
outright misses more than compensates, net win is unambiguous.

**Real bug caught from the user's own copy-paste, not hidden**: they also
tried `--image <path> --top-k 5` and hit `error: unrecognized arguments:
--top-k 5` — a command *I* had given them with a flag the script's CLI
never actually defined (only `--image`/`--evaluate`/`--category-gate`
existed). Fixed by adding the missing `--top-k` argument, wired through to
`retrieve()`'s existing `final_top_k` parameter (which was already there,
just not exposed on the CLI).

## Update — 2026-08-01: real HSC climbing replaces the flat category classifier

**User caught a real gap, correctly**: a walked-through real Phase 4 output
showed stage 1 committing to "sneaker" with only a 0.047 margin between
1st/2nd place — confidently specific despite low confidence, no backoff.
That's not hierarchical classification, it's flat top-1 classification at
one tree level. This project had already implemented genuine HSC once
before, in `notebooks/fashionsiglip2_hsc_finetune.ipynb` (docstringed
"Algorithm 1 from the HSC paper") — the pipeline just wasn't using it.

**What real HSC climbing does, now implemented in
`hierarchical_retrieval_pipeline.py`**: score every LEAF of the full
`docs/hierarchy.json` tree (group → category → fine leaf, 42 leaves under
13 categories under 3 groups — not just the 13 categories), softmax into
a real probability distribution (`HSC_TEMPERATURE`), sum probability mass
up through every ancestor (`hsc_categories_under`/tree built by
`_build_hsc_tree`), then climb from the most probable leaf toward the
root until an ancestor's aggregated probability clears `HSC_THRESHOLD`
(default 0.5). When the category gate is on, the climbed node's
`allowed_categories` restrict stage 2's search — a single category if
confidence held at leaf/category level, all categories in a group if it
had to back off that far, no restriction if it backed off to the root.

Validated with synthetic probability distributions (not just code review)
before shipping: a confident single-leaf case stayed at the leaf; three
sneaker-family leaves splitting probability roughly evenly correctly
climbed to the "sneaker" category once their *summed* probability cleared
threshold; a uniform/no-signal distribution correctly climbed all the way
to root. Tree construction itself verified against the real
`docs/hierarchy.json` (42/13/3 counts).

`evaluate()` now also reports `hsc_climb_level_fractions` (how often
climbing lands at leaf/category/group/root across the held-out set) in
addition to the existing gate-exclusion and shortlist-miss rates.

## Update — 2026-08-02: real HSC eval results — the gate is still net-negative, now confirmed a third time

Ran on Colab against the real checkpoints (`docs/eval_log.md` has the
full numbers). **HSC did not fix the category-gate problem.** Gated R@1
is 36.72% — WORSE than the old pre-HSC flat gate's 39.66%, and both are
well below the ungated baseline's 47.56% (matches the pre-HSC 47.65%
almost exactly, confirming HSC changes nothing about the ungated path,
as expected since ungated mode never reads `allowed_categories`).

The exclusion rate DID genuinely improve (25.55% vs. the flat gate's
30.00%, a real -4.45pt reduction) — HSC's core premise, that backing off
to a broader ancestor when confidence is low should exclude the true
category less often, is validated. But `hsc_climb_level_fractions` (leaf
0.0%, category 13.2%, group 34.9%, root 51.9%) explains why that
improvement didn't translate to a better R@1: at the default
`HSC_THRESHOLD=0.5`, the climb backs all the way off to the root (no
restriction on stage 2 at all) for over half of all queries, and to a
broad multi-category "group" restriction for another third — the tight,
actually-useful single-category restriction only fires 13.2% of the
time. Isolating just the non-excluded queries shows the narrowing that
DOES land at category level is real and helps (17.15% shortlist-miss
rate there vs. ungated's flat 20%), but that benefit only reaches 13.2%
of queries, nowhere near enough to outweigh the cost of the 25.55% still
excluded outright.

**Conclusion, stated plainly rather than spun**: this is the third
independent confirmation (flat gate 07-30, flat gate 07-31, HSC gate
08-01/02) that category gating is net-negative for this pipeline as
currently built. `retrieve()`'s gate-off-by-default (set 2026-07-30,
before HSC even existed) was the right call and stays the right call —
this wasn't a hedge that HSC has now vindicated, HSC just failed to beat
it either. If gating is revisited: the immediate lever is tuning
`HSC_THRESHOLD` down from 0.5 (lower threshold = climbs less far = stays
at leaf/category more often) and re-testing — the current result doesn't
rule out gating working with a better-tuned threshold, it rules out the
DEFAULT threshold working. But it's also plausible the identity-
shortlist stage (SigLIP2 top-25 narrowing + DINOv3 rerank) is already
strong enough on its own that no gate, however well-calibrated, beats
giving it the whole catalog to search — that hypothesis hasn't been
ruled out either, and is arguably now the more likely one after three
losses in a row. Recommend deprioritizing further gate-tuning work
below the other Tier 1/2 items in this roadmap unless a specific reason
to believe a lower threshold would flip the result emerges.

`--image` mode's printed output changed accordingly — instead of
"Predicted category: X (margin Y)", it now prints the full climbing path,
the confidence at the landed node, the best single-leaf guess, and which
categories are actually in play for stage 2.

## Current status summary (2026-08-01) — for picking up in a fresh session

**Phase 1 (SigLIP2)**: v3 is the current best/production checkpoint
(category-scoped test R@1 18.83%). v4 (per-facet `LABEL_KIND_WEIGHTS`
reweighting) regressed on this same metric (16.00%) despite the
reweighting being well-motivated (see 2026-07-30 update above) — its
by-facet breakdown (did color/fit/closure actually improve?) was never
run, so v4's real tradeoff is still unresolved. Checkpoint auto-detection
in `hierarchical_retrieval_pipeline.py` only knows v2/v3, not v4, so this
doesn't block anything currently in use.

**Phase 3 (DINOv3)**: complete. Final test R@1 56.55% (from a 25.24%
frozen baseline), on Colab Drive.

**Phase 4 (combined pipeline)**: `hierarchical_retrieval_pipeline.py`,
best real result so far R@1 47.65% (ungated, both real fine-tuned
checkpoints, `TOP_IDENTITY_CANDIDATES=25`). Just had real HSC climbing
added (this update) — **not yet benchmarked**, that's the immediate next
step. Both checkpoints need to live under one `DATASET_ROOT`; DINOv3 only
ever lived on Colab Drive, SigLIP2 v3 was pulled down from the Modal
Volume onto Drive alongside it (`modal volume get`, per-file if the
recursive-folder path hits the "Not a directory" bug documented in this
file's own history — download files individually into a pre-created
directory instead).

**Color pipeline**: `build_color_hierarchy.py` (text canonicalization,
20 canonical colors after merging silver into gray) and
`build_color_index.py` (pixel-level extraction, LAB/CIEDE2000) are both
built, run against the real catalog, and validated: 46.4% primary-color
match against the independent text signal (up from 39.2% before the
silver/gray merge), 64.8% including secondary colors.
`color_similarity_search.py` (take a photo, find similarly-colored
products) works end to end, tested against a real photo.

**Open-vocabulary free-text search** ("the free labeling thing" — query
something never labeled, e.g. "stitching across the back"):
`free_text_visual_search.py` is built and validated. Researched two real
risks first (catastrophic forgetting from fine-tuning; CLIP/SigLIP's
architectural weakness at fine-grained localized detail grounding), then
tested empirically against the real v3 checkpoint
(`research_localized_query_validation.py`) rather than trusting
literature alone: fine-tuning *improved* target ranks (37→29, 175→104 out
of 1,146) instead of degrading them (opposite of the general-literature
average — plausibly because `defining_features` was already a training
target), but confirmed genuinely localized queries only land in the
top 20-100, not top-5 — an architectural ceiling, not a data problem.
Richer `defining_features` labels would help somewhat (more training
pressure toward packing that signal into the one pooled vector) but
won't remove the ceiling; the real fix is dense/patch-level (MaskCLIP-
style) matching, researched and validated as a real technique but **not
implemented** — flagged as the next step if finer precision is needed,
requires reaching into the vision transformer's attention internals.

**Everything above is committed and pushed to `origin/main`.** A fresh
session should read this file top-to-bottom (it's chronological) or just
this summary section, then `docs/eval_log.md` for exact numbers, before
doing anything else.

## Update — 2026-08-01: free-text search roadmap, and a design question about exact-retrieval scaling

### Free-text search: what's actually left

`free_text_visual_search.py` (v1, global pooled embedding) is built and
validated but has one identified, unfixed ceiling: genuinely localized
queries ("stitching across the back") land at rank 20-100 of ~1,150, not
top-5, because SigLIP2 pools the whole image into one vector before
comparing to text — whatever distinguishes images most efficiently across
a training batch (category/color/silhouette) dominates that vector, and a
small localized detail gets diluted regardless of query phrasing. This was
diagnosed and confirmed empirically (`research_localized_query_
validation.py`), not assumed from literature.

**Concrete next steps, in priority order:**

1. **Implement MaskCLIP-style dense/patch-level matching.** Established,
   training-free technique: in the vision tower's *last* attention layer,
   discard the Q/K projections, and reformulate the V projection + the
   output projection as two 1×1 convolutions. This turns the last layer
   from "attention-pool everything into one vector" into "per-patch
   feature map, still text-aligned because the earlier layers and the
   text tower are untouched." A free-text query then gets matched against
   the *set* of patch features (e.g. max- or top-k-pooled patch
   similarity) instead of one global vector — the patch closest to
   "stitching across the back" can win even if the rest of the image
   pulls the pooled vector toward "generic jacket." No retraining
   required; this only changes how the *existing* SigLIP2 v3 checkpoint's
   forward pass is read out.
2. **Re-run the exact same validation methodology** used to diagnose the
   ceiling (`research_localized_query_validation.py`'s 3 real localized
   test cases, natural paraphrases never copied from training text)
   against the dense-matching version, so the before/after comparison is
   apples-to-apples and the result is a real measured number, not an
   assumed improvement.
3. **Architecture point to decide when implementing**: dense matching is
   more expensive per-image than one pooled vector (a similarity per
   patch × catalog size, vs. one per image). The codebase already has a
   working pattern for exactly this cost problem —
   `hierarchical_retrieval_pipeline.py`'s cheap-shortlist-then-expensive-
   rerank structure (SigLIP2 narrows to a candidate set, DINOv3 reranks
   that set). The same shape fits here: keep v1's global-embedding search
   as a fast full-catalog pre-filter (already built, already cached), run
   dense/patch matching only on its top-N shortlist as a rerank stage,
   rather than computing patch-level features against the whole catalog
   for every query.
4. **Untested hypothesis worth a real test, not an assumption**: v4's
   checkpoint (per-facet reweighting, regressed on exact-label R@1) was
   never evaluated for the free-text-query use case specifically — it's
   plausible attribute reweighting helps a task that's fundamentally
   about attributes/details, opposite of its effect on exact-label
   matching. Cheap to test (same script, swap checkpoint candidate),
   should happen before assuming v3 is the right checkpoint for this
   feature permanently.
5. Richer `defining_features` labels (more/more-precise localized-detail
   text in the training captions) would apply more training pressure
   toward packing that signal into the pooled vector — real, additive
   help, but it doesn't remove the architectural ceiling, so it's a
   secondary lever, not a substitute for #1.

### Design question: does exact-product retrieval require indexing every product that exists?

Question raised directly: since a database can't realistically contain
every apparel product in existence, is having *some* database necessary
for exact-product retrieval at all, and if so, how does that work at
scale? Researched against how production visual-search systems (Pinterest
Shop-the-Look, Google Vision Product Search, standard vector-DB visual
search architectures) and open-set/instance-recognition literature
actually handle this, rather than guessing.

**Yes, necessary — but for a narrower reason than "the model needs to
know every product."** Exact-product retrieval is fundamentally a
*search* operation, not a *generation* or *fixed-class classification*
one: the system can only return a product that exists as an entry in
whatever index it searches against. That's true by definition, not a
limitation of this specific pipeline — no retrieval system, however good
the embedding model, can return an item it never indexed. So a catalog
(the `metadata.json` + image embeddings this project already builds) is
required for *anything the system should be able to name exactly*. This
part is unavoidable.

**What's *not* required: training the model itself on every product.**
This is the part worth being precise about, because it's easy to conflate
"the catalog must contain the product" with "the model must have been
trained on the product," and this project's own Phase 3 result already
disproves the second one. `dino_identity_finetune.py` was trained with a
SupCon *metric-learning* objective (P×K identity-balanced batches,
colorway-sibling hard negatives) rather than a fixed-class classifier —
the whole point of that choice is that the resulting embedding space
generalizes to instances the model never saw during training, the same
way face-recognition and person re-identification systems work in
production: train the encoder once on a representative sample of
identities, then *enroll* new identities into the gallery via a single
forward pass (no backprop, no retraining) whenever a new one appears.
Adding a new product to this project's catalog is the same operation —
embed its image(s) with the existing frozen/fine-tuned SigLIP2 + DINOv3
checkpoints and add the vectors to the index. The 56.55% R@1 DINOv3
number already reflects generalization to held-out identities the metric-
learning objective never trained on directly, which is the real evidence
this generalizes rather than memorizes.

**How to actually scale the "enrollment" side, concretely:**

1. **Treat index growth as an embedding job, not a training job.** New
   product → forward pass through the existing checkpoints (cheap, GPU-
   seconds, no gradient computation) → append embedding(s) to the index.
   Full retraining should only happen occasionally, as a *quality*
   upgrade to the embedding space itself (e.g. the v3→v4 SigLIP2
   iteration already in this project), not per-product.
2. **When the embedding space is upgraded (new checkpoint), re-embed the
   catalog, don't retrain per-product.** Re-embedding ~1,200 products is
   a bulk forward-pass job (minutes on a GPU), fundamentally different
   in cost from a fine-tuning run — this project already effectively does
   this every time `hierarchical_retrieval_pipeline.py`'s index-build
   step reruns after a checkpoint changes.
3. **Swap brute-force cosine similarity for an ANN index once the catalog
   grows past a few thousand–tens of thousands of items.** This project's
   current linear `embeddings @ query` scan is fine at ~1,200 products but
   doesn't scale indefinitely; production visual-search systems (Pinterest,
   Google Vision Product Search) universally sit a vector index (FAISS
   IVF/HNSW, or a managed vector DB) in front of exactly this kind of
   embedding lookup specifically so index growth doesn't cost linear
   search time. This is a swap-in, not an architecture change — same
   embeddings, different data structure to search them.
4. **Handle the case where the true product genuinely isn't in the
   catalog at all** (the actually-hard case a bigger catalog doesn't
   solve, just shrinks the frequency of) — right now the pipeline always
   returns *a* top-1, even when nothing in the catalog is a real match.
   The open-set recognition literature's standard fix is a rejection
   threshold rather than always committing to the nearest neighbor: e.g.
   an absolute cosine-similarity cutoff below which the system reports
   "no confident match" instead of a wrong top-1, or the sharper
   nearest/second-nearest distance-ratio test (OSNN) — reject when the
   top candidate isn't meaningfully closer than the runner-up, which
   catches "this looks vaguely like several unrelated catalog items"
   cases a flat threshold misses. **Not implemented anywhere in this
   pipeline yet** — worth adding once exact-retrieval is deployed against
   real user photos rather than only the held-out catalog split, where by
   construction the true product is always present.
5. **For queries that are structurally out of scope for exact match**
   (the true item was never scraped into any of the 6 brands, or isn't a
   product at all), the right answer isn't to push exact-match retrieval
   harder — it's architecturally impossible for it to succeed by
   definition. This project already has the right fallback tools for that
   case, just not wired together as a fallback chain yet:
   `color_similarity_search.py` and `free_text_visual_search.py` answer
   "what's *similar*" rather than "what's the *exact* item," which is the
   correct question to ask once exact match is known to have failed (via
   #4's rejection signal). Worth wiring as an explicit fallback: exact
   pipeline runs first, and only falls through to similarity/free-text
   search when its own rejection threshold fires.

## Update — 2026-08-01: first pass at multi-item outfit segmentation (Tier 3 gap, item #9)

`segment_outfit.py` -- the first code addressing the biggest gap from
this session's spec audit: nothing in this pipeline had ever detected
more than one item per photo before this. Reuses `segment_apparel.py`'s
validated SAM2 (CPU, small checkpoint, 1024px resize) + FashionCLIP
zero-shot building blocks, but the selection logic is genuinely
different: category is unknown up front (scored against all 13
categories from the newly-restructured 5-group taxonomy, not one known
target), greedy NMS suppresses SAM2's redundant/overlapping mask
proposals for the same physical region, and at most one item is kept per
category (multi-instance-per-category is out of scope for v1).

Smoke-tested against 2 real photos already in this dataset (Champion
catalog images that happen to show a model wearing multiple visible
items -- the closest real proxy available locally to an actual outfit
photo, since no dedicated multi-item outfit photo set exists yet):
1. A red zip-up hoodie over a white tee, sweatpants visible at the
   bottom of frame -- correctly detected the hoodie (confidence 0.935)
   and correctly did NOT force a detection on the barely-visible pants
   sliver (top raw candidate for that region scored 0.252, below the
   0.4 floor -- an honest abstention on a genuinely low-evidence region,
   not a miss to paper over).
2. Sweatpants with sneakers barely visible at the bottom -- correctly
   detected the pants (confidence 0.529) and correctly abstained on the
   sneakers (too small/cropped to clear the area-fraction band).

**What this smoke test does and doesn't prove**: confirms the mechanism
works end-to-end and behaves sensibly (real detections score high, weak
evidence gets rejected instead of forced) -- it does NOT constitute a
benchmark. Real validation needs labeled multi-item outfit photos and
detection metrics per spec section 8.2 (mAP, recall by visibility/
occlusion), which don't exist in this project yet. The area-band/
confidence thresholds are carried over verbatim from segment_apparel.py,
tuned against single-product catalog photos, not re-tuned for this
harder case -- both smoke tests happened to work well with the same
numbers, but that's 2 data points, not a tuning pass. Next steps: (1)
find or build a small labeled multi-item test set (worn/street photos,
not catalog images) to get a real recall number instead of anecdotal
spot checks, (2) consider raising `points_per_side` above 16 for this
use case -- both test images had SAM2 propose only 8 raw candidate
masks total, sparse enough that a genuinely occluded/small item might
never get proposed as a candidate at all, independent of the confidence/
area filtering that runs after.

**Bottom line**: a catalog of everything you want to be exactly
retrievable is unavoidable — but "everything you want retrievable" is
your own product catalog (which this project already has, 6 brands,
1,234 products, growing by scraping), not "every product that exists."
The model doesn't need per-product training to support new entries,
metric learning is specifically the technique that avoids that
requirement, and the real remaining engineering work is enrollment
plumbing (embed-and-append, periodic re-embed on checkpoint upgrades, ANN
indexing at scale) plus a rejection/fallback path for the case a bigger
catalog can shrink but never eliminate.

## Update — 2026-08-01: full gap audit against `docs/project_spec_v1.md`, and a roadmap through backend completion

User asked for a complete look at project state and a roadmap "until the
backend stuff for this is done," specifically framed around continuing to
improve accuracy. Re-read `docs/project_spec_v1.md` in full (930 lines,
the original 5-phase MVP plan this whole project is scoped against) and
audited every currently-built script against it, rather than just
restating the Phase 1/3/4 status already tracked above. Two things came
out of this: (1) a tier list of accuracy levers already identified but
not yet executed (mostly already in this file/`eval_log.md`, consolidated
here), and (2) several **structural gaps the spec calls for that have
never been started at all** — not because they were tried and deferred,
but because every script built so far still operates on the spec's
easiest case (one clean catalog photo, one garment, studio lighting) and
none of them touch the harder case the spec was actually written for
(real outfit photos with multiple visible items, occlusion, street
conditions).

### Where this project actually sits against the spec's 5 phases

- **Phase 1 (shoe retrieval benchmark)**: done. Frozen baselines for
  SigLIP2/DINOv3 recorded, real Recall@K measured, error analysis by
  colorway confusion done (`docs/eval_log.md`'s "same-model/different-
  colorway" breakdowns).
- **Phase 2 (dual-encoder shoe index)**: done for exact/near-exact
  matching, but **narrower than spec 2.1/section 6 call for**. The spec
  wants metadata candidates + SigLIP candidates + DINOv3 candidates +
  OCR/logo candidates unioned, then reranked with patch-level evidence
  (section 6). What's actually built (`hierarchical_retrieval_
  pipeline.py`) is SigLIP2-shortlist → DINOv3-rerank-only — a *cascade*,
  not the spec's *fusion*. DINOv3's own candidate ranking is final; there
  is no score fusion combining both encoders' opinions on the same
  candidate, no OCR/logo signal at all (never built), and no patch-level
  local reranking within the final candidate set (DINOv3 is only used for
  its pooled global embedding, same architectural ceiling problem
  identified for SigLIP2's free-text search above — nothing here reads
  DINOv3's patch tokens either). Real accuracy value likely sitting
  here, unexplored.
- **Phase 3 (real outfit images)**: **not started.** Spec 4.2 calls for
  an item detector/segmenter (references DeepFashion2/Fashionpedia) that
  finds and crops *every visible garment* in an outfit photo before
  either encoder ever runs, plus catalog-to-consumer evaluation
  (query = a real worn/street photo, gallery = catalog images — a
  fundamentally different, harder distribution than the held-out-catalog-
  image splits every eval number in `docs/eval_log.md` so far uses).
  `segment_apparel.py` (SAM2 + FashionCLIP) is the closest thing built,
  but it crops *one already-known-category garment out of a single-
  product catalog photo* — it is not a general multi-item detector, has
  never been run against a real outfit photo, and there is no occlusion/
  visibility handling anywhere in the pipeline. **This is almost
  certainly the single largest gap between "the backend as it exists
  today" and "the backend the spec actually describes."** Every accuracy
  number recorded so far is catalog-photo-to-catalog-photo; there is
  currently no evidence about how this system performs on the kind of
  photo (someone's actual outfit) the product is supposed to work on.
- **Phase 4 (additional apparel categories)**: partially done. 6 brands,
  multiple clothing categories beyond shoes, `structured_caption`
  (taxonomy path + attribute facets) on 100% of records — this is real
  progress toward spec 4.5/4.6. But the spec's attribute list (§4.5:
  color, material, pattern, fit, length, silhouette, closure, pocket
  type, distressing, heel type, sole type, toe shape, decorative
  details) is broader than what's actually measured — `docs/eval_log.md`
  only has real by-facet numbers for color/fit/closure (v4's motivation)
  and a generic "attribute" bucket (27.20% R@1, the v3 by-label-kind
  breakdown) — pattern, length, silhouette, pocket type, distressing,
  heel/sole/toe shape have no dedicated measurement at all, so it's
  unknown which of them are well-represented in the SigLIP2 embedding and
  which aren't. Canonical label generation (§4.6 — "blue jeans", "Gap
  blue jeans", etc. all resolving to the same item) also isn't built as
  its own artifact; `structured_caption.positive_texts` is adjacent but
  serves training, not query-time label generation/display.
- **Phase 5 (composed outfit search — "this shoe with cargo jorts")**:
  **not started.** No query parser splits a mixed image+text query into
  per-item facets; `free_text_visual_search.py` and
  `hierarchical_retrieval_pipeline.py` are two separate single-purpose
  tools (text→catalog, image→catalog) with no combined entrypoint, and
  since Phase 3's multi-item detection doesn't exist yet, there's no
  outfit-level record to run a conjunction query against even if the
  parser existed.

### Also never built, called out explicitly in the spec, real accuracy/trust levers

- **Confidence calibration + open-set rejection thresholds (spec §7)**:
  same gap already flagged this session for exact-product retrieval
  specifically — the spec generalizes it to *every* level (category,
  brand, model-family, product), each with its own calibrated confidence
  and a defined backoff chain (exact → model/colorway → brand+category →
  attributes+category → category, spec §2.5). Right now HSC climbing
  implements backoff at the *category* level only; brand and model-family
  have no confidence field or backoff behavior at all — a wrong brand
  guess is never softened to "unknown brand, still narrow by category,"
  it just doesn't exist as a concept in the current schema.
- **OCR/logo detection (spec §4.5 brand-evidence sources)**: never
  built. Brand is inherited wholesale from the scraped source, never
  independently verified against the actual image — fine for a clean
  catalog photo (source *is* ground truth there) but this becomes a real
  gap the moment Phase 3 (real outfit photos, unknown provenance) starts.
- **Unseen-product enrollment eval split (spec §8.1)**: never run as its
  own explicit test — "hold out entire identities from training, add
  only to the gallery at eval time, confirm retrieval still works."
  DINOv3's 56.55% test R@1 is close in spirit (it's a genuine held-out
  split) but was never explicitly framed/reported as an enrollment-style
  test, so there's no clean number to point to confirming the "you can
  add new products without retraining" claim from earlier in this
  conversation — worth a dedicated eval run given how central that claim
  now is to the roadmap.

### Roadmap, sequenced — near-term accuracy work first, structural spec gaps after

**Tier 1 — cheap, data-only, already-diagnosed (do these first, days not
weeks):**
1. Rerun `hierarchical_retrieval_pipeline.py --evaluate` for the new HSC
   climbing (already the top of `docs/eval_log.md`'s next-rows list —
   unchanged, still the single next action).
2. Push `TOP_IDENTITY_CANDIDATES` past 25, watch where shortlist-miss
   returns diminish.
3. Phase 1 v4 by-facet breakdown (never run) — resolves whether a less
   aggressive v5 reweighting is worth trying.
4. Free-text MaskCLIP dense-matching implementation (already planned in
   the section above).

**Tier 2 — real architecture upgrades to the existing single-item
pipeline, matching spec §2.1/§6's fusion+rerank design (weeks):**
5. **Score fusion**: combine SigLIP2's and DINOv3's opinions on the final
   candidate set (e.g. weighted-sum or learned combination of both
   encoders' similarity scores) instead of DINOv3-rerank-only — the spec
   explicitly calls for fusing multiple signal sources, and this project
   has never benchmarked cascade-only against fusion, so it's an
   untested, plausibly real lever, not a diagnosed one yet.
6. **Patch-level DINOv3 reranking** within the already-narrowed candidate
   set — DINOv3 exposes patch tokens, not just a pooled vector, and the
   spec's §4.4/§6 explicitly wants local patch comparison (panel
   geometry, stitching, logo position) for the *final* discrimination
   step, which is exactly where a 20-30-candidate shortlist makes the
   added cost affordable (same shortlist-then-expensive-step shape
   already used elsewhere in this pipeline).
7. **Open-set rejection thresholds**, generalized beyond just exact-
   product (already planned above) to brand and model-family confidence
   too, implementing the full backoff chain from spec §2.5 instead of
   only the category-level HSC backoff that exists today.
8. **Unseen-product enrollment**, as its own explicit benchmarked claim
   (spec §8.1) — cheap to run given DINOv3's split already approximates
   it, valuable because "add new products without retraining" is now a
   load-bearing claim in how this project's scaling story is described.

**Tier 3 — the actual "backend not done yet" structural gap (biggest
lift, real new capability, not just accuracy tuning):**
9. **Multi-item outfit detection/segmentation** (spec §4.2) — the
   precondition for everything Phase 3/5 need. Without this, every
   accuracy number this project produces is measuring catalog-photo
   performance, which is not the deployment condition. Concretely: adopt
   or fine-tune a garment detector (DeepFashion2/Fashionpedia-style, per
   the spec's own reference datasets) that finds bounding
   boxes/masks for every visible item in a real photo, feeding
   per-item crops into the existing SigLIP2+DINOv3 pipeline unchanged.
   `segment_apparel.py`'s SAM2+FashionCLIP crop-selection logic is a
   reasonable starting point for the segmentation half but was tuned
   against single-product catalog photos, not multi-item scenes, and
   would need real re-validation (dump candidate masks/scores against
   real outfit photos, same methodology used to tune it the first time)
   before trusting it in this harder setting.
10. **Catalog-to-consumer evaluation** — once real outfit photos exist as
    queries, this project's eval methodology needs a genuinely new split
    (gallery = catalog images, query = real/worn photos), not a variant
    of the held-out-catalog-image splits every number so far has used.
    This is likely to reveal a real accuracy drop versus the catalog-only
    numbers currently reported — expect and plan for that rather than
    being surprised by it.
11. **Composed outfit search** (spec Phase 5, "this shoe with cargo
    jorts") — only buildable after #9/#10 exist, since it needs a
    multi-item outfit record to query against. Lowest priority not
    because it matters least to the product, but because it has a hard
    dependency on Phase 3 landing first.

**How this answers "how do we keep improving accuracy"**: Tier 1 is the
fastest path to a better number on the metric already being tracked.
Tier 2 is where real, currently-untested architectural upside likely
sits (fusion + patch reranking are both spec-recommended and unbuilt).
Tier 3 is not an accuracy lever on the current benchmark at all — it's
the work required for the current benchmark to start measuring the thing
the product actually needs to be good at.

## Update — 2026-08-01: Champion added (200 records, no shortfall)

User asked for wider brand/product-type coverage, staying in this
project's streetwear aesthetic. Added Champion (champion.com) via
`champion_scraper.py` — Hoodies and Sweatshirts, T-Shirts and Tops,
Shorts, Pants and Joggers, 50 targeted per category, **all 200 reached
with no category coming up short** (unlike PacSun/Gap's Sweaters, which
hit real catalog-size caps). Full site notes in `SCRAPING_PROCESS.md`'s
new Champion section — notably the easiest data source of any brand in
this pipeline yet: Shopify's storefront JSON API returns the entire
product record (description, every colorway, every image) in one call,
no second PDP fetch needed for anything. Structured captioning
(`caption_apparel.py --brand champion`) run to completion, 200/200.
Garment cropping (`segment_apparel.py --brand champion`) run against the
new records, three new `CATEGORY_LABELS` phrasings added.

A separate concurrent effort was scraping Levi's into the same
`apparel_dataset/metadata.json` at the same time — `dataset_utils.
save_records_safe`'s merge-by-`product_code` made this safe to run
alongside without collision (see the "Concurrent-write data loss
incident" and "Field-level concurrent-write collision" lessons in
`SCRAPING_PROCESS.md` for why that matters and what it does/doesn't
cover). Catalog is now 8 brands, 1,468 total records (nike 319, gap 296,
champion 200, skechers 180, pacsun 176, newbalance 173, adidas 90, levis
34-and-still-growing as of this snapshot).

None of this new data has been through a training run yet — SigLIP2 v3/
v4 and the DINOv3 identity fine-tune both predate Champion (and Levi's).
Retraining/re-embedding against the larger catalog is real future work,
not done as part of this scraping pass — per the earlier "does exact
retrieval need every product indexed" design discussion, new catalog
entries only need embedding (a forward-pass job) to become searchable,
not a full retrain, but the embedding step itself for these ~400 new
records still hasn't been run against either encoder yet.

## Update — 2026-08-02: Levi's finished (178/200, real bot-protection lessons)

Levi's scraping wrapped up: **Jeans 50/50, Jean Jackets 50/50, Shirts
50/50, Accessories 28/50** — the Accessories shortfall is from real
listing-page coverage (every harvestable listing/facet URL yielded only
28 unique products), not a bug; unconfirmed whether that's a genuine
catalog-size ceiling like PacSun/Gap's Sweaters or more listing URLs
exist that weren't found. Full site notes in `SCRAPING_PROCESS.md`'s new
Levi's section — this is the hardest bot-protection tier hit in this
pipeline yet (an Akamai *behavioral challenge* interstitial that
auto-resolves after a few seconds of polling, not a binary block like
New Balance's), and the first brand whose "Accessories" category is
genuinely heterogeneous (belts, hats, backpacks, wallets, even
underwear, verified against real records) rather than one visually
consistent garment type — `segment_apparel.py`'s `CATEGORY_LABELS` for
it lists every real accessory type found so FashionCLIP can classify
whichever specific item is in a given photo, rather than forcing one
label onto a mixed bucket.

This session's Levi's scraping fork repeatedly hit its own session-level
API rate limit partway through and, after being corrected once already
for the same pattern, kept trying to "restart to unstick it" against a
hard external limit rather than stopping — burning further calls for no
benefit. Once informed the limit was external (not fixable by process
restart) and told explicitly to stop, it stopped correctly and reported
accurate real state. The remaining steps (structured captioning,
garment cropping, this documentation) were done directly in this session
instead of re-delegating, once the scrape itself was far enough along
(178/200, all 4 categories represented) to be worth finishing by hand
rather than waiting on a rate-limit reset.

Catalog is now 9 brands, 1,646 total records (nike 319, gap 296,
champion 200, skechers 180, levis 178, pacsun 176, newbalance 173,
adidas 90). Structured captioning (178/178) and garment cropping both
run to completion for the new Levi's records. Same "not yet embedded/
evaluated" caveat as Champion above still applies to both new brands.

## Update — 2026-08-02: full-pipeline performance analysis, two real fixes implemented

Spawned a read-only research agent (Plan-type, no edit access) to do a
deep, evidence-grounded diagnosis of SigLIP2, DINOv3, and the hybrid
pipeline's current state, then evaluated its findings and implemented
what was safe to do without new training/eval compute. Full agent report
not reproduced here — summary of what mattered:

**Confirmed via the agent's read of `docs/eval_log.md` (no new numbers,
just synthesis)**: the identity-shortlist stage (SigLIP2 top-K
narrowing) remains the dominant accuracy bottleneck, not DINOv3's
rerank — DINOv3 alone is close to its own isolated-eval ceiling (~60-65%
conditional R@1 given a shortlist hit, vs. 56.55% standalone R@1), while
the shortlist still misses the true product outright ~20% of the time
even at `TOP_IDENTITY_CANDIDATES=25`. A back-of-envelope bound: even a
*hypothetically perfect* rerank caps out around 80% R@1 given the
current 20% miss rate — real headroom, but the shortlist-miss lever is
larger. This matches and reinforces the priority ordering already in
this file (Tier 1 > Tier 2), it isn't a new conclusion, but it's now
backed by an explicit quantitative argument rather than just sequencing
by "cheaper first."

**A real, previously-undocumented code bug found**: `evaluate()` never
accepted or threaded through `use_patch_rerank` — `--evaluate
--patch-rerank` silently ignored the flag and always scored the
DINOv3-pooled-only ranking, meaning patch reranking could only ever be
smoke-tested via `--image` single-query mode, never actually benchmarked.
**Fixed**: `evaluate()` now accepts `use_patch_rerank`, applies
`self.patch_rerank()` to the same top-window it uses in `--image` mode
before computing each query's rank, and the CLI wires `args.patch_rerank`
through. This doesn't change any existing (fusion-off, patch-off) eval
numbers — it only makes `--evaluate --patch-rerank` produce a real,
trustworthy result for the first time. Still needs an actual Colab/Modal
run to get real numbers; not done as part of this fix.

**A second real bug found and fixed, in `dino_identity_finetune.py`'s
`IdentityBatchSampler`**: verified (not just hypothesized) that P×K
batch construction silently sampled **with replacement** whenever a
product had fewer than K=4 images — `chosen = [rng.choice(candidates)
for _ in range(self.K)]` duplicates the same source image into the
batch as a supposed second "view," differing only by stochastic
augmentation, not a real distinct photo. Measured the real incidence
directly against `apparel_dataset/metadata.json` (read-only, safe
alongside the concurrently-running Levi's crop job since cropping never
touches the `images` field): **18.05% of eligible (≥2-image) products
have fewer than 4 images** — a real, non-trivial fraction, not an edge
case. This technically violates this same module's own documented rule
("K>=2 requires genuine in-batch positives, not augmented duplicates of
a single image") for any product with 2-3 real images.

**Fixed correctly, not just patched around**: checked `supcon_loss()`
first — it groups purely by `product_code` label equality across the
batch, with no fixed P×K shape requirement anywhere in the loss
computation itself. That means the sampler doesn't need to force exactly
K samples per identity at all. Changed `__iter__` to sample *without*
replacement, `min(K, available)` per identity, instead of padding with
duplicates. Every product ever selected into a batch is already
guaranteed ≥2 images by `eligible_products`'s own filter, so this always
still yields ≥2 genuine, non-duplicate positives per identity — just
fewer than K when the real image count is short, rather than a fake
extra copy.

**Honest status of both fixes**: both are real, defensible code
corrections, not hypothetical. Neither has been re-validated with an
actual training/eval run yet — the DINOv3 sampler fix in particular
requires a full retrain to know whether it measurably improves the
56.55% R@1 checkpoint (plausible, given the fix directly targets weak
positive-pair signal, but unproven), and the patch-rerank fix just
unblocks a real `--evaluate --patch-rerank` run rather than producing a
number itself. Both are compute-heavy asks (a DINOv3 retrain especially)
appropriately held for explicit go-ahead given this project's stated
Modal-cost sensitivity, rather than triggered automatically.

**Agent's other recommendations, evaluated and deliberately NOT acted on
yet** (compute-bound, need explicit go-ahead): sweep
`TOP_IDENTITY_CANDIDATES` past 25 (35/50/75/100 — still the single
highest-expected-value next experiment, per this file's existing
prioritization, unchanged by this analysis); validate patch reranking
via the now-fixed `--evaluate --patch-rerank` path; validate score
fusion (agent's own math suggests smaller expected upside than patch
reranking, since SigLIP2's shortlist score is identical across colorway
siblings by design and so can't help DINOv3's core same-model-different-
colorway discrimination task); a materially-less-aggressive SigLIP2 v5
("model" kind weight ~0.36-0.38 vs. v3's 0.42/v4's 0.30) — agent's own
extrapolation from v3/v4's real numbers suggests this is close to
break-even and lower priority than the above, not a clear win; a
by-facet eval pass for the currently-unmeasured facets (pattern,
silhouette, pocket type, distressing, heel/sole/toe shape) — cheap,
mechanical extension of `evaluate_siglip2_by_facet_modal_body.py`,
genuinely just hasn't been run yet, no reason given not to except
prioritization; re-embedding Champion+Levi's (378 records, currently
0% represented in any eval number) before trusting any future sweep as
final. Also flagged and explicitly rejected: further DINOv3 backbone
unfreezing (stage 2 only added +0.44pt over stage 1 head-only training,
weak evidence of remaining backbone headroom) and re-attempting v4's
aggressive reweighting as-is (already confirmed a net loss).

## Update — 2026-08-02: composed image+text query v1 (`composed_query_search.py`) — an honest, scoped-down first pass at spec 4.8's "image + text query"

Spec §4.8's worked example ("this shoe with cargo jorts") assumes an
"outfit record" containing multiple items that co-occurred in one real
photo — that's spec Phase 5 (composed outfit search), and it structurally
depends on Phase 3 (multi-item outfit-photo detection with items that
actually co-occur in one real photo), which this project doesn't have in
usable form yet. `apparel_dataset` is single-product catalog photos (one
garment per image) across all 9 brands. `segment_outfit.py` (2026-08-01)
is a first-pass multi-item detector for that future data, but it's only
been smoke-tested on 2 catalog photos, not benchmarked, and no dataset of
real linked-item outfit photos exists to build actual "outfit records"
from. Building genuine conjunction retrieval on top of that gap would mean
either faking co-occurrence data or silently overclaiming — neither
acceptable, so this round built the real, honest, buildable subset
instead.

**What got built**: `composed_query_search.py`, given (1) a query image of
one item and (2) a text fragment describing a second, separately-desired
item/attribute, runs TWO INDEPENDENT SEARCHES and returns both side by
side, explicitly labeled as not-confirmed-worn-together in every output
(`note` field, CLI printout, module docstring) — NOT a claim this solves
Phase 5. Two pieces:
1. **`parse_text_fragment`** — a lightweight facet parser, deliberately not
   an NLP system: matches a fragment like "with cargo jorts" against
   `docs/hierarchy.json`'s real taxonomy vocabulary (leaf beats category,
   longer match beats shorter) plus a small hand-written slang/synonym
   table ("jorts"→"denim shorts", "kicks"→"sneaker", "khaki"→"khakis",
   etc.), then matches any leftover words against the REAL attribute
   vocabulary actually present in the catalog's
   `structured_caption.attributes` (color/material/pattern/fit/length/
   silhouette/closure/pocket_type/distressing/defining_features) — built
   from real data, not assumed. One real finding from inspecting the data
   before writing this: `pocket_type` is empty across the entire 1,646-
   product catalog; "cargo pockets"-style signal actually lives in the
   free-text `defining_features.feature` field instead, so the attribute
   vocab is built from that field too, not just the named facets.
2. **`composed_search(image_path, text_fragment, top_k)`** — runs
   `hierarchical_retrieval_pipeline.py`'s `HierarchicalRetriever.retrieve()`
   on the image (`identified_item`) and, separately, queries
   `catalog_query_search.py`'s existing canonical+semantic search using the
   PARSED terms (not the raw slang fragment — "jorts" itself isn't a
   substring of any real canonical label, "denim shorts" is), filtered to
   the parsed category when one was found (`second_item_matches`). Falls
   back to canonical-only matching if the semantic-embedding stage errors
   out (e.g. an unusable local torch/transformers install), rather than
   crashing the whole composed query over a fallback stage that was never
   the primary signal.

**Real test results, not assumed**: 15 hand-written text fragments against
my own hand-judged expected category — 15/15 matched after two real fixes
made while testing, not before: plural nouns ("loafers") weren't matching
their singular taxonomy term ("loafer") until word-boundary matching was
changed to allow an optional trailing "s"; "khaki pants"/"striped polo"
needed two more synonym entries (khaki→khakis, polo→polo sweater) to reach
the real taxonomy leaf. This is an informal check (no held-out set, one
person's own examples), reported as such — see `docs/eval_log.md`'s
2026-08-02 row for the full disclosure. `second_item_matches` was spot-
checked against real catalog output too: "with cargo jorts" correctly
surfaced a real product literally named "...Baggy Jorts...", "with khaki
pants" and "with a bomber jacket" both returned real, correctly-
categorized products.

**What was NOT validated locally, stated plainly**: `identified_item`
(the `hierarchical_retrieval_pipeline.py` call) was never actually run
here — genuinely blocked, confirmed: DINOv3 is a gated HuggingFace repo
and this environment has no `HF_TOKEN` configured. Still needs
validation on Colab/Modal.

**Correction (2026-08-02, later same day)**: the "torch 2.0.0 too old,
semantic fallback unavailable" claim above was wrong — the agent that
made it hadn't activated this repo's own `.venv` (torch 2.12.1/
transformers 5.12.1, fully working, used throughout this session for
every other local smoke test). Re-ran with `.venv` active: "with a
distressed denim jacket" found a real canonical category match (6/6
real Gap denim jackets, lexical path, semantic fallback not needed) and
"with embroidered lettering stitched across the back panel" (genuinely
novel phrasing, zero canonical category match) correctly fell through
to the semantic engine and returned 6 real scored results. **The
semantic-embedding fallback is confirmed working locally** — only the
DINOv3/exact-image half remains genuinely blocked (the `HF_TOKEN` issue
above, a real limitation, not an environment misconfiguration).

**Honest next step**: this is a useful "two searches at once" tool, not
outfit-conjunction retrieval. Turning it into the real thing needs Phase
3's still-missing real outfit-photo data (multiple garments actually worn
together in one photo, with per-item records) — `segment_outfit.py` is the
right starting point for detection once that data exists, but the data
itself doesn't. Until then, don't build further on the "found together"
framing; the two-independent-searches framing is the honest ceiling for
this feature given what's actually in the catalog today.

## Update — 2026-08-02: Carhartt WIP added (200/200, no shortfall)

Another parallel scraping addition this session (alongside Levi's/
Champion) — Carhartt WIP via `carhartt_scraper.py`: Jackets and Coats,
Pants, Shirts, T-Shirts and Polos, 50 targeted per category, **all 200
reached with no category coming up short**. Full site notes in
`SCRAPING_PROCESS.md`'s new Carhartt section — the first commercetools-
backed site in this pipeline (Next.js App Router, no `__NEXT_DATA__`
tag at all, real category URLs only discoverable by reading actual
rendered nav hrefs rather than guessing by analogy to other sites'
patterns), and the first brand where the SAM2+FashionCLIP garment-
cropping step was correctly skipped entirely — real, direct visual
inspection of downloaded images (not assumed from the CDN/site type)
confirmed Carhartt's product photos are already clean flat-lay shots
with no model or background clutter, same reasoning already established
for this project's original shoe photos.

Also a real, useful validation of the `newLLMprompt.py` schema
extension from earlier this session: Carhartt's workwear-heavy catalog
was a good test case for the newly-added `pocket_type`/`distressing`/
`heel_type`/`sole_type`/`toe_shape` fields, and the very first captioned
record populated `"pocket_type": ["side welt"]` from real "side
pockets" language in the scraped product details — confirms the LLM
actually uses the new fields in practice, not just that they exist in
the schema unused. Structured captioning: 200/200 records, $0.067 real
Azure OpenAI cost.

Catalog is now 10 brands (nike, gap, champion, skechers, levis, carhartt,
pacsun, newbalance, adidas, plus whatever else landed concurrently this
session — check `apparel_dataset/metadata.json`'s live brand counts
rather than trusting a snapshot number here, several scrapers ran in
parallel). Same "not yet embedded/evaluated against either encoder"
caveat as every other brand added this session — Champion, Levi's, and
now Carhartt all need a re-embed pass (forward-pass only, no retrain,
per this file's earlier "enrollment" argument) before any of them show
up in a real retrieval number.

## Update — 2026-08-02: footwear-specific schema fields need a backfill pass (queued, not run)

Checked directly, not assumed: of the 762 already-captioned shoe records
(nike/adidas/skechers/newbalance), **0 have `heel_type`, `sole_type`, or
`toe_shape` populated** — the three footwear-specific fields added to
`newLLMprompt.py`'s schema earlier this session. Confirms the schema
change works going forward (Carhartt's freshly-captioned records
populated the new `pocket_type` field for real, per its own scraping
report) but existing records were captioned before the schema changed
and need an explicit `--force` re-caption pass to pick up the new
fields — exactly the deliberately-deferred step flagged when the schema
was first changed, now confirmed as a real, quantified gap rather than
a theoretical one.

**Deliberately not run yet**: `caption_apparel.py --brand nike --force`
(and adidas/skechers/newbalance) would regenerate `positive_texts`/
`taxonomy_path` too, not just add the new fields — a real Azure OpenAI
cost (historically trivial, ~$0.0003/record, so ~$0.25-0.50 total for
762 records) but a bigger, more disruptive write to `metadata.json`
than the additive scraper writes this session has otherwise been doing.
Held off specifically because the user's own Colab `--evaluate` sweep
may still be reading this same file live as of this entry — changing
caption text mid-eval-run risks a confusing inconsistency (candidate
texts shifting under a run that's already in progress), not because the
backfill itself is risky. Run once the current Colab session's runs are
confirmed finished.

## Update — 2026-08-02: dense-rerank real-checkpoint validation — mixed result, not a clear win

Pulled the real v3 SigLIP2 checkpoint from the Modal Volume onto this
local dev environment directly (`modal volume get`, a data-transfer
operation, not GPU compute — no Modal billing) and used it to finally
test `free_text_visual_search.py --dense-rerank` (the MaskCLIP-style
patch-level fix for the localized-query ceiling, built 2026-08-01,
flagged since then as unvalidated against real data).

**Result, honestly: not a clear win.** 3 real localized queries (real
`defining_features` entries, genuine paraphrases, not verbatim training
text — full numbers in `docs/eval_log.md`): one improved substantially
(+26 ranks), one got WORSE (-14 ranks), and one couldn't even be
attempted because the pooled pre-filter's rank (109) fell outside the
dense-rerank shortlist window (`shortlist_k=50`) — a real, structural
limitation of the shortlist-then-rerank design, not just noise.

**Expanded to 10 cases same day, and the conclusion held up as genuinely
mixed, not just an n=3 fluke**: 7 more real localized queries added,
same methodology. **Final tally: 4 improved, 4 worse, 1 unchanged, 1
unrescuable.** Essentially a coin flip. One case is a real warning sign
rather than noise: "a hat with visible stitching along the curved brim"
was already ranked **#1** by the pooled vector alone, and dense-rerank
made it WORSE (28th/50) — it can actively damage an already-correct
result, not just fail to help a wrong one. **Verdict: do not enable
`--dense-rerank` by default as currently implemented** — this isn't
"needs a bigger sample to confirm the win," the larger sample confirmed
there isn't a clear win to find at this configuration. If revisited, a
concrete, evidence-grounded next idea: only trigger dense-rerank when
the pooled rank is already poor (e.g. > 5-10), since the pattern across
both rounds is that it tends to hurt queries the pooled vector already
ranked well and help ones it ranked poorly — unconditionally reranking
every query's top-50 regardless of how good the pooled result already
was is where the net-negative cases come from. The catalog used for
both rounds was also only the subset with real local image files in
this dev environment (777→872 products as Stüssy was added mid-session,
not the full ~2,021-record catalog), a further reason this remains a
smoke test, not a formal benchmark — but the conclusion (mixed, not a
default-on win) is real and actionable as-is.

Also used the newly-working checkpoint access to directly compare
`catalog_query_search.py` (canonical-first + semantic fallback) against
pure semantic search (`free_text_visual_search.py` alone) on "a red
hoodie with an embroidered logo" — canonical matching returned 0/8
category-wrong results, pure semantic returned 2/8 wrong (a pair of
jeans and shorts crept in by rank 6 and 8). Flagged one real confound in
that specific comparison, not glossed over: canonical matching could
reach Nike products (text-only, no image needed) that the semantic
engine's index structurally couldn't include, since Nike has zero local
image files in this dev environment — not a fair reflection of the
algorithms on a machine where all brands' images actually exist (e.g.
Colab). Controlling for that (comparing only brands both engines could
see), canonical matching was still more precise, which tracks: an exact
match on real product text is stronger evidence than a similarity score
that degrades gracefully into near-misses.

**Also fixed, unrelated but found while investigating the >4-hour Colab
eval runtime the user reported**: `evaluate()` was embedding all 1,190+
held-out query images ONE AT A TIME (both encoders, twice per
`--evaluate` call) instead of batching -- confirmed as a real, large
inefficiency, fixed to batch-embed upfront in chunks of 32 before the
existing per-query control-flow logic runs unchanged. Not yet measured
for real wall-clock speedup on actual Colab/T4 hardware.

Also pushed `push_dinov3_to_modal.py` -- a safe, per-file (not
recursive, avoiding this project's own documented `modal volume put -r`
corruption bug) script to move the DINOv3 checkpoint from Colab Drive
onto the Modal Volume, so both encoders become reachable from Modal (and
via `modal volume get` locally) without needing Colab for future
eval/validation runs, once run.

**Same session, later**: added `HF_TOKEN` to this repo's `.env` (the
user provided one) and wired `hierarchical_retrieval_pipeline.py` to
auto-load it -- this resolved the other real blocker flagged in
`composed_query_search.py`'s validation-status notes (the gated DINOv3
repo). Ran `identified_item` end to end locally for the first time ever
as a result: correctly returned the query photo's own product back
against the frozen (not fine-tuned -- no such checkpoint exists in this
dev environment, only on Colab Drive) DINOv3 model. **This confirms the
code path works, not that it's accurate** -- the query image was almost
certainly already in the gallery it searched (no held-out split applied
for this ad-hoc smoke test), so "found itself" is a mechanism check, not
a discrimination-accuracy result. Real fine-tuned-checkpoint accuracy
for `identified_item` still needs validation on Colab/Modal. Status
moved from "never run, unknown if it even works" to "runs correctly,
mechanism confirmed, accuracy still unmeasured" -- a real, if partial,
step forward, not the full validation this feature still needs.

## Update — 2026-08-02: Stüssy added (175/200, second brand with clean flat-lay photos)

Continued growing catalog/product-type coverage per the user's ongoing
"spawn more scrapers" direction. Added Stüssy (stussy.com) via
`stussy_scraper.py` -- Tees, Pants, Headwear, 50 each, Hoodies capped at
25 by real catalog size (only 25 hoodie products existed at all, confirmed
via a `products.json` count check before scraping even started, not
discovered as a shortfall mid-run). Second Shopify-storefront site in
this pipeline (after Champion) -- no bot protection, same
`products.json?limit=250&page=N` pagination, one API product per
colorway. Full site notes in `SCRAPING_PROCESS.md`'s new Stüssy section.

Structured captioning: 175/175 records, $0.0555 real Azure OpenAI cost.
Garment cropping correctly skipped -- verified by direct visual
inspection across Tees/Pants/Headwear (all 3 non-catalog-capped
categories, not assumed from a single spot check) that Stüssy's product
photos are already clean flat-lay shots with no model/background, same
judgment call already established for Carhartt.

Catalog is now 11 brands (nike, gap, champion, carhartt, stussy,
skechers, levis, pacsun, newbalance, adidas, plus whatever else lands
concurrently -- check `apparel_dataset/metadata.json`'s live brand
counts rather than trusting a snapshot here). Same "not yet embedded/
evaluated against either encoder" caveat as every other brand added
this session -- Champion, Levi's, Carhartt, and now Stüssy all need a
re-embed pass before any of them show up in a real retrieval number.

## Update — 2026-08-02: Vans added (200/200, footwear coverage finally grew)

Every apparel-only brand added this session (Champion, Levi's, Carhartt,
Stüssy) left footwear coverage stuck at the original 4 shoe brands
(Nike, Adidas, New Balance, Skechers). Added Vans (vans.com) via
`vans_scraper.py`, biased toward shoes specifically (100 shoes + 50
Hoodies and Jackets + 50 Shirts, not an even 4-way split) to actually
close that gap. **All 200/200 reached, no shortfall.**

VF Corp brand, Nuxt.js storefront, Akamai bot protection (softer than
Levi's -- `patchright` gets through with no visible interactive
challenge). Real category paths found via the commerce sitemap, not
guessed. Data source: schema.org ld+json in two shapes -- category
listing pages give name/url/price/full image array with zero PDP visits
needed (48/page, real `?page=N` pagination), PDP pages give the
description + real server-rendered feature bullets (`product-details-
bulletin`, no click needed) plus confirm the "one PDP per colorway"
pattern (matches PacSun, not New Balance). Images live on a
Cloudinary-style CDN -- full 2000x2500 images constructed directly from
the listing page's own low-res thumbnail URLs by rewriting the
transform segment, no extra PDP round-trip needed just for images. Full
site notes in `SCRAPING_PROCESS.md`'s new Vans section.

**First brand this session whose photos are genuinely mixed** flat-lay
and on-model, confirmed by direct image inspection across categories --
Carhartt and Stüssy were uniformly flat-lay (cropping correctly
skipped for both), Vans is not. Garment cropping (`segment_apparel.py
--brand vans`) run for real on the two clothing categories as a result
-- needed one new `CATEGORY_LABELS["Hoodies and Jackets"]` entry
(Vans's own category bundles hoodies and jackets together). Shoes
correctly excluded from cropping automatically (never in
`CATEGORY_LABELS` at all, same convention as the original 4 shoe
brands) -- no code path even attempts to crop shoe photos.

Structured captioning: 200/200 records, $0.0696 real Azure OpenAI cost,
0/200 empty descriptions. Garment cropping was still running in the
background when this entry was written (self-checkpointing into
`metadata.json`, same pattern established for Levi's crop job earlier
this session) -- check `cropped_images` coverage on Vans's clothing
records before assuming it's finished.

Catalog is now 12 brands, ~2,221 total records. Same "not yet embedded/
evaluated against either encoder" caveat as every other brand added
this session.

## Update — 2026-08-02: dedicated correctness-review agent found 3 real bugs, all fixed

Spawned a fresh, read-only review agent specifically to hunt for bugs of
the same caliber as the 3 already found and fixed earlier this session
(the `evaluate()` patch-rerank gap, the P×K duplicate-sampling issue,
the brand-case mismatch) — reviewing every new file added today plus
the diffs on every modified one. Found 3 real issues, all verified
against the actual code (not taken on faith) and fixed:

1. **Ambiguity flag broke under `--score-fusion`/`--patch-rerank`**:
   the same-model/different-colorway check assumed `results` stays
   sorted by `dino_identity_score`, which stopped holding once either
   new rerank arm changed the actual ranking basis (both always return
   the raw pooled DINO score in the tuple, never the fused/patch score
   that determined sort order). A bare subtraction could go negative,
   unconditionally tripping the ambiguity flag — fixed with `abs()`.
2. **DINOv3 identity-index cache fingerprint missing split params**:
   tracked checkpoint + product count, not `SPLIT_SEED`/
   `TEST_IMAGES_PER_PRODUCT`/`VAL_IMAGES_PER_PRODUCT` — editing any of
   those to try a different split would leave count unchanged, silently
   serving a stale cache built from the wrong split. Extended the same
   caching care already given to `--top-identity-candidates`.
3. **`composed_query_search.py`'s attribute vocabulary missing 3 of 5
   new schema fields**: `heel_type`/`sole_type`/`toe_shape` were added
   to `newLLMprompt.py` and both by-facet eval scripts the same day, but
   this file (written slightly after) only picked up 2 of 5 — a query
   like "with a chunky rubber sole" would silently never match.

None of these were reachable via any default configuration (fusion and
patch-rerank are both off by default; the cache bug needs editing a
module constant, not a CLI flag), so no existing eval number is
affected — but all three were real, would have produced silently wrong
output the moment someone exercised the affected path. This review
pattern (a fresh, focused, read-only agent explicitly told what caliber
of bug to look for and given the list of what's already been caught)
has now found 3/3 real issues on first pass — worth repeating whenever
a comparable volume of new code lands in one session.

## Update — 2026-08-02/03: segment_outfit.py's flagged next-step acted on and validated

Picked up the exact hypothesis flagged in this file's earlier
`segment_outfit.py` entry ("consider raising `points_per_side` above 16
... sparse enough that a genuinely occluded/small item might never get
proposed as a candidate at all") with real new test data, since more
real multi-item photos are now available (Vans/Carhartt product photos
that happen to show a model wearing multiple visible garments).

**Real miss found**: a Carhartt photo (navy shirt + light denim jeans,
both clearly visible, model small in frame with a large empty
background) detected **0 items** at the original settings
(`points_per_side=16`, `MIN_AREA_FRACTION=0.03`). Debugged properly —
dumped raw SAM2 candidates rather than guessing — and found only 4 raw
masks proposed total; the dominant one (88% of frame) was correctly
excluded as "the whole photo," but nothing else cleared
`MIN_CONFIDENCE`.

**Fix tested before committing, not applied blind**: `points_per_side`
16→32 found a genuinely correct "jeans" candidate at 0.826 confidence
— but its `area_fraction` (0.028) fell just under the 0.03 floor by a
razor-thin margin. Lowered `MIN_AREA_FRACTION` to 0.02 alongside the
point-density increase; re-ran the real `detect_outfit_items()`
function (not just the raw SAM2 dump) and confirmed the jeans are now
correctly detected. **Regression-checked against the two
already-working test cases** (Champion hoodie, Vans sweatshirt+pants)
at the new settings before committing — both still detect correctly,
same categories, no new false positives.

**Net result across 3 real test images: 4/4 correct detections, 0 false
positives** (up from 3/4 — the Carhartt jeans miss is now a hit). Both
constants updated in `segment_outfit.py` with this reasoning inline.
**Still not a complete fix**: the Carhartt shirt itself was never
detected in any tested configuration — that garment's region never
produced a good candidate mask at either point density, a real
remaining gap. And this is still 3 hand-picked real images, not the
labeled multi-item benchmark this file's own prior entry already
flagged as the real next step — that's still not done.

## Update — 2026-08-02/03: `--top-identity-candidates` sweep — new best result, real returns not yet diminishing

Real Colab run against both fine-tuned checkpoints, sweeping
`--top-identity-candidates` through 35/50/75 (full numbers in
`docs/eval_log.md`). **New best real number: 53.53% R@1 ungated at
K=75**, up from 47.65% at K=25 — a real, substantial ~6pt gain. The
identity-shortlist miss rate — the dominant bottleneck since it was
first diagnosed — dropped from 20.00% (K=25) to 4.87% (K=75), nearly
solved at this setting. Median rank hit 1.0 for the first time: more
than half of held-out queries now get the exact right product as the
literal #1 result.

**The predicted diminishing-returns pattern hasn't shown up yet.**
Marginal R@1 gain per additional candidate held roughly steady across
the whole sweep (~0.11-0.14pt/candidate at every step from 25→35→50→75)
rather than tapering off. Recommend continuing the sweep past 75 (try
100, 150) — there's no evidence yet of where the real ceiling is, only
that it's higher than previously assumed.

**Category gating confirmed net-negative a 4th/5th/6th time**, and the
gap is *widening* with K, not narrowing (gated trails ungated by ~10pt
at K=35, ~13pt at K=75) — further evidence ungated should stay the
permanent default rather than being revisited once K grows.

**One artifact flagged honestly, not silently ignored**: the HSC
climb-level diagnostic (`category 13.2%, group 11.0%, root 75.8%`) is
identical across all 3 new K values (expected, HSC doesn't depend on
K) but doesn't match the 2026-08-01 HSC eval's numbers (`group 34.9%,
root 51.9%`) despite that also being a K=25 run at the same threshold.
Best explanation, not confirmed with certainty: the `evaluate()`
batching fix (2026-08-02, this same session) changed the floating-point
computation path (batched matmul vs. one-image-at-a-time), and HSC's
threshold-based climbing (0.5) is sensitive to exactly this kind of
small numerical shift near a decision boundary. Doesn't change the core
conclusion — gating is clearly net-negative in both the old and new
numbers — but worth knowing this diagnostic isn't perfectly stable
across code versions, only the headline R@1/miss-rate numbers should be
treated as load-bearing for now.
