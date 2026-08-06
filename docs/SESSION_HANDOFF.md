# Session handoff — end of 2026-08-05

Read this first in a new session, then `docs/roadmap_2026-08-05.md` for the
plan and `docs/eval_log.md` for every number. This file is state + hard-won
gotchas; the other docs are the reasoning.

---

## 1. Where the project is

**A working end-to-end product exists.** Point a phone at a garment, get a
spoken answer naming a catalog product, in ~1.5s.

| | |
|---|---|
| Catalog | 12 brands, 2,387 records, 2,230 with images, **2,220 categorised** |
| Best R@1 | **47.65%** (12-brand, K=400, ungated); **58.58%** at fixed gallery N=1000 |
| Serving | live: `hanavm--fashion-serve-fashionservice-api.modal.run` |
| Endpoints | `/identify` `/compose` `/search` `/health` |
| Client | `siri_client.py` verified end-to-end; iOS Shortcut written, **never run** |
| Outfit corpus | 6,860 outfits / 9,971 images / **20,681 detected garments** |
| Co-occurrence | live in `/compose`; jacket→pants = 1,838 real photos |
| Guards | garment gate **calibrated** (AUROC 0.9994) + enforced; open-set **uncalibrated** (0.769) |

**Auth**: `FASHION_API_KEY` in `.env` (gitignored). Never print it.

---

## 2. THE CEILING CALL (the most important thing here)

**Model architecture work is essentially exhausted. Data and product work
are not.** Ten changes were measured against accuracy this session. Two
helped, and both were data plumbing:

| change | result |
|---|---:|
| gallery cap 2 → 6 images/product | **+5.97pt** |
| K 150 → 400 | **+4.60pt** |
| patch rerank | −30pt |
| ArcFace vs SupCon | −13.4pt |
| score fusion | −6.2pt |
| brand boost (OCR) | +0.10pt |
| attribute-head rerank | ~0 |
| logo detector | not wired (see below) |
| category gate | net-negative ×7 |
| distractor margin | 0.00pt |

Three constraints are now **eliminated, not just untried**:

- **Coverage is not the bottleneck.** K=all gives 0.00% shortlist miss and
  R@1 *falls* 0.27pt — coverage 94%→100% is exactly cancelled by
  conditional accuracy 50.65%→47.38%.
- **Latency is not a constraint.** Dense scan = **0.108 ms/query**; encoder
  forward = ~890–1180 ms. ANN/FAISS would optimise something already free
  (1.17 ms at 50k products, 11.4 ms at 500k).
- **Catalog size did not hurt quality.** Fixed-gallery eval: the doubling
  cost ~1.3pt, not the ~12pt raw R@1 suggested.

**What remains is the reranker's discrimination**, and seven attempts to
improve it have measured negative or flat.

### The two data levers with evidence behind them
1. **Human-labelled attributes.** A *perfect* attribute head is worth
   **+8.4pt** (oracle, 3 seeds). The real head at 67–85% accuracy is worth
   ~0. Labels are the limit — and current labels are LLM-generated from
   product *text*, not measured.
2. **Consumer↔catalog training pairs.** Training uses ~4.5 images/identity,
   which is why ArcFace couldn't estimate centroids. `outfit_dataset` is
   6,860 real worn photos — one labelling pass from being exactly the
   pairs the identity encoder is starved of. ~200 hand-labelled pairs would
   also give the **first honest number for the real deployment condition**
   (consumer photo → catalog). Every number we have is catalog→catalog.

---

## 3. DO NOT REDO THESE (measured, mechanism understood)

- **Patch-level rerank** (−30pt). `projection_head` was trained on pooled
  features only; patch tokens are out of distribution.
- **Score fusion** (−6.2pt). SigLIP2 scores are *identity-level*, so every
  colorway sibling gets the same score, flattening what DINOv3 learned.
- **ArcFace** (−13.4pt vs SupCon). 1,077 identities × ~4.5 images cannot
  estimate per-class centroids. Not a hyperparameter gap. (It also *never
  ran* before this session — it crashed on batch 1 with an fp16/fp32
  `scatter_` mismatch under AMP; fixed.)
- **Category gate** — net-negative on seven independent measurements.
- **Widening the shortlist / K past 400** — see K=all above.
- **ANN index for speed** — premature by ~3 orders of magnitude.
- **`segment_outfit.py` threshold tuning** — the bottleneck was SAM2's
  proposals, now replaced by `garment_proposer.py` (human parsing).
- **A logo detector via image-level brand labels.** Built; 98.44% macro
  recall *but four controls prove it never reads the mark* — at 32×32
  where nothing is legible it still scores 83.95%, and it separates
  catalog-photo from real-photo (AUROC 0.967) better than it does brands
  (0.909). It is a brand-*photography-style* classifier. A real one needs
  **mark-level supervision** (boxes on logos).

---

## 4. Operational gotchas (each cost real time)

- **`modal deploy` does NOT cycle a warm container.** It keeps running old
  code AND old secrets. A rotated API key kept accepting the leaked value
  after a "successful" deploy. Always:
  `modal app stop <app> --yes && modal deploy <file>`
- **`modal run` uses LOCAL code; `.spawn()` on a deployed function uses
  DEPLOYED code.** A smoke test passing via `modal run` says nothing about
  what `.spawn()` will execute. Deploy first, then spawn.
- **Never redirect two Modal runs to the same log path.** A stopped run's
  client keeps writing; the file interleaves and produces plausible wrong
  numbers. The tell: two different configs yielding byte-identical results.
- **`modal run --detach` still keeps a client attached** — a reaped shell
  cancelled a corpus run at record 1588. Use `.spawn()`.
- **Python block-buffers stdout to a pipe.** Long jobs look frozen at zero
  for hours. Set `PYTHONUNBUFFERED=1`.
- **Modal cold start** is ~17s normally, but hit **353s** after the catalog
  sync because the serving container absorbed index re-enrolment. Run
  `modal run modal_app_serve.py::build_indexes` after any catalog change.
  **This is still outstanding.**
- **Concurrent writers to `metadata.json` silently lose records** (whole-file
  rewrite). `dataset_utils` now has flock + atomic replace for the outfit
  path. 127 images were lost this way once.
- **Isolate experiment indexes.** `RETRIEVAL_INDEX_DIR=...` — an open-set
  run rebuilds `retrieval_indexes/` with a *reduced* gallery and the live
  service would silently serve it.

---

## 5. Key env knobs (all default to prior behaviour)

```
GALLERY_IMAGES_PER_PRODUCT=6     # default; 2 was the old value, +5.97pt
EVAL_GALLERY_SIZE=0              # >0 = fixed-size gallery, comparable R@1
RETRIEVAL_INDEX_DIR=...          # isolate experiment indexes
GARMENT_GATE_THRESHOLD=0.010     # calibrated on 507 real negatives
IMAGE_LOADER_WORKERS=8           # 32 on Modal; Drive FUSE is fragile
DISTRACTOR_MARGIN=0.0            # measured to buy nothing; kept as record
```

Modal eval app takes `--env "K=V,K2=V2"` to set these per run.

---

## 6. What to do next (ordered)

1. **`build_indexes`** on `fashion-dataset` — fixes the 353s cold start.
2. **Start the production track** — this was agreed. `POST /query` per
   `docs/unified_query_design.md`: one endpoint, rules-based routing
   (image? taxonomy term? companion preposition? exactness cue?), verified
   to reproduce `/identify` `/compose` `/search` exactly. **Do not unify
   the embedding space** — that is score fusion, measured at −6.2pt.
3. **Run the iOS Shortcut once.** `siri/README.md`. Never executed.
4. **Data track**: ~200 hand-labelled consumer→catalog pairs (first honest
   deployment-condition number); human-labelled attributes (+8.4pt oracle).
5. **Grow the catalog on streetwear.** Now safe — use `EVAL_GALLERY_SIZE`
   so growth and quality stay separable. Expect raw R@1 to fall as
   streetwear densifies; that is correct, not regression.

---

## 7. Known gaps / honest limits

- **Brand coverage is 12 brands.** Point it at Uniqlo or North Face and it
  will confidently name something else — open-set rejection can't catch it
  (AUROC 0.769, no usable operating point: 1% false-reject costs 68%
  false-accept). **Frame v1 as "these 12 brands", not "any clothing."**
- **167 products still have no canonical taxonomy** (of 2,387).
- **All accuracy numbers are catalog-photo → catalog-photo.** The real
  condition (consumer photo → catalog) has never been measured.
- **Outfit labels are unvalidated model output**, no ground truth. Every
  surface says so; keep it that way.
- **1,342 Pinterest records have no author** — not traceable for takedown.
  The only item with a closing window (`docs/licensing_review.md`).
- **`reject_threshold` is uncalibrated**, so `rejected_open_set` never
  fires by default. The API reports `reject_threshold_calibrated: false`.
- **Modal spend this session: ~$18 of a $30 budget.**

---

## 8. Untracked backups on disk (intentional, gitignored or unstaged)

```
apparel_dataset/metadata.pre-taxonomy-backfill.json
docs/hierarchy.pre-backfill.json
outfit_cooccurrence.sam2-backup.json
outfit_dataset/metadata.sam2-backup.json
```

Keep until the current state is trusted; they are the rollback path for the
taxonomy backfill and the human-parsing re-detection.

---

## 9. Working norms that paid off

- **Measure before believing, including your own diagnosis.** Five of this
  session's predictions were overturned by measurement — brand evidence,
  synthetic negatives, ArcFace, the detector's distractor rule, and
  attribute learnability.
- **A funnel attributes kills to whatever filter runs first.** The
  "distractor rule causes 66% of losses" finding was an artifact of filter
  order; those masks died at `MIN_CONFIDENCE` regardless.
- **Confidence is not reliability here.** Three times the *wrong* answer
  scored higher than the right one (0.922 vs 0.854 on the jacket miss).
  Don't build a guard on confidence alone.
- **Byte-identical results across different configs mean a bug**, not a
  robust finding.
- **Verify agent claims independently.** They were mostly excellent and
  honest, but the ones that mattered were the ones I checked.
