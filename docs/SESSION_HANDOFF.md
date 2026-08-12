# Session handoff — 2026-08-11

Read this first. Then `docs/production_plan.md` (whose top section carries
the product correction) and `docs/eval_log.md` (every number).

**The previous handoff described an item identifier. That is no longer the
product.** It is still built, still deployed, still works — but it is at
most a feature now.

---

## 1. What the product is

**Outfit search.** A photo of an item plus words for what it should be worn
with, returning **real outfit photos of real people**.

```bash
.venv/bin/python outfit_search.py serve      # http://localhost:7880
```

Use `.venv/bin/python`, never `python3` — system Python has torch 2.0.0 and
`transformers` disables its PyTorch backend below 2.4, producing an error
that looks nothing like a wrong-interpreter error.

The owner's own words when the identifier was demoed: *"the idea is you
give a picture and a text and you get returned a bunch of images of
people's outfits that satisfy picture and text. not at all what i was
thinking. this just gave me what it thought the item was."*

Confirmed with them: **item**-anchored (the photo is a garment, not a mood
board), returning **whole photos**.

### How it ranks

Multimodal — any number of images and phrases, either alone. **Each part is
matched to a different garment in the photo**:

```
part_score = max over the photo's crops of sim(part, crop)   # each crop claimed once
photo_score = mean of the parts' scores
```

Parts claim garments greedily, strongest match first. Scoring every part
against the photo's *best* region would let one jacket-that-is-vaguely-
jeans-like satisfy "this jacket with baggy jeans", which is not the
question.

Two indexes, both local, both rebuilt 2026-08-08:

| | |
|---|---|
| `outfit_search_index.pt` | 16,330 photo vectors (33 MB) |
| `outfit_crop_index.pt` | 31,239 garment-crop vectors (57 MB) |

**Encoder: SigLIP2 v3 only. DINOv3 is deliberately not used here.** The
identity fine-tune bought +31.3pt by pushing colorway siblings *apart* —
the right geometry for "which exact product is this", the wrong one for
"show me outfits like this".

---

## 2. State

| | |
|---|---|
| Catalog | **20 brands / 4,079 records**, all men's |
| Served gallery | **3,922 products**, live, 20.1s cold start |
| Outfit corpus | **10,640 posts / 16,330 images / 31,239 detected garments** |
| Endpoints | `/query` `/identify` `/compose` `/search` `/health` |
| Modal spend | **$24.64 of $30** |

Auth: `FASHION_API_KEY` in `.env` (gitignored). Never print it.

---

## 3. The measurements that matter

**Outfit search has no evaluation at all.** No ground truth, no eval row.
Judge by looking. That is the single biggest gap.

**The encoder is the suspected ceiling, and it is measurable:**

- Mean pairwise similarity between random outfit photos: **0.743**
- "baggy jeans and a black jacket": top-1 0.203, top-50 0.187 — a **0.015**
  spread across 50 results
- **"a photo of a bicycle"** still scores 0.155 top-1

Most of each vector encodes "a person standing in a room", not the clothes.
More corpus buys recall; it cannot fix this. The fix is fine-tuning SigLIP2
on outfit photos, which needs labels.

**Crops matter enormously — the finding of the session:**

| | group agreement | footwear |
|---|---:|---:|
| plain bbox crop | 37.7% | 15% |
| **proposer's masked crop** | **72.4%** | **89%** |

A bbox drags the floor, the wall and the person's legs into the encoder.
Measured paired on 237 garments, replicated at n=15,854.

---

## 4. Two production bugs found this session — do not reintroduce

1. **Serving ran `TOP_IDENTITY_CANDIDATES=25` while every eval used
   150/400.** It is a module constant; `retrieve()` takes it as a kwarg and
   `--evaluate` passed it, but `modal_app_serve.py` never did. So the API
   shortlisted 25 identities out of 1,562 and routinely never saw the true
   product. Now env-overridable, serving sets 400. **400 is a FLOOR, not a
   tuned value** — it was measured at a 2,230-product gallery and K scales
   with catalog size. Sweeping it is cheap and probably worth points.
2. **HTTP 422 killed whole scraper shards.** Arctic Shift returns it
   intermittently on deep cursors; it was not in the backoff set, so
   `raise_for_status()` took down the run. Fixed.

---

## 5. Operational gotchas (each cost real time)

- **`modal deploy` does NOT cycle a warm container** — old code AND old
  secrets survive. Always `modal app stop <app> --yes && modal deploy`.
- **Modal cost is WALL TIME, not request count.** The container stays warm
  between requests, so a slow *local* stage in the loop is billed. Two
  runs: 17,682 garments in 16 min = $0.27; 15,846 garments in 116 min =
  **$2.35**. Precompute, then send densely.
- **`modal volume put -r` corrupts intermittently.** Use one tar per brand
  plus `modal_app_sync_catalog.py::extract`.
- **Long local GPU jobs get killed.** A 40-minute crop build took 10.5
  hours and died four times — macOS was running `ANECompilerService` at
  98% and `BackgroundShortcutRunner` at 64%, load average 24, competing for
  the same silicon MPS needs. Run detached (`nohup … & disown`), checkpoint
  on **elapsed time** (batch-count intervals stretch past the kill
  interval under load), and make the job incremental.
- **Concurrent writers to `metadata.json`** are safe only for *disjoint*
  records. Two scripts writing different fields on the same records is not
  safe — that cost 175 records once.

---

## 6. What needs a human

1. **`pair_eval.py label`** — ~20 min in a browser. Produces the first real
   consumer-photo→catalog number in the project's history, plus the
   calibration data that would let open-set rejection actually fire. The
   build step now samples only from the matchable half of the corpus; the
   first attempt was useless because ~42% of the corpus could never match.
2. **The iOS Shortcut** — written in `siri/README.md`, still never run. Note
   it points at `/identify`, the old product. An outfit-search Shortcut
   would need a small new endpoint.
3. **Judging outfit search by using it.** There is no metric for this.

---

## 7. Honest limits that survive everything above

- **Outfit labels are unvalidated model output.** 31,239 detections from a
  human parser plus zero-shot FashionCLIP over an unlabelled corpus.
  ~91% precision, eyeballed on 40 photos — the same 40 the threshold was
  chosen on. Coverage, not accuracy.
- **`outerwear` sits at 41% group agreement even with good crops**, and it
  is structural: ATR has no Coat class, so a jacket and the tee under it
  come back as one region. Better crops and more brands will not fix it.
- **Brand-targeted sourcing biases the eval optimistically.** Photos found
  by searching "carhartt detroit jacket outfit" over-represent that jacket
  relative to what a real user photographs.
- **`reject_threshold` is uncalibrated**, so `rejected_open_set` never
  fires. The API reports `reject_threshold_calibrated: false`.
- **Confidence is not reliability.** The wrong answer has outscored the
  right one, 0.922 to 0.854.
- Licensing: **every Pinterest record now has an author** (was 1,342
  untraceable). But provenance is not permission — see
  `docs/licensing_review.md`, whose risks 2 and 3 are unchanged.

---

## 8. Working norms that paid off

- **Measure before believing, including your own diagnosis.** Four claims
  were overturned by measurement this session — including three of my own:
  "exact-product matching fails" (it was my crop), "cost is per-request"
  (it is wall time), and "`apparel_dataset_full` is redundant" (it was the
  only local copy of five brands).
- **Verify your own tooling before reporting its output.** The first
  group-agreement diagnostic was wrong twice — it treated canonical
  taxonomy leaves as site categories, then dropped unmapped terms.
- **A subreddit's flair is not evidence.** r/Sneakers `WDYWT` is
  feet-on-pavement. Sample the API, then **audit what the shard actually
  wrote to disk** — that caught r/gorpcore at 8/8 product photos.
