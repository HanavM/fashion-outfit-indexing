# Production plan — 2026-08-05

> ## CORRECTION, later the same day: the product is OUTFIT SEARCH
>
> **Read this before the rest of the file.** Everything below Stage 0 was
> written on the assumption that the product is an item identifier —
> point a phone at a garment, hear which catalog product it is. That is
> what `docs/project_spec_v1.md`, `docs/roadmap.md` and
> `docs/unified_query_design.md` all describe, and it is wrong.
>
> The owner, on seeing the working UI: *"the idea is you give a picture
> and a text and you get returned a bunch of images of people's outfits
> that satisfy picture and text. not at all what i was thinking. this
> just gave me what it thought the item was."*
>
> **The product is: item photo + text → real outfit photos.** The result
> set is `outfit_dataset` (6,860 posts / 9,999 images), not catalog SKUs.
> Confirmed with the owner: *item*-anchored (the photo is a garment, not
> a mood board), returning *whole photos* (not per-garment breakdowns).
>
> ### What this changes
>
> **The encoder.** This file's own Stage 3 quotes the tension without
> noticing it decides the question: "show me blue jeans" wants all blue
> jeans NEAR each other; "show me *this exact* sneaker" wants colorway
> siblings FAR apart. The identify path needs the second geometry, which
> is exactly what DINOv3's identity fine-tune bought (+31.3pt). **Outfit
> search needs the first.** So `outfit_search.py` uses SigLIP2 and does
> not touch DINOv3.
>
> **What the accuracy numbers are worth.** R@1 47.65%, 58.58% fixed
> gallery, the +31.3pt fine-tune, the whole eval log — those measure
> item identification. They are not wrong, and they are not evidence
> about outfit search, which currently has **no measurement at all**.
>
> **What is still load-bearing.** The outfit corpus and its 20,681
> garment detections (the crop index is built from those bboxes); the
> SigLIP2 v3 checkpoint; the co-occurrence index; `dataset_utils` and
> the scraping pipeline. The identity encoder, the garment gate, the
> open-set work, `/query`'s routing and `pair_eval.py` all belong to the
> identifier, which is now at most a *feature* of the product ("what is
> this item") rather than the product itself.
>
> ### What that makes the plan
>
> 1. **Build and judge outfit search.** `outfit_search.py build` then
>    `serve`. Nothing is evaluated; there is no ground truth for "is this
>    a good outfit match." Judge by looking, then decide what to measure.
> 2. **Decide what the identifier is for.** It works and it is deployed.
>    It is plausibly the "shop this look" half of the product. That is a
>    product decision, not an engineering one.
> 3. **Re-scope catalog scraping.** Brand coverage was the top priority
>    when the catalog *was* the result set. For outfit search the corpus
>    that matters is `outfit_dataset`, and growing *that* is the
>    equivalent lever.
> 4. Stages 1–5 below stand as written **only** for the identifier.
>
> The rest of this file is left unedited, as the record of what was
> planned and why, before the premise was corrected.

---

Written after `docs/SESSION_HANDOFF.md`'s ceiling call, which is the
premise this whole plan rests on:

> Model architecture work is essentially exhausted. Data and product work
> are not.

Ten changes were measured against accuracy last session. The only two that
helped were data plumbing (gallery cap 2→6, **+5.97pt**; K 150→400,
**+4.60pt**). Seven attempts to improve the reranker's discrimination
measured negative or flat. Coverage, latency, and catalog size were each
*eliminated* as bottlenecks by measurement rather than left untried.

So "go to production" here does not mean "make the model better." It means
three things, in this order of evidence:

1. **Stop being confidently wrong outside the catalog** — a coverage problem.
2. **Have one interface a real client can call** — a product problem.
3. **Measure the condition we actually ship in** — currently unmeasured.

Everything below is scoped against that. Nothing on this list is a new
ranking idea; the last three of those lost.

---

## Stage 0 — `POST /query` ✅ done, deployed, parity verified

`docs/unified_query_design.md` is the design of record. Shipped
2026-08-05: `query_router.py` (rules), `/query` in `modal_app_serve.py`,
`test_query_router.py` (23 cases), `query_parity_check.py` (10 live
cases, all holding).

The three original endpoints are untouched and still work. `/query` calls
the same `_do_identify` / `_do_compose` / `_do_search` they call, so
parity is by construction rather than by a second implementation kept in
sync by hand.

Two queries that previously mapped to no endpoint at all now work:
`"what brand is this"` and `"find me blue jeans like this"`.

**Explicitly not done, and not an oversight:** the scores are not blended.
That is unifying the *representation* rather than the *interface*, and it
measured −6.22pt R@1 with shortlist miss identical in both arms. Parsed
text also does not filter the identify path, because category gating
measured net-negative on seven independent runs.

---

## Stage 1 — brand coverage (in flight)

**The problem, stated the way the handoff states it:** point this at
Uniqlo or The North Face today and it will confidently name something
else. Open-set rejection cannot catch it — AUROC 0.769, and there is no
usable operating point (1% false-reject costs 68% false-accept). So the
only real fix is to *have the brand in the catalog*.

Four scrapers running concurrently, one brand each, ~200 records × 4
men's categories, following `SCRAPING_PROCESS.md`:

| target | why this brand |
|---|---|
| Uniqlo | the handoff's own named failure case; modern basics |
| The North Face | outerwear, the thinnest category in the catalog |
| Obey | streetwear graphics |
| HUF | streetwear graphics |

Concurrency is safe *only* for scrapers, and only because
`dataset_utils.save_records_safe` merges by `product_code` at save time.
It does **not** make concurrent writers to the same records safe — see
Stage 2.

**Held at four, not six, by disk.** ~7.8 GiB free at the time of writing
and ~1.8 MB/record. `apparel_dataset_full/` is 8 GB and looks like dead
weight, but three live scripts still reference it
(`research_localized_query_validation.py`, `logo_detector.py`,
`brand_evidence.py`), so it is not free to delete. Freeing it is the
prerequisite for the next four brands.

---

## Stage 2 — the post-scrape pipeline, and its ordering hazard

New records are **invisible to text search** until they carry a canonical
taxonomy — that was the bug fixed in commit `2326239`, where half the
catalog was silently unreachable. So this stage is mandatory, not optional
polish.

Run **strictly sequentially**, one at a time:

1. `caption_apparel.py --brand <b>` — structured captions
2. taxonomy backfill — canonical paths (the `2326239` fix)
3. `segment_apparel.py --brand <b>` — **only** for brands whose photos are
   on-model. Each scraper agent reports flat-lay vs on-model from actual
   image inspection; flat-lay brands (Carhartt, Stüssy) correctly skip it.
4. `modal volume put` the new records, **per file, never `-r`** — the
   recursive form has a reproduced corruption bug in this project.
5. `modal run modal_app_serve.py::build_indexes`
6. `modal app stop fashion-serve --yes && modal deploy modal_app_serve.py`

**Why sequential.** Running `caption_apparel.py` alongside
`segment_apparel.py` on the *same brand* lost 175 of 177
`structured_caption` fields. `save_records_safe` merges whole records by
code, so a long-running job's snapshot — taken before the sibling field
landed on disk — overwrites that field on every checkpoint. The merge
logic worked exactly as designed. Lesson 13 in `SCRAPING_PROCESS.md`.

**Step 5 is also the outstanding cold-start fix.** Modal cold start is
~17s normally but hit **353s** after the last catalog sync, because the
serving container absorbed index re-enrolment. `build_indexes` moves that
cost off the request path. It is incremental now (per-product cached
vectors keyed by a `gallery_signature`), so it costs only the new brands.

**Step 6 is not optional.** `modal deploy` does **not** cycle a warm
container — it keeps running old code *and* old secrets. A rotated API key
kept accepting the leaked value after a "successful" deploy.

---

## Stage 3 — re-measure, with growth and quality kept separable

Two numbers, not one:

- **Raw R@1** will *fall* as the catalog grows. That is correct, not a
  regression: more products means more near-neighbours.
- **`EVAL_GALLERY_SIZE=1000`** holds the gallery fixed so quality is
  comparable across catalog sizes. The last doubling cost ~1.3pt by this
  measure, against ~12pt by raw R@1. Report both, lead with the fixed one.

**One thing genuinely needs re-tuning: K.** It is not an absolute — it
scales with catalog size. Doubling the gallery from 1,077 to 2,230 drove
shortlist miss from 1.18% to 19.95% at K=150, and recovering it needed
K=400. Going to ~3,400 products will move it again. This is a sweep of a
known-monotone knob, not a research question.

Note the ceiling that bounds it: K=all gives 0.00% shortlist miss and R@1
*falls* 0.27pt, because coverage 94%→100% is exactly cancelled by
conditional accuracy 50.65%→47.38%. So there is an optimum, and it is not
at either end.

---

## Stage 4 — the number we do not have

**Every accuracy figure in this project is catalog-photo → catalog-photo.**
The real deployment condition — consumer photo → catalog — has never been
measured, not once.

~200 hand-labelled pairs from `outfit_dataset` (6,860 real worn photos,
already collected) would produce the first honest deployment number. The
same labels are the pairs the identity encoder is starved of: training
uses ~4.5 images/identity, which is *why* ArcFace could not estimate
per-class centroids and lost 13.4pt to SupCon.

This is the highest-value item on the list and the only one that can
change what we believe about the system. It is placed after Stages 1–3
only because it wants the grown catalog to label against.

Second data lever, same category: **human-labelled attributes**. A
*perfect* attribute head is worth **+8.4pt** (oracle, 3 seeds). The real
head at 67–85% accuracy is worth ~0. Current labels are LLM-generated from
product *text*, not measured — the labels are the limit, not the head.

---

## Stage 5 — the client path

`siri_client.py` is verified end to end. The iOS Shortcut is written and
**has never been run once**. That is a five-minute task standing between
"a working API" and "a working product," and it should stop being
outstanding.

`/query` also means the two-Shortcut split (one for identify, one for
compose, with Siri's phrase matching doing the routing) can collapse to
one. Worth doing after the Shortcut runs at all.

---

## What is deliberately NOT on this plan

Each of these is measured and understood, not untried. Re-running them is
how this project would waste its next session:

| | |
|---|---|
| patch-level rerank | −30pt (projection head trained on pooled features only) |
| score fusion | −6.2pt (SigLIP2 scores are identity-level) |
| ArcFace over SupCon | −13.4pt (~4.5 images/identity cannot estimate centroids) |
| category gate | net-negative ×7 |
| K past 400 / K=all | coverage gain exactly cancelled by conditional accuracy |
| ANN / FAISS index | premature by ~3 orders of magnitude — dense scan is 0.108 ms/query against ~890–1180 ms for the encoder forward |
| logo detector via image-level brand labels | it is a brand-*photography-style* classifier; 83.95% at 32×32 where no mark is legible. Needs mark-level supervision |
| `segment_outfit.py` threshold tuning | the bottleneck was SAM2's proposals, now replaced by `garment_proposer.py` |

The ANN one is worth restating because `unified_query_design.md`
recommends it: that recommendation was written before the latency
measurement. At 50k products a dense scan is 1.17 ms; at 500k, 11.4 ms.
It is not the constraint and will not be for a long time.

---

## Honest limits that survive this entire plan

- **v1 is "these N brands", not "any clothing."** Growing from 12 to 16
  narrows the gap; it does not close it, and open-set rejection cannot
  cover the remainder at any usable operating point.
- **Outfit labels are unvalidated model output.** Every surface says so.
  Keep it that way.
- **`reject_threshold` is uncalibrated**, so `rejected_open_set` never
  fires by default. The API reports `reject_threshold_calibrated: false`
  rather than letting a client read `false` as "checked and confident."
- **Confidence is not reliability here.** Three times the *wrong* answer
  scored higher than the right one (0.922 vs 0.854 on the jacket miss).
  Do not build a guard on confidence alone.
- **1,342 Pinterest records have no author** and are not traceable for
  takedown — the only item with a closing window
  (`docs/licensing_review.md`).
