# Gap analysis against the spec's intended design — 2026-08-04

Written after the serving layer shipped, by walking `docs/project_spec_v1.md`
component by component rather than by extending the previous plan. The
question asked was "what has NOT been built," so unbuilt and
built-but-disproven items are stated plainly.

## Spec §3 architecture — box by box

| spec component | status |
|---|---|
| Item detection and segmentation | **Weak.** `segment_outfit.py` exists; validated on ~3 photos. On a realistic product-page screenshot it returned 1 item, the whole person, and the wrong garment. |
| Per-item crop and mask | Partial — follows detection's fate |
| SigLIP 2 semantic embedding | **Done**, fine-tuned v3 |
| DINOv3 visual-identity embedding | **Done**, fine-tuned supcon, 59.92% R@1 |
| Category head | **Done** as HSC climbing over 42 leaves |
| Attribute heads | **Not built.** Attributes exist only as offline LLM captions in `structured_caption.attributes`; nothing predicts them at query time |
| **Logo detection and OCR** | **OCR half built and measured (2026-08-04, `brand_evidence.py`); logo half still absent.** OCR reads a brand on 11% of product photos at 100% precision, but wiring it into retrieval moved R@1 +0.10pt (one query in 1,043) even at a weight that amounts to a brand restriction. See eval log |
| Optional VLM structured evidence | Offline only (`caption_apparel.py`), never at query time |
| Confidence calibration and label backoff | **Done** — HSC backoff + open-set rejection, now calibrated (AUROC 0.769) |
| Structured item record | Done for catalog; **absent for outfit photos** |
| Metadata + semantic + identity indexes | **Done** |
| Candidate retrieval | **Done**, and tuned to saturation |
| Patch-level reranking | **Built, measured, HARMFUL** (−30pt). Disabled. The spec asks for it; the evidence says no |
| Multimodal reranking | Score fusion built and **also harmful** (−6.2pt). Disabled |
| **Outfit-level results** | **Built 2026-08-04**, on real but thin evidence. `outfit_cooccurrence.json` aggregates detections over all 6,860 outfit photos; `composed_query_search.py` now grounds companions in observed co-occurrence and falls back to the old two-independent-searches path when the corpus has none. `POST /compose` is NOT yet wired to it |

## Spec §1 — the four named query types

The objective names four queries. The API serves **one and a half**.

| query | endpoint | reality |
|---|---|---|
| "Show me this exact sneaker" | `POST /identify` | ✅ works |
| "Show me this shoe with cargo jorts" | `POST /compose` | ⚠️ two *independent* searches — it does not know what goes with what |
| "Show me blue jeans" | — | ❌ **no endpoint.** `catalog_query_search.py` exists, unserved |
| "Show me gray suede Adidas sneakers" | — | ❌ **no endpoint.** `free_text_visual_search.py` exists, unserved |

Two working scripts are simply not wired to HTTP. This is the cheapest
real gap on the list.

## Things true of the product that the spec never anticipated

1. **Non-clothing input.** A voice assistant gets pointed at anything.
   Nothing currently checks the photo contains a garment before asserting
   a product. A zero-shot gate separated clothing from synthetic
   non-clothing at AUROC 1.0000; being re-measured against real photos.
2. **Off-catalog garments are not rejectable.** AUROC 0.769 with no good
   operating point — 1% false-reject costs 68% false-accept. A real
   limitation to design around, not tune away.
3. **Screenshots are the actual input.** "Seen on screen" means page
   chrome, text and several products, not a clean catalog photo. The
   detector fails on exactly this today.
4. **The end-to-end path has never run.** Client tested against a stub,
   server tested with curl, never joined.

## Roadmap — Phase 10 onward, ordered by product impact

### P0 — the system is currently wrong without these

**10.1 Garment gate** *(in progress)*. Reject non-clothing before the
identity stage. Near-zero cost: reuses the SigLIP2 image embedding already
computed, one matmul against ~21 cached text vectors.

**10.2 Serve the two missing query types.** Wire `catalog_query_search.py`
and `free_text_visual_search.py` to `POST /search`. Half the spec's named
queries become reachable for maybe a day's work. Highest value-to-effort
on this document.

**10.3 End-to-end test.** Point `siri_client.py` at the live API with a
real photo. Roughly five minutes, and it is the only thing that converts
"both halves work" into "the product works."

### P1 — the answer is incomplete without these

**11.1 Outfit-level results.** ~~Run `index_outfits.py` over the 6,860
outfit photos on GPU, build a co-occurrence index, and replace
`/compose`'s two-independent-searches with retrieval against it.~~
**Done 2026-08-04, with a real caveat on how thin the evidence is.**

The corpus run completed on an A10G: all 6,860 records, 0 failures,
10,842 items, ~3.9 GPU-hours (~$5.6 total Modal spend including a
cancelled first attempt). `build_outfit_cooccurrence.py` aggregates it
into `outfit_cooccurrence.json`, and `composed_query_search.py` now
returns an `outfit_evidence` block instead of the blanket "two
independent searches" disclaimer, falling back to the old path (and
naming which of three reasons applied) when the corpus has no evidence.

**Half the corpus contributes nothing.** 14.5% of photos yielded no
detection and 34.1% yielded exactly one item; a single-item photo says
nothing about what goes with what. The index therefore rests on the
**3,530 photos (51.5%) that yielded 2+ distinct categories**, not on
6,860. Mean 1.85 items per productive outfit, from photos of people
wearing four or five garments — so the binding constraint is 11.2's
detector, not the data, and improving it would deepen this index for
free.

Two things measured while building it, both worth not re-learning:
- **`lift`/PMI are unusable here.** 52 of 74 pairs scored below chance,
  which is impossible of clothing. Categories compete for a capped number
  of detections (one item per category, plus NMS), so joint probabilities
  sit below the product of the marginals. Ranking is by count/p(b|a); the
  artifact is recorded in the index's own `diagnostics`.
- **Colour is much weaker than category.** Spot-checking crops, a
  correctly-labelled "loafer" had a box containing wall, ground and both
  legs, so its colour came off the background. Colour re-ranks, never
  filters.

Still open: `POST /compose` in `modal_app_serve.py` reimplements the text
half itself and still returns the old hardcoded "co-occurrence index ...
does not exist yet" note. The index exists; that endpoint needs wiring.

**11.2 Detection that survives a screenshot.** Fix `segment_outfit.py`'s
resize behaviour (`MAX_IMAGE_DIM=1024` shrinks a 1170×2532 screenshot's
subject to ~360×180, which is likely why SAM2 stops proposing per-garment
masks), then validate on a real labelled screenshot set. Until then the
Shortcut relies on a human crop, which currently beats the detector.

*Update 2026-08-05 — the resize hypothesis was not the main cause.* The
detector's weakness is upstream of resolution: SAM2's class-agnostic point
grid does not propose garment-shaped regions at all (the full measurement
is in `segment_outfit.py`'s `DISTRACTOR_MARGIN` block). `garment_proposer.py`
replaces the proposer with human parsing and, on 40 random corpus photos,
takes **1.53 → 2.48 items/photo while roughly doubling crop precision by
eye (~48% → 91%, all 127 crops inspected)**; the 2026-08-05 roadmap entry
has the table and the failure modes. Shipped opt-in as `--proposer
human-parsing`; making it the default means re-running the corpus and
rebuilding `outfit_cooccurrence.json` from it, which is now cheap (~1.4 s
vs ~75 s per photo on CPU). The screenshot case still needs its own
validation — the parser needs a person in frame, which is the one thing a
page screenshot does reliably contain.

**11.3 Brand evidence — logo detection and OCR.** ~~Likely the single
largest untried accuracy lever left.~~ **Built and measured 2026-08-04
(`brand_evidence.py`, `brand_evidence_eval.py`, `--brand-boost`). The
premise was wrong on both halves.**

*"Most product photos carry legible wordmarks"* — they do not. Over
1,186 real catalog images, **51% contain no OCR-legible text at all**
and only **11% yield a brand**. What it does read, it reads perfectly:
132 assertions, zero wrong, an exactly diagonal confusion matrix. The
per-brand spread splits on **wordmark vs. logo** — adidas 40% recall
(prints its name), New Balance 1% and Stüssy 0% (a giant "N" and a
handwritten script, which OCR cannot read by construction). Spec §4.5
lists "Visible logo" and "OCR" as separate sources for exactly this
reason; only the second is built, and the brands it misses are the ones
that need the first.

*"It adds information rather than re-weighting"* — it does not add any
the pipeline lacked. Wired in as a boost (never a filter, so it cannot
repeat the category gate's unrecoverable exclusions), R@1 moved
**65.00% → 65.10%, one query in 1,043**. Re-run at a weight large enough
to function as a brand restriction: **the same 65.10%**. Brand is
already implicit in the fine-tuned identity embedding — products of one
brand look alike, so DINOv3 had already ranked a same-brand product
first on nearly every query where OCR fired.

`--brand-boost` stays in the tree, off by default, harmless and
measured.

**The logo detector this row called for was then built and measured
2026-08-05 (`logo_detector.py`, three eval_log rows). It closes the
wordmark-vs-logo gap on paper and closes nothing in practice, because it
is not a logo detector — it is a brand-photography-style classifier.**

A logistic probe over frozen SigLIP2 features, held out by product
identity, gets **99.08%** on 542 unseen catalog images and lifts macro
per-brand recall from OCR's 11.07% to **98.44%** — precisely on the
brands OCR could not read (Stüssy 0.0% → 100%, New Balance 1.0% → 100%,
Champion 4.0% → 98.3%). Four controls say that number is not about marks:

* **The crops bought nothing.** Multi-crop MIL pooling was the whole
  reason to expect logo reading, and the *full-image* arm is the best arm
  (99.08% vs 98.34% max-pool). Zooming in should help a mark reader.
* **Destroying every mark costs 15pt.** At **32×32**, where no logo,
  wordmark or script is legible, accuracy is still **83.95%** and Stüssy
  still scores 82.8%. ~85% of the performance carries no mark at all.
* **It collapses off catalog photography.** On 300 real outfit photos,
  mean confidence falls 0.852 → 0.406; confidence alone separates
  real-photo from catalog-photo at **AUROC 0.967**, higher than the 0.909
  it manages at its actual job.
* Background removal is survivable (92.64%), so it is not *only* studio
  backdrop — what survives is house style: palette, cut, treatment.

The ranking question is also now settled with a measured cause rather
than an inference. **Nearest-neighbour on the same frozen embedding
already predicts brand at 96.49%** — a 12-way probe adds 2.59pt. That is
the control the `--brand-boost` row never had, and it explains its
+0.10pt exactly. Since the weight-1.0 arm already measured the ceiling of
a 100%-precision brand signal, the only variable a detector changes is
coverage (11% → ~100%), bounding the best case at roughly **+0.9pt** on
an extrapolation that itself assumes information the embedding does not
already have. Nothing was wired into the pipeline and no Modal run was
spent on it.

Open-set is the same story: AUROC **0.9092** vs the DINOv3 path's 0.769,
but **worse where it counts** (20.44% off-catalog recall at 1%
false-reject vs 32%), and against a strictly easier split (whole brands
held out, not identities within brands). Given the 0.967 domain AUROC,
that rejection signal is mostly a "not a catalog photo" detector and
would false-reject real users' in-catalog garments.

**So spec §4.5's "Visible logo" source is still genuinely unbuilt.** What
exists now is a 99%-accurate free brand labeller for *catalog-side*
imagery and a demonstration that whole-image embeddings cannot supply
query-side logo evidence. A real logo detector needs mark-level
supervision — detection boxes on logos, or a mark-crop dataset — not
image-level brand labels, because image-level labels are solvable without
ever looking at the mark and the optimiser will always take that route.
Until then OCR's 100%-precision / 9.17%-recall read remains the only
brand evidence that is actually about the garment rather than the
photographer.

### P2 — worth doing, not blocking

**12.1 Attribute heads** at query time, replacing offline LLM captions.
**BUILT AND MEASURED 2026-08-05** (`attribute_heads.py`, eval_log row).
The heads themselves work: on a product-level holdout, closure 85.3% vs
25.2% majority, material 81.7% vs 50.0%, colour 69.0% vs 25.0%, fit 67.1%
vs 34.0%. Wiring them into ranking earns **nothing** — best delta across
three split seeds is +0.15 / −0.76 / −0.46 pt R@1, i.e. noise, and larger
weights collapse the ranking the way `--patch-rerank` did. Unlike the
other negatives, though, the mechanism is not dead: an ORACLE arm using
the products' own labels gives a consistent **+8pt R@1**, so the ranking
information is real and the head's 67–85% accuracy is what fails to carry
it. So: keep the heads as a query-time descriptor (they can fill the 297
products with no material label and 1,141 with no fit label), do not add
a rerank flag until a head is materially more accurate. All accuracy
numbers are against LLM-generated labels, not ground truth.
**12.2 Catalog-size reset** — push the six local-only brands, re-baseline;
R@1 will drop on distractor count alone.
**12.3 Licensing review** — blocking for a *public* surface, not for
personal use. 1,342 Pinterest records still have no author.

### Explicitly closed — do not revisit

- Patch-level reranking: −30pt, mechanism understood.
- Score fusion: −6.2pt, mechanism understood.
- Category gate: net-negative six independent times.
- ArcFace: −13pt vs SupCon; per-class centroids cannot be estimated from
  ~4.5 images each. Not a hyperparameter gap.
- `--top-identity-candidates` beyond 150: 0.0016 pt/candidate.

## The honest summary

The retrieval core is strong and largely finished. What is missing is
**breadth, not depth**: two of four query types unserved, no brand
evidence, and detection that does not survive the real input modality.
None of these need a better encoder — 59.92% R@1 is not the bottleneck
for any of them.

Outfit reasoning has moved from "not built" to "built on thin, unlabelled
evidence" (11.1). The thing now limiting it is the same detector that
limits 11.2: it recovers 1.85 garments from photos containing four or
five, so half the outfit corpus contributes nothing to co-occurrence.
That makes 11.2 the highest-leverage item on this list — it is the only
one that would improve two boxes at once — and it still needs no better
encoder.
