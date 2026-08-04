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
| **Outfit-level results** | **Not built.** The final box of the spec diagram. `composed_query_search.py` returns two independent searches |

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

**11.1 Outfit-level results.** Run `index_outfits.py` over the 6,860
outfit photos on GPU, build a co-occurrence index, and replace
`/compose`'s two-independent-searches with retrieval against it. This is
the difference between "cargo pants exist in the catalog" and "this
jacket is worn with cargo pants." `modal_app_index_outfits.py` is written
and smoke-tested; the corpus run has not been started.

**11.2 Detection that survives a screenshot.** Fix `segment_outfit.py`'s
resize behaviour (`MAX_IMAGE_DIM=1024` shrinks a 1170×2532 screenshot's
subject to ~360×180, which is likely why SAM2 stops proposing per-garment
masks), then validate on a real labelled screenshot set. Until then the
Shortcut relies on a human crop, which currently beats the detector.

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
measured. The live follow-ups are **a logo detector** (a small trained
classifier over brand marks, which is what the 89% of photos OCR cannot
help with actually needs) and **brand evidence for open-set rejection**
rather than ranking — a brand read matching no catalog brand is positive
evidence the product is off-catalog, and that path badly needs a better
signal (AUROC 0.769).

### P2 — worth doing, not blocking

**12.1 Attribute heads** at query time, replacing offline LLM captions.
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
evidence, no outfit reasoning, and detection that does not survive the
real input modality. None of these need a better encoder — 59.92% R@1 is
not the bottleneck for any of them.
