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
| **Logo detection and OCR** | **Not built at all.** Zero references in the codebase. This is spec §4.5's entire "brand evidence" path |
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

**11.3 Brand evidence — logo detection and OCR.** Spec §4.5, entirely
absent. Most product photos carry legible wordmarks and most screenshots
carry the brand in page text. This is likely the single largest
*untried* accuracy lever left, precisely because it uses a signal the
current pipeline throws away — and unlike patch-rerank and score-fusion,
it adds information rather than re-weighting what is already there.

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
