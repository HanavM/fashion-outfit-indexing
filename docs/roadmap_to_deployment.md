# Roadmap to deployment — from CLI scripts to "Hey Siri, what does this go with?"

Written 2026-08-04. `docs/roadmap.md` covers Phases 0–5 (data, encoders,
retrieval, evaluation), which are substantially done. This document covers
what is left between here and a thing a person can actually use, and it
starts from a re-evaluation rather than from the old plan's assumptions.

## The end goal, stated concretely

> "Hey Siri, what does this jacket look like with cargo pants?"
> — while looking at a product page, a photo, or a screenshot.

Decomposed, that is: capture a screen region → detect the garment in it →
identify it against the catalog → compose it with a text-described second
item → return something speakable and viewable.

## Re-evaluation: what actually exists today

**Strong.** The retrieval core is real and measured, not aspirational.

| capability | state |
|---|---|
| Catalog | 2,387 records / 12 brands local, 1,234 / 6 brands on Colab+Modal |
| SigLIP2 semantic encoder | fine-tuned v3, 18.83% category-scoped R@1 |
| DINOv3 identity encoder | fine-tuned supcon, **59.92% R@1** end-to-end (2026-08-04) |
| Retrieval pipeline | dual-encoder, HSC gate, open-set rejection, GPU-resident |
| Real outfit photos | **6,860 records / 9,971 images** across reddit, wear.jp, pinterest |
| Multi-item detection | `segment_outfit.py` written, smoke-tested on 2 photos only |
| Compose (image + text) | `composed_query_search.py`, two independent searches |

**The gap that matters most: there is no serving layer at all.** Every
entry point in this repo is a CLI script that loads ~1GB of model weights
per invocation and prints to stdout. Nothing is callable over a network,
nothing keeps a model warm, nothing has an API contract. That is the
single biggest thing between the current state and any product surface,
Siri or otherwise.

**A constraint that just expired.** `composed_query_search.py`'s docstring
opens by explaining that it does two *independent* searches because "there
is no dataset yet of real outfit photos with linked/co-occurring items."
That was true when written. As of 2026-08-04 there are 6,860 such photos.
The honest v1 workaround it describes is now replaceable with the real
thing, and that is what turns "here are two unrelated products" into "here
is what actually gets worn together."

## Phase 6 — Serving layer (blocks everything else)

Wrap the pipeline in an HTTP service on Modal, since the checkpoints and
GPU already live there.

1. **Warm model container.** Load SigLIP2 + DINOv3 + all indexes once and
   keep them resident. A cold CLI invocation currently costs ~60s of
   imports and weight loading before it does any work; a voice assistant
   round trip has a budget closer to 2 seconds.
2. **Endpoints.** `POST /identify` (image → ranked products),
   `POST /compose` (image + text → primary item + companions),
   `GET /health`.
3. **Auth + limits.** A shared secret at minimum. This will sit on the
   public internet behind a Shortcut.
4. **Contract.** Typed request/response with confidence and the existing
   open-set rejection surfaced as a real field, so the client can say
   "I'm not sure" instead of asserting a wrong product. The rejection
   threshold work already exists; it must not be dropped at the API edge.

Success: `curl` an image, get JSON in under ~2s warm.

**Status: built and deployed, 2026-08-04** — `modal_app_serve.py`, at
`https://hanavm--fashion-serve-fashionservice-api.modal.run`. Measured
against a real catalog image (Nike AF1 `IR0273-100`, returned at rank 1,
DINOv3 score 0.979): **cold 16.8s** wall (13.1s of that is the container's
model+index load), **warm ~1.5s** wall / ~0.9–1.2s server-side, `/health`
~0.4s. Under the 2s target, but not by much — the budget is spent almost
entirely in the two encoder forwards, so anything Phase 7 adds in front of
this (screenshot segmentation) comes out of ~0.5s of headroom.

Caveats this does NOT resolve:
- It ships the real pipeline file via `add_local_file`, deliberately, so
  it cannot drift the way `hierarchical_retrieval_pipeline_modal_body.py`
  did. Keep it that way.
- `rejected_open_set` is surfaced but the threshold is still uncalibrated
  (`reject_threshold_calibrated: false`), so the default response never
  rejects. Running `--evaluate --open-set-holdout-fraction 0.1` is what
  turns that field from plumbing into a real behaviour, and Phase 9's
  "degrade gracefully" item depends on it.
- The smoke test's query image was in the gallery, so rank-1 there
  confirms the code path, not accuracy. The accuracy number is still
  eval_log.md's 59.92% R@1.
- Volume is still the 6-brand / 1,234-product catalog (1,077 products
  reach the gallery), i.e. the Phase 9 catalog-size reset is untouched.

## Phase 7 — Siri / iOS client

1. **Shortcut**: accept a screenshot or shared image + a spoken phrase,
   POST to the API, speak the top result and show the alternatives.
2. **Screen capture path.** "Seen on screen" means the input is a
   screenshot of arbitrary UI, not a clean catalog photo — the garment is
   surrounded by page chrome, text, and other products. This is where
   `segment_outfit.py` becomes load-bearing rather than experimental.
3. **Speakable output.** The response needs a one-sentence spoken form
   ("a Dickies 874 work pant, and it goes with…") separate from the rich
   visual list.

## Phase 8 — Real outfit intelligence (newly unblocked)

This is what makes the answer good rather than merely correct.

1. **Run `segment_outfit.py` over `outfit_dataset/`** (`index_outfits.py`
   exists uncommitted for exactly this) to turn 6,860 photos into
   multi-item outfit records. GPU work — belongs on Modal, and the
   Dickies segmentation app is a working template.
2. **Build a co-occurrence index**: which garment types, colors and styles
   actually appear together on real people. That is the difference between
   "cargo pants exist in the catalog" and "this jacket is worn with cargo
   pants, and here are the ones that work."
3. **Replace `composed_query_search.py`'s two-independent-searches
   workaround** with retrieval against that index.

Caveat to carry: these photos are unlabeled by deliberate decision, so
everything here is model-derived with no ground truth. Write it into
namespaced fields and never let it contaminate `apparel_dataset`.

## Phase 9 — Hardening before anyone else sees it

1. **Latency budget** end to end; cold-start behaviour on Modal.
2. **Failure modes**: no garment detected, low confidence, API down. The
   assistant must degrade gracefully, not assert nonsense.
3. **Licensing review — genuinely blocking for a public surface.**
   `SCRAPING_PROCESS.md` has carried a standing note since the outfit
   dataset was designed: these are photos of real people, collected under
   varying terms, and 1,342 Pinterest records currently have **no author
   at all**, so they are not traceable for takedown. Using them as
   internal training/eval data is one thing; displaying them in a product
   is a different question that must be answered before launch, not after.
4. **Catalog-size baseline reset.** Every number above is on a
   1,234-product gallery. Pushing the six local-only brands makes it
   2,387 and R@1 will drop on distractor count alone — a new baseline
   block, not a regression.

## Ordering, and why

Phase 6 first, because nothing is demonstrable without it and it is the
only phase that blocks all the others. Phase 7 next, because a working
end-to-end path — even with a mediocre answer — is worth more than a
better answer nobody can invoke. Phase 8 improves answer quality once the
loop exists. Phase 9 is the gate before anyone outside sees it.

Model quality work (ArcFace, resolution, backbone) continues in parallel
per `docs/rerank_improvement_scope.md` and is deliberately NOT on this
critical path: the serving layer is indifferent to which checkpoint it
loads.
