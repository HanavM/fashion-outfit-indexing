# Phase 4 rerank improvement round — scope

Written 2026-08-03, immediately after the `--top-identity-candidates`
sweep closed at K=150 (`docs/eval_log.md`).

## Why this round exists

The sweep that dominated the last several sessions is finished, and it
finished by moving the bottleneck rather than by hitting a wall:

| K | R@1 (ungated) | shortlist miss | conditional R@1 |
|---|---:|---:|---:|
| 50 | 50.76% | 9.66% | 56.2% |
| 75 | 53.53% | 4.87% | 56.3% |
| 100 | 53.87% | 2.77% | 55.4% |
| **150** | **53.95%** | **1.18%** | **54.6%** |

**Shortlist miss is 1.18% — that stage is solved.** Conditional R@1
(accuracy given the true product IS in the candidate set) is flat-to-
declining across the entire sweep and sits at 54.6%. DINOv3 gets it wrong
nearly half the time *while holding the right answer*, and gets slightly
worse as the candidate pool grows, which is what you'd expect from a
harder per-candidate discrimination task.

So: every remaining point of R@1 has to come from **rerank quality**.
Widening retrieval is done. That is a different kind of work from the last
several rounds, and this document scopes it.

Two flags are already ruled out and must not be revisited as
"improvements": `--patch-rerank` (-30pt, confirmed harmful) and
`--score-fusion` (-6.22pt, confirmed harmful). The category gate is
net-negative on its fifth independent measurement. The pipeline's
remaining tuning surface is essentially exhausted; what's left is the
model and the gallery.

## Levers, cheapest first

### A. Raise the gallery image cap — no retraining, config only

Each product's DINOv3 prototype is the mean of its gallery image
embeddings, and that set was hardcoded to the **first 2 images**. The
median catalog product has **5** images (mean 5.6):

| cap | images used | products still truncated |
|---|---:|---:|
| **2 (current)** | **4,729** | **2,139** |
| 3 | 6,868 | 1,862 |
| 4 | 8,730 | 1,663 |
| 6 | 11,441 | 627 |
| all | 13,419 | 0 |

**65% of the gallery imagery is being discarded.** Two views is a thin
basis for a prototype that then has to separate colorway siblings.

This is now a flag: `GALLERY_IMAGES_PER_PRODUCT` (default 2, unchanged,
so nothing shifts unless asked). Changing it invalidates cached vectors
**per product** through the existing `gallery_signature`, so products with
2 images reuse their cache and only products that gain views re-encode.

**Not a foregone conclusion, which is why it's an experiment.** Product
galleries mix front views with detail crops, flat lays and back shots.
More views could sharpen the prototype (more angles, less per-shot noise)
or blur it (a zoomed fabric close-up dragging the mean off the garment).
The distribution above says a cap of 4 captures most of the available
signal while still excluding the long tail of accessory shots.

    GALLERY_IMAGES_PER_PRODUCT=4 python hierarchical_retrieval_pipeline.py \
        --evaluate --top-identity-candidates 150

Run 3, 4 and 6 against the K=150 baseline (53.95%). Cost: one re-encode
per setting, no training. **Do this first** — it is hours of GPU at most
and it either moves the number or rules out a whole hypothesis.

### B. ArcFace vs SupCon — already implemented, never benchmarked

`dino_identity_finetune.py` has `LOSS_TYPE = "supcon" | "arcface"` with
the ArcFace head fully written (margin 0.30, scale 30.0), and its own
docstring says the spec calls for benchmarking the two. **It has never
been run**: `arcface` appears zero times in `docs/eval_log.md`. Output
directories are already namespaced by loss type, so a run cannot clobber
the existing supcon checkpoint.

This matters specifically for the observed failure. SupCon pulls
same-identity images together without enforcing a *margin* between
identities; ArcFace imposes an explicit angular margin, which is the
standard fix for exactly this "right answer is present but ranked second"
behaviour in face/product re-identification. It is the single most
likely-to-work training change available, and the code cost is zero.

Note the ceiling this must beat: stage 2 (last-block unfreeze) bought only
**+0.44pt** over stage 1 (heads-only) on the supcon run, so most of what
supcon can extract is already captured by the projection head. That is
itself evidence the loss, not the capacity, is the binding constraint.

### C. Resolution and backbone — expensive, do last

DINOv3 ViT-B/16 at 224px gives 14x14 patches for a garment that may
occupy a third of the frame. Colorway and trim differences are exactly
the fine detail that resolution buys. Options, in cost order: 224 -> 336,
then ViT-B -> ViT-L.

Deliberately last: both change the checkpoint's compute profile, both
need a full retrain, and neither is testable in isolation until A and B
have said whether the problem is the prototype or the loss. Doing this
first would confound the cheap answers.

## What NOT to do

- **Do not re-litigate the category gate, `--patch-rerank` or
  `--score-fusion`.** All three are measured, all three are net-negative,
  and two have a mechanism recorded in `docs/eval_log.md`.
- **Do not push K past 150.** Marginal return is 0.0016 pt/candidate; the
  shortlist cannot give back more than the 1.18% it still misses.
- **Do not read cropping as a lever here.** `cropped_images` is consumed
  only by `build_color_index.py` — the retrieval pipeline, the DINOv3
  fine-tune and the SigLIP2 fine-tune all train and evaluate on raw
  images. Segmentation coverage cannot move R@1.

## Where each lever can actually run

The Modal `fashion-dataset` Volume is stale (6 brands, no
`finetuned_dinov3_identity_v1_supcon`), and this dev machine holds only a
partial catalog copy. **Colab Drive is the only place the current catalog
and both checkpoints coexist**, so A and B run there unless the Volume is
synced first. Lever C is a training job and is well suited to Modal, but
it needs that sync regardless.

Syncing the Volume is therefore a prerequisite for doing any of this on
Modal, and is worth doing once rather than per-experiment — commands are
in `modal_app_phase4_eval.py`'s docstring.
