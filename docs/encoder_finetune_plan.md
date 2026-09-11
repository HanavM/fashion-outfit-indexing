# Fine-tuning SigLIP2 on outfit photos — what the labels actually support

Written 2026-09-11. Nothing here was run. This is a costed plan plus the
measurement that says whether it can work, and the export script that
feeds it is `prepare_outfit_finetune_data.py`.

Read `docs/SESSION_HANDOFF.md` §4 first for why the encoder is the target.
Short version, all measured: mean pairwise similarity between two random
outfit photos is **0.743**, a real query's top-1 scores 0.203 against
top-50 at 0.187, and **"a photo of a bicycle" still scores 0.155**. Most of
each vector encodes "a person standing in a room". The deployed checkpoint
was fine-tuned on catalog product photos — one garment, white sweep — and
is out of distribution on a phone photo of somebody's bedroom.

---

## 1. Correction to the premise, stated first

The task that produced this document described
`outfit_dataset/product_labels.json` and `product_labels.bbox.json` as
"gpt-4o-mini vision labels". **They are not.** They are the deployed
identifier's output: for each detected garment crop, the ranked CATALOG
PRODUCTS that SigLIP2 + DINOv3 matched it to, carried alongside the
detector's own category/group and (in the bbox file) the heuristic palette
colour. `label_outfit_products.py`'s own docstring says so.

Only `outfit_eval/vlm_labels.json` is gpt-4o-mini, and it covers **4,431
photos**, not the corpus.

This matters because it changes the answer. It is the difference between
"36,550 vision-labelled crops" and "4,431 photos labelled by an
independent judge, plus 36,550 crops annotated by the very stack we are
trying to improve".

---

## 2. The three label files, exactly

| file | records | distinct photos | schema |
|---|---:|---:|---|
| `outfit_dataset/product_labels.json` | 15,869 | 6,391 | `id, post_id, rel, bbox, detected_category, detected_group, detected_color, detector_confidence, parser_score, post_url, source, crop_style, products[], top1_score, margin, garment_gate, rejected_open_set` — all 17 keys on 100% of records |
| `outfit_dataset/product_labels.bbox.json` | 20,681 | 7,806 | the same minus `parser_score` and `crop_style` (15 keys, 100%) |
| `outfit_eval/vlm_labels.json` | 4,431 | 4,431 | `garments: [{garment, colour}], is_outfit_photo` — 2 keys, 100% |

**Overlap.** Zero shared crop `id`s between the two product files (their id
schemes differ), but **15,795 identical `(rel, bbox)` pairs** — so the
masked file is very nearly a re-run of the bbox file over the same
detections with the crop style changed, which is what it was for. Photo
overlap: masked ∩ bbox = 6,391 (total containment), masked ∩ VLM = 1,974,
all three = 1,974. **Union of distinctly-labelled photos: 10,263.** Union
of distinctly-labelled crops (by `(rel, bbox)`): 20,755.

**Files on disk.** Every source photo referenced by every label record is
present: 15,869/15,869, 20,681/20,681, 4,431/4,431, and 43,712/43,712 of
`metadata.json`'s detections. **Photo availability is not the constraint.**
Masked crop availability is — see §4.

**Distributions, and the skew.**

| | masked file | bbox file | VLM |
|---|---|---|---|
| groups | bottoms 29.7 / footwear 25.5 / tops 24.4 / outerwear 14.0 / accessories 6.4 | 29.6 / 24.6 / 23.8 / 14.7 / 7.4 | — |
| categories | 13 classes; pants 26.3%, sneaker 23.7%, jacket 14.0%, socks 0.1% | 13 classes; pants 26.2%, sneaker 22.5% | 13 garments; pants 25.8%, sneaker 16.9%, sweatshirt 0.8% |
| colours | **`detected_color` is `None` on all 15,869** | 19 palette names; **black 41.0%**, white 13.0%, charcoal 11.5%, green 0.1% | 19 names; black 28.5%, white 16.7%, purple 0.2% |

Two things fall out. The masked file has **no colour at all** — that pass
did not store one — so the richer of the two crop files is category-only.
And the bbox file's colour is the heuristic palette vote, measured at
**38.5% agreement** with the VLM with one-directional errors (brown/navy/
blue → black). That is the exact bug `train_colour_head.py` exists to fix.
It is not a training label.

---

## 3. What can form an image–text pair, and what cannot

| candidate text | rows | verdict |
|---|---:|---|
| VLM whole-outfit caption per photo | 3,948 photos, **2,812 distinct captions, 2,345 singletons** | **the strongest signal available** |
| VLM `<colour> <garment>` per crop | 7,851 aligned, **189 distinct strings**, top-10 = 50.6% | usable, weak vocabulary |
| detector category per crop | 15,869, 13 classes, 26% "pants" | opt-in; almost no contrastive information |
| bbox file's palette colour | 20,681 | **no** — 38.5% agreement, and it is the incumbent's bug |
| top-1 catalog product name | 36,550 | **no** — see below |

**Why the product names are not a training label, with the number.** Best
measured R@1 for the identifier is **47.65%**, and that is catalog-photo →
catalog-product, a strictly easier condition than a consumer crop.
`rejected_open_set` is `True` on **0 of 36,550** records, because the
threshold is uncalibrated (AUROC 0.769, no usable operating point), so
nothing filters the guesses. Median `top1_score` is 0.811 (masked) and
0.759 (bbox) with a median margin of ~0.030 — confident-looking scores
with no separation. Training a contrastive objective on "this crop IS a
Gap 10.5in Baggy Denim Short" when that is close to a coin flip teaches
the encoder noise, and the noise is *systematically* aligned with the
encoder's existing mistakes, which is the worst shape for self-training.

**The crop view's real problem is its vocabulary, not its size.** 7,851
pairs sounds fine until you notice there are 189 strings among them. In
the exported train split, **100.0% of rows share their text with another
row in the same split.** An in-batch contrastive loss will treat two
black-pants crops as negatives of each other. This is not fatal — it is
exactly what `dino_identity_finetune.py`'s SupCon setup already handles,
and SupCon beat ArcFace by 13.4pt at this project's data scale — but a
plain CLIP/SigLIP in-batch objective on this view will mostly train on
false negatives. The manifest emits `text_id` on every row so the trainer
can mask them.

**The photo view does not have that problem**: 2,722 distinct captions
over 3,664 rows, 1,747 singleton labels in train, top caption 0.9%. And it
attacks the measured failure head-on — the 0.743 similarity is between
*whole photos*, which is what this view trains.

---

## 4. Masked crops: recoverable, and that is the honest answer

The measurement that makes this section matter (`crop_style_ab.py`, paired,
237 garments): group agreement **40.9% bbox → 72.2% masked**, footwear
**15.8% → 87.7%**. A bbox drags the floor, the wall and the other leg into
the encoder.

A masked crop is produced by `garment_proposer.propose_garment_items`, not
by `segment_outfit.py`: the ATR human parser labels each pixel's garment
class, connected components become regions, `crop_masked_region` cuts the
region's bbox, and then `arr[~sub] = 255` blanks every pixel outside the
mask **inside that bbox**. `segment_outfit.crop_masked_region` on its own
returns the raw bbox crop — the masking is the proposer's line.

**Do the stored labels correspond to masked crops?** `product_labels.json`
is `crop_style: "masked"` on all 15,869 records; `product_labels.bbox.json`
is bbox on all 20,681. So the labels themselves are correctly attributed.

**Are the mask *images* recoverable?** Yes, two ways, and one of them is
free right now.

1. **10,571 of 43,712 detections already have their masked crop on this
   disk** (`detected_items[].crop_path`, 3,709 photos). Verified rather
   than assumed: sampling 200 of them, mean pure-white pixel fraction is
   **0.330** (median 0.313, 176/200 above 20%) and crop dimensions match
   the stored bbox on 200/200. Those are masked crops, and they line up
   with the current detections. *`outfit_search.py::load_crop_records`'s
   docstring says "only 13 of 20,681 exist locally" — that is stale; it was
   true of the `product_labels` subset and is not true of the corpus.*
2. **The rest are regenerable, locally, for free.** Every one of the 43,712
   source photos is on disk, and the proposer is CPU/MPS — no GPU, no API.
   `crop_style_ab.py` ran 100 photos in 8.9 min *including* two API calls
   per crop; the proposer alone is ~1 photo/s, and `outfit_search.py`
   already costs a full-corpus regeneration at **~4 hours**. For the
   2,100 photos this plan actually needs, it is **~35 minutes**.

So: **a fine-tune can use masked crops.** It needs one free local pass
first, which is what `--materialise-crops` is. Until that pass runs, 4,252
of the 7,377 crop rows carry `image_kind: "pending_mask"` and are excluded
from training rather than quietly downgraded to bboxes; `--allow-bbox-crops`
downgrades them explicitly and tags them `image_kind: "bbox"` so a trainer
can weight or drop them.

---

## 5. The export script, and its dry-run numbers

`prepare_outfit_finetune_data.py` — local files in, JSONL manifest out. No
GPU, no paid API, no Modal. Split by **photo** (matching
`train_colour_head.py`), one shared split across both views so a photo's
crops never sit opposite its own full frame; verified 0 photos straddle a
split in the written manifest.

```
.venv/bin/python prepare_outfit_finetune_data.py --dry-run
```

```
43,712 detections · 43,712 with the photo on disk · 10,571 with a MASKED CROP on disk
4,431 VLM-labelled photos, 4,015 judged real outfits

rows 11,041  ·  photos 3,765  ·  distinct texts 2,807
  view crop     7,377   3,550 photos      85 texts
  view photo    3,664   3,664 photos   2,722 texts
image kind: pending_mask 4,252  photo 3,664  masked 3,125
split (by PHOTO, 3,781 photos): train 7,759  val 1,633  test 1,649
TRAINABLE NOW (photo frames + masks on disk): 6,789
pending a mask (regenerable free, --materialise-crops): 4,252

crop view, train (5,198 rows, 85 labels):
  black pants 9.9% · white sneaker 8.1% · black sneaker 7.6% · blue pants 6.7%
  top-10 53.8% · sample_weight 3.18–29.43 · 100.0% of rows share their text

photo view, train (2,561 rows, 2,025 labels):
  top caption 0.9% · top-10 4.0% · 1,747 singleton labels · 31.8% share a text

skipped: crop/no_matching_crop 3,543 · not_an_outfit_photo 416 ·
         photo/too_few_garments 351 · crop/ambiguous_colour 84
dropped: crop/rare_label 474 (104 labels under --min-per-class 12)
```

**So the real training-set size is 6,789 pairs today, 11,041 after the free
local mask pass.** Not 36,550. The label count was never the number.

Imbalance is handled three ways, all reported: `--min-per-class` (default
12, matching the colour head) drops labels too rare to evaluate honestly;
`--cap-per-class` truncates over-represented labels in **train only** (at
200 it takes the crop view's top-1 from 9.9% → 4.8% and top-10 from 53.8%
→ 42.8%); and every class-view row carries
`sample_weight = (N/count) ** 0.5`, the same square-root inverse frequency
`train_colour_head.py` used after "black" at 30% pulled an unweighted head
into predicting it. Free-text photo captions get weight 1.0 — there is no
class there to rebalance, and applying the class rule to them deleted 97%
of the view in the first draft.

---

## 6. Cost and time for the actual run, and the blocker

Rates: A10G at **$1.10/hr**, the constant this repo already uses
(`label_outfit_products.py`). Modal bills **wall time**, not requests —
two measured runs both landed at ~$1.20 per wall-clock hour (17,682
garments in 16.2 min = $0.27; 15,846 in 115.7 min = $2.35), the difference
being a slow local stage inside a warm container.

| step | where | time | cost |
|---|---|---|---|
| Export the manifest | local, CPU | seconds | **$0** |
| Materialise 4,507 masks (2,100 photos) | local, CPU/MPS | ~35 min | **$0** |
| Upload ~11k crops to the Volume | Modal CPU | minutes | ~**$0.02** |
| Fine-tune, stage 1 (heads, 8 epochs) | A10G | ~45–60 min | $0.85–1.10 |
| Fine-tune, stage 2 (last 4 blocks, 6 epochs) | A10G | ~85–120 min | $1.60–2.20 |
| Per-epoch validation retrieval | A10G | included above | — |
| Re-embed the corpus with the new checkpoint | A10G | ~40–60 min | $0.75–1.10 |
| Re-run `outfit_search_eval.py` | local + cached VLM labels | ~10 min | $0 |
| | | **~3–4 h** | **$3.20–$4.45** |

Basis for the training rows: the corpus photos are already on the Volume
(20,545 images, 0 missing), so there is no re-upload. Throughput is
extrapolated from this repo's own measurement of **1.2 s per 32-image
batch** for a SigLIP2-384 forward on an idle A10G (≈26 img/s), with
backward on a frozen backbone roughly forward-dominated and stage 2 at
~9–12 img/s. The epoch counts are v4's actual `STAGE1_EPOCHS=8` /
`STAGE2_EPOCHS=6`. The v3/v4 runs used a 10-hour Modal timeout on ~4,811
train images; this dataset is ~7,800 rows, the same order.

**The blocker.** Spend is **$28.13 of a $30 limit**. Remaining: **$1.87**.
One run of the table above is $3.20–$4.45. **A single fine-tune does not
fit, and neither does the re-embed that would be needed to measure it.**
The outfit-serve container also bills while warm on a 5-minute scaledown,
so the headroom is smaller than $1.87 in practice.

Nobody gets a fine-tune right on the first attempt — v4 was a measured
regression on every metric against v3 — so budget for **two or three
runs plus their re-embeds: $10–15**. Raising the limit to $50 is the
concrete ask. This is a budget decision, not a technical one, and it
should be made before any of §7 is started.

---

## 7. What to do, in order

1. **Label the rest of the corpus with the VLM first — it is the cheapest
   5x available and it is not Modal money.** 4,015 outfit photos are
   labelled of 20,532. At the measured 3.26 photos/s (938 in 4.8 min, 8
   workers, 2 failures) the remaining ~16,500 take **~85 minutes**, and at
   gpt-4o-mini vision rates that is roughly **$5 of Azure**. It takes the
   photo view from 3,664 pairs to ~19,000 and the crop view from 7,377 to
   ~38,000. Do this before spending a dollar of GPU on 6,789 rows.
2. **Materialise the masks locally.** Free, ~35 min for what this manifest
   needs, ~4 h for the whole corpus.
   `nohup .venv/bin/python -u prepare_outfit_finetune_data.py
   --materialise-crops --resume &` — it checkpoints on elapsed time, which
   is the discipline that a 40-minute crop build turning into 10.5 hours
   and four kills bought.
3. **Raise the Modal limit.** §6.
4. **Then train**, photo view as the primary objective and crop view as an
   auxiliary under a multi-positive / SupCon loss, never a plain in-batch
   contrastive one (§3).
5. **Measure with `outfit_search_eval.py` on the held-out photos only.**
   The split is in the manifest. The current number to beat is
   **precision@15 46.0% micro** (garment 72.5%, colour-given-garment
   63.5%).

---

## 8. Honest limits

- **The ceiling of this fine-tune is "reproduce gpt-4o-mini's reading of
  the photo".** Every text in the manifest is machine-generated. The judge
  is independent of the retrieval stack — no shared weights, no shared
  training data — which is what makes it worth training on, but it is not
  ground truth and it miscalls colours under coloured lighting.
  `pair_eval.py` is the human instrument and has still never been run to
  completion.
- **Training and evaluating on the same labeller is circular** unless the
  split is respected. `outfit_search_eval.py --held-out-only` exists for
  exactly this and must be used.
- **7,800 train rows is small, but not disqualifying by this project's own
  history**: v3 fine-tuned on 4,811 catalog images and moved R@1 from
  7.89% to 18.83%. The risk here is not sample size, it is that 85 of the
  training texts carry most of the crop-view mass.
- **`outerwear` sits at 41% group agreement even with masked crops**, and
  it is structural — ATR has no Coat class, so a jacket and the tee under
  it come back as one region. No amount of fine-tuning fixes a region that
  contains two garments.
- **Brand-targeted sourcing biases this optimistically.** Photos found by
  searching "carhartt detroit jacket outfit" over-represent that jacket
  relative to what a user photographs, and the same bias is in the
  training set and the eval set.
- **Nothing in §6 has been run.** The times are extrapolations from this
  repo's measured throughput, not from a fine-tune that happened.
