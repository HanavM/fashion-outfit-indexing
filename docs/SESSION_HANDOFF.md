# Session handoff — 2026-09-11

Read this first. Then `docs/eval_log.md` for every number, and
`docs/production_plan.md` (whose top section carries the product
correction that reframed the whole project).

---

## 0. 2026-09-11: the app runs, and it never could have worked

**The single most important fact: outfit search now works end to end from
the iOS app, and before today it could not have, under any circumstances.**

The app had been written, compiled and never launched. Launching it found
three independent faults, each fatal on its own, all invisible to a build:

1. **One base URL for two services.** `/outfit_search` was being sent to
   the *identifier* host, where that route does not exist. The client
   turned the 404 into `.notDeployed` and the UI reported "Outfit search is
   not deployed" — accurately, in its own terms, for a month, while
   `outfit-serve` was up the whole time. There are now two URLs.
2. **The photo URL was wrapped twice.** The service emits
   `image_url = "/photo?path=<rel>"`; the client only checked for an
   *absolute* URL, fell through to its bare-corpus-path branch and built
   `/photo?path=/photo%3Fpath%3D...`. Every photo, every time.
3. **`AsyncImage` cannot authenticate.** `/photo` answers 401 without a
   Bearer header like every other route, and `AsyncImage` has no way to
   attach one — so every tile said "Photo unavailable" on top of a search
   that had already succeeded. `AuthorizedImageLoader` now carries the key,
   caches, and de-duplicates in-flight requests. **It attaches the header
   only to known hosts**: `image_url` may point at third-party corpus URLs
   (reddit previews among them), and the key must not go to strangers.

The lesson generalises past iOS: **every one of these is a seam between two
components that were each correct.** Compiling proved the Swift was
well-formed. Health checks proved the services were up. Neither could see
that the app was talking to the wrong one, and nothing in the project's
evaluation stack covers the client at all.

### Judged by eye, which was the other open item

`a black leather jacket with baggy jeans` — 4/4 on-query, and genuinely
good. `a green jacket with white sneakers` — 4/4 green, but *white
sneakers* is clearly satisfied in only 1 of 4. That is the documented
decomposition showing up in the wild (garment 72.5%, colour-given-garment
63.5%), and it is the **second clause** that gets dropped.

Two corpus problems visible in four tiles: one result is a **womenswear
catalog product shot** in a corpus that is supposed to be men's-only and
real people — `drop_womens` defaults to *false* in the API — and another
is a flat-lay rather than a person. Both are ranking-visible at position 4
of the very first queries tried, so they are not rare.

### A claim I made today and had to withdraw

Commit `0ca42ce`'s message says `product_labels.json` and
`product_labels.bbox.json` are "gpt-4o-mini output". **They are not, and
the message is wrong.** They are the deployed *identifier's* output —
ranked catalog products per crop from SigLIP2+DINOv3 — plus the detector's
own category/group. `label_outfit_products.py` says so in its first line.
The only VLM labels in the project are `outfit_eval/vlm_labels.json`
(4,431 photos), which was already tracked.

Committing them was still right: they are ~36,550 crops of deployed-pipeline
output that cost GPU wall time to regenerate, against $1.87 of remaining
budget. The reasoning was wrong, not the action. Two consequences worth
carrying:

- the masked file's `detected_color` is `None` on **all 15,869** records,
  and the bbox file's colour is the heuristic palette vote measured at
  38.5% agreement — the very bug the colour head exists to fix. **Neither
  is a colour training label.**
- the two files share **15,795 identical `(rel, bbox)` pairs**: the masked
  file is a re-run of the bbox file with the crop style changed, not a
  second population.

### Masked crops are on disk, and the docstring said otherwise

`outfit_search.py::load_crop_records` claimed "only 13 of 20,681 exist
locally". True of the `product_labels` subset, badly misleading about the
corpus: **10,571 of 43,712 detections have their masked crop on disk**
(counted 2026-09-11, not estimated). Docstring corrected. This matters
because masked-vs-bbox is the 72.4%-vs-37.7% measurement, so "the masks are
gone" would have silently steered the next person to the worse crop.

### The encoder: planned, costed, and deliberately not run

`prepare_outfit_finetune_data.py` + `docs/encoder_finetune_plan.md`.

The usable training signal is **much smaller than the label count
suggests**: 11,041 pairs over 3,765 photos (6,789 trainable today, 4,252
pending a free local mask pass), not 36,550. Catalog product *names* are
excluded on purpose — R@1 is 47.65% catalog-to-catalog, `rejected_open_set`
is True on 0 of 36,550 because the threshold is uncalibrated, so more than
half those texts are wrong *and wrong in a way aligned with the encoder's
existing mistakes*. Training on them would teach the model its own errors.

Split by photo, 0 straddling. The crop view carries only **85 distinct
texts** across 7,377 rows, so **100% of train rows share their text with
another row** — a plain in-batch contrastive loss there trains mostly on
false negatives. The script emits `text_id` for masking and the plan routes
it to a multi-positive SupCon-style loss.

**Cost: ~3–4 h and $3.20–$4.45 per run at the repo's own $1.10/hr A10G.
Budget is $28.13 of $30, so $1.87 — one run does not fit, and neither does
the re-embed needed to measure it.** v4 was a measured regression against
v3, so budget 2–3 runs plus re-embeds: **$10–15**, i.e. raise the limit
to $50.

**The recommendation is to NOT train yet**, and it is an ordering argument
rather than a doubt: 4,015 of 20,532 photos are VLM-labelled, and labelling
the remaining ~16,500 costs ~85 min and roughly **$5 on Azure** — *not*
Modal money — taking the photo view from 3,664 to ~19,000 pairs. Spending
$4 of a $1.87 budget to train on 6,789 rows, when $5 of non-Modal money
buys five times the data, is the wrong order.

### Still not verified in the app

Camera, closet capture, Keychain entry via Settings (the key was injected
through the scheme environment variable instead), the Siri App Intent, and
the try-on surface. Running it once does not mean running all of it.

---

## 1. What the product is

**Outfit search.** A photo of an item plus words for what it should be
worn with, returning **real outfit photos of real people**.

It was an *item identifier* until 2026-08-05, when the owner saw the
working UI and said: *"the idea is you give a picture and a text and you
get returned a bunch of images of people's outfits that satisfy picture
and text. not at all what i was thinking. this just gave me what it
thought the item was."*

Every doc in the repo described the identifier, which is why it got built
first. The identifier still works and is still deployed — it is now at
most a **feature** ("what is this item", "shop this look"), not the
product.

---

## 2. State

| | |
|---|---|
| Catalog | **4,079 records / 20 brands**, all men's |
| Outfit corpus | **14,684 posts / 20,532 photos** |
| Detected garments | **43,712** |
| Skin tone measured | 9,515 of 14,684 |
| Untraceable Pinterest records | **7** (was 1,342) |
| Modal spend | **$28.13 of a $30 limit** — this is now tight |
| Local disk | 14 GiB free (it has hit zero twice) |

### Deployed services

| endpoint | what |
|---|---|
| `hanavm--outfit-serve-outfitservice-api.modal.run` | **outfit search** — `/outfit_search`, `/photo`, `/health`. CPU. 204–426 ms warm, 24.6 s cold |
| `hanavm--fashion-serve-fashionservice-api.modal.run` | **identifier** — `/query` `/identify` `/compose` `/search` `/health`. A10G. ~700 ms warm |

Auth on both: `Authorization: Bearer $FASHION_API_KEY` (in `.env`,
gitignored, never print it).

### Local surfaces

```bash
.venv/bin/python outfit_search.py serve   # the product, localhost:7880
.venv/bin/python try_it.py                # the identifier, localhost:7860
.venv/bin/python pair_eval.py label       # human labelling, localhost:7870
```

**Always `.venv/bin/python`, never `python3`.** System Python has torch
2.0.0; `transformers` disables its PyTorch backend below 2.4 and then
fails with an error that looks nothing like a wrong-interpreter error.

### iOS app — `ios/`

30 Swift files, no third-party dependencies, **builds clean in Debug and
Release** (verified independently with `xcodebuild`). Closet, Search,
Result detail, Feed, App Intents/Siri, and a try-on surface. Every screen
previews against a stub, so none needs the network.

**Run 2026-09-11 for the first time** — see section 0. It builds, launches,
renders, and completes a real `/outfit_search` against the live service
with photographs, after three fixes. Deep links work (`fashion://search?q=`
is how the search was driven). Camera, Keychain-via-Settings, the Siri
snippet and try-on remain unverified.

To drive it from a terminal without touching Xcode:

```bash
xcodebuild -project ios/Fashion.xcodeproj -scheme Fashion \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' build
xcrun simctl install <udid> <path>/Fashion.app
SIMCTL_CHILD_FASHION_API_KEY="$FASHION_API_KEY" \
  xcrun simctl launch <udid> com.hanavm.fashion
xcrun simctl openurl <udid> "fashion://search?q=black%20leather%20jacket"
```

`SIMCTL_CHILD_` is the whole trick: it puts the key in the app's
environment, which `APIConfiguration` already checks first, so no Keychain
or xcconfig is needed. `openurl` raises a system "Open in Fashion?"
confirmation — its default button is Open, so a Return keystroke through
`osascript` clears it without any coordinate maths.

---

## 3. How outfit search actually ranks

Multimodal: any number of images and phrases, either alone. **Each part is
matched to a different garment in the photo.**

```
part_score  = max over the photo's crops of sim(part, crop)   # each crop claimed once
photo_score = mean of the parts' scores
```

Scoring every part against the photo's *best* region would let one
jacket-that-is-vaguely-jeans-like satisfy "this jacket with baggy jeans",
which is not the question.

**Encoder: SigLIP2 only. DINOv3 is deliberately absent.** The identity
fine-tune bought +31.3pt by pushing colorway siblings *apart* — right for
"which exact product is this", wrong for "show me outfits like this".
DINOv3 also has no text tower, so there is nothing for "baggy jeans" to
align against; SigLIP2's towers are trained into one shared space, which
is why both halves of a query use it.

**Structural attribute binding** (the fix for "yellow shoes returned
yellow jackets"): each text clause is parsed for a garment *group*,
a *category* and a *colour*, and crops satisfying more of them are
rewarded. Preferences, never filters — the query classifier is 88%
accurate and a hard filter would give the other 12% nothing but wrong-type
crops.

A sentence is split on with/and/plus/over/commas so "jorts with red shoes"
becomes two constraints. Splitting only fires when clauses name *different*
garment types, so "a black and white jacket" stays one.

---

## 4. Everything measured (the numbers that should stop you re-deriving)

### Outfit search has an evaluation now — `outfit_search_eval.py`

gpt-4o-mini vision (Azure Foundry) labels outfit photos; queries are scored
against those labels. **4,431 photos labelled.**

| precision@15, colour+garment | micro |
|---|---:|
| no attribute binding | 40.9% |
| **with binding (shipped)** | **46.0%** |

Decomposed: **garment correct 72.5%**, **colour correct given the garment
63.5%**. Colour is the weaker half.

Method notes that matter: results are ranked **within the judged pool**
(a plain top-20 gave 1–5 judged results per query and the aggregate was
noise), so the absolute value is optimistic and **the delta across a change
is the trustworthy part**. And it is one model grading another — but the
judge is *independent* of the retrieval stack, which the old
group-agreement proxy was not.

### Things that failed, measured — do not retry these

| attempt | result |
|---|---|
| match query colour in CIELAB not by name | 46.0 → 46.4%, noise |
| re-derive crop colours from `mean_rgb` in CIELAB | 38.5 → 26.7%, **worse** |
| vote in CIELAB on real pixels | 28.6 → 25.6%, no better |
| raise the binding weight 0.05 → 0.40 | byte-identical results (saturation, not a bug — verified) |
| **learned colour head in retrieval** | 39.5 → 40.0%, noise |

The colour head *is* +10.5pt as a classifier (37.7% → 48.2% held-out) and
is deployed, because it is strictly more accurate and free at inference.
**But it does not move retrieval**, and that is the important finding:
colour only breaks ties among crops the encoder has already ordered.

### The encoder is the binding constraint

- mean pairwise similarity between random outfit photos: **0.743**
- "baggy jeans and a black jacket": top-1 0.203 vs top-50 0.187
- **"a photo of a bicycle"** still scores 0.155

Most of each vector encodes "a person standing in a room", not the
clothes. More corpus buys recall; it cannot fix this. SigLIP2 v3 was
fine-tuned on catalog product photos and is out of distribution on people
in bedrooms. **Fine-tuning it on outfit photos is the next real lever, and
it needs labels.**

### Crops: masked beats bbox, enormously

| | group agreement | footwear |
|---|---:|---:|
| plain bbox crop | 37.7% | 15% |
| **proposer's masked crop** | **72.4%** | **89%** |

A bbox drags the floor, the wall and the person's legs into the encoder.
Measured paired on 237 garments, replicated at n=15,854.

### Skin tone is RELATIVE, deliberately

Absolute Monk-scale binning is broken here and was not shipped: the
swatches for lighter tones sit at **L\* 78–94** while photographed skin in
this corpus measures **L\* 11–61**, so half the scale is unreachable. The
first 12 photos all landed 6–10; after correction 26 of 40 still collapsed
onto one tone. The filter compares a photo against a reference, where the
shared lighting bias cancels. **Never label it with tone categories.**

---

## 5. Production bugs found this session — do not reintroduce

1. **Serving ran `TOP_IDENTITY_CANDIDATES=25` while every eval used
   150/400.** A module constant that `--evaluate` passed explicitly and
   `modal_app_serve.py` never did, so the API shortlisted 25 identities out
   of 1,562 and routinely never saw the true product. Now env-overridable,
   serving sets 400. **400 is a FLOOR, not tuned** — it was measured at a
   2,230-product gallery and K scales with catalog size; the gallery is now
   3,922.
2. **HTTP 422 killed whole scraper shards.** Arctic Shift returns it
   intermittently on deep cursors; it was not in the backoff set.
3. **Serial image loading on a network volume.** `encode_images` opened
   files one at a time; on the Modal volume that is 14.9 s per 32-image
   batch with the A10G *idle* — slower than a laptop. Parallelised to
   1.2 s/batch, turning a $2.94 job into $0.32.
4. **`sam2` imported at module scope** in `index_outfits.py` and
   `segment_outfit.py`, making `dominant_color` (pure PIL/numpy)
   unimportable wherever sam2 is absent. Now lazy.
5. **495 photos re-processed forever.** A record with `detected_items: []`
   looks identical to one never tried; they now carry `detection_meta`.

---

## 6. Operational gotchas (each cost real time)

- **`modal deploy` does NOT cycle a warm container** — old code AND old
  secrets survive. Always `modal app stop <app> --yes && modal deploy`.
- **Modal cost is WALL TIME, not request count.** The container stays warm
  between requests, so a slow *local* stage in the loop is billed. Two
  runs: 17,682 garments in 16 min = $0.27; 15,846 in 116 min = **$2.35**.
  Precompute, then send densely.
- **`modal volume put -r` corrupts intermittently.** One tar per chunk,
  plus an `extract` function on the far side.
- **Long local GPU jobs get killed.** A 40-minute crop build took 10.5
  hours and died four times: macOS ran `ANECompilerService` at 98% and
  `BackgroundShortcutRunner` at 64%, load average 24, competing for the
  silicon MPS needs. Run detached (`nohup … & disown`), checkpoint on
  **elapsed time** (batch-count intervals stretch past the kill interval
  under load), and make jobs incremental.
- **One writer at a time on `metadata.json`.** Local passes own
  `author`/`source_link`/`skin_tone`; Modal detection owns
  `detected_items`/`detection_meta`. `modal_app_outfit_pipeline.py::fetch`
  merges per field rather than overwriting — a straight overwrite is how
  175 `structured_caption` fields were lost once.
- **`pgrep -f "<script> <arg>"` matches its own waiting shell.** An
  `until ! pgrep …` loop never exited because of this.
- **Disk.** It has hit zero twice, once mid-write. `~/Library` was 50 GB;
  the reclaimable wins found were uv cache (6 GB), Chrome's on-device AI
  model `OptGuideOnDeviceModel` (4 GB), swiftpm (1.8 GB), Spotify (1.2 GB).
  Chrome's `Default/` is real profile data — do not delete it.

---

## 7. What to do next, in order

1. **Decide the licensing question — it now blocks the stated goal.** The
   README says "working on getting it into the app store right now", and
   `docs/licensing_decision_memo.md` (2026-09-11) measured what that would
   ship: **0 of 20,532 corpus photos carry any licence information.** Not
   zero permissive — the field does not exist in the schema. So
   `licensing_review.md`'s "exclude the NC/ND subset" cannot be executed;
   that row was about `negatives_dataset`, which contributes no photos to
   the product. Worse, its closing reassurance that images are "never
   redistributed" **was reversed by deployment**: `/photo` serves image
   bytes off the volume. The memo ranks the real options; the one that
   keeps the product intact is link-out plus thumbnail, and it is a
   lawyer's question, not an engineering one.
2. **Fix what four tiles exposed.** `drop_womens` defaults to `false`, so
   womenswear catalog shots rank into a men's-only corpus; flat-lays rank
   as outfits. Both appeared at position 4 of the first two queries.
3. **Cover the client.** Nothing in the eval stack touches the app, which
   is how three fatal bugs survived a month and a clean build. A single
   smoke test that runs a real query against the live service and asserts
   a photo actually decodes would have caught all three.
4. **Top up skin tone** — 9,515 of 14,684; ~5,000 new photos have none.
   Local, free, ~90 min: `nohup .venv/bin/python -u extract_skin_tone.py &`
5. **Sweep K on the identifier.** 400 was tuned at 2,230 products, the
   gallery is 3,922. Cheap, and probably worth points.
6. **The encoder.** Still the binding constraint, and today's queries look
   exactly like the documented near-ties (top-4 spread 0.31→0.30). See
   `docs/encoder_finetune_plan.md` for the costed plan and what the label
   data actually supports. **Budget-blocked, not idea-blocked.**

### Budget

**$28.13 of $30.** The outfit-serve container bills while warm (5-minute
scaledown). There is **not** enough left for another full GPU pass. Raise
the limit or plan around it before the next big job.

---

## 8. Honest limits that survive everything above

- **Nothing about outfit search is validated against humans.** The VLM eval
  is a model grading a model. `pair_eval.py` is built for human labelling
  and has still never been run to completion.
- **Outfit labels are unvalidated model output** — 43,712 detections from a
  human parser plus zero-shot FashionCLIP over an unlabelled corpus.
  Coverage, not accuracy.
- **`outerwear` sits at 41% group agreement even with good crops**, and it
  is structural: ATR has no Coat class, so a jacket and the tee under it
  come back as one region.
- **Brand-targeted sourcing biases the eval optimistically.** Photos found
  by searching "carhartt detroit jacket outfit" over-represent that jacket
  relative to what a user photographs.
- **The identifier is wrong often.** Its first real call from the iOS agent
  returned a *different* Obey crewneck at 0.940 with the correct product
  third at 0.921. Confidence is not reliability: the wrong answer has
  outscored the right one 0.922 to 0.854.
- **`reject_threshold` is uncalibrated**, so `rejected_open_set` never
  fires; the API reports `reject_threshold_calibrated: false`.
- **Photographs of real people**, with provenance but not permission.
  Measured 2026-09-11 (`docs/licensing_decision_memo.md`): **0 of 20,532
  photos carry licence information of any kind**, 99.97% are traceable to
  an author and post_url, and of the 920 Pinterest records with a
  `source_link`, **38.6% point at ad/affiliate redirectors** — so a source
  link is not a chain of title. These are also **identifiable people**,
  which puts right-of-publicity and GDPR alongside copyright; `skin_tone`
  is stored for 10,640 posts and used as a ranking signal. Private
  development storage is not publication; the App Store would be.

---

## 9. Working norms that paid off

- **Measure before believing, including your own diagnosis.** Several
  claims were overturned by measurement this session, including my own:
  "exact-product matching fails" (it was my crop), "cost is per-request"
  (it is wall time), "`apparel_dataset_full` is redundant" (it was the only
  local copy of five brands), and "colour naming is the problem" (three
  fixes, all failures).
- **Byte-identical results across configs mean a bug — until you check.**
  The binding-weight sweep produced identical output; that one turned out
  to be genuine saturation, verified by inspecting the actual ranked lists.
- **A flair is not evidence.** r/Sneakers `WDYWT` is feet-on-pavement.
  Sample the API, then **audit what the shard actually wrote to disk** —
  that caught r/gorpcore at 8/8 product photos.
- **Verify agent claims.** They were mostly excellent and honest — the iOS
  agent's `xcodebuild` claim reproduced exactly — but the ones that
  mattered were the ones that got checked.
