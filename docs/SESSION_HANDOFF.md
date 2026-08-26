# Session handoff — 2026-08-26

Read this first. Then `docs/eval_log.md` for every number, and
`docs/production_plan.md` (whose top section carries the product
correction that reframed the whole project).

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

**Nothing in it has ever been RUN** — no simulator, no device existed in
the session that wrote it. Compiling is not rendering. Camera, Keychain,
deep links, the Siri snippet and `/outfit_search` end-to-end are all
unverified.

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

1. **Open the iOS app.** `ios/Fashion.xcodeproj`, Run. It works on demo
   data immediately. This is the only way to find out whether any of it
   renders, which is entirely unverified.
2. **Judge outfit search by using it**, now that corpus, catalog and
   ranking finally point the same way. There is no metric for "is this a
   good outfit"; your eye is the measurement.
3. **Top up skin tone** — 9,515 of 14,684; ~5,000 new photos have none.
   Local, free, ~90 min: `nohup .venv/bin/python -u extract_skin_tone.py &`
4. **Sweep K on the identifier.** 400 was tuned at 2,230 products, the
   gallery is 3,922. Cheap, and probably worth points.
5. **The encoder.** If results still feel like near-ties, that is the
   ceiling and fine-tuning SigLIP2 on outfit photos stops being
   speculative. It needs labels — and `outfit_search_eval.py` already
   produces exactly the kind it needs, at 3.2 photos/s.

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
- **Photographs of real people**, with provenance but not permission
  (`docs/licensing_review.md`). Every Pinterest record now carries an
  author — 7 remain of 1,342 — which answers *whose pin*, not *who owns
  the photograph*. Private development storage is not publication; a public
  link would be.

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
