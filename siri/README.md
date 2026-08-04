# Phase 7 — the Siri / iOS client layer

> "Hey Siri, what does this jacket look like with cargo pants?"
> — while looking at something on screen.

This directory is the **client half** of that sentence. The server half
(`modal_app_serve.py`, Phase 6) is being built separately and is not
touched from here; everything below is written against its contract, not
against its code.

Three things live here:

| file | what it is |
|---|---|
| `README.md` (this file) | the iOS Shortcut recipe — steps a human taps in the Shortcuts app, plus the exact JSON |
| `stub_server.py` | a stdlib fake server that speaks the contract, including every failure mode. No models. |
| `../siri_client.py` | a CLI that performs the identical flow, and is what actually gets tested |

A Shortcut is not code: it cannot be generated as a valid binary
`.shortcut` file from a shell, it cannot be diffed, and it cannot be run
in CI. So the flow is defined twice on purpose — once as taps (below) and
once as `siri_client.py`. **When the two disagree, `siri_client.py` is
the one that was executed.**

---

## 1. The contract

`siri_client.py` and the recipe below both assume exactly this. It is the
single source of truth for the client side; if the serving app lands with
different names, change this file and `siri_client.py`, not the Shortcut
alone.

### `GET /health`

```
Authorization: Bearer <token>      (optional on health)
```
```json
{"status": "ok", "models_loaded": true}
```

### `POST /identify`

```
Content-Type: application/json
Authorization: Bearer <token>
```
```json
{"image_base64": "<base64 JPEG, no data: prefix, no newlines>", "top_k": 5}
```
```json
{
  "spoken": "That looks like a Dickies 874 work pant.",
  "confidence": 0.78,
  "rejected": false,
  "reject_threshold": 0.35,
  "results": [
    {"product_code": "874BK", "brand": "dickies", "name": "Original 874 Work Pant",
     "score": 0.81, "product_url": "https://...", "image_path": "..."}
  ],
  "detection": {"garment_found": true, "category": "bottoms/pants", "bbox": [x1,y1,x2,y2]}
}
```

### `POST /compose`

```json
{"image_base64": "...", "text": "with cargo pants", "top_k": 5}
```
```json
{
  "spoken": "That looks like a Dickies 874 work pant. With cargo pants, try the Eagle Bend Cargo.",
  "confidence": 0.78,
  "rejected": false,
  "primary": { ...one product... },
  "companions": [ ...products... ]
}
```

### Fields that carry the honesty, not just the answer

* **`rejected`** — the open-set rejection that already exists inside
  `hierarchical_retrieval_pipeline.py` as `rejected_open_set`. It must
  survive the trip to the API edge (roadmap Phase 6.4). `siri_client.py`
  accepts either name, and treats the field being **absent** as unsafe —
  a missing rejection signal is not evidence of a match.
* **`confidence`** — the client hedges below `--min-confidence` (0.35 by
  default) regardless of what `spoken` says.
* **`spoken`** — one sentence for TTS, separate from the visual list. It
  is used **verbatim only when the answer is neither rejected nor
  low-confidence.** Otherwise the client writes its own hedge and throws
  the server's sentence away, so that a server which forgets to hedge
  cannot make the assistant assert something wrong out loud.

---

## 2. The Shortcut recipe (~10 minutes)

Two shortcuts, because they are invoked differently. Build **A** first;
**B** is A with two extra actions.

### Prerequisites

* The serving app's URL, e.g. `https://<workspace>--fashion-serve.modal.run`
* The shared token.
* Do **not** paste the token into the shortcut body where a screen
  recording would catch it — put it in a Text action at the top and
  reference it, or store it in a note you paste at setup time.

### Shortcut A — "What is this?"

New Shortcut, name it **What is this** (that name is what you say to Siri).

1. **Receive** — tap the shortcut's ⓘ settings:
   * *Show in Share Sheet* → **on**, accepted types: **Images, Screenshots**
   * *Use with Siri* → on
   This makes `Shortcut Input` the shared image when invoked from the
   share sheet.
2. **If** — `Shortcut Input` **has any value**
   * **Otherwise** branch → **Get Latest Screenshots** (Count: 1), so
     "Hey Siri, what is this" with nothing shared falls back to whatever
     you were just looking at.
   * Set a variable **Photo** in both branches.
3. **Crop Image** *(strongly recommended, see §4)* — set to **Ask Each
   Time** if you want to draw the box, or use *Get Images from Input*
   when the share came from a photo rather than a screen grab. This one
   action is worth more accuracy than anything else in the list.
4. **Convert Image** → **JPEG**, Quality 90. Then **Resize Image** →
   longest edge **1280**. (A 3x retina screenshot is ~12 MB; base64 makes
   it ~16 MB and cellular uploads will time out.)
5. **Base64 Encode** — input: the resized image. **Turn *Line Breaks*
   OFF** in that action's options. This is the single most common reason
   the request 400s: Shortcuts wraps base64 at 76 chars by default and
   most JSON parsers on the far end reject the embedded newlines.
6. **Text** — name it so you can find it; content exactly:
   ```
   {"image_base64":"<Base64 Encoded variable>","top_k":5}
   ```
   Insert the Base64 variable inside the quotes. Do not add spaces.
7. **Get Contents of URL**
   * URL: `https://<your-app>/identify`
   * Method: **POST**
   * Headers:
     * `Content-Type` → `application/json`
     * `Authorization` → `Bearer <token>`
   * Request Body: **File** → the **Text** action from step 6.
     (Choosing "JSON" here makes Shortcuts build the object itself and
     it will mangle a long base64 string; send it as raw text.)
8. **Get Dictionary Value** — Get **Value** for key `rejected` → set
   variable **Rejected**.
9. **Get Dictionary Value** — Get **Value** for key `confidence` → set
   variable **Confidence**.
10. **Get Dictionary Value** — Get **Value** for key `spoken` → set
    variable **Spoken**.
11. **If** — `Spoken` **has any value**
    * **and** `Rejected` **is not** `true` *(add via a nested If — the
      Shortcuts If action takes one condition)*
    * **and** `Confidence` **is greater than** `0.35`
    * → **Speak Text**: `Spoken`
    * **Otherwise** → **Speak Text**:
      `I'm not sure — I couldn't match that to anything I know.`
12. **Get Dictionary Value** — Get **Value** for key `results` →
    **Show Result** (or **Quick Look**) so the ranked alternatives are on
    screen even when the spoken line hedged.

### Shortcut B — "What does this go with?"

Duplicate A, rename to **What does this go with**, and change three things:

* After step 3, add **Ask for Input** (Text): *"With what?"* → variable
  **Phrase**. When run by voice, Siri asks aloud and you answer aloud.
  (You can also add `Shortcut Input` text parsing, but the explicit ask is
  far more reliable than trying to get Siri to pass a phrase through.)
* Step 6's Text becomes:
  ```
  {"image_base64":"<Base64 Encoded>","text":"<Phrase>","top_k":5}
  ```
* Step 7's URL becomes `.../compose`, and step 12 reads the key
  `companions` instead of `results`.

### Invoking it

* "Hey Siri, **what does this go with**" — Siri asks "With what?", you say
  "cargo pants", it uses the latest screenshot.
* Or: screenshot → share sheet → **What does this go with**.

> **Why the latest-screenshot fallback rather than live screen capture:**
> iOS gives shortcuts no API to read the current screen. Taking a
> screenshot (side button + volume up) and letting the shortcut grab the
> most recent one is the only path that exists, and it also means the user
> can crop before running.

---

## 3. Testing it without an iPhone

```bash
export FASHION_API_TOKEN=stub-secret
python3 siri/stub_server.py --port 8000 &            # or --scenario rejected|low|empty|error|slow
python3 siri_client.py --url http://127.0.0.1:8000 --health
python3 siri_client.py --url http://127.0.0.1:8000 --image shot.jpg
python3 siri_client.py --url http://127.0.0.1:8000 --image shot.jpg --text "with cargo pants"
```

`--show-request` prints the exact body the Shortcut's step 6 must
produce, with the base64 redacted. Build step 6 by matching that output.

**What has actually been run** (2026-08-04, no server deployed yet):
every path below was executed against `stub_server.py`, not against a
real model. No retrieval-quality claim is made or implied by any of it.

| scenario | spoken | exit |
|---|---|---|
| healthy, confident | server's `spoken`, verbatim | 0 |
| `/compose` | primary + companion sentence | 0 |
| `rejected: true` (server still sent a confident sentence) | client's hedge; server's sentence discarded | 2 |
| `confidence 0.22` | "I'm not sure. It might be … but the match is weak." | 2 |
| empty results | "I'm not sure — I couldn't match that to anything in the catalog." | 2 |
| `rejected` field missing entirely | "I'm not sure — the service didn't tell me whether that was a real match." | 2 |
| HTTP 500 | "I can't reach the fashion service right now." | 3 |
| timeout | same | 3 |
| connection refused | same | 3 |
| wrong token (401) | same, with `check $FASHION_API_TOKEN` on stderr | 3 |

---

## 4. The screenshot problem, stated honestly

**Every accuracy number this project has ever produced is
catalog-photo-to-catalog-photo**: one clean studio image, one garment,
white background. The 59.92% R@1 is that number. It is not a screenshot
number, and nothing here should be read as one.

What actually arrives from "seen on screen" is a 1170×2532 PNG of a
retail page: a black nav bar, the hero product photo, a price, a size
selector, a "You might also like" strip with three other products, and a
tab bar. The garment of interest is perhaps 30% of the pixels, and there
are competing garments in frame on purpose.

### What was measured (2026-08-04, one image, local CPU)

A synthetic product-page screenshot was built — real Carhartt hero photo
(a model in a brown hoodie and light jeans, the page is selling the
**pants**), page chrome, price text, and a second product thumbnail —
and run through `segment_outfit.py` unmodified:

```
1 item(s) detected
  #1  tops/sweatshirt  (label='a crewneck', confidence=0.544,
                        area_fraction=0.116, bbox=(55,81,417,262))
```

Reading that result honestly:

* **Good:** it did not crop the nav bar, the price text, or the tab bar.
  The distractor labels (`background`, `a wall`, …) did their job — no
  page chrome was returned as a garment.
* **Bad:** the single returned crop is the **entire person**, both
  garments together, not a per-garment crop. Sending that to `/identify`
  asks the identity encoder to match a two-garment crop against
  single-garment catalog photos.
* **Bad:** it was labelled `sweatshirt` when the page's actual subject is
  the pants. A caller cannot tell from the response that it picked the
  wrong garment.
* **Probable cause, not yet confirmed:** `segment_outfit.py` thumbnails
  to `MAX_IMAGE_DIM=1024` on the **longest** edge. A 1170×2532 screenshot
  therefore becomes 473×1024, and the hero photo inside it shrinks to
  roughly 360×180 — at which point SAM2's automatic mask generator stops
  proposing separate masks for the hoodie and the jeans and only proposes
  the person. A portrait screenshot is the worst possible aspect ratio for
  a longest-edge resize. This is a hypothesis with one image behind it,
  which is exactly the amount of evidence it has.

### So what is wired in, and what isn't

* `siri_client.py --segment` runs `segment_outfit.py` locally and sends
  the best crop. It works, it is off by default, and on the evidence above
  **it is a lever to measure, not a fix to assume.**
* `--strict-segment` makes "no garment found" a spoken "I don't see any
  clothing in that image" instead of silently falling back to posting the
  whole screen.
* **The Shortcut does not segment.** iOS cannot run SAM2, and the Shortcut
  has no way to. Its equivalent is step 3's **Crop Image**, done by the
  human. A human dragging a box around the jacket is, today, better than
  the detector — and it costs one tap.

### The honest recommendation

1. Ship with **manual crop** in the Shortcut. It is reliable now.
2. Server-side detection belongs in `/identify` (the contract already has
   a `detection` block for reporting it), not in the client — the client
   cannot carry the models.
3. Before trusting automatic detection on screenshots, fix the resize
   (crop the dominant image region first, or raise `MAX_IMAGE_DIM` for
   tall inputs) and **measure it on real screenshots**, which requires a
   labeled screenshot set that does not exist yet. Phase 8's
   `index_outfits.py` run produces the closest thing — multi-item
   detections at corpus scale — but on outfit photos, not on UI.

---

## 5. Graceful degradation, in one place

The rule, for both clients: **a voice assistant that names the wrong
jacket confidently is worse than one that says "I'm not sure"**, because
the user hears one sentence and cannot see the list to catch the error.

| failure | what the assistant says |
|---|---|
| no image / nothing shared and no screenshot | "I couldn't find an image to look at." |
| `--strict-segment` and no garment detected | "I don't see any clothing in that image." |
| `rejected: true` | "I'm not sure. The closest thing I found was X, but it isn't a confident match." |
| `confidence` below threshold | "I'm not sure. It might be X, but the match is weak." |
| empty results | "I'm not sure — I couldn't match that to anything in the catalog." |
| `rejected` / `confidence` missing from the response | "I'm not sure — the service didn't tell me whether that was a real match." |
| server down, 5xx, timeout, bad token | "I can't reach the fashion service right now." |

In every hedged case the **ranked list is still shown**. The hedge is
about what is spoken aloud, not about withholding results — the user can
look and decide, they just aren't told something false.
