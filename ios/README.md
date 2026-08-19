# Fashion — iOS app

An outfit search engine. You photograph the clothes you own into a **closet**;
you search with any mix of photos and phrases and get back **photographs of
real people** wearing things that match; and you can ask what a piece you own
would look like with something else.

SwiftUI, iOS 17+, Swift 5.9. **Zero third-party dependencies.**

---

## Opening it

```
open ios/Fashion.xcodeproj
```

It builds and runs immediately with no configuration. With no API key it runs
against fixture data and shows a "Demo data" banner on every screen — fake
data is fine, unlabelled fake data is not.

### Giving it the real API key

The key is never in source and never in git. It is resolved at runtime from
three places, in this order — the first one that answers wins:

**1. Scheme environment variable** — best for development. Xcode stores this
in `xcuserdata/`, which is gitignored.

> Product → Scheme → Edit Scheme → Run → Arguments → Environment Variables
> Add `FASHION_API_KEY` with the value from the repo root `.env`.

**2. In-app Settings** — best for running on a device. Open Settings (the gear
in the toolbar), paste the key, it goes into the Keychain. No file editing,
and it survives reinstalls of the project but not of the app.

**3. `Config/Secrets.xcconfig`** — gitignored, substituted into `Info.plist`
at build time.

```sh
cp ios/Config/Secrets.example.xcconfig ios/Config/Secrets.xcconfig
# then paste the key into FASHION_API_KEY
```

> **One gotcha with option 3.** xcconfig treats `//` as a comment *even
> mid-value*. If the key happens to contain two consecutive slashes it gets
> silently truncated. If a valid-looking key gets rejected, that's the first
> thing to check — use option 1 or 2 instead. (This is also why the base URL
> in `Shared.xcconfig` is written `https:/$()/…` — the `$()` splits the
> slashes so the URL survives.)

Settings shows which of the three sources actually supplied the active key, so
a stale Keychain entry quietly beating a fresh xcconfig is visible rather than
a confusing half hour.

`Config/Secrets.xcconfig` is included at the *end* of `Shared.xcconfig`,
because in an xcconfig the later assignment wins — so it overrides the
defaults rather than being overridden by them.

---

## Screens

| Screen | What it does |
|---|---|
| **Closet** | Camera and photo-library capture. Each capture goes to `/identify`, and the review sheet shows **every** candidate with its score for you to pick from — nothing is pre-selected. "None of these" is a first-class outcome. Grid of what you own; tap for detail, where you can re-pick the match, rename it, or start a search built around it. |
| **Search** | Any number of photos and phrases, each shown as a removable part. Filters behind one button: colour (from the index's own vocabulary, or picked from a photo), garment type, US-only, men's-only, and the skin control. Results are a grid of photographs. |
| **Result detail** | The photograph large and unobstructed, the source link directly beneath it, why each query part matched which garment, the detected garments with their caveat, and catalog products of the same *kind* for pieces you don't own. |
| **Fits** | Outfits containing pieces similar to one closet item at a time — you choose which item it's built around. |
| **Siri** | `OutfitSearchIntent` (App Intents, not SiriKit). Speaks a short answer, shows the top 6 as a snippet, each deep-linking to `fashion://outfit/<id>`. |
| **See it on you** | A generated preview, behind `TryOnService`. Labelled as generated in three redundant ways, and there is no way to share or save it. |

---

## The API contract this is coded against

Base `https://hanavm--fashion-serve-fashionservice-api.modal.run`, bearer auth.

**Verified with real calls against the live service** before any code was
written (see `Fashion/API/Models.swift`, and `StubFashionAPI` whose fixtures
are transcribed verbatim from those responses):

- `GET /health` → `{status, device, index: {gallery_products, catalog_products, brands[]}}`
- `POST /identify` `{image_base64, top_k}` → `{results: [{rank, product_code, brand, name, category, model_identity, score}], confidence, rejected_open_set, reject_threshold_calibrated, garment_gate: {score, threshold, looks_like_clothing, calibrated}, predicted_category, spoken, latency_ms}`
- `POST /query` `{image_base64?, text?, top_k?}` → `{query, results: [{product_code, brand, name, match_type, matched_label, score}], spoken, route, latency_ms}`

  One thing worth knowing: on the text-only route `score` is an **integer
  keyword score** (observed 9, 7, 3), not a similarity. `QueryResult.scoreScale`
  carries that distinction so the UI never renders a keyword score as if it
  were a 0–1 similarity.

**Not deployed yet, coded to the agreed contract:**

```
POST /outfit_search
{images: [b64...], texts: [str...], top_k: int,
 colour_name?, colour_rgb?: [r,g,b], skin_image_base64?, skin_tone?: 0..1,
 drop_non_us?: bool, drop_womens?: bool}
-> {results: [{id, image_url, source, post_url, score,
               parts: [{part, score, matched_garment}]}],
    colour_vocab: [...], category_vocab: [...], corpus_posts: int}
```

`OutfitResult` decodes leniently on purpose. The deployed contract promises
`id` and `image_url`; the local engine in `outfit_search.py` emits `rel`,
`path` and `post_id` instead, plus `title`, `author`, `categories`, `colors`
and `whole_frame`. The decoder accepts either and derives what's missing, so
whichever ships, nothing breaks. A 404 or 501 from this path is translated
into a distinct `.notDeployed` error, and the UI says "outfit search isn't
live yet" rather than "something went wrong".

The category filter is sent as an **extra phrase**, not as a new field — the
agreed contract has no category parameter, and inventing one client-side would
be coding against an API that doesn't exist.

### Stub / live split

`FashionAPI` is a protocol with two implementations:

- `LiveFashionAPI` — 90s request timeout for the ~20s cold start, bearer auth,
  FastAPI `detail` extraction, `.notDeployed` mapping.
- `StubFashionAPI` — fixtures, no network, no credentials. Every screen has a
  `#Preview` using it.

The stub's fixtures are deliberately **not** a best case. `identify` returns
the *wrong* product at rank 1, verbatim from a real call: a photo of Obey's
"EST. WORKS BOLD II CREWNECK" came back as a different Obey crewneck (0.940),
then Uniqlo trousers (0.923), then the correct product third (0.921). Any
design showing one confident answer is wrong for this backend, and previewing
against this fixture makes that hard to forget. `StubFashionAPI.Behaviour`
also covers empty results, cold start, arbitrary failures, and the
outfit-search-not-deployed case, so those states get designed rather than
discovered.

`AppEnvironment` picks the implementation: live if a key resolves, stub
otherwise, with `isDemoData` driving the banner.

---

## Local storage: Codable + FileManager, not SwiftData

Three reasons, all specific to this app:

1. The closet is a single-user list of tens of items with no relationships and
   no queries beyond "all of them, newest first". SwiftData's value is
   querying, relationships and cross-context change tracking; none of that is
   being bought here.
2. The heavy part of an item is its photograph, which doesn't belong in a
   database row either way. Images are individual JPEGs; the record stores a
   filename. Once the images are files, one small JSON manifest beside them is
   simpler than running a store alongside them.
3. The identification schema is still moving — the outfit API isn't even
   deployed. A JSON document with optional fields absorbs that; a SwiftData
   model asks for a migration each time.

The trade: a full manifest rewrite on every mutation, and no partial loads. At
closet scale that's free. Revisit if the closet reaches thousands of items or
gains sync.

A closet item stores **the whole candidate list**, not a resolved product,
plus which candidate the user endorsed and whether they endorsed one at all.
An unconfirmed guess is labelled as an unconfirmed guess forever rather than
hardening into a fact.

---

## Design

The rules, and where they were in tension:

- **One accent**, defined once in `Assets.xcassets/AccentColor.colorset` with
  light, dark, and increased-contrast variants. Used in exactly three places.
- **No hardcoded colours in code**, with one deliberate exception:
  `DominantColour.swatch` renders the literal RGB the eyedropper extracted.
  Its entire job is to show you the colour that's about to be sent, so a
  semantic colour would defeat it. It always sits beside text, never conveys
  state, and never becomes a second accent.
- **Glass is for chrome only.** There is no material anywhere in the app —
  navigation and sheets get the system's own treatment, and cards are plain.
  Grep for `Material` returns nothing.
- **Hierarchy from type and space.** There is deliberately no card style with
  a border and a shadow. Photographs tile directly.
- **Light is the default**; no dark hero surface exists. Dark mode works
  because every colour is semantic, but nothing is designed dark.
- **Motion is near-invisible.** Two springs, both ≤300ms, both routed through
  `Theme.Motion` so `accessibilityReduceMotion` collapses them to *no*
  animation rather than a faster one.
- **Dynamic Type reflows**: `fixedSize(horizontal:false, vertical:true)` on
  every wrapping label, and no caveat is ever truncated.
- **44pt touch targets** via `minimumHitTarget()`, which claims the space
  without forcing the visual element to be 44pt — the usual reason this rule
  gets broken.

**Tensions, and how they were called:**

- *Scores vs. clarity.* A progress bar or a green/amber/red pill would read
  better, but this system's ranking isn't calibrated — the wrong answer has
  outscored the right one 0.922 to 0.854, and the spread from rank 1 to rank
  50 in outfit search is about 0.015. So `ScoreLabel` is a bare monospaced
  number with its scale named, and no visual encoding promises more.
- *Selection colour.* Selected states use the accent **plus** a weight change
  or a checkmark glyph, never colour alone.
- *Filters: bar vs. sheet.* A permanent filter bar would put six controls
  above the photographs on every scroll. Filters went into a sheet, and the
  button that opens it names the active ones so state is never hidden.
- *Feed: whole closet vs. one item.* Blending the closet into one feed
  produces results nothing in it can explain, and costs one cold-start request
  per item. It's built around one item at a time, which you choose.
- *Empty states.* "No results" alone can't be told from a broken app, so every
  empty state states the size of the thing that was searched.

---

## Honesty, in the UI rather than the docs

Each of the backend's measured limits has a specific home on screen:

- **Catalog is 20 US brands, men's only.** Named in the closet's empty state,
  in the capture review sheet above the candidates, and in Settings.
  Identification never shows one answer — always the ranked list with scores.
  The uncalibrated open-set threshold is stated too, because the absence of a
  rejection doesn't mean the match is real.
- **Outfit labels are unvalidated model output.** The caveat is baked into the
  `DetectedGarments` component and the garment-type filter's footer, so the
  labels cannot be rendered without it. The `outerwear` merging problem is
  named specifically.
- **The skin control is relative.** No tone categories, no numeric readout, no
  swatches. A reference photo is offered first because it asks for an example
  rather than a self-classification. It's off by default — defaulting it to
  0.5 would silently apply a filter nobody asked for. Its VoiceOver value is
  "N percent along the range", never a tone.
- **Real people, provenance without permission.** `SourceLink` is on every
  outfit, directly under the photograph rather than at the bottom. There is
  **no sharing or export anywhere in the app**, by design.
- **"See it on you" output is generated.** Labelled in body type immediately
  under the image; the disclosure string lives on `TryOnPreview` itself so no
  view can render the image without holding the words; and there's no way to
  save or share it.

---

## What was verified, and what was not

**Verified:**

- `xcodebuild` **Debug and Release both succeed** for the iOS Simulator SDK,
  with **zero errors and zero warnings** from a clean derived-data directory.
- SwiftUI previews **compile** — `#Preview` bodies are compiled into the Debug
  build, so every preview type-checks against the stub. `__preview.dylib` is
  present in the built bundle.
- The App Intents metadata compiled: `Metadata.appintents` is in the bundle,
  so the intent and shortcut phrases registered.
- The asset catalog compiled to `Assets.car` (accent colour with its four
  appearance variants).
- The full credential path works end to end. Built with a dummy
  `Secrets.xcconfig` and confirmed in the built `Info.plist`:
  `FashionAPIKey => "DUMMY-TEST-VALUE-123"`, `FashionAPIBaseURL =>
  "https://hanavm--fashion-serve-fashionservice-api.modal.run"`, and the
  `$()` slash-escaping survives. The dummy file was then deleted.
- `Info.plist` substitution generally: bundle id, `MinimumOSVersion 17.0`,
  camera and photo-library usage strings, and the `fashion://` URL scheme all
  land correctly in the built bundle.
- The `/health`, `/identify` and `/query` response shapes, by calling the live
  API and coding the models against what actually came back.

**Not verified — I have no simulator, no device and no Xcode GUI:**

- **Nothing has ever been run.** Not once. No screen has been rendered, by me
  or by anyone.
- Whether previews *render* — only that they compile. A preview that compiles
  can still crash on the main actor at render time.
- All visual design: spacing, type scale, contrast, how the grids reflow at
  accessibility text sizes, whether the light layout actually reads well.
- Every runtime path: camera capture, photo-library loading, the Keychain
  round-trip, deep-link resolution, the Siri snippet, `AsyncImage` loading of
  corpus photos.
- `/outfit_search` end to end, because it isn't deployed. Everything in Search
  and Fits is coded to a contract, not to observed responses.
- The Azure try-on request and response shapes. **No endpoint existed when
  this was written.** `AzureTryOnService` follows Azure OpenAI's image API
  (`api-key` header, `data[0].b64_json`) and has never been exercised. That's
  exactly why it's behind a protocol.

## Things a human needs to do

1. Open the project and hit Run. It works with demo data immediately.
2. Add the API key by one of the three routes above to get live identification.
3. Set a development team for a device build (simulator needs none).
4. Drop in a real app icon — `AppIcon.appiconset` is declared but has no image.
5. Point `AzureTryOnEndpoint` at a real endpoint once one exists, and check
   `AzureTryOnService`'s request/response shapes against it.
6. Swap `LiveFashionAPI.outfitSearch` to the real response shape if
   `/outfit_search` ships differently from the agreed contract.
