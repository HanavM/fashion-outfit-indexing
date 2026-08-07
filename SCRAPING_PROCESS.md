# Apparel catalog scraping + captioning pipeline — process summary

Context for a fresh Claude session: this documents how a 4-brand shoe image
dataset (Adidas, Nike, New Balance, Skechers — 563 variants, 4334 images)
was scraped, enriched, captioned, and filtered for CLIP finetuning, **and**
how the same `apparel_dataset/` was later extended with 200 Nike men's
clothing records (Tops and T-Shirts, Shorts, Hoodies and Pullovers, Pants
and Tights — 50 each) using an upgraded structured-caption schema and a
SAM2+FashionCLIP garment-cropping step, and then with 176 PacSun men's
clothing records (Pants, Graphic Tees, Sweaters, Shorts — 50 each targeted,
Sweaters capped at 26 by catalog size) — the first extension to an entirely
new *site*, not just a new brand on an already-handled site pattern. Most
recently, 177 Gap men's clothing records (T-Shirts, Shorts, Pants, Sweaters
— 50 each targeted, Sweaters capped at 27 by catalog size) were added via
`gap_scraper.py` — the easiest site of the pipeline (no bot protection
anywhere, plain `requests` for the catalog), notable mainly for a
client-side-rendering trap in the PDP accordions (see "Gap" section below).
Then 200 Champion men's clothing records (Hoodies and Sweatshirts,
T-Shirts and Tops, Shorts, Pants and Joggers — 50 each, all 200 reached
with no category shortfall) were added via `champion_scraper.py` — a
Shopify storefront site, the easiest data source of any brand so far
(one JSON API call returns the entire product record, no second PDP
fetch needed for anything). Most recently, 178 Levi's records (Jeans,
Jean Jackets, Shirts, Accessories — 50 each targeted, Accessories capped
at 28 by real catalog-listing coverage) were added via
`levis_scraper.py` — the hardest bot-protection tier encountered so far
(see "Levi's" section below), and the first brand whose "Accessories"
category is genuinely heterogeneous (belts, hats, backpacks, wallets,
even underwear) rather than one visually consistent garment type.
Then 200 Carhartt WIP records (Jackets and Coats, Pants, Shirts,
T-Shirts and Polos — 50 each, all 200 reached with no category
shortfall) were added via `carhartt_scraper.py` — the first
commercetools-backed site in this pipeline, and the first brand whose
product photos are ALREADY clean flat-lay shots with no model/lifestyle
background, so the SAM2+FashionCLIP garment-cropping step (needed for
Nike/PacSun/Gap/Champion/Levi's) was correctly skipped entirely for this
brand, same reasoning already established for the original shoe photos.
Then 175 Stüssy records (Tees, Pants, Headwear — 50 each, Hoodies capped
at 25 by real catalog size) were added via `stussy_scraper.py` — the
second Shopify-storefront site in this pipeline (after Champion), same
easiest bot-protection tier, and the second brand (after Carhartt) whose
product photos are already clean flat-lay shots, so garment cropping was
again correctly skipped. Most recently, 200 Vans records (Shoes, Hoodies
and Jackets, Shirts — 100/50/50, no shortfall) were added via
`vans_scraper.py`, specifically to grow footwear coverage past the
original 4 shoe brands — a VF Corp/Nuxt.js site with Akamai bot
protection (softer than Levi's, no visible interactive challenge) and
the first brand this session whose product photos are genuinely mixed
flat-lay/on-model rather than uniformly one or the other, so garment
cropping was run for real on the clothing categories (not skipped, and
not applied to shoes). Most recently, 200 Dickies records (Pants,
Shirts, Shorts, Coats and Jackets — 50 each, no shortfall) were added
via `dickies_scraper.py` — the third Shopify-storefront site in this
pipeline (after Champion and Stüssy), same easiest bot-protection tier,
with real on-model lifestyle photos (verified by direct image
inspection, not assumed), so garment cropping was run for real, same as
Vans. Written so the same pattern can be adapted to scrape other
product categories/sites.

Working directory: `/Users/hanavmodasiya/fashion-tests`
Environments: `.venv` (Python 3.14, pip: playwright, patchright, openai,
python-dotenv) for scraping/captioning; conda env `mint` (torch,
open_clip_torch, pillow) for CLIP-based embedding/classification work.

## Goal

Build an image+metadata dataset per shoe colorway variant, suitable for
finetuning a CLIP-style model (Marqo FashionSigLIP) to match a photo of a
worn shoe to the exact product/colorway — for an outfit-identification app.

## Pipeline stages (each is its own script, each is idempotent/checkpointed)

1. **Catalog scrape** (`{brand}_scraper.py`) — discover all color variants
   of every model, download full-res images **directly into
   `apparel_dataset/{brand}/{slug}/{product_code}/image_N.jpg`**, and write
   that brand's records straight into `apparel_dataset/metadata.json`
   (append/update in place, keyed by `product_code`, same checkpoint-every-
   N-records pattern as everything else in this pipeline). There is no
   intermediate `{brand}_catalog/` staging folder and no separate merge
   step — a shoe-brand scraper run before this change wrote to
   `{brand}_catalog/` + `{brand}_products.json` and needed a one-off
   `build_dataset.py` copy pass to bundle into a portable dataset folder;
   new scrapers should skip that indirection and target the shared dataset
   folder from the start.
2. **Detail enrichment** (`enrich_{brand}_details.py`) — revisit each
   variant's PDP, scrape the "product details" copy (description, features,
   materials) into a `details` field on each record **in
   `apparel_dataset/metadata.json`** (filter records by `brand` field,
   same as before but no per-brand JSON file to open).
3. **Captioning** (`caption_shoes.py`) — Azure OpenAI (gpt-4o-mini) turns
   each record's name/color/details into one dense CLIP-training caption,
   written into `apparel_dataset/metadata.json` directly.
4. **View classification** (`classify_views.py`) — zero-shot FashionSigLIP
   classification of every image into a camera-angle category (side/front/
   top/back/hero/on-foot vs. sole/insole/material-closeup/packaging), so
   angles that would never appear in a real "outfit photo" can be excluded
   without touching image files (non-destructive — writes `image_views.json`
   as a separate lookup, doesn't delete anything).
5. **Structured captioning** (`caption_apparel.py`) — added when the dataset
   grew beyond shoes. Same Azure OpenAI text-only pattern as
   `caption_shoes.py`, but fills the JSON-schema prompt in `newLLMprompt.py`
   (taxonomy path, per-attribute color/material/fit/etc., 5-10 `positive_texts`
   at varying specificity) instead of one free-text line, and writes it to a
   **new** `structured_caption` field — `caption` is left untouched
   (non-destructive; it already cost real money to generate). Run once with
   no filter to backfill every existing record, then again per new scrape
   batch. See "Structured captioning" section below.
6. **Garment cropping** (`segment_apparel.py`) — added for the Nike clothing
   expansion only (shoe photos don't have the same "model's face/body
   dominates the frame" problem apparel photos do). SAM2 automatic mask
   generation + FashionCLIP zero-shot label matching crops each apparel photo
   down to just the garment, writing a `cropped_images` field (never touches
   or deletes the original `images`). See "Garment cropping" section below —
   several non-obvious tuning passes were needed to get this right.

## Per-brand scraping notes (each site has a different bot-protection tier
and a different embedded-data source — always check for a JSON blob before
writing CSS-selector scraping code)

### Nike — easiest
- Plain `playwright` (not patchright) works, headless, with
  `--disable-blink-features=AutomationControlled` + webdriver-undefined
  init script. No real bot protection encountered.
- All product data (title, description, **productDetails**,
  **featuresAndBenefits**, price, image gallery) is in a `__NEXT_DATA__`
  JSON script tag — no clicking needed, not even for "product details" (the
  click just reveals a UI accordion; the data is server-rendered already).
  Parse via `<script id="__NEXT_DATA__">...</script>` →
  `data['props']['pageProps']['selectedProduct']['productInfo']`.
- Images: opaque CDN UUIDs, **no per-image view/angle metadata at all** —
  every image shares identical generic alt text.

### Adidas — medium
- `playwright`, `headless=False`, `channel="chrome"`, same anti-automation
  flags. Occasionally an "Account Portal AUTHN" login modal or cookie
  banner intercepts clicks — always `remove_overlays()` (delete
  `dialog[open]`, `[data-mf-id^="ap/"]`, cookie-consent nodes via
  `page.evaluate`) before clicking, and retry once after removal.
- Description/bullets are NOT server-rendered — must scroll down and click
  a `button:has-text("Details")` accordion, then read
  `[data-testid*="accordion"]` elements.
- Price element `[data-testid='main-price']` returns label-prefixed text
  like `"Price\n$90"` — extract with `re.search(r"\$[\d,]+(?:\.\d+)?", ...)`,
  don't try to strip known prefixes (they vary).
- Image gallery: CDN filenames originally carried view codes (`HM1`-`HM9`,
  `HB1`-`HB9` — "Hero Model"/"Hero Back" presumably) used only for
  dedup-by-filename in the scraper, then discarded — **not saved to JSON,
  and lost once files are renamed to `image_N.jpg`**. If you want that
  signal, save it into the JSON at scrape time; it can't be recovered
  after the fact.

### New Balance — hardest (Akamai bot protection)
- Requires `patchright` (a Playwright fork that evades headless
  detection) — plain `playwright` gets blocked outright on category/search
  pages. `headless=False`, `channel="chrome"`.
- Product data comes from a `Product-Variation` SFCC JSON endpoint (visit
  via `page.goto(url)` then `json.loads(page.evaluate("document.body.innerText"))`
  since it returns raw JSON, not HTML): master PDP page also has an
  ld+json `ProductGroup` block listing all colorways + style IDs.
- Master PDP URL requires a `master_id` embedded in the URL path
  (`/pd/{model-slug}/{master_id}.html?dwvar_{master_id}_style={style_id}`)
  that is **not derivable from the style_id (SKU) alone** — this bit us
  hard (see Lessons Learned below). To resolve a master URL from just a
  style code, query NB's site search:
  `GET /on/demandware.store/Sites-NBUS-Site/en_US/Search-UpdateGrid?q={style_id}&start=0&sz=6`
  and regex out `href="(/pd/[^"]+\.html)`.
- Description + Product Details are two separate `<accordion-component>`
  elements (`[data-title="Description"]`, `[data-title="Product Details"]`)
  that are **mutually exclusive** — opening one collapses the other, and
  Playwright's `inner_text()` only returns currently-visible text. **Read
  Description BEFORE clicking the Product Details button**, or you'll
  silently get an empty description on every record.
- Description text has a boilerplate prefix `"Looking for other options?
  Shop the {model}\n\n"` — strip with regex before storing.
- **Akamai will intermittently serve a bot-block page** ("Oops! Something
  went wrong", error code like `0.xxxxxxx.timestamp.xxxxxxxx") mid-run,
  even mid-session — not just a hard one-time wall. Symptoms: some requests
  succeed, then a long unbroken streak fails, sometimes a few succeed again
  later (non-deterministic, looks IP-reputation-based). It can also start
  blocking the *human's own browser* on the same network/IP once triggered
  hard enough. Mitigation implemented: `is_blocked(page)` checks
  `page.title()` / first 200 chars of body text for "Oops! Something went
  wrong", and `goto_with_retry()` retries with escalating backoff
  (5s/10s/15s/20s) up to 4 attempts before giving up and leaving that
  record's `details` field absent (NOT recorded as empty — see lessons).
  If a whole run hits a sustained block, the practical fix is just waiting
  (tens of minutes) before re-running — the script is safe to re-run any
  number of times since it only processes records missing `details`.
- Per-image alt/title text is generic and identical across all images of a
  variant (`"New Balance 9060, U9060GRY"`) — **no per-image view metadata**.
  Only an undocumented numeric suffix in the CDN URL differs
  (`u9060gry_nb_02_i`, `_16_i`, `_05_i`...) with no legend to decode it.

### Skechers — easiest of all, plus the only brand with real view labels
- No real bot protection — plain `requests`/HTTP works, no browser needed
  at all for the catalog scrape.
- Image CDN URLs are **self-describing**:
  `https://images.skechers.com/image/{sku}_HERO_LG`,
  `..._INSOLE`, `..._OUTSOLE`, `..._PROFILE_01`, `..._PROFILE_05` — these
  are genuine, reliable, human-readable view names. This is the only brand
  where you can filter by camera angle from metadata alone without any ML.
- "Key Features" and "Design Details" sections are present directly in the
  static page HTML (no click/JS needed) — just parse with regex/BS4.

### Nike clothing category pages — one product per grouping, not per colorway

Extending `nike_scraper.py`'s pattern to Nike's men's-clothing subcategories
(`nike_clothing_scraper.py`) surfaced a few differences from the shoe scrape:

- **Category left-nav sections resolve to their own `/w/...` category paths**,
  found inside `__NEXT_DATA__`'s `facetNav.categories` list on the parent
  category page (fetch `https://www.nike.com/w/mens-clothing-6ymx6znik1`,
  parse `<script id="__NEXT_DATA__">`, look for `displayText` matching the
  left-nav label — e.g. `"Hoodies and Pullovers"` → `navigation.canonicalUrl`
  gives `/w/mens-hoodies-and-pullovers-6riveznik1`). These paths work as
  the `path` param on the same `product_wall` search API the base scraper
  already uses — no new endpoint needed, just a different `path`.
- **Use `queryType=PRODUCTS`, not `FACETED`**, when paginating a category path
  with `anchor`/`count`. `FACETED` silently under-reports the `pages.next`
  flag on some categories (e.g. "Pants and Tights" reported no next page
  after just 18 unique groupings on page 1, when the category has hundreds
  of products) — `PRODUCTS` (the query type `nike_scraper.py`'s shoe search
  already validated) paginates correctly on every category tested.
- **"Top 50 products" means one record per product, not per colorway.** The
  shoe scraper expands every colorway of a grouping into its own record;
  for a "top N products" clothing scrape, take only the grouping's
  `selectedProduct` (the one variant the PDP actually resolves to), not
  every product in `productGroups[groupIndex]`.
- **The same product can be cross-listed under two different left-nav
  categories** (e.g. a fleece pullover appeared under both "Tops and
  T-Shirts" and "Hoodies and Pullovers"). If you dedupe globally by
  `product_code` across categories (recommended — avoids storing the same
  images/data twice), a category can come up short of its target count
  purely from these collisions. Don't pre-fetch a fixed batch of N groupings
  per category and hope none collide — page the search API as a generator
  and keep pulling additional groupings until the category actually reaches
  N *new* records, skipping (not counting against the target) any grouping
  whose resolved `product_code` turns out to already be in the dataset.
- **Paginating deep into a "mens" category path can drift into
  women's/kids items.** Topping up "Pants and Tights" past the first page
  (to reach 50 after cross-category dedupe losses) pulled in
  `nikeskims-studio-stretch-womens-*` leggings, `*-big-kids-boys-*` tights,
  etc. — this is what Nike's own API returns for that exact category path
  at depth, not a scraper bug, but worth a manual pass if strict gender
  scoping matters more than hitting the exact target count.

## Captioning (`caption_shoes.py`) — Azure AI Foundry / Azure OpenAI

### Setup gotcha (cost real debugging time)
Azure AI Foundry's UI shows the **project** endpoint by default, e.g.
`https://{resource}.services.ai.azure.com/api/projects/{project-name}` —
but the OpenAI Python SDK's `AzureOpenAI` client needs the bare **resource**
root only: `https://{resource}.services.ai.azure.com/` (strip everything
after the domain). Using the project-style URL produces
`BadRequest: API version not supported` regardless of which api-version
string you try — the fix is the URL, not the version. Working
`api_version` for this setup: `2024-10-21`.

Config lives in `.env` (gitignored) copied from `.env.example`
(placeholders only, safe to commit):
```
AZURE_OPENAI_ENDPOINT=https://{resource}.services.ai.azure.com/
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-10-21
```
**Never put a real key in the `.example` file** — only `.env` is
gitignored, `.env.example` is meant to be committed with placeholders.

### Prompt design (iterated several times based on user review)
Caption structure: `{Brand} {Model family} in {exact colorway}, with
{upper material}, {2-4 distinctive visual design details}.`

Key rules that mattered in practice:
- Text-only (no vision calls) — reads scraped `name`/`color_name`/
  `details.description`/`details.features`/etc., not the images. Kept cost
  to **~$0.11 total for 563 captions** with gpt-4o-mini (~490K input
  tokens, ~19K output tokens across the full run + test iterations).
- Ban marketing/vague-adjective filler explicitly by name in the system
  prompt (list of banned words: "playful", "character-inspired",
  "classic silhouette", "sturdy", "stylish", "sleek", "versatile",
  "unique", "iconic", etc.) — **gpt-4o-mini does not reliably honor
  negative/banned-word instructions even at temperature=0**, especially
  when the source scraped text itself uses that language (e.g. Adidas's
  "Pixar Toy Story" themed shoes kept pulling in "character-inspired"
  despite the explicit ban). A retry-loop-with-corrective-followup was
  prototyped and worked, but the user opted for simplicity over strict
  enforcement — current script does a single call, no retry. If caption
  purity matters more next time, either re-add the retry pattern or do a
  cheap regex post-filter pass.
- Capitalize brand names deterministically in code after generation
  (`fix_brand_casing()`) rather than trusting the model — it was
  inconsistent about "Adidas" vs "adidas" even when told to pick one.
- Gender/collection qualifiers ("men's", "junior", "kids") are omitted by
  default UNLESS they're the only thing distinguishing two otherwise
  near-identical SKUs (e.g. a "Junior" size-tier variant of the same
  colorway) — in that case surfacing it is essential for retrieval, so the
  rule is conditional, not a blanket ban.
- Records without scraped `details` (e.g. the ~99 New Balance variants that
  hit the Akamai block and never got enriched) still get captioned from
  just name/color/price — `build_prompt()` degrades gracefully via
  `record.get("details", {})`. Quality is still decent because gpt-4o-mini
  has real-world knowledge of well-known shoe models, but it's inferred
  rather than strictly grounded in scraped text — worth flagging as lower
  trust if this matters downstream.

## View classification (`classify_views.py`)

Since only Skechers has real per-image view labels, a zero-shot classifier
using the same Marqo FashionSigLIP model (`hf-hub:Marqo/marqo-fashionSigLIP`,
loaded via `open_clip`, run on MPS) was used to tag every image in all 4
brands consistently. 10 candidate view/prompt pairs (hero shot, side
profile, front three-quarter, back/heel, top-down, on-foot/lifestyle =
keep; sole/bottom, insole, material close-up, packaging = exclude).
Confidence scores from softmax over these prompts are low in absolute terms
(~0.10, barely above the 1/10 uniform baseline) because SigLIP is trained
with a sigmoid/contrastive objective, not calibrated softmax — **the
relative ranking (argmax) is still reliable**, confirmed via manual spot
image review (4/4 checks correct) before trusting it on the full set.
Result on 4334 images: 84.5% (3664) kept, sole shots were by far the
largest excluded category (662/670 excluded).

Non-destructive by design: writes `image_views.json` (path → view label +
confidence + keep bool) as a separate file; does not move or delete any
image. Physically filtering the dataset (vs. just tagging) was left as a
user decision, not yet applied as of this summary.

## Structured captioning (`caption_apparel.py`)

Added alongside the Nike clothing expansion because the downstream use case
grew from "one CLIP caption per shoe" to structured retrieval-training data
(taxonomy path, per-attribute breakdown, multiple positive texts at varying
specificity) — the schema and prompt live in `newLLMprompt.py`, filled from
each record's `name`/`brand`/`product_code`/`category`/`color_name` plus a
`DESCRIPTION` block built from `details.description` + joined
`features`/`product_details`.

- Sent as a single user message (the template already contains the full
  instruction set + schema + examples, no separate system prompt needed).
  The prompt already mandates "exactly one valid JSON object, no markdown" —
  parse with a strip-code-fences regex first, and if that still fails, retry
  **once** with a corrective follow-up message ("that wasn't valid JSON,
  return only the object") appended to the same conversation rather than
  restarting from scratch. In practice this never needed to fire (0/560 on
  the full backfill run).
- Every existing record needed a `category` field added (a one-off
  migration — all 563 pre-existing records are shoes, tagged
  `category: "sneaker"`) before this could run, since the prompt's
  `{{CATEGORY}}` slot and the later garment-cropping step both need it.
- Cost: **560 records backfilled for ~$0.17** with gpt-4o-mini (~519K input
  / ~153K output tokens) — cheap enough that re-running with `--force` to
  regenerate everyone isn't a real cost concern if the prompt changes again.
- Writes to a **new** `structured_caption` field; never touches or removes
  the old `caption` string field. Keep both — `caption` already cost money
  to generate and some downstream code may still read it.

## Garment cropping (`segment_apparel.py`)

Only applied to the Nike clothing records (shoe photos don't have the
"model's face/body dominates the frame" problem the same way apparel photos
do). Uses SAM2 automatic mask generation + FashionCLIP zero-shot
classification, adapted from the prototype in `fashionCLIP_SAM2.py` — but
getting from that prototype to something that runs safely and correctly
took several non-obvious fixes:

### Performance: never use MPS for SAM2 on Apple Silicon
The prototype script picks `device = "cuda" if available else "cpu"` —
it never uses `"mps"`. Rewriting it to prefer MPS on this Mac (reasonable-
looking "use the GPU if there is one" logic) made `SAM2AutomaticMaskGenerator
.generate()` take **over 2 minutes per image with no progress and no OOM**
(confirmed via direct timing test, not just a hunch) — SAM2 hits slow/
unsupported-op fallbacks on Apple's MPS backend that don't affect plain CPU.
Switching to `device="cpu"` explicitly (matching the original prototype)
dropped the same call to **7.2s**. If a script "based on" a working
prototype changes the device selection "for speed," verify that assumption
before trusting it — the opposite was true here.

### Performance: resize before mask generation
SAM2's automatic mask generator upsamples every candidate mask back to the
*input* image's resolution — these Nike product photos are natively
~2880x3600, and upsampling ~64-256 candidate masks (one per grid point) to
that resolution was the single biggest cost driver. Downsizing to max 1024px
on the long edge before calling `.generate()` (via `PIL.Image.thumbnail`)
took the same image from 42.3s down to 2.9-7.2s depending on point count —
larger effect than any point-count tuning. `points_per_side=16` (vs. the
default 32) plus this resize is the config that ended up safe and fast
enough: ~7s/image, ~2.5 hours for ~1300 images, peak RSS ~5.6GB (stable, not
a leak — confirmed by watching it plateau rather than climb over a full test
run) on a 16GB Mac with no dedicated GPU.

### Model size: switch to the "small" SAM2 checkpoint
`sam2.1_hiera_large` (898MB) is the prototype's default. Combined with MPS
(before that was fixed) it pushed the machine close to a crash — switched to
`sam2.1_hiera_small` (184MB, `configs/sam2.1/sam2.1_hiera_s.yaml`) as a
lighter default; masks are somewhat less precise but adequate for this
bounding-box-crop use case.

### Mask/crop selection: neither "highest score" nor "smallest area" alone works
The intuitive selection rule — among crops whose top FashionCLIP label
matches the record's category, pick the one with the highest confidence
score — **picks the whole person**, not the garment. CLIP-style models tend
to score the "typical" full framing of someone wearing a t-shirt *higher*
for the label "a t-shirt" than a tight fabric-only crop of the same shirt
(confirmed by dumping every candidate mask's area/score for a test image:
a 58%-of-frame "whole person" crop scored 0.991 for "a t-shirt", while a
25.7%-of-frame torso-only crop scored 0.983 — the tighter, better crop lost
by a hair to the looser one). The opposite rule — pick the *smallest*
qualifying crop — swings too far the other way and picks tiny homogeneous
fabric fragments (a sleeve, a waistband sliver) that coincidentally clear
the confidence threshold just by having plausible garment-colored texture,
without containing anything recognizable.

**What worked**: bound candidate masks to a mid-sized area-fraction band
(`0.08 <= area/full_image_area <= 0.55`) *and* a higher confidence floor
(`0.4`, up from an initial `0.25` that let fragments through), then pick the
**highest-scoring** crop only among that filtered band. This excludes both
failure modes (too-big, too-small) and consistently picked the actual garment
region across manual spot checks. These thresholds were tuned by hand against
one test image with full mask/score dumps, not derived analytically — worth
re-checking against a few images from any new category before trusting it.

### Crop shape: bounding box, not a mask-shaped cutout
`crop_masked_region()` crops to the mask's bounding box, not a pixel-exact
mask cutout with the background blanked out. This is deliberate, not a
shortcut: DINOv3/SigLIP-style ViT models are trained on natural, continuous
photos, and a hard mask cutout (background pixels replaced with a flat
fill color) creates out-of-distribution patch artifacts at every mask edge
that tend to hurt embedding/classification quality rather than help it —
production visual-search pipelines generally crop to a bounding box with a
little natural context for the same reason. A bounding-box crop still
includes a sliver of skin/background at the edges sometimes, but that's
preferable to an artificial blank region for anything feeding a ViT-based
model downstream.

### Safety: watch memory when running unattended
Peak RSS for this pipeline (FashionCLIP + SAM2 small, CPU) sits around
5.5-5.7GB and plateaus rather than climbing — but don't assume that from
theory alone on a shared machine. Wrap any long unattended run with a
watchdog loop (`ps -o rss= -p $pid`, kill if it crosses a hard ceiling well
under total system RAM) rather than trusting a "should be fine" estimate,
especially the first time a new model/config combination runs at scale.

### PacSun men's clothing — Pants, Graphic Tees, Sweaters, Shorts

Extending the pipeline to a new *site* (not just a new brand on an existing
site pattern) via `pacsun_scraper.py`, targeting 50 colorway variants per
section (200 requested; PacSun's `mens-sweaters` category only has 26 items
total, so 176 were actually reachable).

- **Plain `playwright`, even non-headless, gets an "Access to this page has
  been denied" bot-block page.** `patchright` (headed, `channel="chrome"`) —
  no extra anti-automation flags needed beyond that — gets through cleanly.
  Same mitigation family as New Balance, but PacSun blocks *harder*: it
  blocked plain Playwright even with the browser visibly open, where New
  Balance only blocked headless.
- **SFCC (Salesforce Commerce Cloud) site — same family as New Balance.**
  Category pages page via `Search-UpdateGrid?cgid={cgid}&start=N&sz=12`, an
  HTML fragment (not JSON) meant for AJAX infinite-scroll — fetched directly
  via `page.goto()` in the same browser context so it inherits the bot-check
  pass, then PDP hrefs are regexed out of the fragment. Page until a request
  returns zero *new* hrefs (matches the New Balance/Nike "page until
  exhausted" pattern) rather than trusting a fixed count.
- **Each PDP embeds a `ProductGroup` ld+json block**, same schema.org pattern
  as New Balance — but PacSun's `hasVariant` list is every *size* of one
  fixed colorway, not multiple colorways like New Balance's. Each colorway
  is already its own separate PDP/URL (the category grid lists colorways as
  distinct tiles) — so unlike the shoe brands, there is no groupKey/colorway-
  expansion step: one PDP visit = one dataset record. `productGroupID` in the
  ld+json (a 13-digit code) is the stable per-colorway `product_code`.
- **Human-readable color name is NOT in the ld+json** — the `hasVariant`
  entries only carry a numeric swatch code (e.g. `"349"`). The actual name
  lives in a rendered `aria-label="Select Color MEDIUM INDIGO"` attribute
  elsewhere on the page — regex that out and title-case it.
- **Description, "Fit & Sizing" bullets, and "Care & Composition" bullets
  (materials) are all server-rendered in accordion `<div>`s already present
  in the initial HTML** — no click/JS needed at all, unlike Adidas/New
  Balance's JS-gated accordions. This meant no separate `enrich_pacsun_*.py`
  detail-enrichment stage was needed — `details` (description/features/
  materials) is captured directly during the catalog scrape, in the same
  PDP visit that grabs images and price.
- **Product images already come as clean full-body lifestyle/model photos**
  (2-3 per colorway) suited to the same SAM2+FashionCLIP cropping step used
  for Nike clothing — no PacSun-specific tuning of `CATEGORY_LABELS` was
  needed beyond adding label phrasings for the 3 new category names not
  already covered (`"Pants"`, `"Graphic Tees"`, `"Sweaters"`; `"Shorts"` was
  already defined from the Nike expansion and reused as-is).
- **`patchright` held up for the entire run, not just the initial page
  load** — ~500 PDP/grid-fragment navigations across the full 176-record
  scrape (plus a full re-scrape after the data-loss incident below) hit zero
  mid-run bot-blocks. `pacsun_scraper.py` still checks every fetched page for
  the "Access to this page has been denied" string and backs off 15s if seen
  (same shape as New Balance's Akamai `is_blocked()` check), but that branch
  never fired in practice — unlike New Balance's Akamai, which blocks
  intermittently *during* a session even after the initial page loads fine.
- **`caption_apparel.py` is safe to run concurrently with `segment_apparel.py`**
  even under tight memory — it's I/O-bound (waiting on the Azure API, no
  local model), using ~50MB RSS, vs. SAM2+FashionCLIP's ~5GB+. Two
  `segment_apparel.py` invocations at once (different `--brand` filters) is
  riskier — one ran concurrently with this PacSun run for a while, and free
  memory dropped to double-digit MB (macOS paged rather than OOM-killing
  either, but it's not a configuration to rely on deliberately).

### Concurrent-write data loss incident (and the fix: `dataset_utils.py`)

While `pacsun_scraper.py` was scraping fresh records into
`apparel_dataset/metadata.json`, a **separate, already-running**
`segment_apparel.py --brand nike` process (started hours earlier, mid a long
unattended run) periodically checkpointed by writing back its own **stale
in-memory copy** of the full record list — loaded before the PacSun records
existed. Every one of its checkpoints silently overwrote the file, erasing
whatever PacSun had appended since. Net effect: a full category's worth of
freshly-scraped PacSun records (Shorts, all 50) vanished with no error from
either script — both scripts "succeeded" from their own point of view.

**Fix**: `dataset_utils.py` now provides `load_records()` /
`save_records_safe(touched: dict)` shared by every checkpointing script
(`pacsun_scraper.py`, `segment_apparel.py`, `caption_apparel.py`).
`save_records_safe` re-reads the current on-disk file at save time and
merges the caller's changed records into it (keyed by `product_code`)
instead of blindly overwriting with an old in-memory list. This doesn't
eliminate the race entirely — two processes could still both read-merge-
write in the same instant and one write could still lose — but it turns
"guaranteed to erase anything written since I started" into "only loses
data in an actual same-instant collision," and recovery is now just
re-running the affected scraper (already-scraped `product_code`s are
skipped, so a re-run only backfills what's actually missing).

**Lesson**: any two of this pipeline's scripts that checkpoint by
periodically rewriting the *entire* shared JSON file are unsafe to run
concurrently unless they merge-on-save. If you add a new script that
touches `apparel_dataset/metadata.json`, use `dataset_utils` — don't
`load()`-once-at-startup-then-blind-`write()`-on-checkpoint.

### Gap men's clothing — T-Shirts, Shorts, Pants, Sweaters

Via `gap_scraper.py`, targeting 50 colorway variants per section (200
requested; Gap's `mens-sweaters` category (cid `5180`) only had 26-27
colorways at scrape time, so 177 were actually reachable — same
"capped by catalog size" situation PacSun's `mens-sweaters` hit).

- **No bot protection anywhere** — plain `requests` (not even a browser)
  works for the entire catalog listing, the easiest tier of any site in
  this pipeline (matches Skechers). Category ids for men's clothing:
  T-Shirts `5225`, Shorts `5156`, Pants `80799`, Sweaters `5180`, all under
  `department=75`.
- **Catalog listing is a clean JSON API**, not an embedded blob or scraped
  HTML: `https://api.gap.com/commerce/search/products/v2/cc?pageSize=200&
  pageNumber=0&cid={cid}&department=75&vendor=constructorio&client_id=0&
  session_id=0&brand=gap&locale=en_US&market=us`. `pageSize=200` covered
  every category tested here in a single page (largest was 171 colorways).
- **One API "style" bundles every colorway inline** as a `styleColors` list
  — no separate per-colorway page visit needed to discover them, unlike
  PacSun (where each colorway was already its own separate PDP requiring an
  extra grid-fragment page visit). `ccId` (9-digit) is the stable
  per-colorway `product_code`; `styleId` is the shared product-family id.
- **Each image position (camera angle) carries ~10 resolution/crop variants
  of the same shot** (thumbnail, quicklook, hero, etc.) under one style
  color; the full-res one to keep is whichever `type` is exactly `"Z"`
  (position 1) or ends in `"_Z"` (`AV1_Z`, `AV2_Z`, ...) — Gap's zoom
  variant, confirmed 1500x2000 in spot checks vs. tiny thumbnails for the
  other types at the same position. Image paths are relative
  (`/webcontent/0061/457/472/cn61457472.jpg`) served off
  `https://www1.assets-gap.com`.
- **PDP url is trivially `https://www.gap.com/browse/product.do?pid={ccId}`**
  — no styleId, category, or slug needed in the URL at all.
- **The "Product details" and "Fabric & care" accordion bullets sit behind
  a React Suspense boundary that a plain `requests.get()` always receives
  as an empty `BAILOUT_TO_CLIENT_SIDE_RENDERING` placeholder — even though
  the heading text ("Product details") IS present in that same raw HTML.**
  This is a genuine trap, not just a missing-click accordion like
  PacSun/New Balance: checking for the heading substring as a "did this
  work" signal looks like success on every single PDP while silently
  returning zero bullets every time (confirmed on 15/15 sampled PDPs before
  catching it). The fix needed a real browser: plain headless `playwright`
  (no patchright, no anti-automation flags — Gap has no bot protection on
  the PDP either) with a ~2s wait renders the actual bullet content
  client-side. No separate `enrich_gap_details.py` stage was needed since
  everything (images/price/color from the API, description/materials from
  the rendered PDP) is captured in one pass per colorway — but that pass
  still needs a browser context open for the whole run, unlike the
  fully-`requests`-based catalog step.
- **Bullet text needs `html.unescape()`** — the rendered PDP HTML contains
  literal `&amp;` and `&nbsp;` entities inside the accordion text (e.g.
  "Career &amp; Enhancement", "Learn more&nbsp;here") that a plain
  tag-strip regex leaves un-decoded.
- Product images are already clean full-body/flat-lay product photos
  suited to the same SAM2+FashionCLIP cropping step used for Nike/PacSun —
  only needed adding `"T-Shirts"` to `segment_apparel.py`'s
  `CATEGORY_LABELS` (Shorts/Pants/Sweaters were already defined from the
  PacSun expansion and reused as-is).

### Champion men's clothing — Hoodies and Sweatshirts, T-Shirts and Tops, Shorts, Pants and Joggers

Via `champion_scraper.py`, targeting 50 colorway variants per section (200
requested, all 200 actually reached this time — no category came up short,
unlike PacSun/Gap/Levi's sweaters/accessories).

- **No bot protection at all, and the easiest data source of any brand in
  this pipeline** — champion.com runs on Shopify, and Shopify's standard
  storefront JSON API (`/collections/{handle}/products.json?limit=250&
  page=N`, paginated until a page returns zero products) returns the
  *entire* product record — full description, every colorway/size variant,
  every image — in one response. No separate PDP visit needed at all for
  anything, unlike every other brand in this pipeline (Gap/PacSun/New
  Balance all need at least one extra fetch for description bullets or
  detail data not present in the catalog listing).
- **Men's-scoped collection handles aren't guessable from the nav
  structure** — Champion's own top-level "men's clothing" URL 404'd, but
  the 404 page still rendered the full site nav HTML, which is where the
  real collection handles were scraped from (`mens-hoodies-sweatshirts`,
  `mens-t-shirt-tops`, `mens-shorts`, `mens-pants`). Deliberately avoided
  the unscoped `joggers`/`sweatpants` handles, which mix genders.
- **Each Shopify "product" is already ONE colorway** (color is baked into
  the title/handle, `options[0]` = Color with exactly one value on every
  product checked) — same "one PDP-equivalent per colorway" shape as
  PacSun/Gap, not Nike/New Balance's multi-colorway grouping. No
  colorway-expansion step needed: one API product = one dataset record.
  Numeric `id` is the stable `product_code`; `handle` is the slug.
- **Images are self-describing by filename position**, printed directly in
  the CDN URL (`..._Front1_...`, `..._Front2_...`, `..._Back1_...`,
  `..._Back2_...`, `..._Detail_...`, `..._Full_Length_...`) — the same
  kind of real, human-readable view-angle signal only Skechers had before
  this. Captured via `image_urls` (raw CDN URLs, filenames intact)
  alongside the renamed `image_N.jpg` files on disk, per lesson #5 below
  (view signal is lost forever once files are renamed unless saved first).
- **`body_html` is a single descriptive paragraph, not a bullet list** of
  materials/features the way Adidas/New Balance/PacSun structure their
  detail sections — stripped of HTML tags and entity-unescaped for
  `details.description`. No separate materials/features arrays populated
  for this brand; captioning still works fine from the description alone
  (same graceful-degradation path already proven for New Balance's
  Akamai-blocked records).
- Product images are full-body/lifestyle model photos, suited to the same
  SAM2+FashionCLIP cropping step as Nike/PacSun/Gap — needed adding
  `"Hoodies and Sweatshirts"`, `"T-Shirts and Tops"`, and `"Pants and
  Joggers"` to `segment_apparel.py`'s `CATEGORY_LABELS` (reusing the same
  label phrasings as the closest existing categories: Hoodies and
  Pullovers, Tops and T-Shirts, Pants and Tights respectively); `"Shorts"`
  was already defined and reused as-is.

### Levi's — Jeans, Jean Jackets, Shirts, Accessories

Via `levis_scraper.py`, targeting 50 colorway variants per section (200
requested; Levi's `Accessories` section only surfaced 28 unique products
across every listing/facet URL harvested, so 178 were actually reachable
— possibly a real catalog-size cap like PacSun/Gap's Sweaters, possibly
some genuinely reachable listing URLs weren't found; worth a follow-up
check if Accessories coverage matters, not confirmed either way).

- **Hardest bot protection of any brand in this pipeline so far** —
  Akamai Bot Manager, but a harder variant than New Balance's binary
  block: plain `requests` gets an immediate edge "Access Denied"
  (`errors.edgesuite.net`) on every page, and even `patchright` (headed,
  `channel="chrome"`) initially loads an interactive Akamai *behavioral
  challenge* interstitial (`sec-if-cpt-container`, "Powered and protected
  by Akamai") rather than a flat allow/block. Unlike New Balance,
  patchright's traffic here looks human enough that the challenge
  auto-resolves and the page reloads itself within ~3-9s if you just
  wait — **poll `page.title()` in a loop until it changes away from
  empty/"Access Denied" rather than treating the first response as
  final**, a short fixed sleep is fragile. This challenge-wait isn't a
  one-time session unlock either — every fresh `page.goto()` to a new PDP
  can re-trigger it, so every page visit needs the same polling logic.
- **Vue.js SSR app** (not Next.js/Nuxt despite `data-v-*` hydration
  markers), with its own `window.__LSCO_INITIAL_STATE__` blob — but this
  is a red herring, not a usable data source: it gets **deleted from
  `window` after hydration completes** (confirmed via
  `Object.getOwnPropertyNames(window)` no longer listing it a few seconds
  after load).
- **Real, stable data source is a schema.org ld+json `ProductGroup`
  block** on every PDP, same family as New Balance/PacSun's pattern:
  `hasVariant` is a list of per-colorway `Product` objects, each with its
  own `sku` (stable `product_code`), `color`, `image` (full CDN URLs, no
  query string), `offers.price`, and a `description` inherited from the
  parent product family. One PDP visit yields every colorway inline, same
  efficiency as Gap's `styleColors` bundling.
- **`hasVariant` sometimes mixes bare `{"url": ...}` sibling-colorway
  links in alongside the real per-colorway `Product` dicts**, not nested
  under a separate key — filtering to `isinstance(v, dict)` before
  reading `v["sku"]` is required, or every single PDP throws `'list'
  object has no attribute 'get'`.
- **Category listing pages cap at exactly 38 PDP links per URL,
  regardless of category** (confirmed identical across jeans/jean-
  jackets/shirts/accessories main pages) — no infinite-scroll or "load
  more" trigger increases this. Getting past 38 unique products per
  section required harvesting PDP links across multiple listing URLs for
  the same section (fit/subcategory pages for jeans, colorgroup facet
  pages for jean jackets/shirts, sub-department pages for accessories)
  and deduping by PDP href.
- **Image CDN is Scene7** (`lscoglobal.scene7.com`, same dynamic-imaging
  family as other Adobe-Scene7-backed retailers). ld+json image URLs are
  usually bare but sometimes already carry a Scene7 preset query string —
  blindly appending explicit-size params produces a malformed double-`?`
  URL that 403s. Always strip any existing query string first
  (`url.split("?")[0]`) before appending size params, same convention as
  Gap's zoom-variant sizing. No CDN-embedded view/angle codes found
  (unlike Skechers/Adidas) — images are just sequentially ordered.
- **"Composition & Care" bullet text is lazy-rendered below the fold** —
  absent from `page.content()` until the page is scrolled, even though
  it's not behind an accordion click (no JS-gated click needed, unlike
  Adidas/New Balance — just scroll-into-view). No separate
  `enrich_levis_details.py` stage needed once scrolled.
- **First brand whose "Accessories" category is genuinely heterogeneous**
  (verified against real scraped records, not assumed): backpacks, hats,
  belts, wallets, bandanas, even underwear, all under one category
  string — unlike every other category in this pipeline, which maps to
  one visually consistent garment silhouette. `segment_apparel.py`'s
  `CATEGORY_LABELS["Accessories"]` lists every real accessory type found
  (`"a hat"`, `"a belt"`, `"a backpack"`, `"a bag"`, `"a wallet"`,
  `"a bandana"`, `"underwear"`) rather than one or two phrasings — its
  crop-selection logic only requires the top-scoring label to be ANY
  phrasing in the category's list, not one fixed label, so this correctly
  classifies whichever specific item is actually in a given photo instead
  of forcing one label onto a mixed bucket. `"Jeans"`, `"Jean Jackets"`,
  and `"Shirts"` also added to `CATEGORY_LABELS`.

### Carhartt WIP — Jackets and Coats, Pants, Shirts, T-Shirts and Polos

Via `carhartt_scraper.py`, targeting 50 colorway variants per section
(200 requested, **all 200 reached, no category shortfall**).

- **No real bot protection** — plain `playwright` (headless,
  `--disable-blink-features=AutomationControlled`), same tier as Nike/
  Gap. A fixed-position `[data-rac]` overlay (cookie-consent/region
  modal) intercepts clicks on first load — removed with a one-line
  `page.evaluate()` (`document.querySelectorAll('[data-rac]').forEach(e
  => { if (getComputedStyle(e).position === 'fixed') e.remove(); })`),
  no dismiss-button click needed.
- **First commercetools-backed site in this pipeline.** Next.js App
  Router (React Server Components) — the classic `__NEXT_DATA__` script
  tag pattern this pipeline usually checks first is **absent entirely**
  here; guessing category URLs by analogy to other sites' patterns
  (`/en/men/clothing/jackets`) 404s outright. Real category listing URLs
  only surfaced by reading actual nav `<a href>` values off the rendered
  homepage: `/en-de/c/{category-slug}` (`men-jackets-and-coats`,
  `men-pants`, `men-shirts`, `men-tshirts-and-polos`), paginated via
  `?page=N`, 48 products/page. Each listing link is already a distinct
  per-colorway PDP (`/en-de/p/{slug}-{trailing-id}`) — one PDP = one
  colorway, no separate expansion step, same shape as PacSun/Gap (not
  New Balance/Levi's, where one PDP inlines every colorway).
- **Real, stable data source is a schema.org ld+json `Product` block**
  (not `ProductGroup`) on every PDP: `name`, `image`, `sku`, `size`,
  `material`, `color`, `brand`, `offers.price`/`priceCurrency`. No
  `description` field in the ld+json itself.
- **Description + feature bullets live in a native HTML `<details>`
  element**, content present in the DOM regardless of open/collapsed
  state (unlike Adidas/New Balance's JS-gated accordions) — `element.
  innerText` reads it directly, no click needed. The accordion text's
  last line is always the base image code (e.g. "I037132_3ZO_XX",
  matching `product_code`) — junk if treated as a feature bullet, must
  be filtered out explicitly.
- **Image CDN is Amplience** (`cdn.media.amplience.net`), a different
  dynamic-imaging vendor than the Scene7 (Adobe)/Shopify CDNs seen on
  other brands. ld+json image URLs carry a small schema-markup preset
  query string — strip it, append `?w=1600&fmt=auto&qlt=default` for
  full-res, same "always strip existing query string first" convention
  as Gap/Levi's zoom sizing. **Enforces hotlink protection**: a plain
  `requests.get()` with only a User-Agent gets a 403 (confirmed via
  direct testing, not assumed) — a `Referer` header pointing at the
  site's own domain is required and sufficient, no cookies/session state
  needed.
- **`product_code` needed deriving, not reading directly** — the
  ld+json `sku` field ("I037532_1") is style-level only, NOT
  per-colorway unique. The real stable per-colorway identifier is the
  filename prefix every one of a PDP's own images shares (e.g. every
  image for one PDP shares "I037532_453_02" before the "-OF-NN"/"-ST-NN"
  view-type suffix) — extracted via regex from the first ld+json image
  URL, with a defensive fallback to the PDP URL's trailing numeric ID if
  that pattern ever fails to match.
- **Product photos are already clean flat-lay shots on a transparent/
  plain background — no model, no lifestyle context.** Confirmed by
  direct visual inspection of multiple real downloaded images across
  different categories (a t-shirt and a jacket both checked), not
  assumed from the CDN/site type. This means the SAM2+FashionCLIP
  garment-cropping step (needed for Nike/PacSun/Gap/Champion/Levi's) was
  correctly **skipped entirely** for this brand — same reasoning this
  project already established for the original shoe photos ("shoe
  photos don't have the same 'model's face/body dominates the frame'
  problem apparel photos do"), just discovered here for a clothing brand
  rather than assumed only to apply to footwear.
- **Schema extension landed the same session, validated in practice**:
  `newLLMprompt.py` had `pocket_type`/`distressing`/`heel_type`/
  `sole_type`/`toe_shape` added to its attribute schema shortly before
  this scrape ran (2026-08-02, closing a real gap where spec section 4.5
  listed these attribute groups but the caption prompt never asked the
  LLM for them). Carhartt WIP's workwear-heavy catalog was a good real
  test: the very first captioned record populated `"pocket_type":
  ["side welt"]` from real "side pockets" language in the scraped
  details — confirms the new fields are being used by the LLM in
  practice, not just present in the schema unused.

### Stüssy — Tees, Pants, Headwear, Hoodies

Via `stussy_scraper.py`, targeting 50 colorway variants per section (200
requested; Stüssy's `hoodies` collection only had 25 products total at
scrape time, confirmed via a `products.json?limit=250` count check before
scraping even started — a real, known-in-advance catalog-size cap, not a
scraping failure, so 175 were reachable).

- **No bot protection encountered** — plain `requests` works for the
  entire catalog + product detail fetch, same easiest tier as Champion/
  Gap/Skechers.
- **Second Shopify-storefront site in this pipeline** (after Champion),
  same `https://www.stussy.com/collections/{handle}/products.json?
  limit=250&page=N` pattern, paginated until a page returns zero
  products. Each Shopify "product" here is already ONE colorway
  (`options[0]` = Color with exactly one value) — same "one API product
  = one dataset record" shape as Champion, no colorway-expansion step
  needed. Full product detail (description bullets, all images, size/
  price/SKU data) is inline in the same response, no separate PDP visit.
- **Real collection handles found via the site's own sitemap**, not
  guessed — initial guesses like `mens-tees` 404'd/returned empty.
  `sitemap.xml` → `sitemap_collections_1.xml?from=...&to=...` (the
  `from`/`to` query params are required, a bare fetch 400s) lists every
  real collection handle. Handles used: `tees`, `hoodies`, `pants`,
  `headwear`.
- **`body_html` is a real bullet list** (`<ul><li>...`), unlike
  Champion's single free-text paragraph — stripped of HTML tags for
  `details.description` with the same `strip_html()` helper regardless.
- **Third brand (after shoes, Carhartt) whose product photos are already
  clean flat-lay shots**, verified by direct visual inspection across
  Tees/Pants/Headwear (not assumed from one category) — garment cropping
  correctly skipped entirely, same judgment call and reasoning as
  Carhartt.
- Structured captioning: 175/175 records, $0.0555 real Azure OpenAI
  cost. Spot-checked output populates `pocket_type`/`closure`/
  `defining_features` meaningfully from real scraped bullet text (e.g.
  a basic tee's `closure: ["crewneck"]`, `defining_features` correctly
  citing "basic logo screenprint" with its real chest/back location) —
  consistent with the newLLMprompt.py schema extension already validated
  working on Carhartt's records.

### Vans — Shoes, Hoodies and Jackets, Shirts

Via `vans_scraper.py`, biased toward footwear (target 100 shoes + 50 each
of two clothing categories, ~200 total) since every apparel-only brand
added earlier this session left footwear coverage stuck at the original 4
shoe brands (Nike, Adidas, New Balance, Skechers) — Vans's real value is
shoe diversity, not more clothing. **All 200/200 reached, no shortfall**:
Shoes 100/100, Hoodies and Jackets 50/50, Shirts 50/50.

- **Real bot protection**: Akamai Bot Manager, same edge "Access Denied"
  (`errors.edgesuite.net`) signature as Levi's/New Balance on plain
  `requests`. Softer than Levi's though — plain `patchright` (headed,
  `channel="chrome"`) gets through cleanly with no visible interactive
  challenge screen at all, unlike Levi's behavioral-challenge
  interstitial. Closer to New Balance's "binary block, patchright passes"
  tier.
- **VF Corp brand, Nuxt.js storefront** — real, stable data source is
  schema.org ld+json, in two different shapes depending on page type:
  category listing pages (`/en-us/c/...`) carry a `CollectionPage` node
  with `mainEntity.itemListElement` (48 products/page, real `?page=N`
  pagination confirmed — different products per page, not a silent
  no-op), each item already including name/url/category/price/full image
  array with NO PDP visit needed for those fields. Product pages
  (`/en-us/p/...`) carry a `ProductGroup` node whose `hasVariant` is SIZE
  variants of one fixed colorway (color constant across every entry) —
  same "one PDP per colorway" pattern as PacSun, not New Balance's
  "one PDP has every colorway." `productGroupID` (e.g. `VN000D9RWVD`) is
  the stable `product_code`, same code embedded in the listing page's own
  image filenames and the PDP URL slug — no cross-referencing needed.
- **Real category paths found via the commerce sitemap**
  (`vans.com/sitemap.xml` → `sitemaps/commerce/commerce-en-us.xml`), not
  guessed — initial guesses (`/en-us/mens-shoes`) 404'd *inside the Nuxt
  app itself* (confirmed via `__NUXT_DATA__`'s own `statusCode: 404`
  field), a genuine in-app 404, not a bot block, and a different failure
  mode than every prior brand's URL-guessing misses (those all just
  returned zero results, not a real error page).
- **Product-detail bullets are server-rendered, no click needed**:
  `data-test-id="product-details-bulletin"` on the PDP, a plain
  `<ul><li>` list readable directly from `page.content()` — real feature/
  construction text (e.g. "Foxing tape inspired by original 90's Osnaburg
  reinforced outsoles"), not marketing filler. 30/200 records have empty
  bullets (some products genuinely lack this section) — same graceful
  degradation as every prior brand's `details.get("features", [])`.
- **Images are on a Cloudinary-style dynamic-imaging CDN**
  (`assets.vans.com/images/t_img/...`). The listing page's own embedded
  image URLs are low-res thumbnails (`t_Thumbnail` transform); real
  full-res (2000x2500, confirmed by downloading and checking actual
  pixel dimensions, not assumed from the URL alone) images are available
  by rewriting the transform segment to
  `t_img/c_fill,g_center,f_auto,h_2500,w_2000/` while keeping the same
  `{SKU}-{VIEW}` image-id fragment (`-HERO`, `-ALT1..4`) already present
  in the listing page's own data — no separate PDP visit needed just to
  discover image URLs.
- **First brand this session whose product photos are genuinely mixed**
  — NOT uniformly flat-lay like Carhartt/Stüssy, confirmed by directly
  inspecting real images across both clothing categories (some
  graphic tees ARE clean flat-lays, but hoodies/jackets and some shirts
  are real on-model lifestyle photos). Garment cropping (`segment_
  apparel.py --brand vans`) run for real here, unlike the last two brand
  additions — needed a new `CATEGORY_LABELS["Hoodies and Jackets"]`
  entry combining both hoodie and jacket phrasings (Vans's own category
  bundles both garment types under one name), reusing `"Shirts"`'s
  existing phrasing from the Levi's expansion. Shoes correctly excluded
  from cropping automatically (not in `CATEGORY_LABELS` at all, matching
  the original 4 shoe brands' convention) — no code path even attempts
  to crop shoe photos.
- Structured captioning: 200/200 records, $0.0696 real Azure OpenAI
  cost.

### Dickies — Pants, Shirts, Shorts, Coats and Jackets

Via `dickies_scraper.py`, targeting 50 colorway variants per section
(200 requested). **All 200/200 reached, no shortfall.**

- **No bot protection** — plain `requests` works for the entire catalog
  + product detail fetch, same easiest tier as Champion/Stüssy/Gap/
  Skechers. Third Shopify-storefront site in this pipeline (after
  Champion and Stüssy).
- **Real collection handles found from the homepage's own nav links**
  (`href="/en-us/collections/..."`), not guessed: `mens-pants`,
  `mens-shirts`, `mens-shorts`, `mens-coats-jackets`. A `products.json?
  limit=250` count check before scraping confirmed real inventory —
  `mens-pants`/`mens-shirts` both hit the 250-per-page cap, meaning
  pagination was needed to reach the real, larger inventory.
- **Standard Shopify storefront JSON API**, same pattern as Champion/
  Stüssy: `https://www.dickies.com/en-us/collections/{handle}/
  products.json?limit=250&page=N`, paginated until a page returns zero
  products. **Note the `/en-us/` locale prefix** — unlike Champion/
  Stüssy's bare `/collections/...`, Dickies's storefront is
  locale-scoped and the `products.json` endpoint 404s without it.
- **Each Shopify "product" is already one colorway** (`options[0]` =
  Color with exactly one value per product) — same "one API product =
  one dataset record" shape as Champion/Stüssy, no colorway-expansion
  step needed. Numeric `id` is the stable `product_code`; `handle` is
  the slug.
- **Full product detail is inline in the same `products.json`
  response** — no separate PDP visit needed, same as Champion/Stüssy.
  `body_html` is a real paragraph (not a bullet list like Stüssy's),
  stripped of HTML tags for `details.description`. Price:
  `variants[0]['price']`, plain decimal string. Sizes/inseams are
  separate `options` entries (Size, Inseam for pants) nested under one
  colorway product — only product-level fields were scraped, not
  per-size variant fields, so no special handling was needed.
- **Real on-model lifestyle photos** — confirmed by direct image
  inspection (not assumed), unlike Carhartt/Stüssy's uniformly flat-lay
  photos. Garment cropping (`segment_apparel.py --brand dickies`) run
  for real, same as Vans — needed one new `CATEGORY_LABELS["Coats and
  Jackets"]` entry (jacket/coat phrasings, reusing the same pattern as
  every prior jacket-family category).
- Structured captioning: 200/200 records.

### Field-level concurrent-write collision (a second, narrower case dataset_utils doesn't fully cover)

Running `caption_apparel.py --brand gap` and `segment_apparel.py --brand gap`
concurrently on the *same* 177 records (following the documented "safe to
run concurrently, I/O-bound vs. SAM2" guidance from the PacSun run) lost 175
of 177 `structured_caption` fields, even though `dataset_utils.
save_records_safe` was used by both scripts. Captioning finished first
(cheap/fast) and merged its `structured_caption` field onto disk; the much
longer `segment_apparel.py` run had already loaded its own in-memory copy of
every gap record *before* captioning wrote anything, so its later
checkpoints kept calling `save_records_safe({code: record})` with a
`record` object that still had no `structured_caption` key — and
`save_records_safe` merges *whole records* keyed by `product_code`, not
individual fields, so each of segment's checkpoints clobbered the caption
that had just landed for that same code.

**Why the earlier PacSun incident's fix didn't catch this**: that fix
(re-read-and-merge-by-`product_code`) fully solves the case where two
scripts touch *disjoint* records (one brand/category appending new records
while another processes a different brand) — the original incident. It does
NOT solve two scripts both mutating fields on the *same* records at the
same time, because the merge granularity is per-record, not per-field. A
script that loads a record, adds one field, and checkpoints the whole
record back will always overwrite any other field added by someone else in
the meantime, `save_records_safe` or not.

**Recovery**: simply re-ran `caption_apparel.py --brand gap` once
`segment_apparel.py` had fully finished — cheap ($0.047, ~175 records) and
safe by design (only processes records missing `structured_caption`).

**Lesson**: "safe to run concurrently" only holds when the two scripts
touch disjoint records. Two checkpointing scripts that will mutate
different fields on the *same* records should still be run sequentially,
not concurrently — `dataset_utils` protects against a stale full-file
overwrite, not a stale full-record overwrite from a script whose in-memory
snapshot predates a field that landed on disk after it loaded.

## Directory / data conventions

Everything lives directly under one shared dataset folder — scrapers write
here from the start, there is no per-brand staging folder to merge later:

```
apparel_dataset/{brand}/{slug}/{product_code}/image_N.jpg  # all images
apparel_dataset/metadata.json                              # all records
image_views.json                                            # image path -> view tag
```

(The original 4-brand shoe run predates this convention and scraped into
`{brand}_catalog/` + `{brand}_products.json` per brand, then used a one-off
`build_dataset.py` to copy/merge into what was then called `shoe_dataset/`
— that folder has since been renamed to `apparel_dataset/` and is now the
single target going forward. New scrapers for other product categories
should append/update records straight into `apparel_dataset/metadata.json`
and download straight into `apparel_dataset/{brand}/...`, tagging each
record's `brand`/category appropriately so multiple product types can
coexist in the same dataset folder.)

`apparel_dataset/metadata.json` record schema (after all enrichment stages;
`category`, `structured_caption`, `cropped_images` were added for the Nike
clothing expansion — older shoe records have `category` backfilled but no
`cropped_images` since cropping was never applied to shoes):
```json
{
  "brand": "...", "category": "sneaker | Tops and T-Shirts | Shorts | ...",
  "name": "...", "color_name": "...", "price": "...",
  "product_code": "SKU", "slug": "...", "product_url": "...",
  "image_count": N, "images": ["path", ...], "image_urls": ["cdn url", ...],
  "cropped_images": ["path", ...],
  "details": {"description": "...", "features": [...], "materials": [...]},
  "caption": "one-line CLIP caption",
  "structured_caption": {
    "product_id": "...", "positive_texts": ["...", ...],
    "taxonomy_path": ["...", ...],
    "attributes": {"color": [...], "material": [...], "defining_features": [...]}
  }
}
```

## Lessons learned (apply these to any new scraping target)

1. **Never `rm -f` a scraped-data JSON to "reset" test state.** Strip the
   specific field you're re-testing on the affected records instead
   (`del record["details"]` on N records, rewrite file). Some fields
   (New Balance's `master_id`-bearing `product_url`) cannot be
   reconstructed from anything else on disk — deleting the file cost real
   recovery time (had to re-derive master IDs via targeted site searches
   per orphaned style code).
2. **Watch disk space aggressively when scraping images.** This machine's
   root volume runs chronically near-full from unrelated things, and a
   scraper crashing mid-`write_text()` due to `ENOSPC` can (rarely, but
   possibly) leave a JSON file truncated — always verify
   `json.load()` succeeds immediately after any crash before assuming data
   is intact. Checkpoint the JSON to disk every N records (10-20), not just
   at the end, so a crash loses at most one checkpoint interval.
3. **When a script "completes successfully" after a network hiccup, verify
   the actual content, not just that a field exists.** The first New
   Balance detail-enrichment run reported "173/173 have details" but 139 of
   those were silently empty (Akamai serving a block page that the script
   didn't detect, so it happily parsed nothing into an empty dict). Add an
   explicit "is this actually the expected page" check
   (title/body-text sniff for the site's known error-page markers) rather
   than trusting HTTP 200 + no-exception as success.
4. **Always check for a hidden JSON blob before writing CSS-selector
   scraping logic.** `__NEXT_DATA__` (Next.js), ld+json `ProductGroup`
   (schema.org, common on SFCC/Salesforce Commerce Cloud sites like New
   Balance), and direct XHR/API endpoints (New Balance's
   `Product-Variation` SFCC endpoint) are far more complete and stable than
   scraping rendered DOM text, and often contain fields the visible page
   doesn't even show.
5. **If you want per-image view/angle labels, capture them at scrape time**
   — check CDN filenames/URLs for embedded codes (Skechers: `_HERO_`,
   `_INSOLE_`, `_OUTSOLE_`, `_PROFILE_NN`; Adidas: `_HM1`-`_HM9`/`_HB1`-`_HB9`)
   and save them into the JSON immediately. Renaming files to generic
   `image_N.jpg` for a clean directory structure destroys this signal
   permanently unless you save it elsewhere first.
6. **Azure AI Foundry endpoint gotcha**: strip the `/api/projects/...`
   suffix the Foundry UI shows you — the OpenAI SDK wants the bare resource
   domain.
7. **gpt-4o-mini does not reliably obey banned-word lists**, even at
   temperature 0, when the source material keeps nudging toward that
   language. If strict compliance matters, verify programmatically
   (substring check) and either retry-with-correction or regex-strip,
   don't rely on the prompt alone.
8. **Don't assume a "based on a working prototype" rewrite preserves its
   working behavior if you change a config value that looks like a pure
   improvement.** Switching a device string from `"cpu"` to a preferred
   `"mps"` looked like a straightforward speedup; it was actually the whole
   cause of a >2min-per-image stall on SAM2. When a script derived from a
   known-working one behaves much worse, diff the actual runtime config
   against the original before assuming the *logic* changed — check what
   changed in *how* it runs, not just what it computes.
9. **The intuitive "pick whichever crop/label scores highest" selection
   rule is often wrong when the goal is a tight/precise region, not a
   confident one.** CLIP-style models reward "typical" framing, which for a
   garment often means the whole photographed person, not a tight crop of
   just the fabric. When tuning any selection-by-score logic meant to
   isolate a *sub-region*, dump the full candidate list (score + size) for
   at least one real example before trusting either "highest score" or
   "smallest region" as the rule — the actual answer here needed both a
   confidence floor *and* an area band, not a single sort key.
10. **Verify a "no OOM, memory looks fine" background run isn't actually
    hung.** Low, non-growing CPU time over a couple of minutes can mean
    "computing efficiently on a GPU/accelerator with little CPU-side work"
    (fine) or "stalled" (not fine) — memory staying flat doesn't distinguish
    these. Time a single unit of work (one image, one record) directly with
    explicit timestamps before trusting a multi-hour unattended run to
    finish in a reasonable time.
11. **Never let a checkpointing script blindly overwrite the shared
    metadata JSON with its own in-memory copy — always merge-on-save.** A
    long-running `segment_apparel.py` loaded the record list once at
    startup, then a separate `pacsun_scraper.py` run appended new records
    to the same file while it was still running; every subsequent
    checkpoint from the first script rewrote the file from its stale
    snapshot and silently erased the new records (see "Concurrent-write
    data loss incident" above). Fixed via `dataset_utils.save_records_safe`,
    which re-reads and merges by `product_code` at save time instead of
    overwriting wholesale. Any new script that checkpoints against
    `apparel_dataset/metadata.json` must use it.
12. **A page's raw HTML can contain a section's heading text while the
    section's actual content never rendered** — don't treat "the label I'm
    looking for is present in the response" as proof the data came through.
    Gap's PDP wraps "Product details"/"Fabric & care" bullets in a React
    Suspense boundary that `requests.get()` always receives as an empty
    `BAILOUT_TO_CLIENT_SIDE_RENDERING` placeholder, but the heading text sits
    outside that boundary and is present either way — so a naive
    `"Product details" in html` check reports success on every PDP while
    silently returning zero bullets every time. Caught by dumping and
    reading one full extracted-bullets result before trusting the batch, not
    by the presence check itself. When a field that should be non-empty
    keeps coming back empty, check for a framework-specific bailout/
    placeholder marker in the raw response, not just the field's own
    absence.
13. **"Safe to run concurrently" (per lesson 11's `dataset_utils` fix) only
    covers two scripts touching *disjoint* records — not two scripts
    mutating *different fields on the same records* at the same time.**
    Running `caption_apparel.py --brand gap` alongside `segment_apparel.py
    --brand gap` (following the earlier PacSun-era guidance that
    caption_apparel is I/O-bound and cheap to run alongside one segment job)
    lost 175/177 `structured_caption` fields: `save_records_safe` merges
    whole records by `product_code`, so segment's long-running in-memory
    copy (loaded before captioning wrote anything) kept overwriting the
    caption field on every checkpoint even though the merge-by-code logic
    worked exactly as designed. Fixed by simply re-running the captioning
    pass once segmentation finished (cheap, idempotent). If two scripts will
    both write fields onto the *same* set of records, run them
    sequentially — `dataset_utils` prevents a stale full-file overwrite, not
    a stale full-record overwrite from a snapshot taken before a sibling
    field landed on disk.
14. **Not every script that touches `metadata.json` was actually updated
    to use `dataset_utils` — lesson 11's fix doesn't protect a script
    nobody remembered to migrate.** `build_hierarchy.py` (analysis/
    canonicalization tooling, not a scraper) still did a bare
    `METADATA_PATH.read_text()` → mutate in memory → `write_text()`
    round trip, the exact pattern lesson 11 fixed everywhere else.
    Running it once (2026-08-01, to regenerate the category tree) while
    the Levi's and Champion scraper forks were actively mid-write reverted
    the file to this script's stale read-time snapshot, dropping Levi's
    from 50 records to 3 — the forks' own safe-merge writes self-healed it
    within a few checkpoints (images were untouched on disk, so recovery
    was just re-fetching the metadata), but it shouldn't have happened.
    Fixed by switching `build_hierarchy.py` to
    `dataset_utils.load_records()` / `save_records_safe()` like everything
    else. **Lesson: when adding ANY new script that reads then writes
    `apparel_dataset/metadata.json` — including one-off analysis/tooling
    scripts, not just scrapers/captioners/segmenters — use `dataset_utils`
    from the start. Don't assume a script is safe just because it "only
    reads for analysis" if it also writes a derived field back.**

## Real-outfit scraping (`outfit_dataset/`) — a SEPARATE dataset, added 2026-08-03

Everything above scrapes **product** photos: one garment, studio lighting,
known category, ground-truth brand from the source. This section covers a
fundamentally different target added 2026-08-03: **photos of real people
wearing real outfits** (influencers, street style, community posts) — the
actual deployment condition the whole system is meant to work on, and the
spec's Phase 3 (`docs/project_spec_v1.md` §4.2, catalog-to-consumer).

### Hard rule: never write these into `apparel_dataset/`

`apparel_dataset/metadata.json` is the **retrieval gallery**. Every eval
number this project has ever produced assumes every record in it is a
clean, single-product, known-identity catalog entry. Dropping unlabeled
consumer photos in there would silently pollute the gallery — the DINOv3
identity index would build embeddings for "products" that aren't products,
and every R@K number would break with no error and no warning. Outfit data
lives in its own tree with its own metadata file:

```
outfit_dataset/{source}/{source_id}/image_N.jpg
outfit_dataset/metadata.json
```

### These images are deliberately UNLABELED

Decided explicitly by the user, 2026-08-03: do **not** try to label which
garments appear in these photos, and do not attempt product linkage.
Customer-review photos on product pages were considered first and rejected
for this purpose — they'd give free ground-truth product linkage, but
they're not what the product needs to show; genuine influencer/worn
outfits are. The labeling will come later from this project's own pipeline
(`segment_outfit.py` → SigLIP2/DINOv3), run unsupervised, with **no ground
truth about whether the labels are right**. That's accepted and understood.

Consequence for scrapers: a scraper's job is images + provenance, nothing
more. Any garment/category/product field must stay absent — not guessed,
not filled with a placeholder. Model-derived fields get written later, by
separate scripts, into their own namespaced keys, so it's always
unambiguous which fields are scraped fact and which are model output.

### `outfit_dataset/metadata.json` record schema

```json
{
  "source": "reddit | lookbook | pexels | ...",
  "source_id": "stable per-source unique id (e.g. reddit post id)",
  "post_url": "...",
  "author": "handle/username, for attribution",
  "title": "post title / caption as published",
  "section": "subreddit, site category, or search term used",
  "created_utc": 1234567890,
  "source_tags": ["only tags the SOURCE provided -- never model-derived"],
  "image_count": 2,
  "images": ["outfit_dataset/reddit/abc123/image_0.jpg"],
  "image_urls": ["https://i.redd.it/..."],
  "scraped_at": "2026-08-03T00:00:00Z",
  "phash": ["perceptual hash per image, for dedup"]
}
```

### Conventions specific to this target

1. **Dedup with perceptual hashing, not URL equality.** Reposts are
   endemic on community sources — the same outfit photo reappears under
   different post ids, different hosts, and resized. URL/id dedup will not
   catch it. Hash every image on download (`imagehash.phash` or
   equivalent), skip near-duplicates within a small Hamming distance.
2. **Scrape broadly now, filter later.** A useful photo shows a person
   wearing multiple visible garments. Verifying that needs a person
   detector, which is exactly the pipeline being built — so don't gate
   scraping on it. Pull widely, keep provenance, filter in a later pass.
3. **Keep provenance complete enough to delete from.** Store post URL and
   author on every record. If a takedown or license question ever comes up,
   the record must be traceable back to its source without re-scraping.
4. **Respect robots.txt and rate limits, and identify the client.** Same
   discipline as the brand scrapers, and more important here: these are
   community sites and individuals' own photos, not corporate catalogs.
5. **Same idempotency/checkpointing rules as every other scraper here** —
   checkpoint `metadata.json` every 10-20 records, never `rm` it to reset
   state, verify `json.load()` after any crash (see "Lessons learned").

### Running it at scale — the procedure, and what it costs

Added 2026-08-03 after the first bulk collection run.

```
./run_reddit_wide.sh        # 17 shards, one per subreddit, ~150-200 images/min
python dedup_outfits.py     # report cross-shard duplicates
python dedup_outfits.py --apply
```

Four operational facts that are not obvious from the scrapers themselves:

1. **Fan out, don't run one process.** The reddit scraper is network-bound,
   not CPU-bound — a single process spends nearly all its wall clock in
   `requests.get` and in the politeness sleeps, and measured ~6 images/min.
   One process per subreddit measured **150-200 images/min** while staying
   under ~15% of one core. The Arctic Shift API is not the limit (1.1s per
   100-post page, no 429s observed with 17 concurrent clients).

2. **Sharding weakens dedup, so sweep afterwards.** Each process loads its
   own in-memory phash list at startup. Dedup is exact within a shard and
   blind across them, and reposts across these subs are endemic. That is
   what `dedup_outfits.py` is for — it is part of the procedure, not
   cleanup. It re-reads under the lock, so it is safe to run while
   scrapers are still going.

3. **Expect ~45% of image fetches to 404, and don't try to fix it.** That
   is Reddit media deleted after the archive snapshotted the post, and it
   rises with post age. Verified on 25 such posts: the `preview` blob
   recovers **zero** of them, so there is no fallback ladder worth adding.
   Budget for it — a 300-image target scans far more posts than 300.

4. **Browser-driven sources are expensive and must be run alone.** Two
   headless Chromium instances took this machine to **load average 68**
   alongside the segmentation job, as the top two CPU consumers at 96% and
   64%. Never run `browser_outfit_scraper.py` concurrently with model or
   segmentation work, and never two of it at once. Reddit is both the
   deepest source and the cheap one; the browser is a supplement for an
   idle machine.

### Verifying a subreddit: sample the API, then audit what the shard wrote

Added 2026-08-06 after two US-men's expansion passes (72 candidate subs
probed between them). Two rules, learned the expensive way:

1. **Never trust a flair name.** `WDYWT`, `Fit`, `On Feet` and `Fit Check`
   mean "photograph of my shoes, cropped at the shin" on sneaker and brand
   subs, and "photograph of a whole person" on outfit subs. The only way to
   tell is to download four real images per candidate flair and look at
   them. That check rejected r/Sneakers, r/Vans, r/newbalance, r/WDYWT and
   r/Converse, and it is also what found the one counterexample worth
   having, r/SneakerFits, whose identical flair names *are* full-body.
   Brand subs for catalog brands (r/Carhartt, r/uniqlo, r/Nike, r/Dickies,
   r/thenorthface, r/RalphLauren...) are product-photo subs, uniformly.
2. **A 4-image sample decides whether to TRY a sub; only an audit of the
   shard's own output decides whether to KEEP it.** r/gorpcore sampled 2/4
   worn and delivered 8/8 product (gear macros, unboxings, flat-lays);
   r/CarharttWIP's `📸 Showcase` was guessed to mean "worn" and delivered
   7/8 hauls, hangers and Grailed screenshots. Both rows are commented out
   in `reddit_outfit_scraper.py` with the evidence. The audit is cheap: a
   contact sheet of 8 random images sampled from what the shard actually
   wrote into `outfit_dataset/`.

Both passes' verdicts -- kept subs with their exact flair gates, and every
rejected sub with the measurement behind the rejection -- live in the
`SUBREDDITS` table and the reject block below it, so no candidate gets
re-probed. Density is not quality: r/fitpics had the highest image-post
density ever measured here (267/300) and is women's boudoir selfies.

### Standing note: these are photos of real people

The brand scrapers collect corporate product photography. This target
collects individuals' own images. Using them as internal training/eval
data is one thing; **publicly displaying them in a product surface is a
different question with real licensing and consent implications**, and the
per-source license terms vary (Reddit's user agreement, Lookbook's terms,
Pexels/Unsplash's explicitly permissive licenses are all different). Worth
resolving deliberately before any of this is shown to end users. Flagged
here once so it's on the record and not rediscovered late.


## Non-clothing scraping (`negatives_dataset/`) — the negative half of a gate, added 2026-08-04

A third dataset, and the first one whose images are deliberately **not**
fashion. It exists because of the finding in `docs/eval_log.md`'s open-set
rejection row: pointed at something that is not in the catalog, the retrieval
pipeline still confidently names a product, and no threshold on the DINOv3
score fixes it (false-accept ~68% at any usable false-reject rate). The
conclusion there was that the fix has to sit **upstream** — refuse to query at
all unless the photo contains a garment.

That gate is a SigLIP2 zero-shot margin, and it had only ever been scored
against **synthetic** negatives (bar charts, solid colour fields, blocks of
text), which gives AUROC 1.0000. That number is true and useless: it measures
how easily SigLIP2 tells a photograph from a chart, when the deployed failure
mode is a photograph of a sofa — and a sofa is a photograph. Calibrating a
shipping threshold on it would be calibrating on the easy half of the problem.

```
negatives_dataset/{source}/{source_id}.jpg
negatives_dataset/metadata.json
```

Note the flat `{source_id}.jpg` layout rather than `outfit_dataset`'s
`{source_id}/image_N.jpg`: these sources return one image per result, not a
gallery per post, so a directory per image would be a directory per file.

### Hard rule: a negative containing prominent clothing is a mislabelled positive

Not noise — a **positive on the wrong side of the split**. A photo of a person
in a jacket filed as a negative pulls the threshold in exactly the direction
that makes the gate reject real clothes, which is the one failure the gate must
not have. Three defences, in order of how much they are trusted:

1. **Query wording.** Object and empty-scene phrasings throughout, and the two
   themes most likely to smuggle in a clothed person — `street` and `screens` —
   are worded toward the empty/technical variants *and* capped at half the
   per-theme quota.
2. **`review_negatives.py`.** Ranks every negative by a person/clothing-presence
   probe and prints a review queue. It never deletes on its own: the ranking
   model is SigLIP2, the same family as the gate being calibrated, so letting it
   choose which negatives survive would delete its own hard cases and inflate
   the very AUROC this exercise exists to deflate. A human passes ids to
   `--drop`.
3. **Looking at them.** The probe cannot see a shirt on a hanger with nobody
   wearing it, so the review also draws a random control block.

### Sources: both keyless, both crawl-friendly

No Pexels/Unsplash — those need an API key that does not exist in this repo,
and obtaining one was explicitly out of scope.

- **Wikimedia Commons** (`commons.wikimedia.org/w/api.php`). Queried with
  `generator=search` + `filetype:bitmap`, **not** `list=categorymembers`:
  Commons categories are curated for topic, not for "is this a usable
  photograph," and `Category:Chairs` genuinely returns `.ogg` pronunciation
  files. `iiurlwidth` asks Commons to pre-scale to the 1536px storage cap, so a
  40 MB original is never transferred to produce a small JPEG.
- **Openverse** (`api.openverse.org`). Widens provider diversity beyond
  Commons' house style (Flickr, museums). It serves **full-size originals**,
  which is the single biggest cost in a run — multi-MB files, occasionally
  tens of MB, and a theme that falls through to Openverse takes ~5-8 minutes
  against Commons' ~1. Budget for it; it is not a hang.

### Conventions shared with the outfit scrapers

Dedup (`imagehash.phash`, Hamming ≤ 6), the 15 KB / 320px size floors, the
1536px long-side cap and the candidate-URL ladder all come from
`outfit_scrape_common.py` rather than being reimplemented, so negatives obey
the same rules as everything else collected here. Provenance per record is
source, source_id, page_url, licence, title, theme, the query used, image_url,
local path, phash and scraped_at.

### Two writers will destroy `metadata.json` — now enforced

Learned by doing it: the collection was accidentally started twice and the two
processes **silently erased 127 already-downloaded images** from each other.
Same mechanism as the `apparel_dataset` incident recorded above — each process
holds the whole record list in memory and rewrites the file wholesale, so
whichever saves last wins. The files stayed on disk as orphans, invisible to
every consumer and un-re-addable, because the scraper skips destinations that
already exist. Nothing crashed and nothing warned; the only symptom was the
count sitting still. `negatives_scraper.py` now refuses to start if another
copy is running (pid file, with stale-pid takeover).
