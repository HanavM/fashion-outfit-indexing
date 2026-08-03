# Real-outfit photo sourcing — where to get them and how

Written 2026-08-03. Companion to SCRAPING_PROCESS.md's "Real-outfit
scraping (`outfit_dataset/`)" section, which defines the *convention*
(schema, dedup, hard separation from `apparel_dataset/`). This document
decides the *source*.

Every number below was measured against the live APIs on 2026-08-03, not
estimated from memory. The probe scripts are throwaway; the numbers are
reproducible from the method notes inline.

## Bottom line

**Primary: Reddit, but flair-filtered instead of score-filtered.** The
current scraper's quality filter is wrong, and fixing it is a ~20-line
change that raises both precision and yield substantially.

**Pinterest is the best-curated source and the least usable one.** Its
`robots.txt` ends in `User-agent: *` / `Disallow: /`, and the official API
reads only your own account. It cannot be scraped without breaking this
project's own rule 4. Two narrow legitimate uses survive — see below.

**Add Pexels as a second, license-clean tier** — it is the only source
here that can be shown in a product surface without a licensing question.

**Consider DeepFashion2 separately** — it is the only option that comes
with ground-truth consumer↔shop pairs, which is the one thing scraping
can never provide.

## Pinterest — assessed, and why it doesn't work as a scrape target

The intuition behind picking Pinterest is correct and worth stating
plainly: **Pinterest content is human-curated, so its precision for
"a person wearing a complete outfit" is far higher than a raw social
feed.** That is a real advantage and the reason it was proposed. The
problem is access, not quality.

What was measured on 2026-08-03:

1. **`robots.txt` is an explicit allowlist, and we are not on it.**
   `https://www.pinterest.com/robots.txt` names ~300 specific crawlers
   (Googlebot, bingbot, Applebot, Yandex, …) and grants them
   `Allow: /resource/*/get/` plus scoped `Disallow`s. It then closes at
   line 997 with:

   ```
   User-agent: *
   Disallow: /
   ```

   Everything not on the allowlist is disallowed from the entire site.
   SCRAPING_PROCESS.md's outfit-scraping rule 4 — "Respect robots.txt and
   rate limits… same discipline as the brand scrapers, and more important
   here" — is a project rule that this would directly violate. That rule
   was written for this exact dataset.

2. **The undocumented JSON endpoint is closed anyway.** The internal
   `/resource/BaseSearchResource/get/` route that scraper tutorials use
   returns **HTTP 403 `Invalid Resource Request`** unauthenticated. The
   public search HTML (`/search/pins/?q=…`) does return 200, but it is a
   1.1 MB React shell; pin data arrives via the same gated resource calls.

3. **The official API v5 cannot do this job.** It requires a Business
   account and app review, and — decisively — it exposes *your own*
   account's pins and boards, not public search or other users' boards.
   Pinterest's developer guidelines additionally restrict storing API
   data beyond campaign analytics. There is no compliant endpoint that
   returns "500 curated streetwear pins."

4. **Copyright sits with third parties, not Pinterest.** Most pins are
   *repins* of images hosted elsewhere — blogs, brand lookbooks,
   Instagram. Reddit's WDYWT posts are, by the subreddit's own rules,
   the poster's own photo of themselves. So Pinterest is *worse* on
   provenance than Reddit, not better, despite looking cleaner.

### The two Pinterest uses that are legitimate

- **Your own curated boards, via the official API.** If a human saves
  pins into a board on an account you control, API v5 can read that
  board. This genuinely captures the curation quality that motivated the
  proposal. Cost: it is manual, so it scales to hundreds of images, not
  tens of thousands. Reasonable for building a small, very high-quality
  **eval** set; not viable for bulk.
- **Pinterest as a discovery index, not an image source.** Browse it
  manually to identify *which* blogs, creators, and lookbook sites the
  good pins point back to, then source from those sites directly under
  their own terms. This converts Pinterest's curation into a target list
  without scraping Pinterest at all.

## Reddit — the volume and noise concerns, measured

Two concerns were raised: not enough volume, and too much noise. The
first is not borne out; the second is real but has a clean fix.

### Volume: ~37,000 outfit posts, and that is a floor

Sampled two independent 14-day windows (2025-06 and 2024-09) per
subreddit, counted image posts carrying an outfit-indicating flair, and
scaled to the 2023-01-01 → 2026-03-01 window the scraper already targets:

| subreddit | img posts/14d | outfit-flaired/14d | est. posts (3.2 yr) |
|---|---:|---:|---:|
| r/mensfashion | 72 | 72 | 5,981 |
| r/fashion | 72 | 72 | 5,899 |
| r/streetwearfits | 66 | 66 | 5,404 |
| r/femalefashion | 58 | 58 | 4,785 |
| r/rawdenim | 46 | 46 | 3,836 |
| r/streetwear | 58 | 43 | 3,548 |
| r/OUTFITS | 70 | 42 | 3,424 |
| r/femalefashionadvice | 16 | 16 | 1,361 |
| r/techwearclothing | 12 | 12 | 1,031 |
| r/malefashion | 26 | 10 | 866 |
| r/japanesestreetwear | 63 | 10 | 784 |
| **total** | | | **~36,900** |

At ~2.4 images/post that is **roughly 60–90k images**. These are floors:
the API caps a page at `limit=100`, so any sub near 100 image-posts per
window is truncated. Against a realistic Phase 3 need of a few thousand
images, Reddit is over-supplied by more than an order of magnitude.
Volume is not the binding constraint.

### Noise: flair fixes it; score never could

The current scraper filters on `score`. That is the wrong signal, for a
reason specific to the archive: **Arctic Shift snapshots a post shortly
after creation, so archived `score` is near-zero regardless of how the
post actually did.** `SCORE_MATURITY_DATE = 2026-03-01` was added to work
around this, but it doesn't: posts 5 months old still show a median score
of 1 across all posts, and 2.5-year-old r/streetwear posts still show
median 1. Score is filtering by *when the archive happened to snapshot*,
not by quality.

One correction to an earlier read of this data: score is not *uniformly*
dead. Restricted to posts that actually have images, r/streetwear's
median score is **34** (68% score >1), while r/malefashion's is **1**
(26% >1). So score is weakly usable on the big subs and useless on small
ones — which makes it a bad global filter, not a worthless one.

**Flair is the signal that actually works.** Measured over 100-post
samples of image posts:

| subreddit | flair distribution (image posts) |
|---|---|
| r/streetwear | **WDYWT 45**, DISCUSSION 6, NEWS 5, INSPO 1 |
| r/malefashion | (none) 18, **WIWT 18**, Inspo 3, Discussion 2 |
| r/OUTFITS | **Outfit of the day 27**, **Felt cute 14**, **My Work Fit 2**, Advice 39 |
| r/japanesestreetwear | DISCUSSION 25, **PICK-UP 12**, AD 2, WDYWT 2 |
| r/techwearclothing | (none) 33, WDYWT 1 |

`WDYWT` ("what did you wear today") and `WIWT` are, by definition, one
person wearing one outfit — exactly the target. On r/streetwear that is
**45 of 57 image posts (79% precision)** before any model-side filtering.

This also exposes a source that should be *dropped or restricted*:
r/japanesestreetwear is mostly `DISCUSSION` and `PICK-UP` (product hauls
— flat-lay photos of purchases, not worn outfits). Scraping it unfiltered
pollutes the set with exactly the studio-ish product shots this dataset
exists to contrast against. Same for r/techwearclothing, where flair is
mostly absent and unusable.

### A third finding: the `removed` filter is over-aggressive

`is_candidate()` drops any post with `removed_by_category` set. That
discards **53–97%** of posts per sub. But most such removals are
`automod_filtered` or `moderator` — subreddit rule violations (missing ID
comment, wrong day thread), not bad images — and the images are usually
still live: **6 of 8 sampled removed posts fetched HTTP 200**. For
comparison, non-removed posts fetched 4 of 5, so baseline image
availability is ~75–80% either way.

Recovering removed-but-live posts roughly **doubles to quadruples** yield
on the subs where removals dominate. The one category to keep excluding
is `deleted` (user-deleted — images usually gone).

## Recommended plan

**Tier 1 — Reddit, flair-gated (bulk, do this first).** Changes to
`reddit_outfit_scraper.py`:

1. Replace the per-sub `min_score` with a per-sub **flair allowlist**;
   keep score only as a weak tiebreak on subs where it's live
   (r/streetwear, r/OUTFITS), not as a gate.
2. Change the removal filter from "drop if `removed_by_category`" to
   "drop only if `removed_by_category == 'deleted'`", and let the
   existing image-fetch failure path absorb the dead ones.
3. Drop r/japanesestreetwear and r/techwearclothing from the default list,
   or restrict them to `WDYWT` only.
4. Delete `SCORE_MATURITY_DATE` and scrape the full window — it exists
   only to protect a score filter that is being removed. This alone
   re-opens 5 months of the most recent posts.

Expected: a few thousand high-precision images in well under an hour of
wall-clock, against a 60–90k-image corpus.

**Tier 2 — Pexels, for anything user-facing.** Free API key, 200 req/hr
and 20k req/month, and the Pexels License permits commercial use without
attribution. This is the only tier that sidesteps SCRAPING_PROCESS.md's
standing note about publicly displaying photos of real people. Volume is
modest (~6k fashion photos) and the look is stock-posed rather than true
deployment condition — so it is a *complement* for demos and product
surfaces, not a replacement for Tier 1's realism.

**Tier 3 — Pinterest, manual + official API, for a small gold eval set.**
Curate a board by hand, read it through API v5. Hundreds of images,
highest precision available. Do this only if Tier 1's precision proves
insufficient in practice.

**Separate track — DeepFashion2, for ground truth.** 491k images and
873k commercial-consumer pairs, with `source: shop|user` and a shared
`pair_id`. This is the only option that makes Phase 3 *measurable*: today
the plan is to label outfit photos unsupervised with no ground truth
(SCRAPING_PROCESS.md, "These images are deliberately UNLABELED"), which
means no R@K number for catalog-to-consumer is possible. DeepFashion2
would give one.

Two caveats before committing: access requires filling a form and signing
an agreement, and **the DeepFashion family prohibits commercial use** —
it can validate the pipeline internally but cannot ship in a product. Its
shop images are also a different catalog (Mogujie), so it validates the
*method*, not retrieval against this project's Nike/Vans/Dickies gallery.

## What was rejected and why

- **Reddit's own API/JSON endpoints** — hard-403 from this machine on
  every documented route, including `old.reddit.com` and public
  redlib/libreddit mirrors. Already documented in the scraper's header.
  Arctic Shift remains the workaround.
- **Server-side score filtering on Arctic Shift** — confirmed unsupported:
  `min_score` returns `Unknown query parameter`, and `sort_type` accepts
  only `default` or `created_utc`. All quality filtering must be
  client-side, which is fine given the corpus depth.
- **Customer-review photos on brand product pages** — previously
  considered and rejected (SCRAPING_PROCESS.md); they'd give free product
  linkage but aren't genuine worn-outfit imagery.
