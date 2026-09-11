# Licensing decision memo — what the corpus actually contains

**Written 2026-09-11.** Companion to `docs/licensing_review.md` (2026-08-04,
resolution note 2026-08-06). That document was an audit. This one is the
decision input: measured composition, the cost of each shipping option, and
an explicit list of what a lawyer has to answer.

**I am not a lawyer and none of this is legal advice.** Everything below is
a count against the data plus the engineering consequence of each count.
Section 6 marks where the questions stop being engineering questions.

Every number here was computed against `outfit_dataset/metadata.json`
(14,684 posts) and `negatives_dataset/metadata.json` (507 records) on
2026-09-11, not carried over from the earlier review.

---

## 1. The one number

**0 of 20,532 photos in the outfit corpus carry any licence information.**

Not "0 carry a permissive licence". Zero carry a `license` field, a
`licence` field, a `rights` field, or a `copyright` field, at the top level
or nested one level deep, in any of the 14,684 records. The key does not
exist in the schema.

So a filter that keeps only records safe for a public commercial surface —
on the basis of a recorded licence — keeps **0 photos, 0.00% of 20,532**.

That is the honest answer to question 3 of this memo's brief. There is no
NC/ND subset to exclude in the outfit corpus, because there is no licensed
subset at all. The NC/ND discussion in `licensing_review.md` is about
`negatives_dataset`, which is a set of furniture, food and landscape photos
used to calibrate a rejection threshold. **It contributes zero photos to the
product surface**, so the review's "public product surface → NC/ND subset
must be excluded" row is answering a question the outfit-search app does not
ask. The row that matters is the other one: `outfit_dataset` → "not without
work". This memo puts a number on "work".

---

## 2. Actual composition

### 2.1 By source

| source | posts | % posts | photos | % photos |
|---|---:|---:|---:|---:|
| Reddit | 6,720 | 45.8% | 12,283 | 59.8% |
| Pinterest | 6,597 | 44.9% | 6,882 | 33.5% |
| wear.jp | 1,367 | 9.3% | 1,367 | 6.7% |
| **total** | **14,684** | | **20,532** | |

There is no fourth source. Reddit spans 27 subreddits (largest:
r/mensfashion 573, r/mensfashionadvice 527, r/femalefashion 502,
r/femalefashionadvice 470, r/streetwear 460). Pinterest spans 179 search
queries. wear.jp is 1,367 posts from a single section, `/men-coordinate/`.

Image bytes came from `preview.redd.it` (8,358), `i.pinimg.com` (6,882),
`i.redd.it` (3,677), `images.wear2.jp` (1,367), `external-preview.redd.it`
(238), and 10 stragglers on imgur/supabase/one retail site.

### 2.2 Provenance fields that exist, by source

| field | Reddit | Pinterest | wear.jp |
|---|---:|---:|---:|
| `source`, `source_id`, `post_url` | 100% | 100% | 100% |
| `author` | 6,720 (100%) | 6,590 (99.9%) | 1,367 (100%) |
| `title` | 100% | 6,414 (97.2%) | 100% |
| `image_urls` | 100% | 100% | 100% |
| `source_link` (outbound) | 0 | 920 (13.9%) | 0 |
| **any licence / rights field** | **0** | **0** | **0** |
| `display_ok` / `commercial_ok` flag | 0 | 0 | 0 |

### 2.3 Traceability

- **Untraceable (no author): 7 posts / 7 photos.** All Pinterest. The
  2026-08-06 backfill claim in `licensing_review.md` holds: 1,342 → 7.
- **Traceable (author *and* post_url both present): 14,677 posts / 20,525
  photos — 99.97%.**

Traceable is not permitted. It answers "who posted this", so a takedown can
be actioned. It does not answer "who owns the photograph", and for Pinterest
it usually cannot, because a pin is typically a repin of someone else's
image.

---

## 3. Claims in `licensing_review.md` checked against the data

### Confirmed

| claim | verdict |
|---|---|
| `negatives_dataset` = 507 records, **all 507 carry a licence** | **true.** 507/507, field name `licence`. |
| 81 non-commercial | **true.** CC by-nc-nd 2.0 (39) + CC by-nc-sa 2.0 (32) + CC by-nc 2.0 (10) = 81. |
| 44 no-derivatives | **true.** CC by-nc-nd 2.0 (39) + CC by-nd 2.0 (5) = 44. |
| 2 GPL | **true.** |
| Pinterest backfill closed the author gap | **true.** 7 remain of 6,597. |
| `outfit_dataset` carries no licence | **true, and now at 2.1x the scale**: 0 of 14,684 posts, up from 0 of 6,860. |

Full `negatives_dataset` licence distribution (507): CC BY 2.0 (124 + 17 as
"CC BY 2.0" = 141), CC BY-SA 4.0 (115), CC BY-NC-ND 2.0 (39), Public domain
(35), CC0 (34 + 8 = 42), CC BY-NC-SA 2.0 (32), CC BY-SA 3.0 (22), CC BY-SA
2.0 (21 + 19 = 40), CC BY 4.0 (13), CC BY-NC 2.0 (10), CC BY-ND 2.0 (5), "No
restrictions" (3), CC BY-SA 2.5 (3), GPL (2), and 5 singletons (PDM-owner,
CC BY 2.0 fr, CC BY 2.5, CC BY 3.0, CC BY 1.0).

Fully unencumbered (CC0 / public domain / PDM / "no restrictions"): **81 of
507**. Commercially usable but with obligations (CC-BY and CC-BY-SA
families): **338** (158 CC-BY, 180 CC-BY-SA). Share-alike obligations
attach to the 180.

### Not supported by the data

1. **The review's summary table says `negatives_dataset` has "author
   recorded: yes". It does not. 0 of 507 records carry an author, creator or
   attribution field** — the key is absent from the schema entirely (the
   full key set is `source`, `source_id`, `page_url`, `licence`, `title`,
   `theme`, `query`, `image_url`, `path`, `phash`, `scraped_at`).

   This matters more than it looks. The single obligation of CC-BY and
   CC-BY-SA is attribution, and **338 of 507 records carry one of those
   licences while none of them records who to attribute.** Satisfying the
   licence would require re-fetching all 338 `page_url`s. The set the review
   calls the well-licensed one is, as stored, not compliant-ready.

2. **The 2026-08-06 note says `source_link` "is the only thing on the page
   that points at the original host".** In practice it mostly does not.
   Of the 920 Pinterest records that now carry one, **355 (38.6%) point at
   an ad or affiliate redirector, not a host**: `cat.us5.us.criteo.com`
   (202), `ad.doubleclick.net` (97), `go.skimresources.com` (41), `amzn.to`
   (12), `link.amazon` (3).

   That leaves **565 of 6,597 Pinterest posts — 8.6% — with an outbound link
   that even plausibly names an original host**, and the top entries there
   are retailers and unrelated sites whose relationship to the photograph is
   unverified: `thewoobles.com` (66, a crochet-kit shop), `usbank.com` (63),
   `levi.com` (33), `instagram.com` (26), `pacsun.com` (18),
   `booking.com` (7). Treat `source_link` as a weak hint, not as chain of
   title.

### Recommendations from the review: implementation status

| # | recommendation | status |
|---|---|---|
| 1 | record a `license` field on every outfit record, even `"unknown"` | **not done.** 0 of 14,684. |
| 2 | mark Pinterest records `display_ok: false` | **not done.** No such key. |
| 3 | add `commercial_ok` to `negatives_dataset` | **not done.** No such key. |
| 4 | decide the outfit-photo question before any public surface | this memo |
| 5 | do not let the index leak images — keep URLs as references, not copies | **reversed by deployment.** See below. |

### The material change since the review: the product now serves copies

`licensing_review.md` closed on "URLs are references, not copies, which is
the right side of the line — keep it that way rather than caching thumbnails
for convenience."

The deployed service does the opposite. `modal_app_outfit_serve.py` exposes
`GET /photo?path=...`, which reads the image off the Modal volume and
returns the bytes; `outfit_search.py` sets `hit["image_url"] =
"/photo?path=..."` and the local UI renders `<img src="/photo?...">`. All
20,532 photos were uploaded to the Modal volume (commit dfc5866, "20,545
images on the volume, 0 missing"). The iOS app consumes that API.

So the current architecture **copies, hosts and redistributes** every photo.
That is not a criticism of the build — it is the only way to make search
fast — but it means the review's most reassuring sentence no longer
describes the system, and option (a) below is a rollback of a shipped
design, not a design choice still open for free.

The good news for option (a): the result objects already carry what
attribution needs. `outfit_search.py` emits `post_url` and `author` on every
hit.

---

## 4. What the risk buckets actually are

Restating the three risks of the original review against current data, plus
one the review does not cover.

**Risk 1 — untraceable records. Effectively closed.** 7 photos of 20,532.
Deleting those 7 costs nothing measurable.

**Risk 2 — no permission, corpus-wide. Open and unchanged, at 2.1x scale.**
20,532 photos, 0 with a licence. Reddit and wear.jp grant rights to Reddit
and to wear.jp; neither sublicenses to us. Pinterest was collected against
`robots.txt` (`User-agent: * / Disallow: /`) through an authenticated
session, a decision the owner made explicitly on 2026-08-03 and which is
fine as a decision but is a ToS matter independent of copyright. wear.jp is
the one source collected robots-clean, with `--respect-robots` on.

**Risk 3 — `negatives_dataset` licence mix. Real but irrelevant to
shipping,** because those images are never displayed and never leave the
threshold calibration. It becomes live only if they appear in a figure, a
demo or a claimed-commercial artifact. The attribution gap in 3.1 above is
the part to fix before that happens.

**Risk 4 — not covered by the review at all: these are identifiable
people, not just copyrighted files.** Copyright is the owner's-permission
question. Separate from it:

- Every photo depicts an identifiable person who did not consent to appear
  in a commercial product. Right of publicity / personality rights are a
  distinct body of law from copyright and are not cured by a licence from a
  photographer.
- Under GDPR-style regimes a photograph of an identifiable person is
  personal data. No lawful basis is recorded, there is no privacy notice,
  and there is no takedown or erasure mechanism in the product.
- `skin_tone` is measured and stored for 10,640 of 14,684 posts and is a
  live ranking signal. Deriving and storing an inferred physical
  characteristic of identifiable individuals sits close to special-category
  processing. This is the item I would put in front of a lawyer first, and
  it is not mentioned anywhere in the existing review.

I am flagging these as questions, not asserting the law. They change the
shape of the decision because **a licensed corpus fixes risk 2 and does not
automatically fix risk 4** — a Pexels photo is licensed for commercial use
by the photographer, which is not the same as the depicted model having
released their likeness for this use (Pexels' own licence text is explicit
that model releases are the user's problem).

---

## 5. The options, with what each costs

Ranked by how much product survives.

### (a) Link-out / thumbnail-and-attribute instead of full display

**Product:** all 20,532 photos stay in the corpus and stay searchable.
Results render as a thumbnail with author and a tap-through to `post_url`
(present on 100% of records; `author` on 99.95%). Ranking, eval, and the
46.0% p@15 number are untouched.

**Engineering:** moderate, and a partial rollback. `post_url` and `author`
are already in the hit payload. What changes is `/photo` (serve a small
derivative, or stop serving bytes at all) and the iOS cell (add attribution,
add the tap-through). Hotlinking the original CDN instead of serving copies
is **not** a clean substitute: 8,596 of the URLs are `preview.redd.it` /
`external-preview.redd.it`, which are signature-bearing and expire, a
further 3,677 are `i.redd.it`, and `i.pinimg.com` is hotlink-hostile. Practically this means "serve a small
thumbnail we host, link out for the full image".

**Risk:** reduced, not resolved. A hosted thumbnail is still a copy. The
argument that a thumbnail plus a link to the source is defensible is a real
argument with real case law behind it in the US, and it is exactly the
argument that needs a lawyer rather than an engineer — it is jurisdiction-
specific, and the App Store is global. It does nothing for risk 4: a
thumbnail of an identifiable person is still a photograph of them.

**Verdict:** preserves the most product. Does not make the question go away;
it makes it a smaller and more arguable question.

### (b) Restrict the corpus to a licensed subset

**Product: 0 photos.** There is no licensed subset. This option does not
exist against this corpus.

The nearest weakened version — "keep the source that was collected without
violating anyone's robots.txt, and that has 100% author coverage" — is
wear.jp alone: **1,367 posts / 1,367 photos, 6.7% of the corpus.** All of it
is `/men-coordinate/`, i.e. Japanese men's outfits, one photo per post. Any
women's query, and any query whose answer lives in a Reddit fit pic, returns
nothing useful. The 46.0% p@15 figure does not transfer — it was measured on
a judged pool drawn from the full corpus, and the handoff already flags that
the absolute value is optimistic.

**Risk:** lower, and still not permission. wear.jp's ToS grants rights to
wear.jp. Robots-clean collection is a statement about the crawl, not about
display rights. Risk 4 is untouched.

**Verdict:** the honest label for this option is "a smaller pile of the same
problem". Worth doing only as a staged de-risking step, not as the answer.

### (c) Replace the corpus with licensed or synthetic imagery

**Pexels.** `docs/outfit_sourcing_plan.md` already costed this: free API
key, 200 req/hr and 20k req/month, ~6,000 fashion photos available, and the
Pexels License permits commercial use without attribution. It is the only
path in this repo that reaches a public surface with real outfit photos and
a licence behind them.

**Product:** roughly 6,000 photos, ~29% of the current corpus, displayed at
full fidelity with no attribution obligation. The sourcing plan's own caveat
stands: Pexels is stock-posed, not deployment-condition, so the corpus gets
cleaner and less real at the same time. Two things in the measured record
argue the size loss matters less than it looks:

- The handoff is explicit that **the encoder is the binding constraint**,
  not corpus size — mean pairwise similarity between random outfit photos is
  0.743, and "a photo of a bicycle" still scores 0.155. "More corpus buys
  recall; it cannot fix this."
- The system was functional and being evaluated at **6,860 posts / 9,971
  photos** (the 2026-08-04 backup), which is the same order as a Pexels-only
  corpus.

**Cost:** a real rebuild — re-scrape, re-detect (the current 43,712
detections do not transfer), re-embed, re-index, re-upload to Modal. Blocked
on budget: **$28.13 of a $30 Modal limit is spent, and the handoff states
there is not enough left for another full GPU pass.** Raise the limit first.

**Synthetic.** The machinery exists (`negatives_dataset/_synthetic`, 400
generated images) but this is a different product. The stated product is
"real outfit photos of real people"; generated outfits are the thing the
owner explicitly did not ask for, and generated-image provenance is its own
unresolved question.

**Risk:** lowest of the shipping options for risk 2, provided the scraped
corpus is removed from the Modal volume rather than merely unlinked. Risk 4
is *reduced but not eliminated* — Pexels licences the photograph, not the
model's likeness for commercial use.

**Verdict:** the only option that plausibly ends with a shippable public app
and a defensible answer to "where did these photos come from".

### (d) Do not ship publicly

**Product:** zero external value. The corpus, the index, the eval and the
iOS app all keep working for internal use, which the review already rates
acceptable.

**Risk:** near zero for risk 2, with one caveat worth checking: App Store
review and TestFlight both involve a third party viewing the content, so
"TestFlight to a few friends" is not obviously the same as "private
development storage". The handoff's own phrasing is the right one — "private
development storage is not publication; a public link would be."

**Verdict:** the default if nothing else is decided. It is also what is true
today.

### (e) Worth naming: ship the catalog surface instead

`apparel_dataset` holds **4,079 products / 20,205 images** of corporate
product photography across 20 brands. `licensing_review.md` calls this "the
safe default", and it ships without any of risk 4 — these are brand
marketing assets of models under contract, not strangers' fit pics.

**Product:** this is the *identifier* ("what is this item", "shop this
look"), which the 2026-08-05 correction demoted from the product to a
feature. It ships a real app; it does not ship this app. It is also not
risk-free — brand product imagery is copyrighted too, just copyrighted by
entities with a commercial interest in it circulating, which is a different
and better-understood risk shape than a stranger's Reddit photo.

**Verdict:** the fallback that gets something into the App Store this
quarter while option (c) is rebuilt.

### Summary

| option | photos on the public surface | does outfit search still work | risk 2 (permission) | risk 4 (likeness/privacy) |
|---|---:|---|---|---|
| (a) link-out + thumbnail | 20,532 | yes, full quality | reduced, unresolved | unchanged |
| (c) Pexels rebuild | ~6,000 | yes, likely near parity — encoder is the binding constraint | largely resolved | reduced, not resolved |
| (b) wear.jp only | 1,367 | degraded; men's Japanese coordinates only | reduced, unresolved | unchanged |
| (e) catalog only | 20,205 | no — different product | different shape, better understood | largely resolved |
| (d) don't ship | 0 | n/a | resolved | resolved |

---

## 6. What a lawyer has to answer

Engineering can produce the counts above. It cannot answer any of these, and
each one changes which option is viable:

1. Is a self-hosted thumbnail plus attribution plus a link to the source
   post defensible for a paid or ad-supported iOS app, in the US and in the
   EU? Option (a) lives or dies on this.
2. Does an App Store listing whose entire content surface is third-party
   photographs collected without permission survive review, and what does
   Apple's IP section actually require of the developer? I have not verified
   the current guideline text; someone must.
3. Pinterest was collected against an explicit `robots.txt` deny through an
   authenticated session. What is the exposure from that as a collection
   act, separately from display — including any contract/CFAA-style theory?
4. Right of publicity: does showing an identifiable stranger's photo in a
   commercial fashion-search app create liability independent of copyright,
   and does a Pexels licence cure it?
5. GDPR/UK-GDPR: are these photographs personal data in this product, what
   is the lawful basis, and does storing an inferred `skin_tone` for 10,640
   posts and using it as a ranking signal implicate Article 9?
6. If the answer to any of the above is "you need a takedown process", what
   does a compliant one look like at 20,532 photos with one operator?

---

## 7. What I would do regardless of which option is chosen

These are cheap, they are reversible, and they are all things the earlier
review recommended and that never got done. None of them is a decision.

1. **Write `licence: "none-recorded"` onto all 14,684 records.** An explicit
   "we looked and there is none" is materially different from a missing key,
   and it stops this from being re-derived a third time.
2. **Delete the 7 authorless Pinterest records.** 7 photos of 20,532. There
   is no argument for keeping them.
3. **Backfill an attribution field onto the 338 CC-BY/CC-BY-SA
   `negatives_dataset` records** from their `page_url`, or stop describing
   that set as the well-licensed one. Right now the licence is recorded and
   the one thing it obliges is not.
4. **Put the decision in the data, not in a doc.** A `display_ok` boolean
   that the serving path actually filters on, so that whichever option is
   chosen is enforced by code rather than remembered.
5. **Do not widen distribution in the meantime.** The `/photo` endpoint
   serving full bytes off the Modal volume is the highest-exposure surface
   in the system, and it is already live behind a bearer token. Keep the
   token.

---

## 8. Bottom line

The corpus is 20,532 photographs of identifiable real people, collected from
three platforms, 99.97% traceable and 0% permitted. There is no licensed
subset to fall back to — the filter everyone assumes exists returns zero
rows. The earlier review's reassurance that the system keeps references
rather than copies is no longer true of the deployed service, which hosts
and serves every image.

That leaves a genuine choice rather than a technicality: degrade the
presentation and argue thumbnails (option a), or rebuild on ~6,000 Pexels
photos and lose a third of the corpus but gain an answer (option c). The
measured record says option (c) costs less product than it appears to,
because the encoder — not corpus size — is what is actually limiting
quality. Option (c) is blocked on the Modal budget, which is the one
concrete prerequisite either way.

The decision is the owner's. This memo is the counts.
