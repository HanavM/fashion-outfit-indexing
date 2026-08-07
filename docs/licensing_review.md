# Licensing and provenance review — 2026-08-04

> ## RESOLVED 2026-08-06: risk 1 (untraceable Pinterest records) is closed
>
> **All 2,553 Pinterest records now carry an `author`. Zero remain
> untraceable.** The 1,342 identified below — and the ~1,200 added since —
> were backfilled at a **100% recovery rate** on every pin that loaded.
> `backfill_pin_authors.py`, ~3.5 s/pin, 87 minutes total.
>
> **Why the original scrape could not have done this**, verified rather
> than assumed: the search-grid DOM exposes only href/alt/image on a pin
> card — no profile link, no domain badge — and the internal
> `/resource/PinResource/get/` JSON route 403s even with a logged-in
> session and a csrftoken header. The pinner is only on the **pin page**,
> inside the `__PWS_INITIAL_PROPS__` blob, which is why this had to be a
> second pass rather than an inline capture.
>
> **What this does and does not fix.** It answers "whose pin is this",
> so a takedown request can now be traced and actioned. It does **not**
> establish copyright: this document's own point stands that Pinterest
> content is mostly repins of third-party images, so the pinner is often
> not the holder. Partly for that reason the backfill also records the
> pin's outbound destination as `source_link` where present (319 records)
> — that link is the only thing on the page that points at the original
> host.
>
> Risks 2 and 3 below are unchanged: `outfit_dataset` still carries
> **provenance but no permission**, and the negatives set is still
> licence-mixed. The judgement calls at the end remain the owner's.

---


Gap analysis item 12.3. `SCRAPING_PROCESS.md` has carried a standing note
since the outfit dataset was designed: these are photos of real people,
collected under varying terms, and displaying them in a product is a
different question from training on them. This is that question, answered
against what the datasets actually contain rather than in the abstract.

**I am not a lawyer and this is not legal advice.** What follows is an
audit of what is and is not traceable, plus the engineering decisions that
follow from it. The judgement calls at the end are the owner's.

## What we actually hold

| dataset | records | licence recorded | author recorded |
|---|---:|---|---|
| `apparel_dataset` | 2,387 | n/a — corporate product photography | n/a |
| `outfit_dataset` | 6,860 | **0 of 6,860** | 5,518 of 6,860 |
| `negatives_dataset` | 507 | **507 of 507** | yes |

That contrast is the headline. The negatives set was collected two days
after the outfit set, from keyless CC sources, and carries a licence
string on every record. The outfit set carries none at all.

## Three distinct risks, most to least severe

### 1. Pinterest: 1,342 records with no author — not traceable

Pin permalinks are `/pin/<id>` with no handle, so `author` is empty on all
1,342. Every record has a `post_url`, so the pin is reachable, but the
**person who owns the photograph is not identified in our data**.

This matters concretely: if a takedown request arrives, we cannot answer
"whose photo is this" without re-scraping, and pins are frequently deleted
or re-hosted. It is also the source whose collection most clearly ran
against the platform's terms — Pinterest's `robots.txt` is
`User-agent: * / Disallow: /`, and browsing was driven through an
authenticated session (see `docs/outfit_sourcing_plan.md`, where this was
raised and the owner decided to proceed).

Pinterest content is additionally mostly **repins of third-party images**,
so the pinner is often not the copyright holder either. Even a complete
author field would not establish who to ask.

### 2. No licence recorded anywhere in `outfit_dataset`

Reddit and wear.jp records carry `author` and `post_url`, which is enough
to trace and to delete. Neither carries a licence, because neither
platform attaches one per post — the terms are the site's user agreement,
which grants the platform rights, not us.

So for all 6,860 outfit records the honest statement is: **we have
provenance but no permission.** Fine for internal training and evaluation
under most readings of research use; not a basis for public display.

### 3. `negatives_dataset` is well-licensed but not uniformly permissive

All 507 carry a licence, and they are mostly CC-BY / CC-BY-SA / public
domain. But:

- **81 are non-commercial (NC)** — CC BY-NC-ND 2.0, CC BY-NC-SA 2.0,
  CC BY-NC 2.0.
- **44 are no-derivatives (ND)**.
- 2 are **GPL**, which is a software licence applied to an image and
  means whatever the uploader intended, which is unclear.

The NC ones cannot be used in a commercial product. The ND ones are the
subtler problem: cropping and resizing to 1536px are arguably derivative
works, and this pipeline does both. CC-BY-SA additionally carries
share-alike obligations.

**None of this affects the current use.** The negatives set exists to
calibrate a threshold; the images are never shown, redistributed, or
shipped — only a scalar margin derived from them is. But that is a
property of *how it is used today*, not of the licence, and the moment
those images appear in a demo, a paper figure, or a shipped model artifact
that is claimed as commercial, the distinction becomes live.

## What follows, by use case

| use | outfit_dataset | negatives_dataset |
|---|---|---|
| internal training / eval | acceptable | acceptable |
| internal demo to the team | acceptable | acceptable |
| **public product surface** | **not without work** | **NC/ND subset must be excluded** |
| paper / blog figures | per-image permission | CC-BY attribution required |
| redistributing the dataset | no | share-alike obligations apply |

## Recommendations, cheapest first

1. **Record a `license` field on every `outfit_dataset` record now**, even
   if the value is `"platform-terms-only"` or `"unknown"`. An explicit
   "unknown" is very different from a missing field: it distinguishes
   *"we checked and there is none"* from *"nobody looked"*, and it stops
   this from being re-derived later. Cheap, and it is the thing that makes
   every later decision auditable.
2. **Mark the 1,342 Pinterest records as display-blocked.** A boolean
   `display_ok: false` in the record is enough for any surface to filter
   on, and it encodes the decision where the data lives rather than in
   someone's memory.
3. **Add a `commercial_ok` flag to `negatives_dataset`** derived from the
   licence string — 81 NC records fail it, 44 ND records fail derivative
   use. Currently the licence is recorded but nothing consumes it.
4. **Before any public surface**, decide the outfit-photo question
   deliberately. The realistic options are: show only catalog product
   photography (corporate, and the safe default); replace the outfit set
   with a permissively-licensed one (Pexels/Unsplash, which the sourcing
   plan already costed); or seek per-creator permission, which does not
   scale to 6,860.
5. **Do not let the co-occurrence index leak images.** It currently stores
   example `post_url`s as evidence. URLs are references, not copies, which
   is the right side of the line — keep it that way rather than caching
   thumbnails for convenience.

## The one thing that is genuinely blocking

Everything above is manageable except the Pinterest author gap, because it
is the only item that cannot be fixed after the fact. Licences can be
recorded retroactively; flags can be added; sources can be swapped. But if
a takedown arrives for a pin that has since been deleted, we cannot
establish who to talk to, and we did not record it at collection time when
it was still available.

If the outfit set is going to matter long-term, backfilling Pinterest
authorship — or dropping those 1,342 records — is the decision with a
closing window. Everything else can wait.
