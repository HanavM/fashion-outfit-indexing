# VALIDATION — fashion outfit search (fash-007)

Written 2026-10-15 by the cloud gardener, per GARDENER.md §0.5 and the owner's
INBOX note of 2026-10-15 ("Don't spare my feelings"). Competitors were **read**
(articles, vendor pages) on this date, not hands-on tried: this run has no
phone and no accounts. Repo numbers are quoted from `docs/SESSION_HANDOFF.md`,
`docs/licensing_decision_memo.md` and `docs/eval_log.md`, not re-measured.

## Verdict: **KILL** (as a product headed for the App Store)

Item photo + text → real photos of people wearing outfits with that item is
not new. **Pinterest shipped exactly this query in 2017**, on a corpus it has
the rights to show, and Google, ChatGPT and the closet apps cover the
neighbouring jobs. The one thing this repo has that a closet app doesn't,
*real strangers' fit pics*, is the thing it legally can't ship: 0 of
20,532 photos carry a licence. The licensed replacement (~6,000 Pexels
photos) is stock-posed, which removes the differentiator. Quality is also
far off the SEED target: 46% p@15 on an optimistic judged pool against a
75% "both clauses in the top 4" goal, with the encoder named as the limit.

Stop building toward SEED.md's definition of done. What would change the
verdict is at the bottom.

## 1. Novelty — what already exists

| product | what it does for "this item + my words → outfits" | source |
|---|---|---|
| **Pinterest Lens "Lens Your Look"** (Nov 2017) | Point the camera at a clothing item, **add a text search like "outfits for the beach"**, and it searches fashion pins for *outfits where that item is a small part*. That is this repo's query, word for word. | [TechCrunch 2017](https://techcrunch.com/?p=1567091) |
| Pinterest "Styled for you" (Oct 2025, US/CA test) | AI collages built from your saved fashion pins; tap an item to swipe through recommended pins to mix and match. | [TechCrunch 2025-10-27](https://techcrunch.com/2025/10/27/pinterest-experiments-with-new-ai-powered-personalized-boards/) |
| Pinterest "Complete the Look" | Visual-search recommendations that complete an outfit (headline only; article not read this run). | [Adweek](https://www.adweek.com/?p=1057603) |
| Google Lens **multisearch** (2022) | Photo or screenshot + "Add to your search" text; Google pitches it for "shopping queries, particularly if you're looking for something that fits in with a particular style". | [Tom's Guide](https://tomsguide.com/news/google-adds-multisearch-tool-combining-text-and-images-heres-how-it-works) |
| Google **AI Mode** visual search (Sep 2025, US) | Upload a photo, refine conversationally ("more options with dark tones"), visual results each linking to their source. Not outfit-specific, but the photo+text-to-images loop exists at Google scale. | [Google blog](https://blog.google/products/search/search-ai-updates-september-2025/) |
| Google **Doppl** (Jun 2025) | Try an outfit on a photo of yourself (generated). Different answer to "how would this look". | [TechRadar](https://www.techradar.com/computing/artificial-intelligence/i-tried-googles-new-gemini-powered-clothing-app-heres-how-you-can-use-ai-to-find-the-perfect-outfit) |
| **ChatGPT** shopping (Oct 2026) | Try On button on shopping results (selfie + item), describe a look and have it shop the pieces, upload celebrity outfits to find buyable versions. | [TechCrunch 2026-10-01](https://techcrunch.com/2026/10/01/chatgpt-can-now-virtually-try-on-clothes-for-you/) |
| **Whering** | "style these shoppable items with your existing wardrobe" before checking out. The *closet* half of "how would this look with my jorts", using your actual jorts. | [Indyx comparison](https://myindyx.com/versus/whering-vs-indyx) (competitor-written; treat as biased) |
| **Indyx** | Lookbook from your own closet plus suggested additions; browse other users' Open Closets; weekly outfits from real stylists. | same |
| **Alta** | AI stylist anchored in your closet; generated look-alike avatar to try "outfits or shopping items before committing". | [Alta blog](https://www.altadaily.com/blog/menswear) |
| **Acloset** | Closet app, outfit suggestions by calendar/weather/mood; claims millions of users. No item-styling feature found. | [startuphub](https://www.startuphub.ai/startups/acloset.md) |
| **Lykdat** | B2B API: visual search, Shop The Look, recommendations for Shopify/BigCommerce retailers. Sells the retrieval layer this repo built. | [Capterra](https://www.capterra.com/p/10017613/Lykdat/) |
| Apple **Visual Intelligence** (iOS 26) | Screenshot anything → search it with Google or ask ChatGPT, system-wide (from search-result headlines; page not read this run). This is the Siri/iOS entry point the SEED wants, already wired to the incumbents. | [Macworld](https://www.macworld.com/article/2879052/how-to-use-visual-intelligence-to-analyze-any-screenshot-in-ios-26.html) |
| Open research: composed image retrieval (FashionIQ, CIRR) | Image + modifying text → images is a named research task with open code (CLIP4Cir, SEARLE). | [CLIP4Cir](https://github.com/ABaldrati/CLIP4CirDemo), [SEARLE](https://github.com/miccunifi/SEARLE) |

Not checked this run: TikTok/Instagram search (hashtag/text search over
real outfit videos; "jorts outfit" is a text query there, and the item
photo adds little for nameable items: see §2).

**Novelty: none at the product level.** The query exists (Pinterest, 2017),
the voice/OS entry point exists (Visual Intelligence → Google/ChatGPT), and
the closet-aware variant exists (Whering). The only uncovered detail is
"real strangers' fit pics, ranked by a model I trained", which is not
visible to a user.

## 2. Importance — who has this problem and what do they do today

- **The person in the README** (shopping, wondering how an item works with
  something they own) types "jorts outfit men" into Pinterest, TikTok or
  Google Images today and gets thousands of real outfit photos, free, in
  apps already on the phone. The photo half of the query only matters when
  the item **can't be named in a few words**. Jorts, white sneakers, a black
  leather jacket, baggy jeans, every example query in this repo's docs, can
  be named.
- When the photo does matter (odd print, specific cut), Pinterest Lens and
  Google Lens already take a photo.
- **Would they switch?** To a new app with ~6k (licensed) or ~20k
  (unlicensable) photos versus Pinterest's whole fashion graph, with a
  measured second-clause failure ("white sneakers" satisfied in 1 of 4 in
  the handoff's own eyeball test)? No. Frequency is also low: this is an
  occasional shopping moment, not a daily habit, so an app has to win it on
  the first try with no retention loop.

**Importance: low.** Real but mild, well served, and the incumbents' answer
is good enough that the switch cost isn't worth paying.

## 3. Feasibility — the hardest risks

1. **Licensing (fatal for a public app).** 0/20,532 photos licensed; the
   corpus is identifiable real people from Reddit, Pinterest (collected
   against a robots.txt deny through an authenticated session) and wear.jp.
   Options per the memo: thumbnails + link-out (still a copy, likeness risk
   unchanged), wear.jp-only (1,367 men's Japanese coordinates), Pexels
   rebuild (~6k stock-posed photos, "less real"), catalog-only (a different
   product), or don't ship. Every shippable option removes the "real
   people's fits" differentiator or keeps a lawyer-sized risk.
2. **Quality.** Encoder is the binding constraint (mean pairwise similarity
   0.743 between random outfit photos; "a photo of a bicycle" scores 0.155).
   Five measured colour/binding fixes did nothing. Fine-tuning needs labels
   and is ~$10–15 of Modal per round. Plausible to improve, not plausible to
   reach "≥75% of top-4 satisfy both clauses" soon from 46% p@15 on a
   pool the handoff itself calls optimistic.
3. **Distribution.** A voice round-trip through Siri Shortcuts competes with
   a system feature (Visual Intelligence) that already hands the same
   screenshot to Google or ChatGPT in one gesture.

A cheap prototype can't retire risk 1; it is a legal question, and the
best engineering answer (Pexels) deletes the point of the product.

## 4. Edge — why would this version win?

It wouldn't, as a product. Pinterest owns the corpus, the rights, the
query and the habit; Google and OpenAI own the OS-level entry point; closet
apps own "with *my* jorts". This repo's assets (masked-crop detection, a
SigLIP2/DINOv3 hybrid, a two-clause eval idea, a Modal deployment) are
engineering skills, not a user-visible edge.

**As a learning project** (the README literally says "Just for fun
project, mostly for myself!") it was worth it and arguably done: it shipped
an end-to-end iOS → Modal retrieval system and produced real findings
(masked crops 72.4% vs bbox 37.7% group agreement; colour is not the
retrieval bottleneck). Private, non-distributed use is option (d) in the
licensing memo and needs no further work from the gardener.

## 5. What happens now

- fash-000 and every open node under it → `killed`. No more gardener runs
  here unless the owner overrules in INBOX.md with a reason.
- fash-004 (licensing decision) is moot under KILL: there is nothing to
  ship. It stays recorded so an overrule can pick it up.
- No code was changed this run.

## 6. What would change my mind

Any one of these, written into INBOX.md, reopens the seed:

1. **A narrower user Pinterest fails.** Evidence (even 10 real
   searches, screenshotted) that Pinterest/Google Lens return bad
   results for some class of item + text that this repo returns well,
   e.g. unnameable items, or a clause Pinterest drops. That is the only
   product-shaped edge left, and it is cheap to test by hand.
2. **A licensed corpus that is still "real people"**: creators opting in
   (a partner program), or a licensed source other than stock. Then the
   differentiator survives licensing.
3. **An explicit research goal instead of a product**: e.g. "fine-tune the
   encoder on outfit photos and publish the two-clause eval as a
   benchmark". That is a legitimate learning seed, but it is a new SEED.md
   with its own validation, not this one.
