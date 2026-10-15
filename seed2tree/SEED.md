# SEED — fashion outfit search

> **DRAFT written by Claude from README.md, docs/production_plan.md and
> docs/SESSION_HANDOFF.md (2026-10-15). Replace any of this with your own
> outline.** This file is yours: the gardener reads it from `main` every run
> and never edits it.

owner_boost: 1.0

## The final product

While shopping, you ask Siri "how would this look with my jorts?" (or open
the iOS app with a photo of an item + a line of text) and get back real
photos of people wearing outfits that contain that item *and* satisfy the
text. Item photo + text → real outfit photos. Fast enough for a voice
round-trip (~2 s), good enough that most top-4 results satisfy both clauses
of the query, and on the App Store.

## Technical direction

- Retrieval: SigLIP2 (semantic, "all blue jeans near each other") for
  outfit search; DINOv3 identity encoder only for "what exactly is this
  item" / shop-the-look.
- Serving on Modal (`outfit-serve`), warm containers, Bearer auth; iOS app
  in SwiftUI; Siri via Shortcuts/App Intents.
- Measure before tuning: `outfit_search_eval.py` exists but it's unclear
  it scores "does the result satisfy both the item and the text". That
  question is the metric that matters.
- Known quality gaps: the second clause of a query gets dropped (colour-
  given-garment 63.5%), womenswear/product shots and flat-lays leak into a
  men's real-people corpus.
- Encoder fine-tune is planned (docs/encoder_finetune_plan.md); more VLM
  labels come first.

## Definition of done

1. An outfit-search eval exists and top-4 satisfies both item and text on
   ≥ 75% of a fixed query set.
2. Corpus filters (gender, flat-lay, product shot) on by default and measured.
3. iOS app + Siri Shortcut work end to end against the deployed service,
   with a smoke test covering the client seam.
4. A corpus/licensing path I have signed off on (see
   docs/licensing_decision_memo.md) — **this is my decision, not the agent's.**
5. App Store submission package ready (submitting is mine to do).

## Out of scope / don't touch

- Spending Modal/Azure money, scraping new sites, or anything touching
  licensing decisions without asking me.
