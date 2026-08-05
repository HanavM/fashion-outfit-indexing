# Design note: a unified `/query` interface — 2026-08-05

**This document replaces nothing.** `docs/roadmap_to_deployment.md` and
`docs/product_gap_analysis.md` remain the plan of record; this is a
forward-looking design for what a general visual search engine over this
catalog would look like, and what our own measurements say about the
obvious way to build one.

## The question

Today a caller must pre-sort their intent into one of three endpoints
before asking anything:

    POST /identify   image        -> ranked products
    POST /compose    image + text -> primary + companions
    POST /search     text         -> ranked products

Nothing routes. `siri_client.py` picks by a one-line rule (`--text`
present -> `/compose`, else `/identify`), and the iOS side ships two
separate Shortcuts so that Siri's own phrase matching does the routing.
No LLM is involved anywhere in the request path.

That works, and it is brittle in a specific way: "what brand are these"
or "find me blue jeans like this" map to no endpoint and have no
fallback. The natural wish is a search engine — one box, text and/or
image in, most-similar things out.

## The trap: unified INTERFACE and unified REPRESENTATION are different

The appealing version is one embedding space: project text and image into
a shared space, one ANN lookup, done. We have already measured a small
version of that, and it lost.

`--score-fusion` blended the SigLIP2 semantic score with the DINOv3
identity score into a single ranking. Result: **−6.22pt R@1** (50.76% ->
44.54%), with shortlist miss *identical* in both arms — a pure ranking
regression, not a coverage one. The mechanism, from
`docs/eval_log.md`:

> SigLIP2 scores are **identity-level**, so every colorway sibling of a
> model receives the SAME score. Blending that into DINOv3's per-product
> score flattens exactly the sibling distinctions DINOv3 was fine-tuned
> to make.

That is the whole tension in one sentence, and it generalises:

| query | wants a metric where... |
|---|---|
| "show me blue jeans" | all blue jeans are NEAR each other |
| "show me *this exact* sneaker" | two colorways of one shoe are FAR apart |

Those are opposite geometries. A single space must pick one, and DINOv3's
identity fine-tune bought **+31.3pt** (25.24% -> 56.55%) precisely by
choosing the second. Collapsing the two spaces spends that.

Spec section 2.2's "two encoder roles" is this argument, made before the
measurements existed. The measurements agree with it.

## What search engines actually do

They do not use one embedding either. The shape is cheap high-recall
retrieval, then expensive rerank, then policy. We already have it:

    SigLIP2 semantic shortlist      (recall)
        v
    DINOv3 identity rerank          (precision)
        v
    HSC backoff, garment gate,      (policy)
    open-set rejection

So the thing we lack is not unified representation. It is the **single
entry point**.

## The proposal: `POST /query`

One endpoint, `{image?, text?, top_k?}`. It infers the query's *shape*
and weights the existing stages accordingly, rather than making the
caller choose a URL.

| input | inferred intent | path |
|---|---|---|
| image only | instance-seeking | identity rerank dominant (today's `/identify`) |
| text only, names a category/attribute | catalog browse | semantic/canonical search, skip DINOv3 entirely (today's `/search`) |
| image + companion phrase ("with X") | outfit composition | anchor identity + co-occurrence (today's `/compose`) |
| image + property question ("what brand") | evidence extraction | brand evidence path |
| image + "this exact" / "like this" | exactness cue | raise identity weight, lower semantic |

**The inference is cheap and mostly non-semantic**: is an image present;
does the text contain a term in the catalog taxonomy; does it contain a
companion preposition ("with", "goes with"); does it contain an exactness
cue. `parse_text_fragment()` already does the taxonomy half.

An LLM router is the expensive version. It buys robustness on phrasings
the vocabulary misses — not new capability — so it should be deferred
until the rule-based version is measured and its failure modes are known.

**This is a routing change, not a retrieval change.** It should be
possible to ship `/query` with the three existing endpoints untouched
behind it, and verify that `/query` reproduces each of them exactly on
the same inputs. If it does not, the routing is wrong and that is
measurable rather than a matter of taste.

## What would have to be true to justify ONE space

If the goal really is a single embedding, the honest path is not
averaging two scores. It is a **multi-vector / late-interaction** model
(ColBERT-style): keep several vectors per product and learn a
query-conditioned combiner, so the metric *adapts* to the query instead
of being a fixed blend. That is a training project, not a refactor.

Before committing to it, note the track record: three attempts to add
signal to the ranking stage have now all lost or tied — patch-rerank
(−30pt), score fusion (−6.2pt), brand boost (+0.10pt) — because each
re-weighted information the identity embedding already encoded. A learned
combiner is a fourth attempt at the same class of thing, and should be
held to the same evidential bar.

## The scaling constraint that arrives first

Independent of the interface, a general visual search engine needs an
**approximate nearest-neighbour index**. Today retrieval is a dense
matmul over the whole gallery, and the identity shortlist is a fixed
top-K.

The 2026-08-04/05 catalog sync made the cost of that concrete: doubling
the gallery from 1,077 to 2,230 products drove shortlist miss from 1.18%
to 19.95% at K=150, and recovering it needed K=400 (miss 5.92%, R@1
43.05% -> 47.65%). **K is not an absolute; it scales with catalog size.**
A catalog an order of magnitude larger makes a fixed-K dense scan both
slow and lossy.

FAISS or HNSW over the identity and semantic spaces is the standard
answer, and it is infrastructure this design needs regardless of how the
query interface evolves. It is probably the first concrete step toward
"search engine" in the scaling sense, and it is independent of everything
above — worth doing on its own merits.

## Summary

- Unify the **interface**: one `/query`, rules-based routing, verified to
  reproduce the existing endpoints exactly.
- Do **not** unify the **representation** by blending scores; we measured
  that at −6.2pt and understand why.
- If one space is genuinely wanted, do it as a learned multi-vector
  combiner, and hold it to the bar the last three ranking ideas failed.
- Add an ANN index regardless; the catalog-size result shows fixed-K
  dense scan does not scale.
