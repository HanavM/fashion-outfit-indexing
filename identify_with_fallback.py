"""Wires the exact-product pipeline's open-set rejection to an actual
similarity-based fallback -- the specific gap flagged as "worth wiring...
not implemented" in docs/roadmap.md's 2026-08-01 exact-retrieval-scaling
design discussion: `hierarchical_retrieval_pipeline.py`'s
`reject_threshold` (added the same day) already lets `retrieve()` decline
to assert an exact-product identification, but nothing downstream of
that decision existed yet -- a rejected query just printed a warning and
still showed DINOv3's raw top-K as unconfirmed "closest visual
candidates." This script is the actual fallback: when rejection fires,
run `color_similarity_search.py` (dominant-color CIEDE2000 match) against
the SAME query image and present those as similar-not-identical items
instead.

Why color similarity and not free-text search as the fallback: this
script's input is a photo, not a text description. `free_text_visual_
search.py` needs a text query to run at all, so it can't be triggered
automatically from an image-only pipeline -- it's a natural SECOND step
a human takes ("describe what you're looking for instead"), not
something this script can invoke on its own. `color_similarity_search.py`
is the one existing tool in this repo that, like the exact pipeline
itself, takes an image and needs no text input -- the natural automatic
fallback.

Honesty note carried over from hierarchical_retrieval_pipeline.py's own
REJECT_SIMILARITY_THRESHOLD comment: this project has never calibrated
a real reject_threshold (that needs off-catalog test images this project
doesn't have yet -- see docs/eval_log.md's reject_threshold_sweep rows
once a real --evaluate run produces them). Without an explicit
--reject-threshold, this script still runs the full fallback chain
end-to-end and is safe/useful for validating the PLUMBING, but the
"should we have rejected this" decision itself is not yet a calibrated
one -- passing no threshold means retrieve() never rejects, so this
script's fallback path is exercised only via --force-fallback (skips
the exact stage and demonstrates the fallback response format directly)
until a real threshold exists.

Usage:
    python3 identify_with_fallback.py --image path/to/query.jpg --reject-threshold 0.35
    python3 identify_with_fallback.py --image path/to/query.jpg --force-fallback  # skip exact stage, smoke-test the fallback path alone
"""

import argparse
import json

from hierarchical_retrieval_pipeline import HierarchicalRetriever, REJECT_SIMILARITY_THRESHOLD
import color_similarity_search


def identify_or_fallback(retriever, image_path, reject_threshold=REJECT_SIMILARITY_THRESHOLD,
                          use_category_gate=False, use_score_fusion=False, use_patch_rerank=False,
                          fallback_top_k=10, force_fallback=False):
    """Returns a dict with `mode`: "exact" (a confident product
    identification, from hierarchical_retrieval_pipeline.py unchanged) or
    "similar_fallback" (rejection fired, or --force-fallback -- a ranked
    list of visually-similar-colored catalog items, explicitly NOT an
    exact-product claim)."""
    if not force_fallback:
        exact_result = retriever.retrieve(
            image_path, use_category_gate=use_category_gate, reject_threshold=reject_threshold,
            use_score_fusion=use_score_fusion, use_patch_rerank=use_patch_rerank,
        )
        if not exact_result["rejected_open_set"]:
            return {"mode": "exact", "exact_result": exact_result}
    else:
        exact_result = None

    fallback = color_similarity_search.search(image_path, top_k=fallback_top_k)
    return {
        "mode": "similar_fallback",
        "exact_result": exact_result,  # None if --force-fallback, else the rejected attempt (kept for context/debugging)
        "fallback_results": fallback["results"],
    }


def print_response(image_path, response):
    print(f"\nQuery: {image_path}")
    if response["mode"] == "exact":
        result = response["exact_result"]
        print(f"CONFIDENT EXACT MATCH (reject_threshold cleared, or none set):")
        for entry in result["results"]:
            print(f"  #{entry['rank']}  {entry['brand']} {entry['name']}  [{entry['product_code']}]  "
                  f"score={entry['dino_identity_score']:.4f}")
        return

    exact = response["exact_result"]
    if exact is not None:
        print(f"NO CONFIDENT EXACT MATCH (best score {exact['results'][0]['dino_identity_score']:.4f} "
              f"below reject_threshold={exact['reject_threshold']:.4f})." if exact["results"]
              else "NO CONFIDENT EXACT MATCH (no identity candidates at all).")
    else:
        print("Exact-match stage skipped (--force-fallback).")
    print("Falling back to similar-COLOR items (NOT an exact-product identification):")
    for entry in response["fallback_results"]:
        print(f"  #{entry['rank']}  {entry['brand']} {entry['name']}  [{entry['product_code']}]  "
              f"color={entry['canonical_color']} ({entry['hex']})  ciede2000={entry['ciede2000_distance']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--reject-threshold", type=float, default=REJECT_SIMILARITY_THRESHOLD,
                         help="See hierarchical_retrieval_pipeline.py's own flag -- not calibrated yet, "
                              "unset means the exact stage never rejects.")
    parser.add_argument("--force-fallback", action="store_true",
                         help="Skip the exact-match stage entirely and go straight to the color-similarity "
                              "fallback -- useful for smoke-testing the fallback path without needing a "
                              "calibrated reject_threshold or the full SigLIP2+DINOv3 pipeline to run.")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    retriever = None if args.force_fallback else HierarchicalRetriever()
    response = identify_or_fallback(
        retriever, args.image, reject_threshold=args.reject_threshold,
        fallback_top_k=args.top_k, force_fallback=args.force_fallback,
    )
    print_response(args.image, response)
