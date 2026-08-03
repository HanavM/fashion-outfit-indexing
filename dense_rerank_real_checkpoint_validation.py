"""Real-checkpoint validation of the MaskCLIP-style dense-rerank fix in
free_text_visual_search.py -- flagged since it was built (2026-08-01) as
"not yet validated against the actual ceiling," blocked purely on not
having the real v3 checkpoint reachable locally. Unblocked 2026-08-02
by pulling the checkpoint down from the Modal Volume directly (data
transfer, no GPU compute/billing).

Same methodology as research_localized_query_validation.py (natural
paraphrases, never copied verbatim from training text, real target
products), but with FRESH test cases -- the original 3 targeted
products in brands with zero local image files in this dev environment
(only Gap/Champion/Levi's/Carhartt/Stussy have real images on disk here;
images are gitignored, this sandbox only has what was scraped directly
into it). One of the 3 original cases (1180436002, Gap "button tabs at
the bottom back hem") happens to still be valid locally and is reused;
2 new cases were built from real `defining_features` entries actually
present in the catalog, picked for genuine spatial localization (back
neckline, wrist, interior lining) -- the exact kind of detail a single
pooled embedding vector dilutes.

Usage:
    python3 dense_rerank_real_checkpoint_validation.py
"""

from pathlib import Path

from free_text_visual_search import FreeTextVisualSearch

TEST_CASES = [
    {
        "product_code": "1180436002",
        "query": "a denim jacket with button tabs near the bottom back hem",
        "note": "reused from research_localized_query_validation.py -- still locally valid",
    },
    {
        "product_code": "8583199555776",
        "query": "a crewneck sweatshirt with a tonal logo script at the back of the neckline",
        "note": "new -- real defining_feature 'tonal Champion script' @ 'back neckline'",
    },
    {
        "product_code": "163650272",
        "query": "a trucker jacket lined with fuzzy sherpa material on the inside",
        "note": "new -- real defining_feature 'fuzzy sherpa lining' @ 'interior'",
    },
]


def run():
    engine = FreeTextVisualSearch()
    total_products = len(engine.records)
    print(f"\nCatalog size for this index: {total_products} products\n")

    for case in TEST_CASES:
        target = case["product_code"]
        query = case["query"]
        print(f"=== Query: '{query}' ===")
        print(f"    ({case['note']})")

        pooled_results = engine.search(query, top_k=total_products)
        pooled_codes = [r["product_code"] for r in pooled_results]
        pooled_rank = pooled_codes.index(target) + 1 if target in pooled_codes else None

        dense_shortlist_k = 50
        dense_results = engine.search_dense_rerank(query, top_k=dense_shortlist_k, shortlist_k=dense_shortlist_k)
        dense_codes = [r["product_code"] for r in dense_results]
        dense_rank = dense_codes.index(target) + 1 if target in dense_codes else None

        print(f"    Pooled-vector rank:       {pooled_rank} / {total_products}")
        print(f"    Dense-rerank rank:        {dense_rank} / {dense_shortlist_k} (within its own shortlist window)"
              if dense_rank else f"    Dense-rerank rank:        not in top {dense_shortlist_k} pooled results "
                                  f"-- outside the shortlist window, dense rerank can't rescue it")
        if pooled_rank and dense_rank:
            delta = pooled_rank - dense_rank
            verdict = "IMPROVED" if delta > 0 else ("WORSE" if delta < 0 else "unchanged")
            print(f"    Delta: {delta:+d} ranks ({verdict})")
        print()


if __name__ == "__main__":
    run()
