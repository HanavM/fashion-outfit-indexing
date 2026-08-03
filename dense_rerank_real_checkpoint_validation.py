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
        "note": "real defining_feature 'tonal Champion script' @ 'back neckline'",
    },
    {
        "product_code": "163650272",
        "query": "a trucker jacket lined with fuzzy sherpa material on the inside",
        "note": "real defining_feature 'fuzzy sherpa lining' @ 'interior'",
    },
    # Expanded 2026-08-02 -- the first 3-case run was explicitly flagged
    # as too thin to draw a real conclusion from (docs/eval_log.md). 7
    # more real cases added, same methodology (genuine paraphrase, real
    # defining_features entries, spatially localized locations only --
    # excludes generic front/chest placements every product has).
    {
        "product_code": "904454002",
        "query": "a jacket with a mesh vent panel on the back",
        "note": "real defining_feature 'breathable mesh vent' @ 'back'",
    },
    {
        "product_code": "845264002",
        "query": "a hat with visible stitching along the curved brim",
        "note": "real defining_feature 'stitching at curved brim' @ 'brim'",
    },
    {
        "product_code": "900853002",
        "query": "a jacket with a two-tone dotted print all over",
        "note": "real defining_feature 'two-tone dotted pattern' @ 'allover'",
    },
    {
        "product_code": "8299539562688",
        "query": "a hooded sweatshirt with ribbed panels along the sides",
        "note": "real defining_feature 'ribbed side paneling' @ 'sides'",
    },
    {
        "product_code": "8583200997568",
        "query": "a sleeveless hoodie with an adjustable drawstring hood",
        "note": "real defining_feature 'adjustable drawcord hood' @ 'hood'",
    },
    {
        "product_code": "8583216627904",
        "query": "a hoodie with a small logo patch on the wrist",
        "note": "real defining_feature 'C logo patch' @ 'wrist'",
    },
    {
        "product_code": "005FM0033",
        "query": "a polo shirt with a distinctive racking design on the collar",
        "note": "real defining_feature 'racking design on the collar' @ 'collar'",
    },
]


def run():
    engine = FreeTextVisualSearch()
    total_products = len(engine.records)
    print(f"\nCatalog size for this index: {total_products} products\n")

    improved, worse, unchanged, unrescuable = 0, 0, 0, 0

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
            if delta > 0:
                improved += 1
            elif delta < 0:
                worse += 1
            else:
                unchanged += 1
        else:
            unrescuable += 1
        print()

    print("=" * 60)
    print(f"SUMMARY across {len(TEST_CASES)} cases: {improved} improved, {worse} worse, "
          f"{unchanged} unchanged, {unrescuable} unrescuable (outside the pooled top-{dense_shortlist_k})")
    print("Still an informal, hand-picked sample, not a rigorous held-out benchmark --")
    print("but a real, honest directional read given the sample size available locally.")


if __name__ == "__main__":
    run()
