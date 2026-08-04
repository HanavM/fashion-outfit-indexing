"""Measure brand evidence as an open-set signal (gap analysis 11.3c).

Test design, and why it uses real data rather than a synthetic split:

The existing open-set eval (`--open-set-holdout-fraction`) holds out
IDENTITIES, not brands -- a held-out Nike shoe still reads "Nike", which
IS a catalog brand, so brand evidence cannot possibly help there. That
split measures the wrong thing for this question.

What it needs is products from brands the deployment genuinely does not
stock, and that exists for free: the Modal Volume serves 6 brands
(adidas, gap, newbalance, nike, pacsun, skechers) while the local catalog
has 12. The other six -- carhartt, champion, dickies, levis, stussy,
vans -- are REAL products that are genuinely off-catalog for the deployed
service. No simulation of "unknown product" is required.

So: point brand_evidence at a 6-brand catalog, then run it over images
from all 12 and ask whether it can tell which half is which.

    python3 open_set_brand_eval.py --per-brand 40
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
from pathlib import Path

IN_CATALOG = {"adidas", "gap", "newbalance", "nike", "pacsun", "skechers"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-brand", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="logs/open_set_brand_eval.json")
    args = parser.parse_args()

    # Must be set BEFORE importing brand_evidence -- it reads the metadata
    # path at import time to build its brand vocabulary, and the whole
    # experiment depends on that vocabulary being the 6-brand one.
    scratch = Path(args.out).parent
    scratch.mkdir(parents=True, exist_ok=True)
    full = json.loads(Path("apparel_dataset/metadata.json").read_text())
    six = [r for r in full if r.get("brand") in IN_CATALOG]
    six_path = scratch / "_meta_6brand.json"
    six_path.write_text(json.dumps(six))
    os.environ["APPAREL_METADATA"] = str(six_path)

    import brand_evidence as be
    import open_set_brand_evidence as osbe

    print(f"catalog vocabulary: {be.catalog_brands()}")
    print(f"gazetteer entries : {len(osbe.off_catalog_alias_table())}")

    rng = random.Random(args.seed)
    by_brand: dict[str, list[str]] = collections.defaultdict(list)
    for record in full:
        brand = record.get("brand")
        if not brand:
            continue
        for rel in record.get("images", []):
            by_brand[brand].append(rel)

    detector = be.BrandDetector()
    rows = []
    for brand in sorted(by_brand):
        paths = by_brand[brand]
        rng.shuffle(paths)
        picked, seen = [], 0
        for rel in paths:
            resolved = be.resolve_image_path(rel)
            if resolved is None:
                continue
            picked.append(resolved)
            if len(picked) >= args.per_brand:
                break
        truth = "in_catalog" if brand in IN_CATALOG else "off_catalog"
        print(f"  {brand:12s} ({truth:11s}) {len(picked)} images", flush=True)
        for path in picked:
            ev = osbe.classify_image(path, reader=detector)
            rows.append({
                "brand": brand, "truth": truth, "image": str(path),
                "verdict": ev.verdict, "pred_brand": ev.brand,
                "reason": ev.reason, "score": ev.score,
                "matched_text": ev.matched_text,
            })

    Path(args.out).write_text(json.dumps(rows, indent=1))
    report(rows)


def report(rows):
    n = len(rows)
    fires = [r for r in rows if r["verdict"] != "no_evidence"]
    off = [r for r in rows if r["truth"] == "off_catalog"]
    inc = [r for r in rows if r["truth"] == "in_catalog"]

    print("\n" + "=" * 68)
    print(f"images: {n}  ({len(inc)} in-catalog, {len(off)} off-catalog)")
    print(f"fired at all (any verdict): {len(fires)} = {len(fires)/n:.1%}")
    print(f"  reason breakdown: {dict(collections.Counter(r['reason'] for r in rows))}")

    print("\n-- the question: does an off_catalog verdict mean off-catalog? --")
    said_off = [r for r in rows if r["verdict"] == "off_catalog"]
    correct_off = [r for r in said_off if r["truth"] == "off_catalog"]
    if said_off:
        print(f"  said off_catalog : {len(said_off)}")
        print(f"  PRECISION        : {len(correct_off)}/{len(said_off)} = "
              f"{len(correct_off)/len(said_off):.2%}")
    else:
        print("  never fired off_catalog")
    if off:
        print(f"  RECALL on off-catalog images: {len(correct_off)}/{len(off)} = "
              f"{len(correct_off)/len(off):.2%}")

    print("\n-- and the reverse: in_catalog verdicts --")
    said_in = [r for r in rows if r["verdict"] == "in_catalog"]
    correct_in = [r for r in said_in if r["truth"] == "in_catalog"]
    if said_in:
        print(f"  said in_catalog  : {len(said_in)}  precision "
              f"{len(correct_in)}/{len(said_in)} = {len(correct_in)/len(said_in):.2%}")

    print("\n-- per-brand off_catalog recall --")
    for brand in sorted({r["brand"] for r in off}):
        b = [r for r in off if r["brand"] == brand]
        hit = [r for r in b if r["verdict"] == "off_catalog"]
        print(f"  {brand:12s} {len(hit):3d}/{len(b):3d} = {len(hit)/len(b):6.1%}")

    wrong = [r for r in said_off if r["truth"] == "in_catalog"]
    if wrong:
        print(f"\n-- FALSE off_catalog calls ({len(wrong)}), the costly error --")
        for r in wrong[:8]:
            print(f"  {r['brand']:12s} read '{r['matched_text']}' as {r['pred_brand']}")


if __name__ == "__main__":
    main()
