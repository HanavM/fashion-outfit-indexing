"""Measure `brand_evidence.py` against the catalog's ground-truth brand.

Every catalog record carries a `brand` field, so a brand read out of the
image is directly checkable. That makes this the deliverable, not the
detector: item 11.3 in `docs/product_gap_analysis.md` says brand evidence
should only be wired into retrieval **if the measurement justifies it**,
because a filter that is wrong some of the time removes the true product
from the candidate set unrecoverably -- the exact way the category gate
failed six times (see `docs/eval_log.md`).

What it reports:

  * **no-text rate** -- how often OCR finds nothing usable at all. This
    bounds how much brand evidence can ever help, so it is printed first.
  * per-brand precision / recall / abstain, at a chosen accept threshold
  * a full confusion matrix including the "abstain" column
  * a sweep of the accept threshold, so the operating point is chosen from
    the curve rather than assumed

Sampling is deterministic (`--seed`), by product then by image, so reruns
are comparable. Results are checkpointed per image to `--out`, so a long
CPU/MPS run can be resumed with `--resume`.

    .venv/bin/python brand_evidence_eval.py --products-per-brand 50 \\
        --images-per-product 2 --out logs/brand_evidence_eval.json
    .venv/bin/python brand_evidence_eval.py --report-only \\
        --out logs/brand_evidence_eval.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from brand_evidence import (
    DEFAULT_ACCEPT_SCORE,
    METADATA_PATH,
    BrandDetector,
    resolve_image_path,
)


def build_sample(products_per_brand: int, images_per_product: int, seed: int) -> list[dict]:
    with open(METADATA_PATH) as fh:
        records = json.load(fh)

    by_brand: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        if rec.get("brand") and rec.get("images"):
            by_brand[rec["brand"]].append(rec)

    rng = random.Random(seed)
    sample: list[dict] = []
    for brand in sorted(by_brand):
        prods = sorted(by_brand[brand], key=lambda r: r.get("product_code", "") or r["slug"])
        rng.shuffle(prods)
        picked = 0
        for rec in prods:
            if picked >= products_per_brand:
                break
            resolvable = [p for p in (resolve_image_path(r) for r in rec["images"]) if p]
            if not resolvable:
                continue
            chosen = resolvable if len(resolvable) <= images_per_product else \
                rng.sample(resolvable, images_per_product)
            for p in chosen:
                sample.append({
                    "brand": brand,
                    "product_code": rec.get("product_code"),
                    "slug": rec.get("slug"),
                    "image": str(p),
                })
            picked += 1
    return sample


def run(args) -> list[dict]:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: dict[str, dict] = {}
    if args.resume and out_path.exists():
        done = {r["image"]: r for r in json.loads(out_path.read_text())}
        print(f"resuming: {len(done)} images already scored")

    sample = build_sample(args.products_per_brand, args.images_per_product, args.seed)
    todo = [s for s in sample if s["image"] not in done]
    print(f"sample: {len(sample)} images over {len(set(s['brand'] for s in sample))} brands; "
          f"{len(todo)} to run")

    det = BrandDetector(device=args.device)
    print(f"device={det.device} max_dim={det.max_dim}")

    results = list(done.values())
    t0 = time.time()
    for i, item in enumerate(todo, 1):
        try:
            # `item["brand"]` is ground truth and the detector's own `brand`
            # field would silently overwrite it, so it is renamed on the way in.
            fields = det.detect(item["image"]).as_dict()
            row = {**item, "brand_pred": fields.pop("brand"), **fields}
        except Exception as exc:                      # a corrupt file must not kill an hour of work
            row = {**item, "brand_pred": None, "score": 0.0,
                   "reason": "error", "error": repr(exc)}
        results.append(row)
        if i % args.checkpoint_every == 0 or i == len(todo):
            out_path.write_text(json.dumps(results, indent=1))
            rate = (time.time() - t0) / i
            print(f"  {i}/{len(todo)}  {rate:.2f}s/img  eta {(len(todo)-i)*rate/60:.1f} min",
                  flush=True)
    out_path.write_text(json.dumps(results, indent=1))
    return results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _accepted(row: dict, threshold: float) -> str | None:
    if row.get("reason") != "match":
        return None
    return row["brand_pred"] if row.get("score", 0.0) >= threshold else None


def report(results: list[dict], threshold: float) -> None:
    brands = sorted({r["brand"] for r in results})
    n = len(results)

    no_text = sum(1 for r in results if r.get("reason") == "no_text")
    no_match = sum(1 for r in results if r.get("reason") == "no_brand_match")
    errors = sum(1 for r in results if r.get("reason") == "error")

    print("\n" + "=" * 78)
    print(f"BRAND EVIDENCE EVAL -- {n} images, {len(brands)} brands, "
          f"accept threshold {threshold:.2f}")
    print("=" * 78)
    print(f"\nNO-TEXT RATE: {no_text/n:6.2%}  ({no_text}/{n} images where OCR found "
          f"nothing above the confidence floor)")
    print(f"text but no brand match: {no_match/n:6.2%}  ({no_match}/{n})")
    if errors:
        print(f"errors: {errors}")

    tr = [r for r in results if r.get("reason") in ("no_text", "no_brand_match", "match")]
    regions = [r.get("n_text_regions", 0) for r in tr]
    print(f"median OCR text regions per image: {sorted(regions)[len(regions)//2] if regions else 0}")

    # ---- per brand -------------------------------------------------------
    tp = Counter(); fp = Counter(); support = Counter(); abstain = Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        truth = r["brand"]
        support[truth] += 1
        pred = _accepted(r, threshold)
        confusion[truth][pred or "(abstain)"] += 1
        if pred is None:
            abstain[truth] += 1
        elif pred == truth:
            tp[pred] += 1
        else:
            fp[pred] += 1

    print(f"\n{'brand':<12}{'n':>5}{'recall':>9}{'precision':>11}{'abstain':>9}"
          f"{'TP':>6}{'FP':>6}")
    print("-" * 78)
    for b in brands:
        pred_total = tp[b] + fp[b]
        prec = tp[b] / pred_total if pred_total else float("nan")
        rec = tp[b] / support[b] if support[b] else float("nan")
        print(f"{b:<12}{support[b]:>5}{rec:>9.2%}{prec:>11.2%}"
              f"{abstain[b]/support[b]:>9.2%}{tp[b]:>6}{fp[b]:>6}")

    total_tp = sum(tp.values()); total_fp = sum(fp.values())
    total_pred = total_tp + total_fp
    print("-" * 78)
    print(f"{'OVERALL':<12}{n:>5}{total_tp/n:>9.2%}"
          f"{(total_tp/total_pred if total_pred else float('nan')):>11.2%}"
          f"{(n-total_pred)/n:>9.2%}{total_tp:>6}{total_fp:>6}")
    print("\nrecall    = correct brand read / all images of that brand")
    print("precision = correct / all images the detector ASSERTED that brand for")
    print("abstain   = detector returned no brand (the safe outcome)")

    # ---- confusion matrix ------------------------------------------------
    cols = brands + ["(abstain)"]
    print("\nCONFUSION MATRIX (rows = truth, cols = prediction)")
    header = f"{'':<12}" + "".join(f"{c[:7]:>8}" for c in cols)
    print(header)
    for b in brands:
        row = f"{b:<12}" + "".join(f"{confusion[b][c] or '.':>8}" for c in cols)
        print(row)

    # ---- threshold sweep -------------------------------------------------
    print("\nACCEPT-THRESHOLD SWEEP")
    print(f"{'thr':>6}{'asserted':>10}{'precision':>11}{'recall':>9}{'wrong':>8}")
    for thr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]:
        t = w = 0
        for r in results:
            pred = _accepted(r, thr)
            if pred is None:
                continue
            if pred == r["brand"]:
                t += 1
            else:
                w += 1
        asserted = t + w
        prec = t / asserted if asserted else float("nan")
        print(f"{thr:>6.2f}{asserted:>10}{prec:>11.2%}{t/n:>9.2%}{w:>8}")
    print("\n'wrong' is the number that decides whether this can be a retrieval\n"
          "filter at all -- each one removes the true product unrecoverably.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products-per-brand", type=int, default=50)
    ap.add_argument("--images-per-product", type=int, default=2)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="logs/brand_evidence_eval.json")
    ap.add_argument("--threshold", type=float, default=DEFAULT_ACCEPT_SCORE)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        results = json.loads(Path(args.out).read_text())
    else:
        results = run(args)
    report(results, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
