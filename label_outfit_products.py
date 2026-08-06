"""Label the garments in real outfit photos with catalog PRODUCTS.

    .venv/bin/python label_outfit_products.py --n 3000
    .venv/bin/python label_outfit_products.py --n 3000 --resume
    .venv/bin/python label_outfit_products.py --report

Runs each detected garment crop through the deployed identification
pipeline (SigLIP2 semantic shortlist -> DINOv3 identity rerank) and stores
the ranked catalog products **with their scores**, so downstream code can
threshold them instead of trusting them.

## Why this exists

`outfit_search.py` ranks by embedding cosine alone, and the measurements
in its own module docstring show why that is weak: outfit photos have
mean pairwise similarity 0.743, so the top 50 for any query sit within
0.015 of each other and a nonsense query still scores 0.155. Structured
labels are the way out -- filter cheaply and precisely on what a photo
CONTAINS, then rerank the survivors by embedding. That cascade (cheap
high-recall shortlist -> expensive precise rerank) is the shape that beat
score fusion by 6.22pt on the identification task.

## What these labels are worth, stated before anyone uses them

**Expect most exact-product labels to be wrong, and design around that.**

- Best measured R@1 is **47.65%**, and that is catalog photo -> catalog
  product. These are crops from consumer photos, a condition this project
  has **never measured** (`pair_eval.py` exists to fix that and has not
  been run).
- The gallery is **12 brands** (the deployed volume catalog; the six new
  brands were scraped but never synced). Most garments in a random Reddit
  or Pinterest outfit are not in it at all.
- Open-set rejection is **uncalibrated** -- AUROC 0.769, no usable
  operating point, so `rejected_open_set` never fires. Nothing stops a
  confident answer for a product we do not stock. An early spot check had
  consumer jeans scoring **0.902** against "Levi's 550".

So every record here keeps `score`, `rank`, the runner-up, and the
garment gate, and `--report` prints the score distribution rather than a
label count. Treat `score` as the only thing standing between a label and
a guess.

**The brand field is the sturdier signal.** Plain nearest-neighbour on
the frozen SigLIP2 embedding gets brand right **96.49%** of the time on
catalog images (`docs/eval_log.md`, 2026-08-05) -- which is exactly why
`--brand-boost` bought only +0.10pt: the embedding already knew. That was
a negative result for boosting rank and is a positive one for labelling.
It is still a catalog-photo number; these are consumer crops.

## Cost

Uses the ALREADY DEPLOYED endpoint, so there is no new app, no deploy,
and no idle container beyond the one serving requests. Prints a running
estimate against `--budget` and stops rather than overrunning it. Check
real spend with `modal billing summary`; the estimate here is wall-clock
x an assumed A10G rate and is not authoritative.
"""

import argparse
import base64
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OUTFIT_METADATA = REPO_ROOT / "outfit_dataset" / "metadata.json"
LABELS_PATH = REPO_ROOT / "outfit_dataset" / "product_labels.json"
DEFAULT_URL = "https://hanavm--fashion-serve-fashionservice-api.modal.run"

# Modal A10G list price at time of writing. Only used for the running
# estimate and the budget guard -- `modal billing summary` is the truth.
A10G_DOLLARS_PER_HOUR = 1.10


def load_api_key():
    key = os.environ.get("FASHION_API_KEY")
    if key:
        return key
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("FASHION_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("FASHION_API_KEY not found in environment or .env")


def sample_crops(count, seed, per_photo):
    """Seeded sample of detected garments, stratified by category.

    Stratified because the raw distribution is dominated by pants (5,411)
    and sneakers (4,660); an unstratified sample would say almost nothing
    about hats, sweaters or shorts. Stratification changes what the score
    distribution represents -- it is a per-category read, not a corpus
    average -- and `--report` breaks results down by category for exactly
    that reason.
    """
    records = json.loads(OUTFIT_METADATA.read_text())
    by_category = {}
    for record in records:
        post_id = f"{record.get('source')}:{record.get('source_id')}"
        seen_in_photo = {}
        for order, item in enumerate(record.get("detected_items") or []):
            source_image = item.get("source_image")
            bbox = item.get("bbox")
            if not source_image or not bbox:
                continue
            if not (REPO_ROOT / source_image).exists():
                continue
            # Cap per photo so one busy collage cannot dominate.
            seen_in_photo[source_image] = seen_in_photo.get(source_image, 0) + 1
            if seen_in_photo[source_image] > per_photo:
                continue
            by_category.setdefault(item.get("category") or "unknown", []).append({
                "id": f"{post_id}#{order}",
                "post_id": post_id,
                "rel": source_image,
                "bbox": bbox,
                "detected_category": item.get("category"),
                "detected_group": item.get("category_group"),
                "detected_color": (item.get("color") or {}).get("name"),
                "detector_confidence": item.get("confidence"),
                "post_url": record.get("post_url"),
                "source": record.get("source"),
            })

    rng = random.Random(seed)
    for items in by_category.values():
        rng.shuffle(items)

    # Round-robin across categories until the quota is met, so small
    # categories are represented without starving large ones.
    ordered, cursors = [], {c: 0 for c in by_category}
    while len(ordered) < count:
        progressed = False
        for category in sorted(by_category):
            cursor = cursors[category]
            if cursor < len(by_category[category]):
                ordered.append(by_category[category][cursor])
                cursors[category] = cursor + 1
                progressed = True
                if len(ordered) >= count:
                    break
        if not progressed:
            break
    return ordered


def crop_bytes(item, cache):
    from PIL import Image, ImageOps

    path = str(REPO_ROOT / item["rel"])
    image = cache.get(path)
    if image is None:
        with Image.open(path) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
        cache.clear()          # single-entry cache; crops arrive grouped by photo
        cache[path] = image
    left, top, right, bottom = item["bbox"]
    if right - left < 8 or bottom - top < 8:
        return None
    import io

    buffer = io.BytesIO()
    image.crop((left, top, right, bottom)).save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


def label(args):
    import requests

    key = load_api_key()
    existing = json.loads(LABELS_PATH.read_text()) if (args.resume and LABELS_PATH.exists()) else {}
    items = sample_crops(args.n, args.seed, args.per_photo)
    todo = [i for i in items if i["id"] not in existing]
    print(f"  {len(items):,} sampled · {len(existing):,} already labelled · {len(todo):,} to do")
    if not todo:
        return

    budget_seconds = args.budget / A10G_DOLLARS_PER_HOUR * 3600
    print(f"  budget ${args.budget:.2f} ~= {budget_seconds/60:.0f} min of A10G wall time")
    print(f"  (real spend: `modal billing summary`)\n")

    session = requests.Session()
    lock = threading.Lock()
    started = time.time()
    state = {"done": 0, "failed": 0, "stop": False}
    caches = {}

    def work(item):
        if state["stop"]:
            return
        cache = caches.setdefault(threading.get_ident(), {})
        try:
            payload = crop_bytes(item, cache)
            if payload is None:
                return
            response = session.post(
                f"{args.url.rstrip('/')}/identify",
                json={"image_base64": base64.b64encode(payload).decode(),
                      "top_k": args.top_k},
                headers={"Authorization": f"Bearer {key}"}, timeout=300)
            response.raise_for_status()
            result = response.json()
        except Exception as error:  # noqa: BLE001
            with lock:
                state["failed"] += 1
            if state["failed"] <= 3:
                print(f"  [warn] {item['id']}: {error}")
            return

        products = [{"rank": r["rank"], "product_code": r["product_code"],
                     "brand": r["brand"], "name": r["name"],
                     "category": r.get("category"), "score": r["score"]}
                    for r in result.get("results", [])]
        record = {**{k: v for k, v in item.items() if k != "bbox"},
                  "bbox": item["bbox"],
                  "products": products,
                  "top1_score": result.get("confidence"),
                  # The margin between #1 and #2 is a better "is this a
                  # real match" signal than the raw score, because the
                  # score's absolute scale drifts with the crop's quality.
                  "margin": (products[0]["score"] - products[1]["score"])
                            if len(products) > 1 else None,
                  "garment_gate": result.get("garment_gate"),
                  "rejected_open_set": result.get("rejected_open_set")}

        with lock:
            existing[item["id"]] = record
            state["done"] += 1
            elapsed = time.time() - started
            if state["done"] % args.checkpoint == 0:
                LABELS_PATH.write_text(json.dumps(existing, indent=2))
                rate = state["done"] / elapsed
                cost = elapsed / 3600 * A10G_DOLLARS_PER_HOUR
                remaining = (len(todo) - state["done"]) / rate if rate else 0
                print(f"  {state['done']:5d}/{len(todo)}  {rate:.1f}/s  "
                      f"~${cost:.2f} so far  ~{remaining/60:.0f} min left")
            if elapsed > budget_seconds:
                state["stop"] = True

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))

    LABELS_PATH.write_text(json.dumps(existing, indent=2))
    elapsed = time.time() - started
    print(f"\n  labelled {state['done']:,} ({state['failed']} failed) in "
          f"{elapsed/60:.1f} min · estimated ${elapsed/3600*A10G_DOLLARS_PER_HOUR:.2f}")
    if state["stop"]:
        print("  STOPPED ON BUDGET — rerun with --resume to continue")
    print(f"  -> {LABELS_PATH}")
    print("  now: .venv/bin/python label_outfit_products.py --report")


def report(args):
    if not LABELS_PATH.exists():
        raise SystemExit(f"{LABELS_PATH} not found — run the labelling pass first")
    labels = json.loads(LABELS_PATH.read_text())
    import collections
    import statistics

    print(f"\n  {len(labels):,} garments labelled\n")

    by_category = collections.defaultdict(list)
    brands = collections.Counter()
    gate_fail = 0
    for record in labels.values():
        if record.get("top1_score") is not None:
            by_category[record.get("detected_category") or "?"].append(record)
        if record.get("products"):
            brands[record["products"][0]["brand"]] += 1
        if (record.get("garment_gate") or {}).get("looks_like_clothing") is False:
            gate_fail += 1

    print(f"  {'category':14} {'n':>5} {'median':>8} {'p90':>8} {'median':>8}")
    print(f"  {'':14} {'':>5} {'score':>8} {'score':>8} {'margin':>8}")
    for category, records in sorted(by_category.items(), key=lambda kv: -len(kv[1])):
        scores = [r["top1_score"] for r in records]
        margins = [r["margin"] for r in records if r.get("margin") is not None]
        print(f"  {category:14} {len(records):5d} {statistics.median(scores):8.3f} "
              f"{sorted(scores)[int(len(scores)*.9)-1]:8.3f} "
              f"{statistics.median(margins) if margins else 0:8.3f}")

    print(f"\n  top-1 brand distribution (12-brand gallery):")
    for brand, count in brands.most_common(12):
        print(f"    {count:5d}  {brand}")

    print(f"\n  {gate_fail:,} crops failed the garment gate "
          f"({gate_fail/max(len(labels),1):.1%}) — the gate IS calibrated "
          "(AUROC 0.9994), so these are the detector's bad crops.")

    all_margins = [r["margin"] for r in labels.values() if r.get("margin") is not None]
    if all_margins:
        margins = sorted(all_margins)
        print(f"\n  margin (#1 minus #2) — the usable confidence signal:")
        for q in (0.5, 0.75, 0.9, 0.95, 0.99):
            print(f"    p{int(q*100):<3} {margins[int(len(margins)*q)-1]:.4f}")
        print("\n  A near-zero margin means #1 and #2 were interchangeable, so the"
              "\n  exact product is not identified even when the score looks high."
              "\n  Nothing here says a label is CORRECT — only pair_eval.py, with"
              "\n  human labels, can say that. Run it before ranking on products.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command")

    l = sub.add_parser("label", help="run the labelling pass (default)")
    for parser in (ap, l):
        parser.add_argument("--n", type=int, default=3000, help="garments to label")
        parser.add_argument("--seed", type=int, default=23)
        parser.add_argument("--per-photo", type=int, default=3)
        parser.add_argument("--top-k", type=int, default=5)
        parser.add_argument("--workers", type=int, default=4,
                            help="concurrent requests; the service allows 4 per container")
        parser.add_argument("--checkpoint", type=int, default=100)
        parser.add_argument("--budget", type=float, default=1.50,
                            help="stop after roughly this many dollars of A10G time")
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--report", action="store_true",
                            help="summarise existing labels and exit")
        parser.add_argument("--url", default=os.environ.get("FASHION_API_URL", DEFAULT_URL))

    args = ap.parse_args()
    if args.report:
        report(args)
    else:
        label(args)


if __name__ == "__main__":
    main()
