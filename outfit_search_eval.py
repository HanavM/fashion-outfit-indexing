"""Measure outfit search against VLM-derived ground truth.

    .venv/bin/python outfit_search_eval.py label --n 600   # VLM-label a sample
    .venv/bin/python outfit_search_eval.py score           # run queries, score them
    .venv/bin/python outfit_search_eval.py report

## Why this exists

Outfit search has had **no evaluation of any kind**. Every number in this
project measures item identification (catalog photo -> catalog product);
none of it says whether "red sweater" returns red sweaters. Judgement by
eye caught two real bugs, but it cannot tell you whether a change helped
by 2% or hurt by 5%, and it cannot be re-run after every edit.

`pair_eval.py` needs a human. This does not: a **stronger, independent
model** (gpt-4o-mini vision, via the project's existing Azure Foundry
deployment) looks at each outfit photo and lists the garments and their
colours. Those labels become the ground truth that queries are scored
against.

## What this can and cannot claim

**It is not human ground truth.** It is one model grading another. That is
worth stating plainly, and it is still worth a great deal, because:

- The labeller is **independent of the retrieval stack**. It shares no
  weights, no training data and no failure modes with SigLIP2 or the ATR
  parser. When it disagrees with our detector, that disagreement is
  informative rather than circular -- unlike the group-agreement proxy,
  where both sides came from our own pipeline.
- It is **far stronger** at the specific question ("is there a red sweater
  in this photo") than a zero-shot FashionCLIP crop label.
- It is **repeatable**, so a ranking change can be measured rather than
  eyeballed.

Its own errors are real: VLMs miscall colours under coloured lighting,
confuse sweater/sweatshirt/hoodie exactly as our pipeline does, and miss
small items. So treat a score of 0.7 as "0.7 by this judge", and treat
CHANGES in the score across a code change as the trustworthy signal --
the judge's bias is constant across arms, so it cancels in a comparison.

## What it measures

For each (garment, colour) pair the VLM found often enough to query:

  **precision@k** -- of the top k photos returned for "<colour> <garment>",
  how many does the VLM independently say contain that garment in that
  colour.

Reported per query and in aggregate, split by colour-only, garment-only
and combined queries, because those fail differently -- the colour+garment
binding was the bug that motivated this.
"""

import argparse
import base64
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("APPAREL_DATASET_ROOT", str(REPO_ROOT / "apparel_dataset"))

OUTFIT_METADATA = REPO_ROOT / "outfit_dataset" / "metadata.json"
OUT_DIR = REPO_ROOT / "outfit_eval"
LABELS_PATH = OUT_DIR / "vlm_labels.json"
RESULTS_PATH = OUT_DIR / "results.json"

# The vocabularies the search side actually uses. The VLM is asked to map
# into THESE rather than free-form, so scoring is a set comparison instead
# of fuzzy string matching -- and so a miss is a real miss rather than a
# synonym mismatch ("crewneck" vs "sweatshirt").
GARMENTS = ["t-shirt", "shirt", "sweater", "hoodie", "sweatshirt", "tank top",
            "jacket", "pants", "shorts", "sneaker", "loafer", "hat", "socks"]
COLOURS = ["black", "white", "gray", "charcoal", "light gray", "navy", "blue",
           "light blue", "red", "green", "olive", "yellow", "orange", "pink",
           "purple", "brown", "beige", "cream", "tan"]

PROMPT = f"""You are labelling a photo of a person wearing an outfit, to build
an evaluation set for a clothing search engine.

List ONLY the garments clearly visible ON THE PERSON. Ignore background,
other people, and anything held rather than worn.

For each garment use EXACTLY one value from each list.
garment: {", ".join(GARMENTS)}
colour: {", ".join(COLOURS)}

Pick the single closest option. If a garment is not close to any listed
type, omit it. If you cannot tell the colour, omit the garment.

Reply with JSON only, no prose, no code fence:
{{"garments": [{{"garment": "...", "colour": "..."}}], "is_outfit_photo": true}}

Set is_outfit_photo false if this is a product/retail shot, a flat-lay, a
collage, or a close-up that does not show a person wearing clothes."""


def load_env():
    for line in (REPO_ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def client():
    from openai import AzureOpenAI

    load_env()
    return AzureOpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"],
                       api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                       azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"])


def parse_json(text):
    """The model wraps JSON in a fence perhaps half the time even when told
    not to -- gpt-4o-mini does not reliably obey formatting instructions,
    already recorded as lesson 7 in SCRAPING_PROCESS.md."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(),
                     flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


# ----------------------------------------------------------------------
# label
# ----------------------------------------------------------------------

def matchable(record):
    section = str(record.get("section") or "").lower()
    if record.get("source") == "wear" or "korean" in section:
        return False
    return not any(k in section for k in
                   ("femalefashion", "petitefashionadvice", "womens"))


def label(args):
    OUT_DIR.mkdir(exist_ok=True)
    existing = json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else {}
    records = json.loads(OUTFIT_METADATA.read_text())

    pool = []
    for record in records:
        if args.pinterest_only and record.get("source") != "pinterest":
            continue
        if not matchable(record):
            continue
        for rel in (record.get("images") or [])[:1]:
            if (REPO_ROOT / rel).exists() and rel not in existing:
                pool.append(rel)
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    todo = pool[:args.n]
    print(f"  {len(existing):,} already labelled · {len(todo):,} to do")
    if not todo:
        return

    api = client()
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    lock = threading.Lock()
    state = {"done": 0, "failed": 0}
    started = time.time()

    def work(rel):
        try:
            encoded = base64.b64encode((REPO_ROOT / rel).read_bytes()).decode()
            response = api.chat.completions.create(
                model=deployment, max_tokens=400, temperature=0,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}]}])
            parsed = parse_json(response.choices[0].message.content)
        except Exception as error:  # noqa: BLE001
            with lock:
                state["failed"] += 1
                if state["failed"] <= 3:
                    print(f"  [warn] {rel}: {error}")
            return
        if not parsed:
            with lock:
                state["failed"] += 1
            return

        # Keep only in-vocabulary pairs, so scoring is a clean set test.
        clean = [{"garment": g.get("garment"), "colour": g.get("colour")}
                 for g in (parsed.get("garments") or [])
                 if g.get("garment") in GARMENTS and g.get("colour") in COLOURS]
        with lock:
            existing[rel] = {"garments": clean,
                             "is_outfit_photo": bool(parsed.get("is_outfit_photo", True))}
            state["done"] += 1
            if state["done"] % 50 == 0:
                LABELS_PATH.write_text(json.dumps(existing, indent=2))
                rate = state["done"] / (time.time() - started)
                print(f"  {state['done']:5,}/{len(todo):,}  {rate:.1f}/s  "
                      f"~{(len(todo)-state['done'])/rate/60:.0f} min left")

    with ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
        list(pool_exec.map(work, todo))
    LABELS_PATH.write_text(json.dumps(existing, indent=2))
    print(f"\n  labelled {state['done']:,} ({state['failed']} failed) in "
          f"{(time.time()-started)/60:.1f} min -> {LABELS_PATH}")


# ----------------------------------------------------------------------
# score
# ----------------------------------------------------------------------

def build_queries(labels, min_support):
    """Queries worth asking: the (colour, garment) pairs the judge saw often
    enough that a top-k could plausibly be filled with true positives."""
    import collections

    pairs = collections.Counter()
    garments = collections.Counter()
    for entry in labels.values():
        if not entry.get("is_outfit_photo"):
            continue
        seen = set()
        for item in entry["garments"]:
            pairs[(item["colour"], item["garment"])] += 1
            seen.add(item["garment"])
        for g in seen:
            garments[g] += 1

    queries = []
    for (colour, garment), count in pairs.most_common():
        if count >= min_support:
            queries.append({"text": f"{colour} {garment}", "kind": "colour+garment",
                            "colour": colour, "garment": garment, "support": count})
    for garment, count in garments.most_common():
        if count >= min_support:
            queries.append({"text": garment, "kind": "garment",
                            "colour": None, "garment": garment, "support": count})
    return queries


def score(args):
    import outfit_search

    if not LABELS_PATH.exists():
        raise SystemExit("no VLM labels yet — run `label` first")
    labels = json.loads(LABELS_PATH.read_text())
    truth = {rel: entry for rel, entry in labels.items() if entry.get("is_outfit_photo")}
    print(f"  {len(truth):,} labelled outfit photos "
          f"({len(labels) - len(truth):,} judged not-an-outfit and excluded)")

    queries = build_queries(labels, args.min_support)
    if args.max_queries:
        queries = queries[:args.max_queries]
    print(f"  {len(queries)} queries with >= {args.min_support} supporting photos\n")

    engine = outfit_search.OutfitSearch()
    rows = []
    for query in queries:
        # Retrieve DEEP, then keep the top k among photos the judge saw.
        #
        # Scoring a plain top-20 gave 1-5 judged results per query, because
        # the labelled pool is ~9% of the corpus -- so each precision was
        # 0/1 or 3/3 and the aggregate was noise. Ranking within the judged
        # pool is the standard fix: every one of the k results is scoreable,
        # and the number measures ORDERING rather than label coverage.
        #
        # It does make the task easier than production (a smaller haystack),
        # so the absolute value is optimistic. Comparisons across arms stay
        # valid, which is what this is for.
        result = engine.search([{"kind": "text", "value": query["text"]}],
                               top_k=args.pool, drop_non_us=True, drop_womens=True,
                               type_preference=args.type_preference,
                               colour_match=args.colour_match)
        judged = hits = 0
        for hit in result["results"]:
            entry = truth.get(hit["rel"])
            if entry is None:
                continue
            if judged >= args.k:
                break
            judged += 1
            for item in entry["garments"]:
                if item["garment"] != query["garment"]:
                    continue
                if query["colour"] is None or item["colour"] == query["colour"]:
                    hits += 1
                    break
        rows.append({**query, "judged": judged, "hits": hits,
                     "precision": (hits / judged) if judged else None})
        mark = f"{rows[-1]['precision']:.0%}" if judged else "  -"
        print(f"  {query['text']:24} {mark:>5}  ({hits}/{judged} judged, "
              f"support {query['support']})")

    OUT_DIR.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(
        {"k": args.k, "type_preference": args.type_preference, "rows": rows}, indent=2))
    summarise(rows, args)


def summarise(rows, args):
    import statistics

    print(f"\n  --- precision@{args.k}, judged by gpt-4o-mini vision ---")
    for kind in ("colour+garment", "garment"):
        subset = [r for r in rows if r["kind"] == kind and r["precision"] is not None]
        if not subset:
            continue
        total_hits = sum(r["hits"] for r in subset)
        total_judged = sum(r["judged"] for r in subset)
        print(f"  {kind:16} micro {total_hits/total_judged:6.1%}  "
              f"macro {statistics.mean(r['precision'] for r in subset):6.1%}  "
              f"({len(subset)} queries, {total_judged} judged results)")

    weakest = sorted((r for r in rows if r["precision"] is not None),
                     key=lambda r: r["precision"])[:5]
    if weakest:
        print("\n  weakest queries:")
        for row in weakest:
            print(f"    {row['precision']:5.0%}  {row['text']}")
    print("\n  One model grading another, not human ground truth. The judge is\n"
          "  independent of the retrieval stack (no shared weights or training\n"
          "  data), which is what makes it informative -- but its own colour and\n"
          "  sweater/hoodie errors are real. Trust CHANGES across a code change\n"
          "  more than the absolute number; the judge's bias cancels in a diff.")


def report(args):
    if not RESULTS_PATH.exists():
        raise SystemExit("no results yet — run `score` first")
    payload = json.loads(RESULTS_PATH.read_text())
    args.k = payload["k"]
    summarise(payload["rows"], args)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    l = sub.add_parser("label", help="VLM-label a sample of outfit photos")
    l.add_argument("--n", type=int, default=600)
    l.add_argument("--seed", type=int, default=41)
    l.add_argument("--workers", type=int, default=8)
    l.add_argument("--pinterest-only", action="store_true")
    l.set_defaults(func=label)

    s = sub.add_parser("score", help="run queries and score them")
    s.add_argument("--k", type=int, default=20)
    s.add_argument("--pool", type=int, default=3000,
                   help="retrieval depth searched to find k judged results")
    s.add_argument("--min-support", type=int, default=8)
    s.add_argument("--max-queries", type=int, default=25)
    s.add_argument("--colour-match", default="both", choices=("name","lab","both"))
    s.add_argument("--type-preference", type=float, default=0.05,
                   help="0 disables the type/colour binding, for A/B")
    s.set_defaults(func=score)

    r = sub.add_parser("report", help="re-print the last score")
    r.set_defaults(func=report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
