"""Attribute heads on frozen SigLIP2 image embeddings -- product gap item 12.1.

Predicts `structured_caption.attributes` facets (material, fit, closure,
canonical_color, pattern, length) from an image at QUERY TIME, so the system
stops depending on an offline LLM captioning pass to know what a garment is
made of or how it fits.

## Why this is worth building at all

The 2026-08-04 ceiling-check row in docs/eval_log.md: a linear probe on
FROZEN SigLIP2 features already recovers these facets far above
majority-class (closure +57.9pt, canonical_color +42.9pt, fit +26.9pt,
material +14.3pt). The information is in the embedding and simply is not
used at query time. That check was 700 products with ONE image each and no
product-level holdout; this script is the real version.

## THE LABELS ARE NOT GROUND TRUTH -- read this before quoting any number

Every label here was written by `caption_apparel.py`, an LLM captioner that
had access to the product's marketing TEXT as well as its image. So:

  * a label may encode the description rather than the appearance
    ("regenerative cotton", "vegan", "workwear twill" are not visual
    categories, they are copy);
  * a head trained on them can score well by learning PRODUCT IDENTITY --
    all four images of one product share one label, and SigLIP2 clusters a
    product's images tightly -- rather than by learning the attribute.

Product-level splitting (below) removes the trivial form of the second
problem, but not the subtler one: brands are visually distinctive and a
brand's catalog shares vocabulary, so "Carhartt-looking => duck canvas" is
still learnable without seeing any canvas. **Accuracy here is accuracy at
reproducing the captioner, not accuracy against reality.** Every number
this prints carries that caveat.

## Bucketing the long tail -- decided, not silent

The raw label space is unusable as-is: 116 distinct materials, 76 fits, 56
closures, most of them singletons, many of them the same thing spelled
differently ('laces' / 'lace up' / 'lace', 'zip fly' / 'zip-fly',
'organic cotton' / 'regenerative cotton' / '100% cotton'). Two explicit
stages, both reported by --report-buckets:

  1. **Canonicalisation** (`CANONICAL_RULES`): ordered first-match keyword
     rules per facet, collapsing spellings and sub-varieties into a family
     ('nubuck suede' -> suede, 'synthetic leather' -> leather, 'cotton
     twill' -> twill). This is a JUDGEMENT CALL and it is lossy in a
     specific direction: it treats material as a visual family, so real/
     vegan/synthetic leather become one class. That is the right call for a
     head reading pixels -- nothing in the image distinguishes vegan
     leather from cowhide -- but it means this head cannot answer a
     sustainability question.
  2. **Rare-class folding**: any canonical class with fewer than
     --min-class-count training examples is folded into an explicit
     `other` class. Rare classes are NOT dropped; dropping them would
     inflate accuracy by removing the hard examples, and would leave the
     head unable to say "not one of the things I know". `other` is scored
     like any other class and its share of the data is reported.

## Honest eval

Split is by PRODUCT, not by image (two images of one product cannot
straddle the split -- otherwise this measures memorisation). Reported per
facet: image-level accuracy, product-level accuracy (mean of the product's
image probabilities), and the majority-class baseline computed as the TRAIN
majority class evaluated on TEST -- the number a system that always guessed
the most common thing would get.

## Does it help retrieval? (--retrieval-eval)

The question that actually matters, and the reason this file also contains
a retrieval experiment rather than stopping at accuracy. Setup:

  * query = one held-out image of a test-split product; gallery = every
    catalog product, represented by the mean of its remaining images
    (the same prototype construction the real pipeline uses);
  * baseline = cosine ranking on SigLIP2 v3 embeddings, i.e. the pipeline's
    shortlist stage in isolation (this is NOT the full pipeline's R@1 --
    there is no DINOv3 rerank here -- so compare arms within this script,
    never to a Phase 4 row);
  * treatment = rerank the top-K by adding w * sum over facets of
    log P(head predicts the CANDIDATE's label | query image).

Usage:
    python3 attribute_heads.py --report-buckets
    python3 attribute_heads.py --train
    python3 attribute_heads.py --train --mlp
    python3 attribute_heads.py --retrieval-eval
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

METADATA_PATH = Path("apparel_dataset/metadata.json")
# Written by modal_app_attribute_embed.py on the Volume, pulled down with
# `modal volume get`. Deliberately NOT inside apparel_dataset/ -- that tree is
# read-only for this work (it is the labelled gallery every eval number
# depends on) and nothing here should be able to touch it.
EMBED_DIR = Path(os.environ.get("ATTRIBUTE_EMBED_DIR", "attribute_head_embeddings"))
FACETS = ["material", "fit", "closure", "canonical_color", "pattern", "length"]
OTHER = "other"

# Ordered (regex, canonical) rules per facet. First match wins, so the more
# specific pattern must come first ('cotton twill' must hit twill before
# cotton). Anything matching nothing keeps its normalised string and then
# faces the min-count fold.
CANONICAL_RULES = {
    "material": [
        (r"suede|nubuck", "suede"),
        (r"leather", "leather"),          # real/vegan/faux/synthetic all collapse
        (r"denim", "denim"),
        (r"corduroy", "corduroy"),
        (r"fleece|terry|pile|sherpa", "fleece"),
        (r"mesh", "mesh"),
        (r"canvas|duck", "canvas"),
        (r"twill|ripstop|poplin|oxford|flannel|chambray", "twill"),
        (r"linen", "linen"),
        (r"wool|acrylic|boucle|cashmere", "wool"),
        (r"nylon|polyamide|supplex|gore tex", "nylon"),
        (r"poly", "polyester"),
        (r"knit|jersey", "knit"),
        (r"cotton", "cotton"),
        (r"vegan|synthetic|textile|fabric|tpu|flex|rubber|quilt", "synthetic"),
    ],
    "fit": [
        (r"crop", "cropped"),
        (r"relax|roomy|generous|easy|spacious", "relaxed"),
        (r"loose|baggy|oversized|boxy|wide|big|tall|larger", "loose"),
        (r"slim|skinny|snug|tight|fitted|streamlined|athletic|taper", "slim"),
        (r"straight|bootcut|flare", "straight"),
        (r"adjust|customiz|elastic|cinch", "adjustable"),
        (r"regular|standard|classic|true to size|modern|timeless|versatile", "regular"),
    ],
    "closure": [
        (r"snapback|strapback", "strapback"),
        (r"button fly|fly.*button|button.*fly", "button fly"),
        (r"fly", "zip fly"),
        (r"lace", "lace-up"),
        (r"zip", "zip"),
        (r"snap", "snap"),
        (r"button|placket", "button"),
        (r"crew|v neck|pullover|hood", "pullover"),
        (r"drawcord|drawstring|bungee|elastic", "drawstring"),
        (r"buckle|hook|velcro|adjust|strap", "adjustable strap"),
        (r"none|no closure|slip on", "none"),
    ],
    "length": [
        (r"crop|cutoff|3 4|7 8|above the knee|mid thigh|short", "cropped"),
        (r"knee", "knee length"),
        (r"full|long|maxi", "full length"),
        (r"(\d+(\.\d+)?)\s*(inch|in\b|inseam)", "__inches__"),
    ],
    "canonical_color": [],   # already only 18 classes; min-count fold is enough
    "pattern": [],
}


def normalize(value):
    return re.sub(r"\s+", " ", str(value).lower().replace("-", " ").replace("/", " ")).strip()


def canonicalize(facet, raw):
    text = normalize(raw)
    if not text:
        return None
    for pattern, target in CANONICAL_RULES.get(facet, []):
        match = re.search(pattern, text)
        if match:
            if target == "__inches__":
                # Inseams split cleanly into two real garments: shorts are
                # quoted at 5-15", trousers at 29-34". Anything between is
                # noise, not a third category.
                inches = float(match.group(1))
                return "short inseam" if inches <= 20 else "full length"
            return target
    return text


def primary_label(record, facet):
    value = record.get("structured_caption", {}).get("attributes", {}).get(facet)
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None or not str(value).strip():
        return None
    return canonicalize(facet, value)


def load_records():
    records = json.loads(METADATA_PATH.read_text())
    return {r["product_code"]: r for r in records}


def load_embeddings(encoder):
    index = json.loads((EMBED_DIR / "index.json").read_text())
    embeddings = np.load(EMBED_DIR / f"{encoder}_image_embeddings.npy")
    assert len(index) == embeddings.shape[0], "index/embedding length mismatch"
    return index, embeddings


def product_split(codes, seed=13, fractions=(0.7, 0.15, 0.15)):
    """Split by PRODUCT. Two images of one product must never straddle the
    split -- otherwise the head can memorise the product and the accuracy is
    meaningless."""
    ordered = sorted(codes)
    random.Random(seed).shuffle(ordered)
    n_train = int(len(ordered) * fractions[0])
    n_val = int(len(ordered) * (fractions[0] + fractions[1]))
    return set(ordered[:n_train]), set(ordered[n_train:n_val]), set(ordered[n_val:])


def build_facet_dataset(facet, index, embeddings, records, train_codes, min_count):
    """Returns (rows, classes) where rows are (row_index, product_code, class_index)."""
    labels = {code: primary_label(records[code], facet)
              for code in {row["product_code"] for row in index} if code in records}
    train_counts = Counter(label for code, label in labels.items()
                           if label and code in train_codes)
    kept = {label for label, count in train_counts.items() if count >= min_count}
    classes = sorted(kept) + [OTHER]
    class_index = {name: i for i, name in enumerate(classes)}

    rows = []
    for i, row in enumerate(index):
        label = labels.get(row["product_code"])
        if label is None:
            continue           # unlabelled product: cannot supervise, excluded
        rows.append((i, row["product_code"], class_index.get(label, class_index[OTHER])))
    return rows, classes, labels


def train_head(x_train, y_train, x_val, y_val, n_classes, mlp=False, epochs=60, seed=0):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    dim = x_train.shape[1]
    model = (nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, n_classes))
             if mlp else nn.Linear(dim, n_classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    xt, yt = torch.tensor(x_train), torch.tensor(y_train)
    xv, yv = torch.tensor(x_val), torch.tensor(y_val)

    best_state, best_acc = None, -1.0
    batch = 256
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(xt))
        for start in range(0, len(xt), batch):
            idx = order[start:start + batch]
            optimizer.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            acc = (model(xv).argmax(dim=1) == yv).float().mean().item() if len(xv) else 0.0
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model, best_acc


def evaluate_facet(model, x_test, y_test, codes_test, classes):
    import torch

    with torch.no_grad():
        logits = model(torch.tensor(x_test))
        probabilities = torch.softmax(logits, dim=1).numpy()
    image_pred = probabilities.argmax(axis=1)
    image_acc = float((image_pred == y_test).mean()) if len(y_test) else 0.0

    by_product = defaultdict(list)
    for i, code in enumerate(codes_test):
        by_product[code].append(i)
    correct = 0
    for code, idxs in by_product.items():
        pooled = probabilities[idxs].mean(axis=0).argmax()
        correct += int(pooled == y_test[idxs[0]])
    product_acc = correct / max(1, len(by_product))
    return image_acc, product_acc, image_pred, probabilities


def macro_f1(y_true, y_pred, n_classes):
    scores = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def report_buckets(records, min_count):
    for facet in FACETS:
        raw = Counter()
        canon = Counter()
        for record in records.values():
            value = record.get("structured_caption", {}).get("attributes", {}).get(facet)
            if isinstance(value, list):
                value = value[0] if value else None
            if value is None or not str(value).strip():
                continue
            raw[normalize(value)] += 1
            canon[canonicalize(facet, value)] += 1
        kept = {k for k, v in canon.items() if v >= min_count}
        folded = sum(v for k, v in canon.items() if k not in kept)
        total = sum(canon.values())
        print(f"\n=== {facet}: {total} labelled products, {len(raw)} raw -> "
              f"{len(canon)} canonical -> {len(kept) + 1} classes (incl. '{OTHER}')")
        print(f"    folded into '{OTHER}': {folded} products "
              f"({folded / max(1, total):.1%}) across {len(canon) - len(kept)} rare classes")
        for name, count in sorted(canon.items(), key=lambda kv: -kv[1])[:14]:
            mark = "" if name in kept else "  -> other"
            print(f"      {name:<20} {count:>5}{mark}")


def run_training(args):
    import torch

    records = load_records()
    index, embeddings = load_embeddings(args.encoder)
    codes = sorted({row["product_code"] for row in index} & set(records))
    train_codes, val_codes, test_codes = product_split(codes, seed=args.seed)
    print(f"encoder={args.encoder}  images={len(index)}  products={len(codes)}  "
          f"split {len(train_codes)}/{len(val_codes)}/{len(test_codes)} (train/val/test, by product)")
    print("NOTE: labels are LLM-generated (caption_apparel.py, which could see product TEXT). "
          "Accuracy below is accuracy at reproducing the captioner, not against reality.\n")

    results = {}
    heads = {}
    for facet in FACETS:
        rows, classes, labels = build_facet_dataset(
            facet, index, embeddings, records, train_codes, args.min_class_count)
        if len(classes) <= 2 or len(rows) < 100:
            print(f"{facet:<16} SKIPPED -- only {len(rows)} labelled images / "
                  f"{len(classes)} classes after bucketing; too sparse to train or trust")
            results[facet] = None
            continue

        def subset(code_set):
            picked = [(i, c, y) for i, c, y in rows if c in code_set]
            x = embeddings[[i for i, _, _ in picked]]
            y = np.array([y for _, _, y in picked], dtype=np.int64)
            return x, y, [c for _, c, _ in picked]

        x_train, y_train, _ = subset(train_codes)
        x_val, y_val, _ = subset(val_codes)
        x_test, y_test, codes_test = subset(test_codes)
        if len(x_test) == 0 or len(x_train) < 50:
            print(f"{facet:<16} SKIPPED -- {len(x_train)} train / {len(x_test)} test images")
            results[facet] = None
            continue

        model, val_acc = train_head(x_train, y_train, x_val, y_val, len(classes),
                                    mlp=args.mlp, epochs=args.epochs, seed=args.seed)
        image_acc, product_acc, image_pred, _ = evaluate_facet(
            model, x_test, y_test, codes_test, classes)

        majority_class = Counter(y_train.tolist()).most_common(1)[0][0]
        majority_acc = float((y_test == majority_class).mean())
        f1 = macro_f1(y_test, image_pred, len(classes))
        other_share = float((y_test == len(classes) - 1).mean())

        results[facet] = dict(classes=classes, n_classes=len(classes),
                              train_images=len(x_train), test_images=len(x_test),
                              test_products=len(set(codes_test)),
                              image_acc=image_acc, product_acc=product_acc,
                              majority_acc=majority_acc, majority_class=classes[majority_class],
                              macro_f1=f1, other_share=other_share, val_acc=val_acc)
        heads[facet] = (model, classes)
        print(f"{facet:<16} classes={len(classes):>3}  test_img={len(x_test):>5}  "
              f"acc={image_acc:.1%}  product-acc={product_acc:.1%}  "
              f"majority={majority_acc:.1%} ({classes[majority_class]})  "
              f"lift={image_acc - majority_acc:+.1%}  macroF1={f1:.3f}  other={other_share:.1%}")

    scored = [(f, r) for f, r in results.items() if r]
    if scored:
        worst = min(scored, key=lambda kv: kv[1]["image_acc"] - kv[1]["majority_acc"])
        print(f"\n=== confusion matrix, worst facet by lift: {worst[0]} "
              f"({worst[1]['image_acc'] - worst[1]['majority_acc']:+.1%} over majority)")
        print_confusion(worst[0], args, records, index, embeddings, train_codes, val_codes, test_codes)

    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        for facet, (model, classes) in heads.items():
            torch.save(model.state_dict(), out / f"{facet}.pt")
            (out / f"{facet}_classes.json").write_text(json.dumps(classes))
        (out / "summary.json").write_text(json.dumps(
            {"encoder": args.encoder, "mlp": args.mlp, "min_class_count": args.min_class_count,
             "results": {k: v for k, v in results.items() if v}}, indent=1))
        print(f"\nsaved heads to {out}")
    return results


def print_confusion(facet, args, records, index, embeddings, train_codes, val_codes, test_codes, top=8):
    rows, classes, _ = build_facet_dataset(facet, index, embeddings, records,
                                           train_codes, args.min_class_count)

    def subset(code_set):
        picked = [(i, c, y) for i, c, y in rows if c in code_set]
        return (embeddings[[i for i, _, _ in picked]],
                np.array([y for _, _, y in picked], dtype=np.int64),
                [c for _, c, _ in picked])

    x_train, y_train, _ = subset(train_codes)
    x_val, y_val, _ = subset(val_codes)
    x_test, y_test, codes_test = subset(test_codes)
    model, _ = train_head(x_train, y_train, x_val, y_val, len(classes),
                          mlp=args.mlp, epochs=args.epochs, seed=args.seed)
    _, _, pred, _ = evaluate_facet(model, x_test, y_test, codes_test, classes)

    counts = Counter(y_test.tolist())
    shown = [c for c, _ in counts.most_common(top)]
    header = "true \\ pred".ljust(18) + "".join(classes[c][:9].rjust(10) for c in shown)
    print(header)
    for t in shown:
        line = classes[t][:17].ljust(18)
        for p in shown:
            line += str(int(((y_test == t) & (pred == p)).sum())).rjust(10)
        total = int((y_test == t).sum())
        hit = int(((y_test == t) & (pred == t)).sum())
        print(line + f"   | {hit}/{total} = {hit / max(1, total):.0%}")


def run_retrieval_eval(args):
    """Does predicting attributes at query time improve RETRIEVAL?

    Baseline is SigLIP2-v3 cosine over product prototypes -- the pipeline's
    shortlist stage alone, no DINOv3 rerank. Absolute R@1 here is therefore
    NOT comparable to any Phase 4 row; only the two arms in this run are
    comparable to each other.
    """
    import torch

    records = load_records()
    index, embeddings = load_embeddings(args.encoder)
    codes = sorted({row["product_code"] for row in index} & set(records))
    train_codes, val_codes, test_codes = product_split(codes, seed=args.seed)

    rows_by_product = defaultdict(list)
    for i, row in enumerate(index):
        if row["product_code"] in records:
            rows_by_product[row["product_code"]].append(i)

    # Train one head per facet on the TRAIN products only, so nothing the head
    # learned came from a query product.
    heads = {}
    facet_labels = {}
    for facet in FACETS:
        rows, classes, labels = build_facet_dataset(
            facet, index, embeddings, records, train_codes, args.min_class_count)
        if len(classes) <= 2 or len(rows) < 100:
            continue

        def subset(code_set):
            picked = [(i, c, y) for i, c, y in rows if c in code_set]
            if not picked:
                return np.zeros((0, embeddings.shape[1]), dtype=np.float32), np.zeros(0, np.int64)
            return (embeddings[[i for i, _, _ in picked]],
                    np.array([y for _, _, y in picked], dtype=np.int64))

        x_train, y_train = subset(train_codes)
        x_val, y_val = subset(val_codes)
        if len(x_train) < 50:
            continue
        model, _ = train_head(x_train, y_train, x_val, y_val, len(classes),
                              mlp=args.mlp, epochs=args.epochs, seed=args.seed)
        heads[facet] = (model, {name: i for i, name in enumerate(classes)})
        facet_labels[facet] = labels
    print(f"heads available for retrieval: {sorted(heads)}")

    # Gallery: every product with >=2 images. Query: one held-out image of a
    # TEST-split product; the product's prototype is the mean of its OTHER
    # images, so the query image itself is never in the gallery vector.
    gallery_codes, prototypes, queries = [], [], []
    for code, idxs in rows_by_product.items():
        if len(idxs) < 2:
            continue
        # Query set = val + test products: every product whose LABELS the heads
        # never saw. Using test alone halves the sample for no extra rigour --
        # val only ever picked the early-stopping epoch.
        if code in test_codes or code in val_codes:
            query_row = idxs[0]
            rest = idxs[1:]
            queries.append((code, query_row))
        else:
            rest = idxs
        vector = embeddings[rest].mean(axis=0)
        vector /= np.linalg.norm(vector) + 1e-9
        gallery_codes.append(code)
        prototypes.append(vector)
    prototypes = np.stack(prototypes)
    code_position = {c: i for i, c in enumerate(gallery_codes)}
    print(f"gallery {len(gallery_codes)} products, {len(queries)} held-out queries "
          f"(test-split products only)")

    query_matrix = embeddings[[r for _, r in queries]]
    similarity = query_matrix @ prototypes.T

    # P(candidate's label | query image), per facet, precomputed for every query.
    facet_logprob = {}
    for facet, (model, class_index) in heads.items():
        with torch.no_grad():
            probabilities = torch.softmax(model(torch.tensor(query_matrix)), dim=1).numpy()
        labels = facet_labels[facet]
        candidate_class = np.full(len(gallery_codes), -1, dtype=np.int64)
        for code, position in code_position.items():
            label = labels.get(code)
            if label is not None:
                candidate_class[position] = class_index.get(label, class_index.get(OTHER, -1))
        logp = np.log(np.clip(probabilities, 1e-6, 1.0))
        matrix = np.zeros((len(queries), len(gallery_codes)), dtype=np.float32)
        known = candidate_class >= 0
        matrix[:, known] = logp[:, candidate_class[known]]
        # Unlabelled candidates get the query's mean log-prob: neither rewarded
        # nor punished for a missing label (punishing them would make the boost
        # a label-coverage prior instead of an attribute signal).
        matrix[:, ~known] = logp.mean(axis=1, keepdims=True)
        facet_logprob[facet] = matrix

    # ORACLE arm: replace the head's prediction with the query product's OWN
    # LLM label, i.e. a head with 100% accuracy against the labels. This bounds
    # the whole idea -- if perfect attribute knowledge does not improve ranking,
    # no better head can, and the negative below is structural rather than a
    # head-quality problem.
    oracle_bonus = {}
    for facet, (model, class_index) in heads.items():
        labels = facet_labels[facet]
        candidate_class = np.full(len(gallery_codes), -1, dtype=np.int64)
        for code, position in code_position.items():
            label = labels.get(code)
            if label is not None:
                candidate_class[position] = class_index.get(label, class_index.get(OTHER, -1))
        query_class = np.array([class_index.get(labels.get(code), -1) for code, _ in queries])
        matrix = np.zeros((len(queries), len(gallery_codes)), dtype=np.float32)
        for i, qc in enumerate(query_class):
            if qc < 0:
                continue              # query product unlabelled for this facet
            matrix[i] = np.where(candidate_class == qc, 1.0,
                                 np.where(candidate_class < 0, 0.5, 0.0))
        oracle_bonus[facet] = matrix

    truth = np.array([code_position[c] for c, _ in queries])

    def recall_at(scores, k):
        order = np.argsort(-scores, axis=1)[:, :k]
        return float(np.mean([truth[i] in order[i] for i in range(len(truth))]))

    baseline_r1 = recall_at(similarity, 1)
    baseline_r5 = recall_at(similarity, 5)
    print(f"\nBASELINE (SigLIP2 {args.encoder} cosine, no attributes): "
          f"R@1 {baseline_r1:.2%}  R@5 {baseline_r5:.2%}   [shortlist stage only -- "
          f"not comparable to a Phase 4 pipeline row]")

    total_logprob = sum(facet_logprob.values()) if facet_logprob else np.zeros_like(similarity)
    # Only rerank the top-K, matching how a rerank would actually be deployed.
    topk = min(args.rerank_k, similarity.shape[1])
    shortlist = np.argsort(-similarity, axis=1)[:, :topk]
    print(f"shortlist coverage @K={topk}: "
          f"{np.mean([truth[i] in shortlist[i] for i in range(len(truth))]):.2%}")

    mask = np.zeros_like(similarity, dtype=bool)
    for i in range(len(truth)):
        mask[i, shortlist[i]] = True

    # Predicted-argmax agreement, as an alternative to log-probability: a flat
    # bonus when the head's top prediction equals the candidate's label. Scale-
    # free, so it cannot be dismissed as a log-prob calibration artefact.
    agreement = np.zeros_like(similarity)
    for facet, (model, class_index) in heads.items():
        with torch.no_grad():
            predicted = torch.softmax(model(torch.tensor(query_matrix)), dim=1).numpy().argmax(axis=1)
        labels = facet_labels[facet]
        candidate_class = np.array(
            [class_index.get(labels.get(c), -1) for c in gallery_codes], dtype=np.int64)
        agreement += (predicted[:, None] == candidate_class[None, :]).astype(np.float32)

    arms = {
        "head log-prob": total_logprob,
        "head argmax agreement": agreement,
        "ORACLE (true LLM labels)": sum(oracle_bonus.values()) if oracle_bonus else np.zeros_like(similarity),
    }
    best_overall = (None, 0.0, baseline_r1)
    for name, bonus in arms.items():
        print(f"\n--- {name}")
        print(f"{'weight':>8} {'R@1':>9} {'delta':>9} {'R@5':>9}")
        print(f"{0.0:>8.4f} {baseline_r1:>8.2%} {'--':>9} {baseline_r5:>8.2%}")
        for weight in args.weights:
            scores = np.where(mask, similarity + weight * bonus, -np.inf)
            r1, r5 = recall_at(scores, 1), recall_at(scores, 5)
            print(f"{weight:>8.4f} {r1:>8.2%} {r1 - baseline_r1:>+8.2%} {r5:>8.2%}")
            if r1 > best_overall[2]:
                best_overall = (name, weight, r1)
    if best_overall[0] is None:
        print(f"\nNo arm at any weight beat the baseline R@1 of {baseline_r1:.2%}.")
    else:
        print(f"\nbest: {best_overall[0]} @ weight {best_overall[1]} -> R@1 {best_overall[2]:.2%} "
              f"({best_overall[2] - baseline_r1:+.2%} vs baseline)")
    print("Reminder: the candidate-side labels are the same LLM captions the heads were "
          "trained on, so this measures agreement between a head and a captioner, "
          "not attribute correctness.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-buckets", action="store_true",
                        help="print the long-tail bucketing decisions and stop")
    parser.add_argument("--train", action="store_true", help="train + evaluate heads")
    parser.add_argument("--retrieval-eval", action="store_true",
                        help="measure whether query-time attributes improve R@1")
    parser.add_argument("--encoder", default="v3", choices=["v3", "base"],
                        help="v3 = the checkpoint the real shortlist runs; base = frozen pretrained")
    parser.add_argument("--mlp", action="store_true", help="1-hidden-layer head instead of linear")
    parser.add_argument("--min-class-count", type=int, default=15,
                        help="canonical classes rarer than this fold into 'other'")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save", default="", help="directory to write trained heads to")
    parser.add_argument("--rerank-k", type=int, default=50)
    parser.add_argument("--weights", type=float, nargs="*",
                        default=[0.001, 0.003, 0.01, 0.03, 0.1])
    args = parser.parse_args()

    if args.report_buckets:
        report_buckets(load_records(), args.min_class_count)
        return
    if args.train:
        run_training(args)
    if args.retrieval_eval:
        run_retrieval_eval(args)
    if not (args.train or args.retrieval_eval):
        parser.print_help()


if __name__ == "__main__":
    main()
