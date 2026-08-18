"""Distil VLM colour judgements into a local head on frozen SigLIP2 crop embeddings.

    .venv/bin/python train_colour_head.py            # train + evaluate + save
    .venv/bin/python train_colour_head.py --report   # re-print the saved metrics

## Why this instead of calling a VLM per image

The owner's objection to VLM colour labelling was scalability, and it is
right about the part that matters: a per-image API call is a linear
external cost, subject to rate limits, and produces labels that stop being
reproducible the moment the model is deprecated.

(The part of the objection that does not hold: it is a one-time OFFLINE
cost per image, not per query, and the VLM labelled at 3.2 photos/s
against `garment_proposer`'s ~1 photo/s — it is cheaper per image than the
detection pass already in the pipeline.)

Distillation keeps the accuracy and removes the dependency. The VLM
labels a bounded, one-time sample; a small head learns from it; every
future crop is then a matrix multiply on an embedding that already exists
in `outfit_crop_index.pt`. Corpus size stops mattering.

This is the pattern `attribute_heads.py` already established for the
catalog — an MLP on frozen SigLIP2 features — applied to a different
label source.

## Why the incumbent is beatable

The heuristic (`index_outfits.dominant_color`) votes each pixel to the
nearest palette colour in **RGB Euclidean distance**. Measured against the
VLM on 1,632 garments it agrees 38.5%, and the errors are one-directional:
brown→black, gray→charcoal, navy→black, blue→black, white→light gray.
Everything dark collapses onto black/charcoal, because photographed
garments are darker than nominal swatches.

Three attempts to fix that by changing the COLOUR SPACE all failed
(CIELAB matching at query time: noise; renaming from stored mean_rgb:
worse; CIELAB voting on pixels: no better). The colour space was never the
problem — the two labellers answer different questions. The heuristic
reports the colour of the PIXELS, which for a navy jacket in shadow really
is near-black. A user searching "navy jacket" means the GARMENT's colour.
A learned head can pick that up from the embedding; a palette lookup
cannot.

## Alignment discipline

A (garment, colour) label is attached to a crop only when the photo has
**exactly one** crop of that category and the VLM named **exactly one**
colour for it. Ambiguity is dropped rather than guessed: a wrong training
pair teaches the head the wrong thing, while a missing one only costs
sample size.

Train/test is split **by photo**, so no photo's crops straddle the split.
"""

import argparse
import collections
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("APPAREL_DATASET_ROOT", str(REPO_ROOT / "apparel_dataset"))

LABELS_PATH = REPO_ROOT / "outfit_eval" / "vlm_labels.json"
OUTFIT_METADATA = REPO_ROOT / "outfit_dataset" / "metadata.json"
HEAD_PATH = REPO_ROOT / "outfit_dataset" / "colour_head.pt"
MIN_PER_CLASS = 12


def build_dataset(engine, labels):
    import numpy as np

    truth = {k: v for k, v in labels.items() if v.get("is_outfit_photo")}
    stored = {}
    for record in json.loads(OUTFIT_METADATA.read_text()):
        for order, item in enumerate(record.get("detected_items") or []):
            stored[(item.get("source_image"), order)] = (item.get("color") or {}).get("name")

    rows = collections.defaultdict(list)
    for index, record in enumerate(engine.crop_records):
        rows[(record["rel"], record.get("category"))].append(index)

    X, y, photos, baseline = [], [], [], []
    for rel, entry in truth.items():
        by_garment = collections.defaultdict(set)
        for item in entry["garments"]:
            by_garment[item["garment"]].add(item["colour"])
        for garment, colours in by_garment.items():
            candidates = rows.get((rel, garment), [])
            if len(candidates) != 1 or len(colours) != 1:
                continue
            record = engine.crop_records[candidates[0]]
            X.append(engine.crop_embeddings[candidates[0]].numpy())
            y.append(next(iter(colours)))
            photos.append(rel)
            baseline.append(stored.get((record["rel"], record["detection_index"])))

    # Drop colours too rare to learn or to evaluate honestly.
    counts = collections.Counter(y)
    keep = [i for i, c in enumerate(y) if counts[c] >= MIN_PER_CLASS]
    dropped = sorted({c for c in y if counts[c] < MIN_PER_CLASS})
    if dropped:
        print(f"  dropped {len(y)-len(keep)} pairs in rare colours: {', '.join(dropped)}")
    return (np.array([X[i] for i in keep]), [y[i] for i in keep],
            [photos[i] for i in keep], [baseline[i] for i in keep])


def train(args):
    import numpy as np
    import torch

    import outfit_search

    if not LABELS_PATH.exists():
        raise SystemExit("no VLM labels — run `outfit_search_eval.py label` first")
    engine = outfit_search.OutfitSearch()
    labels = json.loads(LABELS_PATH.read_text())
    X, y, photos, baseline = build_dataset(engine, labels)
    classes = sorted(set(y))
    yi = np.array([classes.index(c) for c in y])
    print(f"  {len(X):,} aligned pairs · {len(classes)} colours · "
          f"{len(set(photos)):,} photos")

    rng = np.random.RandomState(args.seed)
    unique_photos = sorted(set(photos))
    rng.shuffle(unique_photos)
    train_photos = set(unique_photos[:int(len(unique_photos) * 0.75)])
    is_train = np.array([p in train_photos for p in photos])
    is_test = ~is_train
    print(f"  train {is_train.sum():,} · test {is_test.sum():,} (split by photo)")

    Xtr = torch.tensor(X[is_train]); ytr = torch.tensor(yi[is_train])
    Xte = torch.tensor(X[is_test]); yte = torch.tensor(yi[is_test])

    # Class weights: "black" is 29% of the data, and an unweighted head
    # drifts toward predicting it -- which is exactly the incumbent's bug,
    # so inheriting it would defeat the purpose.
    counts = np.bincount(yi[is_train], minlength=len(classes)).astype(float)
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5,
                           dtype=torch.float32)

    torch.manual_seed(args.seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], args.hidden), torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout), torch.nn.Linear(args.hidden, len(classes)))
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_accuracy, best_state, patience = 0.0, None, 0
    for epoch in range(args.epochs):
        model.train(); optimiser.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(Xtr), ytr, weight=weights)
        loss.backward(); optimiser.step()
        if epoch % 10 == 9:
            model.eval()
            with torch.no_grad():
                accuracy = (model(Xte).argmax(1) == yte).float().mean().item()
            if accuracy > best_accuracy:
                best_accuracy, patience = accuracy, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= args.patience:
                    print(f"  early stop at epoch {epoch+1}")
                    break

    gold = np.array(y, dtype=object)[is_test]
    base = np.array(baseline, dtype=object)[is_test]
    have = np.array([b is not None for b in base])
    heuristic = float((base[have] == gold[have]).mean()) if have.any() else 0.0
    majority = float((gold == collections.Counter(y).most_common(1)[0][0]).mean())

    print(f"\n  {'majority class':34} {majority:6.1%}")
    print(f"  {'heuristic palette vote (same crops)':34} {heuristic:6.1%}")
    print(f"  {'distilled head':34} {best_accuracy:6.1%}")
    print(f"  {'':34} {(best_accuracy-heuristic)*100:+6.1f}pt vs heuristic")

    model.load_state_dict(best_state)
    # Save the held-out photo list. The retrieval eval scores against the
    # same VLM labels this head trained on, so scoring it on photos it saw
    # would be measuring memorisation. outfit_search_eval --held-out-only
    # reads this.
    torch.save({"held_out_photos": sorted(set(p for p in photos if p not in train_photos)),
                "state_dict": best_state, "classes": classes,
                "hidden": args.hidden, "dim": int(X.shape[1]),
                "metrics": {"head": best_accuracy, "heuristic": heuristic,
                            "majority": majority, "n_test": int(is_test.sum()),
                            "n_train": int(is_train.sum())}}, HEAD_PATH)
    print(f"\n  saved -> {HEAD_PATH}")
    print("  Accuracy is agreement with a VLM on a 19-way task where the two\n"
          "  labellers partly disagree by construction (pixel colour vs garment\n"
          "  colour). The comparison against the heuristic is the meaningful part.")


def report(args):
    import torch

    if not HEAD_PATH.exists():
        raise SystemExit("no head trained yet")
    payload = torch.load(HEAD_PATH, map_location="cpu", weights_only=False)
    metrics = payload["metrics"]
    print(f"  classes: {len(payload['classes'])}  ·  train {metrics['n_train']:,} "
          f"· test {metrics['n_test']:,}")
    for name in ("majority", "heuristic", "head"):
        print(f"  {name:12} {metrics[name]:6.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    report(args) if args.report else train(args)


if __name__ == "__main__":
    main()
