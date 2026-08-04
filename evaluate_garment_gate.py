"""Calibrate the SigLIP2 zero-shot garment gate against REAL negatives.

The gate is the thing that stands between the Siri surface and the finding in
docs/eval_log.md's open-set row: pointed at a chair, the retrieval pipeline
confidently names a jacket ~68% of the time at any usable false-reject rate,
and no threshold on the DINOv3 score fixes it. The conclusion there was that
the fix has to be upstream -- refuse to query at all unless the photo contains
a garment.

That gate had only ever been scored against SYNTHETIC negatives (bar charts,
solid colour fields, blocks of text), which gives AUROC 1.0000. That number is
real but it is not the number the gate ships on: it measures how easily SigLIP2
tells a photograph from a chart. The deployed failure mode is a photograph of a
sofa, and a sofa is a photograph. So this scores the same gate against
`negatives_dataset/` -- real photos of furniture, cars, food, landscapes,
buildings, interiors, electronics, animals, plants, books, kitchenware, tools,
streets and screenshots -- and reports the two-sided operating table, in the
same shape as the open-set rows, because a single AUROC cannot tell you what
you give up to get what.

`--negatives synthetic` reproduces the old number in this same script, so the
optimism is measurable rather than asserted.

Score, per image:
    max cosine over GARMENT_PROMPTS  -  max cosine over NON_GARMENT_PROMPTS
A margin rather than a bare garment similarity, because raw cosine to any one
prompt drifts with image statistics; the difference is what a gate can threshold.

Positives are drawn from BOTH halves of the real distribution the gate will
see: `apparel_dataset/` product photos (clean studio, easy) and
`outfit_dataset/` worn outfits (cluttered, whole-body, hard). Reporting them
separately is not decoration -- they are the two ends of the difficulty range
and the threshold lives between them.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor


# roc_auc / open_set_threshold_table are copied from
# hierarchical_retrieval_pipeline.py rather than imported. That module is
# deliberately standalone (it is shipped as a single file to Colab and Modal,
# which is also why its AUROC is hand-written instead of pulled from sklearn),
# and importing it here fails outright: its module level touches
# /content/drive/MyDrive. Keeping the two error rates computed the SAME way as
# the open-set rows matters more than avoiding forty duplicated lines, because
# the whole point is that these tables are read side by side.

def roc_auc(positive_scores, negative_scores):
    """Rank-based (Mann-Whitney U) AUROC with average-rank tie handling.

    Interpretation here: probability that a randomly chosen real garment photo
    scores higher than a randomly chosen non-garment photo.
    """
    positives = np.asarray(positive_scores, dtype=float)
    negatives = np.asarray(negative_scores, dtype=float)
    if len(positives) == 0 or len(negatives) == 0:
        return None

    combined = np.concatenate([positives, negatives])
    order = combined.argsort(kind="mergesort")
    ranks = np.empty(len(combined), dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=float)

    sorted_values = combined[order]
    start = 0
    while start < len(sorted_values):
        stop = start
        while (stop + 1 < len(sorted_values)
               and sorted_values[stop + 1] == sorted_values[start]):
            stop += 1
        if stop > start:
            ranks[order[start:stop + 1]] = (start + 1 + stop + 1) / 2.0
        start = stop + 1

    positive_rank_sum = ranks[:len(positives)].sum()
    return float((positive_rank_sum - len(positives) * (len(positives) + 1) / 2.0)
                 / (len(positives) * len(negatives)))


def _threshold_row(threshold, positive_scores, negative_scores):
    return {
        "threshold": float(threshold),
        # Real garment photos this threshold would refuse to query at all.
        "false_reject_rate": float(np.mean(positive_scores < threshold)),
        # Non-garment photos this threshold would let through to the pipeline,
        # which will then confidently name a product for a sofa.
        "false_accept_rate": float(np.mean(negative_scores >= threshold)),
    }


def open_set_threshold_table(positive_scores, negative_scores):
    """For each candidate threshold, BOTH error rates it buys. Percentile rows
    are anchored on the positive distribution; best_balanced maximises
    Youden's J over every observed score."""
    rows = {}
    for percentile in (1, 2, 5, 10, 20, 30, 50):
        rows[f"p{percentile}"] = _threshold_row(
            np.percentile(positive_scores, percentile), positive_scores, negative_scores)

    grid = np.unique(np.concatenate([positive_scores, negative_scores]))
    best = min(grid, key=lambda t: np.mean(positive_scores < t)
               + np.mean(negative_scores >= t))
    rows["best_balanced"] = _threshold_row(best, positive_scores, negative_scores)
    return rows

REPO = Path(__file__).resolve().parent
APPAREL_METADATA = REPO / "apparel_dataset" / "metadata.json"
OUTFIT_METADATA = REPO / "outfit_dataset" / "metadata.json"
NEGATIVES_METADATA = REPO / "negatives_dataset" / "metadata.json"

# Base, not the fine-tuned checkpoint: the gate has to work on images the
# fine-tune never saw (a sofa is not in any fashion training set), and a
# fashion-specialised encoder is exactly the wrong tool for deciding whether
# something is fashion at all.
MODEL_ID = "google/siglip2-base-patch16-384"
BATCH_SIZE = 16

GARMENT_PROMPTS = [
    "a photo of clothing",
    "a photo of a garment",
    "a photo of a shirt",
    "a photo of a jacket",
    "a photo of trousers",
    "a photo of a dress",
    "a photo of shoes",
    "a person wearing an outfit",
    "a fashion product photo of an item of clothing",
    "a flat lay of a piece of clothing",
]

NON_GARMENT_PROMPTS = [
    "a photo of furniture",
    "a photo of a vehicle",
    "a photo of food",
    "a photo of a landscape",
    "a photo of a building",
    "a photo of an electronic device",
    "a photo of an animal",
    "a photo of a plant",
    "a photo of a book or document",
    "a photo of kitchenware",
    "a photo of tools",
    "a screenshot of a computer screen",
    "a photo of an object that is not clothing",
]


def load_model(device):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    return model, processor


def extract_embeddings(output):
    """SigLIP2's get_*_features returns a bare tensor on some transformers
    versions and a ModelOutput on others; same shape either way. Mirrors
    embed_catalog_siglip2.extract_embeddings (not imported -- that module pulls
    in pandas, which this environment does not have)."""
    if torch.is_tensor(output):
        return output
    for attribute in ("text_embeds", "image_embeds", "pooler_output"):
        value = getattr(output, attribute, None)
        if value is not None:
            return value
    raise TypeError(f"Could not extract embeddings from {type(output)}")


@torch.no_grad()
def embed_texts(model, processor, texts, device):
    inputs = processor(text=texts, padding="max_length", truncation=True,
                       max_length=64, return_tensors="pt").to(device)
    features = extract_embeddings(model.get_text_features(**inputs))
    return torch.nn.functional.normalize(features.float(), p=2, dim=-1)


@torch.no_grad()
def score_images(model, processor, paths, garment_text, other_text, device):
    """Margin score per image; returns (scores, kept_paths).

    Unreadable files are dropped rather than scored as zero -- a decode failure
    is not evidence about the gate, and folding it in as a mid-range score
    would quietly move the threshold.
    """
    scores, kept = [], []
    for start in range(0, len(paths), BATCH_SIZE):
        batch_paths, images = [], []
        for path in paths[start:start + BATCH_SIZE]:
            try:
                images.append(Image.open(path).convert("RGB"))
                batch_paths.append(path)
            except Exception:
                continue
        if not images:
            continue
        inputs = processor(images=images, return_tensors="pt").to(device)
        features = extract_embeddings(model.get_image_features(**inputs))
        features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
        margin = (features @ garment_text.T).max(dim=1).values \
            - (features @ other_text.T).max(dim=1).values
        scores.extend(margin.float().cpu().tolist())
        kept.extend(batch_paths)
        print(f"    scored {len(kept)}/{len(paths)}", end="\r")
    print()
    return np.asarray(scores, dtype=float), kept


def sample_apparel(limit, rng):
    records = json.loads(APPAREL_METADATA.read_text())
    # One image per product, not one per file: products carry up to 18 near
    # duplicate views, and letting a handful of products dominate the positive
    # set would make the threshold a statement about those products.
    paths = []
    for record in records:
        on_disk = [p for p in record.get("images", []) if (REPO / p).is_file()]
        if on_disk:
            paths.append(str(REPO / on_disk[0]))
    rng.shuffle(paths)
    return paths[:limit]


def sample_outfits(limit, rng):
    records = json.loads(OUTFIT_METADATA.read_text())
    paths = []
    for record in records:
        on_disk = [p for p in record.get("images", []) if (REPO / p).is_file()]
        if on_disk:
            paths.append(str(REPO / on_disk[0]))
    rng.shuffle(paths)
    return paths[:limit]


def load_negatives():
    records = json.loads(NEGATIVES_METADATA.read_text())
    paths, themes = [], []
    for record in records:
        path = REPO / record["path"]
        if path.is_file():
            paths.append(str(path))
            themes.append(record["theme"])
    return paths, themes


def make_synthetic(count, out_dir, rng):
    """The old negative set: charts, solid colours, text blocks.

    Kept in the same script as the real negatives specifically so the two can
    be run back to back and the gap between them stated as a measurement.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        kind = index % 3
        image = Image.new("RGB", (512, 512), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        if kind == 0:  # bar chart
            for bar in range(8):
                height = rng.randint(40, 440)
                draw.rectangle([30 + bar * 58, 480 - height, 78 + bar * 58, 480],
                               fill=(rng.randint(0, 200), rng.randint(0, 200), rng.randint(0, 200)))
            draw.line([20, 480, 500, 480], fill=(0, 0, 0), width=3)
        elif kind == 1:  # solid colour field
            image = Image.new("RGB", (512, 512),
                              (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
        else:  # block of text
            for line in range(20):
                draw.text((20, 20 + line * 24),
                          " ".join(rng.choice(["lorem", "ipsum", "dolor", "sit", "amet",
                                               "consectetur", "adipiscing", "elit"])
                                   for _ in range(9)),
                          fill=(20, 20, 20))
        path = out_dir / f"synthetic_{index:04d}.png"
        image.save(path)
        paths.append(str(path))
    return paths


def describe(name, scores):
    if len(scores) == 0:
        return f"  {name:<22} (empty)"
    return (f"  {name:<22} n={len(scores):<5} mean={scores.mean():+.4f}  "
            f"min={scores.min():+.4f}  max={scores.max():+.4f}  "
            f"median={np.median(scores):+.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negatives", choices=["real", "synthetic"], default="real")
    parser.add_argument("--apparel-samples", type=int, default=300)
    parser.add_argument("--outfit-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}  model={MODEL_ID}  negatives={args.negatives}")

    model, processor = load_model(device)
    garment_text = embed_texts(model, processor, GARMENT_PROMPTS, device)
    other_text = embed_texts(model, processor, NON_GARMENT_PROMPTS, device)

    apparel_paths = sample_apparel(args.apparel_samples, rng)
    outfit_paths = sample_outfits(args.outfit_samples, rng)
    if args.negatives == "real":
        negative_paths, negative_themes = load_negatives()
    else:
        negative_paths = make_synthetic(
            400, REPO / "negatives_dataset" / "_synthetic", rng)
        negative_themes = ["synthetic"] * len(negative_paths)

    print(f"positives: {len(apparel_paths)} apparel + {len(outfit_paths)} outfits;"
          f" negatives: {len(negative_paths)}")

    print("  apparel...")
    apparel_scores, _ = score_images(model, processor, apparel_paths,
                                     garment_text, other_text, device)
    print("  outfits...")
    outfit_scores, _ = score_images(model, processor, outfit_paths,
                                    garment_text, other_text, device)
    print("  negatives...")
    negative_scores, negative_kept = score_images(model, processor, negative_paths,
                                                  garment_text, other_text, device)
    theme_by_path = dict(zip(negative_paths, negative_themes))

    positive_scores = np.concatenate([apparel_scores, outfit_scores])
    auroc = roc_auc(positive_scores, negative_scores)
    table = open_set_threshold_table(positive_scores, negative_scores)

    print("\n=== score distributions (garment margin) ===")
    print(describe("apparel (product)", apparel_scores))
    print(describe("outfit (worn)", outfit_scores))
    print(describe("ALL POSITIVES", positive_scores))
    print(describe(f"negatives ({args.negatives})", negative_scores))

    print("\n=== per-theme negatives (hardest last) ===")
    per_theme = {}
    for path, score in zip(negative_kept, negative_scores):
        per_theme.setdefault(theme_by_path[path], []).append(score)
    for theme, values in sorted(per_theme.items(), key=lambda kv: np.mean(kv[1])):
        values = np.asarray(values)
        print(f"  {theme:<14} n={len(values):<4} mean={values.mean():+.4f}  "
              f"max={values.max():+.4f}")

    print(f"\n=== AUROC (positives vs negatives): {auroc:.4f} ===")
    print(f"{'row':<15}{'threshold':>12}{'false-reject':>15}{'false-accept':>15}")
    for name, row in table.items():
        print(f"{name:<15}{row['threshold']:>12.4f}"
              f"{row['false_reject_rate']*100:>14.2f}%"
              f"{row['false_accept_rate']*100:>14.2f}%")

    # Overlap is the headline for a gate, so state it rather than leaving it to
    # be inferred from the table: how many worlds does each side share.
    overlap_positives = float(np.mean(positive_scores < negative_scores.max()))
    overlap_negatives = float(np.mean(negative_scores > positive_scores.min()))
    print(f"\npositives below the highest negative: {overlap_positives*100:.2f}%")
    print(f"negatives above the lowest positive:  {overlap_negatives*100:.2f}%")

    if args.output:
        Path(args.output).write_text(json.dumps({
            "model": MODEL_ID,
            "negatives": args.negatives,
            "auroc": auroc,
            "counts": {"apparel": len(apparel_scores), "outfit": len(outfit_scores),
                       "negatives": len(negative_scores)},
            "means": {"apparel": float(apparel_scores.mean()),
                      "outfit": float(outfit_scores.mean()),
                      "negatives": float(negative_scores.mean())},
            "per_theme_mean": {t: float(np.mean(v)) for t, v in per_theme.items()},
            "threshold_table": table,
        }, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
