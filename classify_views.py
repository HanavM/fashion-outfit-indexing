"""
Zero-shot classify every scraped apparel image by camera angle/view, using
the same Marqo FashionSigLIP model already used for the search index. Goal:
tag which images resemble a realistic "outfit photo" shot (side/front/angled
views of the whole item) vs. views that would never appear in a real photo
of someone wearing it (sole, insole, extreme material close-ups, packaging).

Runs in Colab against Google Drive, not local disk -- mount Drive first.

Does NOT delete or move any images — writes a JSON report per image with
the predicted view label + a keep/exclude flag, so filtering stays a
reversible, inspectable decision.

Usage:
    python classify_views.py --limit 40   # quick test sample across brands
    python classify_views.py              # full run over apparel_dataset/
"""

import argparse, json
from pathlib import Path

import torch
from PIL import Image

DATASET_DIR = Path("/content/drive/MyDrive/apparel_dataset")
OUT_FILE = DATASET_DIR / "image_views.json"
MODEL_NAME = "hf-hub:Marqo/marqo-fashionSigLIP"
BATCH_SIZE = 32

# (label, prompt, keep_in_dataset)
CANDIDATES = [
    ("hero / main product shot", "a hero product photo of a full shoe", True),
    ("side profile view", "a side profile view photo of a full shoe", True),
    ("front three-quarter view", "a front three-quarter angle photo of a full shoe", True),
    ("back / heel view", "a back heel view photo of a full shoe", True),
    ("top-down view", "a top-down view photo of a full shoe", True),
    ("on-foot / lifestyle shot", "a photo of a shoe being worn on a foot", True),
    ("sole / bottom view", "a photo of the bottom sole of a shoe", False),
    ("insole / inside view", "a close-up photo of a shoe's insole from above", False),
    ("material close-up", "an extreme close-up photo of shoe fabric or stitching texture", False),
    ("packaging / box", "a photo of a shoe box or packaging", False),
]


def load_model():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}")
    return model, preprocess, device


def resolve_image_path(raw_path: str) -> Path | None:
    """
    metadata.json's `images` entries carry inconsistent prefixes -- some
    products still have a stale "shoe_dataset/..." prefix baked in from
    before the apparel_dataset rename, others have the correct
    "apparel_dataset/..." prefix. Real files always live at
    DATASET_DIR/<brand>/<slug>/<product_code>/image_N.jpg, so reconstruct
    from the last 4 path components regardless of what prefix is present.
    """
    raw_path = Path(raw_path)

    candidates = [raw_path, DATASET_DIR / raw_path]

    if raw_path.parts and raw_path.parts[0] == DATASET_DIR.name:
        candidates.append(DATASET_DIR.joinpath(*raw_path.parts[1:]))

    if len(raw_path.parts) >= 4:
        candidates.append(DATASET_DIR.joinpath(*raw_path.parts[-4:]))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return None


def collect_image_paths(limit: int | None) -> list[str]:
    meta = json.loads((DATASET_DIR / "metadata.json").read_text())

    def resolved(raw_images):
        out = []
        for raw in raw_images:
            resolved_path = resolve_image_path(raw)
            if resolved_path is not None:
                out.append(str(resolved_path))
        return out

    if limit:
        # spread the sample across brands rather than just the first N
        by_brand: dict[str, list[str]] = {}
        for r in meta:
            by_brand.setdefault(r["brand"], []).extend(resolved(r["images"]))
        paths = []
        per_brand = max(1, limit // len(by_brand))
        for imgs in by_brand.values():
            paths.extend(imgs[:per_brand])
        paths = paths[:limit]
    else:
        paths = [p for r in meta for p in resolved(r["images"])]

    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    paths = collect_image_paths(args.limit)
    print(f"Classifying {len(paths)} images...")

    model, preprocess, device = load_model()
    import open_clip
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    labels = [c[0] for c in CANDIDATES]
    prompts = [c[1] for c in CANDIDATES]
    keep_flags = {c[0]: c[2] for c in CANDIDATES}

    text_tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    results = {}
    for batch_start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[batch_start:batch_start + BATCH_SIZE]
        tensors, valid_paths = [], []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(preprocess(img))
                valid_paths.append(p)
            except Exception as e:
                print(f"  [warn] skip {p}: {e}")

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            img_feats = model.encode_image(batch_tensor)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            sims = img_feats @ text_feats.T
            probs = sims.softmax(dim=-1)

        for i, p in enumerate(valid_paths):
            top_idx = probs[i].argmax().item()
            label = labels[top_idx]
            results[p] = {
                "view": label,
                "confidence": round(probs[i][top_idx].item(), 4),
                "keep": keep_flags[label],
            }

        done = min(batch_start + BATCH_SIZE, len(paths))
        print(f"  [{done}/{len(paths)}] classified")

    OUT_FILE.write_text(json.dumps(results, indent=2))

    from collections import Counter
    dist = Counter(r["view"] for r in results.values())
    print(f"\nDone. {len(results)} images classified.")
    print("Distribution:")
    for label, count in dist.most_common():
        flag = "KEEP" if keep_flags[label] else "EXCLUDE"
        print(f"  [{flag:7}] {label}: {count}")
    keep_n = sum(1 for r in results.values() if r["keep"])
    print(f"\nWould keep {keep_n}/{len(results)} images ({keep_n/len(results)*100:.1f}%)")
    print(f"Report -> {OUT_FILE}")


if __name__ == "__main__":
    main()
