"""Triage negatives_dataset/ for the one thing that would poison it: clothing.

A negative containing a prominent garment is not noise, it is a mislabelled
positive, and it drags the gate's threshold in exactly the direction that makes
the gate reject real clothes. Search terms are not enough of a guarantee --
"empty street road" returns pedestrians, "office interior" returns people at
desks -- so every negative gets looked at, in an order that puts the likely
offenders first.

Two things this deliberately does NOT do:

It does not delete anything on its own. The ranking model is SigLIP2, the same
family as the gate being calibrated; letting it choose which negatives survive
would quietly delete its own hard cases and inflate the AUROC this whole
exercise exists to deflate. It prints a queue; a human drops ids with --drop.

It does not rank by the gate's own garment margin, for the same reason. It
ranks by a person/clothing-presence probe, which is a different question
("is someone in this frame") from the one being calibrated ("is this a photo
of a garment"). Reviewing in that order still means a garment with no person
in it -- a shirt on a hanger in a shop window -- can sit low in the queue, so
--sample also draws a random control block to check what the ranking misses.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from evaluate_garment_gate import (
    MODEL_ID,
    NEGATIVES_METADATA,
    REPO,
    embed_texts,
    extract_embeddings,
    load_model,
)

PERSON_PROMPTS = [
    "a photo of a person wearing clothes",
    "a photo of people",
    "a portrait of a person",
    "a person standing in the frame",
    "a close-up of someone's clothing",
]

NO_PERSON_PROMPTS = [
    "a photo of an empty scene with no people",
    "a photo of an object on its own",
    "a landscape with nobody in it",
    "an empty room",
]


@torch.no_grad()
def rank(records, model, processor, device, batch_size=16):
    person_text = embed_texts(model, processor, PERSON_PROMPTS, device)
    empty_text = embed_texts(model, processor, NO_PERSON_PROMPTS, device)

    scored = []
    for start in range(0, len(records), batch_size):
        batch, images = [], []
        for record in records[start:start + batch_size]:
            try:
                images.append(Image.open(REPO / record["path"]).convert("RGB"))
                batch.append(record)
            except Exception:
                continue
        if not images:
            continue
        inputs = processor(images=images, return_tensors="pt").to(device)
        features = extract_embeddings(model.get_image_features(**inputs))
        features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
        margin = (features @ person_text.T).max(dim=1).values \
            - (features @ empty_text.T).max(dim=1).values
        scored.extend(zip(batch, margin.float().cpu().tolist()))
        print(f"  ranked {len(scored)}/{len(records)}", end="\r")
    print()
    scored.sort(key=lambda pair: -pair[1])
    return scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=40,
                        help="how many likeliest-offender rows to print")
    parser.add_argument("--sample", type=int, default=20,
                        help="random control rows, to catch what the ranking misses")
    parser.add_argument("--drop", default="",
                        help="comma-separated source_ids to delete (file + record)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = json.loads(NEGATIVES_METADATA.read_text())

    if args.drop:
        drop = {value.strip() for value in args.drop.split(",") if value.strip()}
        kept = []
        removed = 0
        for record in records:
            if record["source_id"] in drop:
                (REPO / record["path"]).unlink(missing_ok=True)
                removed += 1
                continue
            kept.append(record)
        NEGATIVES_METADATA.write_text(json.dumps(kept, indent=2))
        print(f"dropped {removed} records, {len(kept)} remain")
        return

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    model, processor = load_model(device)
    scored = rank(records, model, processor, device)

    print(f"\n=== {args.top} most likely to contain a person / clothing ===")
    for record, score in scored[:args.top]:
        print(f"  {score:+.4f}  {record['theme']:<12} {record['source_id']:<12} "
              f"{record['path']}  {record['title'][:60]}")

    rng = random.Random(args.seed)
    control = rng.sample(scored, min(args.sample, len(scored)))
    print(f"\n=== {len(control)} random control rows ===")
    for record, score in control:
        print(f"  {score:+.4f}  {record['theme']:<12} {record['source_id']:<12} "
              f"{record['path']}")

    values = np.asarray([score for _, score in scored])
    print(f"\nperson-probe margin: mean={values.mean():+.4f} "
          f"max={values.max():+.4f} min={values.min():+.4f}; "
          f"{int((values > 0).sum())} of {len(values)} score above zero")


if __name__ == "__main__":
    main()
