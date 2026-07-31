"""One-off empirical test (not a permanent pipeline component): does
free-text query -> SigLIP2 text embedding -> cosine similarity against
cached image embeddings work for genuinely novel, localized/compositional
queries never seen verbatim in training? And does the v3 fine-tune's
catastrophic forgetting risk (partial unfreezing of last 4 blocks) show up
against the base model on this specific task?

Test cases: real products from the catalog whose defining_features mention
a rare, spatially localized structural detail. Queries are natural
paraphrases, NOT verbatim copies of the training label text, to simulate
a genuinely novel user prompt.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

DATASET_ROOT = Path("apparel_dataset_full")
METADATA_PATH = DATASET_ROOT / "metadata.json"
BASE_MODEL_ID = "google/siglip2-base-patch16-384"
V3_CHECKPOINT = "/tmp/v3_checkpoint/stage2_lastnblocks_best"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

TEST_CASES = [
    {"code": "0120611140009", "query": "a jersey with embroidered lettering on the back"},
    {"code": "1180436002", "query": "a denim jacket with button tabs near the bottom back hem"},
    {"code": "M990GL6", "query": "a running shoe with a plastic tab at the back of the heel"},
]


def resolve_image_path(raw_path):
    raw_path = Path(raw_path)
    candidates = [raw_path, DATASET_ROOT / raw_path, DATASET_ROOT.parent / raw_path]
    if raw_path.parts and raw_path.parts[0] == DATASET_ROOT.name:
        candidates.append(DATASET_ROOT.joinpath(*raw_path.parts[1:]))
    if len(raw_path.parts) >= 4:
        candidates.append(DATASET_ROOT.joinpath(*raw_path.parts[-4:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def extract_embeddings(output):
    if torch.is_tensor(output):
        return output
    for attribute in ("text_embeds", "image_embeds", "pooler_output"):
        value = getattr(output, attribute, None)
        if value is not None:
            return value
    raise TypeError(f"Cannot extract embeddings from {type(output)}")


@torch.inference_mode()
def encode_images(model, processor, records, batch_size=16):
    embeddings, kept = [], []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        images, batch_kept = [], []
        for r in batch:
            try:
                with Image.open(r["path"]) as img:
                    images.append(ImageOps.exif_transpose(img).convert("RGB"))
                batch_kept.append(r)
            except Exception:
                continue
        if not images:
            continue
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        out = extract_embeddings(model.get_image_features(**inputs)).float()
        embeddings.append(F.normalize(out, dim=-1).cpu())
        kept.extend(batch_kept)
        if start % (batch_size * 10) == 0:
            print(f"  encoded {start}/{len(records)}", flush=True)
    return torch.cat(embeddings, dim=0), kept


@torch.inference_mode()
def encode_text(model, processor, texts):
    inputs = processor(text=texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
    out = extract_embeddings(model.get_text_features(**inputs)).float()
    return F.normalize(out, dim=-1).cpu()


def run_for_model(label, model_path):
    print(f"\n=== {label} ({model_path}) ===")
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    records = []
    for p in metadata:
        code = str(p.get("product_code", "")).strip()
        images = p.get("images") or []
        if not images or not code:
            continue
        path = resolve_image_path(images[0])
        if path:
            records.append({"code": code, "path": path, "brand": p.get("brand"), "name": p.get("name")})

    print(f"Encoding {len(records)} catalog images (one per product)...")
    image_embeddings, kept_records = encode_images(model, processor, records)
    codes = [r["code"] for r in kept_records]

    queries = [tc["query"] for tc in TEST_CASES]
    text_embeddings = encode_text(model, processor, queries)

    for tc, text_embedding in zip(TEST_CASES, text_embeddings):
        similarities = image_embeddings @ text_embedding
        order = torch.argsort(similarities, descending=True)
        ranked_codes = [codes[i] for i in order.tolist()]
        target = tc["code"]
        rank = ranked_codes.index(target) + 1 if target in ranked_codes else None
        top5 = [(kept_records[i]["brand"], kept_records[i]["name"], codes[i]) for i in order[:5].tolist()]
        print(f"\nQuery: '{tc['query']}'")
        print(f"  Target product {target} rank: {rank} (out of {len(ranked_codes)})")
        print("  Top 5 results:")
        for b, n, c in top5:
            print(f"    {b} {n} [{c}]")

    del model
    return


if __name__ == "__main__":
    run_for_model("BASE (untrained)", BASE_MODEL_ID)
    run_for_model("v3 FINE-TUNED", V3_CHECKPOINT)
