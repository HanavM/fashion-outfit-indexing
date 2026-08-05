"""Modal app: dump FROZEN SigLIP2 image embeddings for the whole catalog, once.

Why this exists (item 12.1, attribute heads): the 2026-08-04 ceiling check
established that a linear probe on frozen SigLIP2 features recovers
`structured_caption.attributes` far above majority-class, but it did so on
700 products with ONE image each -- enough to say the signal exists, not
enough to train a head, and with no way to hold out by product.

Training the head itself needs no GPU: the head is a linear/1-hidden-layer
map on top of a frozen 768-d vector. What needs a GPU is producing those
vectors for every catalog image once. So this app does exactly that and
nothing else -- encode, save to the Volume, stop. Everything downstream
(`attribute_heads.py`) is CPU work on a 30 MB .npy.

Two encoders in ONE pass over the images, because the run is bound by
per-file network-filesystem latency (~3-6 img/s), not by the GPU:
  * `google/siglip2-base-patch16-384` -- frozen base, matches the ceiling
    check, and is the "is the information in the pretrained embedding"
    question;
  * `finetuned_siglip2_hierarchical_v3/stage2_lastnblocks_best` -- the
    checkpoint the real retrieval shortlist actually runs, so any retrieval
    experiment built on these vectors is comparable to the pipeline rather
    than to a model nothing serves.

Nothing here writes to metadata.json, the shared retrieval_indexes/, or any
image. Output is two new files under a dedicated directory.

Usage:
    python3 -m modal run modal_app_attribute_embed.py
    python3 -m modal run modal_app_attribute_embed.py --images-per-product 4
    modal volume get fashion-dataset apparel_dataset/attribute_head_embeddings .
"""

from pathlib import Path

import modal

app = modal.App("fashion-attribute-embed")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "transformers", "pillow", "tqdm", "numpy",
        "accelerate", "safetensors", "sentencepiece",
    )
)

volume = modal.Volume.from_name("fashion-dataset", create_if_missing=False)
hf_secret = modal.Secret.from_name("hf-token")

DATA_ROOT = "/data/apparel_dataset"
OUT_DIR = f"{DATA_ROOT}/attribute_head_embeddings"


@app.function(image=image, gpu="A10G", volumes={"/data": volume},
              secrets=[hf_secret], timeout=120 * 60)
def embed(images_per_product: int = 4, limit_products: int = 0):
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch
    from PIL import Image as PILImage
    from transformers import AutoModel, AutoProcessor

    root = Path(DATA_ROOT)
    records = json.loads((root / "metadata.json").read_text())
    if limit_products:
        records = records[:limit_products]
    print(f"{len(records)} products in volume metadata.json", flush=True)

    def resolve(raw):
        # Same candidate ladder as hierarchical_retrieval_pipeline.resolve_image_path:
        # metadata paths are written relative to several different historical
        # roots ("apparel_dataset/...", "shoe_dataset/<brand>/...").
        p = Path(raw)
        candidates = [p, root / p, root.parent / p]
        if p.parts and p.parts[0] == root.name:
            candidates.append(root.joinpath(*p.parts[1:]))
        if len(p.parts) >= 4:
            candidates.append(root.joinpath(*p.parts[-4:]))
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    entries = []          # (product_code, resolved_path, original_path)
    unresolved = 0
    for record in records:
        kept = 0
        for raw in record.get("images", []):
            if kept >= images_per_product:
                break
            path = resolve(raw)
            if path is None:
                unresolved += 1
                continue
            entries.append((record["product_code"], path, raw))
            kept += 1
    print(f"{len(entries)} images to encode, {unresolved} unresolved paths", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_id = "google/siglip2-base-patch16-384"
    processor = AutoProcessor.from_pretrained(base_id)

    models = {"base": AutoModel.from_pretrained(base_id, torch_dtype=torch.float32).to(device).eval()}
    checkpoint = root / "finetuned_siglip2_hierarchical_v3" / "stage2_lastnblocks_best"
    if checkpoint.exists():
        models["v3"] = AutoModel.from_pretrained(str(checkpoint), torch_dtype=torch.float32).to(device).eval()
        print(f"loaded v3 checkpoint {checkpoint}", flush=True)
    else:
        print(f"WARNING: no v3 checkpoint at {checkpoint}; base only", flush=True)

    def load(path):
        try:
            with PILImage.open(path) as im:
                return im.convert("RGB")
        except Exception:
            return None

    def unwrap(output):
        # Same defensive unwrap as hierarchical_retrieval_pipeline's
        # extract_siglip_embeddings: get_image_features returns a bare tensor
        # on some transformers versions and a wrapped output object on others.
        if torch.is_tensor(output):
            return output
        for attribute in ("image_embeds", "pooler_output", "last_hidden_state"):
            value = getattr(output, attribute, None)
            if value is not None:
                return value.mean(dim=1) if attribute == "last_hidden_state" else value
        raise TypeError(f"cannot extract embeddings from {type(output)}")

    batch = 64
    outputs = {name: [] for name in models}
    kept_entries = []
    pool = ThreadPoolExecutor(max_workers=32)
    for start in range(0, len(entries), batch):
        chunk = entries[start:start + batch]
        images = list(pool.map(lambda e: load(e[1]), chunk))
        pairs = [(e, im) for e, im in zip(chunk, images) if im is not None]
        if not pairs:
            continue
        inputs = processor(images=[im for _, im in pairs], return_tensors="pt").to(device)
        with torch.no_grad():
            for name, model in models.items():
                feats = unwrap(model.get_image_features(**inputs))
                feats = feats / feats.norm(dim=-1, keepdim=True)
                outputs[name].append(feats.float().cpu().numpy())
        kept_entries.extend(e for e, _ in pairs)
        if start % (batch * 20) == 0:
            print(f"  {start + len(chunk)}/{len(entries)}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, chunks in outputs.items():
        arr = np.concatenate(chunks, axis=0).astype(np.float32)
        np.save(f"{OUT_DIR}/{name}_image_embeddings.npy", arr)
        print(f"saved {name} {arr.shape}", flush=True)
    Path(f"{OUT_DIR}/index.json").write_text(json.dumps(
        [{"product_code": c, "path": raw} for c, _, raw in kept_entries], indent=1))
    print(f"saved index of {len(kept_entries)} rows", flush=True)
    volume.commit()


@app.local_entrypoint()
def main(images_per_product: int = 4, limit_products: int = 0):
    embed.remote(images_per_product=images_per_product, limit_products=limit_products)
