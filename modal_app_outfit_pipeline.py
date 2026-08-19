"""Run outfit detection and index building on Modal instead of the laptop.

    modal run modal_app_outfit_pipeline.py::detect          # garments in new photos
    modal run modal_app_outfit_pipeline.py::build_indexes   # photo + crop vectors
    modal run modal_app_outfit_pipeline.py::fetch           # pull results down

## Why this exists

These two jobs are the reason this project keeps stalling. Locally:

- the crop index took **10.5 hours** and was killed four times, because
  macOS's ANE compiler and Spotlight were saturating the same silicon MPS
  needs (load average 24, batches degrading from 6s to 86s)
- free disk hit zero twice, once mid-write, and a truncated
  `metadata.json` is a real risk this project has already documented

On an A10G the same work is roughly **$0.60-1.20 and under an hour**, with
the images already on the `outfit-index` Volume. The laptop stops being
the bottleneck and stops being a failure mode.

## The state-ownership rule

While a Modal job is running, **the Volume's `metadata.json` is the source
of truth** and nothing local should write to it. Both copies are the same
file today; if a local writer (the author backfill, the skin-tone pass)
runs concurrently, the two diverge silently and one set of fields is lost
on the next sync. That failure has happened in this project before, with
175 `structured_caption` fields.

So: push local edits up, run the Modal job, pull the result down. One
writer at a time.
"""

import modal

app = modal.App("outfit-pipeline")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "transformers", "pillow", "numpy",
                 "tqdm", "accelerate", "safetensors", "sentencepiece", "scipy")
    .add_local_file("garment_proposer.py", "/root/garment_proposer.py")
    .add_local_file("segment_outfit.py", "/root/segment_outfit.py")
    .add_local_file("index_outfits.py", "/root/index_outfits.py")
    .add_local_file("outfit_search.py", "/root/outfit_search.py")
    .add_local_file("free_text_visual_search.py", "/root/free_text_visual_search.py")
    .add_local_file("dataset_utils.py", "/root/dataset_utils.py")
    .add_local_file("docs/hierarchy.json", "/root/docs/hierarchy.json")
)

outfit_volume = modal.Volume.from_name("outfit-index", create_if_missing=False)
# The SigLIP2 checkpoint lives on the catalog volume; mounting it read-only
# keeps one copy rather than duplicating 1.4 GB.
catalog_volume = modal.Volume.from_name("fashion-dataset", create_if_missing=False)
hf_secret = modal.Secret.from_name("hf-token")

VOLUMES = {"/v": outfit_volume, "/data": catalog_volume}


@app.function(image=image, gpu="A10G", volumes=VOLUMES, secrets=[hf_secret],
              timeout=6 * 60 * 60)
def detect(limit: int = 0, images_per_post: int = 1, force: bool = False):
    """Human-parsing detection over photos that have no `detected_items` yet.

    Incremental by construction: a photo already carrying detections is
    skipped, so an interrupted run resumes for free.
    """
    import json
    import os
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, "/root")
    os.chdir("/v")
    from PIL import Image, ImageOps
    from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

    import garment_proposer
    import index_outfits

    root = Path("/v/outfit_dataset")
    metadata_path = root / "metadata.json"
    records = json.loads(metadata_path.read_text())

    todo = []
    for record in records:
        if record.get("detected_items"):
            continue
        # A record with detection_meta but no items was already processed and
        # genuinely contains nothing detectable -- 495 of them. Empty and
        # never-tried look identical through `detected_items` alone, so
        # without this the same photos are re-run on every pass. Confirmed
        # by a 20-photo probe that returned 0 garments and 0 failures.
        if record.get("detection_meta") and not force:
            continue
        for rel in (record.get("images") or [])[:images_per_post]:
            name = rel.split("outfit_dataset/", 1)[-1]
            if (root / name).is_file():
                todo.append((record, rel, root / name))
    if limit:
        todo = todo[:limit]
    print(f"{len(todo):,} photos need detection (of {len(records):,} records)")
    if not todo:
        return 0

    processor, model = garment_proposer.load_human_parser("cuda")
    clip_processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    clip_model = AutoModelForZeroShotImageClassification.from_pretrained(
        "patrickjohncyh/fashion-clip").to("cuda")

    started = time.time()
    done = failed = items = 0
    for index, (record, rel, path) in enumerate(todo, 1):
        try:
            proposals = garment_proposer.propose_garment_items(
                str(path), processor, model, clip_processor, clip_model, "cuda")
        except Exception as error:  # noqa: BLE001
            failed += 1
            if failed <= 3:
                print(f"  [warn] {rel}: {error}")
            continue

        detected = []
        for rank, proposal in enumerate(proposals, start=1):
            detected.append({
                "rank": rank,
                "label": proposal.get("label"),
                "category": proposal.get("category"),
                "category_group": proposal.get("category_group"),
                "confidence": proposal.get("confidence"),
                "bbox": list(proposal["bbox"]),
                "area_fraction": proposal.get("area_fraction"),
                "source_image": rel,
                # Colour from the MASKED crop, matching index_outfits'
                # own call site -- a bbox crop drags in background and
                # measured far worse.
                "color": index_outfits.dominant_color(proposal["crop"]),
            })
        record["detected_items"] = detected
        record["detection_meta"] = {
            "detected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": "cuda", "num_items": len(detected),
            "params": {"proposer": "human-parsing",
                       "min_parser_score": garment_proposer.MIN_PARSER_SCORE,
                       "version": 2, "where": "modal"},
        }
        items += len(detected)
        done += 1

        if done % 250 == 0:
            metadata_path.write_text(json.dumps(records, ensure_ascii=False))
            outfit_volume.commit()
            rate = index / (time.time() - started)
            print(f"  {index:6,}/{len(todo):,}  {rate:.1f}/s  "
                  f"~{(len(todo)-index)/rate/60:.0f} min left  ${(time.time()-started)/3600*1.10:.2f}")

    metadata_path.write_text(json.dumps(records, ensure_ascii=False))
    outfit_volume.commit()
    elapsed = time.time() - started
    print(f"\ndetected {done:,} photos, {items:,} garments ({failed} failed) in "
          f"{elapsed/60:.1f} min ~ ${elapsed/3600*1.10:.2f}")
    print("Labels are UNVALIDATED model output; there is no ground truth for "
          "what garments are in these photos.")
    return done


@app.function(image=image, gpu="A10G", volumes=VOLUMES, secrets=[hf_secret],
              timeout=6 * 60 * 60)
def build_indexes(rebuild: bool = False):
    """Photo and crop embedding indexes, written to the Volume."""
    import os
    import sys
    import time

    sys.path.insert(0, "/root")
    os.chdir("/v")
    # outfit_search resolves paths relative to its own file; point both the
    # metadata and the checkpoint search at the mounted volumes.
    os.environ["APPAREL_DATASET_ROOT"] = "/data/apparel_dataset"
    # The volume is a network filesystem: serial loading measured 0.46 s
    # per image with the GPU idle, slower than a laptop. Same value
    # modal_app_serve.py uses for the catalog, for the same reason.
    os.environ["IMAGE_LOADER_WORKERS"] = "32"

    import outfit_search

    outfit_search.REPO_ROOT = __import__("pathlib").Path("/v")
    outfit_search.OUTFIT_METADATA = outfit_search.REPO_ROOT / "outfit_dataset" / "metadata.json"
    outfit_search.PHOTO_INDEX_PATH = outfit_search.REPO_ROOT / "outfit_dataset" / "outfit_search_index.pt"
    outfit_search.CROP_INDEX_PATH = outfit_search.REPO_ROOT / "outfit_dataset" / "outfit_crop_index.pt"

    started = time.time()

    class Args:
        index = "both"
        limit = None

    Args.rebuild = rebuild
    outfit_search.build(Args())
    outfit_volume.commit()
    elapsed = time.time() - started
    print(f"\nindexes built in {elapsed/60:.1f} min ~ ${elapsed/3600*1.10:.2f}")


@app.function(image=image, volumes=VOLUMES, timeout=60 * 60)
def _read(name: str) -> bytes:
    from pathlib import Path

    return (Path("/v/outfit_dataset") / name).read_bytes()


# Fields each side owns. Modal detection writes the first group; local
# passes (author backfill, skin-tone extraction) write the second. A
# straight overwrite in either direction silently destroys the other --
# which is exactly how 175 structured_caption fields were lost once
# before -- so metadata is MERGED per field, not replaced.
REMOTE_OWNED = ("detected_items", "detection_meta")
LOCAL_OWNED = ("author", "source_link", "skin_tone")


@app.local_entrypoint()
def fetch():
    """Pull the indexes down, and MERGE remote metadata into local."""
    import json
    from pathlib import Path

    target = Path(__file__).resolve().parent / "outfit_dataset"

    for name in ("outfit_search_index.pt", "outfit_crop_index.pt"):
        try:
            payload = _read.remote(name)
        except Exception as error:  # noqa: BLE001
            print(f"  {name}: not on the volume ({error})")
            continue
        (target / name).write_bytes(payload)
        print(f"  {name}: {len(payload)/1e6:.0f} MB")

    try:
        remote = json.loads(_read.remote("metadata.json"))
    except Exception as error:  # noqa: BLE001
        print(f"  metadata.json: not fetched ({error})")
        return

    local_path = target / "metadata.json"
    local = json.loads(local_path.read_text())
    key = lambda r: f"{r.get('source')}:{r.get('source_id')}"
    remote_by_key = {key(r): r for r in remote}

    merged = gained = kept = 0
    for record in local:
        counterpart = remote_by_key.get(key(record))
        if not counterpart:
            continue
        for field in REMOTE_OWNED:
            if counterpart.get(field) is not None and record.get(field) != counterpart.get(field):
                record[field] = counterpart[field]
                gained += 1
        kept += sum(1 for f in LOCAL_OWNED if record.get(f) is not None)
        merged += 1

    # Records that exist only on the volume (scraped elsewhere) still come
    # across, so a merge never loses rows either.
    local_keys = {key(r) for r in local}
    added = [r for r in remote if key(r) not in local_keys]
    local.extend(added)

    local_path.write_text(json.dumps(local, ensure_ascii=False))
    print(f"  metadata.json: merged {merged:,} records "
          f"({gained:,} remote fields applied, {kept:,} local fields preserved, "
          f"{len(added):,} new rows)")
