"""Unpack outfit chunks onto the `outfit-index` Volume, and keep metadata current.

    modal run modal_app_sync_outfits.py::extract_all
    modal run modal_app_sync_outfits.py::put_metadata
    modal run modal_app_sync_outfits.py::verify

Companion to `sync_outfits_to_modal.py`, which uploads chunked tars.
Chunks rather than `modal volume put -r` because the recursive form has a
reproduced corruption bug in this project.
"""

import json
from pathlib import Path

import modal

app = modal.App("outfit-sync")
image = modal.Image.debian_slim(python_version="3.11")
volume = modal.Volume.from_name("outfit-index", create_if_missing=False)

LOCAL_METADATA = Path(__file__).resolve().parent / "outfit_dataset" / "metadata.json"


@app.function(image=image, volumes={"/v": volume}, timeout=60 * 60)
def extract_all():
    """Unpack every tar in /v/_incoming into /v/outfit_dataset, then remove it."""
    import tarfile

    incoming = Path("/v/_incoming")
    target = Path("/v/outfit_dataset")
    target.mkdir(parents=True, exist_ok=True)
    if not incoming.is_dir():
        print("nothing in _incoming")
        return 0

    before = sum(1 for _ in target.rglob("image_*.jpg"))
    tars = sorted(incoming.glob("*.tar"))
    for archive_path in tars:
        with tarfile.open(archive_path) as archive:
            # `data` filter: a member escaping the target would write into
            # the volume root. The tar is ours, but that is not a reason to
            # extract unfiltered.
            archive.extractall(target, filter="data")
        archive_path.unlink()
        print(f"  extracted {archive_path.name}")
    after = sum(1 for _ in target.rglob("image_*.jpg"))
    volume.commit()
    print(f"images {before:,} -> {after:,} (+{after-before:,}) from {len(tars)} tar(s)")
    return after - before


@app.function(image=image, volumes={"/v": volume}, timeout=30 * 60)
def _write_metadata(payload: bytes):
    records = json.loads(payload)
    path = Path("/v/outfit_dataset/metadata.json")
    before = 0
    if path.is_file():
        try:
            before = len(json.loads(path.read_text()))
        except json.JSONDecodeError:
            before = -1
    path.write_text(json.dumps(records, ensure_ascii=False))
    volume.commit()
    print(f"metadata.json: {before:,} -> {len(records):,} records")
    return len(records)


@app.local_entrypoint()
def put_metadata():
    """Push the local metadata.json up. Sent as bytes so the whole sync is
    one mechanism rather than a mix of volume-put and function calls."""
    _write_metadata.remote(LOCAL_METADATA.read_bytes())


@app.function(image=image, volumes={"/v": volume}, timeout=30 * 60)
def verify():
    """Do the images the metadata references actually exist on the volume?"""
    root = Path("/v/outfit_dataset")
    records = json.loads((root / "metadata.json").read_text())
    referenced = missing = 0
    with_images = 0
    for record in records:
        images = record.get("images") or []
        if images:
            with_images += 1
        for rel in images:
            referenced += 1
            # Records store repo-relative paths ("outfit_dataset/reddit/...");
            # on the volume that root IS /v/outfit_dataset.
            name = rel.split("outfit_dataset/", 1)[-1]
            if not (root / name).is_file():
                missing += 1
    present = sum(1 for _ in root.rglob("image_*.jpg"))
    print(f"records {len(records):,} ({with_images:,} with images)")
    print(f"referenced {referenced:,}  ·  files on volume {present:,}  ·  missing {missing:,}")
    return missing
