"""Sync newly-scraped brands onto the `fashion-dataset` Volume.

    # one brand at a time, from sync_catalog.sh
    modal volume put fashion-dataset obey.tar _incoming/obey.tar
    modal run modal_app_sync_catalog.py::extract --tar-name obey.tar

Why a tar and not `modal volume put -r`: the recursive form of
`modal volume put`/`get` has a reproduced corruption bug in this project
(it intermittently creates a stray *file* where a directory should be, and
then fails with "Not a directory"). It was reproduced on both Drive and
plain local disk, so it is not a FUSE artifact. The documented workaround
is one file per call -- which for ~4,000 images is impractical, so this
sends ONE file per brand and unpacks it inside the container instead.

`verify` exists because a sync that silently half-lands is worse than one
that fails: it re-reads the volume and reports the per-brand file counts
and the metadata record counts side by side.
"""

import modal

app = modal.App("fashion-sync-catalog")
image = modal.Image.debian_slim(python_version="3.11")
volume = modal.Volume.from_name("fashion-dataset", create_if_missing=False)


@app.function(image=image, volumes={"/data": volume}, timeout=60 * 60)
def extract(tar_name: str):
    """Unpack /data/_incoming/<tar_name> into /data/apparel_dataset/."""
    import os
    import tarfile
    from pathlib import Path

    source = Path("/data/_incoming") / tar_name
    if not source.is_file():
        raise SystemExit(f"{source} not on the volume — did `modal volume put` run?")

    target = Path("/data/apparel_dataset")
    target.mkdir(parents=True, exist_ok=True)

    before = sum(1 for _ in target.rglob("*.jpg"))
    with tarfile.open(source) as archive:
        members = archive.getmembers()
        # The tar is ours, but extraction is still filtered: a member
        # escaping the target directory would be writing into the volume
        # root, and `data` filter is the stdlib's own guard for that.
        archive.extractall(target, filter="data")
    after = sum(1 for _ in target.rglob("*.jpg"))

    os.unlink(source)
    volume.commit()
    print(f"{tar_name}: {len(members):,} members, "
          f"images {before:,} -> {after:,} (+{after - before:,})")
    return after - before


@app.function(image=image, volumes={"/data": volume}, timeout=30 * 60)
def put_metadata(payload: bytes):
    """Replace the volume's metadata.json.

    Sent as bytes through the function call rather than via
    `modal volume put`, so the whole sync is one mechanism. At ~7 MB it is
    comfortably inside Modal's argument limit.
    """
    import json
    from pathlib import Path

    path = Path("/data/apparel_dataset/metadata.json")
    before = 0
    if path.is_file():
        try:
            before = len(json.loads(path.read_text()))
        except json.JSONDecodeError:
            before = -1
    records = json.loads(payload)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    volume.commit()
    print(f"metadata.json: {before:,} -> {len(records):,} records")
    return len(records)


@app.function(image=image, volumes={"/data": volume}, timeout=30 * 60)
def verify():
    """Per-brand image counts on the volume vs record counts in metadata."""
    import collections
    import json
    from pathlib import Path

    root = Path("/data/apparel_dataset")
    records = json.loads((root / "metadata.json").read_text())
    by_brand = collections.Counter(r.get("brand") for r in records)

    print(f"{'brand':16} {'records':>8} {'image files':>12} {'referenced':>11} {'missing':>8}")
    total_missing = 0
    for brand in sorted(by_brand):
        folder = root / brand
        files = sum(1 for _ in folder.rglob("*.jpg")) if folder.is_dir() else 0
        referenced = missing = 0
        for record in records:
            if record.get("brand") != brand:
                continue
            for relative in record.get("images") or []:
                referenced += 1
                # Two path conventions coexist. The four original shoe
                # brands predate the current layout and are stored as
                # "shoe_dataset/<brand>/<slug>/<code>/image_N.jpg", while
                # everything since uses "apparel_dataset/...". A naive
                # split on "apparel_dataset/" leaves the legacy paths
                # untouched and reports every shoe image as missing --
                # which it did, 4,973 false alarms, before this was fixed.
                #
                # Mirrors free_text_visual_search.resolve_image_path: try
                # the path as-is, then fall back to its last four parts
                # (brand/slug/code/file) joined onto the dataset root.
                parts = Path(relative).parts
                candidates = [root / relative]
                if len(parts) >= 4:
                    candidates.append(root.joinpath(*parts[-4:]))
                if not any(candidate.is_file() for candidate in candidates):
                    missing += 1
        total_missing += missing
        print(f"{brand:16} {by_brand[brand]:8,} {files:12,} {referenced:11,} {missing:8,}")
    print(f"\nTOTAL records {len(records):,}, missing image files {total_missing:,}")
    return total_missing
