"""Delta-sync `outfit_dataset/` to the `outfit-index` Modal Volume.

    .venv/bin/python sync_outfits_to_modal.py --plan     # what would move
    .venv/bin/python sync_outfits_to_modal.py            # do it

## Why

The corpus is the substrate for detection, index builds and any future
fine-tuning, and this laptop cannot host that work reliably: a crop-index
build took 10.5 hours and was killed four times, and free disk has hit
zero twice, once mid-write. Putting the images where the GPU is fixes
both, and is the same pattern `apparel_dataset` already uses.

**Delta, not a full upload.** The volume already holds the 2026-08-04
corpus (9,999 images / 6,860 records); only what has been scraped since
needs to move. The manifest is fetched from the volume and diffed locally,
so a re-run after an interruption costs only what is still missing.

**Chunked tars, not `modal volume put -r`.** The recursive form has a
reproduced corruption bug in this project (it intermittently creates a
file where a directory should be). One tar per chunk sidesteps it, and
chunks are sized so the tar plus its source fit in free disk — this
machine has run out of space mid-operation before, and a tar that dies on
ENOSPC uploads a truncated archive without complaining.
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OUTFIT_DIR = REPO_ROOT / "outfit_dataset"
VOLUME = "outfit-index"
MIN_FREE_GB = 2.5


def free_gb():
    stat = os.statvfs("/System/Volumes/Data")
    return stat.f_bavail * stat.f_frsize / 1e9


def remote_manifest():
    """Relative image paths already on the volume."""
    helper = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    helper.write('''
import json
import modal
app = modal.App("outfit-manifest")
vol = modal.Volume.from_name("outfit-index", create_if_missing=False)

@app.function(image=modal.Image.debian_slim(python_version="3.11"),
              volumes={"/v": vol}, timeout=1800)
def manifest():
    from pathlib import Path
    root = Path("/v/outfit_dataset")
    if not root.is_dir():
        return []
    return [str(p.relative_to(root)) for p in root.rglob("image_*.jpg")]

@app.local_entrypoint()
def main():
    from pathlib import Path
    Path("/tmp/outfit_remote_manifest.json").write_text(json.dumps(manifest.remote()))
    print("manifest written")
''')
    helper.close()
    subprocess.run(["modal", "run", helper.name], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    data = json.loads(Path("/tmp/outfit_remote_manifest.json").read_text())
    os.unlink(helper.name)
    return set(data)


def local_images():
    return {str(p.relative_to(OUTFIT_DIR)): p
            for p in OUTFIT_DIR.rglob("image_*.jpg")}


def chunk(missing, local, max_bytes):
    """Group missing files into tars under `max_bytes`, keeping each post's
    directory intact so a partial upload never splits one record."""
    batches, current, size = [], [], 0
    for rel in sorted(missing):
        path = local[rel]
        try:
            file_size = path.stat().st_size
        except OSError:
            continue
        if current and size + file_size > max_bytes:
            batches.append(current)
            current, size = [], 0
        current.append(rel)
        size += file_size
    if current:
        batches.append(current)
    return batches


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="report and exit")
    ap.add_argument("--chunk-gb", type=float, default=1.0)
    ap.add_argument("--limit-chunks", type=int)
    args = ap.parse_args()

    local = local_images()
    print(f"  local: {len(local):,} images")
    print("  fetching the volume's manifest…")
    remote = remote_manifest()
    print(f"  volume: {len(remote):,} images")

    missing = [r for r in local if r not in remote]
    total = sum(local[r].stat().st_size for r in missing)
    print(f"  to upload: {len(missing):,} images, {total/1e9:.2f} GB")
    if not missing:
        print("  nothing to do")
        return

    batches = chunk(missing, local, int(args.chunk_gb * 1e9))
    if args.limit_chunks:
        batches = batches[:args.limit_chunks]
    print(f"  {len(batches)} chunk(s) of <= {args.chunk_gb} GB\n")
    if args.plan:
        return

    for index, batch in enumerate(batches, 1):
        if free_gb() < MIN_FREE_GB:
            print(f"  STOPPING: only {free_gb():.1f} GB free "
                  f"(need {MIN_FREE_GB}). Re-run to continue.")
            return
        tar_path = Path(tempfile.gettempdir()) / f"outfits_{index}.tar"
        started = time.time()
        with tarfile.open(tar_path, "w") as archive:
            for rel in batch:
                archive.add(local[rel], arcname=rel)
        size = tar_path.stat().st_size
        subprocess.run(["modal", "volume", "put", VOLUME, str(tar_path),
                        f"_incoming/outfits_{index}.tar", "--force"], check=True,
                       stdout=subprocess.DEVNULL)
        tar_path.unlink()
        print(f"  chunk {index}/{len(batches)}: {len(batch):,} images, "
              f"{size/1e9:.2f} GB, {time.time()-started:.0f}s")

    print("\n  uploaded. Extract with:")
    print("    modal run modal_app_sync_outfits.py::extract_all")


if __name__ == "__main__":
    main()
