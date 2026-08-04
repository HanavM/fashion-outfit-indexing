"""Sync Colab Drive's apparel_dataset -> the Modal `fashion-dataset` Volume.

Run this IN COLAB. Colab Drive is the only place the current catalog and
both fine-tuned checkpoints coexist: this dev machine holds a partial copy
(carhartt/champion/dickies/levis/stussy/vans but not adidas/gap/newbalance/
nike/pacsun/skechers) and the Modal Volume holds the complementary half
plus a stale metadata.json. Colab also has a far faster uplink than a home
connection, which is the other reason the sync belongs here.

What it does:
  1. Diffs Drive against the Volume and prints a plan with sizes.
  2. Uploads only what is missing, one brand at a time.
  3. Force-overwrites metadata.json (the Volume's is the stale 6-brand one).
  4. Re-lists the Volume and reports what is still absent.

Safe to re-run: completed brands are skipped on the next pass, so a
disconnect costs only the brand that was in flight.

NOT uploaded, on purpose:
  - `retrieval_indexes/` -- derived data. Leaving the Volume's copy alone
    is deliberate: build_or_load_identity_index fingerprints on the
    checkpoint path, so introducing the real DINOv3 checkpoint invalidates
    it correctly and it rebuilds. Copying a stale index over would not
    make it any fresher, and copying a *newer* one risks a fingerprint
    that claims to describe vectors built somewhere else.
  - Anything under a brand dir that Drive does not have. This is a
    one-way push, never a delete.

Usage in a Colab cell:

    !wget -q -O sync.py https://raw.githubusercontent.com/HanavM/fashion-outfit-indexing/main/colab_sync_drive_to_modal.py
    !python sync.py

or paste the file into a cell and run it.
"""

import os
import subprocess
import sys
from pathlib import Path

VOLUME = "fashion-dataset"
DRIVE_ROOT = Path("/content/drive/MyDrive/apparel_dataset")
REMOTE_ROOT = "apparel_dataset"

# Pushed even though they are not brand directories.
EXTRA_PATHS = [
    "finetuned_dinov3_identity_v1_supcon",   # the missing checkpoint -- the whole point
    "finetuned_siglip2_hierarchical_v3",     # skipped automatically if already present
]

SKIP = {"retrieval_indexes", ".DS_Store", "__pycache__", ".ipynb_checkpoints"}


def run(args, **kwargs):
    return subprocess.run(args, text=True, capture_output=True, **kwargs)


def ensure_modal():
    try:
        import modal  # noqa: F401
    except ImportError:
        print("Installing modal...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "modal"], check=True)

    # Token from env if provided (paste MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
    # from your local ~/.modal.toml to skip the browser flow entirely).
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        print("Using MODAL_TOKEN_ID / MODAL_TOKEN_SECRET from the environment.")
        return
    probe = run([sys.executable, "-m", "modal", "volume", "list"])
    if probe.returncode == 0:
        print("Modal already authenticated.")
        return
    print("Modal is not authenticated. Running `modal setup` -- open the URL it prints.\n")
    subprocess.run([sys.executable, "-m", "modal", "setup"], check=True)


def mount_drive():
    if DRIVE_ROOT.exists():
        return
    try:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount("/content/drive")
    except ImportError:
        sys.exit("Not running in Colab and Drive is not mounted -- nothing to sync from.")
    if not DRIVE_ROOT.exists():
        sys.exit(f"{DRIVE_ROOT} does not exist. Check the path.")


def remote_entries():
    result = run([sys.executable, "-m", "modal", "volume", "ls", VOLUME, REMOTE_ROOT])
    if result.returncode != 0:
        sys.exit(f"Could not list the volume:\n{result.stderr}")
    names = set()
    for line in result.stdout.splitlines():
        for token in line.replace("│", " ").split():
            if token.startswith(REMOTE_ROOT + "/"):
                tail = token[len(REMOTE_ROOT) + 1:].strip("/")
                if tail and tail != ".":
                    names.add(tail.split("/")[0])
    return names


def dir_size_mb(path: Path) -> float:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total / 1e6


def main():
    mount_drive()
    ensure_modal()

    present = remote_entries()
    print(f"\nVolume already has: {', '.join(sorted(present)) or '(nothing)'}\n")

    local_dirs = sorted(
        p.name for p in DRIVE_ROOT.iterdir()
        if p.is_dir() and p.name not in SKIP
    )
    wanted = [d for d in local_dirs if d in EXTRA_PATHS or not d.startswith("finetuned_")]
    todo = [d for d in wanted if d not in present]

    print("PLAN")
    if not todo:
        print("  nothing to upload -- every directory is already on the volume")
    for name in todo:
        print(f"  upload {name:42s} {dir_size_mb(DRIVE_ROOT / name):9.0f} MB")
    print(f"  overwrite {REMOTE_ROOT}/metadata.json (volume's copy is stale)")
    total = sum(dir_size_mb(DRIVE_ROOT / d) for d in todo)
    print(f"\n  total to upload: {total:.0f} MB\n")

    # metadata.json first: if the sync is interrupted, a current catalog
    # pointing at some-images-missing is a recoverable state, whereas
    # images with a stale catalog silently look complete and are not.
    meta = DRIVE_ROOT / "metadata.json"
    if meta.is_file():
        print("Uploading metadata.json (force)...")
        result = subprocess.run(
            [sys.executable, "-m", "modal", "volume", "put", "-f", VOLUME,
             str(meta), f"{REMOTE_ROOT}/metadata.json"])
        if result.returncode != 0:
            print("  !! metadata.json failed -- fix this before trusting any eval run")

    failed = []
    for index, name in enumerate(todo, 1):
        local = DRIVE_ROOT / name
        print(f"\n[{index}/{len(todo)}] {name} ({dir_size_mb(local):.0f} MB)")
        result = subprocess.run(
            [sys.executable, "-m", "modal", "volume", "put", VOLUME,
             str(local), f"{REMOTE_ROOT}/{name}"])
        if result.returncode != 0:
            print(f"  !! {name} FAILED -- re-run this script to retry just this one")
            failed.append(name)

    print("\n" + "=" * 60)
    after = remote_entries()
    still_missing = [d for d in wanted if d not in after]
    print(f"Volume now has: {', '.join(sorted(after))}")
    if still_missing:
        print(f"STILL MISSING: {', '.join(still_missing)}  -- re-run to retry")
    else:
        print("All directories present.")
    if failed:
        print(f"Failed this run: {', '.join(failed)}")

    checkpoint = "finetuned_dinov3_identity_v1_supcon"
    if checkpoint in after:
        print(f"\n{checkpoint} is on the volume -- Modal eval runs will now use the "
              "real fine-tuned DINOv3 instead of falling back to the frozen base model.")
    else:
        print(f"\nWARNING: {checkpoint} is NOT on the volume. Modal eval runs will "
              "silently fall back to frozen base DINOv3 and score far lower. "
              "Check that the directory exists in Drive under that exact name:")
        print("   ", DRIVE_ROOT / checkpoint)


if __name__ == "__main__":
    main()
