"""Run this ON COLAB (where the DINOv3 checkpoint actually lives, under
Drive) to push it onto the `fashion-dataset` Modal Volume -- the same
Volume SigLIP2 v3 already lives on. Once it's there, both encoders are
reachable from Modal (and via `modal volume get` from anywhere else,
including this local dev environment) without needing Colab at all for
future eval/validation runs.

Why this script exists instead of the one-line `modal volume put ... -r`
command documented in modal_app_phase4_eval.py's docstring: this
project's own SCRAPING_PROCESS.md documents a real, reproduced bug with
the recursive form of `modal volume get`/`put` -- it intermittently
corrupts (creates a stray *file* where a directory should be, "Not a
directory" error) on multi-file directory transfers, on both Drive and
plain local disk, not Drive-FUSE-specific. The documented workaround is
uploading files individually into a pre-created directory structure, one
`modal volume put ... <exact-file>` per file, not the whole folder at
once -- which is exactly what this script automates, with retry-on-
failure per file rather than one shot at the whole tree.

Usage (in a Colab cell, after `!pip install modal` and `modal token new`
or setting MODAL_TOKEN_ID/MODAL_TOKEN_SECRET if you already have them
from this project's existing Modal setup):

    !python3 push_dinov3_to_modal.py

Or import and call push() directly if you want to point it at a
different local source directory / remote target.
"""

import subprocess
import sys
import time
from pathlib import Path

# Adjust if your DINOv3 checkpoint lives somewhere else on Drive --
# this matches dino_identity_finetune.py's OUTPUT_DIR convention and the
# path hierarchical_retrieval_pipeline.py's DINOV3_CHECKPOINT_CANDIDATES
# already expects to find it at.
LOCAL_SOURCE_DIR = Path("/content/drive/MyDrive/apparel_dataset/finetuned_dinov3_identity_v1_supcon")
VOLUME_NAME = "fashion-dataset"
REMOTE_TARGET_PREFIX = "apparel_dataset/finetuned_dinov3_identity_v1_supcon"

MAX_RETRIES_PER_FILE = 3
RETRY_DELAY_SECONDS = 5


def push(local_source_dir=LOCAL_SOURCE_DIR, volume_name=VOLUME_NAME, remote_target_prefix=REMOTE_TARGET_PREFIX):
    local_source_dir = Path(local_source_dir)
    if not local_source_dir.is_dir():
        print(f"ERROR: {local_source_dir} does not exist or is not a directory -- check the path "
              f"(e.g. Drive might be mounted at a different location, or the checkpoint might be under "
              f"a different stage name like stage1_head_best instead of stage2_lastblock_best's parent dir).")
        sys.exit(1)

    all_files = sorted(p for p in local_source_dir.rglob("*") if p.is_file())
    print(f"Found {len(all_files)} files under {local_source_dir}")

    failed = []
    for index, local_path in enumerate(all_files, start=1):
        relative_path = local_path.relative_to(local_source_dir)
        remote_path = f"{remote_target_prefix}/{relative_path}"
        print(f"[{index}/{len(all_files)}] {relative_path} -> {volume_name}:{remote_path}")

        success = False
        for attempt in range(1, MAX_RETRIES_PER_FILE + 1):
            result = subprocess.run(
                ["modal", "volume", "put", volume_name, str(local_path), remote_path, "--force"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                success = True
                break
            print(f"  attempt {attempt}/{MAX_RETRIES_PER_FILE} failed: {result.stderr.strip()[-300:]}")
            if attempt < MAX_RETRIES_PER_FILE:
                time.sleep(RETRY_DELAY_SECONDS)

        if not success:
            failed.append(str(relative_path))

    print(f"\nDone. {len(all_files) - len(failed)}/{len(all_files)} files uploaded successfully.")
    if failed:
        print(f"FAILED ({len(failed)}), re-run this script to retry -- it will just re-upload everything "
              f"(--force overwrites, so safe to re-run, no partial-state cleanup needed):")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("All files uploaded. Verify with:")
        print(f"  !modal volume ls {volume_name} {remote_target_prefix}")


if __name__ == "__main__":
    push()
