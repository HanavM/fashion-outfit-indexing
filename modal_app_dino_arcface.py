"""Benchmark ArcFace against SupCon for the DINOv3 identity encoder.

`dino_identity_finetune.py` has implemented both losses since it was
written, and its own docstring says the spec calls for benchmarking them
by flipping LOSS_TYPE -- but "arcface" appears zero times in
docs/eval_log.md. It has never actually been run. This app runs it.

## Why this is the lever worth pulling

The `--top-identity-candidates` sweep closed at K=150 on 2026-08-03 with
shortlist miss at 1.18%: retrieval breadth is solved. What did NOT improve
is **conditional R@1** -- accuracy given the true product IS in the
candidate set -- which is flat-to-declining across the whole sweep and
sits at 54.6%. DINOv3 is wrong nearly half the time while holding the
right answer, so every remaining point has to come from rerank quality.

SupCon pulls same-identity images together but enforces no *margin*
between different identities. ArcFace imposes an explicit angular margin,
which is the standard fix for exactly this "right answer present but
ranked second" failure in face and product re-identification. Supporting
evidence that the loss, not the capacity, is the binding constraint: on
the supcon run, stage 2 (last-block unfreeze) bought only +0.44pt over
stage 1 (heads-only) -- almost everything the loss can express is already
captured by a projection head.

## Safety of running this

Output directories are namespaced by loss type
(`finetuned_dinov3_identity_v1_arcface`), which the training script does
deliberately -- its own comment explains that loss-agnostic paths would
let a second run see stage1_complete/stage2_complete from the first,
skip training, and silently report the FIRST loss's metrics mislabelled
under the second. So this **cannot** clobber the supcon checkpoint that
every current number depends on.

The run is also resumable: the script checkpoints per stage and writes
resume_state.json, so a timeout continues rather than restarting.

## Comparability

Trains against the Volume's catalog, which as of 2026-08-03 is 1,234
products / 6 brands -- identical to Colab's, so the resulting checkpoint
is directly comparable to the 53.95% baseline. Do NOT compare it to
numbers produced on the 12-brand local catalog, which has nearly twice
the gallery and is a different problem.

Usage:
    python3 -m modal run modal_app_dino_arcface.py                 # arcface
    python3 -m modal run modal_app_dino_arcface.py --loss supcon   # control rerun
"""

from pathlib import Path

import modal

app = modal.App("fashion-dino-arcface")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "transformers", "pillow", "tqdm", "numpy",
        "accelerate", "safetensors", "sentencepiece",
    )
    .add_local_file(
        str(Path(__file__).parent / "dino_identity_finetune.py"),
        "/root/dino_identity_finetune.py",
    )
)

volume = modal.Volume.from_name("fashion-dataset", create_if_missing=False)
hf_secret = modal.Secret.from_name("hf-token")


@app.function(image=image, gpu="A10G", volumes={"/data": volume},
              secrets=[hf_secret], timeout=6 * 60 * 60)
def train(loss: str = "arcface"):
    import os
    import subprocess
    import sys
    import time

    import torch
    print(f"cuda: {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}",
          flush=True)

    env = {
        **os.environ,
        "APPAREL_DATASET_ROOT": "/data/apparel_dataset",
        "LOSS_TYPE": loss,
        # Block buffering to a pipe is why an earlier long run's logs sat
        # frozen for hours while it was in fact working.
        "PYTHONUNBUFFERED": "1",
    }

    started = time.time()
    result = subprocess.run([sys.executable, "/root/dino_identity_finetune.py"],
                            cwd="/data", env=env)
    elapsed = (time.time() - started) / 60
    print(f"\n{loss}: exit={result.returncode} in {elapsed:.1f} min", flush=True)

    # Commit even on failure -- the script checkpoints per stage, so a
    # partial run is resumable only if what it wrote actually persists.
    volume.commit()
    if result.returncode != 0:
        raise RuntimeError(f"dino_identity_finetune.py failed ({result.returncode})")

    out = Path(f"/data/apparel_dataset/finetuned_dinov3_identity_v1_{loss}")
    print(f"\nwrote: {sorted(p.name for p in out.iterdir()) if out.exists() else 'NOTHING'}")


@app.local_entrypoint()
def main(loss: str = "arcface"):
    train.remote(loss=loss)
