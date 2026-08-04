"""Modal app for the real Phase 4 end-to-end eval: the full
SigLIP2 + DINOv3 hierarchical retrieval pipeline, run with the category
gate both on and off against the fashion-dataset Volume.

Usage:
    python3 -m modal run modal_app_phase4_eval.py
    # or, for the deployed fire-and-forget path:
    modal deploy modal_app_phase4_eval.py && python3 modal_trigger_phase4_eval.py

## This now runs the REAL pipeline file

It used to run `hierarchical_retrieval_pipeline_modal_body.py`, a
hand-maintained Modal copy of the pipeline. That copy drifted badly --
890 lines against the pipeline's 1,700+ -- so this app was silently
evaluating months-old code, including none of the 2026-08-03 GPU work.
The only thing the copy actually changed for Modal was DATASET_ROOT, and
the pipeline already supports that through its APPAREL_DATASET_ROOT
environment override, so the duplicate bought nothing and cost
correctness. It has been deleted (recoverable from git history); this
ships the real file.

## Before trusting any number this prints, check the Volume

As of 2026-08-03 the `fashion-dataset` Volume is STALE in two ways that
both silently deflate results rather than erroring:

  1. **It holds 6 brands** (adidas, gap, newbalance, nike, pacsun,
     skechers) against the catalog's current 12, with the matching old
     `metadata.json`.
  2. **`finetuned_dinov3_identity_v1_supcon` is missing entirely.** The
     pipeline's checkpoint auto-detection then falls back to frozen base
     DINOv3 with a printed warning, so the identity-rerank stage -- the
     stage that produces the headline R@1 -- is not the trained model.
     (SigLIP2 v3 trained on Modal directly, so that one IS here.)

Neither is fixable from the dev machine: the local `apparel_dataset/` is
itself only a partial copy (it has carhartt/champion/dickies/levis/
stussy/vans but not the volume's six), and the DINOv3 checkpoint exists
only on Colab Drive, where dino_identity_finetune.py ran. Syncing has to
be driven from Colab, e.g.:

    modal volume put fashion-dataset \\
        /content/drive/MyDrive/apparel_dataset/metadata.json \\
        apparel_dataset/metadata.json
    modal volume put fashion-dataset \\
        /content/drive/MyDrive/apparel_dataset/finetuned_dinov3_identity_v1_supcon \\
        apparel_dataset/finetuned_dinov3_identity_v1_supcon -r
    # plus each brand directory the volume is missing

Until that happens this is a working harness pointed at incomplete data.
Its R@K belongs in a scratch note, not docs/eval_log.md. A 2026-08-03 T4
run against the volume as-is scored 11.22% gated / 17.07% ungated, which
is a floor set by the missing checkpoint, not a regression.
"""

from pathlib import Path

import modal

app = modal.App("fashion-phase4-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "transformers", "pillow", "tqdm", "numpy",
        "accelerate", "safetensors", "sentencepiece",
    )
    .add_local_file(
        str(Path(__file__).parent / "hierarchical_retrieval_pipeline.py"),
        "/root/hierarchical_retrieval_pipeline.py",
    )
    .add_local_file(
        str(Path(__file__).parent / "docs" / "hierarchy.json"),
        "/root/docs/hierarchy.json",
    )
)

volume = modal.Volume.from_name("fashion-dataset", create_if_missing=False)
hf_secret = modal.Secret.from_name("hf-token")

# Images live on the Volume, i.e. a network filesystem, and the encode
# loop is bound by per-file round-trip latency rather than by the GPU --
# measured at ~3.2 images/s on a T4 (2026-08-03). Oversubscribing the
# loader is the right response to latency, hence 32 here rather than the
# pipeline's own CUDA default of 16.
ENV = {
    "APPAREL_DATASET_ROOT": "/data/apparel_dataset",
    "IMAGE_LOADER_WORKERS": "32",
    "CATALOG_VERIFY_WORKERS": "64",
}


@app.function(image=image, gpu="A10G", volumes={"/data": volume},
              secrets=[hf_secret], timeout=90 * 60)
def evaluate(extra_args: str = "", index_dir: str = ""):
    """`extra_args` is appended to --evaluate (e.g. "--open-set-holdout-fraction 0.1").

    `index_dir` sets RETRIEVAL_INDEX_DIR. Pass it for ANY run whose flags
    change the identity index fingerprint -- open-set holdout is the live
    example, since it removes whole identities from the gallery and would
    otherwise rebuild the shared retrieval_indexes/ in place. modal_app_serve.py
    loads that directory at container start, so a shared rebuild means the
    next serving cold start quietly answers from a smaller catalog.
    """
    import os
    import shlex
    import subprocess
    import sys

    env = {**os.environ, **ENV, "PYTHONUNBUFFERED": "1"}
    if index_dir:
        env["RETRIEVAL_INDEX_DIR"] = index_dir
        print(f"index dir isolated to {index_dir} (serving index untouched)", flush=True)

    args = [sys.executable, "/root/hierarchical_retrieval_pipeline.py", "--evaluate"]
    args += shlex.split(extra_args)
    print(f"running: {' '.join(args)}", flush=True)

    result = subprocess.run(args, env=env)
    # Persists the catalog + index caches this run just built, so the next
    # invocation skips catalog verification and index building entirely.
    volume.commit()
    if result.returncode != 0:
        raise RuntimeError(f"evaluate failed ({result.returncode})")


@app.local_entrypoint()
def main(extra_args: str = "", index_dir: str = ""):
    evaluate.remote(extra_args=extra_args, index_dir=index_dir)
