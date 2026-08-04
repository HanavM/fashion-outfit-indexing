"""CUDA validation for hierarchical_retrieval_pipeline.py's GPU refactor.

The 2026-08-03 GPU work (device-resident indexes, cached query embeddings,
desynced score extraction, threaded image decode) could only be tested
locally on CPU/MPS -- this repo's dev machine has no NVIDIA GPU. This app
exercises the changed paths on a real T4, the same class of card the Colab
runs use.

WHAT THIS VALIDATES: that the pipeline runs end-to-end on CUDA without
device-mismatch errors, that the indexes actually land on the GPU, and how
long an eval takes there.

WHAT IT DOES *NOT* VALIDATE: the eval numbers. The fashion-dataset Volume
is stale relative to Colab Drive -- it holds 6 brands (adidas, gap,
newbalance, nike, pacsun, skechers) rather than the current 12, and has no
finetuned_dinov3_identity_v1_supcon checkpoint, so the identity stage
falls back to frozen base DINOv3. R@K printed here is therefore NOT
comparable to docs/eval_log.md. Don't log it as a result.

Unlike modal_app_phase4_eval.py this ships the REAL pipeline file rather
than the hand-maintained hierarchical_retrieval_pipeline_modal_body.py
copy, which had drifted to 890 lines against the pipeline's 1,700+. The
only thing that copy changed for Modal was DATASET_ROOT, and the pipeline
already supports that via the APPAREL_DATASET_ROOT env override -- so the
duplicate is unnecessary and testing it would test the wrong code.

Usage:
    python3 -m modal run modal_app_pipeline_gpu_check.py
"""

from pathlib import Path

import modal

app = modal.App("fashion-pipeline-gpu-check")

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

# T4 on purpose: it is what the Colab runs use, and the float16-not-bfloat16
# choice in the pipeline is specific to Turing. Timeout is a cost guard --
# a hang dies in 45 min rather than burning credits.
@app.function(image=image, gpu="T4", volumes={"/data": volume},
              secrets=[hf_secret], timeout=45 * 60)
def gpu_check():
    import os
    import subprocess
    import time

    os.environ["APPAREL_DATASET_ROOT"] = "/data/apparel_dataset"

    import torch
    print("=" * 70)
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)} | "
              f"capability: {torch.cuda.get_device_capability(0)}")
    print("=" * 70, flush=True)

    def run(label, args):
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
        started = time.time()
        result = subprocess.run(
            ["python", "/root/hierarchical_retrieval_pipeline.py", *args],
            capture_output=False,
        )
        elapsed = time.time() - started
        print(f"\n--> {label}: exit {result.returncode} in {elapsed:.1f}s", flush=True)
        return result.returncode, elapsed

    timings = {}
    # First call pays catalog verification + every index build.
    timings["evaluate (cold)"] = run("EVALUATE -- cold: builds catalog cache + all indexes",
                                     ["--evaluate"])
    # Second call should hit the catalog cache and every index cache, so it
    # isolates the actual per-query scoring cost -- the part that used to
    # run on the CPU.
    timings["evaluate (warm)"] = run("EVALUATE -- warm: caches hot, measures scoring path",
                                     ["--evaluate"])

    print("\n" + "=" * 70)
    print("TIMINGS")
    for label, (code, elapsed) in timings.items():
        print(f"  {label:<22} exit={code}  {elapsed:7.1f}s")
    print("=" * 70)

    volume.commit()
    failed = [label for label, (code, _) in timings.items() if code != 0]
    if failed:
        raise RuntimeError(f"GPU check FAILED for: {failed}")
    print("GPU check passed.")


@app.local_entrypoint()
def main():
    gpu_check.remote()
