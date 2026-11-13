"""
Run the StripedHyena labelling on a cloud GPU with Modal (https://modal.com)
---------------------------------------------------------------------------
Modal's Starter plan includes $30 of free compute per month with no credit
card. This script rents an NVIDIA L4 (24 GB, new enough for flash-attn),
runs label_dataset.py there, and downloads the labels to ./labels/.

One-time setup on your Mac (inside the project venv):
    pip install modal
    modal setup                      # opens the browser to log in / sign up

Pilot (200 questions):
    modal run modal_label.py --limit 200
Full run later (continues where the pilot stopped; labels live in a Modal Volume):
    modal run modal_label.py --limit 0

Notes
  * First run builds the container image (installs CUDA PyTorch + flash-attn)
    and downloads ~15 GB of model weights into a persistent Volume: expect
    10-20 minutes before the first question. Later runs reuse both.
  * The whole project folder is uploaded except data/, labels*/ and caches;
    kb/ must exist (run build_kb.py first).
  * Library versions are pinned to ones contemporary with StripedHyena's
    code (late 2023 / 2024); newer transformers releases may break its
    remote code.
"""

import os

import modal

app = modal.App("bayesian-epistemic-routing-labelling")

volume = modal.Volume.from_name("ber-cache", create_if_missing=True)   # HF weights + labels persist here

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.3.1", "transformers==4.41.2", "accelerate==0.30.1", "sentencepiece",
        "numpy<2", "packaging", "ninja", "wheel",
        "pgmpy", "spacy>=3.7,<3.8", "sentence-transformers==3.0.1", "faiss-cpu", "scikit-learn", "requests",
        # spaCy 3.7 imports click directly, but newer typer releases no longer pull it in.
        "click",
    )
    # Prebuilt flash-attn wheel matching torch 2.3 / CUDA 12 / Python 3.11 (no long compile).
    .run_commands("pip install flash-attn==2.5.9.post1 --no-build-isolation")
    .run_commands("python -m spacy download en_core_web_sm")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(".", remote_path="/root/project",
                   ignore=["data", "data/**", "labels*", "labels*/**", "**/__pycache__", "**/.DS_Store",
                           ".venv", ".venv/**", "*.docx", "*.pdf", "*.png", "*.jpeg"])
)


@app.function(image=image, gpu="L4", volumes={"/cache": volume}, timeout=8 * 3600)
def label(limit: int = 200, llm: str = "stripedhyena") -> dict:
    import subprocess
    os.chdir("/root/project")
    out_dir = "/cache/labels"
    cmd = ["python", "label_dataset.py", "--llm", llm, "--out", out_dir, "--limit", str(limit)]
    try:
        subprocess.run(cmd, check=True)
    finally:
        volume.commit()                         # keep partial progress even if the run fails
    return {name: open(os.path.join(out_dir, name), encoding="utf-8").read()
            for name in os.listdir(out_dir)}


@app.function(image=image, gpu="L4", volumes={"/cache": volume}, timeout=1800)
def smoke_test() -> str:
    """Load StripedHyena and answer one question - checks the install before a long run."""
    import sys
    import time
    sys.path.insert(0, "/root/project")
    from llm_backends import make_backend
    t0 = time.time()
    llm = make_backend("stripedhyena")
    load_s = time.time() - t0
    t0 = time.time()
    reply = llm.generate("Context:\n[1] Paris is the capital of France.\n\nQuestion: What is the capital of France?")
    volume.commit()
    return f"loaded in {load_s:.0f}s, answered {reply!r} in {time.time() - t0:.1f}s"


@app.local_entrypoint()
def main(limit: int = 200, llm: str = "stripedhyena", smoke: bool = False):
    if smoke:
        print(smoke_test.remote())
        return
    files = label.remote(limit, llm)
    os.makedirs("labels", exist_ok=True)
    for name, text in files.items():
        with open(os.path.join("labels", name), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"downloaded labels/{name}")
