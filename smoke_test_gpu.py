"""
GPU smoke test for StripedHyena on a local machine (e.g. RTX 5050 in WSL2/Linux)
-------------------------------------------------------------------------------
Checks, in order, and stops at the first problem with a clear message:
  1. PyTorch sees the GPU and supports its architecture
  2. flash-attn imports and runs a real kernel on this GPU
  3. bitsandbytes 4-bit works on this GPU
  4. StripedHyena loads (NF4) and answers - cached and uncached decoding must agree
Prints memory use and seconds per answer.

    python smoke_test_gpu.py                 # NF4 + flash-attn (the DL project's setup)
    python smoke_test_gpu.py --noflash       # NF4, flash-attn off (fallback)
"""

import argparse
import sys
import time


def step(msg):
    print(f"\n== {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noflash", action="store_true")
    ap.add_argument("--compute-dtype", default=None, help="float16 | bfloat16 (default: bfloat16 if supported)")
    args = ap.parse_args()

    step("1. PyTorch + GPU")
    import torch
    print("torch", torch.__version__, "| CUDA build", torch.version.cuda)
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU visible. In WSL2: update the Windows NVIDIA driver, then check `nvidia-smi` works.")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info()
    print(f"{name} | compute capability {cap[0]}.{cap[1]} | {total / 2**30:.1f} GB VRAM ({free / 2**30:.1f} free)")
    arch = f"sm_{cap[0]}{cap[1]}"
    if arch not in torch.cuda.get_arch_list():
        sys.exit(f"This PyTorch build has no kernels for {arch} ({name}). RTX 50-series needs a CUDA 12.8+ build:\n"
                 "  pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128")
    x = torch.randn(256, 256, device="cuda")
    print("matmul OK:", float((x @ x).sum()) == float((x @ x).sum()))

    if not args.noflash:
        step("2. flash-attn")
        try:
            import flash_attn
            from flash_attn import flash_attn_func
            q = torch.randn(1, 64, 8, 64, device="cuda", dtype=torch.float16)
            out = flash_attn_func(q, q, q, causal=True)
            print("flash-attn", flash_attn.__version__, "kernel ran on this GPU:", tuple(out.shape))
        except Exception as e:
            sys.exit(f"flash-attn not usable on this GPU: {type(e).__name__}: {e}\n"
                     "Fallback: python smoke_test_gpu.py --noflash   (and label with --llm stripedhyena-nf4-noflash)")

    step("3. bitsandbytes 4-bit")
    try:
        import bitsandbytes as bnb
        lin = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.float16, quant_type="nf4").cuda()
        lin(torch.randn(2, 64, device="cuda", dtype=torch.float16))
        print("bitsandbytes", bnb.__version__, "NF4 layer ran on this GPU")
    except Exception as e:
        sys.exit(f"bitsandbytes 4-bit failed: {type(e).__name__}: {e}\n  try: pip install --upgrade bitsandbytes")

    step("4. StripedHyena-Nous-7B (NF4" + (", no flash-attn)" if args.noflash else ", flash-attn)"))
    from llm_backends import make_backend
    spec = "stripedhyena-nf4" + ("-noflash" if args.noflash else "")
    kwargs = {"compute_dtype": args.compute_dtype} if args.compute_dtype else {}
    t0 = time.time()
    llm = make_backend(spec, **kwargs)
    print(f"loaded {llm.name} in {time.time() - t0:.0f}s | GPU memory in use "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GB")
    prompts = [
        "Context:\n[1] Paris is the capital and largest city of France.\n\nQuestion: What is the capital of France?",
        "Context:\n[1] Victor Herbert composed the song Dream On (An Indian Lullaby) in 1912.\n\nQuestion: Who was the composer of Dream On (An Indian Lullaby)?",
        "Context:\n[1] Emma Thompson wrote the screenplay for the 2001 film Wit.\n\nQuestion: Who was the screenwriter for Wit?",
    ]
    for p in prompts:      # the first long-enough answer also self-tests the cached decoding modes
        t0 = time.time()
        print(repr(llm.generate(p)), f"in {time.time() - t0:.1f}s | decoding mode: {llm._mode}")
    if llm._mode is None:
        print("answers too short to self-test the cache yet - it will be tested on the first labelled question")
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    print("\nAll good. Pilot (delete any old labels/ files first):\n  python label_dataset.py --llm " + spec +
          (f" --compute-dtype {args.compute_dtype}" if args.compute_dtype else "") + " --limit 200")


if __name__ == "__main__":
    main()
