"""
Where do the garbled StripedHyena answers come from?  MEASURE, don't guess.
---------------------------------------------------------------------------
One run prints facts about the loaded model, each checked against a reference:

  1. inventory      - what precision every part of the model is actually in (4-bit, fp16, bf16, fp32)
  2. rotary         - the attention layers' position encoding (RoPE): stored frequencies and the
                      cos/sin tables actually used, vs. the exact formula; and flash-attn's Triton
                      rotary kernel vs. a plain-PyTorch version on THIS GPU
  3. activations    - largest value leaving each block, inf/NaN count (float16 overflows at 65,504)
  4. health         - how well the model predicts ordinary Wikipedia text (from kb/docs.jsonl),
                      split by position in the text. A healthy LM gets BETTER further into a text
                      (more context). If it gets WORSE with position, its long-range machinery
                      (positions / long filters) is broken - matching "long prompts get garbled".
  5. ablation       - the same health test + the 18 questions again with the rotary step done in
                      plain PyTorch instead of the Triton kernel (no reload needed)

Run twice on Kaggle (2x T4), then compare:
    python diagnose_model.py labels/<old labels>.jsonl --llm stripedhyena-nf4-noflash   # as labelled
    python diagnose_model.py labels/<old labels>.jsonl --llm stripedhyena-noflash        # NO 4-bit (fp16, both GPUs)
If the second is clean and the first isn't, 4-bit quantisation is the cause; etc. (verdict printed at the end).
"""

import argparse
import collections
import json
import math
import re
import time

import torch

STUTTER = re.compile(r"\b(\w{1,6}) (\1\w+)")      # "B Bacigalupi", "Bur Burton"
GLUED = re.compile(r"[a-z][A-Z][a-z]")             # "HelpWanted", "ofLaos"


def damaged(reply: str) -> bool:
    return (not reply or "###" in reply or bool(STUTTER.search(reply)) or bool(GLUED.search(reply))
            or bool(re.search(r"\bof[A-Z]", reply)) or reply.endswith((" is", " of", " was")))


def kind_of(name: str) -> str:
    if "rotary_emb" in name:
        return "attention: rotary freqs"
    if "inner_mha_cls" in name:
        return "attention: Wqkv/out_proj"
    if ".filter." in name:
        return "hyena: long filter (poles, D, short conv)"
    if "projections" in name or "out_filter_dense" in name:
        return "hyena: projections"
    if "norm" in name:
        return "norms"
    if ".mlp." in name:
        return "MLP"
    if "embedding" in name or "unembed" in name or "lm_head" in name:
        return "embedding / output head"
    return "other"


def inventory(model):
    rows = collections.defaultdict(collections.Counter)
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        label = "4-bit (bnb)" if type(t).__name__ == "Params4bit" else str(t.dtype).replace("torch.", "")
        rows[kind_of(name)][label] += t.numel()
    print("\n== 1. INVENTORY: precision of each part (number of values)")
    for k in sorted(rows):
        print(f"   {k:<44} " + ", ".join(f"{d}: {n:,}" for d, n in rows[k].items()))


# ---------------------------------------------------------------- rotary
def rotary_modules(model):
    return [m for m in model.modules() if type(m).__name__ == "RotaryEmbedding"]


def exact_inv_freq(mod, device):
    return 1.0 / (mod.base ** (torch.arange(0, mod.dim, 2, device=device, dtype=torch.float32) / mod.dim))


def check_rotary(model):
    mods = rotary_modules(model)
    print(f"\n== 2a. ROTARY tables ({len(mods)} attention layers)")
    if not mods:
        print("   no RotaryEmbedding modules found")
        return
    worst_freq, worst_cos, dtypes, sizes = 0.0, 0.0, collections.Counter(), set()
    for m in mods:
        ref = exact_inv_freq(m, m.inv_freq.device)
        dtypes[str(m.inv_freq.dtype)] += 1
        worst_freq = max(worst_freq, ((m.inv_freq.float() - ref).abs() / ref).max().item())
        cos = getattr(m, "_cos_cached", None)
        if cos is not None:
            sizes.add(tuple(cos.shape))
            pos = torch.arange(cos.shape[0], device=cos.device, dtype=torch.float32)
            ref_cos = torch.cos(torch.outer(pos, ref.to(cos.device)))
            worst_cos = max(worst_cos, (cos.float() - ref_cos).abs().max().item())
    m = mods[0]
    print(f"   base {m.base}, dim {m.dim}, interleaved {m.interleaved}, stored freqs dtype {dict(dtypes)}, "
          f"pos_idx_in_fp32 {getattr(m, 'pos_idx_in_fp32', '?')}")
    print(f"   stored frequencies vs exact formula: worst relative error {worst_freq:.2e}")
    if sizes:
        verdict = "OK" if worst_cos < 5e-3 else "WRONG - attention gets wrong positions"
        print(f"   cos table actually used {sorted(sizes)}: worst error {worst_cos:.2e}  -> {verdict}")


def check_triton_rotary():
    print("\n== 2b. ROTARY kernel: flash-attn Triton vs plain PyTorch on this GPU")
    try:
        from flash_attn.layers.rotary import apply_rotary_emb_func, apply_rotary_emb_torch
    except Exception as e:
        print(f"   could not import flash-attn rotary: {e}")
        return
    torch.manual_seed(0)
    s, h, d = 600, 8, 128
    x = torch.randn(1, s, h, d, device="cuda", dtype=torch.float16)
    inv = 1.0 / (500000 ** (torch.arange(0, d, 2, device="cuda", dtype=torch.float32) / d))
    freqs = torch.outer(torch.arange(s, device="cuda", dtype=torch.float32), inv)
    cos, sin = freqs.cos().half(), freqs.sin().half()
    ref = apply_rotary_emb_torch(x.float(), cos.float(), sin.float()).half()
    try:
        out = apply_rotary_emb_func(x, cos, sin)
        err = (out.float() - ref.float()).abs().max().item()
        print(f"   max difference {err:.2e}  -> {'OK' if err < 1e-2 else 'WRONG - Triton rotary kernel is broken here'}")
    except Exception as e:
        print(f"   Triton kernel errored: {type(e).__name__}: {e}")


def use_torch_rotary():
    """Replace flash-attn's Triton rotary functions with plain PyTorch (same maths, exact float32)."""
    import flash_attn.layers.rotary as R

    def rot(x, cos, sin, interleaved, offsets):
        s = x.shape[1]
        o = int(offsets) if not torch.is_tensor(offsets) or offsets.numel() == 1 else int(offsets[0])
        return R.apply_rotary_emb_torch(x.float(), cos[o:o + s].float(), sin[o:o + s].float(),
                                        interleaved).to(x.dtype)

    def qkv_(qkv, cos, sin, cos_k=None, sin_k=None, interleaved=False, seqlen_offsets=0, num_heads_q=None, **_):
        cos_k, sin_k = (cos if cos_k is None else cos_k), (sin if sin_k is None else sin_k)
        if qkv.dim() == 5:                                   # (b, s, 3, h, d)
            q, k = qkv[:, :, 0], qkv[:, :, 1]
        else:                                                # (b, s, hq + 2*hk, d)  grouped-query
            hk = (qkv.shape[2] - num_heads_q) // 2
            q, k = qkv[:, :, :num_heads_q], qkv[:, :, num_heads_q:num_heads_q + hk]
        q.copy_(rot(q, cos, sin, interleaved, seqlen_offsets))
        k.copy_(rot(k, cos_k, sin_k, interleaved, seqlen_offsets))
        return qkv

    def func(x, cos, sin, interleaved=False, inplace=False, seqlen_offsets=0, cu_seqlens=None, max_seqlen=None):
        out = rot(x, cos, sin, interleaved, seqlen_offsets)
        return x.copy_(out) if inplace else out

    def kv_(kv, cos, sin, interleaved=False, seqlen_offsets=0):
        k = kv[:, :, 0]
        k.copy_(rot(k, cos, sin, interleaved, seqlen_offsets))
        return kv

    R.apply_rotary_emb_qkv_, R.apply_rotary_emb_func, R.apply_rotary_emb_kv_ = qkv_, func, kv_
    R.apply_rotary_emb = func


# ---------------------------------------------------------------- activations + health
def health_texts(kb_dir, tokenizer, n=8, min_tokens=700):
    docs = [json.loads(l) for l in open(f"{kb_dir}/docs.jsonl", encoding="utf-8")]
    by_title = {}
    for d in docs:
        by_title.setdefault(d["title"], []).append(d["text"])
    titles = sorted(by_title)[::max(1, len(by_title) // 200)]
    texts, cur = [], ""
    for t in titles:
        cur += ("\n\n" if cur else "") + " ".join(by_title[t])
        if len(tokenizer(cur).input_ids) >= min_tokens:
            texts.append(cur)
            cur = ""
            if len(texts) == n:
                break
    return texts


BUCKETS = [(1, 64), (64, 128), (128, 256), (256, 512), (512, 768)]


def health(llm, texts, watch_activations=False):
    model, tok = llm.model, llm.tokenizer
    losses = collections.defaultdict(list)
    stats = collections.defaultdict(lambda: [0.0, 0, None])
    hooks = []
    if watch_activations:
        blocks = [m for n, m in model.named_modules() if re.search(r"blocks\.\d+$", n)]
        for i, b in enumerate(blocks):
            def hook(mod, inp, out, i=i, name=type(b).__name__):
                y = out[0] if isinstance(out, tuple) else out
                s = stats[(i, name)]
                s[0] = max(s[0], y.detach().float().abs().nan_to_num(posinf=1e9).max().item())
                s[1] += int((~torch.isfinite(y)).sum().item())
                s[2] = str(y.dtype).replace("torch.", "")
            hooks.append(b.register_forward_hook(hook))
    with torch.no_grad():
        for text in texts:
            ids = tok(text, return_tensors="pt").input_ids[:, :768].to(model.device)
            out = model(input_ids=ids, use_cache=False)
            logits = (out.logits if hasattr(out, "logits") else out[0]).float()
            nll = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="none")
            for pos, v in enumerate(nll.tolist(), start=1):
                losses[pos].append(v)
    for h in hooks:
        h.remove()
    if watch_activations:
        print("\n== 3. ACTIVATIONS leaving each block (float16 overflows at 65,504)")
        for (i, name), (mx, bad, dt) in sorted(stats.items()):
            flag = "  <-- inf/NaN!" if bad else ("  <-- near float16 limit" if mx > 30000 else "")
            print(f"   block {i:2d} {name:<26} {dt:<9} max |value| {mx:>10.1f}  inf/NaN {bad}{flag}")
    row = {}
    for lo, hi in BUCKETS:
        v = [x for p in range(lo, hi) for x in losses.get(p, [])]
        row[f"{lo}-{hi}"] = sum(v) / len(v) if v else float("nan")
    return row


def print_health(label, row):
    cells = "  ".join(f"{k:>8}: {v:5.2f} (ppl {math.exp(v):5.1f})" for k, v in row.items())
    print(f"   {label:<16} {cells}")


# ---------------------------------------------------------------- questions
def question_prompts(labels_path, kb_dir, n_bad=14, n_good=4):
    from feature_extractors import FeatureExtractor, KnowledgeBase
    from llm_backends import build_prompt
    rows = [json.loads(l) for l in open(labels_path, encoding="utf-8")]
    bad = [r for r in rows if damaged(r["reply"])][:n_bad]
    good = [r for r in rows if r["correct"] and not damaged(r["reply"])][:n_good]
    kb = KnowledgeBase.load(kb_dir)
    fx = FeatureExtractor(kb, popularity=json.load(open(f"{kb_dir}/popularity.json")))
    prompts = []
    for r in bad + good:
        res = fx.extract(r["question"])
        ranked = sorted(zip(res.details["rerank_scores"], res.details["chunks"]), key=lambda x: -x[0])
        prompts.append(build_prompt(r["question"], [c["text"] for _, c in ranked], 3))
    return bad + good, prompts


def answer_all(llm, sample, prompts, label):
    from label_dataset import grade
    t0 = time.time()
    replies = [llm.generate(p) for p in prompts]
    dmg = sum(damaged(x) for x in replies)
    cor = sum(grade(x, r["answers"])["correct"] for x, r in zip(replies, sample))
    print(f"   {label:<16} damaged {dmg}/{len(replies)}   correct {cor}/{len(replies)}   ({time.time() - t0:.0f}s)")
    return replies, dmg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", help="the old labels file (to pick the 14 damaged + 4 clean questions)")
    ap.add_argument("--llm", default="stripedhyena-nf4-noflash")
    ap.add_argument("--compute-dtype", default="float16")
    ap.add_argument("--kb", default="kb")
    args = ap.parse_args()

    from llm_backends import make_backend
    print(f"GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    sample, prompts = question_prompts(args.labels, args.kb)
    kw = {"compute_dtype": args.compute_dtype} if "nf4" in args.llm or "fp4" in args.llm else {"dtype": "float16"}
    t0 = time.time()
    llm = make_backend(args.llm, **kw)
    print(f"loaded {llm.name} in {time.time() - t0:.0f}s")
    texts = health_texts(args.kb, llm.tokenizer)

    inventory(llm.model)
    check_triton_rotary()
    base = health(llm, texts, watch_activations=True)
    check_rotary(llm.model)                       # after a forward pass, so the cos tables exist

    print("\n== 4. HEALTH: average loss on Wikipedia text by position (lower = better; healthy = flat or falling)")
    print_health("as loaded", base)
    print("\n== 5. QUESTIONS (14 that were garbled + 4 that were fine)")
    replies0, d0 = answer_all(llm, sample, prompts, "as loaded")

    use_torch_rotary()
    for m in rotary_modules(llm.model):           # rebuild tables from the exact formula, in float32
        m.inv_freq = exact_inv_freq(m, m.inv_freq.device)
        m._seq_len_cached, m._cos_cached, m._sin_cached = 0, None, None
    llm._mode = None                              # re-test cached decoding under the patch
    print("\n== ABLATION: rotary in plain PyTorch + exact frequencies")
    alt = health(llm, texts)
    print_health("torch rotary", alt)
    replies1, d1 = answer_all(llm, sample, prompts, "torch rotary")

    print("\nSide by side:")
    for r, a, b in zip(sample, replies0, replies1):
        print(f"   {r['question'][:40]:<41} {a[:32]!r:<35} {b[:32]!r}")

    print("\n== VERDICT")
    rising = base["512-768"] - base["1-64"]
    print(f"   loss change from start to end of text: {rising:+.2f}  "
          f"({'gets WORSE with position: long-range machinery broken' if rising > 0.3 else 'normal'})")
    if d1 < d0 - 3 or alt["512-768"] < base["512-768"] - 0.3:
        print("   the PyTorch rotary fixes it -> the Triton rotary kernel / rotary tables are the cause")
    else:
        print("   rotary is NOT the cause (patch changed nothing important)")
    print(f"   record for this setup ({llm.name}): damaged {d0}/18, late-position loss {base['512-768']:.2f}"
          "\n   -> run again with the other --llm and compare the two records")


if __name__ == "__main__":
    main()
