"""
LLM plug-ins for Bayesian Epistemic Routing
-------------------------------------------
Everything else in the project is LLM-agnostic; only this file knows how to
talk to a model. Each backend exposes ONE method:

    generate(prompt: str, max_new_tokens: int = 32) -> str

and a `name` used to tag learned parameters (a profile is only valid for the
model it was learned with).

Backends
  StripedHyenaBackend - togethercomputer/StripedHyena-Nous-7B via Hugging Face
                        transformers (needs an NVIDIA GPU, Ampere/RTX-30xx or
                        newer, ~16 GB VRAM in 16-bit; flash-attn is required by
                        the model code)
  HFCausalBackend     - any ordinary Hugging Face causal LM (for comparisons)
  OpenAICompatBackend - any local server speaking the OpenAI chat API
                        (Ollama, llama.cpp server, LM Studio, vLLM, ...)
  EchoBackend         - offline stand-in for tests (answers from the context)

Usage
    llm = make_backend("stripedhyena-nf4-noflash")  # or "hf:<model id>", "openai:<model>@<url>", "echo"
    print(llm.generate(build_prompt(question, passages)))
"""

from __future__ import annotations

import re
from typing import Optional


# --------------------------------------------------------------------------
# Prompt: the same RAG prompt for every backend; each backend wraps it in
# its own chat format.
# --------------------------------------------------------------------------

RAG_INSTRUCTION = (
    "Answer the question using the context passages if they help. "
    "Reply with the answer only, in a few words. "
    "If you do not know the answer, reply exactly: I don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def build_prompt(question: str, passages: list[str], max_passages: int = 3, conversation=None) -> str:
    """conversation: optional earlier turns [[question, answer], ...] (follow-up questions need them)."""
    context = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages[:max_passages])) or "(none)"
    prompt = RAG_INSTRUCTION.format(context=context, question=question)
    if conversation:
        turns = "\n".join(f"Q: {q}\nA: {a}" for q, a in conversation)
        prompt = prompt.replace("\n\nQuestion: ", f"\n\nConversation so far:\n{turns}\n\nQuestion: ", 1)
    return prompt


# Chat-format headers models echo before/after the answer: "### Response:", "### Reply:", ...
HEADER_LINE = re.compile(r"^#{2,}\s*[A-Za-z][A-Za-z ]*:?\s*$")
HEADER_PREFIX = re.compile(r"^#{2,}\s*[A-Za-z][A-Za-z ]*:\s*")


def _answer_lines(text: str):
    """Yield answer lines, skipping blank lines and echoed '### Something:' headers."""
    for line in text.replace("</s>", "").split("\n"):
        line = line.strip()
        if not line or HEADER_LINE.match(line):
            continue
        line = HEADER_PREFIX.sub("", line).strip()      # "### Response: Paris" -> "Paris"
        if line:
            yield line


def clean_answer(text: str) -> str:
    """
    The first real answer line. StripedHyena often writes
    '### Response:\nThe composer of Rent is Jonathan Larson.\n</s>' -
    the header is skipped, not treated as the end of the answer.
    """
    return next(_answer_lines(text), "")


def answer_complete(text: str) -> bool:
    """True once a real answer line has been written AND ended by a newline (or </s>)."""
    lines = text.replace("</s>", "\n").split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or HEADER_LINE.match(line) or not HEADER_PREFIX.sub("", line).strip():
            continue
        return i < len(lines) - 1
    return False


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

class HFCausalBackend:
    """Any Hugging Face causal LM, greedy decoding."""

    prompt_template = "{prompt}"

    def __init__(self, model_id: str, load_in_4bit: bool = False, dtype: Optional[str] = None,
                 trust_remote_code: bool = False, config_overrides: Optional[dict] = None,
                 quant: Optional[str] = None, compute_dtype: Optional[str] = None,
                 double_quant: bool = False, adapter: Optional[str] = None):
        """
        quant:         None (16-bit) | "nf4" | "fp4"   (bitsandbytes 4-bit; load_in_4bit=True means "nf4")
        compute_dtype: dtype the 4-bit layers compute in ("float16" / "bfloat16"); default = best the GPU supports
        double_quant:  bitsandbytes' double quantisation (QLoRA option)
        adapter:       path or hub id of a PEFT/LoRA adapter to load on top (e.g. a fine-tune from the DL project)
        """
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        self.name = model_id
        self.torch = torch
        kwargs = {"trust_remote_code": trust_remote_code}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"          # spreads a 16-bit 7B model over both Kaggle T4s
        if config_overrides:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            for key, value in config_overrides.items():
                setattr(config, key, value)
            kwargs["config"] = config
        best = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if torch.cuda.is_available() \
            else "float32"                         # CPU: float32
        quant = quant or ("nf4" if load_in_4bit else None)
        if quant:                              # e.g. NF4: an 8B model in ~5 GB, as in QLoRA setups
            from transformers import BitsAndBytesConfig
            compute_dtype = compute_dtype or best
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type=quant,
                bnb_4bit_compute_dtype=getattr(torch, compute_dtype),
                bnb_4bit_use_double_quant=double_quant)
            # Precision of everything that is NOT 4-bit (embeddings, norms, Hyena filters).
            # Set explicitly: recent transformers otherwise pick a default (e.g. the
            # checkpoint's bfloat16) that older GPUs like the T4 can't run natively.
            dtype = dtype or compute_dtype
            kwargs["dtype"] = getattr(torch, dtype)
            self.name += f"-{quant}-{compute_dtype}" + (f"-params{dtype}" if dtype != compute_dtype else "") \
                         + ("-dq" if double_quant else "")
        else:
            dtype = dtype or best
            kwargs["dtype"] = getattr(torch, dtype)
            self.name += f"-{dtype}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.name += "+" + adapter.rstrip("/").split("/")[-1]
        self.model.eval()

    def _format(self, prompt: str) -> str:
        """Instruct/chat models get their own chat template (the format they were trained on)."""
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                enable_thinking=False)          # Qwen3-style models: answer directly, no hidden reasoning
        return self.prompt_template.format(prompt=prompt)

    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        inputs = self.tokenizer(self._format(prompt), return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None,
                                      pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        self.last_raw = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return clean_answer(self.tokenizer.decode(new_tokens, skip_special_tokens=True))


class StripedHyenaBackend(HFCausalBackend):
    """
    StripedHyena-Nous-7B (Together AI + Nous Research), Apache-2.0.
    Prompt format from the model card: "### Instruction:\\n{prompt}\\n\\n### Response:\\n".
    Precision (checked on Kaggle): the checkpoint stores poles/residues in float16 and the
    remote code keeps them in float32 after loading, so they are NOT a source of error.
    """

    prompt_template = "### Instruction:\n{prompt}\n\n### Response:\n"

    def __init__(self, model_id: str = "togethercomputer/StripedHyena-Nous-7B", load_in_4bit: bool = False,
                 no_flash: bool = False, **quant_kwargs):
        overrides = None
        if no_flash:
            # Older GPUs (e.g. Kaggle/Colab T4) cannot run flash-attn kernels. The model's
            # config can switch them off: attention then runs in plain PyTorch. The
            # flash_attn PYTHON package is still imported for its RoPE helper (Triton,
            # works on T4), so make the package importable even without its CUDA kernels.
            _make_flash_attn_importable()
            overrides = {"use_flash_attn": False, "use_flash_rmsnorm": False,
                         "use_flash_depthwise": False, "use_flashfft": False}
        overrides = dict(overrides or {}, use_cache=True)
        rms_note = ""
        if not no_flash:
            # The config also turns on flash-attn's fused RMSNorm, which needs the separately
            # compiled `dropout_layer_norm` extension that prebuilt flash-attn wheels don't ship.
            # Keep flash ATTENTION, but compute RMSNorm in plain PyTorch if that piece is missing
            # (same formula, tiny rounding differences).
            try:
                import dropout_layer_norm  # noqa: F401
            except ImportError:
                overrides["use_flash_rmsnorm"] = False
                rms_note = "-torchrms"
                print("[StripedHyena] flash-attn attention ON; RMSNorm in PyTorch "
                      "(fused flash RMSNorm extension not installed)")
        super().__init__(model_id, load_in_4bit=load_in_4bit, trust_remote_code=True,
                         config_overrides=overrides, **quant_kwargs)
        if no_flash:
            self.name += "-noflash"
        self.name += rms_note
        self._mode = None          # decoding mode, chosen by self-test on the first question

    # Ways to feed StripedHyena its own cache. Which one its remote code expects
    # depends on its version, so the first generate() call TESTS them against
    # uncached decoding (always correct, just slower) and keeps the first that
    # reproduces it token for token.
    CACHE_MODES = ("prepare", "last_token", "full_sequence")

    def generate(self, prompt: str, max_new_tokens: int = 48) -> str:
        """
        Greedy decoding written out by hand instead of model.generate():
        recent transformers versions build their own Transformer-style cache
        inside generate(), which StripedHyena's remote code doesn't support
        (it keeps its own Hyena + attention state).
        """
        torch = self.torch
        text = self.prompt_template.format(prompt=prompt)
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
        prompt_len = ids.shape[1]
        with torch.no_grad():
            if getattr(self, "_mode", None) is None:
                out = self._decode(ids, max_new_tokens, "no_cache")       # always correct
                if out.shape[1] - prompt_len >= 4:                          # long enough to test the cache on
                    self._mode = self._choose_mode(ids, max_new_tokens, out)
            else:
                out = self._decode(ids, max_new_tokens, self._mode)
        self.last_raw = self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=False)
        return clean_answer(self.last_raw)

    def _choose_mode(self, ids, max_new_tokens, reference) -> str:
        for mode in self.CACHE_MODES:
            try:
                candidate = self._decode(ids, max_new_tokens, mode)
            except Exception as e:
                print(f"[StripedHyena] cache mode {mode!r} errored: {type(e).__name__}: {e}")
                continue
            if candidate.shape == reference.shape and bool((candidate == reference).all()):
                print(f"[StripedHyena] cached decoding mode {mode!r} matches uncached output - using it")
                return mode
            print(f"[StripedHyena] cache mode {mode!r} gives DIFFERENT tokens than uncached - rejected")
        print("[StripedHyena] no cached mode matches; using uncached decoding (correct, slower)")
        return "no_cache"

    def _has_own_prepare(self) -> bool:
        fn = getattr(type(self.model), "prepare_inputs_for_generation", None)
        return fn is not None and "transformers.generation" not in getattr(fn, "__module__", "")

    def _decode(self, ids, max_new_tokens, mode):
        torch = self.torch
        past = None
        start = ids.shape[1]
        if mode == "prepare" and not self._has_own_prepare():
            raise RuntimeError("model has no prepare_inputs_for_generation of its own")
        for step in range(max_new_tokens):
            if mode == "no_cache":
                out = self.model(input_ids=ids, use_cache=False)
            elif mode == "prepare":
                inputs = self.model.prepare_inputs_for_generation(ids, past_key_values=past, use_cache=True)
                out = self.model(**inputs)
            elif mode == "last_token":
                feed = ids if step == 0 else ids[:, -1:]
                out = self.model(input_ids=feed, use_cache=True, past_key_values=past)
            else:  # "full_sequence"
                out = self.model(input_ids=ids, use_cache=True, past_key_values=past)
            logits = out.logits if hasattr(out, "logits") else out[0]
            if mode != "no_cache":
                past = out.past_key_values if hasattr(out, "past_key_values") else out[1]
            nxt = logits[:, -1, :].float().argmax(dim=-1, keepdim=True).to(ids.device)
            ids = torch.cat([ids, nxt], dim=1)
            if self.tokenizer.eos_token_id is not None and nxt.item() == self.tokenizer.eos_token_id:
                break
            # stop at the end of the first answer line (we only keep that line anyway)
            if answer_complete(self.tokenizer.decode(ids[0, start:], skip_special_tokens=True)):
                break
        return ids


def _make_flash_attn_importable() -> None:
    """
    flash_attn/__init__.py imports its compiled CUDA extension (flash_attn_2_cuda).
    If that extension wasn't built (installed with FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE),
    register an empty stand-in so the pure-Python/Triton parts can still be imported.
    The stand-in is never called because use_flash_attn=False.
    """
    import importlib
    import sys
    import types
    try:
        importlib.import_module("flash_attn_2_cuda")
    except ImportError:
        sys.modules["flash_attn_2_cuda"] = types.ModuleType("flash_attn_2_cuda")


class OpenAICompatBackend:
    """Any local server with an OpenAI-style /v1/chat/completions endpoint."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1"):
        import requests
        self.name = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.session = requests.Session()

    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        r = self.session.post(self.url, json={
            "model": self.name, "temperature": 0, "max_tokens": max_new_tokens,
            "messages": [{"role": "user", "content": prompt}]}, timeout=120)
        r.raise_for_status()
        return clean_answer(r.json()["choices"][0]["message"]["content"])


class EchoBackend:
    """Test stand-in: answers with the first capitalised phrase after the subject, or 'I don't know'."""

    name = "echo-test"

    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        context = prompt.split("Context:\n", 1)[1].split("\n\nQuestion:", 1)[0]
        if context.strip() == "(none)":
            return "I don't know"
        words = re.findall(r"[A-Z][\w'-]+(?: [A-Z][\w'-]+)*", context.split(":", 1)[-1])
        return words[1] if len(words) > 1 else "I don't know"


# Widely used open instruct models (plain transformers, no custom code, no flash-attn needed).
MODEL_ALIASES = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",     # Apache-2.0, not gated - the default
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct", # tiny: for testing the pipeline on a laptop CPU
    "qwen3-8b": "Qwen/Qwen3-8B",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",   # gated: accept the licence on Hugging Face first
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",  # gated
}


def make_backend(spec: str, **kwargs):
    """
    "qwen2.5-7b" etc. (see MODEL_ALIASES)  -> that model, 16-bit; add "-nf4" for 4-bit ("qwen2.5-7b-nf4")
    "stripedhyena[-nf4|-fp4][-noflash]"  -> StripedHyena-Nous-7B, e.g.
        "stripedhyena"                    16-bit, flash-attn (RTX 30xx+ / L4 / A100)
        "stripedhyena-nf4"                NF4 4-bit, flash-attn       (the DL project's setup)
        "stripedhyena-nf4-noflash"        NF4 4-bit, plain attention  (free Kaggle/Colab T4)
        "stripedhyena-noflash"            16-bit, plain attention     (2x T4)
      extra kwargs: compute_dtype="float16"|"bfloat16", double_quant=True, adapter="<LoRA path>"
    "hf:<model id>"                -> any HF causal LM (same kwargs)
    "openai:<model>@<base url>"    -> local OpenAI-compatible server
    "echo"                         -> offline test stand-in
    """
    if spec.startswith("stripedhyena"):
        flags = set(spec.split("-")[1:])
        unknown = flags - {"nf4", "fp4", "4bit", "noflash"}
        if unknown:
            raise ValueError(f"unknown StripedHyena option(s): {sorted(unknown)}")
        quant = "fp4" if "fp4" in flags else "nf4" if flags & {"nf4", "4bit"} else None
        return StripedHyenaBackend(no_flash="noflash" in flags, quant=quant, **kwargs)
    base, _, q = spec.partition("-nf4")
    if base in MODEL_ALIASES:
        return HFCausalBackend(MODEL_ALIASES[base], quant="nf4" if spec.endswith("-nf4") else None, **kwargs)
    if spec.startswith("hf:"):
        return HFCausalBackend(spec[3:], **kwargs)
    if spec.startswith("openai:"):
        model, _, url = spec[7:].partition("@")
        return OpenAICompatBackend(model, url or "http://localhost:11434/v1")
    if spec == "echo":
        return EchoBackend()
    raise ValueError(f"unknown backend spec: {spec}")
