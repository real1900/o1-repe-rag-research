"""Concrete `LlamaModelRunner` -- implements the duck-typed runner interface
the protocol expects.

Loads a HuggingFace causal LM (anything with `model.model.layers` -- Llama,
Qwen2, Mistral, etc.), exposes mean-pooled activations at any layer, and
generates with an optional ablation hook at a target layer.

Device-agnostic (CUDA / MPS / CPU). Matches the patterns from
pilot_real_rag.py so behaviors are consistent across the project.
"""
from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _make_ablate_hook(unit: torch.Tensor):
    """Build a forward hook that ablates (projects out) the `unit` direction
    from EVERY position of the residual stream at this layer.

    Ablation removes the FULL component along `unit` -- scale-free, no alpha.

    Critical: we ablate at every position, not just position [-1]. The earlier
    last-position-only variant gave near-null effects across all 12 corpus
    tasks because during prefill the K/V cache for positions 0..N-2 is stored
    unablated -- downstream attention then attends back through that unablated
    prefix. The all-positions variant gives Δbase=-0.65 on refusal at layer 7
    (refusal rate 0.65 -> 0.00); the last-position variant gives -0.05 (noise).
    See probe_hook.py / probe_hook_results.json for the diagnostic sweep.
    """
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        hf = h.float()
        coef = hf @ unit                                  # [batch, seq]
        hn = h - (coef.unsqueeze(-1) * unit).to(h.dtype)  # ablate ALL positions
        if isinstance(output, tuple):
            return (hn,) + output[1:]
        return hn
    return hook


class LlamaModelRunner:
    """Runner for Llama-style models (anything with `model.model.layers`).

    Implements the duck-typed `ModelRunner` interface used by
    `protocol.evaluate_task`:
      - `mean_act(texts, layer) -> [len(texts), d_model] float32 tensor`
      - `generate(prompt, hook, layer, max_new_tokens) -> str`
    """

    def __init__(self, model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
                 dtype: torch.dtype = torch.float16,
                 device: str | None = None,
                 max_input_tokens: int = 1024):
        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.device = device
        self.model_id = model_id
        self.max_input_tokens = max_input_tokens

        print(f"[runner] loading {model_id} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
        self.model.eval()
        self.layers = self.model.model.layers
        self.d_model = self.model.config.hidden_size
        print(f"[runner] ready: hidden={self.d_model}, "
              f"layers={len(self.layers)}, device={device}")

    # ----------------------------------------------------------------------
    # mean_act: mean-pooled hidden state at `layer` for each text
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def mean_act(self, texts: list[str], layer: int) -> torch.Tensor:
        """Return float32 [len(texts), d_model] of mean-pooled hidden states at `layer`."""
        cache = {}
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            cache["v"] = h.detach()

        outs = []
        for text in texts:
            inp = self.tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=self.max_input_tokens).to(self.device)
            handle = self.layers[layer].register_forward_hook(hook)
            try:
                self.model(**inp)
            finally:
                handle.remove()
            # mean over sequence -> [d_model], float32 for stable downstream math
            outs.append(cache["v"].mean(dim=1).squeeze(0).float())
        return torch.stack(outs)  # [n, d_model]

    # ----------------------------------------------------------------------
    # generate: greedy decoding with optional ablation hook at a target layer
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt: str, hook: tuple | None, layer: int,
                 max_new_tokens: int = 64) -> str:
        """Greedy continuation. `hook` is None or `("ablate", unit_direction)`."""
        inp = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=self.max_input_tokens * 2).to(self.device)
        in_len = inp.input_ids.shape[1]

        handle = None
        if hook is not None:
            mode, unit = hook
            if mode != "ablate":
                raise NotImplementedError(f"Unknown hook mode: {mode}")
            unit = unit.to(self.device).float()
            handle = self.layers[layer].register_forward_hook(_make_ablate_hook(unit))

        try:
            out = self.model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        finally:
            if handle is not None:
                handle.remove()

        return self.tokenizer.decode(out[0][in_len:], skip_special_tokens=True)


# --------------------------------------------------------------------------
# Self-contained smoke test: `python -m steering_taxonomy.runner`
# --------------------------------------------------------------------------
if __name__ == "__main__":
    runner = LlamaModelRunner()

    # mean_act on two tiny texts
    acts = runner.mean_act(
        ["The capital of France is Paris.", "The capital of Japan is Tokyo."],
        layer=7,
    )
    print(f"[smoke] mean_act shape={tuple(acts.shape)}, "
          f"dtype={acts.dtype}, device={acts.device}")

    # generate without a hook
    no_hook = runner.generate("The capital of France is",
                              hook=None, layer=7, max_new_tokens=8)
    print(f"[smoke] generate (no hook): {no_hook!r}")

    # generate with a random ablation hook (should still produce a string)
    rand = torch.randn(runner.d_model)
    rand = rand / rand.norm()
    with_hook = runner.generate("The capital of France is",
                                hook=("ablate", rand), layer=7, max_new_tokens=8)
    print(f"[smoke] generate (random ablate): {with_hook!r}")
    print("[smoke] OK")
