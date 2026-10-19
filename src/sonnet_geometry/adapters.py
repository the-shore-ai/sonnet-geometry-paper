"""Fail-closed instrumentation for native Transformers causal text decoders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Module layouts are verified at runtime as well as constrained by native class.
MODEL_CLASSES = {
    "llama": "LlamaModel",
    "qwen2": "Qwen2Model",
    "qwen3": "Qwen3Model",
    "gemma": "GemmaModel",
    "gemma2": "Gemma2Model",
    "gemma3_text": "Gemma3TextModel",
}
OPTIONAL_CAPTURES = {
    "q_proj": ("self_attn.q_proj", "output", "Query linear projection before head reshape, QK normalization (if present), and RoPE"),
    "k_proj": ("self_attn.k_proj", "output", "Key linear projection before head reshape, QK normalization (if present), RoPE, and KV repetition"),
    "v_proj": ("self_attn.v_proj", "output", "Value linear projection before head reshape and KV repetition"),
    "attention_output": ("self_attn.o_proj", "output", "Attention output projection before any post-attention normalization and residual addition"),
    "mlp_intermediate": ("mlp.down_proj", "input", "Input to MLP down projection, after activation and multiplicative gate"),
    "mlp_output": ("mlp", "output", "MLP output before any post-feedforward normalization and residual addition"),
    "attention_weights": ("self_attn", "weights", "Eager attention probabilities after softmax (evaluation mode, no dropout)"),
}


class UnsupportedArchitecture(ValueError):
    pass


@dataclass(frozen=True)
class DecoderAdapter:
    model: Any
    model_type: str
    layers: Any
    norm: Any

    def module_for(self, layer: int, capture: str) -> tuple[Any, str, str]:
        if capture not in OPTIONAL_CAPTURES:
            raise UnsupportedArchitecture(f"Unknown capture: {capture}")
        path, mode, semantics = OPTIONAL_CAPTURES[capture]
        node = self.layers[layer]
        for component in path.split("."):
            if not hasattr(node, component):
                raise UnsupportedArchitecture(f"{self.model_type} layer {layer} lacks {path}; no guessed substitute")
            node = getattr(node, component)
        if not hasattr(node, "register_forward_hook"):
            raise UnsupportedArchitecture(f"{path} is not a hookable module")
        return node, mode, semantics

    def describe(self) -> dict[str, Any]:
        return {
            "adapter_version": 1,
            "model_type": self.model_type,
            "native_class": type(self.model).__name__,
            "native_module": type(self.model).__module__,
            "layer_count": len(self.layers),
            "embedding_semantics": "Actual input to first decoder block, including architecture embedding scaling",
            "block_semantics": "Output of each complete decoder block, including residual additions; final block is pre-final-norm",
            "final_normalized_semantics": "Final decoder normalization output; stored separately from residual stream",
        }


def get_adapter(model: Any) -> DecoderAdapter:
    config = model.config
    model_type = config.model_type
    if model_type in {"gemma4_text", "muse_glimmer_text"}:
        from .modern_adapters import get_modern_adapter
        return get_modern_adapter(model)
    expected = MODEL_CLASSES.get(model_type)
    if not expected or type(model).__name__ != expected or not type(model).__module__.startswith("transformers.models."):
        raise UnsupportedArchitecture(f"No verified adapter for {model_type!r}/{type(model).__name__}; use a native base text model")
    if getattr(config, "is_encoder_decoder", False) or getattr(config, "use_bidirectional_attention", False):
        raise UnsupportedArchitecture("Prefix extraction requires a causal decoder")
    for name in ("layers", "norm", "embed_tokens"):
        if not hasattr(model, name):
            raise UnsupportedArchitecture(f"Missing required module: {name}")
    if not len(model.layers) or len(model.layers) != config.num_hidden_layers:
        raise UnsupportedArchitecture("Decoder layer count does not match configuration")
    return DecoderAdapter(model, model_type, model.layers, model.norm)
