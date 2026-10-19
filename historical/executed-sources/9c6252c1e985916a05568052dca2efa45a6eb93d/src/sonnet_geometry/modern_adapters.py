"""Explicit native text-backbone adapters audited against Transformers 5.20.1.

This module deliberately excludes Gemma4 MoE, shared-KV, and per-layer-input
variants. The selected 31B dense checkpoint has none of those features.
"""
from __future__ import annotations

import gc
from typing import Any

from .adapters import DecoderAdapter, UnsupportedArchitecture

AUDITED_TRANSFORMERS_VERSION = "5.20.1"
MODERN_PARENT_TYPES = {"gemma4": ("Gemma4Model", "gemma4_text"),
                       "muse_glimmer": ("MuseGlimmerModel", "muse_glimmer_text")}
MODERN_TEXT_TYPES = {"gemma4_text": "Gemma4TextModel", "muse_glimmer_text": "MuseGlimmerTextModel"}


def _require_audited_version() -> None:
    import transformers
    if transformers.__version__ != AUDITED_TRANSFORMERS_VERSION:
        raise UnsupportedArchitecture(f"Modern adapters require audited Transformers {AUDITED_TRANSFORMERS_VERSION}; got {transformers.__version__}")


class ModernDecoderAdapter(DecoderAdapter):
    def describe(self) -> dict[str, Any]:
        result = super().describe()
        result["adapter_version"] = 2
        result["audited_transformers_version"] = AUDITED_TRANSFORMERS_VERSION
        result["text_only"] = True
        result["embedding_semantics"] = (
            "Actual first-block input after scaled word embeddings" if self.model_type == "gemma4_text"
            else "Actual first-block input after weightless RMS normalization of word embeddings"
        )
        result["layer_architecture"] = []
        for i, layer in enumerate(self.layers):
            attention = layer.self_attn
            result["layer_architecture"].append({
                "layer_index": i, "attention_type": self.model.config.layer_types[i],
                "head_dim": attention.head_dim,
                "query_heads": attention.q_proj.out_features // attention.head_dim,
                "key_value_heads": attention.k_proj.out_features // attention.head_dim,
                "distinct_value_projection": attention.v_proj is not None,
                "value_source": "v_proj" if attention.v_proj is not None else "shared pre-normalization k_proj output; separate v_norm",
                "attention_output_gate": self.model_type == "muse_glimmer_text",
                "rope_enabled": (bool(self.model.config.layer_rope_theta[i])
                                 if self.model_type == "muse_glimmer_text" else True),
            })
        return result


def get_modern_adapter(model: Any) -> ModernDecoderAdapter:
    _require_audited_version()
    config = model.config
    expected = MODERN_TEXT_TYPES.get(config.model_type)
    if expected is None or type(model).__name__ != expected or not type(model).__module__.startswith("transformers.models."):
        raise UnsupportedArchitecture("Expected an audited native Gemma4 or Muse Glimmer text backbone")
    if getattr(config, "is_encoder_decoder", False):
        raise UnsupportedArchitecture("Causal text decoder required")
    if config.model_type == "gemma4_text":
        if config.use_bidirectional_attention not in (None, "vision"):
            raise UnsupportedArchitecture("Gemma4 all-token bidirectional attention is excluded")
        if config.enable_moe_block or config.num_kv_shared_layers or config.hidden_size_per_layer_input:
            raise UnsupportedArchitecture("Gemma4 MoE, shared-KV, and per-layer-input variants require separate audits")
    if not all(hasattr(model, name) for name in ("layers", "norm", "embed_tokens")):
        raise UnsupportedArchitecture("Native text model lacks required residual modules")
    if not len(model.layers) or len(model.layers) != config.num_hidden_layers or len(config.layer_types) != len(model.layers):
        raise UnsupportedArchitecture("Text layer configuration mismatch")
    if any(kind not in {"sliding_attention", "full_attention"} for kind in config.layer_types):
        raise UnsupportedArchitecture("Unknown attention type")
    if any(not layer.self_attn.is_causal for layer in model.layers):
        raise UnsupportedArchitecture("All selected text attention layers must be causal")
    return ModernDecoderAdapter(model, config.model_type, model.layers, model.norm)


def load_text_backbone_from_parent(
    model_id: str, *, revision: str | None, dtype: Any, device: str,
    attention_implementation: str = "eager", local_files_only: bool = False,
):
    """Load checkpoint through its real parent class and verify all text weights.

    Explicit GPU placement streams weights to the H200; it does not materialize a
    60GB CPU model before .to(cuda). The native base wrapper contains no LM head.
    This lower-level helper also accepts tiny local checkpoint fixtures.
    """
    import torch
    from transformers import AutoConfig, AutoModel

    _require_audited_version()
    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False,
                                        local_files_only=local_files_only)
    if config.model_type not in MODERN_PARENT_TYPES:
        raise UnsupportedArchitecture(f"Unsupported multimodal parent: {config.model_type}")
    parent, loading = AutoModel.from_pretrained(
        model_id, revision=revision, config=config, trust_remote_code=False,
        use_safetensors=True, dtype=dtype, device_map={"": device},
        attn_implementation=attention_implementation, output_loading_info=True,
        local_files_only=local_files_only,
    )
    expected_parent, expected_text = MODERN_PARENT_TYPES[config.model_type]
    if type(parent).__name__ != expected_parent or not hasattr(parent, "language_model") or hasattr(parent, "lm_head"):
        raise UnsupportedArchitecture("Unexpected native parent/text hierarchy")
    unexpected = set(loading.get("unexpected_keys", ())) - {"lm_head.weight"}
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs") or unexpected:
        raise RuntimeError("Checkpoint loading is incomplete or incompatible; refusing research extraction")
    text = parent.language_model
    if text.config.model_type != expected_text:
        raise UnsupportedArchitecture("Wrong text backbone configuration")
    get_modern_adapter(text)
    text._sonnet_geometry_parent_config = config.to_dict()
    text._sonnet_geometry_loading_info = {key: sorted(value) if isinstance(value, set) else value for key, value in loading.items()}
    del parent
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return text.eval()


def load_modern_model(model_id: str, revision: str, *, dtype: str = "bfloat16", device: str = "cuda",
                      attention_implementation: str = "eager"):
    import torch
    from transformers import AutoTokenizer
    from .extraction import require_revision

    revision = require_revision(revision)
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Captured floating precision is required")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer offsets are required")
    model = load_text_backbone_from_parent(model_id, revision=revision, dtype=getattr(torch, dtype), device=device,
                                          attention_implementation=attention_implementation)
    model._sonnet_geometry_provenance = {
        "model_id": model_id, "revision": revision, "weights_source": "huggingface_pretrained_safetensors",
        "trust_remote_code": False, "quantized": False, "text_only_backbone": True,
    }
    return model, tokenizer
