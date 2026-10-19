"""Native modern adapters; isolated upgrade fixtures, never pretrained downloads."""
import json

import pytest
import transformers

if transformers.__version__ != "5.20.1":
    pytest.skip("Modern adapters are audited only with Transformers 5.20.1", allow_module_level=True)

import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig, Gemma4TextModel,
                          MuseGlimmerConfig, MuseGlimmerForConditionalGeneration, MuseGlimmerTextConfig,
                          MuseGlimmerTextModel, MuseGlimmerVisionConfig, PreTrainedTokenizerFast)

from sonnet_geometry.adapters import UnsupportedArchitecture, get_adapter
from sonnet_geometry import extraction
from sonnet_geometry.modern_adapters import get_modern_adapter, load_text_backbone_from_parent


def sample_inputs():
    raw = Tokenizer(WordLevel({"[UNK]": 0, "my": 1, "sun": 2, "love": 3}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
    record = {"sonnet_id": "sonnet-001", "number": 1, "text": "my sun\nmy love", "lines": ["my sun", "my love"],
              "line_spans": [{"line_number": 1, "start": 0, "end": 6, "source_start": 0},
                             {"line_number": 2, "start": 7, "end": 14, "source_start": 7}],
              "stanza_spans": [{"stanza_number": 1, "start": 0, "end": 14}]}
    return tokenizer, record


def model_config(kind):
    common = dict(vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8, pad_token_id=0,
                  bos_token_id=1, eos_token_id=2, sliding_window=8, max_position_embeddings=64,
                  layer_types=["sliding_attention", "full_attention"])
    if kind == "gemma":
        config = Gemma4TextConfig(
            **common, global_head_dim=16, num_global_key_value_heads=1,
            attention_k_eq_v=True, hidden_size_per_layer_input=0, num_kv_shared_layers=0,
            enable_moe_block=False, use_bidirectional_attention="vision",
            rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 10000.},
                             "full_attention": {"rope_type": "proportional", "rope_theta": 1000000., "partial_rotary_factor": .25}},
        )
    else:
        config = MuseGlimmerTextConfig(**common, layer_rope_theta=[500000., 0.])
    config._attn_implementation = "eager"
    return config


def wrapper(kind):
    text = model_config(kind)
    if kind == "gemma":
        config = Gemma4Config(text_config=text, vision_config=None, audio_config=None)
        cls = Gemma4ForConditionalGeneration
    else:
        vision = MuseGlimmerVisionConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=1,
                                        num_attention_heads=4, patch_size=2, pos_emb_height=4,
                                        pos_emb_width=4, merge_size=2)
        config = MuseGlimmerConfig(text_config=text, vision_config=vision,
                                  out_hidden_size=128, projector_hidden_size=32)
        cls = MuseGlimmerForConditionalGeneration
    config._attn_implementation = "eager"
    return cls(config).eval()


@pytest.mark.parametrize("kind", ["gemma", "muse"])
def test_parent_checkpoint_loads_exact_text_weights_without_lm_head(tmp_path, kind):
    torch.manual_seed(91)
    original = wrapper(kind)
    original.save_pretrained(tmp_path)
    loaded = load_text_backbone_from_parent(str(tmp_path), revision=None, dtype=torch.float32,
                                           device="cpu", local_files_only=True)
    assert not hasattr(loaded, "lm_head")
    assert not loaded._sonnet_geometry_loading_info["missing_keys"]
    assert loaded._sonnet_geometry_parent_config["model_type"] in {"gemma4", "muse_glimmer"}
    for name, value in loaded.state_dict().items():
        assert torch.equal(value, original.model.language_model.state_dict()[name]), name
    tokenizer, record = sample_inputs()
    manifest = extraction.extract_sonnet(
        loaded, tokenizer, record, tmp_path / "outputs", model_id=f"local/{kind}-saved-random",
        revision="0" * 40, run_id="parent", provenance={"weights_source": "random_initialization"},
    )
    assert "parent-model-config.json" in manifest["artifacts"]
    assert "loading-info.json" in manifest["artifacts"]
    assert manifest["request"]["parent_config_sha256"] == manifest["artifacts"]["parent-model-config.json"]["sha256"]
    saved_parent = json.loads((tmp_path / "outputs/parent/parent-model-config.json").read_text())
    assert saved_parent["model_type"] == original.config.model_type


def test_missing_pretrained_text_weight_fails_closed(tmp_path):
    original = wrapper("muse")
    original.save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    del tensors["model.language_model.layers.0.self_attn.q_proj.weight"]
    save_file(tensors, path, metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="incomplete"):
        load_text_backbone_from_parent(str(tmp_path), revision=None, dtype=torch.float32,
                                      device="cpu", local_files_only=True)


@pytest.mark.parametrize("kind", ["gemma", "muse"])
def test_all_residuals_hooks_and_prefix_validation(tmp_path, kind):
    torch.manual_seed(92)
    model = (Gemma4TextModel if kind == "gemma" else MuseGlimmerTextModel)(model_config(kind)).eval()
    adapter = get_adapter(model)
    assert adapter.describe()["layer_architecture"][1]["attention_type"] == "full_attention"
    if kind == "gemma":
        assert not adapter.describe()["layer_architecture"][1]["distinct_value_projection"]
        with pytest.raises(UnsupportedArchitecture):
            adapter.module_for(1, "v_proj")
    else:
        assert not adapter.describe()["layer_architecture"][1]["rope_enabled"]
        assert adapter.describe()["layer_architecture"][0]["attention_output_gate"]
    tokenizer, record = sample_inputs()
    extraction.extract_sonnet(model, tokenizer, record, tmp_path, model_id=f"local/{kind}-random",
                              revision="0" * 40, run_id="fixture", provenance={"weights_source": "random_initialization"},
                              capture=["q_proj", "k_proj", "v_proj", "attention_output", "mlp_intermediate", "mlp_output", "attention_weights"],
                              capture_layers=[0])
    tensors = load_file(tmp_path / "fixture" / "activations.safetensors")
    assert len(tensors) == 11
    assert tensors["residual.block.001"].shape == (4, 32)
    assert tensors["intermediate.block.000.attention_weights"].shape == (4, 4, 4)
    with torch.inference_mode():
        assert torch.equal(model.norm(tensors["residual.block.001"]), tensors["final_normalized"])
    validation = extraction.validate_causal_prefixes(model, record, tmp_path / "fixture", line_numbers=(1, 2))
    assert validation["eligible_count"] == 2
    assert validation["passed"]


def test_gemma_unreviewed_variants_fail_closed():
    config = model_config("gemma")
    model = Gemma4TextModel(config)
    config.num_kv_shared_layers = 1
    with pytest.raises(UnsupportedArchitecture, match="separate audits"):
        get_modern_adapter(model)
