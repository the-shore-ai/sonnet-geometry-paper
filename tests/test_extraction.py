"""Offline architecture and numerical checks; no pretrained downloads."""
import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (Gemma2Config, Gemma2Model, Gemma3TextConfig, Gemma3TextModel,
                          LlamaConfig, LlamaModel, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, Qwen3Model)

from sonnet_geometry.adapters import UnsupportedArchitecture, get_adapter
from sonnet_geometry.extraction import extract_sonnet, load_model, require_revision, validate_causal_prefixes
from sonnet_geometry.storage import IntegrityError, verify_run

REVISION = "0" * 40


@pytest.fixture
def record():
    return {"sonnet_id": "sonnet-001", "number": 1, "text": "my sun\nmy love", "lines": ["my sun", "my love"],
            "line_spans": [{"line_number": 1, "start": 0, "end": 6, "source_start": 0},
                           {"line_number": 2, "start": 7, "end": 14, "source_start": 7}],
            "stanza_spans": [{"stanza_number": 1, "start": 0, "end": 14}]}


@pytest.fixture
def tokenizer():
    raw = Tokenizer(WordLevel({"[UNK]": 0, "my": 1, "sun": 2, "love": 3}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")


def tiny_model(config_class, model_class):
    torch.manual_seed(314)
    config = config_class(vocab_size=8, hidden_size=32, intermediate_size=48,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=8, max_position_embeddings=64, attention_dropout=0.0,
                          sliding_window=16, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    config._attn_implementation = "eager"
    return model_class(config).eval()


def extract(model, tokenizer, record, output_dir, **kwargs):
    return extract_sonnet(model, tokenizer, record, output_dir,
                          model_id="local/tiny-random", revision=REVISION,
                          provenance={"weights_source": "random_initialization"}, **kwargs)


@pytest.mark.parametrize("config_class,model_class", [
    (LlamaConfig, LlamaModel), (Qwen3Config, Qwen3Model),
    (Gemma2Config, Gemma2Model), (Gemma3TextConfig, Gemma3TextModel),
])
def test_residual_stream_final_norm_and_optional_hooks(tmp_path, tokenizer, record, config_class, model_class):
    model = tiny_model(config_class, model_class)
    capture = ["q_proj", "k_proj", "v_proj", "attention_output", "mlp_intermediate", "mlp_output", "attention_weights"]
    manifest = extract(model, tokenizer, record, tmp_path, run_id="test", capture=capture, capture_layers=[0])
    run = tmp_path / "test"
    tensors = load_file(run / "activations.safetensors")
    assert len(tensors) == 11
    assert tensors["residual.embedding"].shape == (4, 32)
    assert tensors["intermediate.block.000.q_proj"].shape == (4, 32)
    assert tensors["intermediate.block.000.k_proj"].shape == (4, 16)
    assert tensors["intermediate.block.000.mlp_intermediate"].shape == (4, 48)
    assert tensors["intermediate.block.000.attention_weights"].shape == (4, 4, 4)
    assert torch.allclose(tensors["intermediate.block.000.attention_weights"].sum(-1), torch.ones(4, 4))
    assert not torch.equal(tensors["residual.block.001"], tensors["final_normalized"])
    with torch.inference_mode():
        assert torch.equal(model.norm(tensors["residual.block.001"]), tensors["final_normalized"])
        expected = model(torch.tensor([[1, 2, 1, 3]]), output_hidden_states=True, use_cache=False)
    assert torch.equal(tensors["residual.embedding"], expected.hidden_states[0][0])
    assert torch.equal(tensors["residual.block.000"], expected.hidden_states[1][0])
    assert torch.equal(tensors["final_normalized"], expected.last_hidden_state[0])
    assert manifest["execution"]["logits"] is False
    assert manifest["request"]["weights_source"] == "random_initialization"
    assert verify_run(run)["run_id"] == "test"
    # Newer Transformers retains its own output-recorder hooks. Extraction must
    # remove its handles without deleting hooks owned by the framework/caller.
    assert all(getattr(hook, "__module__", None) != "sonnet_geometry.extraction"
               for module in model.modules()
               for hook in list(module._forward_hooks.values()) + list(module._forward_pre_hooks.values()))


def test_prefix_numerical_equivalence_and_resume(tmp_path, tokenizer, record):
    model = tiny_model(LlamaConfig, LlamaModel)
    first = extract(model, tokenizer, record, tmp_path, run_id="full")
    tokens = json.loads((tmp_path / "full" / "tokens.json").read_text())
    assert all(p["exact_token_prefix"] for p in tokens["prefix_checks"])
    tensors = load_file(tmp_path / "full" / "activations.safetensors")
    with torch.inference_mode():
        prefix = model(torch.tensor([[1, 2]]), output_hidden_states=True, use_cache=False)
    torch.testing.assert_close(prefix.last_hidden_state[0], tensors["final_normalized"][:2], atol=1e-6, rtol=1e-5)
    second = extract(model, tokenizer, record, tmp_path, run_id="full", resume=True)
    assert first == second
    with pytest.raises(FileExistsError):
        extract(model, tokenizer, record, tmp_path, run_id="full")
    with pytest.raises(ValueError, match="differs"):
        extract(model, tokenizer, record, tmp_path, run_id="full", resume=True, capture=["q_proj"], capture_layers=[0])
    (tmp_path / "full" / "tokens.json").write_text("tampered")
    with pytest.raises(IntegrityError):
        extract(model, tokenizer, record, tmp_path, run_id="full", resume=True)


def test_fail_closed_and_restore_hooks(tmp_path, tokenizer, record):
    model = tiny_model(LlamaConfig, LlamaModel)
    with pytest.raises(ValueError, match="40-character"):
        require_revision("main")
    with pytest.raises(ValueError, match="Explicit capture_layers"):
        extract(model, tokenizer, record, tmp_path, capture=["q_proj"])
    model.config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="eager"):
        extract(model, tokenizer, record, tmp_path, capture=["attention_weights"], capture_layers=[0])
    model.config.model_type = "unknown_future_model"
    with pytest.raises(UnsupportedArchitecture):
        get_adapter(model)


def test_bfloat16_roundtrip(tmp_path, tokenizer, record):
    model = tiny_model(LlamaConfig, LlamaModel).to(torch.bfloat16)
    extract(model, tokenizer, record, tmp_path, run_id="bf16")
    tensors = load_file(tmp_path / "bf16" / "activations.safetensors")
    assert all(t.dtype == torch.bfloat16 for t in tensors.values())
    with torch.inference_mode():
        expected = model(torch.tensor([[1, 2, 1, 3]]), use_cache=False).last_hidden_state[0]
    assert torch.equal(expected, tensors["final_normalized"])
    original_state = {name: value.clone() for name, value in model.state_dict().items()}
    original_buffers = {name: value.clone() for name, value in model.named_buffers()}
    validation = validate_causal_prefixes(model, record, tmp_path / "bf16", line_numbers=(1, 2, 14),
                                          fp32_diagnostic=True)
    assert validation["dtype"] == "torch.bfloat16"
    assert validation["eligible_count"] == 2
    assert validation["passed"]
    assert validation["fixed_shape_causality"]["passed"]
    assert validation["fixed_shape_causality"]["eligible_count"] == 1
    assert validation["fp32_diagnostic"]["enabled"]
    assert validation["fp32_diagnostic"]["passed"]
    assert len(validation["results"][0]["layers"]) == 4
    assert validation["results"][-1]["eligible"] is False
    for name, value in model.state_dict().items():
        assert value.dtype == original_state[name].dtype
        assert torch.equal(value, original_state[name])
    for name, value in model.named_buffers():
        assert value.dtype == original_buffers[name].dtype
        assert torch.equal(value, original_buffers[name])


def test_validation_reports_length_sensitivity_separately_from_causality(tmp_path, tokenizer, record):
    model = tiny_model(LlamaConfig, LlamaModel)
    extract(model, tokenizer, record, tmp_path, run_id="baseline")

    def length_sensitive_norm(_module, arguments):
        value = arguments[0]
        return (value + 0.25,) if value.shape[1] < 4 else arguments

    handle = model.norm.register_forward_pre_hook(length_sensitive_norm)
    try:
        validation = validate_causal_prefixes(model, record, tmp_path / "baseline", line_numbers=(1,))
    finally:
        handle.remove()
    assert validation["variable_length_equivalent"] is False
    assert validation["fixed_shape_causality"]["passed"] is True
    assert validation["passed"] is True
    assert validation["pass_criterion"] == "bitwise_future_token_invariance_at_fixed_full_sequence_shape"


def test_future_token_leakage_fails_causal_validation(tmp_path, tokenizer, record):
    model = tiny_model(LlamaConfig, LlamaModel)

    def leaking_norm(_module, arguments):
        value = arguments[0]
        return (value + value.mean(dim=1, keepdim=True),)

    handle = model.norm.register_forward_pre_hook(leaking_norm)
    try:
        extract(model, tokenizer, record, tmp_path, run_id="leaking")
        validation = validate_causal_prefixes(model, record, tmp_path / "leaking", line_numbers=(1,))
    finally:
        handle.remove()
    assert validation["fixed_shape_causality"]["passed"] is False
    assert validation["passed"] is False


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_standard_checkpoint_loads_complete_base_and_persists_info(tmp_path, tokenizer, record, tie_word_embeddings):
    config = tiny_model(Qwen3Config, Qwen3Model).config
    config.tie_word_embeddings = tie_word_embeddings
    original = Qwen3ForCausalLM(config).eval()
    original.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    model, loaded_tokenizer = load_model(str(tmp_path), REVISION, dtype="float32", device="cpu")
    loading = model._sonnet_geometry_loading_info
    assert not loading["missing_keys"]
    assert not loading["mismatched_keys"]
    assert set(loading["unexpected_keys"]) <= {"lm_head.weight"}
    for name, value in model.state_dict().items():
        assert torch.equal(value, original.model.state_dict()[name]), name
    manifest = extract_sonnet(model, loaded_tokenizer, record, tmp_path / "outputs", model_id=str(tmp_path),
                              revision=REVISION, run_id="checkpoint")
    saved = json.loads((tmp_path / "outputs/checkpoint/loading-info.json").read_text())
    assert saved == loading
    assert "loading-info.json" in manifest["artifacts"]
    verify_run(tmp_path / "outputs/checkpoint")


@pytest.mark.parametrize("fault", ["missing", "unexpected"])
def test_standard_checkpoint_incomplete_or_extra_weights_fail_closed(tmp_path, tokenizer, fault):
    original = Qwen3ForCausalLM(tiny_model(Qwen3Config, Qwen3Model).config)
    original.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    if fault == "missing":
        del tensors["model.layers.0.self_attn.q_proj.weight"]
    else:
        tensors["unrecognized_adapter.weight"] = torch.ones(2, 2)
    save_file(tensors, path, metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="incomplete or incompatible"):
        load_model(str(tmp_path), REVISION, dtype="float32", device="cpu")
