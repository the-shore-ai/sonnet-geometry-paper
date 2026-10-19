# Model selection and verification

Research decision recorded 2026-10-17 (America/Los_Angeles). Sources below were checked on that date. Model cards can change; `configs/models.lock.json`, `configs/models-candidates.lock.json`, downloaded configurations, tokenizer files, weight revisions, and each run manifest take precedence over this survey. A published checkpoint is not evidence that this repository has validated its adapter.

## Decision

Begin validation and the first corpus extraction with **`Qwen/Qwen3-8B`**, BF16, without generation or a chat template. This is an intentionally conservative, small, conventional causal-transformer reference, **not the latest Qwen release**. Its exact revision is `b968826d9c46dd6066d109eabc6255188de91218` in the initial lock file. Use the tokenizer from that same revision. Compare instruction-tuned and base checkpoints only as a separately identified sensitivity experiment.

The current larger replication candidates are **`Qwen/Qwen3.8-27B`**, **`google/gemma-4-31B-it`**, and **`meta-models/Muse-Glimmer-30B`**. They fit the single-H200 weight budget in BF16, but their newer architectures need explicit extraction adapters and numerical validation. Do not silently substitute them into a Qwen3/Llama adapter. The older, simpler candidates in the initial lock file remain fallbacks, not claims about the newest models.

Independent unauthenticated Hugging Face API checks resolved the following immutable revisions, and their SHA-specific configurations were downloaded and SHA-256 hashed in the candidate lock file. All three report ungated Apache-2.0 metadata. No full weight download or forward pass was performed by this metadata check.

| Candidate | Revision | Registered architecture |
|---|---|---|
| Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `Qwen3_5ForConditionalGeneration` |
| Gemma4-31B-it | `842da3794eaa0b77d5f08bae87a17459d91ff475` | `Gemma4ForConditionalGeneration` |
| Muse-Glimmer-30B | `a4e59da52a7bc87ae7251dd5545c0dd437c44b68` | `MuseGlimmerForConditionalGeneration` |

Muse Spark Open metadata resolves to `0fb02478b02565b2ce7fc57e6051c91172774ac9`, but its public config fetch returned HTTP 401 and API metadata reports a manual gate. Its architecture metadata identifies `MuseSparkForConditionalGeneration`. This distinction matters: readable documentation does not imply downloadable checkpoint access.

## Verified checkpoint survey

| Checkpoint | Architecture / accessibility | License | Approximate BF16 weights only | Decision |
|---|---|---|---:|---|
| `Qwen/Qwen3-8B` | 36 dense causal blocks, width 4096; GQA 32 query / 8 KV heads, head width 128; ordinary Qwen3 support in Transformers >=4.51 | Apache 2.0 | 16.4 GB (8.2B × 2 bytes) | Initial validation and pilot |
| `Qwen/Qwen3.8-27B` | 64 blocks, width 5120; repeated groups of three Gated DeltaNet blocks and one gated-attention block; dense FFNs, multimodal wrapper | Apache 2.0 as displayed on model repository | 54 GB nominal | Latest fitting Qwen candidate found; hybrid adapter required |
| `google/gemma-4-31B-it` | 60 text blocks, width 5376; five sliding-attention layers then one global layer; window 1024; global/local head sizes differ, global keys and values tied | Apache 2.0 | About 61.4 GB (30.7B nominal) | Current dense Gemma replication candidate; text-backbone adapter required |
| `meta-models/Muse-Glimmer-30B` | 52 text blocks, width 6656; local/local/local/global pattern, window 2048; GQA 32/2; multimodal wrapper | Apache 2.0 | About 59.2–60 GB | Current fitting Meta candidate found; Muse-specific adapter required |
| `google/gemma-3-12b-pt` | Gemma 3 multimodal wrapper with text backbone; local/global attention | Gemma terms; initial API check reports manual gate | About 24 GB nominal | Older fallback; requires account access and adapter validation |
| `meta-llama/Llama-3.1-8B` | Dense causal decoder, GQA; conventional Llama instrumentation | Llama 3.1 Community License; initial API check reports manual gate | About 16 GB nominal | Older practical Llama fallback; requires account access |

Architecture facts come from the official [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-8B) and [configuration](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json), [Qwen3.8 card](https://huggingface.co/Qwen/Qwen3.8-27B), [Gemma 4 card](https://huggingface.co/google/gemma-4-31B-it) and [configuration](https://huggingface.co/google/gemma-4-31B-it/raw/main/config.json), and [Muse Glimmer card](https://huggingface.co/meta-models/Muse-Glimmer-30B) and [configuration](https://huggingface.co/meta-models/Muse-Glimmer-30B/raw/main/config.json). The memory column is arithmetic from nominal parameter counts, not measured peak GPU memory. GB here is decimal; preserve actual CUDA byte counts in the run record.

## What “Meta Muse Open” refers to

The name should not be dismissed as nonexistent. Meta's official documentation identifies **Muse Spark Open**, announced October 15, 2026, with BF16 checkpoint **`meta-models/Muse-Spark-Open-816B-A42B`**. This is an 816B-total, 42B-active sparse mixture-of-experts model with a custom Muse Model License. Meta reports about **1.65 TB of BF16 weights**; the NVFP4 release is still about 537 GB. Active parameters determine only part of compute, not total weight residency. Neither checkpoint is appropriate for the requested single-H200, unquantized experiment. See [Meta's model documentation](https://dev.meta.ai/docs/muse-spark-open) and [checkpoint table](https://dev.meta.ai/docs/muse-spark-open/get-the-model).

Meta separately provides **Muse Glimmer**, a smaller dense checkpoint. Its official download page lists an ungated BF16 release. It is distilled from Muse Spark: a model from a different organization is not necessarily independently trained from scratch, and inter-family agreement cannot establish independent training-data provenance. [Meta Glimmer overview](https://dev.meta.ai/docs/muse-glimmer), [download documentation](https://dev.meta.ai/docs/muse-glimmer/get-the-model).

For a Llama-only replication, use the gated 3.1 8B fallback at native BF16 precision. Llama 3.3 70B weights alone consume roughly 140 GB, leaving inadequate comfortable extraction headroom on an H200; Llama 4 Scout is 109B total despite 17B active, roughly 218 GB BF16. Neither is a sensible single-device baseline here. [Llama 3.3 card](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct), [Scout card](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct), [Llama 3.1 card and license](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

## H200 and extraction compatibility

NVIDIA specifies 141 GB HBM3e for H200. Fit must include weights, forward working memory, CUDA allocation overhead, captured tensors, and any attention matrices; nominal parameter arithmetic is only a lower bound. No gradients, optimizer, generation, or KV cache are needed for a full-poem forward pass. Capture to CPU promptly and measure peak allocation on one poem before starting a corpus job. [NVIDIA specification](https://nvidianews.nvidia.com/news/nvidia-supercharges-hopper-the-worlds-leading-ai-computing-platform).

For a model with L blocks, token count T and hidden width D, one embedding plus L block-output streams requires approximately `2*(L+1)*T*D` bytes at BF16. Qwen3-8B at T=256 needs about 77.6 MB per sonnet or 12.0 GB for 154 sonnets for these streams alone. This is an illustrative token count, not a corpus measurement. Full attention probabilities scale as `L*heads*T*T`; MLP intermediates can exceed residual storage. Keep every requested activation's estimate separate and use actual token counts for launch approval.

`output_hidden_states=True` is not a universal residual-stream contract. A library may return its final normalized state in place of the final raw block output; embeddings may be scaled before entering the first block. Hook the input to the first decoder block and the output of every block, label those semantics, and retain final normalization separately. Queries captured at a projection are pre-RoPE and potentially pre-normalization; keys/values can have fewer heads than queries. Document pre/post transformation semantics in each tensor manifest. Do not infer attention weights from projection outputs.

Gemma4 and Muse use nested text configurations and model wrappers. Qwen3.8 mixes recurrent/linear-attention and softmax-attention blocks; it does not have an ordinary attention-probability matrix at every layer. Uniformly named tensors are not uniform mathematical objects. Before enabling an adapter, pin Transformers and implementation versions, verify hooks/shapes against the actual module tree, and run causal-prefix and save/reload checks. A Transformers version embedded in a config is descriptive metadata, not a proven minimum supported version.

### Native package audit

The published `transformers==5.20.1` wheel was downloaded from PyPI and inspected without installing it into the research environment. Wheel SHA-256: `5f51ce444193e5d0a751e3fa489ddbc2224ff12f6675a5cfefd4044b09757c19`. It contains native `gemma4` and `muse_glimmer` modeling modules and conditional-generation architecture registrations. Gemma4 also has a text-only causal-model registration; Muse Glimmer should be loaded through its verified native architecture or multimodal registration, not assumed to work through a generic Llama loader. This establishes available upstream code, not a completed local adapter or successful checkpoint forward pass. [PyPI version metadata](https://pypi.org/pypi/transformers/5.20.1/json).

Adjacent stable wheel inventories show Gemma4 code absent in 5.4.0 and present in 5.5.0; Muse Glimmer code absent in 5.14.0 and present in 5.15.0. These bound initial family availability in stable minor releases, but do not guarantee that every present-day checkpoint works with that first implementation. For replication prefer the audited 5.20.1 release after tests. Its declared requirements include Python >=3.10, Hugging Face Hub >=1.31 and <3.0, tokenizers >=0.23.1 and <0.24, and safetensors >=0.8; its PyTorch extra requires torch >=2.6 and accelerate >=1.1. Pin the actual tested environment rather than mixing an old dependency lock with a new model family.

## Sparse features and model independence

Gemma Scope originally targets Gemma 2; Gemma Scope 2 targets Gemma 3. Their existence does **not** make those SAEs compatible with Gemma4, Qwen, or Muse. Any SAE experiment must pin the exact source checkpoint, layer, hook site, normalization and dictionary revision and report reconstruction fidelity. Leave sparse-feature extraction optional until this contract is satisfied. [Gemma Scope paper](https://aclanthology.org/2024.blackboxnlp-1.19/), [current Google Scope documentation](https://ai.google.dev/gemma/docs/gemma_scope).

## Required preflight record

For every selected model save resolved commit SHA; downloaded config and tokenizer hashes; architecture class; total and trainable parameter counts; weight dtype and any quantization flag; tokenizer implementation/version and special-token policy; attention implementation; extraction hooks; runtime package versions; CUDA/driver/GPU information; source code commit; and a measured short forward pass. Access, successful download, and successful instrumented execution are three distinct statuses. No model in a survey table counts as a completed replication.
