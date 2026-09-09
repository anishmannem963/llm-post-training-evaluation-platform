# LLM Post-Training & Evaluation Platform

A reproducible ML engineering platform for post-training open-source language models, evaluating model quality, and benchmarking inference efficiency.

## What is implemented

### Iteration 1 — Baseline Foundation

The baseline pipeline:

1. Loads and deterministically samples an instruction dataset.
2. Normalizes examples into prompt/reference pairs.
3. Runs an untouched instruction-tuned base model.
4. Saves every prediction to JSONL.
5. Records latency, generated-token throughput, model, and device metadata.
6. Produces the frozen baseline used by later fine-tuning comparisons.

Default baseline:

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset: `yahma/alpaca-cleaned`
- Samples: 25 for the initial smoke benchmark
- Devices: CUDA, Apple MPS, or CPU selected automatically

### Iteration 2 — SFT + LoRA

The post-training pipeline adds:

- conversational prompt-completion formatting for TRL
- completion-only supervised fine-tuning
- PEFT LoRA adapters on Qwen attention projections
- trainable/total parameter accounting and parameter-reduction reporting
- checkpoint and training-metric persistence
- an M1-friendly smoke-training profile using small batches and gradient accumulation

Default LoRA smoke run:

- LoRA rank: 16
- LoRA alpha: 32
- Target modules: `q_proj`, `v_proj`
- Training examples: 500
- Max training steps: 50
- Effective batch size: 8 through gradient accumulation
- Maximum sequence length: 512

### Iteration 3 — Held-out Base vs LoRA Evaluation

The evaluation pipeline now:

- reserves a deterministic held-out slice starting after the 500 training examples
- loads either the untouched base model or the saved PEFT adapter
- generates responses for the same 100 held-out prompts
- measures token F1 and ROUGE-L F1 against references
- measures mean/p95 latency and generated-token throughput
- writes per-example predictions plus base, LoRA, and delta summaries
- releases accelerator memory between model runs for 8 GB Apple Silicon machines

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Run the untouched baseline:

```bash
python scripts/run_baseline.py --config configs/baseline.yaml
```

Train the LoRA adapter:

```bash
python scripts/train_sft_lora.py --config configs/sft_lora.yaml
```

Evaluate the frozen base model against the trained adapter:

```bash
python scripts/evaluate_base_vs_lora.py --config configs/evaluation.yaml
```

Training prints the number and percentage of trainable parameters before optimization. The final adapter and `training_metrics.json` are written under `checkpoints/qwen2.5-0.5b-lora/`.

## Generated artifacts

Experiment artifacts are intentionally ignored by Git:

- `outputs/baseline_predictions.jsonl`
- `outputs/base_vs_lora_predictions.jsonl`
- `benchmarks/results/baseline_metrics.json`
- `benchmarks/results/base_vs_lora_metrics.json`
- `checkpoints/`

## Roadmap

- [x] Iteration 1: dataset + base-model inference + baseline benchmarking
- [x] Iteration 2: supervised fine-tuning + LoRA training pipeline
- [x] Iteration 3: held-out base-vs-adapter quality and systems evaluation
- [ ] Iteration 4: preference optimization (DPO)
- [ ] Iteration 5: quantization and efficiency benchmarking
- [ ] Iteration 6: model serving API
- [ ] Iteration 7: reproducible benchmark report and final documentation

## Benchmark integrity

Repository documentation reports measured results only after an experiment has been executed. Target or estimated resume metrics are not recorded as benchmark results.
