# LLM Post-Training & Evaluation Platform

A reproducible ML engineering platform for post-training open-source language models, evaluating model quality, and benchmarking inference efficiency.

## Iteration 1 — Baseline Foundation

The first milestone establishes a reproducible baseline before any fine-tuning:

1. Load and deterministically sample an instruction dataset.
2. Normalize examples into prompt/reference pairs.
3. Run an untouched instruction-tuned base model.
4. Save every prediction to JSONL.
5. Record latency, generated-token throughput, model, and device metadata.
6. Use the resulting artifacts as the baseline for later SFT/LoRA comparisons.

### Default experiment

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset: `yahma/alpaca-cleaned`
- Samples: 25 for the initial smoke benchmark
- Devices: CUDA, Apple MPS, or CPU selected automatically

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python scripts/run_baseline.py --config configs/baseline.yaml
```

Generated artifacts are intentionally ignored by Git:

- `outputs/baseline_predictions.jsonl`
- `benchmarks/results/baseline_metrics.json`

## Roadmap

- [x] Iteration 1: dataset + base-model inference + baseline benchmarking
- [ ] Iteration 2: supervised fine-tuning + LoRA
- [ ] Iteration 3: model-quality evaluation framework
- [ ] Iteration 4: preference optimization (DPO)
- [ ] Iteration 5: quantization and efficiency benchmarking
- [ ] Iteration 6: model serving API
- [ ] Iteration 7: reproducible benchmark report and final documentation

## Benchmark integrity

Repository documentation reports measured results only after an experiment has been executed. Target or estimated resume metrics are not recorded as benchmark results.
