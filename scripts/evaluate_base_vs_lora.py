from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import yaml

from llm_platform.benchmark import summarize, write_json
from llm_platform.data import load_instruction_samples
from llm_platform.evaluation import summarize_quality
from llm_platform.inference import BaseModelRunner


def release_accelerator_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_model(
    label: str,
    model_name: str,
    adapter_path: str | None,
    samples,
    max_new_tokens: int,
) -> list[dict]:
    runner = BaseModelRunner(model_name, adapter_path=adapter_path)
    records: list[dict] = []

    for index, sample in enumerate(samples):
        result = runner.generate(sample["prompt"], max_new_tokens=max_new_tokens)
        records.append(
            {
                "model": label,
                "id": index,
                "prompt": sample["prompt"],
                "reference": sample["reference"],
                "prediction": result.response,
                "latency_seconds": result.latency_seconds,
                "generated_tokens": result.generated_tokens,
            }
        )

    del runner
    release_accelerator_memory()
    return records


def summarize_model(records: list[dict]) -> dict:
    return {**summarize(records), **summarize_quality(records)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    samples = load_instruction_samples(**config["data"])
    model_name = config["model"]["name"]
    adapter_path = config["model"]["adapter_path"]
    max_new_tokens = int(config["generation"]["max_new_tokens"])

    base_records = run_model("base", model_name, None, samples, max_new_tokens)
    lora_records = run_model("lora", model_name, adapter_path, samples, max_new_tokens)

    output_path = Path(config["output"]["predictions"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in base_records + lora_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    base_metrics = summarize_model(base_records)
    lora_metrics = summarize_model(lora_records)
    payload = {
        "model": model_name,
        "adapter_path": adapter_path,
        "held_out_offset": config["data"]["offset"],
        "base": base_metrics,
        "lora": lora_metrics,
        "delta": {
            "token_f1": lora_metrics["mean_token_f1"] - base_metrics["mean_token_f1"],
            "rouge_l_f1": lora_metrics["mean_rouge_l_f1"] - base_metrics["mean_rouge_l_f1"],
            "mean_latency_seconds": (
                lora_metrics["mean_latency_seconds"] - base_metrics["mean_latency_seconds"]
            ),
            "mean_tokens_per_second": (
                lora_metrics["mean_tokens_per_second"] - base_metrics["mean_tokens_per_second"]
            ),
        },
    }
    write_json(config["output"]["metrics"], payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
