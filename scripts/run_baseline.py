from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from llm_platform.benchmark import summarize, write_json
from llm_platform.data import load_instruction_samples
from llm_platform.inference import BaseModelRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = config["model"]
    data_cfg = config["data"]
    output_cfg = config["output"]

    samples = load_instruction_samples(**data_cfg)
    runner = BaseModelRunner(model_cfg["name"])

    prediction_path = Path(output_cfg["predictions"])
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    records = []

    with prediction_path.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            result = runner.generate(
                sample["prompt"], max_new_tokens=model_cfg["max_new_tokens"]
            )
            record = {
                "id": index,
                "prompt": sample["prompt"],
                "reference": sample["reference"],
                "prediction": result.response,
                "latency_seconds": result.latency_seconds,
                "generated_tokens": result.generated_tokens,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = summarize(records)
    metrics["model"] = model_cfg["name"]
    metrics["device"] = runner.device
    write_json(output_cfg["metrics"], metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
