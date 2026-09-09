from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from llm_platform.data import load_instruction_samples
from llm_platform.training import build_trainer, parameter_stats, to_prompt_completion_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_lora.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    samples = load_instruction_samples(**config["data"])
    train_dataset = to_prompt_completion_dataset(samples)

    trainer = build_trainer(
        model_name=config["model"]["name"],
        dataset=train_dataset,
        lora_cfg=config["lora"],
        training_cfg=config["training"],
    )

    stats = parameter_stats(trainer.model)
    print(
        json.dumps(
            {
                "trainable_parameters": stats.trainable,
                "total_parameters": stats.total,
                "trainable_percent": round(stats.trainable_percent, 4),
                "parameter_reduction_percent": round(stats.reduction_percent, 4),
            },
            indent=2,
        )
    )

    train_result = trainer.train()
    trainer.save_model(config["training"]["output_dir"])
    trainer.save_state()

    metrics_path = Path(config["training"]["output_dir"]) / "training_metrics.json"
    metrics_path.write_text(json.dumps(train_result.metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved adapter and training metrics to {config['training']['output_dir']}")


if __name__ == "__main__":
    main()
