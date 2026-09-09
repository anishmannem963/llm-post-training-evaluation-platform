from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


@dataclass(frozen=True)
class ParameterStats:
    trainable: int
    total: int

    @property
    def trainable_percent(self) -> float:
        return 0.0 if self.total == 0 else 100.0 * self.trainable / self.total

    @property
    def reduction_percent(self) -> float:
        return 100.0 - self.trainable_percent


def to_prompt_completion_dataset(samples: Dataset) -> Dataset:
    """Convert normalized baseline samples into TRL prompt-completion format."""
    required = {"prompt", "reference"}
    missing = required.difference(samples.column_names)
    if missing:
        raise ValueError(f"Samples are missing required columns: {sorted(missing)}")

    def convert(row: dict) -> dict:
        return {"prompt": row["prompt"], "completion": row["reference"]}

    return samples.map(convert, remove_columns=samples.column_names)


def parameter_stats(model) -> ParameterStats:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return ParameterStats(trainable=trainable, total=total)


def build_trainer(model_name: str, dataset: Dataset, lora_cfg: dict, training_cfg: dict):
    """Construct an SFTTrainer configured for parameter-efficient LoRA training."""
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg["target_modules"]),
    )

    args = SFTConfig(
        output_dir=training_cfg["output_dir"],
        learning_rate=float(training_cfg["learning_rate"]),
        num_train_epochs=float(training_cfg["num_train_epochs"]),
        max_steps=int(training_cfg["max_steps"]),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        max_length=int(training_cfg["max_length"]),
        logging_steps=int(training_cfg["logging_steps"]),
        save_steps=int(training_cfg["save_steps"]),
        seed=int(training_cfg["seed"]),
        completion_only_loss=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        save_total_limit=2,
    )

    return SFTTrainer(
        model=model_name,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
