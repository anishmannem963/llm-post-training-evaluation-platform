from __future__ import annotations

from datasets import Dataset, load_dataset


def load_instruction_samples(
    dataset_name: str,
    split: str = "train",
    sample_size: int = 25,
    seed: int = 42,
) -> Dataset:
    """Load a deterministic subset and normalize it to prompt/reference fields."""
    dataset = load_dataset(dataset_name, split=split)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    count = min(sample_size, len(dataset))
    dataset = dataset.shuffle(seed=seed).select(range(count))

    required = {"instruction", "input", "output"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    def normalize(row: dict) -> dict:
        instruction = row["instruction"].strip()
        context = row["input"].strip()
        prompt = instruction if not context else f"{instruction}\n\nContext:\n{context}"
        return {"prompt": prompt, "reference": row["output"].strip()}

    return dataset.map(normalize, remove_columns=dataset.column_names)
