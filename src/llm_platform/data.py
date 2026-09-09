from __future__ import annotations

from datasets import Dataset, load_dataset


def load_instruction_samples(
    dataset_name: str,
    split: str = "train",
    sample_size: int = 25,
    seed: int = 42,
    offset: int = 0,
) -> Dataset:
    """Load a deterministic slice and normalize it to prompt/reference fields."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if offset < 0:
        raise ValueError("offset must be non-negative")

    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.shuffle(seed=seed)
    start = min(offset, len(dataset))
    stop = min(start + sample_size, len(dataset))
    dataset = dataset.select(range(start, stop))

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
