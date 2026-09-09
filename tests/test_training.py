import torch
from datasets import Dataset

from llm_platform.training import parameter_stats, to_prompt_completion_dataset


def test_prompt_completion_conversion():
    samples = Dataset.from_dict(
        {
            "prompt": ["Explain gradient descent."],
            "reference": ["It iteratively updates parameters to reduce a loss function."],
        }
    )

    converted = to_prompt_completion_dataset(samples)

    assert converted.column_names == ["prompt", "completion"]
    assert converted[0]["prompt"] == [
        {"role": "user", "content": "Explain gradient descent."}
    ]
    assert converted[0]["completion"] == [
        {
            "role": "assistant",
            "content": "It iteratively updates parameters to reduce a loss function.",
        }
    ]


def test_parameter_stats_counts_trainable_parameters():
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    for parameter in model[0].parameters():
        parameter.requires_grad = False

    stats = parameter_stats(model)

    assert stats.total == 23
    assert stats.trainable == 8
    assert stats.trainable_percent == 100.0 * 8 / 23
    assert stats.reduction_percent == 100.0 - (100.0 * 8 / 23)
