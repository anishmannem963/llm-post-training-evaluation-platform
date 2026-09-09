import pytest

from llm_platform.data import load_instruction_samples


def test_non_positive_sample_size_rejected():
    with pytest.raises(ValueError):
        load_instruction_samples("yahma/alpaca-cleaned", sample_size=0)
