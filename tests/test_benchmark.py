import pytest

from llm_platform.benchmark import summarize


def test_summarize_records():
    metrics = summarize(
        [
            {"latency_seconds": 1.0, "generated_tokens": 10},
            {"latency_seconds": 2.0, "generated_tokens": 10},
        ]
    )
    assert metrics["samples"] == 2
    assert metrics["mean_latency_seconds"] == pytest.approx(1.5)
    assert metrics["mean_tokens_per_second"] == pytest.approx(7.5)
    assert metrics["total_generated_tokens"] == 20


def test_empty_benchmark_rejected():
    with pytest.raises(ValueError):
        summarize([])
