import pytest

from llm_platform.evaluation import rouge_l_f1, summarize_quality, token_f1


def test_token_f1_exact_match():
    assert token_f1("a b c", "a b c") == 1.0


def test_token_f1_partial_overlap():
    assert token_f1("a b", "a c") == pytest.approx(0.5)


def test_rouge_l_f1_exact_match():
    assert rouge_l_f1("the quick fox", "the quick fox") == 1.0


def test_rouge_l_f1_partial_sequence():
    assert rouge_l_f1("a b c", "a x c") == pytest.approx(2 / 3)


def test_quality_summary():
    metrics = summarize_quality(
        [
            {"prediction": "a b", "reference": "a b"},
            {"prediction": "a c", "reference": "a b"},
        ]
    )
    assert metrics["samples"] == 2
    assert metrics["mean_token_f1"] == pytest.approx(0.75)
    assert metrics["mean_rouge_l_f1"] == pytest.approx(0.75)


def test_empty_quality_evaluation_rejected():
    with pytest.raises(ValueError):
        summarize_quality([])
