from __future__ import annotations

import statistics


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def token_f1(prediction: str, reference: str) -> float:
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred or not ref:
        return 1.0 if pred == ref else 0.0

    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = sum(min(count, ref_counts.get(token, 0)) for token, count in pred_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for token_left in left:
        current = [0]
        for index, token_right in enumerate(right, start=1):
            if token_left == token_right:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred or not ref:
        return 1.0 if pred == ref else 0.0
    lcs = _lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def summarize_quality(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Cannot summarize an empty evaluation")
    token_scores = [token_f1(row["prediction"], row["reference"]) for row in records]
    rouge_scores = [rouge_l_f1(row["prediction"], row["reference"]) for row in records]
    return {
        "samples": len(records),
        "mean_token_f1": statistics.fmean(token_scores),
        "mean_rouge_l_f1": statistics.fmean(rouge_scores),
    }
