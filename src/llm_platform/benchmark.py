from __future__ import annotations

import json
import statistics
from pathlib import Path


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Cannot summarize an empty benchmark")

    latencies = [float(row["latency_seconds"]) for row in records]
    token_rates = [
        row["generated_tokens"] / row["latency_seconds"]
        for row in records
        if row["latency_seconds"] > 0
    ]
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))

    return {
        "samples": len(records),
        "mean_latency_seconds": statistics.fmean(latencies),
        "p95_latency_seconds": ordered[p95_index],
        "mean_tokens_per_second": statistics.fmean(token_rates),
        "total_generated_tokens": sum(row["generated_tokens"] for row in records),
    }


def write_json(path: str, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
