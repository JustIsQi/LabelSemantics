"""Batch test against the running FastAPI service.

Replays a JSONL file (defaults to ``data/eval.jsonl``) through ``POST /predict``
sequentially and computes the same per-label recall / extras / accuracy that
``tests/batch_test.py`` prints for the in-process model. Reuses
``batch_test``'s gold extraction and scoring helpers (those only need stdlib at
runtime), so the HTTP path is held to the same accuracy as the in-process
batch test.

Examples
--------

    python tests/fastapi_batch_test.py
    python tests/fastapi_batch_test.py \\
        --base-url http://127.0.0.1:8000 \\
        --input-file data/eval.jsonl \\
        --output-file tests/fastapi_batch_test_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests._fastapi_client import ServiceClient  # noqa: E402
from tests.batch_test import (  # noqa: E402
    evaluate_match,
    gold_entities_from_row,
    load_jsonl,
    normalize_entities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch test against the FastAPI NER service.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--input-file", default="data/eval.jsonl")
    parser.add_argument(
        "--output-file",
        default="tests/fastapi_batch_test_predictions.jsonl",
        help="Per-row predictions in batch_test schema (set to '' to skip writing).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Sequential warmup calls before the timed loop.",
    )
    parser.add_argument(
        "--include-time-regex",
        action="store_true",
        help="请求中带上 include_time_regex=true。",
    )
    return parser.parse_args()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> int:
    args = parse_args()
    client = ServiceClient(args.base_url, timeout=args.timeout)
    include_time_regex = True if args.include_time_regex else None

    label_descriptions = client.labels()
    if not isinstance(label_descriptions, dict) or not label_descriptions:
        print(f"GET /labels returned unexpected payload: {label_descriptions!r}", file=sys.stderr)
        return 1
    print(f"Labels from service: {list(label_descriptions.keys())}", flush=True)

    input_path = _resolve_path(args.input_file)
    rows = load_jsonl(str(input_path), args.limit)
    print(f"Loaded {len(rows)} rows from {input_path}", flush=True)

    for row in rows[: args.warmup]:
        client.predict(row.get("ori_query", ""), include_time_regex=include_time_regex)

    output_path: Optional[Path] = _resolve_path(args.output_file) if args.output_file else None
    out_handle = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_handle = output_path.open("w", encoding="utf-8", newline="\n")

    correct = 0
    total = 0
    server_ms_samples: List[float] = []
    client_ms_samples: List[float] = []
    label_hits: Dict[str, int] = defaultdict(int)
    label_total: Dict[str, int] = defaultdict(int)
    label_extras_rows: Dict[str, int] = defaultdict(int)
    label_extras_count: Dict[str, int] = defaultdict(int)
    label_missing_rows: Dict[str, int] = defaultdict(int)
    label_missing_count: Dict[str, int] = defaultdict(int)
    rows_with_any_extras = 0
    total_extras = 0

    wall_start = time.perf_counter()
    try:
        for index, row in enumerate(rows, start=1):
            query = row.get("ori_query", "")
            gold_entities = gold_entities_from_row(row, label_descriptions)

            client_start = time.perf_counter()
            try:
                response: Dict[str, Any] = client.predict(
                    query, include_time_regex=include_time_regex
                )
            except RuntimeError as exc:
                print(f"row {index} request failed: {exc}", file=sys.stderr)
                return 1
            client_ms = (time.perf_counter() - client_start) * 1000
            server_ms = float(response.get("inference_ms", 0.0))

            pred_entities = normalize_entities(response.get("entities", {}), label_descriptions)
            is_correct, per_label = evaluate_match(pred_entities, gold_entities)
            correct += int(is_correct)
            total += 1
            server_ms_samples.append(server_ms)
            client_ms_samples.append(client_ms)

            row_extras_total = 0
            extras_dump: Dict[str, List[str]] = {}
            missing_dump: Dict[str, List[str]] = {}
            for label, info in per_label.items():
                label_total[label] += 1
                if info["hit"]:
                    label_hits[label] += 1
                if info["extras"]:
                    label_extras_rows[label] += 1
                    label_extras_count[label] += len(info["extras"])
                    row_extras_total += len(info["extras"])
                    extras_dump[label] = info["extras"]
                if info["missing"]:
                    label_missing_rows[label] += 1
                    label_missing_count[label] += len(info["missing"])
                    missing_dump[label] = info["missing"]
            if row_extras_total:
                rows_with_any_extras += 1
                total_extras += row_extras_total

            if out_handle is not None:
                gold_compact = {k: v for k, v in gold_entities.items() if v}
                pred_compact = {k: v for k, v in pred_entities.items() if v}
                out_handle.write(
                    json.dumps(
                        {
                            "ori_query": query,
                            "correct": is_correct,
                            "gold": gold_compact,
                            "pred": pred_compact,
                            "extras": extras_dump,
                            "missing": missing_dump,
                            "server_ms": round(server_ms, 3),
                            "client_ms": round(client_ms, 3),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if index % 100 == 0:
                accuracy = correct / total if total else 0.0
                avg_server = statistics.fmean(server_ms_samples)
                avg_client = statistics.fmean(client_ms_samples)
                print(
                    f"processed={index} accuracy={accuracy:.6f} "
                    f"avg_server_ms={avg_server:.3f} avg_client_ms={avg_client:.3f}",
                    flush=True,
                )
    finally:
        if out_handle is not None:
            out_handle.close()

    wall_elapsed = time.perf_counter() - wall_start
    accuracy = correct / total if total else 0.0
    avg_server = statistics.fmean(server_ms_samples) if server_ms_samples else 0.0
    avg_client = statistics.fmean(client_ms_samples) if client_ms_samples else 0.0
    max_client = max(client_ms_samples) if client_ms_samples else 0.0

    if output_path is not None:
        print(f"Output: {output_path}", flush=True)
    print(
        f"total={total} correct={correct} accuracy={accuracy:.6f} "
        f"avg_server_ms={avg_server:.3f} avg_client_ms={avg_client:.3f} "
        f"max_client_ms={max_client:.3f} wall_s={wall_elapsed:.3f}",
        flush=True,
    )
    print("Per-label recall (gold ⊆ pred) and extras:", flush=True)
    for label in sorted(label_total):
        ttl = label_total[label]
        hit = label_hits[label]
        recall = hit / ttl if ttl else 0.0
        ex_rows = label_extras_rows[label]
        ex_cnt = label_extras_count[label]
        ms_rows = label_missing_rows[label]
        ms_cnt = label_missing_count[label]
        print(
            f"  {label:>8s}: recall={recall:.4f} ({hit}/{ttl})  "
            f"extras_rows={ex_rows} extras_total={ex_cnt}  "
            f"missing_rows={ms_rows} missing_total={ms_cnt}",
            flush=True,
        )
    print(
        f"Rows with any extras: {rows_with_any_extras}  total extras mentions: {total_extras}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
