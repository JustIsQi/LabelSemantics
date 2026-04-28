"""Concurrent stress test for the FastAPI service.

Spawns ``--concurrency`` client threads that hammer ``POST /predict`` for a
fixed total of ``--requests`` calls (or one full pass over ``--input-file``).
Reports throughput (QPS) and latency distribution (avg / p50 / p95 / p99 / max)
for both the server-reported inference time and the client-side wall time.
This is the easiest way to verify that a multi-worker server (``--workers N``)
actually gives parallel speedup.

Examples
--------

    # 8 concurrent clients, 1000 requests, cycling through canned queries
    python tests/fastapi_stress_test.py --concurrency 8 --requests 1000

    # Replay 500 rows of eval.jsonl with 16 concurrent clients
    python tests/fastapi_stress_test.py \\
        --concurrency 16 --input-file data/eval.jsonl --limit 500
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests._fastapi_client import ServiceClient  # noqa: E402
from tests.batch_test import load_jsonl  # noqa: E402


CANNED_QUERIES = [
    "茅台2023年的营收情况",
    "中信证券2024年Q1对宁德时代的研报",
    "比亚迪最近的股价表现",
    "宁德时代H1净利润同比增长多少",
    "中金公司近期对新能源板块的观点",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concurrent stress test against the FastAPI NER service."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of concurrent client threads.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=200,
        help="Total number of requests to send. With --input-file, queries cycle "
        "through the file; pass --requests $(wc -l < file) for exactly one pass.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Optional JSONL file; queries are taken from each row's ori_query.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Take only the first N rows of --input-file.",
    )
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


def _build_query_pool(args: argparse.Namespace) -> List[str]:
    if args.input_file:
        path = Path(args.input_file)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        rows = load_jsonl(str(path), args.limit)
        queries = [row.get("ori_query", "") for row in rows if row.get("ori_query")]
        if not queries:
            raise SystemExit(f"no queries found in {path}")
        return queries
    return list(CANNED_QUERIES)


def _percentile(samples: List[float], pct: float) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0]
    ordered = sorted(samples)
    if pct <= 0:
        return ordered[0]
    if pct >= 100:
        return ordered[-1]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summarise(name: str, samples: List[float]) -> str:
    if not samples:
        return f"{name}: <no samples>"
    return (
        f"{name}: n={len(samples)} "
        f"avg={statistics.fmean(samples):.3f} "
        f"p50={_percentile(samples, 50):.3f} "
        f"p95={_percentile(samples, 95):.3f} "
        f"p99={_percentile(samples, 99):.3f} "
        f"max={max(samples):.3f}"
    )


def main() -> int:
    args = parse_args()
    if args.concurrency < 1:
        print("--concurrency must be >= 1", file=sys.stderr)
        return 2
    if args.requests < 1:
        print("--requests must be >= 1", file=sys.stderr)
        return 2

    client = ServiceClient(args.base_url, timeout=args.timeout)
    include_time_regex = True if args.include_time_regex else None

    query_pool = _build_query_pool(args)
    total_requests = args.requests
    queries = [query_pool[i % len(query_pool)] for i in range(total_requests)]

    print(
        f"target: base_url={args.base_url} concurrency={args.concurrency} "
        f"requests={total_requests} pool_size={len(query_pool)}",
        flush=True,
    )

    for query in queries[: args.warmup]:
        try:
            client.predict(query, include_time_regex=include_time_regex)
        except RuntimeError as exc:
            print(f"warmup failed: {exc}", file=sys.stderr)
            return 1

    def _one(query: str) -> Tuple[float, float, bool, str]:
        client_start = time.perf_counter()
        try:
            response = client.predict(query, include_time_regex=include_time_regex)
        except RuntimeError as exc:
            client_ms = (time.perf_counter() - client_start) * 1000
            return 0.0, client_ms, False, str(exc)
        client_ms = (time.perf_counter() - client_start) * 1000
        server_ms = float(response.get("inference_ms", 0.0))
        return server_ms, client_ms, True, ""

    server_ms_samples: List[float] = []
    client_ms_samples: List[float] = []
    error_count = 0
    first_error: str = ""

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(_one, query) for query in queries]
        completed = 0
        for future in as_completed(futures):
            server_ms, client_ms, ok, err = future.result()
            completed += 1
            if ok:
                server_ms_samples.append(server_ms)
                client_ms_samples.append(client_ms)
            else:
                error_count += 1
                if not first_error:
                    first_error = err
            if completed % 100 == 0:
                avg_client = statistics.fmean(client_ms_samples) if client_ms_samples else 0.0
                print(
                    f"completed={completed}/{total_requests} errors={error_count} "
                    f"avg_client_ms={avg_client:.3f}",
                    flush=True,
                )
    wall_elapsed = time.perf_counter() - wall_start
    qps = (total_requests - error_count) / wall_elapsed if wall_elapsed > 0 else 0.0

    print()
    print(
        f"requests={total_requests} ok={total_requests - error_count} errors={error_count} "
        f"concurrency={args.concurrency} wall_s={wall_elapsed:.3f} qps={qps:.2f}",
        flush=True,
    )
    print(_summarise("server_inference_ms", server_ms_samples), flush=True)
    print(_summarise("client_total_ms   ", client_ms_samples), flush=True)
    if error_count:
        print(f"first error: {first_error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
