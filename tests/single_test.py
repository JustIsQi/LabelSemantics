"""Simplest possible single-query test against the FastAPI service.

Sends one ``POST /predict`` request and prints the inference result together
with a millisecond-level timing breakdown so you can see where every chunk of
the end-to-end latency went:

- ``tokenize_ms`` / ``model_ms`` / ``decode_ms``: per-stage timings reported
  by the server inside the ``timings`` field of the response body.
- ``inference_ms``: sum of the three stages above (== legacy ``inference_ms``).
- ``server_handler_ms``: handler-side wall time (inference + normalize +
  optional time-regex merge).
- ``server_request_total_ms``: full ASGI handling time (request parse +
  validation + handler + response serialize), reported by the server via the
  ``X-Request-Total-Ms`` response header.
- ``client_total_ms``: client-side wall time (everything above + network
  round-trip + JSON encode/decode on the client).

The gaps tell you where to optimize:

    client_total_ms - server_request_total_ms  -> network + client-side cost
    server_request_total_ms - server_handler_ms -> ASGI / pydantic / JSON serialize
    server_handler_ms - inference_ms           -> normalize + post-processing
    inference_ms - model_ms                    -> tokenize + tag decode

Examples
--------

    python tests/fastapi_single_test.py "银河证券、国泰海通、招商证券去年增资哪些行业"
    python tests/fastapi_single_test.py --base-url http://127.0.0.1:8000 "中信证券Q1研报"
    python tests/fastapi_single_test.py --repeat 5 "比亚迪最近股价表现"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests._fastapi_client import ServiceClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-query test for the FastAPI NER service.")
    parser.add_argument("query", default="2026年3月2日中泰证券研报中，通过迪士尼业务收入表格显示FY25体验业务收入超350亿美元的是哪篇？")
    parser.add_argument("--base-url", default="http://10.100.0.2:8601")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="连续发送 N 次同一 query；当 N>1 时额外打印 server/client 平均、p50、最大延迟。",
    )
    parser.add_argument(
        "--include-time-regex",
        action="store_true",
        help="请求中带上 include_time_regex=true。",
    )
    return parser.parse_args()


def _format_timing_line(
    response: dict, server_request_total_ms: float, client_total_ms: float, prefix: str = ""
) -> str:
    timings = response.get("timings") or {}
    parts = [
        f"tokenize_ms={timings.get('tokenize_ms', 0):.3f}",
        f"model_ms={timings.get('model_ms', 0):.3f}",
        f"decode_ms={timings.get('decode_ms', 0):.3f}",
        f"inference_ms={timings.get('inference_ms', 0):.3f}",
        f"server_handler_ms={timings.get('server_handler_ms', 0):.3f}",
        f"server_request_total_ms={server_request_total_ms:.3f}",
        f"client_total_ms={client_total_ms:.3f}",
    ]
    return prefix + "  ".join(parts)


def main() -> int:
    args = parse_args()
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 2

    client = ServiceClient(args.base_url, timeout=args.timeout)
    include_time_regex = True if args.include_time_regex else None

    inference_samples = []
    handler_samples = []
    server_total_samples = []
    client_samples = []
    last_response = None

    for index in range(1, args.repeat + 1):
        start = time.perf_counter()
        try:
            response = client.predict(args.query, include_time_regex=include_time_regex)
        except RuntimeError as exc:
            print(f"request failed: {exc}", file=sys.stderr)
            return 1
        client_ms = (time.perf_counter() - start) * 1000
        server_request_total_ms = client.server_request_total_ms() or 0.0
        timings = response.get("timings") or {}
        inference_samples.append(float(timings.get("inference_ms", 0.0)))
        handler_samples.append(float(timings.get("server_handler_ms", 0.0)))
        server_total_samples.append(server_request_total_ms)
        client_samples.append(client_ms)
        last_response = response

        if args.repeat == 1:
            print(json.dumps(response, ensure_ascii=False, indent=2))
            print(_format_timing_line(response, server_request_total_ms, client_ms))
        else:
            print(
                _format_timing_line(
                    response,
                    server_request_total_ms,
                    client_ms,
                    prefix=f"[{index}/{args.repeat}] ",
                )
            )

    if args.repeat > 1:
        assert last_response is not None
        print()
        print(json.dumps(last_response, ensure_ascii=False, indent=2))
        print(
            "summary  "
            f"avg_inference_ms={statistics.fmean(inference_samples):.3f}  "
            f"avg_handler_ms={statistics.fmean(handler_samples):.3f}  "
            f"avg_server_total_ms={statistics.fmean(server_total_samples):.3f}  "
            f"avg_client_ms={statistics.fmean(client_samples):.3f}  "
            f"p50_client_ms={statistics.median(client_samples):.3f}  "
            f"max_client_ms={max(client_samples):.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
