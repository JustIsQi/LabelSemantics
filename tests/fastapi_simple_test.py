from __future__ import annotations

import argparse
import json
import sys
import time
from urllib import error as urllib_error
from urllib import request as urllib_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal single-query test for the FastAPI NER service.")
    parser.add_argument("query", help="原始 query 文本")
    parser.add_argument("--base-url", default="http://10.100.0.2:8601")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    url = f"{args.base_url.rstrip('/')}/predict"
    payload = json.dumps({"query": args.query}, ensure_ascii=False).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib_request.urlopen(req, timeout=args.timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        return 1
    except urllib_error.URLError as exc:
        print(f"request failed: {exc.reason}", file=sys.stderr)
        return 1
    client_ms = (time.perf_counter() - start) * 1000

    response = json.loads(body)
    print(json.dumps(response, ensure_ascii=False, indent=2))
    print(f"client_total_ms={client_ms:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
