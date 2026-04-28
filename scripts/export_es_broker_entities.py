#!/usr/bin/env python3
"""Export unique broker organization names from Elasticsearch to JSON.

The FastAPI service can load this JSON file to copy company-side predictions
that are known broker names into the BROKER field.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _json_request(
    url: str,
    *,
    user: str,
    password: str,
    payload: Optional[dict] = None,
    method: Optional[str] = None,
    timeout: float,
) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if user or password:
        token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {token}"

    req = urllib_request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ES HTTP {exc.code} for {url}: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"ES request failed for {url}: {exc.reason}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ES returned non-JSON response for {url}") from exc


def _hit_values(hit: dict, field: str) -> Iterable[str]:
    source = hit.get("_source") or {}
    value = source.get(field)
    if value is None:
        fields = hit.get("fields") or {}
        value = fields.get(field)
    if isinstance(value, list):
        for item in value:
            yield str(item)
    elif value is not None:
        yield str(value)


def export_entities(args: argparse.Namespace) -> List[str]:
    base_url = f"http://{args.host}:{args.port}"
    index = urllib_parse.quote(args.index, safe="")
    search_url = f"{base_url}/{index}/_search?scroll={urllib_parse.quote(args.scroll)}"

    payload = {
        "size": args.batch_size,
        "_source": [args.field],
        "query": {"exists": {"field": args.field}},
    }
    response = _json_request(
        search_url,
        user=args.user,
        password=args.password,
        payload=payload,
        timeout=args.timeout,
    )

    names = set()
    scroll_id = response.get("_scroll_id")
    try:
        while True:
            hits = (response.get("hits") or {}).get("hits") or []
            if not hits:
                break
            for hit in hits:
                for value in _hit_values(hit, args.field):
                    name = value.strip()
                    if name:
                        names.add(name)

            if not scroll_id:
                break
            response = _json_request(
                f"{base_url}/_search/scroll",
                user=args.user,
                password=args.password,
                payload={"scroll": args.scroll, "scroll_id": scroll_id},
                timeout=args.timeout,
            )
            scroll_id = response.get("_scroll_id", scroll_id)
    finally:
        if scroll_id:
            try:
                _json_request(
                    f"{base_url}/_search/scroll",
                    user=args.user,
                    password=args.password,
                    payload={"scroll_id": [scroll_id]},
                    method="DELETE",
                    timeout=args.timeout,
                )
            except RuntimeError:
                pass

    return sorted(names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export deduplicated broker SRC_ORG_NAME values from Elasticsearch."
    )
    parser.add_argument("--host", default=os.getenv("ELASTICSEARCH_SEARCH_URL", "10.100.0.100"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("ELASTICSEARCH_SEARCH_PORT", "9200")),
    )
    parser.add_argument("--user", default=os.getenv("ELASTICSEARCH_SEARCH_USER", "elastic"))
    parser.add_argument(
        "--password",
        default=os.getenv("ELASTICSEARCH_SEARCH_PASSWORD", "6YCuCbNUf2ap"),
    )
    parser.add_argument("--index", default="securities_firm_with_pinyin")
    parser.add_argument("--field", default="SRC_ORG_NAME")
    parser.add_argument(
        "--output",
        default="data/securities_firm_src_org_names.json",
        help="Output JSON path, relative to project root unless absolute.",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--scroll", default="2m")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        names = export_entities(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(names, output, ensure_ascii=False, indent=2)
        output.write("\n")

    print(f"wrote {len(names)} unique {args.field} values to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
