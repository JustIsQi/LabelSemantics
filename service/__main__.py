"""Multi-worker launcher: ``python -m service``.

This wraps ``uvicorn.run`` with sensible defaults and exposes the most common
knobs as CLI flags. Configuration of model/label paths is still picked up from
``LS_*`` environment variables (see ``service.app``) so they propagate cleanly
to every worker process.
"""

from __future__ import annotations

import argparse
import os

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LabelSemantics NER FastAPI service with multiple workers."
    )
    parser.add_argument("--host", default=os.environ.get("LS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LS_PORT", "8000")))
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("LS_WORKERS", "1")),
        help="number of uvicorn worker processes; each loads its own model copy.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LS_LOG_LEVEL", "info").lower(),
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="enable auto-reload (dev only; forces --workers 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workers = 1 if args.reload else max(1, args.workers)
    uvicorn.run(
        "service.app:app",
        host=args.host,
        port=args.port,
        workers=workers,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
