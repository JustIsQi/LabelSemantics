"""FastAPI app exposing the LabelSemantics NER model.

Inference logic mirrors ``tests/batch_test.py`` (encode_query / predict_entities /
trim_entity_span / extract_time_mentions / normalize_entities) so that the HTTP
service produces predictions identical to the batch evaluation script.

Configuration is read from environment variables so the same module can be
launched directly with ``uvicorn service.app:app --workers N``:

- ``LS_MODEL_PATH``   self-contained model directory (default ``outputs/best_model``)
- ``LS_LABEL_FILE``   labels.json path (default ``data/excel_ner_data/labels.json``)
- ``LS_MAX_LENGTH``   tokenizer max length (default ``128``)
- ``LS_DEVICE``       torch device, e.g. ``cuda``, ``cuda:0``, ``cpu`` (default: auto)
- ``LS_WARMUP_QUERIES``  optional ``;`` separated warmup queries
- ``LS_INCLUDE_TIME_REGEX``  ``1`` to also merge regex-extracted TIME mentions
  into the response (mirrors ``gold_entities_from_row`` in batch_test); default ``0``.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from label_semantics.labels import build_tag_maps, load_label_descriptions
from label_semantics.metrics import get_entities
from label_semantics.model import LabelSemanticsNER

logger = logging.getLogger("label_semantics.service")

# Prefer orjson for response serialization (2-5x faster than stdlib json on
# small dicts, which dominates per-request overhead in this service). Fall back
# silently to JSONResponse if orjson isn't installed.
try:
    import orjson  # noqa: F401
    from fastapi.responses import ORJSONResponse as _DefaultResponseClass
except ImportError:  # pragma: no cover
    from fastapi.responses import JSONResponse as _DefaultResponseClass


# --- inference helpers (kept in sync with tests/batch_test.py) ---------------

ENTITY_BOUNDARY_STRIP_CHARS = set(
    " \t\u3000、，,。.；;：:!！?？/\\()()[]【】《》<>\"'`~“”‘’\r\n"
)

TIME_PATTERNS = (
    r"截至\d{4}年\d{1,2}月\d{1,2}日",
    r"\d{4}年\d{1,2}月\d{1,2}日",
    r"\d{4}\s*[-—至到]\s*\d{4}年",
    r"\d{4}年(?:第[一二三四1234]季度|[一二三四1234]季度|Q[1-4]|H[12]|上半年|下半年|中报|年报|全年)",
    r"\d{4}Q[1-4]",
    r"\d{4}年",
    r"\d{1,2}月\d{1,2}日",
    r"(?:最近|近|过去|未来|前)\d*[一二三四五六七八九十0-9]*(?:个)?(?:交易日|工作日|日|天|周|月|季度|季|半年|年)",
    r"(?:近|过去)[一二三四五六七八九十0-9]+年",
    r"年初至今",
    r"(?:最近|最新|近期|当前|目前|今年|去年|本周|本月|本季度|本年|上周|上月|上季度|上年|上半年|下半年|上市以来)",
    r"[一二三四]季度",
    r"Q[1-4]",
)
TIME_RE = re.compile("|".join(f"(?:{pattern})" for pattern in TIME_PATTERNS))


def trim_entity_span(query: str, start: int, end: int):
    while start <= end and query[start] in ENTITY_BOUNDARY_STRIP_CHARS:
        start += 1
    while end >= start and query[end] in ENTITY_BOUNDARY_STRIP_CHARS:
        end -= 1
    return start, end


def encode_query(query: str, tokenizer, max_length: int, device: torch.device):
    chars = list(query)
    inputs = tokenizer(
        chars,
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    word_ids = inputs.word_ids(batch_index=0)
    model_inputs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "token_type_ids": inputs.get(
            "token_type_ids", torch.zeros_like(inputs["input_ids"])
        ).to(device),
    }
    return model_inputs, word_ids


def predict_entities(
    query: str,
    model: LabelSemanticsNER,
    tokenizer,
    id2tag: Dict[int, str],
    label_descriptions: Dict[str, str],
    max_length: int,
    device: torch.device,
) -> Dict[str, List[str]]:
    """Backward-compatible wrapper used by the warmup path. The hot serving
    path uses :func:`predict_entities_timed` so it can return a per-stage
    timing breakdown without paying for an extra ``time.perf_counter`` call
    when timing isn't needed."""
    if not query:
        return {label: [] for label in label_descriptions}

    model_inputs, word_ids = encode_query(query, tokenizer, max_length, device)
    with torch.no_grad():
        _, pred_ids = model(**model_inputs, use_label_cache=True)

    tags = ["O"] * len(query)
    previous_word_id = None
    for pred_id, word_id in zip(pred_ids[0].detach().cpu().tolist(), word_ids):
        if word_id is None or word_id >= len(tags):
            previous_word_id = word_id
            continue
        # Only the first sub-token of each word carries the supervised label
        # (matches training-time alignment).
        if word_id == previous_word_id:
            continue
        tags[word_id] = id2tag[pred_id]
        previous_word_id = word_id

    entities: Dict[str, List[str]] = {label: [] for label in label_descriptions}
    for entity_type, start, end in get_entities(tags):
        if entity_type not in entities:
            continue
        start, end = trim_entity_span(query, start, end)
        if start > end:
            continue
        entities[entity_type].append(query[start:end + 1])
    return entities


def predict_entities_timed(
    query: str,
    model: LabelSemanticsNER,
    tokenizer,
    id2tag: Dict[int, str],
    label_descriptions: Dict[str, str],
    max_length: int,
    device: torch.device,
):
    """Same predictions as :func:`predict_entities`, but returns
    ``(entities, timings)`` where ``timings`` is a dict with millisecond
    measurements for ``tokenize``, ``model`` (forward + implicit CUDA sync via
    ``.cpu()``) and ``decode``.

    The implementation is duplicated rather than wrapped so that the hot path
    avoids the extra ``time.perf_counter`` overhead when timing isn't needed.
    """
    if not query:
        return (
            {label: [] for label in label_descriptions},
            {"tokenize_ms": 0.0, "model_ms": 0.0, "decode_ms": 0.0, "total_ms": 0.0},
        )

    t0 = time.perf_counter()
    model_inputs, word_ids = encode_query(query, tokenizer, max_length, device)
    t1 = time.perf_counter()

    with torch.no_grad():
        _, pred_ids = model(**model_inputs, use_label_cache=True)
    # ``.cpu().tolist()`` implicitly synchronizes with the CUDA stream, so this
    # captures real GPU work without an extra ``torch.cuda.synchronize`` call.
    pred_list = pred_ids[0].detach().cpu().tolist()
    t2 = time.perf_counter()

    tags = ["O"] * len(query)
    previous_word_id = None
    for pred_id, word_id in zip(pred_list, word_ids):
        if word_id is None or word_id >= len(tags):
            previous_word_id = word_id
            continue
        if word_id == previous_word_id:
            continue
        tags[word_id] = id2tag[pred_id]
        previous_word_id = word_id

    entities: Dict[str, List[str]] = {label: [] for label in label_descriptions}
    for entity_type, start, end in get_entities(tags):
        if entity_type not in entities:
            continue
        start, end = trim_entity_span(query, start, end)
        if start > end:
            continue
        entities[entity_type].append(query[start:end + 1])
    t3 = time.perf_counter()

    timings = {
        "tokenize_ms": (t1 - t0) * 1000,
        "model_ms": (t2 - t1) * 1000,
        "decode_ms": (t3 - t2) * 1000,
        "total_ms": (t3 - t0) * 1000,
    }
    return entities, timings


def extract_time_mentions(query: str) -> List[str]:
    mentions: List[str] = []
    seen = set()
    for match in TIME_RE.finditer(query):
        mention = match.group(0).strip()
        if mention and mention not in seen:
            mentions.append(mention)
            seen.add(mention)
    return mentions


def normalize_entities(
    entities: Dict[str, List[str]],
    label_descriptions: Dict[str, str],
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for label in label_descriptions:
        values = entities.get(label, []) or []
        out[label] = sorted(
            {str(value).strip() for value in values if str(value).strip()}
        )
    return out


# --- per-worker model bundle -------------------------------------------------


class ModelBundle:
    """Per-process container. Each uvicorn worker has its own instance."""

    def __init__(
        self,
        tokenizer,
        model: LabelSemanticsNER,
        id2tag: Dict[int, str],
        label_descriptions: Dict[str, str],
        device: torch.device,
        max_length: int,
        merge_time_regex: bool,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.id2tag = id2tag
        self.label_descriptions = label_descriptions
        self.device = device
        self.max_length = max_length
        self.merge_time_regex = merge_time_regex
        # The model + label cache are not thread-safe to mutate concurrently, so
        # we serialize inference within a single worker and rely on multi-worker
        # processes for parallelism.
        self.lock = threading.Lock()


def _resolve_path(value: str) -> str:
    """Resolve a config path: absolute paths are kept, relative paths are
    anchored to PROJECT_ROOT so workers behave the same regardless of cwd."""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def _build_bundle() -> ModelBundle:
    model_path = _resolve_path(os.environ.get("LS_MODEL_PATH", "outputs/best_model"))
    label_file = _resolve_path(
        os.environ.get("LS_LABEL_FILE", "data/excel_ner_data/labels.json")
    )
    max_length = int(os.environ.get("LS_MAX_LENGTH", "128"))
    device_name = os.environ.get("LS_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    merge_time_regex = os.environ.get("LS_INCLUDE_TIME_REGEX", "0").lower() in (
        "1",
        "true",
        "yes",
    )

    device = torch.device(device_name)
    pid = os.getpid()
    logger.info(
        "[pid=%s] loading model from %s on device=%s (max_length=%s)",
        pid,
        model_path,
        device,
        max_length,
    )

    label_descriptions = load_label_descriptions(label_file)
    tag2id, id2tag = build_tag_maps(label_descriptions)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = LabelSemanticsNER(model_path, tag2id, label_descriptions).to(device)
    model.eval()

    warmup_env = os.environ.get("LS_WARMUP_QUERIES", "").strip()
    warmup_queries = [q for q in warmup_env.split(";") if q.strip()] if warmup_env else [
        "示例查询",
        "茅台2023年财报",
    ]
    for query in warmup_queries:
        try:
            predict_entities(
                query, model, tokenizer, id2tag, label_descriptions, max_length, device
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[pid=%s] warmup query %r failed: %s", pid, query, exc)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    logger.info("[pid=%s] model ready", pid)
    return ModelBundle(
        tokenizer=tokenizer,
        model=model,
        id2tag=id2tag,
        label_descriptions=label_descriptions,
        device=device,
        max_length=max_length,
        merge_time_regex=merge_time_regex,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=os.environ.get("LS_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    bundle = _build_bundle()
    app.state.bundle = bundle
    try:
        yield
    finally:
        # Help free GPU memory on graceful shutdown.
        app.state.bundle = None
        if bundle.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass


app = FastAPI(
    title="LabelSemantics NER Service",
    description="HTTP wrapper around tests/batch_test.py inference.",
    version="0.1.0",
    lifespan=lifespan,
    default_response_class=_DefaultResponseClass,
)


# Pure ASGI middleware: lower per-request overhead than BaseHTTPMiddleware
# (which wraps the response in an extra streaming context). Captures the
# wall-clock time spent inside the ASGI stack — i.e. uvicorn parse + FastAPI
# route handling + response build, but excluding the actual TCP send of the
# response body. Reported back to the client as ``X-Request-Total-Ms``.
class _TimingHeaderMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()

        async def _send(message):
            if message.get("type") == "http.response.start":
                elapsed_ms = (time.perf_counter() - start) * 1000
                headers = list(message.get("headers", []))
                headers.append((b"x-request-total-ms", f"{elapsed_ms:.3f}".encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, _send)


app.add_middleware(_TimingHeaderMiddleware)


# --- HTTP schema -------------------------------------------------------------
#
# These pydantic models are kept primarily for OpenAPI docs / IDE auto-complete
# on the client side. The route handlers themselves return raw ``dict`` objects
# (without ``response_model=...``) so we skip a second round of pydantic
# validation/serialization on the response, which measurably shaves milliseconds
# off the per-request critical path.


class PredictRequest(BaseModel):
    query: str = Field(..., min_length=1, description="原始 query 文本")
    include_time_regex: Optional[bool] = Field(
        default=None,
        description="是否在 TIME 字段中合并正则抽取的时间词；默认沿用 LS_INCLUDE_TIME_REGEX。",
    )


class TimingBreakdown(BaseModel):
    tokenize_ms: float
    model_ms: float
    decode_ms: float
    inference_ms: float
    server_handler_ms: float


class PredictResponse(BaseModel):
    query: str
    entities: Dict[str, List[str]]
    inference_ms: float
    timings: TimingBreakdown


class BatchPredictRequest(BaseModel):
    queries: List[str] = Field(..., min_length=1, max_length=128)
    include_time_regex: Optional[bool] = None


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]


# --- routes ------------------------------------------------------------------


@app.get("/health")
async def health():
    bundle: Optional[ModelBundle] = getattr(app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "status": "ok",
        "pid": os.getpid(),
        "device": str(bundle.device),
        "labels": list(bundle.label_descriptions.keys()),
    }


@app.get("/labels")
async def labels():
    bundle: Optional[ModelBundle] = getattr(app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return bundle.label_descriptions


def _run_predict(bundle: ModelBundle, query: str, include_time_regex: Optional[bool]):
    """Returns ``(entities_dict, timings_dict)``.

    ``timings_dict`` contains ``tokenize_ms`` / ``model_ms`` / ``decode_ms`` /
    ``inference_ms`` (sum of the previous three) and ``server_handler_ms``
    (handler-side wall time including normalize + optional time-regex merge).
    """
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    handler_start = time.perf_counter()
    with bundle.lock:
        raw, infer_timings = predict_entities_timed(
            query,
            bundle.model,
            bundle.tokenizer,
            bundle.id2tag,
            bundle.label_descriptions,
            bundle.max_length,
            bundle.device,
        )

    entities = normalize_entities(raw, bundle.label_descriptions)

    use_time_regex = (
        bundle.merge_time_regex if include_time_regex is None else include_time_regex
    )
    if use_time_regex and "TIME" in entities:
        merged = set(entities["TIME"]) | set(extract_time_mentions(query))
        entities["TIME"] = sorted(m for m in merged if m)
    handler_ms = (time.perf_counter() - handler_start) * 1000

    timings = {
        "tokenize_ms": round(infer_timings["tokenize_ms"], 3),
        "model_ms": round(infer_timings["model_ms"], 3),
        "decode_ms": round(infer_timings["decode_ms"], 3),
        "inference_ms": round(infer_timings["total_ms"], 3),
        "server_handler_ms": round(handler_ms, 3),
    }
    return entities, timings


@app.post("/predict")
def predict(req: PredictRequest):
    bundle: Optional[ModelBundle] = getattr(app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    entities, timings = _run_predict(bundle, req.query, req.include_time_regex)
    return {
        "query": req.query,
        "entities": entities,
        "inference_ms": timings["inference_ms"],
        "timings": timings,
    }


@app.post("/predict_batch")
def predict_batch(req: BatchPredictRequest):
    bundle: Optional[ModelBundle] = getattr(app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    results = []
    for query in req.queries:
        entities, timings = _run_predict(bundle, query, req.include_time_regex)
        results.append(
            {
                "query": query,
                "entities": entities,
                "inference_ms": timings["inference_ms"],
                "timings": timings,
            }
        )
    return {"results": results}
