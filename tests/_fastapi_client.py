"""Tiny stdlib HTTP client shared by the fastapi_*_test.py scripts.

Built on ``http.client`` (not ``urllib.request``) so the underlying TCP socket
can be reused across calls via HTTP keep-alive — that alone removes one of the
biggest contributors to the gap between ``client_total_ms`` and the server-side
inference time, especially on localhost.

Each :class:`ServiceClient` instance keeps a per-thread connection so it is safe
to use from multiple threads (the stress test shares a single client across a
``ThreadPoolExecutor``); concurrent calls do not serialize on a single socket.

Response headers from the most recent call are exposed via
:attr:`ServiceClient.last_headers` (lowercased keys), which is how the test
scripts read the ``X-Request-Total-Ms`` header injected by the server-side
ASGI middleware.
"""

from __future__ import annotations

import http.client
import json
import threading
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit


class ServiceClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        parsed = urlsplit(base_url.rstrip("/"))
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"unsupported scheme in base_url: {base_url!r}")
        self._scheme = parsed.scheme
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._base_path = parsed.path  # e.g. "" or "/api"
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Per-thread keep-alive connection so concurrent callers do not
        # serialize on a single socket.
        self._tls = threading.local()
        # Lowercased response headers from the most recent call (per-thread
        # would also be reasonable; we keep it simple — callers that need
        # strict isolation can use one ServiceClient per thread).
        self.last_headers: Dict[str, str] = {}
        self.last_status: int = 0

    def _get_conn(self) -> http.client.HTTPConnection:
        conn: Optional[http.client.HTTPConnection] = getattr(self._tls, "conn", None)
        if conn is None:
            if self._scheme == "https":
                conn = http.client.HTTPSConnection(
                    self._host, self._port, timeout=self.timeout
                )
            else:
                conn = http.client.HTTPConnection(
                    self._host, self._port, timeout=self.timeout
                )
            self._tls.conn = conn
        return conn

    def _close_conn(self) -> None:
        conn: Optional[http.client.HTTPConnection] = getattr(self._tls, "conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._tls.conn = None

    def close(self) -> None:
        self._close_conn()

    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> Any:
        full_path = self._base_path + path
        body: Optional[bytes] = None
        headers = {
            "Accept": "application/json",
            "Connection": "keep-alive",
        }
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))

        # One transparent retry: keep-alive sockets can be closed by the
        # server (idle timeout, worker restart) and the next request will
        # raise RemoteDisconnected / BadStatusLine. Re-open and retry once.
        last_exc: Optional[BaseException] = None
        for attempt in range(2):
            conn = self._get_conn()
            try:
                conn.request(method, full_path, body=body, headers=headers)
                resp = conn.getresponse()
                response_body = resp.read()
                self.last_status = resp.status
                self.last_headers = {k.lower(): v for k, v in resp.getheaders()}
                if resp.status >= 400:
                    detail = response_body.decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"{method} {self.base_url}{path} -> HTTP {resp.status}: {detail}"
                    )
                if not response_body:
                    return None
                try:
                    return json.loads(response_body)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"{method} {self.base_url}{path} returned non-JSON body: "
                        f"{response_body[:200]!r}"
                    ) from exc
            except (
                http.client.RemoteDisconnected,
                http.client.BadStatusLine,
                ConnectionResetError,
                BrokenPipeError,
            ) as exc:
                last_exc = exc
                self._close_conn()
                if attempt == 1:
                    break
                continue
            except OSError as exc:
                self._close_conn()
                raise RuntimeError(f"{method} {self.base_url}{path} failed: {exc}") from exc
        raise RuntimeError(
            f"{method} {self.base_url}{path} failed after retry: {last_exc!r}"
        ) from last_exc

    def server_request_total_ms(self) -> Optional[float]:
        """Return the ``X-Request-Total-Ms`` header from the most recent
        response (full server-side ASGI handling time), or ``None`` if the
        header is missing."""
        value = self.last_headers.get("x-request-total-ms")
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def labels(self) -> Dict[str, str]:
        return self._request("GET", "/labels")

    def predict(
        self, query: str, include_time_regex: Optional[bool] = None
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"query": query}
        if include_time_regex is not None:
            payload["include_time_regex"] = include_time_regex
        return self._request("POST", "/predict", payload)

    def predict_batch(
        self, queries: List[str], include_time_regex: Optional[bool] = None
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"queries": queries}
        if include_time_regex is not None:
            payload["include_time_regex"] = include_time_regex
        return self._request("POST", "/predict_batch", payload)
