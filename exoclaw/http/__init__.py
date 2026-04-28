"""Cross-runtime async HTTP/1.1 client.

CPython delegates to ``httpx`` (battle-tested connection pooling,
retries, robust HTTP parsing). MicroPython implements a hand-rolled
HTTP/1.1 path on top of ``asyncio.open_connection`` — chunked-
transfer request bodies (so big prompts don't materialise as one
buffer), chunked-transfer response parsing, line-oriented SSE
iteration. Same source either way; the runtime gate selects which
submodule is loaded.

Plugin authors that need HTTP should import ``HTTPClient`` from
here (and annotate against ``ClientProto`` / ``ResponseProto`` /
``StreamCMProto`` if they need static types) instead of taking a
hard ``httpx`` dependency. ``httpx`` doesn't run on MicroPython
(C-extension dependencies, threading model, sniffio); a plugin
that imports it directly excludes itself from the chip target.

Surface:

- ``HTTPClient(*, timeout=60.0, ssl_context=None)`` — async client.
  ``aclose()`` to release. CPython pools connections via httpx; MP
  opens a fresh socket per call.
- ``HTTPClient.stream_post(url, *, headers=None, content=None,
  timeout=None, method="POST")`` — async context manager yielding
  a streaming response. ``content`` accepts bytes or
  ``AsyncIterable[bytes]``; the iterable variant enables the
  streaming-prompt memory win. Set ``method="GET"`` (or other
  bodyless methods) to drive a chunked-response read without
  sending a body — the chip web_fetch path needs this so that a
  large HTML response can be parsed incrementally without ever
  materialising the whole body in heap.
- Response interface — ``status_code``, ``headers`` (lower-cased
  keys), ``aread()``, ``text`` (post-aread), ``raise_for_status()``,
  ``aiter_lines()``.
- Exceptions: ``HTTPError`` (base), ``HTTPConnectError``,
  ``HTTPReadTimeout``, ``HTTPWriteTimeout``, ``HTTPStatusError``.

Layout: split into per-runtime submodules so the MP heap only loads
the bytecode it actually needs. Importing the public name
``HTTPClient`` triggers loading exactly one of ``_cpython`` /
``_mp``; the other is never compiled.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol

from exoclaw._compat import IS_MICROPYTHON

if TYPE_CHECKING:
    # ``collections.abc`` doesn't ship on MicroPython; pulled in for
    # type-checking only. ``from __future__ import annotations`` at
    # the top stringifies all annotations so the runtime never
    # resolves these names.
    import ssl
    from collections.abc import AsyncIterable, AsyncIterator


# ── Exceptions ───────────────────────────────────────────────────


class HTTPError(Exception):
    """Base class for ``exoclaw.http`` transport errors."""


class HTTPConnectError(HTTPError):
    """TCP / TLS connect failure (DNS, refused, handshake)."""


class HTTPReadTimeout(HTTPError):  # noqa: N818
    """Server didn't send bytes within the timeout.

    Named without an ``Error`` suffix to mirror httpx's
    ``httpx.ReadTimeout`` / ``httpx.WriteTimeout`` taxonomy that
    plugin authors are likely to be familiar with."""


class HTTPWriteTimeout(HTTPError):  # noqa: N818
    """Couldn't push request bytes within the timeout."""


class HTTPStatusError(HTTPError):
    """4xx / 5xx response. ``raise_for_status`` raises this."""

    def __init__(self, status_code: int, message: str = "") -> None:
        self.status_code = status_code
        super().__init__(message or f"HTTP {status_code}")


# ── Public typing surface ───────────────────────────────────────


class ResponseProto(Protocol):
    """Common surface for streaming responses on both runtimes."""

    @property
    def status_code(self) -> int: ...

    @property
    def headers(self) -> dict[str, str]: ...

    async def aread(self) -> bytes: ...

    @property
    def text(self) -> str: ...

    def raise_for_status(self) -> None: ...

    def aiter_lines(self) -> AsyncIterator[str]: ...


class StreamCMProto(Protocol):
    """Async context manager for one streaming POST."""

    async def __aenter__(self) -> ResponseProto: ...

    async def __aexit__(self, *exc: object) -> None: ...


class ClientProto(Protocol):
    """Common surface for the async HTTP client on both runtimes."""

    async def aclose(self) -> None: ...

    def stream_post(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        content: AsyncIterable[bytes] | bytes | None = None,
        timeout: float | None = None,
        method: str = "POST",
    ) -> StreamCMProto: ...


# ── URL parsing (used by both runtimes) ──────────────────────────


def _parse_url(url: str) -> tuple[str, str, int, str]:
    """Return (scheme, host, port, path-with-query).

    Minimal URL parser — handles ``http://`` / ``https://`` URLs.
    No userinfo support, no IPv6 brackets."""
    if url.startswith("https://"):
        scheme = "https"
        rest = url[8:]
        default_port = 443
    elif url.startswith("http://"):
        scheme = "http"
        rest = url[7:]
        default_port = 80
    else:
        raise ValueError(f"unsupported URL scheme: {url!r}")

    slash = rest.find("/")
    if slash == -1:
        authority = rest
        path = "/"
    else:
        authority = rest[:slash]
        path = rest[slash:] or "/"

    colon = authority.rfind(":")
    if colon == -1:
        host = authority
        port = default_port
    else:
        host = authority[:colon]
        port = int(authority[colon + 1 :])
    return scheme, host, port, path


# ── Public façade ───────────────────────────────────────────────


def HTTPClient(  # noqa: N802  — factory named for ergonomic ``HTTPClient(...)`` call site
    *, timeout: float = 60.0, ssl_context: "ssl.SSLContext | bool | None" = None
) -> ClientProto:
    """Return a runtime-appropriate async HTTP client.

    CPython: thin wrapper over a pooled ``httpx.AsyncClient``.
    MicroPython: hand-rolled HTTP/1.1 over ``asyncio.open_connection``.

    Both expose the same interface — ``aclose()`` and
    ``stream_post(url, *, headers, content, timeout)``."""
    if IS_MICROPYTHON:  # pragma: no cover (cpython)
        from exoclaw.http._mp import MPClient

        return MPClient(timeout=timeout, ssl_context=ssl_context)
    from exoclaw.http._cpython import HttpxClient  # pragma: no cover (micropython)

    return HttpxClient(timeout=timeout, ssl_context=ssl_context)  # pragma: no cover (micropython)


# ── JSON helper ─────────────────────────────────────────────────


async def post_json(
    client: ClientProto,
    url: str,
    payload: dict[str, object],
    *,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> object:
    """POST a JSON dict, return the parsed JSON response.

    Builds ``Content-Type: application/json``, encodes the payload
    once, runs the request, raises on HTTP errors, returns the
    decoded JSON. For streaming responses use
    ``HTTPClient.stream_post`` directly."""
    body = json.dumps(payload).encode("utf-8")
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)
    async with client.stream_post(
        url, headers=merged_headers, content=body, timeout=timeout
    ) as resp:
        await resp.aread()
        resp.raise_for_status()
        return json.loads(resp.text)


# CPython-only re-export: tests / advanced callers can wrap an
# existing ``httpx.AsyncClient`` instead of using the ``HTTPClient``
# factory. Not exposed on MP — ``_cpython`` isn't staged there.
if not IS_MICROPYTHON:  # pragma: no cover (micropython)
    from exoclaw.http._cpython import from_httpx  # noqa: F401


__all__ = [
    "ClientProto",
    "HTTPClient",
    "HTTPConnectError",
    "HTTPError",
    "HTTPReadTimeout",
    "HTTPStatusError",
    "HTTPWriteTimeout",
    "ResponseProto",
    "StreamCMProto",
    "post_json",
]
# ``from_httpx`` is CPython-only — the helper lives in
# ``_cpython.py`` and exists to wrap a pre-built
# ``httpx.AsyncClient`` for tests. On MicroPython the symbol
# isn't defined, so listing it unconditionally in ``__all__``
# would break ``from exoclaw.http import *``.
if not IS_MICROPYTHON:  # pragma: no cover (micropython)
    __all__.append("from_httpx")
