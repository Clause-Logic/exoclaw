"""MicroPython implementation of ``exoclaw.http``.

Hand-rolled HTTP/1.1 over ``asyncio.open_connection``. Loaded only
on MicroPython by ``exoclaw.http.HTTPClient``; never imported on
CPython.

Always uses chunked transfer encoding for the request body so the
caller can stream it (``AsyncIterable[bytes]``) without ever
materialising the full body. Response side handles both
chunked-transfer and content-length framing, and yields decoded
UTF-8 lines for SSE iteration.

Async iterators here use the explicit ``__aiter__`` / ``__anext__``
class protocol rather than ``async def`` + ``yield``. MicroPython
1.27 collapses ``async def`` with ``yield`` into a plain generator
that doesn't expose ``__aiter__``, so a CPython-style ``async for
chunk in agen()`` raises ``AttributeError`` on MP. The class-based
protocol works identically on both runtimes.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from exoclaw._compat import monotonic_diff_ms, monotonic_ms
from exoclaw.http import (
    HTTPConnectError,
    HTTPError,
    HTTPReadTimeout,
    HTTPStatusError,
    HTTPWriteTimeout,
)

if TYPE_CHECKING:
    import ssl
    from collections.abc import AsyncIterable


def _deadline_ms(timeout_s: float) -> int:
    """Compute a deadline off the monotonic clock (not wall clock).

    On MP this maps to ``time.ticks_ms`` and pairs with
    ``monotonic_diff_ms`` to handle wrap-around safely. Wall clock
    (``time.time()``) is unsafe on a chip: NTP jumps push the
    deadline arbitrarily, and on a freshly booted board with no RTC
    the value is undefined."""
    return monotonic_ms() + int(timeout_s * 1000)


def _expired(deadline_ms: int) -> bool:
    """Has the deadline passed? Wrap-safe via ``monotonic_diff_ms``."""
    return monotonic_diff_ms(deadline_ms, monotonic_ms()) <= 0


async def _read_until_double_crlf(
    reader: asyncio.StreamReader,
    deadline_ms: int,
    max_bytes: int = 32 * 1024,
) -> bytes:
    """Read response head (status line + headers) up to ``\\r\\n\\r\\n``.

    Deadline-checked between reads instead of wrapped in
    ``asyncio.wait_for``. On MicroPython, ``wait_for(coro_doing_IO,
    t)`` puts the calling task into ``_task_queue`` (sleeping) while
    the IO-wrapping inner task registers in ``IOQueue``; combined
    with SSL streaming reads on the same socket this hits a
    pairheap-double-push assert in MP's asyncio. Deadline-only
    bounding keeps the task on exactly one wait queue at a time."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        if len(buf) > max_bytes:
            raise HTTPError("response head exceeded {} bytes".format(max_bytes))
        if _expired(deadline_ms):
            raise HTTPReadTimeout("response head deadline exceeded")
        chunk = await reader.read(1024)
        if not chunk:
            raise HTTPError("connection closed before response head complete")
        buf += chunk
    return buf


def _parse_response_head(head: bytes) -> tuple[int, dict[str, str], bytes]:
    """Return (status_code, lower-cased headers, leftover-body-bytes).

    Splits at ``\\r\\n\\r\\n``; bytes after are the first slice of
    the body that came in the same recv. Headers split on FIRST
    ``:`` to preserve values containing colons."""
    sep = head.find(b"\r\n\r\n")
    head_part = head[:sep]
    body_start = head[sep + 4 :]
    lines = head_part.split(b"\r\n")
    status_parts = lines[0].split(b" ", 2)
    if len(status_parts) < 2:
        raise HTTPError("malformed status line: {!r}".format(lines[0]))
    # ``int()`` raises ``ValueError`` on a non-numeric status code
    # (``HTTP/1.1 ABC OK``). Translate to ``HTTPError`` so callers
    # only need to handle the public taxonomy.
    try:
        status_code = int(status_parts[1])
    except ValueError:
        raise HTTPError("malformed status line: {!r}".format(lines[0]))
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        colon = line.find(b":")
        if colon == -1:
            continue
        k = line[:colon].decode("ascii", "replace").strip().lower()
        v = line[colon + 1 :].decode("ascii", "replace").strip()
        headers[k] = v
    return status_code, headers, body_start


class _ChunkedBodyIter:
    """Async iterator over a chunked-transfer-encoded body. Yields
    raw body bytes one HTTP chunk at a time. Reads from
    ``initial_buf`` first, then from the socket."""

    def __init__(self, reader: asyncio.StreamReader, initial_buf: bytes) -> None:
        self._reader = reader
        self._buf = initial_buf
        self._done = False

    def __aiter__(self) -> "_ChunkedBodyIter":
        return self

    async def _readline(self) -> bytes:
        while b"\r\n" not in self._buf:
            data = await self._reader.read(1024)
            if not data:
                raise HTTPError("chunked body closed mid-frame")
            self._buf += data
        idx = self._buf.find(b"\r\n")
        line = self._buf[:idx]
        self._buf = self._buf[idx + 2 :]
        return line

    async def _readexactly(self, n: int) -> bytes:
        while len(self._buf) < n:
            data = await self._reader.read(max(n - len(self._buf), 1024))
            if not data:
                raise HTTPError("chunked body closed mid-chunk")
            self._buf += data
        out = self._buf[:n]
        self._buf = self._buf[n:]
        return out

    async def __anext__(self) -> bytes:
        if self._done:  # pragma: no cover (micropython)
            # Defensive — ``async for`` never re-enters past
            # ``StopAsyncIteration``. Only fires if user code calls
            # ``__anext__`` directly after the iter completes; not
            # reachable from the normal usage path the test rig
            # exercises.
            raise StopAsyncIteration
        size_line = await self._readline()
        semi = size_line.find(b";")
        size_hex = size_line[:semi] if semi != -1 else size_line
        try:
            chunk_size = int(size_hex.strip(), 16)
        except ValueError:
            raise HTTPError("bad chunk size: {!r}".format(size_hex))
        if chunk_size == 0:
            # Discard trailers + final blank line.
            while True:
                trailer = await self._readline()
                if not trailer:
                    break
            self._done = True
            raise StopAsyncIteration
        data = await self._readexactly(chunk_size)
        # Trailing CRLF after the chunk data.
        await self._readexactly(2)
        return data


class _ContentLengthBodyIter:
    """Async iterator over a content-length-bounded body. Reads
    from ``initial_buf`` first, then from the socket."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        initial_buf: bytes,
        content_length: int,
    ) -> None:
        self._reader = reader
        self._buf = initial_buf
        self._remaining = content_length

    def __aiter__(self) -> "_ContentLengthBodyIter":
        return self

    async def __anext__(self) -> bytes:
        if self._remaining <= 0:
            raise StopAsyncIteration
        if self._buf:
            head = self._buf[: self._remaining]
            self._buf = self._buf[len(head) :]
            self._remaining -= len(head)
            return head
        data = await self._reader.read(min(self._remaining, 4096))
        if not data:
            self._remaining = 0
            raise StopAsyncIteration
        self._remaining -= len(data)
        return data


class _UntilCloseBodyIter:
    """Read-until-EOF body iterator. Used when the server sends
    no content-length and no chunked encoding (rare — some SSE
    proxies do this)."""

    def __init__(self, reader: asyncio.StreamReader, initial_buf: bytes) -> None:
        self._reader = reader
        self._buf = initial_buf
        self._eof = False

    def __aiter__(self) -> "_UntilCloseBodyIter":
        return self

    async def __anext__(self) -> bytes:
        if self._eof:
            raise StopAsyncIteration
        if self._buf:
            out = self._buf
            self._buf = b""
            return out
        data = await self._reader.read(4096)
        if not data:
            self._eof = True
            raise StopAsyncIteration
        return data


class _LineIter:
    """Async iterator yielding decoded UTF-8 lines from a body
    iterator. Splits on ``\\n``; strips trailing ``\\r``."""

    def __init__(self, body_iter: object) -> None:
        self._body = body_iter
        self._buf = b""
        self._exhausted = False

    def __aiter__(self) -> "_LineIter":
        return self

    async def __anext__(self) -> str:
        while True:
            nl = self._buf.find(b"\n")
            if nl != -1:
                line = self._buf[:nl]
                if line.endswith(b"\r"):
                    line = line[:-1]
                self._buf = self._buf[nl + 1 :]
                return line.decode("utf-8", "replace")
            if self._exhausted:
                if self._buf:
                    line = self._buf
                    self._buf = b""
                    return line.decode("utf-8", "replace")
                raise StopAsyncIteration
            try:
                chunk = await self._body.__anext__()  # type: ignore[attr-defined]
            except StopAsyncIteration:
                self._exhausted = True
                continue
            self._buf += chunk


class MPResponse:
    """``ResponseProto`` impl for MicroPython.

    Reads the response body lazily — ``aread()`` materialises it
    once and caches; ``aiter_lines()`` streams without buffering
    the full body."""

    def __init__(
        self,
        status_code: int,
        headers: dict[str, str],
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        initial_body: bytes,
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self._reader = reader
        self._writer = writer
        self._initial = initial_body
        self._read_body: bytes | None = None

    def _make_body_iter(self) -> object:
        te = self.headers.get("transfer-encoding", "").lower()
        if "chunked" in te:
            return _ChunkedBodyIter(self._reader, self._initial)
        cl = self.headers.get("content-length")
        if cl is not None:
            return _ContentLengthBodyIter(self._reader, self._initial, int(cl))
        return _UntilCloseBodyIter(self._reader, self._initial)

    async def aread(self) -> bytes:
        if self._read_body is not None:
            return self._read_body
        chunks: list[bytes] = []
        # ``async for`` so the body iterator's ``__aiter__`` runs
        # alongside ``__anext__`` — keeps the iter protocol exercised
        # symmetrically (matters for the MP coverage gate, which
        # wants both halves traced).
        async for chunk in self._make_body_iter():  # type: ignore[attr-defined]
            chunks.append(chunk)
        self._read_body = b"".join(chunks)
        return self._read_body

    @property
    def text(self) -> str:
        if self._read_body is None:
            raise RuntimeError("call aread() before .text on MP response")
        return self._read_body.decode("utf-8")

    def raise_for_status(self) -> None:
        if 400 <= self.status_code < 600:
            raise HTTPStatusError(self.status_code)

    def aiter_lines(self) -> _LineIter:
        return _LineIter(self._make_body_iter())


class MPStreamCM:
    """MP async context manager for one streaming POST.

    Connect → send request line + headers → stream request body
    with chunked transfer → read response head → hand off to
    ``MPResponse`` for body iteration."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None,
        content: "AsyncIterable[bytes] | bytes | None",
        timeout: float,
        ssl_context: "ssl.SSLContext | bool | None",
    ) -> None:
        self._url = url
        self._headers = headers or {}
        self._content = content
        self._timeout = timeout
        self._ssl_context = ssl_context
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    # ── Real network path: ``__aenter__`` opens a real socket via
    # ``asyncio.open_connection``, which the unit-test rig (no
    # network access) can't drive. The body of the method delegates
    # to helpers that ARE testable (``_send_request`` against a fake
    # writer; ``_parse_response_head`` against canned bytes), so the
    # uncovered surface is just the connect/wire-up wrapper. Mark
    # those lines pragma-no-cover for the MP coverage gate; CPython
    # exercises this entire flow via ``test_http.py``.
    async def __aenter__(self) -> MPResponse:  # pragma: no cover (micropython)
        from exoclaw.http import _parse_url

        scheme, host, port, path = _parse_url(self._url)
        ssl_arg: "ssl.SSLContext | bool | None"
        if scheme == "https":
            if self._ssl_context is not None:
                ssl_arg = self._ssl_context
            else:
                import ssl as _ssl

                # MP's stdlib ``ssl`` ships ``create_default_context``
                # only on the unix port; bare-metal builds expose a
                # smaller surface. Fall back to ``True`` for default
                # verification.
                if hasattr(_ssl, "create_default_context"):
                    ssl_arg = _ssl.create_default_context()
                else:
                    ssl_arg = True
        else:
            ssl_arg = None

        # Deadline-only bounding: ``wait_for(io_coro, t)`` on
        # MicroPython schedules the caller's task on _task_queue
        # (sleep) while the wrapped IO task registers in IOQueue —
        # collisions on long SSL streams hit a pairheap assert. We
        # use a single deadline that covers connect + send + head
        # read, checked between IO ops, so the task is on exactly
        # one wait queue at any moment.
        deadline = _deadline_ms(self._timeout)
        try:
            reader, writer = await asyncio.open_connection(host, port, ssl=ssl_arg)
        except OSError as e:
            raise HTTPConnectError("connect failed to {}:{}: {}".format(host, port, e)) from e
        # Assign before the deadline check so ``_close_writer`` can
        # find the socket on the leak path. ``_writer = None`` after
        # close keeps ``__aexit__`` idempotent.
        self._reader = reader
        self._writer = writer
        if _expired(deadline):
            await self._close_writer()
            raise HTTPConnectError("connect deadline exceeded for {}:{}".format(host, port))

        try:
            await self._send_request(writer, host, path, deadline)
            head = await _read_until_double_crlf(reader, deadline)
        except (HTTPReadTimeout, HTTPWriteTimeout):
            await self._close_writer()
            raise
        except OSError as e:
            await self._close_writer()
            raise HTTPConnectError("send/recv failed: {}".format(e)) from e

        status_code, headers, initial_body = _parse_response_head(head)
        return MPResponse(status_code, headers, reader, writer, initial_body)

    async def __aexit__(self, *exc: object) -> None:  # pragma: no cover (micropython)
        # Only invoked from a real ``async with client.stream_post(...)``
        # which requires a live socket — covered by the CPython httpx
        # path. Pair-pragma'd with ``__aenter__``.
        await self._close_writer()

    async def _close_writer(self) -> None:  # pragma: no cover (micropython)
        # See ``__aexit__`` — invoked only from real-network code
        # paths.
        writer = self._writer
        if writer is None:
            return
        try:
            close = getattr(writer, "close", None)
            if close is not None:
                close()
            wait_closed = getattr(writer, "wait_closed", None)
            if wait_closed is not None:
                await wait_closed()
        except Exception:
            pass
        self._writer = None

    async def _send_request(
        self, writer: asyncio.StreamWriter, host: str, path: str, deadline_ms: int
    ) -> None:
        """Build and send the HTTP/1.1 request.

        Always uses chunked transfer encoding for the body so the
        same code path serves streaming-iterable and one-shot bytes.

        ``deadline_ms`` bounds total send time. Drains run unwrapped
        — wrapping with ``asyncio.wait_for`` mixes a sleep into the
        same task that's also doing IO via ``_io_queue.queue_write``,
        which on MP races into a pairheap-double-push assert.
        """
        lines = ["POST {} HTTP/1.1".format(path), "Host: {}".format(host)]
        seen = {k.lower() for k in self._headers}
        if "user-agent" not in seen:
            lines.append("User-Agent: exoclaw-http/1.0")
        if "accept" not in seen:
            lines.append("Accept: */*")
        lines.append("Transfer-Encoding: chunked")
        for k, v in self._headers.items():
            if k.lower() == "transfer-encoding":
                continue
            lines.append("{}: {}".format(k, v))
        lines.append("")
        lines.append("")
        head = "\r\n".join(lines).encode("ascii")
        writer.write(head)
        drain = getattr(writer, "drain", None)
        if drain is not None:
            if _expired(deadline_ms):
                raise HTTPWriteTimeout("header write deadline exceeded")
            await drain()

        content = self._content
        if content is None:
            writer.write(b"0\r\n\r\n")
            if drain is not None:
                if _expired(deadline_ms):
                    raise HTTPWriteTimeout("body write deadline exceeded")
                await drain()
            return

        if isinstance(content, (bytes, bytearray)):
            if content:
                writer.write(
                    "{:x}\r\n".format(len(content)).encode("ascii") + bytes(content) + b"\r\n"
                )
            writer.write(b"0\r\n\r\n")
            if drain is not None:
                if _expired(deadline_ms):
                    raise HTTPWriteTimeout("body write deadline exceeded")
                await drain()
            return

        # Async iterable of bytes — pump through chunked framing.
        # Two iteration styles to support:
        #   * Class-based async iterators (``__aiter__`` /
        #     ``__anext__``) — driven via ``await iter.__anext__()``.
        #   * MP ``async def`` + ``yield`` — produces a plain
        #     generator with neither ``__aiter__`` nor ``__anext__``.
        #     Iterate with sync ``for``; ``await`` inside the
        #     generator body collapses to ``yield from`` so the
        #     scheduler still drives any I/O it does.
        if hasattr(content, "__aiter__") or hasattr(content, "__anext__"):
            content_iter = content.__aiter__() if hasattr(content, "__aiter__") else content
            while True:
                try:
                    chunk = await content_iter.__anext__()
                except StopAsyncIteration:
                    break
                if not chunk:
                    continue
                writer.write("{:x}\r\n".format(len(chunk)).encode("ascii") + bytes(chunk) + b"\r\n")
                if drain is not None:
                    if _expired(deadline_ms):
                        raise HTTPWriteTimeout("body write deadline exceeded")
                    await drain()
        else:
            for chunk in content:
                if not chunk:
                    continue
                writer.write("{:x}\r\n".format(len(chunk)).encode("ascii") + bytes(chunk) + b"\r\n")
                if drain is not None:
                    if _expired(deadline_ms):
                        raise HTTPWriteTimeout("body write deadline exceeded")
                    await drain()
        writer.write(b"0\r\n\r\n")
        if drain is not None:
            if _expired(deadline_ms):
                raise HTTPWriteTimeout("body write deadline exceeded")
            await drain()


class MPClient:
    """``ClientProto`` impl for MicroPython.

    No connection pool — each ``stream_post`` opens a fresh socket.
    The chip workload is one-call-at-a-time, so pooling wouldn't
    pay back the bookkeeping. ``ssl_context`` is forwarded to
    ``asyncio.open_connection`` for ``https`` URLs."""

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        ssl_context: "ssl.SSLContext | bool | None" = None,
    ) -> None:
        self._timeout = timeout
        self._ssl_context = ssl_context

    async def aclose(self) -> None:
        return None

    def stream_post(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        content: "AsyncIterable[bytes] | bytes | None" = None,
        timeout: float | None = None,
    ) -> MPStreamCM:
        return MPStreamCM(url, headers, content, timeout or self._timeout, self._ssl_context)
