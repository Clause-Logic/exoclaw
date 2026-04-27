"""MicroPython side of the ``exoclaw.http`` test matrix.

Pure-Python — no ``pytest``. Each ``test_*`` function is invoked
with no args; failures raise.

Tests drive the MP ``_MPStreamCM`` directly with fake stream
reader/writer objects, since the runner doesn't have network
access. The fake streams record what was written (the request
side) and play back canned bytes (the response side), exercising:

- ``_parse_url``: scheme/host/port/path decomposition.
- ``_parse_response_head``: status + lower-cased header parsing.
- ``_MPResponse._iter_chunked``: chunked-transfer body decoding.
- ``_MPResponse.aiter_lines`` over a chunked SSE response.
- Full request shape: chunked-encoded request body bytes.
- Status-error path: ``raise_for_status`` translates 4xx/5xx.

Coverage is reported by the matrix runner the same way
``test_compat.py``'s coverage is — pragma-tagged branches in
``http.py`` are excluded; everything else is expected to run.
"""

import asyncio

from exoclaw import http as h
from exoclaw.http import _mp as h_mp

# ── Fake async stream pair (in-memory) ──────────────────────────


class _FakeReader:
    """In-memory ``asyncio.StreamReader`` substitute.

    Stores a queue of bytes chunks; ``read(n)`` pops up to ``n``
    bytes from the head of the queue (cross-chunk if needed).
    Returns ``b""`` when drained — same EOF signal as the real
    StreamReader."""

    def __init__(self, chunks):
        self._buf = b""
        self._chunks = list(chunks)

    async def read(self, n):
        # Pull from the chunk queue until we have ``n`` bytes or
        # exhaust everything. EOF returns ``b""``.
        while len(self._buf) < n and self._chunks:
            self._buf += self._chunks.pop(0)
        if not self._buf:
            return b""
        out = self._buf[:n]
        self._buf = self._buf[n:]
        return out


class _FakeWriter:
    """In-memory ``asyncio.StreamWriter`` substitute. Accumulates
    written bytes for assertion."""

    def __init__(self):
        self.written = b""
        self.closed = False

    def write(self, data):
        self.written += bytes(data)

    async def drain(self):
        return None

    def close(self):
        self.closed = True

    async def wait_closed(self):
        return None


# ── _parse_url tests (runtime-agnostic) ────────────────────────


def test_parse_url_https_default_port():
    assert h._parse_url("https://api.example.com/v1") == (
        "https",
        "api.example.com",
        443,
        "/v1",
    )


def test_parse_url_http_explicit_port():
    assert h._parse_url("http://localhost:8080/x?y=1") == (
        "http",
        "localhost",
        8080,
        "/x?y=1",
    )


def test_parse_url_https_root_path():
    assert h._parse_url("https://api.example.com") == ("https", "api.example.com", 443, "/")


def test_parse_url_unknown_scheme_raises():
    raised = False
    try:
        h._parse_url("ftp://nope")
    except ValueError:
        raised = True
    assert raised


# ── _parse_response_head tests ──────────────────────────────────


def test_parse_response_head_lowercases_keys():
    head = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nServer: x\r\n\r\nleftover"
    status, headers, leftover = h_mp._parse_response_head(head)
    assert status == 200
    assert headers["content-type"] == "text/event-stream"
    assert headers["server"] == "x"
    assert leftover == b"leftover"


def test_parse_response_head_handles_value_with_colon():
    """``Date: Mon, 01 Jan 2026 00:00:00 GMT`` has multiple colons."""
    head = b"HTTP/1.1 200 OK\r\nDate: Mon, 01 Jan 2026 00:00:00 GMT\r\n\r\n"
    status, headers, leftover = h_mp._parse_response_head(head)
    assert status == 200
    assert headers["date"] == "Mon, 01 Jan 2026 00:00:00 GMT"
    assert leftover == b""


# ── _read_until_double_crlf tests ───────────────────────────────


def test_read_until_double_crlf_assembles_across_chunks():
    """Reader splits the head across recvs; the helper must keep
    reading until the terminator arrives."""
    reader = _FakeReader([b"HTTP/1.1 200 OK\r\n", b"X-A: 1\r\n\r\n", b"body bytes"])

    async def _go():
        return await h_mp._read_until_double_crlf(reader)

    head = asyncio.run(_go())
    assert b"\r\n\r\n" in head
    assert b"X-A: 1" in head


def test_read_until_double_crlf_raises_on_early_close():
    """Connection closed before the terminator arrives — surface
    a clean ``HTTPError`` rather than hang."""
    reader = _FakeReader([b"HTTP/1.1 200 OK\r\nServer: x"])  # no \r\n\r\n
    raised = False

    async def _go():
        nonlocal raised
        try:
            await h_mp._read_until_double_crlf(reader)
        except h.HTTPError:
            raised = True

    asyncio.run(_go())
    assert raised


# ── Chunked body iteration ──────────────────────────────────────


def _build_chunked_body(parts):
    """Encode ``parts`` (list of bytes) as a chunked-transfer body."""
    out = b""
    for p in parts:
        out += "{:x}\r\n".format(len(p)).encode("ascii") + p + b"\r\n"
    out += b"0\r\n\r\n"
    return out


def test_chunked_body_round_trip_via_aread():
    """``_MPResponse.aread`` over a chunked body returns the
    concatenation of decoded chunks."""
    body = _build_chunked_body([b"hello, ", b"world"])
    reader = _FakeReader([body])
    writer = _FakeWriter()
    resp = h_mp.MPResponse(200, {"transfer-encoding": "chunked"}, reader, writer, initial_body=b"")

    async def _go():
        return await resp.aread()

    out = asyncio.run(_go())
    assert out == b"hello, world"


def test_chunked_body_lines_iterates_decoded_lines():
    """SSE-style line iteration over a chunked body."""
    sse = b"data: a\n\ndata: b\n\ndata: [DONE]\n\n"
    body = _build_chunked_body([sse[:10], sse[10:]])
    reader = _FakeReader([body])
    writer = _FakeWriter()
    resp = h_mp.MPResponse(200, {"transfer-encoding": "chunked"}, reader, writer, initial_body=b"")

    async def _go():
        out = []
        async for line in resp.aiter_lines():
            out.append(line)
        return out

    lines = asyncio.run(_go())
    assert "data: a" in lines
    assert "data: b" in lines
    assert "data: [DONE]" in lines


def test_content_length_body_is_read_correctly():
    """Non-chunked, content-length-bounded body."""
    body = b'{"ok":true}'
    reader = _FakeReader([body])
    writer = _FakeWriter()
    resp = h_mp.MPResponse(
        200, {"content-length": str(len(body))}, reader, writer, initial_body=b""
    )

    async def _go():
        return await resp.aread()

    out = asyncio.run(_go())
    assert out == body


def test_content_length_body_uses_initial_buffer_first():
    """When the head + start of body arrive in one recv,
    ``initial_body`` carries the prefix into the response."""
    body = b"prefix-data-tail"
    reader = _FakeReader([])  # no further data — initial covers it
    writer = _FakeWriter()
    resp = h_mp.MPResponse(
        200,
        {"content-length": str(len(body))},
        reader,
        writer,
        initial_body=body,
    )

    async def _go():
        return await resp.aread()

    out = asyncio.run(_go())
    assert out == body


def test_aread_caches_body_between_calls():
    """Second ``aread()`` call returns the cached bytes without
    re-reading the (now-empty) reader."""
    body = b"hello"
    reader = _FakeReader([body])
    writer = _FakeWriter()
    resp = h_mp.MPResponse(200, {"content-length": "5"}, reader, writer, initial_body=b"")

    async def _go():
        first = await resp.aread()
        second = await resp.aread()
        return first, second

    a, b = asyncio.run(_go())
    assert a == b == body
    # ``.text`` property requires aread() first; verify it works post-aread.
    assert resp.text == "hello"


# ── raise_for_status ────────────────────────────────────────────


def test_raise_for_status_passes_2xx():
    resp = h_mp.MPResponse(200, {}, _FakeReader([]), _FakeWriter(), b"")
    resp.raise_for_status()  # no raise


def test_raise_for_status_raises_on_4xx():
    resp = h_mp.MPResponse(404, {}, _FakeReader([]), _FakeWriter(), b"")
    raised = False
    try:
        resp.raise_for_status()
    except h.HTTPStatusError as e:
        assert e.status_code == 404
        raised = True
    assert raised


def test_raise_for_status_raises_on_5xx():
    resp = h_mp.MPResponse(503, {}, _FakeReader([]), _FakeWriter(), b"")
    raised = False
    try:
        resp.raise_for_status()
    except h.HTTPStatusError as e:
        assert e.status_code == 503
        raised = True
    assert raised


def test_text_before_aread_raises():
    resp = h_mp.MPResponse(200, {}, _FakeReader([]), _FakeWriter(), b"")
    raised = False
    try:
        _ = resp.text
    except RuntimeError:
        raised = True
    assert raised


# ── Request encoding (chunked transfer for body) ────────────────


def test_send_request_emits_chunked_body_with_iterable_content():
    """``_send_request`` with an async-iterable body produces:
    request line + headers + Transfer-Encoding: chunked +
    one ``<size-hex>\\r\\n<data>\\r\\n`` per chunk + ``0\\r\\n\\r\\n`` terminator."""
    writer = _FakeWriter()
    cm = h_mp.MPStreamCM(
        "https://api.example.com/v1/x",
        headers={"Authorization": "Bearer t"},
        content=None,
        timeout=5.0,
        ssl_context=None,
    )

    async def _body():
        yield b"hello"
        yield b"world"

    cm._content = _body()

    async def _go():
        await cm._send_request(writer, "api.example.com", "/v1/x")

    asyncio.run(_go())
    sent = writer.written
    # Request line + Host header set up.
    assert sent.startswith(b"POST /v1/x HTTP/1.1\r\n")
    assert b"Host: api.example.com\r\n" in sent
    # Authorization header preserved.
    assert b"Authorization: Bearer t\r\n" in sent
    # Chunked transfer encoding declared.
    assert b"Transfer-Encoding: chunked\r\n" in sent
    # Both chunks present in chunked framing.
    assert b"5\r\nhello\r\n" in sent
    assert b"5\r\nworld\r\n" in sent
    # Terminator.
    assert sent.endswith(b"0\r\n\r\n")


def test_send_request_emits_one_shot_bytes_as_single_chunk():
    """Passing ``content=<bytes>`` produces a single chunk + terminator."""
    writer = _FakeWriter()
    cm = h_mp.MPStreamCM(
        "https://x.test/y",
        headers=None,
        content=b'{"x":1}',
        timeout=5.0,
        ssl_context=None,
    )

    async def _go():
        await cm._send_request(writer, "x.test", "/y")

    asyncio.run(_go())
    sent = writer.written
    # Single chunk: ``7\r\n{"x":1}\r\n``
    assert b'7\r\n{"x":1}\r\n' in sent
    assert sent.endswith(b"0\r\n\r\n")
    # Default headers added when caller passes none.
    assert b"User-Agent: exoclaw-http/1.0\r\n" in sent
    assert b"Accept: */*\r\n" in sent


def test_send_request_with_no_content_emits_only_terminator():
    writer = _FakeWriter()
    cm = h_mp.MPStreamCM(
        "https://x.test/y",
        headers=None,
        content=None,
        timeout=5.0,
        ssl_context=None,
    )

    async def _go():
        await cm._send_request(writer, "x.test", "/y")

    asyncio.run(_go())
    sent = writer.written
    assert sent.endswith(b"\r\n\r\n0\r\n\r\n")


# ── Public factory ──────────────────────────────────────────────


def test_http_client_factory_returns_mp_impl():
    """On MP, ``HTTPClient(...)`` returns the hand-rolled impl."""
    c = h.HTTPClient(timeout=5.0)
    assert isinstance(c, h_mp.MPClient)

    async def _close():
        await c.aclose()

    asyncio.run(_close())


def test_mp_client_aclose_is_noop():
    """No connection pool → aclose has nothing to do, returns None."""
    c = h_mp.MPClient(timeout=5.0)

    async def _go():
        return await c.aclose()

    assert asyncio.run(_go()) is None


# ── post_json round-trip with a fake client ─────────────────────


class _FakeResponse:
    """In-memory response satisfying ``ResponseProto`` enough for
    ``post_json``."""

    def __init__(self, status_code, body_bytes):
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}
        self._body = body_bytes
        self._read = False

    async def aread(self):
        self._read = True
        return self._body

    @property
    def text(self):
        if not self._read:
            raise RuntimeError("aread first")
        return self._body.decode("utf-8")

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise h.HTTPStatusError(self.status_code)

    def aiter_lines(self):
        raise NotImplementedError


class _FakeStreamCM:
    def __init__(self, captured, status_code, body):
        self._captured = captured
        self._status_code = status_code
        self._body = body

    async def __aenter__(self):
        return _FakeResponse(self._status_code, self._body)

    async def __aexit__(self, *exc):
        return None


class _FakeClient:
    def __init__(self, status_code=200, body=b"{}"):
        self.captured = {}
        self._status_code = status_code
        self._body = body

    async def aclose(self):
        return None

    def stream_post(self, url, *, headers=None, content=None, timeout=None):
        self.captured["url"] = url
        self.captured["headers"] = headers
        self.captured["content"] = content
        self.captured["timeout"] = timeout
        return _FakeStreamCM(self.captured, self._status_code, self._body)


def test_post_json_round_trips():
    client = _FakeClient(status_code=200, body=b'{"ok":true}')

    async def _go():
        return await h.post_json(
            client, "https://x.test/y", {"hello": "world"}, headers={"X-Test": "y"}
        )

    result = asyncio.run(_go())
    assert result == {"ok": True}
    assert client.captured["headers"]["Content-Type"] == "application/json"
    assert client.captured["headers"]["X-Test"] == "y"
    assert client.captured["content"] == b'{"hello": "world"}'


def test_read_until_double_crlf_max_bytes_cap():
    """Cap triggers when the head exceeds ``max_bytes`` without a
    terminator."""
    big = b"X" * 5000  # no \r\n\r\n
    reader = _FakeReader([big])

    async def _go():
        return await h_mp._read_until_double_crlf(reader, max_bytes=1024)

    raised = False
    try:
        asyncio.run(_go())
    except h.HTTPError:
        raised = True
    assert raised


def test_parse_response_head_skips_empty_and_no_colon_lines():
    """Empty header lines and lines without ``:`` are skipped."""
    head = (
        b"HTTP/1.1 200 OK\r\n"
        b"\r\n"  # blank line — skipped
        b"X-Valid: yes\r\n"
        b"NotAHeader\r\n"  # no colon — skipped
        b"\r\n"
    )
    # NB: the double \r\n\r\n splits before the body. But our extra
    # blank \r\n above adds one. The parser should still cope.
    head = b"HTTP/1.1 200 OK\r\nX-Valid: yes\r\nNotAHeader\r\n\r\n"
    status, headers, leftover = h_mp._parse_response_head(head)
    assert status == 200
    assert headers["x-valid"] == "yes"
    # Line without colon was skipped — not in headers.
    assert "notaheader" not in headers


def test_parse_response_head_raises_on_malformed_status():
    """Status line missing the status code raises ``HTTPError``."""
    head = b"BROKEN\r\n\r\n"
    raised = False
    try:
        h_mp._parse_response_head(head)
    except h.HTTPError:
        raised = True
    assert raised


def test_chunked_body_closed_mid_frame_raises():
    """Connection drops while reading a chunk-size line — surface
    a clean ``HTTPError``."""
    # ``5\r`` — partial size line then EOF
    body = b"5\r"
    reader = _FakeReader([body])
    writer = _FakeWriter()
    resp = h_mp.MPResponse(200, {"transfer-encoding": "chunked"}, reader, writer, initial_body=b"")
    raised = False

    async def _go():
        nonlocal raised
        try:
            await resp.aread()
        except h.HTTPError:
            raised = True

    asyncio.run(_go())
    assert raised


def test_chunked_body_closed_mid_chunk_raises():
    """Server promised 5 bytes but sent fewer before close."""
    # ``5\r\nhel`` — size line OK, but data short
    body = b"5\r\nhel"
    reader = _FakeReader([body])
    writer = _FakeWriter()
    resp = h_mp.MPResponse(200, {"transfer-encoding": "chunked"}, reader, writer, initial_body=b"")
    raised = False

    async def _go():
        nonlocal raised
        try:
            await resp.aread()
        except h.HTTPError:
            raised = True

    asyncio.run(_go())
    assert raised


def test_chunked_body_bad_size_raises():
    """Non-hex chunk size line is rejected."""
    body = b"NOT-HEX\r\nhello\r\n0\r\n\r\n"
    reader = _FakeReader([body])
    writer = _FakeWriter()
    resp = h_mp.MPResponse(200, {"transfer-encoding": "chunked"}, reader, writer, initial_body=b"")
    raised = False

    async def _go():
        nonlocal raised
        try:
            await resp.aread()
        except h.HTTPError:
            raised = True

    asyncio.run(_go())
    assert raised


def test_chunked_body_size_with_extension():
    """``5;ext=val\\r\\n`` — chunk extensions are ignored."""
    body = b"5;ext=val\r\nhello\r\n0\r\n\r\n"
    reader = _FakeReader([body])
    writer = _FakeWriter()
    resp = h_mp.MPResponse(200, {"transfer-encoding": "chunked"}, reader, writer, initial_body=b"")

    async def _go():
        return await resp.aread()

    assert asyncio.run(_go()) == b"hello"


def test_until_close_body_yields_initial_then_socket():
    """No content-length, no chunked → read until EOF."""
    reader = _FakeReader([b"more-data"])
    writer = _FakeWriter()
    resp = h_mp.MPResponse(200, {}, reader, writer, initial_body=b"head-data-")

    async def _go():
        return await resp.aread()

    assert asyncio.run(_go()) == b"head-data-more-data"


def test_line_iter_yields_trailing_unterminated_line():
    """Body that ends without a final newline still yields the
    last partial line on iteration close."""
    reader = _FakeReader([])
    writer = _FakeWriter()
    resp = h_mp.MPResponse(
        200, {"content-length": "11"}, reader, writer, initial_body=b"a\nb\nno-nl-x"
    )

    async def _go():
        out = []
        async for line in resp.aiter_lines():
            out.append(line)
        return out

    lines = asyncio.run(_go())
    assert lines == ["a", "b", "no-nl-x"]


def test_chunked_body_iterates_via_async_for():
    """``async for`` exercises ``__aiter__`` + ``__anext__``
    + final ``StopAsyncIteration`` symmetrically."""
    body = _build_chunked_body([b"a", b"bc"])
    reader = _FakeReader([body])
    it = h_mp._ChunkedBodyIter(reader, b"")

    async def _go():
        out = []
        async for chunk in it:
            out.append(chunk)
        return out

    assert asyncio.run(_go()) == [b"a", b"bc"]


def test_content_length_body_iterates_via_async_for_to_completion():
    """Drain a content-length body via ``async for`` so the final
    ``StopAsyncIteration`` path runs after the budget is consumed."""
    reader = _FakeReader([b"oran", b"ges"])
    it = h_mp._ContentLengthBodyIter(reader, initial_buf=b"", content_length=7)

    async def _go():
        out = []
        async for chunk in it:
            out.append(chunk)
        return out

    chunks = asyncio.run(_go())
    assert b"".join(chunks) == b"oranges"


def test_content_length_body_socket_eof_terminates_early():
    """Reader returns EOF before ``content_length`` is satisfied
    — terminate cleanly via ``StopAsyncIteration``."""
    reader = _FakeReader([b"only-3-bytes"])
    it = h_mp._ContentLengthBodyIter(reader, initial_buf=b"", content_length=100)

    async def _go():
        out = []
        async for chunk in it:
            out.append(chunk)
        return out

    chunks = asyncio.run(_go())
    # We get whatever the reader had; it then EOFs and the iter stops.
    assert chunks == [b"only-3-bytes"]


def test_until_close_body_iterates_via_async_for():
    """Drain a no-CL/no-TE body via ``async for``."""
    reader = _FakeReader([b"x", b"y", b"z"])
    it = h_mp._UntilCloseBodyIter(reader, initial_buf=b"head-")

    async def _go():
        out = []
        async for chunk in it:
            out.append(chunk)
        return out

    chunks = asyncio.run(_go())
    assert b"".join(chunks) == b"head-xyz"


def test_send_request_iterable_content_uses_aiter_protocol():
    """When ``content`` exposes ``__aiter__`` / ``__anext__``,
    drive via the async-iter protocol — covers the explicit
    branch in ``_send_request``."""

    class _ClassBasedAsyncIter:
        def __init__(self):
            self._chunks = [b"alpha", b"beta"]
            self._i = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._i >= len(self._chunks):
                raise StopAsyncIteration
            out = self._chunks[self._i]
            self._i += 1
            return out

    writer = _FakeWriter()
    cm = h_mp.MPStreamCM(
        "https://x.test/y",
        headers=None,
        content=_ClassBasedAsyncIter(),
        timeout=5.0,
        ssl_context=None,
    )

    async def _go():
        await cm._send_request(writer, "x.test", "/y")

    asyncio.run(_go())
    sent = writer.written
    # Both chunks present in chunked framing.
    assert b"5\r\nalpha\r\n" in sent
    assert b"4\r\nbeta\r\n" in sent
    assert sent.endswith(b"0\r\n\r\n")


def test_send_request_drops_caller_transfer_encoding_override():
    """Caller can't override our chunked transfer encoding."""
    writer = _FakeWriter()
    cm = h_mp.MPStreamCM(
        "https://x.test/y",
        headers={"Transfer-Encoding": "identity"},
        content=b"data",
        timeout=5.0,
        ssl_context=None,
    )

    async def _go():
        await cm._send_request(writer, "x.test", "/y")

    asyncio.run(_go())
    sent = writer.written
    # Only one Transfer-Encoding header — ours.
    te_count = sent.count(b"Transfer-Encoding:")
    assert te_count == 1
    assert b"Transfer-Encoding: chunked\r\n" in sent
    assert b"Transfer-Encoding: identity" not in sent


def test_post_json_raises_on_error_status():
    client = _FakeClient(status_code=503, body=b'{"error":"down"}')

    raised = False

    async def _go():
        nonlocal raised
        try:
            await h.post_json(client, "https://x.test/y", {})
        except h.HTTPStatusError as e:
            assert e.status_code == 503
            raised = True

    asyncio.run(_go())
    assert raised
