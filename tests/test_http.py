"""CPython side of the ``exoclaw.http`` test matrix.

The CPython path is a thin wrapper over ``httpx``; tests use
``httpx.MockTransport`` to drive the request/response shape and
verify ``HTTPClient`` translates exceptions to the
``exoclaw.http`` taxonomy. The MicroPython path is exercised by
``tests/micro/test_http.py`` (see the matrix runner).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest

from exoclaw.http import (
    HTTPClient,
    HTTPConnectError,
    HTTPReadTimeout,
    HTTPStatusError,
    _parse_url,
    post_json,
)
from exoclaw.http._cpython import HttpxClient


def test_parse_url_basics() -> None:
    """``_parse_url`` covers the URLs plugin authors actually write."""
    assert _parse_url("https://api.openai.com/v1/chat") == ("https", "api.openai.com", 443, "/v1/chat")
    assert _parse_url("http://localhost:8080/x") == ("http", "localhost", 8080, "/x")
    assert _parse_url("https://api.example.com") == ("https", "api.example.com", 443, "/")
    assert _parse_url("https://api.example.com:9443/v1") == ("https", "api.example.com", 9443, "/v1")


def test_parse_url_unknown_scheme_raises() -> None:
    with pytest.raises(ValueError):
        _parse_url("ftp://nope")


@pytest.mark.asyncio
async def test_http_client_factory_returns_cpython_impl() -> None:
    """On CPython, ``HTTPClient(...)`` returns the httpx-backed impl."""
    client = HTTPClient(timeout=10.0)
    try:
        assert isinstance(client, HttpxClient)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_stream_post_round_trip_via_mock_transport() -> None:
    """End-to-end: drive a fake server with ``httpx.MockTransport``,
    verify status / headers / body / lines round-trip through the
    ``StreamingResponse`` wrapper."""
    body = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        # Verify the request body came through (chunked or not — httpx
        # collapses async-iterable content into the request stream).
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
        )

    transport = httpx.MockTransport(handler)
    raw = httpx.AsyncClient(transport=transport)

    # Wrap the test client with our adapter manually since the public
    # factory doesn't accept a pre-built httpx.AsyncClient.
    client = HttpxClient.__new__(HttpxClient)
    client._client = raw
    try:
        async with client.stream_post(
            "https://example.test/v1/chat",
            headers={"Authorization": "Bearer test"},
            content=b'{"x":1}',
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "text/event-stream"
            lines = [line async for line in resp.aiter_lines()]
            assert "data: [DONE]" in lines
    finally:
        await raw.aclose()


@pytest.mark.asyncio
async def test_stream_post_translates_connect_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated connect failure")

    transport = httpx.MockTransport(handler)
    raw = httpx.AsyncClient(transport=transport)
    client = HttpxClient.__new__(HttpxClient)
    client._client = raw

    try:
        with pytest.raises(HTTPConnectError):
            async with client.stream_post("https://example.test/x", content=b""):
                pass
    finally:
        await raw.aclose()


@pytest.mark.asyncio
async def test_stream_post_translates_read_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated read timeout")

    transport = httpx.MockTransport(handler)
    raw = httpx.AsyncClient(transport=transport)
    client = HttpxClient.__new__(HttpxClient)
    client._client = raw

    try:
        with pytest.raises(HTTPReadTimeout):
            async with client.stream_post("https://example.test/x", content=b""):
                pass
    finally:
        await raw.aclose()


@pytest.mark.asyncio
async def test_raise_for_status_translates_to_exoclaw_taxonomy() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"boom")

    transport = httpx.MockTransport(handler)
    raw = httpx.AsyncClient(transport=transport)
    client = HttpxClient.__new__(HttpxClient)
    client._client = raw

    try:
        async with client.stream_post("https://example.test/x", content=b"") as resp:
            await resp.aread()
            with pytest.raises(HTTPStatusError) as exc_info:
                resp.raise_for_status()
            assert exc_info.value.status_code == 503
    finally:
        await raw.aclose()


@pytest.mark.asyncio
async def test_async_iterable_request_body_streamed() -> None:
    """``content`` accepts an ``AsyncIterable[bytes]``; the chunks
    arrive at the server reassembled. This is the streaming-prompt
    win the openai provider depends on."""
    captured: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        # ``content`` is a property that materialises the stream.
        captured["body"] = bytes(request.content)
        return httpx.Response(200, headers={"content-type": "application/json"}, content=b"{}")

    transport = httpx.MockTransport(handler)
    raw = httpx.AsyncClient(transport=transport)
    client = HttpxClient.__new__(HttpxClient)
    client._client = raw

    async def _body() -> AsyncIterator[bytes]:
        yield b'{"a":'
        yield b"1"
        yield b"}"

    try:
        async with client.stream_post("https://example.test/x", content=_body()) as resp:
            await resp.aread()
            assert resp.status_code == 200
        assert captured["body"] == b'{"a":1}'
    finally:
        await raw.aclose()


@pytest.mark.asyncio
async def test_post_json_helper_round_trips() -> None:
    """``post_json`` builds the request, parses the response,
    handles error status codes."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = bytes(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b'{"ok":true}',
        )

    transport = httpx.MockTransport(handler)
    raw = httpx.AsyncClient(transport=transport)
    client = HttpxClient.__new__(HttpxClient)
    client._client = raw

    try:
        result = await post_json(
            client, "https://example.test/x", {"hello": "world"}, headers={"X-Test": "y"}
        )
        assert result == {"ok": True}
        assert captured["body"] == json.dumps({"hello": "world"}).encode("utf-8")
        # Caller header overrides default? No — defaults win? Both
        # get sent; the caller's value rides on top of our content-
        # type default for cases the caller doesn't override.
        sent = captured["headers"]
        assert sent["content-type"] == "application/json"
        assert sent["x-test"] == "y"
    finally:
        await raw.aclose()


@pytest.mark.asyncio
async def test_post_json_raises_on_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, content=b'{"error":"unauthorized"}')

    transport = httpx.MockTransport(handler)
    raw = httpx.AsyncClient(transport=transport)
    client = HttpxClient.__new__(HttpxClient)
    client._client = raw

    try:
        with pytest.raises(HTTPStatusError) as exc_info:
            await post_json(client, "https://example.test/x", {})
        assert exc_info.value.status_code == 401
    finally:
        await raw.aclose()


