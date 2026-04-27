"""CPython implementation of ``exoclaw.http`` — wraps ``httpx``.

Loaded only on CPython by ``exoclaw.http.HTTPClient``; never
imported on MicroPython, so the httpx dependency stays optional
and the MP heap doesn't have to compile httpx-specific bytecode."""

from __future__ import annotations

from typing import TYPE_CHECKING

from exoclaw.http import (
    HTTPConnectError,
    HTTPReadTimeout,
    HTTPStatusError,
    HTTPWriteTimeout,
)

if TYPE_CHECKING:
    import ssl
    from collections.abc import AsyncIterable, AsyncIterator

    import httpx


class HttpxResponse:
    """``ResponseProto`` impl for CPython — wraps ``httpx.Response``.

    Exposes lower-cased headers, async body read, and the same line
    iterator surface the MP path provides."""

    def __init__(self, resp: "httpx.Response") -> None:
        self._resp = resp
        self._read_body: bytes | None = None

    @property
    def status_code(self) -> int:
        return int(self._resp.status_code)

    @property
    def headers(self) -> dict[str, str]:
        return {k.lower(): v for k, v in self._resp.headers.items()}

    async def aread(self) -> bytes:
        self._read_body = bytes(await self._resp.aread())
        return self._read_body

    @property
    def text(self) -> str:
        return str(self._resp.text)

    def raise_for_status(self) -> None:
        import httpx

        try:
            self._resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPStatusError(int(e.response.status_code), str(e)) from e

    def aiter_lines(self) -> "AsyncIterator[str]":
        return self._resp.aiter_lines()


class HttpxStreamCM:
    """Async context manager wrapping ``httpx.AsyncClient.stream``.

    Translates httpx's exception taxonomy to ``exoclaw.http``'s
    flatter set so callers don't have to know about httpx."""

    def __init__(
        self,
        client: "httpx.AsyncClient",
        url: str,
        headers: dict[str, str] | None,
        content: "AsyncIterable[bytes] | bytes | None",
        timeout: float | None,
    ) -> None:
        self._client = client
        self._url = url
        self._headers = headers
        self._content = content
        self._timeout = timeout
        self._cm: object = None

    async def __aenter__(self) -> HttpxResponse:
        import httpx

        cm = self._client.stream(
            "POST",
            self._url,
            headers=self._headers or {},
            content=self._content,
            timeout=self._timeout,
        )
        self._cm = cm
        try:
            resp = await cm.__aenter__()
        except httpx.ConnectError as e:
            raise HTTPConnectError(str(e)) from e
        except httpx.ReadTimeout as e:
            raise HTTPReadTimeout(str(e)) from e
        except httpx.WriteTimeout as e:
            raise HTTPWriteTimeout(str(e)) from e
        return HttpxResponse(resp)

    async def __aexit__(self, *exc: object) -> None:
        if self._cm is not None:
            await self._cm.__aexit__(*exc)  # type: ignore[attr-defined]


class HttpxClient:
    """``ClientProto`` impl for CPython — owns one ``httpx.AsyncClient``.

    ``ssl_context`` falls through to ``httpx``'s ``verify`` arg
    (``True`` for default verification, ``False`` to disable, or a
    custom ``ssl.SSLContext`` for pinned certs)."""

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        ssl_context: "ssl.SSLContext | bool | None" = None,
    ) -> None:
        import httpx

        self._client = httpx.AsyncClient(
            timeout=timeout,
            verify=ssl_context if ssl_context is not None else True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def stream_post(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        content: "AsyncIterable[bytes] | bytes | None" = None,
        timeout: float | None = None,
    ) -> HttpxStreamCM:
        return HttpxStreamCM(self._client, url, headers, content, timeout)


def from_httpx(client: "httpx.AsyncClient") -> HttpxClient:
    """Wrap an existing ``httpx.AsyncClient`` as a ``ClientProto``.

    Useful for tests that drive an ``httpx.MockTransport`` and want
    the rest of the call site (provider streaming protocol, error
    translation, etc.) to go through the standard
    ``exoclaw.http`` plumbing. Production callers should construct
    via ``HTTPClient(...)`` instead.

    CPython only — not exported from the MP package because httpx
    isn't available there."""
    instance = HttpxClient.__new__(HttpxClient)
    instance._client = client
    return instance
