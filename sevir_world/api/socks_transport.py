"""Small dependency-free JSON transport shared by model discovery and chat.

TLS connections are pooled per (proxy, target) pair and reused across calls.
Every Agent model call previously paid for a fresh TCP connection to the
local SOCKS5 proxy, the SOCKS5 handshake and the remote TLS handshake before
a single request byte was sent; a keep-alive connection amortises all of that
to zero after the first call.  A pooled connection that the proxy or server
closed in the meantime is rebuilt transparently once before failing.
"""
from __future__ import annotations

import http.client
import json
import socket
import ssl
import threading
from typing import Any
from urllib.parse import ParseResult, urlparse

# Pooled TLS connections keyed by (proxy URL, target host, target port).
_CONNECTIONS: dict[tuple[str, str, int], ssl.SSLSocket] = {}
_CONNECTIONS_LOCK = threading.Lock()


def _close_quietly(connection: socket.socket | ssl.SSLSocket | None) -> None:
    if connection is None:
        return
    try:
        connection.close()
    except OSError:
        pass


def _take_pooled_connection(key: tuple[str, str, int]) -> ssl.SSLSocket | None:
    with _CONNECTIONS_LOCK:
        return _CONNECTIONS.pop(key, None)


def _store_pooled_connection(
    key: tuple[str, str, int], connection: ssl.SSLSocket
) -> None:
    with _CONNECTIONS_LOCK:
        previous = _CONNECTIONS.pop(key, None)
        _CONNECTIONS[key] = connection
    _close_quietly(previous)


def _recv_exact(connection: socket.socket | ssl.SSLSocket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise RuntimeError("SOCKS5 proxy closed the connection unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def request_json_via_socks5(
    method: str,
    url: str,
    api_key: str,
    proxy_url: str,
    timeout_seconds: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send one HTTPS JSON request through an unauthenticated SOCKS5 proxy."""

    endpoint = urlparse(url)
    proxy = urlparse(proxy_url)
    if proxy.scheme not in {"socks5", "socks5h"} or not proxy.hostname:
        raise RuntimeError("QWEN_SOCKS5_PROXY must use socks5:// or socks5h://")
    host = endpoint.hostname
    if endpoint.scheme != "https" or not host:
        raise RuntimeError("SOCKS5 model transport currently requires an HTTPS endpoint")
    port = endpoint.port or 443
    key = (proxy_url.strip().lower(), host.lower(), int(port))
    connection = _take_pooled_connection(key)
    reused = connection is not None
    if connection is None:
        connection = _open_socks5_tls(proxy, host, port, timeout_seconds)
    try:
        while True:
            try:
                connection.settimeout(timeout_seconds)
                parsed, reusable = _send_http_json(
                    connection, method, endpoint, host, api_key, payload
                )
            except (ConnectionError, http.client.HTTPException):
                # A pooled connection may have been closed by the proxy or the
                # server since the previous call.  Rebuild it once, then let a
                # second failure surface to the caller unchanged.
                _close_quietly(connection)
                if not reused:
                    raise
                reused = False
                connection = _open_socks5_tls(proxy, host, port, timeout_seconds)
                continue
            break
    except BaseException:
        _close_quietly(connection)
        raise
    if reusable:
        _store_pooled_connection(key, connection)
    else:
        _close_quietly(connection)
    return parsed


def _open_socks5_tls(
    proxy: ParseResult, host: str, port: int, timeout_seconds: float
) -> ssl.SSLSocket:
    """Open one TLS connection through the SOCKS5 proxy to the target."""

    connection = socket.create_connection(
        (proxy.hostname, proxy.port or 1080), timeout=timeout_seconds
    )
    try:
        connection.sendall(b"\x05\x01\x00")
        if _recv_exact(connection, 2) != b"\x05\x00":
            raise RuntimeError("SOCKS5 proxy rejected unauthenticated negotiation")
        encoded_host = host.encode("idna")
        if len(encoded_host) > 255:
            raise RuntimeError("model API hostname is too long for SOCKS5")
        connection.sendall(
            b"\x05\x01\x00\x03"
            + bytes([len(encoded_host)])
            + encoded_host
            + int(port).to_bytes(2, "big")
        )
        reply = _recv_exact(connection, 4)
        if reply[1] != 0:
            raise RuntimeError(f"SOCKS5 proxy connect failed with code {reply[1]}")
        address_type = reply[3]
        if address_type == 1:
            _recv_exact(connection, 4)
        elif address_type == 3:
            _recv_exact(connection, _recv_exact(connection, 1)[0])
        elif address_type == 4:
            _recv_exact(connection, 16)
        else:
            raise RuntimeError("SOCKS5 proxy returned an invalid address type")
        _recv_exact(connection, 2)
        return ssl.create_default_context().wrap_socket(connection, server_hostname=host)
    except BaseException:
        _close_quietly(connection)
        raise


def _send_http_json(
    connection: ssl.SSLSocket,
    method: str,
    endpoint: ParseResult,
    host: str,
    api_key: str,
    payload: dict[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    """Run one HTTP/1.1 request; return the parsed JSON and keep-alive status."""

    body = (
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if payload is not None
        else b""
    )
    path = endpoint.path or "/"
    if endpoint.query:
        path += "?" + endpoint.query
    header_lines = [
        f"{method.upper()} {path} HTTP/1.1",
        f"Host: {host}",
        f"Authorization: Bearer {api_key}",
        "Accept: application/json",
    ]
    if body:
        header_lines.extend([
            "Content-Type: application/json",
            f"Content-Length: {len(body)}",
        ])
    # HTTP/1.1 defaults to keep-alive; the response object reports below
    # whether the server nevertheless asked to close the connection.
    header_lines.extend(["", ""])
    connection.sendall("\r\n".join(header_lines).encode("ascii") + body)
    response = http.client.HTTPResponse(connection)
    response.begin()
    raw = response.read()
    reusable = not response.will_close
    text = raw.decode("utf-8", errors="replace")
    if response.status < 200 or response.status >= 300:
        raise RuntimeError(f"API returned HTTP {response.status}: {text[:600]}")
    if not text.strip():
        raise RuntimeError("模型API返回空响应（HTTP body 长度为 0），未得到可解析的 JSON")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"模型API返回了截断或非法 JSON（body 长度 {len(raw)}，前 300 字符：{text[:300]!r}）"
        ) from error
    if not isinstance(parsed, dict):
        raise RuntimeError("模型API JSON响应不是对象")
    return parsed, reusable
