"""SOCKS5/TLS connections with standard-library HTTP framing and pooling."""
from __future__ import annotations

import http.client
import socket
import ssl
import threading
from collections.abc import Mapping
from urllib.parse import ParseResult, urlparse


class _Socks5Connection(http.client.HTTPConnection):
    """Use the standard HTTP client over a SOCKS5-negotiated TLS socket."""

    def __init__(self, proxy: ParseResult, host: str, port: int, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout)
        self.proxy = proxy

    def connect(self) -> None:
        self.sock = _open_socks5_tls(self.proxy, self.host, self.port, self.timeout)

# Pooled TLS connections keyed by (proxy URL, target host, target port).
_CONNECTIONS: dict[tuple[str, str, int], _Socks5Connection] = {}
_CONNECTIONS_LOCK = threading.Lock()


def _close_quietly(connection: socket.socket | http.client.HTTPConnection | None) -> None:
    if connection is None:
        return
    try:
        connection.close()
    except OSError:
        pass


def _take_pooled_connection(key: tuple[str, str, int]) -> _Socks5Connection | None:
    with _CONNECTIONS_LOCK:
        return _CONNECTIONS.pop(key, None)


def _store_pooled_connection(
    key: tuple[str, str, int], connection: _Socks5Connection
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


def request_via_socks5(
    method: str,
    url: str,
    headers: Mapping[str, str],
    proxy_url: str,
    timeout_seconds: float,
    body: bytes | None = None,
) -> tuple[int, bytes]:
    """Return HTTP status and bytes; the caller owns API and JSON semantics."""

    endpoint = urlparse(url)
    proxy = urlparse(proxy_url)
    if proxy.scheme not in {"socks5", "socks5h"} or not proxy.hostname:
        raise RuntimeError("QWEN_SOCKS5_PROXY must use socks5:// or socks5h://")
    host = endpoint.hostname
    if endpoint.scheme != "https" or not host:
        raise RuntimeError("SOCKS5 model transport currently requires an HTTPS endpoint")
    port = endpoint.port or 443
    path = endpoint.path or "/"
    if endpoint.query:
        path += "?" + endpoint.query
    key = (proxy_url.strip().lower(), host.lower(), int(port))
    connection = _take_pooled_connection(key)
    reused = connection is not None
    if connection is None:
        connection = _Socks5Connection(proxy, host, port, timeout_seconds)
    try:
        while True:
            try:
                connection.timeout = timeout_seconds
                if connection.sock is not None:
                    connection.sock.settimeout(timeout_seconds)
                connection.request(method, path, body=body, headers=headers)
                response = connection.getresponse()
                with response:
                    raw = response.read()
                    status, reusable = response.status, not response.will_close
            except (ConnectionError, http.client.HTTPException):
                # A pooled connection may have been closed by the proxy or the
                # server since the previous call.  Rebuild it once, then let a
                # second failure surface to the caller unchanged.
                _close_quietly(connection)
                if not reused:
                    raise
                reused = False
                connection = _Socks5Connection(proxy, host, port, timeout_seconds)
                continue
            break
    except BaseException:
        _close_quietly(connection)
        raise
    if reusable:
        _store_pooled_connection(key, connection)
    else:
        _close_quietly(connection)
    return status, raw


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
