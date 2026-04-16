"""
BEP 14 — Local Service Discovery (LSD).

Announces the client's presence to peers on the same LAN by sending
periodic multicast UDP packets to the LSD group address.  Any BitTorrent
client listening on the same subnet will receive the announcement and can
connect directly — no tracker required.

Protocol:
  - IPv4 multicast group: 239.192.152.143, port 6771
  - IPv6 multicast group: [ff15::efc0:988f], port 6771
  - Messages are HTTP-like text with BT-SEARCH headers::

      BT-SEARCH * HTTP/1.1\\r\\n
      Host: 239.192.152.143:6771\\r\\n
      Port: <our_port>\\r\\n
      Infohash: <info_hash_hex>\\r\\n
      cookie: <random_cookie>\\r\\n
      \\r\\n\\r\\n

  - We listen on the multicast group and add any discovered peers to the
    peer queue.  Our own announcements are suppressed via the cookie.

Usage::

    async with LSDService(info_hash, port=6881) as lsd:
        peers = await lsd.discover(timeout=5.0)
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import struct
import time
from typing import Callable

log = logging.getLogger(__name__)

LSD_MCAST_ADDR  = "239.192.152.143"
LSD_PORT        = 6771
ANNOUNCE_INTERVAL = 300   # seconds between periodic re-announces (BEP 14 spec)


def _make_announce(info_hash_hex: str, port: int, cookie: str) -> bytes:
    """Build a BT-SEARCH announce packet."""
    lines = [
        "BT-SEARCH * HTTP/1.1",
        f"Host: {LSD_MCAST_ADDR}:{LSD_PORT}",
        f"Port: {port}",
        f"Infohash: {info_hash_hex}",
        f"cookie: {cookie}",
        "",
        "",
    ]
    return "\r\n".join(lines).encode()


def _parse_announce(data: bytes) -> tuple[str | None, int | None, str | None]:
    """Parse a BT-SEARCH packet.

    Returns (info_hash_hex, port, cookie) or (None, None, None) on failure.
    """
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return None, None, None

    if not text.startswith("BT-SEARCH"):
        return None, None, None

    info_hash_hex = None
    port          = None
    cookie        = None

    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("infohash:"):
            info_hash_hex = line.split(":", 1)[1].strip().lower()
        elif line.lower().startswith("port:"):
            try:
                port = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.lower().startswith("cookie:"):
            cookie = line.split(":", 1)[1].strip()

    return info_hash_hex, port, cookie


class LSDService:
    """Sends and receives BEP 14 LSD multicast announcements.

    Args:
        info_hash:    20-byte torrent info hash.
        port:         Our listening port for peer connections.
        on_peer:      Optional callback ``(host, port)`` called for each discovered peer.
        announce_interval: Seconds between periodic re-announces.
    """

    def __init__(
        self,
        info_hash: bytes,
        port: int,
        *,
        on_peer: "Callable[[str, int], None] | None" = None,
        announce_interval: float = ANNOUNCE_INTERVAL,
    ) -> None:
        self._info_hash       = info_hash
        self._info_hash_hex   = info_hash.hex()
        self._port            = port
        self._on_peer         = on_peer
        self._cookie          = os.urandom(4).hex()
        self._announce_interval = announce_interval
        self._transport: asyncio.DatagramTransport | None = None
        self._discovered: list[tuple[str, int]] = []
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Open the UDP multicast socket and start the announce loop."""
        loop = asyncio.get_running_loop()

        # Create a UDP socket that can send multicast
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except AttributeError:
            pass  # not available on all platforms (e.g. Windows)

        # Bind to the LSD port so we receive multicast packets
        sock.bind(("", LSD_PORT))

        # Join the multicast group
        mreq = struct.pack(
            "4sL",
            socket.inet_aton(LSD_MCAST_ADDR),
            socket.INADDR_ANY,
        )
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except OSError as exc:
            log.warning("LSD: cannot join multicast group: %s", exc)

        sock.setblocking(False)

        protocol = _LSDProtocol(self)
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: protocol,
            sock=sock,
        )
        log.debug("LSD: listening on %s:%s", LSD_MCAST_ADDR, LSD_PORT)

        # Announce immediately, then periodically
        self._task = asyncio.create_task(self._announce_loop())

    async def stop(self) -> None:
        """Stop the announce loop and close the socket."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    async def announce_once(self) -> None:
        """Send a single LSD multicast announcement."""
        if self._transport is None:
            return
        msg = _make_announce(self._info_hash_hex, self._port, self._cookie)
        try:
            self._transport.sendto(msg, (LSD_MCAST_ADDR, LSD_PORT))
            log.debug("LSD: announced infohash %s on port %s", self._info_hash_hex, self._port)
        except OSError as exc:
            log.debug("LSD: send failed: %s", exc)

    async def discover(self, timeout: float = 5.0) -> list[tuple[str, int]]:
        """Announce and collect peers discovered within *timeout* seconds."""
        await self.announce_once()
        await asyncio.sleep(timeout)
        return list(self._discovered)

    def _handle_datagram(self, data: bytes, addr: tuple[str, int]) -> None:
        """Called by the protocol when a UDP packet arrives."""
        info_hash_hex, port, cookie = _parse_announce(data)
        if info_hash_hex is None or port is None:
            return
        if cookie == self._cookie:
            return  # ignore our own announcement
        if info_hash_hex != self._info_hash_hex:
            return  # different torrent
        peer = (addr[0], port)
        if peer not in self._discovered:
            self._discovered.append(peer)
            log.info("LSD: discovered peer %s:%s", *peer)
            if self._on_peer is not None:
                self._on_peer(*peer)

    async def _announce_loop(self) -> None:
        """Send announcements at regular intervals."""
        while True:
            await self.announce_once()
            await asyncio.sleep(self._announce_interval)

    # Context manager support
    async def __aenter__(self) -> "LSDService":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()


class _LSDProtocol(asyncio.DatagramProtocol):
    """asyncio datagram protocol that routes packets to LSDService."""

    def __init__(self, service: LSDService) -> None:
        self._service = service

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self._service._handle_datagram(data, addr)

    def error_received(self, exc: Exception) -> None:
        log.debug("LSD: socket error: %s", exc)

    def connection_lost(self, exc: Exception | None) -> None:
        pass
