"""
Tracker communication — HTTP (BEP 3/23) and UDP (BEP 15).

HTTP tracker:
  - info_hash and peer_id are percent-encoded byte-by-byte in the query string.
  - BEP 23 compact peer format: 6 bytes per peer (4-byte IPv4 + 2-byte port).

UDP tracker (BEP 15):
  - Two-step protocol: connect (get a connection_id) then announce.
  - All integers are big-endian.
  - Connect request: 16 bytes; connect response: 16 bytes.
  - Announce request: 98 bytes; announce response: >= 20 bytes + 6 per peer.
"""

from __future__ import annotations

import asyncio
import os
import socket
import struct
import urllib.parse
from dataclasses import dataclass, field

import aiohttp

from bittorrent.bencode import decode, DecodeError


# Azureus-style peer ID: -<2-char client><4-digit version>-<12 random bytes>
_PEER_ID_PREFIX = b"-BC0001-"  # BC = BitClient


def generate_peer_id() -> bytes:
    """Return a fresh 20-byte peer ID."""
    return _PEER_ID_PREFIX + os.urandom(12)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TrackerResponse:
    """Parsed response from a tracker announce."""
    interval: int                 # seconds until next mandatory re-announce
    peers: list[tuple[str, int]]  # [(ip_str, port), ...]
    min_interval: int = 0         # minimum re-announce interval (optional)
    complete: int = 0             # number of seeders
    incomplete: int = 0           # number of leechers (peers without full copy)


class TrackerError(Exception):
    """Raised when a tracker request fails or returns a failure response."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def announce(
    announce_url: str,
    info_hash: bytes,
    peer_id: bytes,
    port: int,
    *,
    uploaded: int = 0,
    downloaded: int = 0,
    left: int = 0,
    event: str = "",
    numwant: int = 200,
    timeout: int = 15,
) -> TrackerResponse:
    """Announce to a tracker and return the parsed response.

    Dispatches to the UDP (BEP 15) or HTTP (BEP 3) implementation based on
    the URL scheme.

    Args:
        announce_url:  The tracker URL from the .torrent file.
        info_hash:     20-byte SHA-1 of the bencoded info dict.
        peer_id:       20-byte peer identity (use generate_peer_id()).
        port:          Port we are listening on (use 6881 if not yet listening).
        uploaded:      Bytes uploaded so far.
        downloaded:    Bytes downloaded so far.
        left:          Bytes remaining to download.
        event:         "started", "stopped", "completed", or "" for regular.
        timeout:       Request timeout in seconds.

    Raises:
        TrackerError on network failure or tracker-reported error.
    """
    scheme = urllib.parse.urlparse(announce_url).scheme.lower()
    kwargs = dict(
        uploaded=uploaded, downloaded=downloaded, left=left,
        event=event, numwant=numwant, timeout=timeout,
    )
    if scheme == "udp":
        return await _announce_udp(announce_url, info_hash, peer_id, port, **kwargs)
    return await _announce_http(announce_url, info_hash, peer_id, port, **kwargs)


# ---------------------------------------------------------------------------
# HTTP tracker
# ---------------------------------------------------------------------------

async def _announce_http(
    announce_url: str,
    info_hash: bytes,
    peer_id: bytes,
    port: int,
    *,
    uploaded: int = 0,
    downloaded: int = 0,
    left: int = 0,
    event: str = "",
    numwant: int = 200,
    timeout: int = 15,
) -> TrackerResponse:
    url = _build_url(announce_url, info_hash, peer_id, port,
                     uploaded=uploaded, downloaded=downloaded,
                     left=left, event=event, numwant=numwant)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    raise TrackerError(f"Tracker returned HTTP {resp.status}")
                body = await resp.read()
    except aiohttp.ClientError as exc:
        raise TrackerError(f"HTTP request failed: {exc}") from exc

    return _parse_response(body)


def _build_url(
    base_url: str,
    info_hash: bytes,
    peer_id: bytes,
    port: int,
    *,
    uploaded: int = 0,
    downloaded: int = 0,
    left: int = 0,
    event: str = "",
    numwant: int = 200,
) -> str:
    """Build the full tracker announce URL with a percent-encoded query string."""
    parts = [
        f"info_hash={_percent_encode(info_hash)}",
        f"peer_id={_percent_encode(peer_id)}",
        f"port={port}",
        f"uploaded={uploaded}",
        f"downloaded={downloaded}",
        f"left={left}",
        "compact=1",
        f"numwant={numwant}",
    ]
    if event:
        parts.append(f"event={urllib.parse.quote(event, safe='')}")

    separator = "&" if "?" in base_url else "?"
    return base_url + separator + "&".join(parts)


def _percent_encode(data: bytes) -> str:
    """Percent-encode every byte of *data* (safe='')."""
    return urllib.parse.quote(data, safe="")


def _parse_response(body: bytes) -> TrackerResponse:
    """Parse a raw bencoded tracker response body."""
    try:
        resp = decode(body)
    except DecodeError as exc:
        raise TrackerError(f"Cannot decode tracker response: {exc}") from exc

    if not isinstance(resp, dict):
        raise TrackerError("Tracker response is not a bencoded dict")

    if b"failure reason" in resp:
        reason = resp[b"failure reason"]
        if isinstance(reason, bytes):
            reason = reason.decode("utf-8", errors="replace")
        raise TrackerError(f"Tracker failure: {reason}")

    interval = resp.get(b"interval", 0)
    if not isinstance(interval, int):
        raise TrackerError("'interval' must be an integer")

    min_interval = resp.get(b"min interval", 0)
    if not isinstance(min_interval, int):
        min_interval = 0

    complete = resp.get(b"complete", 0)
    incomplete = resp.get(b"incomplete", 0)

    peers_raw = resp.get(b"peers", b"")
    if isinstance(peers_raw, bytes):
        peers = _parse_compact_peers(peers_raw)
    elif isinstance(peers_raw, list):
        peers = _parse_dict_peers(peers_raw)
    else:
        raise TrackerError(f"Unexpected peers type: {type(peers_raw).__name__}")

    # BEP 7: IPv6 compact peers (18 bytes each: 16-byte addr + 2-byte port)
    peers6_raw = resp.get(b"peers6", b"")
    if isinstance(peers6_raw, bytes) and peers6_raw:
        peers.extend(_parse_compact_peers6(peers6_raw))

    return TrackerResponse(
        interval=interval,
        peers=peers,
        min_interval=min_interval,
        complete=complete if isinstance(complete, int) else 0,
        incomplete=incomplete if isinstance(incomplete, int) else 0,
    )


def _parse_compact_peers(data: bytes) -> list[tuple[str, int]]:
    """Parse BEP 23 compact peer list (6 bytes per peer)."""
    if len(data) % 6 != 0:
        raise TrackerError(
            f"Compact peers data length {len(data)} is not a multiple of 6"
        )
    peers: list[tuple[str, int]] = []
    for i in range(0, len(data), 6):
        ip = ".".join(str(b) for b in data[i : i + 4])
        (port,) = struct.unpack("!H", data[i + 4 : i + 6])
        peers.append((ip, port))
    return peers


def _parse_compact_peers6(data: bytes) -> list[tuple[str, int]]:
    """Parse BEP 7 compact IPv6 peer list (18 bytes per peer)."""
    if len(data) % 18 != 0:
        raise TrackerError(
            f"Compact peers6 data length {len(data)} is not a multiple of 18"
        )
    peers: list[tuple[str, int]] = []
    for i in range(0, len(data), 18):
        ip = socket.inet_ntop(socket.AF_INET6, data[i : i + 16])
        (port,) = struct.unpack("!H", data[i + 16 : i + 18])
        peers.append((ip, port))
    return peers


def _parse_dict_peers(entries: list) -> list[tuple[str, int]]:
    """Parse non-compact (list-of-dicts) peer format."""
    peers: list[tuple[str, int]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise TrackerError("Each peer entry must be a dict")
        ip_raw = entry.get(b"ip", b"")
        port = entry.get(b"port", 0)
        if not isinstance(ip_raw, bytes):
            raise TrackerError("Peer 'ip' must be a byte string")
        if not isinstance(port, int):
            raise TrackerError("Peer 'port' must be an integer")
        peers.append((ip_raw.decode("utf-8"), port))
    return peers


# ---------------------------------------------------------------------------
# UDP tracker (BEP 15)
# ---------------------------------------------------------------------------

_UDP_MAGIC    = 0x41727101980  # magic connection_id for initial connect
_UDP_CONNECT  = 0
_UDP_ANNOUNCE = 1
_UDP_ERROR    = 3


def _encode_connect_request(transaction_id: int) -> bytes:
    """Build a 16-byte UDP connect request packet."""
    return struct.pack("!QII", _UDP_MAGIC, _UDP_CONNECT, transaction_id)


def _decode_connect_response(data: bytes, transaction_id: int) -> int:
    """Parse a UDP connect response and return the connection_id.

    Raises TrackerError on bad length, wrong action, or txid mismatch.
    """
    if len(data) < 16:
        raise TrackerError(
            f"UDP connect response too short: {len(data)} bytes (need 16)"
        )
    action, txid, connection_id = struct.unpack("!IIQ", data[:16])
    if action != _UDP_CONNECT:
        raise TrackerError(
            f"UDP connect: expected action 0, got {action}"
        )
    if txid != transaction_id:
        raise TrackerError("UDP connect transaction ID mismatch")
    return connection_id


def _encode_announce_request(
    connection_id: int,
    transaction_id: int,
    info_hash: bytes,
    peer_id: bytes,
    *,
    downloaded: int = 0,
    left: int = 0,
    uploaded: int = 0,
    event: str = "",
    numwant: int = 200,
    port: int = 6881,
    key: int = 0,
) -> bytes:
    """Build a 98-byte UDP announce request packet.

    Event encoding (BEP 15): 0=none, 1=completed, 2=started, 3=stopped.
    """
    event_id = {"started": 2, "stopped": 3, "completed": 1}.get(event, 0)
    return struct.pack(
        "!QII20s20sQQQIIIIH",
        connection_id,
        _UDP_ANNOUNCE,
        transaction_id,
        info_hash,
        peer_id,
        downloaded,
        left,
        uploaded,
        event_id,
        0,        # ip_address: 0 = let tracker detect our IP
        key,
        numwant,
        port,
    )


def _decode_announce_response(data: bytes, transaction_id: int) -> TrackerResponse:
    """Parse a UDP announce response into a TrackerResponse.

    Raises TrackerError on bad length, error action, wrong action, or txid mismatch.
    """
    if len(data) < 20:
        raise TrackerError(
            f"UDP announce response too short: {len(data)} bytes (need 20)"
        )
    action, txid, interval, leechers, seeders = struct.unpack("!IIIII", data[:20])
    if action == _UDP_ERROR:
        msg = data[8:].decode("utf-8", errors="replace")
        raise TrackerError(f"UDP tracker error: {msg}")
    if action != _UDP_ANNOUNCE:
        raise TrackerError(
            f"UDP announce: expected action 1, got {action}"
        )
    if txid != transaction_id:
        raise TrackerError("UDP announce transaction ID mismatch")
    peers = _parse_compact_peers(data[20:])
    return TrackerResponse(
        interval=interval,
        peers=peers,
        complete=seeders,
        incomplete=leechers,
    )


async def _udp_transact(
    host: str,
    port: int,
    request: bytes,
    *,
    timeout: float = 5.0,
) -> bytes:
    """Send *request* to host:port via UDP and return the first response.

    Resolves the hostname, sends the packet, and waits for a reply.
    Raises TrackerError on DNS failure, socket error, or timeout.
    """
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_DGRAM)
    except OSError as exc:
        raise TrackerError(f"Cannot resolve UDP tracker {host!r}: {exc}") from exc
    if not infos:
        raise TrackerError(f"Cannot resolve UDP tracker {host!r}")

    addr = infos[0][4][:2]   # (ip_string, port)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    try:
        await asyncio.wait_for(loop.sock_sendto(sock, request, addr), timeout=timeout)
        data, _ = await asyncio.wait_for(
            loop.sock_recvfrom(sock, 65536), timeout=timeout
        )
        return data
    except asyncio.TimeoutError:
        raise TrackerError(
            f"UDP tracker {host}:{port} timed out after {timeout:.0f}s"
        )
    except OSError as exc:
        raise TrackerError(f"UDP socket error: {exc}") from exc
    finally:
        sock.close()


async def _announce_udp(
    url: str,
    info_hash: bytes,
    peer_id: bytes,
    port: int,
    *,
    uploaded: int = 0,
    downloaded: int = 0,
    left: int = 0,
    event: str = "",
    numwant: int = 200,
    timeout: int = 15,
) -> TrackerResponse:
    """Announce to a UDP tracker (BEP 15).

    Two-step: connect (obtain a connection_id) then announce.
    """
    parsed = urllib.parse.urlparse(url)
    host         = parsed.hostname
    tracker_port = parsed.port

    if not host or tracker_port is None:
        raise TrackerError(f"Invalid UDP tracker URL: {url!r}")

    # Step 1: connect — obtain a connection_id valid for ~1 minute
    tx_connect  = int.from_bytes(os.urandom(4), "big")
    connect_req = _encode_connect_request(tx_connect)
    connect_resp = await _udp_transact(host, tracker_port, connect_req, timeout=timeout)
    connection_id = _decode_connect_response(connect_resp, tx_connect)

    # Step 2: announce
    tx_announce  = int.from_bytes(os.urandom(4), "big")
    key          = int.from_bytes(os.urandom(4), "big")
    announce_req = _encode_announce_request(
        connection_id, tx_announce, info_hash, peer_id,
        downloaded=downloaded, left=left, uploaded=uploaded,
        event=event, numwant=numwant, port=port, key=key,
    )
    announce_resp = await _udp_transact(
        host, tracker_port, announce_req, timeout=timeout
    )
    return _decode_announce_response(announce_resp, tx_announce)
