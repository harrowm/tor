"""
HTTP Tracker communication (BEP 3, BEP 23).

Sends a compact announce request and parses the peer list from the response.

Key facts about the wire format:
  - info_hash and peer_id are 20-byte binary values; they must be
    percent-encoded byte-by-byte in the query string (not base64).
  - Trackers that support BEP 23 return peers in compact format:
    a byte string of 6-byte chunks, each being 4 bytes of IPv4 address
    followed by 2 bytes of port in network (big-endian) byte order.
  - We always request compact=1; non-compact fallback is also handled.
"""

from __future__ import annotations

import os
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
    """Send an HTTP tracker announce and return the parsed response.

    Args:
        announce_url:  The tracker URL from the .torrent file.
        info_hash:     20-byte SHA-1 of the bencoded info dict.
        peer_id:       20-byte peer identity (use generate_peer_id()).
        port:          Port we are listening on (use 6881 if not yet listening).
        uploaded:      Bytes uploaded so far.
        downloaded:    Bytes downloaded so far.
        left:          Bytes remaining to download.
        event:         "started", "stopped", "completed", or "" for regular.
        timeout:       HTTP request timeout in seconds.

    Raises:
        TrackerError on network failure or tracker-reported error.
    """
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


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------

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
    """Build the full tracker announce URL with a percent-encoded query string.

    Binary fields (info_hash, peer_id) must be percent-encoded byte-by-byte.
    urllib.parse.urlencode encodes bytes as their repr, not as raw bytes, so
    we build the query string manually.
    """
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


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(body: bytes) -> TrackerResponse:
    """Parse a raw bencoded tracker response body."""
    try:
        resp = decode(body)
    except DecodeError as exc:
        raise TrackerError(f"Cannot decode tracker response: {exc}") from exc

    if not isinstance(resp, dict):
        raise TrackerError("Tracker response is not a bencoded dict")

    # Tracker-reported failure
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
        raise TrackerError(
            f"Unexpected peers type: {type(peers_raw).__name__}"
        )

    return TrackerResponse(
        interval=interval,
        peers=peers,
        min_interval=min_interval,
        complete=complete if isinstance(complete, int) else 0,
        incomplete=incomplete if isinstance(incomplete, int) else 0,
    )


def _parse_compact_peers(data: bytes) -> list[tuple[str, int]]:
    """Parse BEP 23 compact peer list (6 bytes per peer).

    Each peer is 4 bytes of IPv4 address + 2 bytes port, big-endian.
    """
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
