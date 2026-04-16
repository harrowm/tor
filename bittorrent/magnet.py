"""
Magnet URI support — parsing (BEP 9) and metadata resolution (BEP 10).

parse_magnet() extracts info_hash, display name, and tracker URLs from a
magnet URI.  resolve_magnet() announces to each tracker to collect peers,
then tries peers one by one until the info dict is fetched and verified,
and returns a ready-to-use Torrent object.

Magnet URI format:
    magnet:?xt=urn:btih:<hex|base32>&dn=<name>&tr=<tracker-url>&...

info_hash encoding:
    40-char hex   e.g. a1b2c3...  (SHA-1 in hex)
    32-char base32 e.g. ABCDEF...  (SHA-1 in base32, case-insensitive)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import urllib.parse
from dataclasses import dataclass, field

from bittorrent.dht import DHTClient
from bittorrent.metadata import fetch_metadata
from bittorrent.peer import PeerConnection, PeerError
from bittorrent.torrent import Torrent, parse as parse_torrent
from bittorrent.tracker import TrackerError, announce as tracker_announce

log = logging.getLogger(__name__)

_UT_METADATA = b"ut_metadata"


class MagnetError(Exception):
    """Raised when a magnet URI is invalid or cannot be resolved."""


# Well-known public trackers used as fallback when a magnet link has none.
# Many clients (qBittorrent, Deluge, etc.) ship a similar built-in list.
FALLBACK_TRACKERS: list[str] = [
    "udp://tracker.opentrackr.org:1337/announce",
    "udp://open.tracker.cl:1337/announce",
    "udp://tracker.openbittorrent.com:6969/announce",
    "udp://opentracker.io:6969/announce",
    "udp://tracker.torrent.eu.org:451/announce",
]


@dataclass
class MagnetLink:
    """Parsed magnet URI."""
    info_hash: bytes
    name: str | None = None
    trackers: list[str] = field(default_factory=list)

    @property
    def info_hash_hex(self) -> str:
        return self.info_hash.hex()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_magnet(uri: str) -> MagnetLink:
    """Parse a magnet URI and return a MagnetLink.

    Supports 40-char hex and 32-char base32 info_hash encodings.
    Multiple ``tr=`` parameters are all collected.

    Raises MagnetError on invalid URI or unrecognised hash format.
    """
    if not uri.lower().startswith("magnet:"):
        raise MagnetError(f"Not a magnet URI: {uri!r}")

    parsed = urllib.parse.urlparse(uri)
    params = urllib.parse.parse_qs(parsed.query, keep_blank_values=False)

    info_hash: bytes | None = None
    for xt in params.get("xt", []):
        if xt.lower().startswith("urn:btih:"):
            info_hash = _decode_btih(xt[len("urn:btih:"):])
            break

    if info_hash is None:
        raise MagnetError("Magnet URI has no valid xt=urn:btih: parameter")

    name_list = params.get("dn", [])
    name = urllib.parse.unquote_plus(name_list[0]) if name_list else None

    trackers = [urllib.parse.unquote_plus(t) for t in params.get("tr", [])]

    return MagnetLink(info_hash=info_hash, name=name, trackers=trackers)


def _decode_btih(s: str) -> bytes:
    """Decode a btih hash string to 20 bytes.

    Accepts 40-char hex or 32-char base32 (case-insensitive).
    Raises MagnetError on invalid input.
    """
    if len(s) == 40:
        try:
            return bytes.fromhex(s)
        except ValueError as exc:
            raise MagnetError(f"Invalid hex info_hash {s!r}") from exc
    if len(s) == 32:
        try:
            return base64.b32decode(s.upper())
        except Exception as exc:
            raise MagnetError(f"Invalid base32 info_hash {s!r}") from exc
    raise MagnetError(
        f"info_hash must be 40 hex chars or 32 base32 chars, got {len(s)}: {s!r}"
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

async def resolve_magnet(
    magnet: MagnetLink,
    peer_id: bytes,
    port: int = 6881,
    *,
    max_peers: int = 10,
    peer_timeout: float = 30.0,
    dht_timeout: float = 30.0,
) -> Torrent:
    """Fetch torrent metadata from peers and return a Torrent.

    Announces to all trackers in the magnet link to gather peers.  If no
    tracker peers are found (or there are no trackers), falls back to a DHT
    get_peers lookup (BEP 5).  Then tries peers in order until the metadata
    is fetched and SHA-1 verified.

    Args:
        magnet:       Parsed magnet link.
        peer_id:      20-byte peer identity.
        port:         Port to advertise to trackers.
        max_peers:    Maximum number of peers to try for metadata.
        peer_timeout: Seconds allowed per peer for the full metadata fetch.
        dht_timeout:  Seconds allowed for DHT bootstrap + get_peers lookup.

    Raises:
        MagnetError: No peers found at all, or no peer delivers valid metadata.
    """
    peers: list[tuple[str, int]] = []

    # Use fallback public trackers when the magnet link carries none.
    tracker_urls = magnet.trackers or FALLBACK_TRACKERS
    if not magnet.trackers:
        log.info("Magnet has no trackers — using built-in public tracker list")

    async def _try_tracker(url: str) -> list[tuple[str, int]]:
        try:
            resp = await tracker_announce(
                url, magnet.info_hash, peer_id, port,
                left=0, event="started",
            )
            log.info("Tracker %s: %d peers", url, len(resp.peers))
            return resp.peers
        except TrackerError as exc:
            log.warning("Tracker %s failed: %s", url, exc)
            return []

    # Announce to all trackers concurrently — avoids stacking timeouts.
    results = await asyncio.gather(*[_try_tracker(u) for u in tracker_urls])
    for result in results:
        peers.extend(result)

    # Fall back to DHT when trackers yield no peers (or there are no trackers)
    if not peers:
        log.info("No tracker peers — trying DHT lookup")
        dht_peers = await _dht_get_peers(magnet.info_hash, timeout=dht_timeout)
        peers.extend(dht_peers)
        log.info("DHT returned %d peers", len(dht_peers))

    if not peers:
        raise MagnetError("No peers found from trackers or DHT")

    last_exc: Exception = MagnetError("No peers tried")
    for host, peer_port in peers[:max_peers]:
        try:
            info_bytes = await _metadata_from_peer(
                host, peer_port, magnet.info_hash, peer_id,
                timeout=peer_timeout,
            )
        except Exception as exc:
            log.debug("Peer %s:%s: %s", host, peer_port, exc)
            last_exc = exc
            continue

        announce = magnet.trackers[0] if magnet.trackers else ""
        torrent = parse_torrent(_build_torrent_bytes(info_bytes, announce))
        log.info("Metadata resolved from %s:%s", host, peer_port)
        return torrent

    raise MagnetError(
        f"Could not fetch metadata from any of "
        f"{min(len(peers), max_peers)} peers: {last_exc}"
    )


async def _dht_get_peers(
    info_hash: bytes,
    *,
    timeout: float = 30.0,
) -> list[tuple[str, int]]:
    """Bootstrap DHT and perform a get_peers lookup for *info_hash*.

    Returns a list of (ip, port) tuples.  Never raises; returns [] on failure.
    """
    try:
        async with DHTClient() as dht:
            await asyncio.wait_for(dht.bootstrap(), timeout=timeout / 2)
            remaining = timeout / 2
            return await dht.get_peers(info_hash, timeout=remaining)
    except Exception as exc:
        log.warning("DHT lookup failed: %s", exc)
        return []


async def _metadata_from_peer(
    host: str,
    port: int,
    info_hash: bytes,
    peer_id: bytes,
    *,
    timeout: float,
) -> bytes:
    """Connect to one peer and fetch the info dict via BEP 10/9.

    Raises PeerError or MagnetError on failure.
    """
    conn = await PeerConnection.open(
        host, port, info_hash, peer_id,
        timeout=min(timeout, 10.0),
        extension_protocol=True,
    )
    try:
        if not conn.remote_supports_extensions:
            raise PeerError("Peer does not support BEP 10 extension protocol")
        await conn.do_extension_handshake({_UT_METADATA: 1})
        return await asyncio.wait_for(
            fetch_metadata(conn, info_hash),
            timeout=timeout,
        )
    finally:
        await conn.close()


def _build_torrent_bytes(info_bytes: bytes, announce: str = "") -> bytes:
    """Wrap raw bencoded info bytes in a minimal .torrent dict.

    Produces a bencoded dict parseable by torrent.parse().  Keys are in
    sorted order as required by bencode: 'announce' < 'info'.

    The announce URL may be empty (when the magnet link has no trackers).
    """
    ann = announce.encode("utf-8")
    return (
        b"d"
        + b"8:announce"
        + str(len(ann)).encode()
        + b":"
        + ann
        + b"4:info"
        + info_bytes
        + b"e"
    )
