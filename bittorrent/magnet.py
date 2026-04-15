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

from bittorrent.metadata import fetch_metadata
from bittorrent.peer import PeerConnection, PeerError
from bittorrent.torrent import Torrent, parse as parse_torrent
from bittorrent.tracker import TrackerError, announce as tracker_announce

log = logging.getLogger(__name__)

_UT_METADATA = b"ut_metadata"


class MagnetError(Exception):
    """Raised when a magnet URI is invalid or cannot be resolved."""


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
) -> Torrent:
    """Fetch torrent metadata from peers and return a Torrent.

    Announces to all trackers in the magnet link to gather peers, then
    tries peers in order until the metadata is fetched and SHA-1 verified.

    Args:
        magnet:       Parsed magnet link.
        peer_id:      20-byte peer identity.
        port:         Port to advertise to trackers.
        max_peers:    Maximum number of peers to try.
        peer_timeout: Seconds allowed per peer for the full metadata fetch.

    Raises:
        MagnetError: No trackers return peers, or no peer delivers valid metadata.
    """
    peers: list[tuple[str, int]] = []

    for tracker_url in magnet.trackers:
        try:
            resp = await tracker_announce(
                tracker_url, magnet.info_hash, peer_id, port,
                left=0, event="started",
            )
            peers.extend(resp.peers)
            log.info("Tracker %s: %d peers", tracker_url, len(resp.peers))
        except TrackerError as exc:
            log.warning("Tracker %s failed: %s", tracker_url, exc)

    if not peers:
        raise MagnetError("No peers found from any tracker in the magnet link")

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
