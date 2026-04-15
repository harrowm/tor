"""
BEP 19 — Web Seeds (HTTP/FTP seeding).

Web seeds allow pieces to be fetched from ordinary HTTP servers when peers
are unavailable.  Each URL in the torrent's ``url-list`` is a base URL from
which files are served.

URL construction (BEP 19):
  Single-file torrent:  <base_url>/<torrent_name>
  Multi-file torrent:   <base_url>/<torrent_name>/<path/to/file>

Pieces are fetched using HTTP Range requests (``Range: bytes=<start>-<end>``).
For pieces that span multiple files (multi-file torrents), we issue one
Range request per file and concatenate the results.

Integration: WebSeedClient.fetch_piece() returns raw bytes that have already
been SHA-1 verified; callers can write them directly to Storage.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import quote as _urlencode

if TYPE_CHECKING:
    from bittorrent.torrent import Torrent

FETCH_TIMEOUT = 30.0   # seconds per HTTP request


class WebSeedError(Exception):
    """Raised when a web seed fetch fails (HTTP error, timeout, hash mismatch)."""


@dataclass
class _FileRegion:
    """A contiguous byte range in the concatenated torrent space, backed by one file."""
    url: str             # full HTTP URL for this file
    torrent_offset: int  # byte offset within the whole torrent
    length: int          # file length in bytes

    @property
    def torrent_end(self) -> int:
        return self.torrent_offset + self.length


class WebSeedClient:
    """Fetches torrent pieces from a BEP 19 web seed URL.

    Args:
        torrent: The parsed Torrent metadata.
        base_url: The base URL from the torrent's ``url-list``.

    Example::

        async with aiohttp.ClientSession() as session:
            client = WebSeedClient(torrent, "https://example.com/files/")
            data = await client.fetch_piece(session, piece_index=0)
    """

    def __init__(self, torrent: "Torrent", base_url: str) -> None:
        self._torrent  = torrent
        self._base_url = base_url.rstrip("/")
        self._regions  = self._build_regions()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_piece(
        self,
        session: "aiohttp.ClientSession",
        piece_index: int,
        *,
        timeout: float = FETCH_TIMEOUT,
    ) -> bytes:
        """Download and verify one piece from the web seed.

        Args:
            session:     An aiohttp.ClientSession to use for HTTP requests.
            piece_index: Which piece to download (0-based).
            timeout:     Seconds to wait per HTTP request.

        Returns:
            Raw piece bytes (SHA-1 verified).

        Raises:
            WebSeedError: On HTTP error, network timeout, or hash mismatch.
        """
        torrent = self._torrent
        if piece_index < 0 or piece_index >= torrent.num_pieces:
            raise WebSeedError(
                f"piece_index {piece_index} out of range [0, {torrent.num_pieces})"
            )

        piece_start = piece_index * torrent.piece_length
        piece_len   = min(torrent.piece_length, torrent.total_length - piece_start)

        data = await self._fetch_bytes(session, piece_start, piece_len, timeout=timeout)

        # Verify
        expected = torrent.piece_hashes[piece_index]
        actual   = hashlib.sha1(data).digest()
        if actual != expected:
            raise WebSeedError(
                f"Piece {piece_index} hash mismatch from web seed "
                f"(expected {expected.hex()}, got {actual.hex()})"
            )

        return data

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_regions(self) -> list[_FileRegion]:
        """Return ordered list of _FileRegion objects for this torrent."""
        torrent = self._torrent
        base    = self._base_url
        regions: list[_FileRegion] = []
        offset = 0

        if torrent.is_multi_file:
            for file_info in torrent.files:
                # URL-encode each path component and join
                path_parts = [_urlencode(p, safe="") for p in file_info.path]
                url = (
                    base
                    + "/"
                    + _urlencode(torrent.name, safe="")
                    + "/"
                    + "/".join(path_parts)
                )
                regions.append(_FileRegion(
                    url=url,
                    torrent_offset=offset,
                    length=file_info.length,
                ))
                offset += file_info.length
        else:
            url = base + "/" + _urlencode(torrent.name, safe="")
            regions.append(_FileRegion(
                url=url,
                torrent_offset=0,
                length=torrent.length,
            ))

        return regions

    async def _fetch_bytes(
        self,
        session: "aiohttp.ClientSession",
        torrent_offset: int,
        length: int,
        *,
        timeout: float,
    ) -> bytes:
        """Fetch *length* bytes at *torrent_offset* from the appropriate URL(s).

        Handles pieces that span multiple files in multi-file torrents by
        issuing one Range request per file.
        """
        import aiohttp

        result   = bytearray()
        end_want = torrent_offset + length

        for region in self._regions:
            if end_want <= region.torrent_offset:
                break
            if torrent_offset >= region.torrent_end:
                continue

            # Overlap between [torrent_offset, end_want) and region
            fetch_start = max(torrent_offset, region.torrent_offset)
            fetch_end   = min(end_want, region.torrent_end)

            file_start  = fetch_start - region.torrent_offset
            file_end    = fetch_end   - region.torrent_offset - 1  # inclusive

            range_header = f"bytes={file_start}-{file_end}"

            try:
                async with session.get(
                    region.url,
                    headers={"Range": range_header},
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status not in (200, 206):
                        raise WebSeedError(
                            f"HTTP {resp.status} from {region.url} "
                            f"(range {range_header})"
                        )
                    chunk = await resp.read()
            except aiohttp.ClientError as exc:
                raise WebSeedError(
                    f"HTTP error fetching {region.url}: {exc}"
                ) from exc
            except TimeoutError as exc:
                raise WebSeedError(
                    f"Timeout fetching {region.url}"
                ) from exc

            result.extend(chunk)

        return bytes(result)


# ---------------------------------------------------------------------------
# Standalone helper used by PeerManager
# ---------------------------------------------------------------------------

def build_webseed_clients(torrent: "Torrent") -> list[WebSeedClient]:
    """Return one WebSeedClient per URL in torrent.url_list."""
    return [WebSeedClient(torrent, url) for url in torrent.url_list]
