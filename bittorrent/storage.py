"""
File storage — pre-allocation, random-access piece writes, SHA-1 verification.

Single-file torrents: straightforward — piece offset = piece_index * piece_length.

Multi-file torrents: a piece can span two adjacent files. We resolve the
piece's byte range against an ordered list of (file_path, start_offset,
end_offset) regions and split writes across file boundaries.

All file I/O is synchronous (using plain open/seek/write) since piece writes
are infrequent and short-lived. Switching to aiofiles is straightforward if
needed later.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from bittorrent.torrent import Torrent

log = logging.getLogger(__name__)


@dataclass
class _Region:
    """Maps a contiguous byte range in the torrent to a file path."""
    path: Path
    torrent_offset: int   # byte offset within the whole torrent where this file starts
    length: int           # file length in bytes

    @property
    def torrent_end(self) -> int:
        return self.torrent_offset + self.length


class StorageError(Exception):
    """Raised when a storage operation fails."""


class Storage:
    """Manages on-disk storage for one torrent.

    Call allocate() once before writing any pieces.
    """

    def __init__(self, torrent: Torrent, base_dir: Path | str) -> None:
        self._torrent  = torrent
        self._base_dir = Path(base_dir)
        self._regions  = _build_regions(torrent, self._base_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(self) -> None:
        """Pre-allocate all files to their final size (sparse where supported).

        Creates parent directories as needed. Files are opened and truncated
        to the correct length so random writes never go out of bounds.
        """
        for region in self._regions:
            region.path.parent.mkdir(parents=True, exist_ok=True)
            if not region.path.exists() or region.path.stat().st_size != region.length:
                with open(region.path, "ab") as fh:
                    fh.truncate(region.length)

    def write_piece(self, piece_index: int, data: bytes) -> None:
        """Write *data* for *piece_index* to the correct location(s) on disk.

        Raises StorageError if piece_index is out of range or data length
        doesn't match the expected piece size.
        """
        piece_start, piece_len = self._piece_range(piece_index)
        if len(data) != piece_len:
            raise StorageError(
                f"Piece {piece_index}: expected {piece_len} bytes, got {len(data)}"
            )
        self._write_bytes(piece_start, data)
        log.info("DISK WRITE piece %d: %d bytes at offset %d", piece_index, len(data), piece_start)

    def read_piece(self, piece_index: int) -> bytes:
        """Read back the bytes for *piece_index* from disk."""
        piece_start, piece_len = self._piece_range(piece_index)
        return self._read_bytes(piece_start, piece_len)

    def verify_piece(self, piece_index: int, data: bytes) -> bool:
        """Return True if SHA-1(data) matches the expected hash for this piece."""
        expected = self._torrent.piece_hashes[piece_index]
        actual   = hashlib.sha1(data).digest()
        return actual == expected

    def is_complete(self) -> bool:
        """Return True if every piece on disk passes its hash check."""
        for i in range(self._torrent.num_pieces):
            try:
                data = self.read_piece(i)
            except (OSError, StorageError):
                return False
            if not self.verify_piece(i, data):
                return False
        return True

    def scan_pieces(
        self,
        progress_cb: "Callable[[int, int], None] | None" = None,
    ) -> list[int]:
        """Read and hash-verify every piece on disk; return indices of good ones.

        Used at startup to resume a partial download: any piece whose SHA-1
        matches is marked complete so it won't be re-downloaded.

        Args:
            progress_cb: Optional callable(done, total) called after each piece,
                         useful for showing a progress indicator to the user.

        Returns:
            Sorted list of piece indices that are already complete on disk.
        """
        from typing import Callable  # local to avoid circular imports at module level
        good: list[int] = []
        total = self._torrent.num_pieces
        for i in range(total):
            if progress_cb:
                progress_cb(i, total)
            try:
                data = self.read_piece(i)
            except (OSError, StorageError):
                continue
            if self.verify_piece(i, data):
                good.append(i)
        return good

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _piece_range(self, piece_index: int) -> tuple[int, int]:
        """Return (torrent_byte_offset, length) for this piece."""
        n = self._torrent.num_pieces
        if piece_index < 0 or piece_index >= n:
            raise StorageError(
                f"piece_index {piece_index} out of range [0, {n})"
            )
        piece_length  = self._torrent.piece_length
        total_length  = self._torrent.total_length
        start         = piece_index * piece_length
        end           = min(start + piece_length, total_length)
        return start, end - start

    def _write_bytes(self, torrent_offset: int, data: bytes) -> None:
        """Write *data* starting at *torrent_offset*, splitting across files."""
        data_pos = 0
        for region in self._regions:
            if data_pos >= len(data):
                break
            if torrent_offset + len(data) <= region.torrent_offset:
                break
            if torrent_offset >= region.torrent_end:
                continue

            # Overlap between [torrent_offset, torrent_offset+len(data)) and region
            write_start_in_torrent = max(torrent_offset, region.torrent_offset)
            write_end_in_torrent   = min(torrent_offset + len(data), region.torrent_end)

            file_offset  = write_start_in_torrent - region.torrent_offset
            chunk_len    = write_end_in_torrent - write_start_in_torrent
            data_slice   = data[data_pos: data_pos + chunk_len]

            with open(region.path, "r+b") as fh:
                fh.seek(file_offset)
                fh.write(data_slice)

            data_pos += chunk_len

    def _read_bytes(self, torrent_offset: int, length: int) -> bytes:
        """Read *length* bytes starting at *torrent_offset*, splicing across files."""
        result  = bytearray()
        needed  = length

        for region in self._regions:
            if needed <= 0:
                break
            if torrent_offset + length <= region.torrent_offset:
                break
            if torrent_offset >= region.torrent_end:
                continue

            read_start_in_torrent = max(torrent_offset, region.torrent_offset)
            read_end_in_torrent   = min(torrent_offset + length, region.torrent_end)

            file_offset = read_start_in_torrent - region.torrent_offset
            chunk_len   = read_end_in_torrent - read_start_in_torrent

            with open(region.path, "rb") as fh:
                fh.seek(file_offset)
                chunk = fh.read(chunk_len)

            result.extend(chunk)
            needed -= chunk_len

        return bytes(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_regions(torrent: Torrent, base_dir: Path) -> list[_Region]:
    """Build ordered list of _Region objects for a torrent."""
    regions: list[_Region] = []
    offset = 0

    if torrent.is_multi_file:
        torrent_name = torrent.name
        for file_info in torrent.files:
            # Path components are joined under <torrent_name>/
            rel = Path(torrent_name).joinpath(*file_info.path)
            regions.append(_Region(
                path=base_dir / rel,
                torrent_offset=offset,
                length=file_info.length,
            ))
            offset += file_info.length
    else:
        regions.append(_Region(
            path=base_dir / torrent.name,
            torrent_offset=0,
            length=torrent.length,
        ))

    return regions
