"""
Tests for bittorrent.peer_manager — download orchestration.

We mock PeerConnection.open() to return a fake connection whose
download_piece() returns pre-canned data, avoiding real network I/O.
The Storage is real but uses tmp_path for isolation.
"""

import asyncio
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bittorrent.peer_manager import DownloadStats, PeerManager
from bittorrent.piece_manager import PieceManager
from bittorrent.storage import Storage
from bittorrent.torrent import Torrent


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

PIECE_LENGTH = 512
INFO_HASH    = b"\x00" * 20
PEER_ID      = b"-BC0001-" + b"X" * 12
PEERS        = [("1.2.3.4", 6881), ("5.6.7.8", 6882)]


def make_pieces(n: int, piece_length: int = PIECE_LENGTH) -> list[bytes]:
    return [bytes([i % 256] * piece_length) for i in range(n)]


def make_torrent(pieces: list[bytes], piece_length: int = PIECE_LENGTH) -> Torrent:
    total = sum(len(p) for p in pieces)
    return Torrent(
        announce="http://tracker.example.com/announce",
        info_hash=INFO_HASH,
        info_hash_hex=INFO_HASH.hex(),
        name="test.bin",
        piece_length=piece_length,
        piece_hashes=[hashlib.sha1(p).digest() for p in pieces],
        length=total,
    )


def make_storage(torrent: Torrent, tmp_path: Path) -> Storage:
    return Storage(torrent, tmp_path)


def make_pm(torrent: Torrent) -> PieceManager:
    return PieceManager(
        torrent.num_pieces,
        torrent.piece_length,
        torrent.total_length,
    )


def make_manager(torrent, pm, storage) -> PeerManager:
    return PeerManager(torrent, pm, storage, INFO_HASH, PEER_ID)


class FakePeer:
    """Stands in for PeerConnection. Returns pre-canned piece data."""

    def __init__(self, host, port, pieces: list[bytes], bitfield: bytes = b""):
        self.host    = host
        self.port    = port
        self.bitfield = bytearray(bitfield)
        self._pieces  = pieces

    async def download_piece(self, piece_index, piece_size, expected_hash):
        return self._pieces[piece_index]

    async def close(self):
        pass


def patch_open(fake_peer: FakePeer):
    """Context manager: patch PeerConnection.open to return *fake_peer*."""
    return patch(
        "bittorrent.peer_manager.PeerConnection.open",
        new=AsyncMock(return_value=fake_peer),
    )


def patch_open_error(exc: Exception):
    """Patch PeerConnection.open to raise *exc*."""
    return patch(
        "bittorrent.peer_manager.PeerConnection.open",
        new=AsyncMock(side_effect=exc),
    )


# ---------------------------------------------------------------------------
# DownloadStats
# ---------------------------------------------------------------------------

class TestDownloadStats:
    def test_percent_zero_when_nothing_done(self):
        s = DownloadStats(pieces_total=10)
        assert s.percent == 0.0

    def test_percent_full(self):
        s = DownloadStats(pieces_complete=10, pieces_total=10)
        assert s.percent == 100.0

    def test_percent_half(self):
        s = DownloadStats(pieces_complete=5, pieces_total=10)
        assert s.percent == 50.0

    def test_percent_zero_total(self):
        s = DownloadStats(pieces_total=0)
        assert s.percent == 0.0


# ---------------------------------------------------------------------------
# run() — complete single-peer download
# ---------------------------------------------------------------------------

class TestRun:
    async def test_single_piece_downloaded(self, tmp_path):
        pieces  = make_pieces(1)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        fake = FakePeer("1.2.3.4", 6881, pieces, bitfield=b"\x80")
        with patch_open(fake):
            await manager.run([("1.2.3.4", 6881)])

        assert pm.is_complete()

    async def test_multiple_pieces_all_downloaded(self, tmp_path):
        pieces  = make_pieces(4)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        # bitfield: all 4 pieces → \xf0 (pieces 0-3 in high 4 bits)
        fake = FakePeer("1.2.3.4", 6881, pieces, bitfield=b"\xf0")
        with patch_open(fake):
            await manager.run([("1.2.3.4", 6881)])

        assert pm.is_complete()
        assert pm.num_complete == 4

    async def test_data_written_to_storage(self, tmp_path):
        pieces  = make_pieces(2)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        fake = FakePeer("1.2.3.4", 6881, pieces, bitfield=b"\xc0")
        with patch_open(fake):
            await manager.run([("1.2.3.4", 6881)])

        assert storage.read_piece(0) == pieces[0]
        assert storage.read_piece(1) == pieces[1]

    async def test_stats_updated_after_download(self, tmp_path):
        pieces  = make_pieces(3)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        fake = FakePeer("1.2.3.4", 6881, pieces, bitfield=b"\xe0")
        with patch_open(fake):
            await manager.run([("1.2.3.4", 6881)])

        assert manager.stats.pieces_complete == 3
        assert manager.stats.bytes_downloaded == 3 * PIECE_LENGTH

    async def test_on_progress_called_per_piece(self, tmp_path):
        pieces  = make_pieces(3)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        calls = []
        async def on_progress(stats):
            calls.append(stats.pieces_complete)

        fake = FakePeer("1.2.3.4", 6881, pieces, bitfield=b"\xe0")
        with patch_open(fake):
            await manager.run([("1.2.3.4", 6881)], on_progress=on_progress)

        assert calls == [1, 2, 3]

    async def test_raises_when_peers_exhausted_incomplete(self, tmp_path):
        pieces  = make_pieces(4)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        # Peer has only pieces 0 and 1 (bitfield \xc0)
        fake = FakePeer("1.2.3.4", 6881, pieces, bitfield=b"\xc0")
        with patch_open(fake), pytest.raises(RuntimeError, match="incomplete"):
            await manager.run([("1.2.3.4", 6881)])


# ---------------------------------------------------------------------------
# Peer connection failure handling
# ---------------------------------------------------------------------------

class TestConnectionFailure:
    async def test_connect_error_skipped(self, tmp_path):
        """A peer that fails to connect should be skipped, not crash the run."""
        pieces  = make_pieces(1)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        from bittorrent.peer import PeerError

        good_peer = FakePeer("5.6.7.8", 6882, pieces, bitfield=b"\x80")

        async def open_side_effect(host, port, *args, **kwargs):
            if port == 6881:
                raise PeerError("connection refused")
            return good_peer

        with patch(
            "bittorrent.peer_manager.PeerConnection.open",
            new=AsyncMock(side_effect=open_side_effect),
        ):
            await manager.run([("1.2.3.4", 6881), ("5.6.7.8", 6882)])

        assert pm.is_complete()

    async def test_all_peers_fail_raises(self, tmp_path):
        pieces  = make_pieces(1)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        from bittorrent.peer import PeerError
        with patch_open_error(PeerError("all fail")):
            with pytest.raises(RuntimeError, match="incomplete"):
                await manager.run([("1.2.3.4", 6881)])

    async def test_piece_returned_to_missing_on_peer_error(self, tmp_path):
        """If a peer fails mid-piece, that piece must go back to MISSING."""
        pieces  = make_pieces(2)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        from bittorrent.peer import PeerError

        class FailOnPiece0:
            host     = "1.2.3.4"
            port     = 6881
            bitfield = bytearray(b"\xc0")

            async def download_piece(self, idx, size, h):
                if idx == 0:
                    raise PeerError("piece 0 failed")
                return pieces[idx]

            async def close(self): pass

        good_peer = FakePeer("5.6.7.8", 6882, pieces, bitfield=b"\xc0")

        call_count = 0
        async def open_side_effect(host, port, *a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FailOnPiece0()
            return good_peer

        with patch(
            "bittorrent.peer_manager.PeerConnection.open",
            new=AsyncMock(side_effect=open_side_effect),
        ):
            await manager.run([("1.2.3.4", 6881), ("5.6.7.8", 6882)])

        assert pm.is_complete()
