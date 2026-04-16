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
        # BEP 10 extension protocol attributes (default: no extension support)
        self.remote_supports_extensions = False
        self._pending: list = []

    async def download_piece(self, piece_index, piece_size, expected_hash, **kwargs):
        return self._pieces[piece_index]

    async def do_extension_handshake(self, extensions, *, timeout=15.0):
        pass

    def peer_ext_id(self, name):
        return None

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

            async def download_piece(self, idx, size, h, **kwargs):
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


# ---------------------------------------------------------------------------
# Parallel download across multiple peers
# ---------------------------------------------------------------------------

class TestParallelDownload:
    async def test_two_peers_each_with_half_the_pieces(self, tmp_path):
        """Peer A has pieces 0-1, Peer B has pieces 2-3; combined = complete."""
        pieces  = make_pieces(4)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        # \xc0 = 11000000 → pieces 0,1 ; \x30 = 00110000 → pieces 2,3
        peer_a = FakePeer("1.2.3.4", 6881, pieces, bitfield=b"\xc0")
        peer_b = FakePeer("5.6.7.8", 6882, pieces, bitfield=b"\x30")

        async def open_side_effect(host, port, *a, **kw):
            return peer_a if host == "1.2.3.4" else peer_b

        with patch(
            "bittorrent.peer_manager.PeerConnection.open",
            new=AsyncMock(side_effect=open_side_effect),
        ):
            await manager.run([("1.2.3.4", 6881), ("5.6.7.8", 6882)])

        assert pm.is_complete()
        assert pm.num_complete == 4

    async def test_six_pieces_split_across_three_peers(self, tmp_path):
        """Each of three peers has exactly two pieces; all six must be downloaded."""
        pieces  = make_pieces(6)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        # \xc0 = pieces 0-1 ; \x30 = pieces 2-3 ; \x0c = pieces 4-5
        peer_map = {
            "1.1.1.1": FakePeer("1.1.1.1", 1, pieces, bitfield=b"\xc0"),
            "2.2.2.2": FakePeer("2.2.2.2", 2, pieces, bitfield=b"\x30"),
            "3.3.3.3": FakePeer("3.3.3.3", 3, pieces, bitfield=b"\x0c"),
        }

        async def open_side_effect(host, port, *a, **kw):
            return peer_map[host]

        with patch(
            "bittorrent.peer_manager.PeerConnection.open",
            new=AsyncMock(side_effect=open_side_effect),
        ):
            await manager.run([("1.1.1.1", 1), ("2.2.2.2", 2), ("3.3.3.3", 3)])

        assert pm.is_complete()

    async def test_correct_data_written_from_two_peers(self, tmp_path):
        """Verify that pieces written to storage come from the correct peers."""
        pieces  = make_pieces(4)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        peer_a = FakePeer("1.2.3.4", 6881, pieces, bitfield=b"\xc0")  # 0,1
        peer_b = FakePeer("5.6.7.8", 6882, pieces, bitfield=b"\x30")  # 2,3

        async def open_side_effect(host, port, *a, **kw):
            return peer_a if host == "1.2.3.4" else peer_b

        with patch(
            "bittorrent.peer_manager.PeerConnection.open",
            new=AsyncMock(side_effect=open_side_effect),
        ):
            await manager.run([("1.2.3.4", 6881), ("5.6.7.8", 6882)])

        for i, expected in enumerate(pieces):
            assert storage.read_piece(i) == expected

    async def test_no_piece_downloaded_twice(self, tmp_path):
        """Under concurrent peers, no piece should be downloaded more than once."""
        pieces  = make_pieces(4)
        torrent = make_torrent(pieces)
        # Disable end-game (threshold=0) so each piece is only given to one worker.
        # End-game intentionally allows in-progress pieces to be sent to multiple
        # workers; this test is checking normal (non-end-game) behaviour.
        pm      = PieceManager(torrent.num_pieces, torrent.piece_length,
                               torrent.total_length, end_game_threshold=0)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        download_counts: dict[int, int] = {i: 0 for i in range(4)}

        class CountingPeer(FakePeer):
            async def download_piece(self, idx, size, h, **kwargs):
                download_counts[idx] += 1
                return self._pieces[idx]

        # Both peers have all 4 pieces (\xf0 = 11110000)
        peer_a = CountingPeer("1.2.3.4", 6881, pieces, bitfield=b"\xf0")
        peer_b = CountingPeer("5.6.7.8", 6882, pieces, bitfield=b"\xf0")

        call_n = 0
        async def open_side_effect(host, port, *a, **kw):
            nonlocal call_n
            call_n += 1
            return peer_a if call_n == 1 else peer_b

        with patch(
            "bittorrent.peer_manager.PeerConnection.open",
            new=AsyncMock(side_effect=open_side_effect),
        ):
            await manager.run([("1.2.3.4", 6881), ("5.6.7.8", 6882)])

        assert pm.is_complete()
        for i, count in download_counts.items():
            assert count == 1, f"piece {i} downloaded {count} times"

    async def test_stall_wait_allows_in_flight_piece_recovery(self, tmp_path):
        """Worker that finds no MISSING pieces waits for in-flight pieces to resolve.

        Scenario:
          - Peer A has piece 0 only.
          - Peer B has piece 1 only, but fails immediately.
          - Peer C has piece 1 only, succeeds.

        Without the stall-wait fix, Worker A finishes piece 0, sees piece 1 is
        IN_PROGRESS (not its bitfield), exits, grabs Peer C, but Peer C also
        sees piece 1 as IN_PROGRESS and exits — leaving no worker when Peer B
        fails.  With the fix, the worker waits until piece 1 resolves.
        """
        pieces  = make_pieces(2)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        from bittorrent.peer import PeerError

        class FailPeer:
            host     = "2.2.2.2"
            port     = 2
            bitfield = bytearray(b"\x40")  # piece 1

            async def download_piece(self, idx, size, h, **kwargs):
                await asyncio.sleep(0)   # yield so other tasks run first
                raise PeerError("peer B failed")

            async def close(self): pass

        peer_map = {
            "1.1.1.1": FakePeer("1.1.1.1", 1, pieces, bitfield=b"\x80"),  # piece 0
            "2.2.2.2": FailPeer(),
            "3.3.3.3": FakePeer("3.3.3.3", 3, pieces, bitfield=b"\x40"),  # piece 1
        }

        async def open_side_effect(host, port, *a, **kw):
            return peer_map[host]

        with patch(
            "bittorrent.peer_manager.PeerConnection.open",
            new=AsyncMock(side_effect=open_side_effect),
        ):
            await manager.run([("1.1.1.1", 1), ("2.2.2.2", 2), ("3.3.3.3", 3)])

        assert pm.is_complete()


# ---------------------------------------------------------------------------
# Disconnection / timeout handling
# ---------------------------------------------------------------------------

class TestDisconnectionHandling:
    async def test_piece_retried_after_peer_timeout(self, tmp_path):
        """A peer that raises PeerError (simulating timeout) hands the piece to the next peer."""
        pieces  = make_pieces(2)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        from bittorrent.peer import PeerError

        class TimeoutPeer:
            host     = "1.2.3.4"
            port     = 6881
            bitfield = bytearray(b"\xc0")  # pieces 0,1

            async def download_piece(self, idx, size, h, **kwargs):
                raise PeerError("simulated block timeout")

            async def close(self): pass

        good_peer = FakePeer("5.6.7.8", 6882, pieces, bitfield=b"\xc0")

        call_n = 0
        async def open_side_effect(host, port, *a, **kw):
            nonlocal call_n
            call_n += 1
            return TimeoutPeer() if call_n == 1 else good_peer

        with patch(
            "bittorrent.peer_manager.PeerConnection.open",
            new=AsyncMock(side_effect=open_side_effect),
        ):
            await manager.run([("1.2.3.4", 6881), ("5.6.7.8", 6882)])

        assert pm.is_complete()

    async def test_in_progress_piece_returned_to_missing_on_timeout(self, tmp_path):
        """After all peers are exhausted, a failed piece is MISSING not IN_PROGRESS."""
        pieces  = make_pieces(1)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        from bittorrent.peer import PeerError

        class TimeoutPeer:
            host     = "1.2.3.4"
            port     = 6881
            bitfield = bytearray(b"\x80")

            async def download_piece(self, idx, size, h, **kwargs):
                raise PeerError("timeout")

            async def close(self): pass

        with patch(
            "bittorrent.peer_manager.PeerConnection.open",
            new=AsyncMock(return_value=TimeoutPeer()),
        ):
            with pytest.raises(RuntimeError, match="incomplete"):
                await manager.run([("1.2.3.4", 6881)])

        from bittorrent.piece_manager import PieceState
        assert pm.piece_state(0) == PieceState.MISSING

    async def test_partial_peer_then_timeout_then_recovery(self, tmp_path):
        """Peer A gets piece 0, times out on pieces 1+2; Peer B finishes the rest."""
        pieces  = make_pieces(3)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        from bittorrent.peer import PeerError

        class PartialTimeoutPeer:
            host     = "1.2.3.4"
            port     = 6881
            bitfield = bytearray(b"\xe0")  # pieces 0,1,2

            async def download_piece(self, idx, size, h, **kwargs):
                if idx == 0:
                    return pieces[0]
                raise PeerError("timeout on piece %d" % idx)

            async def close(self): pass

        good_peer = FakePeer("5.6.7.8", 6882, pieces, bitfield=b"\xe0")

        call_n = 0
        async def open_side_effect(host, port, *a, **kw):
            nonlocal call_n
            call_n += 1
            return PartialTimeoutPeer() if call_n == 1 else good_peer

        with patch(
            "bittorrent.peer_manager.PeerConnection.open",
            new=AsyncMock(side_effect=open_side_effect),
        ):
            await manager.run([("1.2.3.4", 6881), ("5.6.7.8", 6882)])

        assert pm.is_complete()

    async def test_disconnected_peer_mid_download(self, tmp_path):
        """A peer that drops the connection mid-download is replaced cleanly."""
        pieces  = make_pieces(2)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        from bittorrent.peer import PeerError

        class DisconnectPeer:
            host     = "1.2.3.4"
            port     = 6881
            bitfield = bytearray(b"\xc0")

            async def download_piece(self, idx, size, h, **kwargs):
                raise PeerError("connection reset by peer")

            async def close(self): pass

        good_peer = FakePeer("5.6.7.8", 6882, pieces, bitfield=b"\xc0")

        call_n = 0
        async def open_side_effect(host, port, *a, **kw):
            nonlocal call_n
            call_n += 1
            return DisconnectPeer() if call_n == 1 else good_peer

        with patch(
            "bittorrent.peer_manager.PeerConnection.open",
            new=AsyncMock(side_effect=open_side_effect),
        ):
            await manager.run([("1.2.3.4", 6881), ("5.6.7.8", 6882)])

        assert pm.is_complete()


# ---------------------------------------------------------------------------
# End-game mode
# ---------------------------------------------------------------------------

class TestEndGame:
    def _make_eg_setup(self, tmp_path, num_pieces: int, threshold: int):
        """Helper: build torrent/pm/storage/manager with a custom eg threshold."""
        pieces  = make_pieces(num_pieces)
        torrent = make_torrent(pieces)
        pm      = PieceManager(
            torrent.num_pieces,
            torrent.piece_length,
            torrent.total_length,
            end_game_threshold=threshold,
        )
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)
        return pieces, torrent, pm, storage, manager

    async def test_end_game_completes_download(self, tmp_path):
        """End-game mode: two peers each racing on the last piece → download completes."""
        pieces, torrent, pm, storage, manager = self._make_eg_setup(
            tmp_path, num_pieces=4, threshold=2
        )

        # Both peers have all 4 pieces; threshold=2 means end-game fires
        # when 2 pieces remain.  With two concurrent workers, they should
        # race and both finish correctly.
        peer_a = FakePeer("1.2.3.4", 6881, pieces, bitfield=b"\xf0")
        peer_b = FakePeer("5.6.7.8", 6882, pieces, bitfield=b"\xf0")

        call_n = 0
        async def open_side_effect(host, port, *a, **kw):
            nonlocal call_n
            call_n += 1
            return peer_a if call_n == 1 else peer_b

        with patch(
            "bittorrent.peer_manager.PeerConnection.open",
            new=AsyncMock(side_effect=open_side_effect),
        ):
            await manager.run([("1.2.3.4", 6881), ("5.6.7.8", 6882)])

        assert pm.is_complete()

    async def test_end_game_no_double_count_in_stats(self, tmp_path):
        """When two workers race on the same piece, pieces_complete is not inflated."""
        pieces, torrent, pm, storage, manager = self._make_eg_setup(
            tmp_path, num_pieces=4, threshold=4  # threshold=4 → end-game from start*
        )
        # *We need complete > 0 for end-game; we pre-complete piece 0 manually.
        pm.mark_complete(0)
        storage.allocate()
        storage.write_piece(0, pieces[0])
        manager._stats.pieces_complete = 1
        manager._stats.bytes_downloaded = len(pieces[0])

        # Both peers have pieces 1-3 and will both try to download them in end-game.
        peer_a = FakePeer("1.2.3.4", 6881, pieces, bitfield=b"\xf0")
        peer_b = FakePeer("5.6.7.8", 6882, pieces, bitfield=b"\xf0")

        call_n = 0
        async def open_side_effect(host, port, *a, **kw):
            nonlocal call_n
            call_n += 1
            return peer_a if call_n == 1 else peer_b

        with patch(
            "bittorrent.peer_manager.PeerConnection.open",
            new=AsyncMock(side_effect=open_side_effect),
        ):
            await manager.run([("1.2.3.4", 6881), ("5.6.7.8", 6882)])

        assert pm.is_complete()
        # stats.pieces_complete must equal num_pieces exactly — no double-counting
        assert manager.stats.pieces_complete == 4

    async def test_end_game_data_correct_after_race(self, tmp_path):
        """Data written to storage is correct even when two peers race on a piece."""
        pieces, torrent, pm, storage, manager = self._make_eg_setup(
            tmp_path, num_pieces=3, threshold=2
        )

        peer_a = FakePeer("1.2.3.4", 6881, pieces, bitfield=b"\xe0")
        peer_b = FakePeer("5.6.7.8", 6882, pieces, bitfield=b"\xe0")

        call_n = 0
        async def open_side_effect(host, port, *a, **kw):
            nonlocal call_n
            call_n += 1
            return peer_a if call_n == 1 else peer_b

        with patch(
            "bittorrent.peer_manager.PeerConnection.open",
            new=AsyncMock(side_effect=open_side_effect),
        ):
            await manager.run([("1.2.3.4", 6881), ("5.6.7.8", 6882)])

        for i, expected in enumerate(pieces):
            assert storage.read_piece(i) == expected

    async def test_end_game_not_triggered_early(self, tmp_path):
        """With a high threshold, end-game fires; with threshold=0 it never fires."""
        pieces, torrent, pm, storage, manager = self._make_eg_setup(
            tmp_path, num_pieces=4, threshold=0  # disabled
        )

        # Only one peer has all pieces; should still complete via normal path
        fake = FakePeer("1.2.3.4", 6881, pieces, bitfield=b"\xf0")
        with patch_open(fake):
            await manager.run([("1.2.3.4", 6881)])

        assert pm.is_complete()


# ---------------------------------------------------------------------------
# No-bitfield peers (seeder that skips BITFIELD message)
# ---------------------------------------------------------------------------

class TestNoBitfield:
    async def test_peer_with_no_bitfield_downloads_all_pieces(self, tmp_path):
        """A peer that sends no BITFIELD (empty bytearray) should be treated
        as able to serve any piece — common for seeders."""
        pieces  = make_pieces(4)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        # bitfield=b"" simulates a peer that never sent a BITFIELD message
        fake = FakePeer("1.2.3.4", 6881, pieces, bitfield=b"")
        with patch_open(fake):
            await manager.run([("1.2.3.4", 6881)])

        assert pm.is_complete()
        for i, expected in enumerate(pieces):
            assert storage.read_piece(i) == expected

    async def test_peer_with_empty_bitfield_serves_all_pieces(self, tmp_path):
        """Explicitly confirm empty bitfield => None passed to next_piece => any piece eligible."""
        pieces  = make_pieces(2)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        fake = FakePeer("1.2.3.4", 6881, pieces, bitfield=b"")
        with patch_open(fake):
            await manager.run([("1.2.3.4", 6881)])

        assert pm.num_complete == 2


# ---------------------------------------------------------------------------
# BEP 11 — Peer Exchange (PEX)
# ---------------------------------------------------------------------------

class TestPEX:
    """Tests for BEP 11 PEX peer exchange support."""

    # --- Unit tests for decode_pex_peers / encode_pex_peers ---

    def test_decode_pex_peers_basic(self):
        from bittorrent.messages import decode_pex_peers
        import socket, struct
        raw = socket.inet_aton("1.2.3.4") + struct.pack("!H", 6881)
        assert decode_pex_peers(raw) == [("1.2.3.4", 6881)]

    def test_decode_pex_peers_multiple(self):
        from bittorrent.messages import decode_pex_peers
        import socket, struct
        raw = (
            socket.inet_aton("1.1.1.1") + struct.pack("!H", 1111) +
            socket.inet_aton("2.2.2.2") + struct.pack("!H", 2222)
        )
        peers = decode_pex_peers(raw)
        assert peers == [("1.1.1.1", 1111), ("2.2.2.2", 2222)]

    def test_decode_pex_peers_empty(self):
        from bittorrent.messages import decode_pex_peers
        assert decode_pex_peers(b"") == []

    def test_decode_pex_peers_skips_port_zero(self):
        from bittorrent.messages import decode_pex_peers
        import socket, struct
        raw = socket.inet_aton("1.2.3.4") + struct.pack("!H", 0)
        assert decode_pex_peers(raw) == []

    def test_decode_pex_peers_partial_entry_ignored(self):
        """Trailing bytes that don't form a complete 6-byte entry are ignored."""
        from bittorrent.messages import decode_pex_peers
        import socket, struct
        raw = socket.inet_aton("1.2.3.4") + struct.pack("!H", 6881) + b"\x01\x02"
        assert decode_pex_peers(raw) == [("1.2.3.4", 6881)]

    def test_encode_pex_peers_basic(self):
        from bittorrent.messages import encode_pex_peers
        import socket, struct
        result = encode_pex_peers([("1.2.3.4", 6881)])
        assert result == socket.inet_aton("1.2.3.4") + struct.pack("!H", 6881)

    def test_encode_pex_peers_empty(self):
        from bittorrent.messages import encode_pex_peers
        assert encode_pex_peers([]) == b""

    def test_encode_decode_roundtrip(self):
        from bittorrent.messages import decode_pex_peers, encode_pex_peers
        peers = [("10.0.0.1", 6881), ("192.168.1.1", 6882)]
        assert decode_pex_peers(encode_pex_peers(peers)) == peers

    # --- Integration: PEX peers are added to queue during download ---

    async def test_pex_peers_added_to_queue(self, tmp_path):
        """PEX message received during download adds new peers to the queue."""
        import socket, struct
        from bittorrent.bencode import encode as bencode
        from bittorrent.messages import MSG_EXTENDED, PEX_LOCAL_ID, PeerMessage

        pieces = make_pieces(2)
        torrent = make_torrent(pieces)
        pm = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        # Encode a PEX message with one peer
        pex_peer_compact = socket.inet_aton("9.8.7.6") + struct.pack("!H", 9999)
        pex_payload = bencode({b"added": pex_peer_compact})
        pex_msg = PeerMessage(MSG_EXTENDED, bytes([PEX_LOCAL_ID]) + pex_payload)

        class PEXPeer(FakePeer):
            """After piece 0, injects a PEX message into _pending."""
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.remote_supports_extensions = False
                self._pending = []
                self._pex_injected = False

            async def download_piece(self, idx, size, h, **kwargs):
                data = await super().download_piece(idx, size, h, **kwargs)
                if not self._pex_injected:
                    self._pending.append(pex_msg)
                    self._pex_injected = True
                return data

            async def do_extension_handshake(self, *a, **kw):
                pass

            def peer_ext_id(self, name):
                return None

        pex_peer = PEXPeer("1.2.3.4", 6881, pieces, bitfield=b"\xc0")

        with patch_open(pex_peer):
            await manager.run([("1.2.3.4", 6881)])

        # Download completes with 2 pieces
        assert pm.num_complete == 2

    async def test_extension_handshake_attempted_when_supported(self, tmp_path):
        """Extension handshake is called when peer supports extensions."""
        pieces = make_pieces(1)
        torrent = make_torrent(pieces)
        pm = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        hs_calls = []

        class ExtPeer(FakePeer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.remote_supports_extensions = True
                self._pending = []
                self._peer_ext_ids = {}

            async def do_extension_handshake(self, extensions, *, timeout=15.0):
                hs_calls.append(extensions)

            def peer_ext_id(self, name):
                return None

        ext_peer = ExtPeer("1.2.3.4", 6881, pieces, bitfield=b"\x80")

        with patch_open(ext_peer):
            await manager.run([("1.2.3.4", 6881)])

        assert len(hs_calls) == 1
        assert b"ut_pex" in hs_calls[0]

    async def test_extension_handshake_skipped_when_not_supported(self, tmp_path):
        """Extension handshake is NOT called when peer doesn't support extensions."""
        pieces = make_pieces(1)
        torrent = make_torrent(pieces)
        pm = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        hs_calls = []

        class NoExtPeer(FakePeer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.remote_supports_extensions = False
                self._pending = []

            async def do_extension_handshake(self, extensions, *, timeout=15.0):
                hs_calls.append(extensions)

            def peer_ext_id(self, name):
                return None

        peer = NoExtPeer("1.2.3.4", 6881, pieces, bitfield=b"\x80")

        with patch_open(peer):
            await manager.run([("1.2.3.4", 6881)])

        assert len(hs_calls) == 0

    async def test_pex_failure_does_not_abort_download(self, tmp_path):
        """If extension handshake raises PeerError, download continues."""
        from bittorrent.peer import PeerError

        pieces = make_pieces(1)
        torrent = make_torrent(pieces)
        pm = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        class FailExtPeer(FakePeer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.remote_supports_extensions = True
                self._pending = []
                self._peer_ext_ids = {}

            async def do_extension_handshake(self, extensions, *, timeout=15.0):
                raise PeerError("extension handshake failed")

            def peer_ext_id(self, name):
                return None

        peer = FailExtPeer("1.2.3.4", 6881, pieces, bitfield=b"\x80")

        with patch_open(peer):
            await manager.run([("1.2.3.4", 6881)])

        assert pm.is_complete()


# ---------------------------------------------------------------------------
# BEP 29 — uTP fallback
# ---------------------------------------------------------------------------

class TestUTPFallback:
    """Tests for uTP (BEP 29) as a TCP fallback."""

    async def test_utp_fallback_used_when_tcp_fails(self, tmp_path):
        """When TCP fails and use_utp=True, uTP connection is attempted."""
        from bittorrent.peer import PeerError

        pieces  = make_pieces(1)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = PeerManager(torrent, pm, storage, INFO_HASH, PEER_ID, use_utp=True)

        fake = FakePeer("1.2.3.4", 6881, pieces, bitfield=b"\x80")

        with (
            patch("bittorrent.peer_manager.PeerConnection.open",
                  new=AsyncMock(side_effect=PeerError("TCP refused"))),
            patch("bittorrent.peer_manager.PeerConnection.open_utp",
                  new=AsyncMock(return_value=fake)),
        ):
            await manager.run([("1.2.3.4", 6881)])

        assert pm.is_complete()

    async def test_utp_not_attempted_when_disabled(self, tmp_path):
        """With use_utp=False (default), uTP is never tried when TCP fails."""
        from bittorrent.peer import PeerError

        pieces  = make_pieces(1)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)  # use_utp defaults to False

        utp_calls = []

        async def fake_open_utp(*args, **kwargs):
            utp_calls.append(True)
            return FakePeer("1.2.3.4", 6881, pieces, bitfield=b"\x80")

        with (
            patch("bittorrent.peer_manager.PeerConnection.open",
                  new=AsyncMock(side_effect=PeerError("TCP refused"))),
            patch("bittorrent.peer_manager.PeerConnection.open_utp",
                  new=AsyncMock(side_effect=fake_open_utp)),
        ):
            with pytest.raises(RuntimeError, match="incomplete"):
                await manager.run([("1.2.3.4", 6881)])

        assert utp_calls == []  # uTP was never tried

    async def test_raises_when_both_tcp_and_utp_fail(self, tmp_path):
        """When both TCP and uTP fail, the download raises RuntimeError."""
        from bittorrent.peer import PeerError
        from bittorrent.utp import UTPError

        pieces  = make_pieces(1)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = PeerManager(torrent, pm, storage, INFO_HASH, PEER_ID, use_utp=True)

        with (
            patch("bittorrent.peer_manager.PeerConnection.open",
                  new=AsyncMock(side_effect=PeerError("TCP refused"))),
            patch("bittorrent.peer_manager.PeerConnection.open_utp",
                  new=AsyncMock(side_effect=UTPError("uTP timeout"))),
        ):
            with pytest.raises(RuntimeError, match="incomplete"):
                await manager.run([("1.2.3.4", 6881)])

    async def test_utp_download_completes_correctly(self, tmp_path):
        """A download via uTP produces the same result as via TCP."""
        pieces  = make_pieces(3)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = PeerManager(torrent, pm, storage, INFO_HASH, PEER_ID, use_utp=True)

        from bittorrent.peer import PeerError

        utp_peer = FakePeer("1.2.3.4", 6881, pieces, bitfield=b"\xe0")

        with (
            patch("bittorrent.peer_manager.PeerConnection.open",
                  new=AsyncMock(side_effect=PeerError("TCP refused"))),
            patch("bittorrent.peer_manager.PeerConnection.open_utp",
                  new=AsyncMock(return_value=utp_peer)),
        ):
            await manager.run([("1.2.3.4", 6881)])

        assert pm.is_complete()
        for i, expected in enumerate(pieces):
            assert storage.read_piece(i) == expected


# ---------------------------------------------------------------------------
# on_piece_complete callback
# ---------------------------------------------------------------------------

class TestOnPieceComplete:
    async def test_callback_called_for_each_new_piece(self, tmp_path):
        """on_piece_complete is invoked with the piece index after each new piece."""
        pieces = make_pieces(3)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)

        completed_indices = []

        def record(idx):
            completed_indices.append(idx)

        manager = PeerManager(
            torrent, pm, storage, INFO_HASH, PEER_ID,
            on_piece_complete=record,
        )

        peer = FakePeer("1.2.3.4", 6881, pieces, bitfield=b"\xe0")
        with patch("bittorrent.peer_manager.PeerConnection.open",
                   new=AsyncMock(return_value=peer)):
            await manager.run([("1.2.3.4", 6881)])

        assert len(completed_indices) == 3
        assert sorted(completed_indices) == [0, 1, 2]

    async def test_callback_not_called_for_already_complete_pieces(self, tmp_path):
        """on_piece_complete fires only for newly completed pieces, not duplicates."""
        pieces = make_pieces(2)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)

        # Pre-complete piece 0
        storage.allocate()
        storage.write_piece(0, pieces[0])
        pm.mark_complete(0)

        calls = []
        manager = PeerManager(
            torrent, pm, storage, INFO_HASH, PEER_ID,
            on_piece_complete=lambda idx: calls.append(idx),
        )

        peer = FakePeer("1.2.3.4", 6881, pieces, bitfield=b"\xc0")
        with patch("bittorrent.peer_manager.PeerConnection.open",
                   new=AsyncMock(return_value=peer)):
            await manager.run([("1.2.3.4", 6881)])

        # Only piece 1 should trigger the callback
        assert calls == [1]

    async def test_no_callback_is_fine(self, tmp_path):
        """PeerManager works normally when on_piece_complete is None."""
        pieces = make_pieces(1)
        torrent = make_torrent(pieces)
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = PeerManager(torrent, pm, storage, INFO_HASH, PEER_ID)

        peer = FakePeer("1.2.3.4", 6881, pieces, bitfield=b"\x80")
        with patch("bittorrent.peer_manager.PeerConnection.open",
                   new=AsyncMock(return_value=peer)):
            await manager.run([("1.2.3.4", 6881)])

        assert pm.is_complete()
