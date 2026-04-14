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
        pm      = make_pm(torrent)
        storage = make_storage(torrent, tmp_path)
        manager = make_manager(torrent, pm, storage)

        download_counts: dict[int, int] = {i: 0 for i in range(4)}

        class CountingPeer(FakePeer):
            async def download_piece(self, idx, size, h):
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

            async def download_piece(self, idx, size, h):
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

            async def download_piece(self, idx, size, h):
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

            async def download_piece(self, idx, size, h):
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

            async def download_piece(self, idx, size, h):
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

            async def download_piece(self, idx, size, h):
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
