"""
Tests for bittorrent.main — CLI argument parsing and wiring.

We don't test the full end-to-end download (that requires a real network);
instead we verify:
  - Argument parsing
  - Error paths (bad torrent file, tracker failure)
  - The happy path is wired up correctly (mocked tracker + peer manager)
"""

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bittorrent.main import _parse_args, _run
from bittorrent.bencode import encode
from bittorrent.tracker import TrackerResponse


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_positional_source(self):
        args = _parse_args(["foo.torrent"])
        assert args.source == "foo.torrent"

    def test_default_output_dir(self):
        args = _parse_args(["foo.torrent"])
        assert args.output_dir == "."

    def test_custom_output_dir(self):
        args = _parse_args(["foo.torrent", "--output-dir", "/tmp/dl"])
        assert args.output_dir == "/tmp/dl"

    def test_short_output_dir(self):
        args = _parse_args(["foo.torrent", "-o", "/tmp/dl"])
        assert args.output_dir == "/tmp/dl"

    def test_default_port(self):
        args = _parse_args(["foo.torrent"])
        assert args.port == 6881

    def test_custom_port(self):
        args = _parse_args(["foo.torrent", "--port", "6889"])
        assert args.port == 6889

    def test_verbose_flag(self):
        args = _parse_args(["foo.torrent", "--verbose"])
        assert args.verbose is True

    def test_verbose_default_false(self):
        args = _parse_args(["foo.torrent"])
        assert args.verbose is False


# ---------------------------------------------------------------------------
# _run() error paths
# ---------------------------------------------------------------------------

def make_torrent_file(tmp_path: Path) -> Path:
    """Write a minimal valid .torrent file and return its path."""
    piece_data = b"\xab" * 512
    piece_hash = hashlib.sha1(piece_data).digest()
    info = {
        b"length": 512,
        b"name": b"test.bin",
        b"piece length": 512,
        b"pieces": piece_hash,
    }
    meta = encode({
        b"announce": b"http://tracker.example.com/announce",
        b"info": info,
    })
    path = tmp_path / "test.torrent"
    path.write_bytes(meta)
    return path


class TestRunErrors:
    async def test_missing_torrent_file_returns_1(self, tmp_path):
        args = _parse_args([str(tmp_path / "nonexistent.torrent")])
        code = await _run(args)
        assert code == 1

    async def test_invalid_torrent_file_returns_1(self, tmp_path):
        bad = tmp_path / "bad.torrent"
        bad.write_bytes(b"this is not bencode")
        args = _parse_args([str(bad)])
        code = await _run(args)
        assert code == 1

    async def test_tracker_error_returns_1(self, tmp_path):
        torrent_path = make_torrent_file(tmp_path)
        args = _parse_args([str(torrent_path), "-o", str(tmp_path)])

        from bittorrent.tracker import TrackerError
        with patch("bittorrent.main.announce", new=AsyncMock(
            side_effect=TrackerError("tracker down")
        )):
            code = await _run(args)
        assert code == 1

    async def test_no_peers_returns_1(self, tmp_path):
        torrent_path = make_torrent_file(tmp_path)
        args = _parse_args([str(torrent_path), "-o", str(tmp_path)])

        empty_response = TrackerResponse(interval=1800, peers=[])
        with patch("bittorrent.main.announce", new=AsyncMock(return_value=empty_response)):
            code = await _run(args)
        assert code == 1


class TestRunSuccess:
    async def test_successful_download_returns_0(self, tmp_path):
        torrent_path = make_torrent_file(tmp_path)
        args = _parse_args([str(torrent_path), "-o", str(tmp_path)])

        peers    = [("1.2.3.4", 6881)]
        response = TrackerResponse(interval=1800, peers=peers, complete=1)

        with (
            patch("bittorrent.main.announce", new=AsyncMock(return_value=response)),
            patch("bittorrent.main.PeerManager") as MockPM,
            patch("bittorrent.main.Storage.is_complete", return_value=False),
        ):
            instance = AsyncMock()
            instance.run = AsyncMock()
            MockPM.return_value = instance

            code = await _run(args)

        instance.run.assert_called_once()
        assert code == 0

    async def test_already_complete_returns_0(self, tmp_path):
        torrent_path = make_torrent_file(tmp_path)
        args = _parse_args([str(torrent_path), "-o", str(tmp_path)])

        response = TrackerResponse(interval=1800, peers=[("1.2.3.4", 6881)])
        with (
            patch("bittorrent.main.announce", new=AsyncMock(return_value=response)),
            patch("bittorrent.main.Storage.is_complete", return_value=True),
        ):
            code = await _run(args)
        assert code == 0


# ---------------------------------------------------------------------------
# _announce_all — multi-tracker support
# ---------------------------------------------------------------------------

class TestAnnounceAll:
    async def test_announces_to_all_trackers_in_announce_list(self, tmp_path):
        """All trackers in announce-list are tried in addition to announce."""
        from bittorrent.main import _announce_all
        from bittorrent.torrent import Torrent
        from bittorrent.tracker import TrackerResponse
        import hashlib

        torrent = Torrent(
            announce="http://t1.example.com/announce",
            announce_list=[
                ["http://t2.example.com/announce"],
                ["http://t3.example.com/announce"],
            ],
            info_hash=b"\x00" * 20,
            info_hash_hex="00" * 20,
            name="test",
            piece_length=512,
            piece_hashes=[hashlib.sha1(b"x").digest()],
            length=512,
        )

        called_urls = []
        async def fake_announce(url, *a, **kw):
            called_urls.append(url)
            return TrackerResponse(interval=1800, peers=[("1.2.3.4", 6881)])

        from rich.console import Console
        console = Console(stderr=True)
        with patch("bittorrent.main.announce", new=fake_announce):
            peers = await _announce_all(torrent, b"-BC0001-" + b"X" * 12, 6881, console)

        assert "http://t1.example.com/announce" in called_urls
        assert "http://t2.example.com/announce" in called_urls
        assert "http://t3.example.com/announce" in called_urls

    async def test_deduplicates_peers_across_trackers(self, tmp_path):
        """Same peer returned by multiple trackers only appears once."""
        from bittorrent.main import _announce_all
        from bittorrent.torrent import Torrent
        from bittorrent.tracker import TrackerResponse
        import hashlib

        torrent = Torrent(
            announce="http://t1.example.com/announce",
            announce_list=[["http://t2.example.com/announce"]],
            info_hash=b"\x00" * 20,
            info_hash_hex="00" * 20,
            name="test",
            piece_length=512,
            piece_hashes=[hashlib.sha1(b"x").digest()],
            length=512,
        )

        async def fake_announce(url, *a, **kw):
            # Both trackers return the same peer
            return TrackerResponse(interval=1800, peers=[("1.2.3.4", 6881)])

        from rich.console import Console
        console = Console(stderr=True)
        with patch("bittorrent.main.announce", new=fake_announce):
            peers = await _announce_all(torrent, b"-BC0001-" + b"X" * 12, 6881, console)

        assert peers.count(("1.2.3.4", 6881)) == 1

    async def test_partial_tracker_failure_still_returns_peers(self, tmp_path):
        """If one tracker fails, peers from the others are still returned."""
        from bittorrent.main import _announce_all
        from bittorrent.torrent import Torrent
        from bittorrent.tracker import TrackerResponse, TrackerError
        import hashlib

        torrent = Torrent(
            announce="http://bad.example.com/announce",
            announce_list=[["http://good.example.com/announce"]],
            info_hash=b"\x00" * 20,
            info_hash_hex="00" * 20,
            name="test",
            piece_length=512,
            piece_hashes=[hashlib.sha1(b"x").digest()],
            length=512,
        )

        async def fake_announce(url, *a, **kw):
            if "bad" in url:
                raise TrackerError("timeout")
            return TrackerResponse(interval=1800, peers=[("5.6.7.8", 6882)])

        from rich.console import Console
        console = Console(stderr=True)
        with patch("bittorrent.main.announce", new=fake_announce):
            peers = await _announce_all(torrent, b"-BC0001-" + b"X" * 12, 6881, console)

        assert ("5.6.7.8", 6882) in peers

    async def test_no_trackers_returns_empty(self, tmp_path):
        """Torrent with no trackers returns empty peer list."""
        from bittorrent.main import _announce_all
        from bittorrent.torrent import Torrent
        import hashlib

        torrent = Torrent(
            announce="",
            announce_list=[],
            info_hash=b"\x00" * 20,
            info_hash_hex="00" * 20,
            name="test",
            piece_length=512,
            piece_hashes=[hashlib.sha1(b"x").digest()],
            length=512,
        )

        from rich.console import Console
        console = Console(stderr=True)
        peers = await _announce_all(torrent, b"-BC0001-" + b"X" * 12, 6881, console)
        assert peers == []
