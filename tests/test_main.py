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
