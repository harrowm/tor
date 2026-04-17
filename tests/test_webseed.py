"""
Tests for bittorrent.webseed — BEP 19 HTTP seeding.

Tests cover:
  - Torrent.url_list parsing
  - WebSeedClient URL construction (single-file and multi-file)
  - fetch_piece: success, HTTP error, hash mismatch, timeout
  - PeerManager web seed worker integration (mocked HTTP)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bittorrent.bencode import encode
from bittorrent.torrent import Torrent, FileInfo, parse
from bittorrent.webseed import (
    WebSeedClient,
    WebSeedError,
    build_webseed_clients,
    _FileRegion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_torrent_bytes(
    *,
    name: str = "test.bin",
    piece_length: int = 512,
    total_length: int = 512,
    url_list: list[str] | str | None = None,
    multi_file: bool = False,
) -> bytes:
    """Build a minimal bencoded .torrent with optional url-list."""
    piece_data = b"\xab" * total_length
    piece_hash = hashlib.sha1(piece_data[:piece_length]).digest()

    if multi_file:
        info: dict = {
            b"name": name.encode(),
            b"piece length": piece_length,
            b"pieces": piece_hash,
            b"files": [
                {b"length": total_length // 2, b"path": [b"a.txt"]},
                {b"length": total_length // 2, b"path": [b"b.txt"]},
            ],
        }
    else:
        info = {
            b"name": name.encode(),
            b"piece length": piece_length,
            b"pieces": piece_hash,
            b"length": total_length,
        }

    meta: dict = {
        b"announce": b"http://tracker.example.com/announce",
        b"info": info,
    }

    if url_list is not None:
        if isinstance(url_list, list):
            meta[b"url-list"] = [u.encode() for u in url_list]
        else:
            meta[b"url-list"] = url_list.encode()

    return encode(meta)


def single_file_torrent(
    name: str = "test.bin",
    piece_length: int = 512,
    pieces_data: list[bytes] | None = None,
    url_list: list[str] | None = None,
) -> Torrent:
    if pieces_data is None:
        pieces_data = [b"\xab" * piece_length]
    total = sum(len(p) for p in pieces_data)
    return Torrent(
        announce="http://tracker.example.com/announce",
        info_hash=b"\x00" * 20,
        info_hash_hex="00" * 20,
        name=name,
        piece_length=piece_length,
        piece_hashes=[hashlib.sha1(p).digest() for p in pieces_data],
        length=total,
        url_list=url_list or [],
    )


def multi_file_torrent(
    name: str = "mydir",
    piece_length: int = 512,
    files: list[tuple[list[str], bytes]] | None = None,
    url_list: list[str] | None = None,
) -> Torrent:
    if files is None:
        files = [(["a.txt"], b"\xaa" * 300), (["b.txt"], b"\xbb" * 300)]
    total_data = b"".join(data for _, data in files)
    n_pieces = -(-len(total_data) // piece_length)
    pieces_data = [
        total_data[i * piece_length: (i + 1) * piece_length]
        for i in range(n_pieces)
    ]
    file_infos = [
        FileInfo(path=path_parts, length=len(data))
        for path_parts, data in files
    ]
    return Torrent(
        announce="http://tracker.example.com/announce",
        info_hash=b"\x00" * 20,
        info_hash_hex="00" * 20,
        name=name,
        piece_length=piece_length,
        piece_hashes=[hashlib.sha1(p).digest() for p in pieces_data],
        files=file_infos,
        url_list=url_list or [],
    )


# ---------------------------------------------------------------------------
# Torrent.url_list parsing
# ---------------------------------------------------------------------------

class TestUrlListParsing:
    def test_single_url_as_string(self):
        raw = make_torrent_bytes(url_list="https://example.com/files/")
        t = parse(raw)
        assert t.url_list == ["https://example.com/files/"]

    def test_multiple_urls_as_list(self):
        raw = make_torrent_bytes(
            url_list=["https://seed1.example.com/", "https://seed2.example.com/"]
        )
        t = parse(raw)
        assert t.url_list == [
            "https://seed1.example.com/",
            "https://seed2.example.com/",
        ]

    def test_no_url_list(self):
        raw = make_torrent_bytes()
        t = parse(raw)
        assert t.url_list == []

    def test_url_list_empty_list(self):
        from bittorrent.bencode import encode
        meta = {
            b"announce": b"http://tracker.example.com/announce",
            b"info": {
                b"name": b"test.bin",
                b"piece length": 512,
                b"pieces": b"\x00" * 20,
                b"length": 512,
            },
            b"url-list": [],
        }
        t = parse(encode(meta))
        assert t.url_list == []


# ---------------------------------------------------------------------------
# WebSeedClient — URL construction
# ---------------------------------------------------------------------------

class TestWebSeedClientURLs:
    def test_single_file_url(self):
        t = single_file_torrent(name="ubuntu.iso")
        client = WebSeedClient(t, "https://releases.ubuntu.com/")
        assert len(client._regions) == 1
        assert client._regions[0].url == "https://releases.ubuntu.com/ubuntu.iso"

    def test_single_file_url_trailing_slash_stripped(self):
        t = single_file_torrent(name="file.bin")
        client = WebSeedClient(t, "https://example.com/files")
        assert client._regions[0].url == "https://example.com/files/file.bin"

    def test_multi_file_url_construction(self):
        t = multi_file_torrent(
            name="mydir",
            files=[(["a.txt"], b"\xaa" * 100), (["b.txt"], b"\xbb" * 200)],
        )
        client = WebSeedClient(t, "https://example.com/")
        urls = [r.url for r in client._regions]
        assert urls[0] == "https://example.com/mydir/a.txt"
        assert urls[1] == "https://example.com/mydir/b.txt"

    def test_multi_file_nested_path(self):
        t = multi_file_torrent(
            files=[(["sub", "deep.txt"], b"\x00" * 50)],
        )
        client = WebSeedClient(t, "https://example.com/")
        assert client._regions[0].url == "https://example.com/mydir/sub/deep.txt"

    def test_region_offsets_correct(self):
        t = multi_file_torrent(
            name="mydir",
            files=[(["a.txt"], b"\xaa" * 100), (["b.txt"], b"\xbb" * 200)],
        )
        client = WebSeedClient(t, "https://example.com/")
        assert client._regions[0].torrent_offset == 0
        assert client._regions[0].length == 100
        assert client._regions[1].torrent_offset == 100
        assert client._regions[1].length == 200

    def test_name_is_url_encoded(self):
        t = single_file_torrent(name="my file.bin")
        client = WebSeedClient(t, "https://example.com/")
        assert client._regions[0].url == "https://example.com/my%20file.bin"

    def test_build_webseed_clients_creates_one_per_url(self):
        t = single_file_torrent(url_list=["https://a.com/", "https://b.com/"])
        clients = build_webseed_clients(t)
        assert len(clients) == 2
        assert clients[0]._base_url == "https://a.com"
        assert clients[1]._base_url == "https://b.com"

    def test_build_webseed_clients_empty_list(self):
        t = single_file_torrent(url_list=[])
        assert build_webseed_clients(t) == []


# ---------------------------------------------------------------------------
# WebSeedClient.fetch_piece — mocked HTTP responses
# ---------------------------------------------------------------------------

def _make_mock_session(
    *,
    status: int = 206,
    data: bytes = b"",
    raise_exc: Exception | None = None,
) -> MagicMock:
    """Return an aiohttp.ClientSession mock."""
    import aiohttp

    resp = AsyncMock()
    resp.status = status
    resp.read   = AsyncMock(return_value=data)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__  = AsyncMock(return_value=False)

    if raise_exc is not None:
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(side_effect=raise_exc)
        cm.__aexit__  = AsyncMock(return_value=False)
    else:
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__  = AsyncMock(return_value=False)

    session = MagicMock()
    session.get = MagicMock(return_value=cm)
    return session


class TestFetchPiece:
    async def test_fetches_correct_bytes(self):
        piece = b"\xde\xad\xbe\xef" * 128   # 512 bytes
        t = single_file_torrent(pieces_data=[piece])
        client = WebSeedClient(t, "https://example.com/")
        session = _make_mock_session(status=206, data=piece)
        result = await client.fetch_piece(session, 0)
        assert result == piece

    async def test_sends_range_header(self):
        piece = b"\xab" * 512
        t = single_file_torrent(pieces_data=[piece])
        client = WebSeedClient(t, "https://example.com/")
        session = _make_mock_session(status=206, data=piece)
        await client.fetch_piece(session, 0)
        call_kwargs = session.get.call_args[1]
        assert "Range" in call_kwargs["headers"]
        assert call_kwargs["headers"]["Range"] == "bytes=0-511"

    async def test_range_header_for_second_piece(self):
        p0 = b"\xaa" * 512
        p1 = b"\xbb" * 512
        t = single_file_torrent(pieces_data=[p0, p1])
        client = WebSeedClient(t, "https://example.com/")

        session = _make_mock_session(status=206, data=p1)
        await client.fetch_piece(session, 1)
        call_kwargs = session.get.call_args[1]
        assert call_kwargs["headers"]["Range"] == "bytes=512-1023"

    async def test_range_header_for_short_last_piece(self):
        p0 = b"\xaa" * 512
        p1 = b"\xbb" * 200
        t = single_file_torrent(pieces_data=[p0, p1])
        client = WebSeedClient(t, "https://example.com/")

        session = _make_mock_session(status=206, data=p1)
        await client.fetch_piece(session, 1)
        call_kwargs = session.get.call_args[1]
        assert call_kwargs["headers"]["Range"] == "bytes=512-711"

    async def test_raises_on_http_error(self):
        piece = b"\xab" * 512
        t = single_file_torrent(pieces_data=[piece])
        client = WebSeedClient(t, "https://example.com/")
        session = _make_mock_session(status=404, data=b"Not Found")
        with pytest.raises(WebSeedError, match="HTTP 404"):
            await client.fetch_piece(session, 0)

    async def test_raises_on_hash_mismatch(self):
        piece = b"\xab" * 512
        t = single_file_torrent(pieces_data=[piece])
        client = WebSeedClient(t, "https://example.com/")
        # Serve wrong data that won't hash correctly
        session = _make_mock_session(status=206, data=b"\x00" * 512)
        with pytest.raises(WebSeedError, match="hash mismatch"):
            await client.fetch_piece(session, 0)

    async def test_raises_on_client_error(self):
        import aiohttp
        piece = b"\xab" * 512
        t = single_file_torrent(pieces_data=[piece])
        client = WebSeedClient(t, "https://example.com/")
        session = _make_mock_session(raise_exc=aiohttp.ClientConnectionError("refused"))
        with pytest.raises(WebSeedError, match="HTTP error"):
            await client.fetch_piece(session, 0)

    async def test_raises_out_of_range(self):
        piece = b"\xab" * 512
        t = single_file_torrent(pieces_data=[piece])
        client = WebSeedClient(t, "https://example.com/")
        session = _make_mock_session(status=206, data=piece)
        with pytest.raises(WebSeedError, match="out of range"):
            await client.fetch_piece(session, 99)

    async def test_200_ok_also_accepted(self):
        """Some servers return 200 instead of 206 for Range requests."""
        piece = b"\xcd" * 512
        t = single_file_torrent(pieces_data=[piece])
        client = WebSeedClient(t, "https://example.com/")
        session = _make_mock_session(status=200, data=piece)
        result = await client.fetch_piece(session, 0)
        assert result == piece


# ---------------------------------------------------------------------------
# Multi-file piece fetching
# ---------------------------------------------------------------------------

class TestMultiFileFetch:
    async def test_spanning_piece_fetched_from_two_files(self):
        """A piece that spans a.txt and b.txt is assembled from two Range requests."""
        file_a = b"\xaa" * 300
        file_b = b"\xbb" * 300
        total  = file_a + file_b

        t = multi_file_torrent(
            piece_length=512,
            files=[(["a.txt"], file_a), (["b.txt"], file_b)],
        )
        client = WebSeedClient(t, "https://example.com/")

        # Piece 0 = bytes 0-511 → 300 from a.txt + 212 from b.txt
        piece0 = total[:512]
        responses_given = []

        def make_response(chunk):
            resp = AsyncMock()
            resp.status = 206
            resp.read   = AsyncMock(return_value=chunk)
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=resp)
            cm.__aexit__  = AsyncMock(return_value=False)
            return cm

        call_count = 0
        def get_side_effect(url, headers, timeout):
            nonlocal call_count
            call_count += 1
            if "a.txt" in url:
                return make_response(file_a)   # all of a.txt for bytes 0-511
            else:
                return make_response(file_b[:212])  # first 212 bytes of b.txt

        session = MagicMock()
        session.get = MagicMock(side_effect=get_side_effect)

        result = await client.fetch_piece(session, 0)
        assert result == piece0
        assert call_count == 2   # one request per file

    async def test_piece_entirely_within_one_file(self):
        """A piece that fits entirely within the first file uses only one URL."""
        file_a = b"\xaa" * 600
        file_b = b"\xbb" * 600
        # piece_length=512, piece 0 = bytes 0-511 → entirely in file_a (600 bytes)

        t = multi_file_torrent(
            piece_length=512,
            files=[(["a.txt"], file_a), (["b.txt"], file_b)],
        )
        client = WebSeedClient(t, "https://example.com/")

        piece0 = file_a[:512]
        call_count = 0

        def get_side_effect(url, headers, timeout):
            nonlocal call_count
            call_count += 1
            resp = AsyncMock()
            resp.status = 206
            resp.read   = AsyncMock(return_value=piece0)
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=resp)
            cm.__aexit__  = AsyncMock(return_value=False)
            return cm

        session = MagicMock()
        session.get = MagicMock(side_effect=get_side_effect)

        result = await client.fetch_piece(session, 0)
        assert result == piece0
        assert call_count == 1   # only one request needed


# ---------------------------------------------------------------------------
# PeerManager: web seed worker integration
# ---------------------------------------------------------------------------

class TestWebSeedIntegration:
    """Verify PeerManager spawns web seed workers and completes download."""

    async def test_downloads_via_webseed_when_no_peers(self, tmp_path):
        """With no peers but a web seed, the download completes via HTTP."""
        from bittorrent.piece_manager import PieceManager
        from bittorrent.peer_manager import PeerManager
        from bittorrent.storage import Storage

        piece = b"\xde\xad\xbe\xef" * 128   # 512 bytes
        torrent = single_file_torrent(
            pieces_data=[piece],
            url_list=["https://example.com/"],
        )

        pm      = PieceManager(1, 512, 512)
        storage = Storage(torrent, tmp_path)
        manager = PeerManager(torrent, pm, storage, b"\x00" * 20, b"\x00" * 20)

        mock_session = AsyncMock()
        mock_session.close = AsyncMock()

        with (
            patch("bittorrent.peer_manager.build_webseed_clients") as mock_build,
            patch.object(manager, "_create_http_session", return_value=mock_session),
        ):
            mock_client = MagicMock()
            mock_client.fetch_piece = AsyncMock(return_value=piece)
            mock_build.return_value = [mock_client]

            # No peers — download must come entirely from web seed
            await manager.run([], allocate=True)

        assert pm.is_complete()

    async def test_webseed_worker_retries_on_error(self, tmp_path):
        """A WebSeedError on one piece causes a retry rather than crashing."""
        from bittorrent.piece_manager import PieceManager
        from bittorrent.peer_manager import PeerManager
        from bittorrent.storage import Storage

        piece = b"\xab" * 512
        torrent = single_file_torrent(
            pieces_data=[piece],
            url_list=["https://example.com/"],
        )

        pm      = PieceManager(1, 512, 512)
        storage = Storage(torrent, tmp_path)
        manager = PeerManager(torrent, pm, storage, b"\x00" * 20, b"\x00" * 20)

        call_count = 0

        async def flaky_fetch(session, piece_index):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise WebSeedError("transient error")
            return piece

        mock_session = AsyncMock()
        mock_session.close = AsyncMock()

        with (
            patch("bittorrent.peer_manager.build_webseed_clients") as mock_build,
            patch.object(manager, "_create_http_session", return_value=mock_session),
        ):
            mock_client = MagicMock()
            mock_client.fetch_piece = flaky_fetch
            mock_build.return_value = [mock_client]

            await manager.run([], allocate=True)

        assert pm.is_complete()
        assert call_count == 2   # first attempt failed, second succeeded

    async def test_webseed_and_peers_work_together(self, tmp_path):
        """Web seed and peer workers coexist; both contribute to completion."""
        from bittorrent.piece_manager import PieceManager
        from bittorrent.peer_manager import PeerManager
        from bittorrent.storage import Storage

        # 2 pieces: peer gets piece 0, web seed gets piece 1
        pieces = [b"\xaa" * 512, b"\xbb" * 512]
        torrent = Torrent(
            announce="http://tracker.example.com/announce",
            info_hash=b"\x00" * 20,
            info_hash_hex="00" * 20,
            name="test.bin",
            piece_length=512,
            piece_hashes=[hashlib.sha1(p).digest() for p in pieces],
            length=1024,
            url_list=["https://example.com/"],
        )

        pm      = PieceManager(2, 512, 1024)
        storage = Storage(torrent, tmp_path)
        manager = PeerManager(torrent, pm, storage, b"\x00" * 20, b"\x00" * 20)

        class FakePeer:
            host     = "1.2.3.4"
            port     = 6881
            bitfield = bytearray(b"\x80")  # piece 0 only
            remote_supports_extensions = False
            _pending = []

            async def download_piece(self, idx, size, h, **kwargs):
                return pieces[idx]

            async def do_extension_handshake(self, *a, **kw): pass
            def peer_ext_id(self, name): return None
            async def close(self): pass

        async def webseed_fetch(session, piece_index):
            if piece_index == 1:
                return pieces[1]
            raise WebSeedError("piece 0 not served by web seed")

        mock_session = AsyncMock()
        mock_session.close = AsyncMock()

        with (
            patch("bittorrent.peer_manager.PeerConnection.open",
                  new=AsyncMock(return_value=FakePeer())),
            patch("bittorrent.peer_manager.build_webseed_clients") as mock_build,
            patch.object(manager, "_create_http_session", return_value=mock_session),
        ):
            mock_client = MagicMock()
            mock_client.fetch_piece = webseed_fetch
            mock_build.return_value = [mock_client]

            await manager.run([("1.2.3.4", 6881)])

        assert pm.is_complete()
