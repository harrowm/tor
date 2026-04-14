"""
Tests for bittorrent.torrent — .torrent file parser.

We build synthetic .torrent files using our bencode encoder so tests
have no external file dependencies. A few tests verify the critical
info_hash computation is correct.
"""

import hashlib
import pytest
from bittorrent.bencode import encode
from bittorrent.torrent import parse, ParseError, Torrent, FileInfo


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_pieces(n: int = 1) -> bytes:
    """Return n * 20 bytes of fake SHA-1 piece hashes."""
    return bytes(range(20)) * n


def make_single_file_torrent(
    *,
    name: str = "test.txt",
    length: int = 1024,
    piece_length: int = 512,
    pieces: bytes | None = None,
    announce: str = "http://tracker.example.com/announce",
    extra_info: dict | None = None,
    extra_meta: dict | None = None,
) -> bytes:
    """Build a minimal single-file .torrent as bytes."""
    n_pieces = -(-length // piece_length)  # ceiling division
    if pieces is None:
        pieces = make_pieces(n_pieces)

    info: dict = {
        b"length": length,
        b"name": name.encode(),
        b"piece length": piece_length,
        b"pieces": pieces,
    }
    if extra_info:
        info.update(extra_info)

    meta: dict = {
        b"announce": announce.encode(),
        b"info": info,
    }
    if extra_meta:
        meta.update(extra_meta)

    return encode(meta)


def make_multi_file_torrent(
    *,
    name: str = "mydir",
    files: list[tuple[list[str], int]] | None = None,
    piece_length: int = 512,
    announce: str = "http://tracker.example.com/announce",
) -> bytes:
    """Build a minimal multi-file .torrent as bytes."""
    if files is None:
        files = [(["a.txt"], 100), (["sub", "b.txt"], 200)]

    total = sum(length for _, length in files)
    n_pieces = -(-total // piece_length)
    pieces = make_pieces(n_pieces)

    file_list = [
        {
            b"length": length,
            b"path": [p.encode() for p in path],
        }
        for path, length in files
    ]

    info: dict = {
        b"files": file_list,
        b"name": name.encode(),
        b"piece length": piece_length,
        b"pieces": pieces,
    }

    return encode({
        b"announce": announce.encode(),
        b"info": info,
    })


# ---------------------------------------------------------------------------
# Basic parsing — single-file
# ---------------------------------------------------------------------------

class TestSingleFile:
    def test_returns_torrent_instance(self):
        data = make_single_file_torrent()
        t = parse(data)
        assert isinstance(t, Torrent)

    def test_announce(self):
        t = parse(make_single_file_torrent(announce="http://example.com/announce"))
        assert t.announce == "http://example.com/announce"

    def test_name(self):
        t = parse(make_single_file_torrent(name="hello.txt"))
        assert t.name == "hello.txt"

    def test_length(self):
        t = parse(make_single_file_torrent(length=99999))
        assert t.length == 99999

    def test_piece_length(self):
        t = parse(make_single_file_torrent(piece_length=262144))
        assert t.piece_length == 262144

    def test_piece_hashes_count(self):
        # 1000 bytes / 512 bytes per piece = 2 pieces
        t = parse(make_single_file_torrent(length=1000, piece_length=512))
        assert t.num_pieces == 2
        assert len(t.piece_hashes) == 2

    def test_piece_hashes_are_20_bytes(self):
        t = parse(make_single_file_torrent())
        for h in t.piece_hashes:
            assert len(h) == 20

    def test_is_not_multi_file(self):
        t = parse(make_single_file_torrent())
        assert not t.is_multi_file
        assert t.files == []

    def test_total_length_equals_length(self):
        t = parse(make_single_file_torrent(length=5000))
        assert t.total_length == 5000


# ---------------------------------------------------------------------------
# Basic parsing — multi-file
# ---------------------------------------------------------------------------

class TestMultiFile:
    def test_is_multi_file(self):
        t = parse(make_multi_file_torrent())
        assert t.is_multi_file

    def test_length_is_zero(self):
        t = parse(make_multi_file_torrent())
        assert t.length == 0

    def test_files_parsed(self):
        data = make_multi_file_torrent(
            files=[(["a.txt"], 100), (["sub", "b.txt"], 200)]
        )
        t = parse(data)
        assert len(t.files) == 2
        assert t.files[0].path == ["a.txt"]
        assert t.files[0].length == 100
        assert t.files[1].path == ["sub", "b.txt"]
        assert t.files[1].length == 200

    def test_total_length(self):
        data = make_multi_file_torrent(
            files=[(["a.txt"], 100), (["b.txt"], 200), (["c.txt"], 700)]
        )
        t = parse(data)
        assert t.total_length == 1000

    def test_name(self):
        t = parse(make_multi_file_torrent(name="myalbum"))
        assert t.name == "myalbum"


# ---------------------------------------------------------------------------
# info_hash — the critical test
# ---------------------------------------------------------------------------

class TestInfoHash:
    def test_info_hash_is_20_bytes(self):
        t = parse(make_single_file_torrent())
        assert len(t.info_hash) == 20

    def test_info_hash_hex_matches(self):
        t = parse(make_single_file_torrent())
        assert t.info_hash_hex == t.info_hash.hex()
        assert len(t.info_hash_hex) == 40

    def test_info_hash_correct_value(self):
        """The info_hash must be SHA-1 of the bencoded info dict bytes."""
        # Build the info dict separately so we know exactly what to hash.
        piece_length = 512
        length = 1024
        pieces = make_pieces(2)

        info = {
            b"length": length,
            b"name": b"test.txt",
            b"piece length": piece_length,
            b"pieces": pieces,
        }
        info_encoded = encode(info)
        expected_hash = hashlib.sha1(info_encoded).digest()

        meta = encode({b"announce": b"http://tracker.example.com/announce", b"info": info})
        t = parse(meta)

        assert t.info_hash == expected_hash

    def test_different_torrents_different_hash(self):
        t1 = parse(make_single_file_torrent(name="a.txt"))
        t2 = parse(make_single_file_torrent(name="b.txt"))
        assert t1.info_hash != t2.info_hash

    def test_same_content_same_hash(self):
        data = make_single_file_torrent()
        t1 = parse(data)
        t2 = parse(data)
        assert t1.info_hash == t2.info_hash


# ---------------------------------------------------------------------------
# announce-list (BEP 12)
# ---------------------------------------------------------------------------

class TestAnnounceList:
    def test_no_announce_list(self):
        t = parse(make_single_file_torrent())
        assert t.announce_list == []

    def test_announce_list_parsed(self):
        data = make_single_file_torrent(
            extra_meta={
                b"announce-list": [
                    [b"http://tracker1.com/announce"],
                    [b"http://tracker2.com/announce", b"http://tracker3.com/announce"],
                ]
            }
        )
        t = parse(data)
        assert t.announce_list == [
            ["http://tracker1.com/announce"],
            ["http://tracker2.com/announce", "http://tracker3.com/announce"],
        ]


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestParseErrors:
    def test_not_a_dict(self):
        with pytest.raises(ParseError):
            parse(encode([1, 2, 3]))

    def test_missing_announce(self):
        data = encode({
            b"info": {
                b"length": 100,
                b"name": b"x",
                b"piece length": 512,
                b"pieces": make_pieces(1),
            }
        })
        with pytest.raises(ParseError, match="announce"):
            parse(data)

    def test_missing_info(self):
        data = encode({b"announce": b"http://x.com"})
        with pytest.raises(ParseError, match="info"):
            parse(data)

    def test_missing_name(self):
        data = encode({
            b"announce": b"http://x.com",
            b"info": {
                b"length": 100,
                b"piece length": 512,
                b"pieces": make_pieces(1),
            },
        })
        with pytest.raises(ParseError, match="name"):
            parse(data)

    def test_missing_piece_length(self):
        data = encode({
            b"announce": b"http://x.com",
            b"info": {
                b"length": 100,
                b"name": b"x",
                b"pieces": make_pieces(1),
            },
        })
        with pytest.raises(ParseError, match="piece length"):
            parse(data)

    def test_pieces_not_multiple_of_20(self):
        data = encode({
            b"announce": b"http://x.com",
            b"info": {
                b"length": 100,
                b"name": b"x",
                b"piece length": 512,
                b"pieces": b"tooshort",
            },
        })
        with pytest.raises(ParseError, match="multiple of 20"):
            parse(data)

    def test_invalid_bencode(self):
        with pytest.raises(ParseError):
            parse(b"not bencode at all!!!")

    def test_empty_bytes(self):
        with pytest.raises(ParseError):
            parse(b"")
