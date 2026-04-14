"""
Tests for bittorrent.storage — allocation, piece writes, reads, verification.

Uses tmp_path (pytest fixture) so all file I/O is isolated to a temp directory
that is cleaned up automatically.
"""

import hashlib
from pathlib import Path

import pytest

from bittorrent.bencode import encode
from bittorrent.storage import Storage, StorageError, _build_regions
from bittorrent.torrent import Torrent, FileInfo


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def make_piece_hashes(pieces_data: list[bytes]) -> list[bytes]:
    return [hashlib.sha1(p).digest() for p in pieces_data]


def single_file_torrent(
    name: str = "test.bin",
    piece_length: int = 512,
    pieces_data: list[bytes] | None = None,
    total_length: int | None = None,
) -> Torrent:
    if pieces_data is None:
        pieces_data = [b"\xab" * piece_length]
    if total_length is None:
        total_length = sum(len(p) for p in pieces_data)
    return Torrent(
        announce="http://tracker.example.com/announce",
        info_hash=b"\x00" * 20,
        info_hash_hex="00" * 20,
        name=name,
        piece_length=piece_length,
        piece_hashes=make_piece_hashes(pieces_data),
        length=total_length,
    )


def multi_file_torrent(
    name: str = "mydir",
    piece_length: int = 512,
    files: list[tuple[list[str], bytes]] | None = None,
) -> Torrent:
    if files is None:
        files = [(["a.txt"], b"\xaa" * 300), (["b.txt"], b"\xbb" * 300)]

    total_data = b"".join(data for _, data in files)
    n_pieces   = -(-len(total_data) // piece_length)
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
        piece_hashes=make_piece_hashes(pieces_data),
        files=file_infos,
    )


# ---------------------------------------------------------------------------
# _build_regions
# ---------------------------------------------------------------------------

class TestBuildRegions:
    def test_single_file_one_region(self, tmp_path):
        t = single_file_torrent(name="hello.bin", total_length=1000)
        regions = _build_regions(t, tmp_path)
        assert len(regions) == 1
        assert regions[0].path == tmp_path / "hello.bin"
        assert regions[0].torrent_offset == 0
        assert regions[0].length == 1000

    def test_multi_file_regions_ordered(self, tmp_path):
        t = multi_file_torrent(
            name="mydir",
            files=[
                (["a.txt"], b"\xaa" * 100),
                (["b.txt"], b"\xbb" * 200),
            ],
        )
        regions = _build_regions(t, tmp_path)
        assert len(regions) == 2
        assert regions[0].path == tmp_path / "mydir" / "a.txt"
        assert regions[0].torrent_offset == 0
        assert regions[0].length == 100
        assert regions[1].path == tmp_path / "mydir" / "b.txt"
        assert regions[1].torrent_offset == 100
        assert regions[1].length == 200

    def test_multi_file_nested_path(self, tmp_path):
        t = multi_file_torrent(
            files=[(["sub", "deep.txt"], b"\x00" * 50)]
        )
        regions = _build_regions(t, tmp_path)
        assert regions[0].path == tmp_path / "mydir" / "sub" / "deep.txt"


# ---------------------------------------------------------------------------
# allocate
# ---------------------------------------------------------------------------

class TestAllocate:
    def test_creates_file(self, tmp_path):
        t = single_file_torrent(total_length=1024)
        s = Storage(t, tmp_path)
        s.allocate()
        assert (tmp_path / "test.bin").exists()

    def test_file_correct_size(self, tmp_path):
        t = single_file_torrent(total_length=1024)
        s = Storage(t, tmp_path)
        s.allocate()
        assert (tmp_path / "test.bin").stat().st_size == 1024

    def test_creates_parent_dirs(self, tmp_path):
        t = multi_file_torrent(
            files=[(["sub", "file.dat"], b"\x00" * 100)]
        )
        s = Storage(t, tmp_path)
        s.allocate()
        assert (tmp_path / "mydir" / "sub" / "file.dat").exists()

    def test_idempotent(self, tmp_path):
        t = single_file_torrent(total_length=512)
        s = Storage(t, tmp_path)
        s.allocate()
        s.allocate()   # second call must not raise
        assert (tmp_path / "test.bin").stat().st_size == 512

    def test_multi_file_all_created(self, tmp_path):
        t = multi_file_torrent(
            files=[
                (["a.txt"], b"\xaa" * 100),
                (["b.txt"], b"\xbb" * 200),
            ],
        )
        s = Storage(t, tmp_path)
        s.allocate()
        assert (tmp_path / "mydir" / "a.txt").stat().st_size == 100
        assert (tmp_path / "mydir" / "b.txt").stat().st_size == 200


# ---------------------------------------------------------------------------
# write_piece / read_piece — single-file
# ---------------------------------------------------------------------------

class TestSingleFileWriteRead:
    def test_write_and_read_single_piece(self, tmp_path):
        piece = b"\xde\xad\xbe\xef" * 128   # 512 bytes
        t = single_file_torrent(piece_length=512, pieces_data=[piece])
        s = Storage(t, tmp_path)
        s.allocate()
        s.write_piece(0, piece)
        assert s.read_piece(0) == piece

    def test_write_multiple_pieces(self, tmp_path):
        p0 = b"\xaa" * 512
        p1 = b"\xbb" * 512
        t = single_file_torrent(piece_length=512, pieces_data=[p0, p1])
        s = Storage(t, tmp_path)
        s.allocate()
        s.write_piece(0, p0)
        s.write_piece(1, p1)
        assert s.read_piece(0) == p0
        assert s.read_piece(1) == p1

    def test_pieces_do_not_overlap(self, tmp_path):
        p0 = b"\x11" * 512
        p1 = b"\x22" * 512
        t = single_file_torrent(piece_length=512, pieces_data=[p0, p1])
        s = Storage(t, tmp_path)
        s.allocate()
        s.write_piece(0, p0)
        s.write_piece(1, p1)
        # Overwrite piece 0 — piece 1 must be unaffected
        p0_new = b"\x33" * 512
        s.write_piece(0, p0_new)
        assert s.read_piece(1) == p1

    def test_last_piece_shorter(self, tmp_path):
        p0    = b"\xaa" * 512
        p1    = b"\xbb" * 200   # shorter last piece
        total = len(p0) + len(p1)
        t = single_file_torrent(piece_length=512, pieces_data=[p0, p1],
                                total_length=total)
        s = Storage(t, tmp_path)
        s.allocate()
        s.write_piece(0, p0)
        s.write_piece(1, p1)
        assert s.read_piece(0) == p0
        assert s.read_piece(1) == p1

    def test_out_of_range_raises(self, tmp_path):
        t = single_file_torrent(pieces_data=[b"\x00" * 512])
        s = Storage(t, tmp_path)
        s.allocate()
        with pytest.raises(StorageError, match="out of range"):
            s.write_piece(99, b"\x00" * 512)

    def test_wrong_data_length_raises(self, tmp_path):
        piece = b"\xab" * 512
        t = single_file_torrent(pieces_data=[piece])
        s = Storage(t, tmp_path)
        s.allocate()
        with pytest.raises(StorageError, match="expected"):
            s.write_piece(0, b"\xab" * 100)


# ---------------------------------------------------------------------------
# write_piece / read_piece — multi-file, including cross-file pieces
# ---------------------------------------------------------------------------

class TestMultiFileWriteRead:
    def test_each_file_written_correctly(self, tmp_path):
        piece_length = 600
        file_a_data  = b"\xaa" * 300
        file_b_data  = b"\xbb" * 300
        total        = file_a_data + file_b_data   # 600 bytes = exactly 1 piece

        t = multi_file_torrent(
            piece_length=piece_length,
            files=[
                (["a.txt"], file_a_data),
                (["b.txt"], file_b_data),
            ],
        )
        s = Storage(t, tmp_path)
        s.allocate()
        s.write_piece(0, total)

        assert (tmp_path / "mydir" / "a.txt").read_bytes() == file_a_data
        assert (tmp_path / "mydir" / "b.txt").read_bytes() == file_b_data

    def test_piece_spanning_two_files(self, tmp_path):
        # piece_length=512; file_a=300 bytes, file_b=300 bytes
        # piece 0: bytes 0-511 → 300 bytes of a.txt + 212 bytes of b.txt
        # piece 1: bytes 512-599 → remaining 88 bytes of b.txt
        file_a = b"\xaa" * 300
        file_b = b"\xbb" * 300
        total  = file_a + file_b   # 600 bytes

        t = multi_file_torrent(
            piece_length=512,
            files=[(["a.txt"], file_a), (["b.txt"], file_b)],
        )
        s = Storage(t, tmp_path)
        s.allocate()

        # Write both pieces
        s.write_piece(0, total[:512])
        s.write_piece(1, total[512:])

        # Verify file contents are exactly right
        assert (tmp_path / "mydir" / "a.txt").read_bytes() == file_a
        assert (tmp_path / "mydir" / "b.txt").read_bytes() == file_b

    def test_read_spanning_piece(self, tmp_path):
        file_a = b"\x11" * 300
        file_b = b"\x22" * 300
        total  = file_a + file_b

        t = multi_file_torrent(
            piece_length=512,
            files=[(["a.txt"], file_a), (["b.txt"], file_b)],
        )
        s = Storage(t, tmp_path)
        s.allocate()
        s.write_piece(0, total[:512])
        s.write_piece(1, total[512:])

        assert s.read_piece(0) == total[:512]
        assert s.read_piece(1) == total[512:]


# ---------------------------------------------------------------------------
# verify_piece
# ---------------------------------------------------------------------------

class TestVerifyPiece:
    def test_correct_data_passes(self, tmp_path):
        piece = b"\xde\xad" * 256
        t = single_file_torrent(pieces_data=[piece])
        s = Storage(t, tmp_path)
        assert s.verify_piece(0, piece) is True

    def test_wrong_data_fails(self, tmp_path):
        piece = b"\xde\xad" * 256
        t = single_file_torrent(pieces_data=[piece])
        s = Storage(t, tmp_path)
        assert s.verify_piece(0, b"\x00" * len(piece)) is False

    def test_one_byte_corruption_fails(self, tmp_path):
        piece = b"\xab" * 512
        t = single_file_torrent(pieces_data=[piece])
        s = Storage(t, tmp_path)
        corrupted = bytearray(piece)
        corrupted[100] ^= 0xFF
        assert s.verify_piece(0, bytes(corrupted)) is False


# ---------------------------------------------------------------------------
# is_complete
# ---------------------------------------------------------------------------

class TestIsComplete:
    def test_complete_when_all_pieces_correct(self, tmp_path):
        p0 = b"\xaa" * 512
        p1 = b"\xbb" * 512
        t  = single_file_torrent(pieces_data=[p0, p1])
        s  = Storage(t, tmp_path)
        s.allocate()
        s.write_piece(0, p0)
        s.write_piece(1, p1)
        assert s.is_complete() is True

    def test_not_complete_when_piece_missing(self, tmp_path):
        p0 = b"\xaa" * 512
        p1 = b"\xbb" * 512
        t  = single_file_torrent(pieces_data=[p0, p1])
        s  = Storage(t, tmp_path)
        s.allocate()
        s.write_piece(0, p0)
        # p1 not written — file is pre-allocated with zeros which won't match
        assert s.is_complete() is False

    def test_not_complete_when_file_missing(self, tmp_path):
        piece = b"\xaa" * 512
        t = single_file_torrent(pieces_data=[piece])
        s = Storage(t, tmp_path)
        # Don't call allocate — file doesn't exist
        assert s.is_complete() is False
