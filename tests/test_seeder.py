"""
Tests for bittorrent.seeder — Seeder and _UploadPeer.

We test without real TCP sockets by:
  - Directly calling _UploadPeer.serve() with fake StreamReader/StreamWriter pairs
  - Testing Seeder._rechoke() by inspecting which peers got choked/unchoked
  - Testing broadcast_have() with mock _UploadPeer objects
"""

from __future__ import annotations

import asyncio
import hashlib
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bittorrent.messages import (
    MSG_BITFIELD,
    MSG_CHOKE,
    MSG_HAVE,
    MSG_PIECE,
    MSG_REQUEST,
    MSG_UNCHOKE,
    encode_bitfield,
    encode_have,
    encode_handshake,
    encode_interested,
    encode_piece,
    encode_request,
    encode_unchoke,
)
from bittorrent.peer import PeerConnection, PeerError
from bittorrent.piece_manager import PieceManager
from bittorrent.seeder import MAX_UPLOAD_SLOTS, Seeder, _UploadPeer
from bittorrent.torrent import Torrent


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

INFO_HASH = bytes(range(20))
PEER_ID   = b"-BC0001-" + b"X" * 12
THEIR_ID  = b"-QT0000-" + b"Y" * 12


def make_torrent(num_pieces: int = 4, piece_length: int = 512) -> Torrent:
    total = num_pieces * piece_length
    piece_hashes = [hashlib.sha1(b"\x00" * piece_length).digest()] * num_pieces
    return Torrent(
        announce="http://t.example.com/announce",
        info_hash=INFO_HASH,
        info_hash_hex=INFO_HASH.hex(),
        name="test.bin",
        piece_length=piece_length,
        piece_hashes=piece_hashes,
        length=total,
    )


def make_pm(num_pieces: int = 4, piece_length: int = 512, all_complete: bool = True) -> PieceManager:
    total = num_pieces * piece_length
    pm = PieceManager(num_pieces, piece_length, total)
    if all_complete:
        for i in range(num_pieces):
            pm.mark_complete(i)
    return pm


def make_storage(num_pieces: int = 4, piece_length: int = 512) -> MagicMock:
    storage = MagicMock()
    storage.read_piece = MagicMock(return_value=b"\xab" * piece_length)
    return storage


class MockWriter:
    """Minimal writer that captures written data; supports get_extra_info."""

    def __init__(self, peername=("5.6.7.8", 12345)):
        self.buffer = bytearray()
        self.closed = False
        self._peername = peername

    def write(self, data: bytes) -> None:
        self.buffer.extend(data)

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass

    def get_extra_info(self, key, default=None):
        return self._peername if key == "peername" else default

    def messages_sent(self) -> list[tuple[int | None, bytes]]:
        """Parse written bytes into (msg_id, payload) tuples.

        The first 68 bytes are the handshake (not length-prefixed) and are
        skipped; everything after is parsed as length-prefixed messages.
        """
        from bittorrent.messages import HANDSHAKE_LEN
        result = []
        buf = bytes(self.buffer)
        pos = HANDSHAKE_LEN   # skip raw handshake
        if pos > len(buf):
            return result
        while pos + 4 <= len(buf):
            (length,) = struct.unpack("!I", buf[pos:pos + 4])
            pos += 4
            if length == 0:
                result.append((None, b""))
                continue
            if pos + length > len(buf):
                break
            mid  = buf[pos]
            payload = buf[pos + 1: pos + length]
            result.append((mid, payload))
            pos += length
        return result

    def sent_ids(self) -> list[int | None]:
        return [m[0] for m in self.messages_sent()]


def make_incoming_stream(extra_msgs: bytes = b"") -> tuple[asyncio.StreamReader, MockWriter]:
    """Return a (reader, writer) pair with their handshake pre-loaded."""
    their_hs = encode_handshake(INFO_HASH, THEIR_ID)
    reader = asyncio.StreamReader()
    reader.feed_data(their_hs + extra_msgs)
    reader.feed_eof()
    writer = MockWriter()
    return reader, writer


# ---------------------------------------------------------------------------
# PieceManager.bitfield_bytes
# ---------------------------------------------------------------------------

class TestBitfieldBytes:
    def test_all_complete(self):
        pm = make_pm(8, 512)
        bf = pm.bitfield_bytes()
        assert bf == bytearray(b"\xff")

    def test_none_complete(self):
        pm = make_pm(8, 512, all_complete=False)
        bf = pm.bitfield_bytes()
        assert bf == bytearray(b"\x00")

    def test_first_piece_complete(self):
        pm = make_pm(8, 512, all_complete=False)
        pm.mark_complete(0)
        bf = pm.bitfield_bytes()
        assert bf[0] == 0x80

    def test_last_piece_complete(self):
        pm = make_pm(8, 512, all_complete=False)
        pm.mark_complete(7)
        bf = pm.bitfield_bytes()
        assert bf[0] == 0x01

    def test_non_multiple_of_8(self):
        pm = make_pm(3, 512, all_complete=False)
        pm.mark_complete(0)
        pm.mark_complete(2)
        bf = pm.bitfield_bytes()
        # piece 0 = 0x80, piece 2 = 0x20
        assert bf[0] & 0x80
        assert bf[0] & 0x20
        assert not (bf[0] & 0x40)  # piece 1 not set


# ---------------------------------------------------------------------------
# _UploadPeer
# ---------------------------------------------------------------------------

class TestUploadPeerServe:
    async def test_sends_bitfield_after_handshake(self):
        # No extra messages — serve() will exit with PeerError on EOF
        reader, writer = make_incoming_stream()
        torrent = make_torrent(4)
        pm      = make_pm(4, all_complete=True)
        storage = make_storage(4)

        peer = _UploadPeer(torrent, storage, pm)
        # serve() completes handshake/bitfield/unchoke then hits EOF in read_request
        try:
            await peer.serve(reader, writer, INFO_HASH, PEER_ID, unchoke=True)
        except PeerError:
            pass

        sent_ids = writer.sent_ids()
        # After handshake, should have sent BITFIELD then UNCHOKE
        assert MSG_BITFIELD in sent_ids
        assert MSG_UNCHOKE in sent_ids

    async def test_no_unchoke_when_unchoke_false(self):
        reader, writer = make_incoming_stream()
        torrent = make_torrent(4)
        pm      = make_pm(4, all_complete=True)
        storage = make_storage(4)

        peer = _UploadPeer(torrent, storage, pm)
        try:
            await peer.serve(reader, writer, INFO_HASH, PEER_ID, unchoke=False)
        except PeerError:
            pass

        sent_ids = writer.sent_ids()
        assert MSG_BITFIELD in sent_ids
        assert MSG_UNCHOKE not in sent_ids

    async def test_serves_a_request(self):
        piece_data = b"\xcd" * 512
        # Build: INTERESTED + REQUEST for piece 0 block 0
        request_msgs = encode_interested() + encode_request(0, 0, 512)
        reader, writer = make_incoming_stream(request_msgs)

        torrent = make_torrent(4, 512)
        pm      = make_pm(4, 512, all_complete=True)
        storage = MagicMock()
        storage.read_piece = MagicMock(return_value=piece_data)

        peer = _UploadPeer(torrent, storage, pm)
        # serve() handles the request then hits EOF
        try:
            await peer.serve(reader, writer, INFO_HASH, PEER_ID, unchoke=True)
        except PeerError:
            pass

        sent_ids = writer.sent_ids()
        assert MSG_PIECE in sent_ids

    async def test_skips_request_when_piece_not_complete(self):
        request_msgs = encode_interested() + encode_request(0, 0, 512)
        reader, writer = make_incoming_stream(request_msgs)

        torrent = make_torrent(4, 512)
        pm      = make_pm(4, 512, all_complete=False)  # no pieces
        storage = MagicMock()

        peer = _UploadPeer(torrent, storage, pm)
        try:
            await peer.serve(reader, writer, INFO_HASH, PEER_ID, unchoke=True)
        except PeerError:
            pass

        sent_ids = writer.sent_ids()
        assert MSG_PIECE not in sent_ids
        storage.read_piece.assert_not_called()

    async def test_am_choking_property_before_serve(self):
        torrent = make_torrent()
        pm      = make_pm()
        storage = make_storage()
        peer    = _UploadPeer(torrent, storage, pm)
        assert peer.am_choking is True   # no conn yet

    async def test_close_sets_closed(self):
        torrent = make_torrent()
        pm      = make_pm()
        storage = make_storage()
        peer    = _UploadPeer(torrent, storage, pm)
        await peer.close()
        assert peer.closed is True


# ---------------------------------------------------------------------------
# Seeder._rechoke
# ---------------------------------------------------------------------------

class TestSeederRechoke:
    def _make_mock_peer(self, am_choking=True, peer_interested=True, closed=False):
        peer = MagicMock()
        peer.am_choking = am_choking
        peer.peer_interested = peer_interested
        peer.closed = closed
        peer.send_unchoke_safe = AsyncMock()
        peer.send_choke_safe = AsyncMock()
        return peer

    async def test_unchokes_up_to_max_slots(self):
        torrent = make_torrent()
        pm      = make_pm()
        storage = make_storage()
        seeder  = Seeder(torrent, storage, pm, info_hash=INFO_HASH, peer_id=PEER_ID)

        # Add MAX_UPLOAD_SLOTS + 2 interested peers, all currently choked
        peers = [self._make_mock_peer(am_choking=True) for _ in range(MAX_UPLOAD_SLOTS + 2)]
        seeder._peers = peers

        seeder._rechoke()

        unchoke_calls = sum(1 for p in peers if p.send_unchoke_safe.called)
        assert unchoke_calls == MAX_UPLOAD_SLOTS

    async def test_chokes_excess_peers(self):
        torrent = make_torrent()
        pm      = make_pm()
        storage = make_storage()
        seeder  = Seeder(torrent, storage, pm, info_hash=INFO_HASH, peer_id=PEER_ID)

        # MAX_UPLOAD_SLOTS + 2 peers currently unchoked
        peers = [self._make_mock_peer(am_choking=False) for _ in range(MAX_UPLOAD_SLOTS + 2)]
        seeder._peers = peers

        seeder._rechoke()

        choke_calls = sum(1 for p in peers if p.send_choke_safe.called)
        assert choke_calls == 2

    async def test_ignores_uninterested_peers(self):
        torrent = make_torrent()
        pm      = make_pm()
        storage = make_storage()
        seeder  = Seeder(torrent, storage, pm, info_hash=INFO_HASH, peer_id=PEER_ID)

        # One interested peer, many uninterested — only the one interested should be unchoked
        uninterested = [self._make_mock_peer(peer_interested=False) for _ in range(5)]
        interested   = [self._make_mock_peer(peer_interested=True, am_choking=True)]
        seeder._peers = uninterested + interested

        seeder._rechoke()

        assert interested[0].send_unchoke_safe.called
        for p in uninterested:
            assert not p.send_unchoke_safe.called

    async def test_ignores_closed_peers(self):
        torrent = make_torrent()
        pm      = make_pm()
        storage = make_storage()
        seeder  = Seeder(torrent, storage, pm, info_hash=INFO_HASH, peer_id=PEER_ID)

        closed_peer = self._make_mock_peer(closed=True)
        open_peer   = self._make_mock_peer(am_choking=True)
        seeder._peers = [closed_peer, open_peer]

        seeder._rechoke()

        assert not closed_peer.send_unchoke_safe.called


# ---------------------------------------------------------------------------
# Seeder.broadcast_have
# ---------------------------------------------------------------------------

class TestSeederBroadcastHave:
    async def test_sends_to_all_open_peers(self):
        torrent = make_torrent()
        pm      = make_pm()
        storage = make_storage()
        seeder  = Seeder(torrent, storage, pm, info_hash=INFO_HASH, peer_id=PEER_ID)

        peers = [MagicMock(closed=False) for _ in range(3)]
        for p in peers:
            p.send_have_safe = AsyncMock()
        seeder._peers = peers

        seeder.broadcast_have(5)
        # broadcast_have calls send_have_safe() (creating the coroutine) synchronously
        for p in peers:
            p.send_have_safe.assert_called_once_with(5)

    async def test_skips_closed_peers(self):
        torrent = make_torrent()
        pm      = make_pm()
        storage = make_storage()
        seeder  = Seeder(torrent, storage, pm, info_hash=INFO_HASH, peer_id=PEER_ID)

        open_peer   = MagicMock(closed=False)
        open_peer.send_have_safe = AsyncMock()
        closed_peer = MagicMock(closed=True)
        closed_peer.send_have_safe = AsyncMock()
        seeder._peers = [open_peer, closed_peer]

        seeder.broadcast_have(2)

        open_peer.send_have_safe.assert_called_once_with(2)
        closed_peer.send_have_safe.assert_not_called()


# ---------------------------------------------------------------------------
# CLI --leech flag
# ---------------------------------------------------------------------------

class TestLeechFlag:
    def test_leech_flag_present(self):
        from bittorrent.main import _parse_args
        args = _parse_args(["foo.torrent", "--leech"])
        assert args.leech is True

    def test_leech_short_flag(self):
        from bittorrent.main import _parse_args
        args = _parse_args(["foo.torrent", "-l"])
        assert args.leech is True

    def test_leech_default_false(self):
        from bittorrent.main import _parse_args
        args = _parse_args(["foo.torrent"])
        assert args.leech is False
