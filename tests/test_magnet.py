"""
Tests for the magnet link stack:
  - messages.py additions  (decode_handshake_full, encode_extended, etc.)
  - peer.py extension protocol (do_extension_handshake, peer_ext_id, …)
  - metadata.py (fetch_metadata via ut_metadata)
  - magnet.py  (parse_magnet, resolve_magnet, _build_torrent_bytes)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import math
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bittorrent.bencode import encode as bencode
from bittorrent.magnet import (
    MagnetError,
    MagnetLink,
    _build_torrent_bytes,
    _decode_btih,
    parse_magnet,
    resolve_magnet,
)
from bittorrent.messages import (
    EXT_PROTOCOL_RESERVED,
    MSG_EXTENDED,
    decode_handshake_full,
    encode_extended,
    encode_handshake,
    supports_extension_protocol,
)
from bittorrent.metadata import fetch_metadata
from bittorrent.peer import PeerConnection, PeerError
from bittorrent.torrent import parse as parse_torrent

INFO_HASH = bytes(range(20))
PEER_ID   = b"-BC0001-" + b"X" * 12


# ===========================================================================
# messages.py — extension protocol additions
# ===========================================================================

class TestDecodeHandshakeFull:
    def test_returns_three_values(self):
        data = encode_handshake(INFO_HASH, PEER_ID)
        result = decode_handshake_full(data)
        assert len(result) == 3

    def test_info_hash_and_peer_id(self):
        data = encode_handshake(INFO_HASH, PEER_ID)
        ih, pid, _ = decode_handshake_full(data)
        assert ih == INFO_HASH
        assert pid == PEER_ID

    def test_plain_reserved_bytes(self):
        data = encode_handshake(INFO_HASH, PEER_ID)
        _, _, reserved = decode_handshake_full(data)
        assert reserved == b"\x00" * 8

    def test_extension_reserved_bytes(self):
        data = encode_handshake(INFO_HASH, PEER_ID, reserved=EXT_PROTOCOL_RESERVED)
        _, _, reserved = decode_handshake_full(data)
        assert reserved == EXT_PROTOCOL_RESERVED

    def test_short_data_raises(self):
        from bittorrent.messages import MessageError
        with pytest.raises(MessageError):
            decode_handshake_full(b"\x00" * 30)


class TestSupportsExtensionProtocol:
    def test_plain_reserved_returns_false(self):
        assert not supports_extension_protocol(b"\x00" * 8)

    def test_ext_reserved_returns_true(self):
        assert supports_extension_protocol(EXT_PROTOCOL_RESERVED)

    def test_bit_position_exact(self):
        # Bit 20 from right = byte[5] & 0x10
        reserved = bytearray(8)
        reserved[5] = 0x10
        assert supports_extension_protocol(bytes(reserved))

    def test_other_bits_do_not_trigger(self):
        reserved = bytearray(8)
        reserved[5] = 0x08   # adjacent bit — should NOT trigger
        assert not supports_extension_protocol(bytes(reserved))

    def test_short_reserved_returns_false(self):
        assert not supports_extension_protocol(b"\x10")  # only 1 byte


class TestEncodeExtended:
    def test_message_id_20(self):
        from bittorrent.messages import read_message
        data = encode_extended(0, bencode({b"m": {}}))
        # The outer 4-byte length prefix + 1-byte msg_id
        msg_id = data[4]
        assert msg_id == MSG_EXTENDED

    def test_ext_id_at_payload_offset(self):
        data = encode_extended(3, b"hello")
        # bytes 0-3: length, byte 4: msg_id=20, byte 5: ext_id
        assert data[5] == 3

    def test_payload_follows_ext_id(self):
        payload = bencode({b"msg_type": 0, b"piece": 5})
        data = encode_extended(1, payload)
        assert data[6:] == payload

    def test_encode_handshake_reserved_kwarg(self):
        # encode_handshake with reserved= sets extension bit
        h = encode_handshake(INFO_HASH, PEER_ID, reserved=EXT_PROTOCOL_RESERVED)
        assert h[20:28] == EXT_PROTOCOL_RESERVED

    def test_encode_handshake_default_reserved_is_zeros(self):
        h = encode_handshake(INFO_HASH, PEER_ID)
        assert h[20:28] == b"\x00" * 8

    def test_encode_handshake_wrong_reserved_length(self):
        with pytest.raises(ValueError, match="reserved must be 8 bytes"):
            encode_handshake(INFO_HASH, PEER_ID, reserved=b"\x00" * 7)


# ===========================================================================
# peer.py — extension protocol methods
# ===========================================================================

def _make_conn_with_streams(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> PeerConnection:
    return PeerConnection._from_streams("1.2.3.4", 6881, reader, writer)


def _feed_message(reader: asyncio.StreamReader, msg_id: int, payload: bytes) -> None:
    """Push a length-prefixed message into a StreamReader."""
    length = 1 + len(payload)
    reader.feed_data(struct.pack("!I", length) + bytes([msg_id]) + payload)


def _feed_extended(
    reader: asyncio.StreamReader,
    ext_id: int,
    payload: bytes,
) -> None:
    """Push an extended message (id=20) into a StreamReader."""
    _feed_message(reader, MSG_EXTENDED, bytes([ext_id]) + payload)


class TestExtensionHandshake:
    async def test_sets_peer_ext_ids(self):
        reader = asyncio.StreamReader()
        writer = MagicMock(spec=asyncio.StreamWriter)
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        conn = _make_conn_with_streams(reader, writer)

        # Peer's extension handshake: they have ut_metadata at ID 3
        peer_hs = bencode({b"m": {b"ut_metadata": 3}, b"metadata_size": 512})
        _feed_extended(reader, 0, peer_hs)

        await conn.do_extension_handshake({b"ut_metadata": 1})
        assert conn.peer_ext_id(b"ut_metadata") == 3

    async def test_sets_metadata_size(self):
        reader = asyncio.StreamReader()
        writer = MagicMock(spec=asyncio.StreamWriter)
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        conn = _make_conn_with_streams(reader, writer)

        peer_hs = bencode({b"m": {b"ut_metadata": 1}, b"metadata_size": 1024})
        _feed_extended(reader, 0, peer_hs)

        await conn.do_extension_handshake({b"ut_metadata": 1})
        assert conn.metadata_size == 1024

    async def test_sends_our_extension_handshake(self):
        reader = asyncio.StreamReader()
        writer = MagicMock(spec=asyncio.StreamWriter)
        sent_data = []
        writer.write = lambda d: sent_data.append(d)
        writer.drain = AsyncMock()
        conn = _make_conn_with_streams(reader, writer)

        # Feed peer response immediately
        peer_hs = bencode({b"m": {b"ut_metadata": 1}})
        _feed_extended(reader, 0, peer_hs)

        await conn.do_extension_handshake({b"ut_metadata": 1})
        combined = b"".join(sent_data)
        # The message should contain msg_id=20 and ext_id=0
        assert bytes([MSG_EXTENDED]) in combined

    async def test_skips_bitfield_and_defers_other_messages(self):
        """BITFIELD before the ext handshake is consumed; other msgs are deferred."""
        from bittorrent.messages import MSG_BITFIELD, MSG_HAVE

        reader = asyncio.StreamReader()
        writer = MagicMock(spec=asyncio.StreamWriter)
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        conn = _make_conn_with_streams(reader, writer)

        # Push: BITFIELD, HAVE(piece=0), extension handshake
        bitfield = bytearray([0xFF])
        _feed_message(reader, MSG_BITFIELD, bytes(bitfield))
        _feed_message(reader, MSG_HAVE, struct.pack("!I", 0))
        peer_hs = bencode({b"m": {b"ut_metadata": 2}})
        _feed_extended(reader, 0, peer_hs)

        await conn.do_extension_handshake({b"ut_metadata": 1})

        assert conn.bitfield == bitfield
        # HAVE message should be deferred back into _pending
        assert len(conn._pending) == 1
        assert conn._pending[0].msg_id == MSG_HAVE

    async def test_timeout_raises_peer_error(self):
        reader = asyncio.StreamReader()
        writer = MagicMock(spec=asyncio.StreamWriter)
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        conn = _make_conn_with_streams(reader, writer)
        # No data fed — should time out
        with pytest.raises(PeerError, match="timed out"):
            await conn.do_extension_handshake({b"ut_metadata": 1}, timeout=0.05)

    def test_peer_ext_id_unknown_returns_none(self):
        reader = asyncio.StreamReader()
        writer = MagicMock(spec=asyncio.StreamWriter)
        conn = _make_conn_with_streams(reader, writer)
        assert conn.peer_ext_id(b"ut_metadata") is None

    async def test_read_extension_payload_skips_non_extended(self):
        from bittorrent.messages import MSG_HAVE

        reader = asyncio.StreamReader()
        writer = MagicMock(spec=asyncio.StreamWriter)
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        conn = _make_conn_with_streams(reader, writer)

        # Push HAVE, then an extended message
        _feed_message(reader, MSG_HAVE, struct.pack("!I", 5))
        _feed_extended(reader, 7, b"payload")

        ext_id, payload = await conn.read_extension_payload()
        assert ext_id == 7
        assert payload == b"payload"

    async def test_send_extension_writes_correct_bytes(self):
        reader = asyncio.StreamReader()
        writer = MagicMock(spec=asyncio.StreamWriter)
        sent = []
        writer.write = lambda d: sent.append(d)
        writer.drain = AsyncMock()
        conn = _make_conn_with_streams(reader, writer)

        await conn.send_extension(5, b"hello")
        combined = b"".join(sent)
        assert combined[4] == MSG_EXTENDED
        assert combined[5] == 5
        assert combined[6:] == b"hello"

    async def test_open_with_extension_protocol_sets_flag(self):
        """open(extension_protocol=True) sets reserved bit in handshake."""
        info_hash = bytes(range(20))
        peer_id   = b"-BC0001-" + b"Z" * 12
        fake_peer_id = b"-BX0001-" + b"A" * 12

        reader = asyncio.StreamReader()
        writer = MagicMock(spec=asyncio.StreamWriter)
        sent = []
        writer.write = lambda d: sent.append(d)
        writer.drain = AsyncMock()
        writer.is_closing = MagicMock(return_value=False)

        # Feed peer's handshake — WITH extension bit set
        peer_hs = encode_handshake(info_hash, fake_peer_id,
                                   reserved=EXT_PROTOCOL_RESERVED)
        reader.feed_data(peer_hs)
        # No bitfield follows

        with patch("asyncio.open_connection", return_value=(reader, writer)):
            conn = await PeerConnection.open(
                "1.2.3.4", 6881, info_hash, peer_id,
                extension_protocol=True,
            )

        assert conn.remote_supports_extensions is True
        # Verify our sent handshake has the extension bit set
        our_hs = b"".join(sent)
        assert our_hs[20:28] == EXT_PROTOCOL_RESERVED

    async def test_open_without_extension_protocol_no_ext_bit(self):
        """open() without extension_protocol sends plain reserved bytes."""
        info_hash = bytes(range(20))
        peer_id   = b"-BC0001-" + b"Z" * 12
        fake_peer_id = b"-BX0001-" + b"A" * 12

        reader = asyncio.StreamReader()
        writer = MagicMock(spec=asyncio.StreamWriter)
        sent = []
        writer.write = lambda d: sent.append(d)
        writer.drain = AsyncMock()
        writer.is_closing = MagicMock(return_value=False)

        reader.feed_data(encode_handshake(info_hash, fake_peer_id))

        with patch("asyncio.open_connection", return_value=(reader, writer)):
            conn = await PeerConnection.open("1.2.3.4", 6881, info_hash, peer_id)

        assert conn.remote_supports_extensions is False
        our_hs = b"".join(sent)
        assert our_hs[20:28] == b"\x00" * 8


# ===========================================================================
# metadata.py — fetch_metadata
# ===========================================================================

def _make_info_dict(name: str = "test", size: int = 512) -> dict:
    piece_hash = hashlib.sha1(b"\xab" * size).digest()
    return {
        b"name": name.encode(),
        b"piece length": size,
        b"pieces": piece_hash,
        b"length": size,
    }


def _make_metadata_conn(
    info_bytes: bytes,
    *,
    ut_id: int = 1,
    piece_size: int = 16_384,
) -> PeerConnection:
    """Return a PeerConnection stub already set up for ut_metadata exchange."""
    reader = asyncio.StreamReader()
    writer = MagicMock(spec=asyncio.StreamWriter)
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    conn = PeerConnection._from_streams("1.2.3.4", 6881, reader, writer)
    conn._peer_ext_ids = {b"ut_metadata": ut_id}
    conn.metadata_size = len(info_bytes)

    # Pre-load reader with data responses for each piece
    num_pieces = math.ceil(len(info_bytes) / piece_size)
    for i in range(num_pieces):
        start = i * piece_size
        chunk = info_bytes[start : start + piece_size]
        header = bencode({b"msg_type": 1, b"piece": i, b"total_size": len(info_bytes)})
        _feed_extended(reader, ut_id, header + chunk)

    return conn


class TestFetchMetadata:
    async def test_single_piece_success(self):
        info_dict = _make_info_dict(size=512)
        info_bytes = bencode(info_dict)
        info_hash = hashlib.sha1(info_bytes).digest()
        conn = _make_metadata_conn(info_bytes)

        result = await fetch_metadata(conn, info_hash)
        assert result == info_bytes

    async def test_multi_piece_success(self):
        # Build an info dict big enough to span multiple 16 KiB pieces
        piece_hashes = hashlib.sha1(b"\xcd" * 512).digest() * 4
        info_dict = {
            b"name": b"big",
            b"piece length": 512,
            b"pieces": piece_hashes,
            b"length": 512 * 4,
        }
        info_bytes = bencode(info_dict)
        # Pad to more than one metadata piece (need > 16 KiB of info bytes)
        # For a realistic test, use many piece hashes to bloat the info dict
        big_pieces = hashlib.sha1(b"\xaa").digest() * 1000   # 20 000 bytes
        info_dict[b"pieces"] = big_pieces
        info_bytes = bencode(info_dict)
        info_hash = hashlib.sha1(info_bytes).digest()
        conn = _make_metadata_conn(info_bytes)

        result = await fetch_metadata(conn, info_hash)
        assert result == info_bytes

    async def test_hash_mismatch_raises(self):
        info_dict = _make_info_dict()
        info_bytes = bencode(info_dict)
        wrong_hash = b"\x00" * 20
        conn = _make_metadata_conn(info_bytes)

        with pytest.raises(PeerError, match="SHA-1 mismatch"):
            await fetch_metadata(conn, wrong_hash)

    async def test_no_ut_metadata_raises(self):
        reader = asyncio.StreamReader()
        writer = MagicMock(spec=asyncio.StreamWriter)
        writer.drain = AsyncMock()
        conn = PeerConnection._from_streams("1.2.3.4", 6881, reader, writer)
        # _peer_ext_ids is empty — peer_ext_id returns None

        with pytest.raises(PeerError, match="ut_metadata"):
            await fetch_metadata(conn, b"\x00" * 20)

    async def test_no_metadata_size_raises(self):
        reader = asyncio.StreamReader()
        writer = MagicMock(spec=asyncio.StreamWriter)
        writer.drain = AsyncMock()
        conn = PeerConnection._from_streams("1.2.3.4", 6881, reader, writer)
        conn._peer_ext_ids = {b"ut_metadata": 1}
        conn.metadata_size = 0

        with pytest.raises(PeerError, match="metadata_size"):
            await fetch_metadata(conn, b"\x00" * 20)

    async def test_reject_raises(self):
        reader = asyncio.StreamReader()
        writer = MagicMock(spec=asyncio.StreamWriter)
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        conn = PeerConnection._from_streams("1.2.3.4", 6881, reader, writer)
        conn._peer_ext_ids = {b"ut_metadata": 1}
        conn.metadata_size = 100

        # Feed a reject message
        reject = bencode({b"msg_type": 2, b"piece": 0})
        _feed_extended(reader, 1, reject)

        with pytest.raises(PeerError, match="rejected"):
            await fetch_metadata(conn, b"\x00" * 20)

    async def test_skips_other_ext_ids(self):
        """Messages for a different extension ID are ignored."""
        info_dict = _make_info_dict(size=512)
        info_bytes = bencode(info_dict)
        info_hash = hashlib.sha1(info_bytes).digest()

        reader = asyncio.StreamReader()
        writer = MagicMock(spec=asyncio.StreamWriter)
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        conn = PeerConnection._from_streams("1.2.3.4", 6881, reader, writer)
        conn._peer_ext_ids = {b"ut_metadata": 2}
        conn.metadata_size = len(info_bytes)

        # Push a message for a different extension (id=99), then the real one
        _feed_extended(reader, 99, b"noise")
        header = bencode({b"msg_type": 1, b"piece": 0, b"total_size": len(info_bytes)})
        _feed_extended(reader, 2, header + info_bytes)

        result = await fetch_metadata(conn, info_hash)
        assert result == info_bytes


# ===========================================================================
# magnet.py — parse_magnet / _decode_btih
# ===========================================================================

class TestDecodeBtih:
    def test_hex_40_chars(self):
        h = "a1" * 20
        result = _decode_btih(h)
        assert result == bytes.fromhex(h)

    def test_hex_case_insensitive(self):
        h = "A1" * 20
        result = _decode_btih(h)
        assert result == bytes.fromhex(h)

    def test_base32_32_chars(self):
        raw = bytes(range(20))
        b32 = base64.b32encode(raw).decode()
        result = _decode_btih(b32)
        assert result == raw

    def test_base32_lowercase_accepted(self):
        raw = bytes(range(20))
        b32 = base64.b32encode(raw).decode().lower()
        result = _decode_btih(b32)
        assert result == raw

    def test_invalid_length_raises(self):
        with pytest.raises(MagnetError, match="40 hex"):
            _decode_btih("abc")

    def test_invalid_hex_raises(self):
        with pytest.raises(MagnetError, match="hex"):
            _decode_btih("Z" * 40)

    def test_invalid_base32_raises(self):
        with pytest.raises(MagnetError, match="base32"):
            _decode_btih("!" * 32)


class TestParseMagnet:
    _HEX_HASH = "a" * 40
    _INFO_HASH = bytes.fromhex(_HEX_HASH)

    def _uri(self, extra: str = "") -> str:
        return f"magnet:?xt=urn:btih:{self._HEX_HASH}{extra}"

    def test_info_hash_extracted(self):
        result = parse_magnet(self._uri())
        assert result.info_hash == self._INFO_HASH

    def test_no_name_defaults_to_none(self):
        result = parse_magnet(self._uri())
        assert result.name is None

    def test_name_decoded(self):
        result = parse_magnet(self._uri("&dn=My+Torrent"))
        assert result.name == "My Torrent"

    def test_single_tracker(self):
        result = parse_magnet(self._uri("&tr=udp%3A%2F%2Ftracker.example.com%3A6969"))
        assert len(result.trackers) == 1
        assert result.trackers[0] == "udp://tracker.example.com:6969"

    def test_multiple_trackers(self):
        uri = self._uri("&tr=http://t1.example.com&tr=http://t2.example.com")
        result = parse_magnet(uri)
        assert len(result.trackers) == 2

    def test_no_trackers(self):
        result = parse_magnet(self._uri())
        assert result.trackers == []

    def test_not_magnet_raises(self):
        with pytest.raises(MagnetError, match="Not a magnet"):
            parse_magnet("http://example.com/file.torrent")

    def test_missing_xt_raises(self):
        with pytest.raises(MagnetError, match="xt=urn:btih"):
            parse_magnet("magnet:?dn=name")

    def test_base32_hash(self):
        raw = bytes(range(20))
        b32 = base64.b32encode(raw).decode()
        result = parse_magnet(f"magnet:?xt=urn:btih:{b32}")
        assert result.info_hash == raw

    def test_info_hash_hex_property(self):
        result = parse_magnet(self._uri())
        assert result.info_hash_hex == self._HEX_HASH


# ===========================================================================
# magnet.py — _build_torrent_bytes
# ===========================================================================

class TestBuildTorrentBytes:
    def _make_info_bytes(self) -> bytes:
        piece_hash = hashlib.sha1(b"\xab" * 512).digest()
        return bencode({
            b"length": 512,
            b"name": b"test.bin",
            b"piece length": 512,
            b"pieces": piece_hash,
        })

    def test_parseable_by_torrent_parse(self):
        info_bytes = self._make_info_bytes()
        torrent_bytes = _build_torrent_bytes(info_bytes, "http://tracker.example.com")
        torrent = parse_torrent(torrent_bytes)
        assert torrent.name == "test.bin"
        assert torrent.total_length == 512

    def test_info_hash_is_preserved(self):
        """The info_hash computed from _build_torrent_bytes matches sha1(info_bytes)."""
        info_bytes = self._make_info_bytes()
        expected_hash = hashlib.sha1(info_bytes).digest()
        torrent_bytes = _build_torrent_bytes(info_bytes, "http://tracker.example.com")
        torrent = parse_torrent(torrent_bytes)
        assert torrent.info_hash == expected_hash

    def test_empty_announce_is_ok(self):
        info_bytes = self._make_info_bytes()
        torrent_bytes = _build_torrent_bytes(info_bytes, "")
        torrent = parse_torrent(torrent_bytes)
        assert torrent.announce == ""

    def test_announce_url_stored(self):
        info_bytes = self._make_info_bytes()
        torrent_bytes = _build_torrent_bytes(info_bytes, "http://example.com/ann")
        torrent = parse_torrent(torrent_bytes)
        assert torrent.announce == "http://example.com/ann"


# ===========================================================================
# magnet.py — resolve_magnet (mocked)
# ===========================================================================

class TestResolveMagnet:
    _INFO_HASH = bytes(range(20))

    def _make_info_bytes(self) -> bytes:
        piece_hash = hashlib.sha1(b"\xab" * 512).digest()
        return bencode({
            b"length": 512,
            b"name": b"test.bin",
            b"piece length": 512,
            b"pieces": piece_hash,
        })

    def _make_magnet(self) -> MagnetLink:
        return MagnetLink(
            info_hash=self._INFO_HASH,
            name="test",
            trackers=["udp://tracker.example.com:6969/announce"],
        )

    async def test_successful_resolution(self, monkeypatch):
        info_bytes = self._make_info_bytes()
        real_hash = hashlib.sha1(info_bytes).digest()
        magnet = MagnetLink(
            info_hash=real_hash,
            name="test",
            trackers=["udp://tracker.example.com:6969/announce"],
        )

        from bittorrent.tracker import TrackerResponse
        mock_announce = AsyncMock(return_value=TrackerResponse(
            interval=1800, peers=[("1.2.3.4", 6881)]
        ))
        mock_metadata = AsyncMock(return_value=info_bytes)

        monkeypatch.setattr("bittorrent.magnet.tracker_announce", mock_announce)
        monkeypatch.setattr("bittorrent.magnet._metadata_from_peer", mock_metadata)

        torrent = await resolve_magnet(magnet, PEER_ID)
        assert torrent.name == "test.bin"

    async def test_no_peers_raises_magnet_error(self, monkeypatch):
        from bittorrent.tracker import TrackerResponse
        mock_announce = AsyncMock(return_value=TrackerResponse(interval=1800, peers=[]))
        monkeypatch.setattr("bittorrent.magnet.tracker_announce", mock_announce)
        monkeypatch.setattr("bittorrent.magnet._dht_get_peers", AsyncMock(return_value=[]))

        with pytest.raises(MagnetError, match="No peers"):
            await resolve_magnet(self._make_magnet(), PEER_ID)

    async def test_tracker_failure_raises_magnet_error(self, monkeypatch):
        from bittorrent.tracker import TrackerError
        mock_announce = AsyncMock(side_effect=TrackerError("down"))
        monkeypatch.setattr("bittorrent.magnet.tracker_announce", mock_announce)
        # Suppress DHT fallback so the test stays deterministic
        monkeypatch.setattr("bittorrent.magnet._dht_get_peers", AsyncMock(return_value=[]))

        with pytest.raises(MagnetError, match="No peers"):
            await resolve_magnet(self._make_magnet(), PEER_ID)

    async def test_all_peers_fail_raises_magnet_error(self, monkeypatch):
        from bittorrent.tracker import TrackerResponse
        mock_announce = AsyncMock(return_value=TrackerResponse(
            interval=1800, peers=[("1.2.3.4", 6881), ("5.6.7.8", 6882)]
        ))
        mock_metadata = AsyncMock(side_effect=PeerError("connection refused"))
        monkeypatch.setattr("bittorrent.magnet.tracker_announce", mock_announce)
        monkeypatch.setattr("bittorrent.magnet._metadata_from_peer", mock_metadata)

        with pytest.raises(MagnetError, match="Could not fetch"):
            await resolve_magnet(self._make_magnet(), PEER_ID)

    async def test_first_peer_fails_second_succeeds(self, monkeypatch):
        info_bytes = self._make_info_bytes()
        real_hash = hashlib.sha1(info_bytes).digest()
        magnet = MagnetLink(
            info_hash=real_hash,
            trackers=["http://tracker.example.com/announce"],
        )

        from bittorrent.tracker import TrackerResponse
        mock_announce = AsyncMock(return_value=TrackerResponse(
            interval=1800, peers=[("bad.host", 1), ("1.2.3.4", 6881)]
        ))
        call_count = 0

        async def mock_metadata(host, port, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise PeerError("first peer failed")
            return info_bytes

        monkeypatch.setattr("bittorrent.magnet.tracker_announce", mock_announce)
        monkeypatch.setattr("bittorrent.magnet._metadata_from_peer", mock_metadata)

        torrent = await resolve_magnet(magnet, PEER_ID)
        assert torrent.name == "test.bin"
        assert call_count == 2

    async def test_multiple_trackers_all_called(self, monkeypatch):
        from bittorrent.tracker import TrackerError, TrackerResponse
        calls = []

        async def mock_announce(url, *args, **kwargs):
            calls.append(url)
            if url == "http://t1.example.com":
                raise TrackerError("t1 down")
            return TrackerResponse(interval=1800, peers=[("1.2.3.4", 6881)])

        info_bytes = self._make_info_bytes()
        real_hash = hashlib.sha1(info_bytes).digest()
        magnet = MagnetLink(
            info_hash=real_hash,
            trackers=["http://t1.example.com", "http://t2.example.com"],
        )
        monkeypatch.setattr("bittorrent.magnet.tracker_announce", mock_announce)
        monkeypatch.setattr(
            "bittorrent.magnet._metadata_from_peer",
            AsyncMock(return_value=info_bytes),
        )

        await resolve_magnet(magnet, PEER_ID)
        assert "http://t1.example.com" in calls
        assert "http://t2.example.com" in calls


# ===========================================================================
# main.py — magnet URI dispatch
# ===========================================================================

class TestMainMagnetDispatch:
    async def test_magnet_uri_triggers_magnet_flow(self, tmp_path, monkeypatch):
        from bittorrent.main import _parse_args, _run
        import hashlib
        from bittorrent.bencode import encode as bencode

        piece_hash = hashlib.sha1(b"\xab" * 512).digest()
        info_bytes = bencode({
            b"length": 512,
            b"name": b"test.bin",
            b"piece length": 512,
            b"pieces": piece_hash,
        })
        info_hash = hashlib.sha1(info_bytes).digest()
        hex_hash = info_hash.hex()

        from bittorrent.torrent import parse as parse_torrent
        torrent_bytes = _build_torrent_bytes(info_bytes, "http://t.example.com")
        fake_torrent = parse_torrent(torrent_bytes)

        from bittorrent.tracker import TrackerResponse
        monkeypatch.setattr(
            "bittorrent.main.parse_magnet",
            MagicMock(return_value=MagnetLink(
                info_hash=info_hash, trackers=["http://t.example.com"]
            )),
        )
        monkeypatch.setattr(
            "bittorrent.main.resolve_magnet",
            AsyncMock(return_value=fake_torrent),
        )
        monkeypatch.setattr(
            "bittorrent.main.announce",
            AsyncMock(return_value=TrackerResponse(
                interval=1800, peers=[("1.2.3.4", 6881)]
            )),
        )
        with patch("bittorrent.main.PeerManager") as MockPM:
            inst = AsyncMock()
            inst.run = AsyncMock()
            MockPM.return_value = inst
            with patch("bittorrent.main.Storage.is_complete", return_value=False):
                args = _parse_args([f"magnet:?xt=urn:btih:{hex_hash}", "-o", str(tmp_path)])
                code = await _run(args)

        assert code == 0

    async def test_invalid_magnet_returns_1(self, tmp_path):
        from bittorrent.main import _parse_args, _run
        args = _parse_args(["magnet:?dn=no_xt_field", "-o", str(tmp_path)])
        code = await _run(args)
        assert code == 1
