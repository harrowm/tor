"""
Tests for bittorrent.messages — peer wire protocol encoding/decoding.

read_message() uses asyncio.StreamReader which supports feed_data() for
in-process testing — no mocking required.
"""

import asyncio
import struct
import pytest

from bittorrent.messages import (
    BLOCK_SIZE,
    HANDSHAKE_LEN,
    MSG_BITFIELD,
    MSG_CANCEL,
    MSG_CHOKE,
    MSG_HAVE,
    MSG_INTERESTED,
    MSG_NOT_INTERESTED,
    MSG_PIECE,
    MSG_REQUEST,
    MSG_UNCHOKE,
    MessageError,
    PeerMessage,
    decode_handshake,
    encode_bitfield,
    encode_cancel,
    encode_choke,
    encode_handshake,
    encode_have,
    encode_interested,
    encode_keepalive,
    encode_not_interested,
    encode_piece,
    encode_request,
    encode_unchoke,
    read_message,
)


INFO_HASH = bytes(range(20))
PEER_ID   = b"-BC0001-" + b"X" * 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_reader(data: bytes) -> asyncio.StreamReader:
    """Return a StreamReader pre-loaded with *data*."""
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


def raw_message(msg_id: int, payload: bytes = b"") -> bytes:
    """Build a raw length-prefixed message."""
    length = 1 + len(payload)
    return struct.pack("!I", length) + bytes([msg_id]) + payload


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_block_size(self):
        assert BLOCK_SIZE == 16_384

    def test_handshake_len(self):
        assert HANDSHAKE_LEN == 68


# ---------------------------------------------------------------------------
# Handshake encode
# ---------------------------------------------------------------------------

class TestEncodeHandshake:
    def test_length(self):
        assert len(encode_handshake(INFO_HASH, PEER_ID)) == 68

    def test_pstrlen(self):
        data = encode_handshake(INFO_HASH, PEER_ID)
        assert data[0] == 19

    def test_protocol_string(self):
        data = encode_handshake(INFO_HASH, PEER_ID)
        assert data[1:20] == b"BitTorrent protocol"

    def test_reserved_bytes(self):
        data = encode_handshake(INFO_HASH, PEER_ID)
        assert data[20:28] == b"\x00" * 8

    def test_info_hash_position(self):
        data = encode_handshake(INFO_HASH, PEER_ID)
        assert data[28:48] == INFO_HASH

    def test_peer_id_position(self):
        data = encode_handshake(INFO_HASH, PEER_ID)
        assert data[48:68] == PEER_ID

    def test_wrong_info_hash_length(self):
        with pytest.raises(ValueError, match="info_hash"):
            encode_handshake(b"tooshort", PEER_ID)

    def test_wrong_peer_id_length(self):
        with pytest.raises(ValueError, match="peer_id"):
            encode_handshake(INFO_HASH, b"tooshort")


# ---------------------------------------------------------------------------
# Handshake decode
# ---------------------------------------------------------------------------

class TestDecodeHandshake:
    def test_roundtrip(self):
        data = encode_handshake(INFO_HASH, PEER_ID)
        info_hash, peer_id = decode_handshake(data)
        assert info_hash == INFO_HASH
        assert peer_id   == PEER_ID

    def test_different_info_hash(self):
        other_hash = bytes(reversed(range(20)))
        data = encode_handshake(other_hash, PEER_ID)
        info_hash, _ = decode_handshake(data)
        assert info_hash == other_hash

    def test_too_short_raises(self):
        with pytest.raises(MessageError, match="short"):
            decode_handshake(b"\x13BitTorrent protocol" + b"\x00" * 8)

    def test_wrong_pstrlen_raises(self):
        # pstrlen=18 but we pad to 68 bytes so the length check passes first
        bad = b"\x12" + b"BitTorrent protocol" + b"\x00" * 8 + INFO_HASH + PEER_ID
        with pytest.raises(MessageError, match="pstrlen"):
            decode_handshake(bad)

    def test_wrong_protocol_string_raises(self):
        bad = b"\x13" + b"BitTorrent Protocol" + b"\x00" * 8 + INFO_HASH + PEER_ID
        with pytest.raises(MessageError, match="protocol"):
            decode_handshake(bad)


# ---------------------------------------------------------------------------
# Message encoders — structure checks
# ---------------------------------------------------------------------------

class TestEncodeKeepalive:
    def test_is_four_zero_bytes(self):
        assert encode_keepalive() == b"\x00\x00\x00\x00"


class TestEncodeFixedMessages:
    """Messages with no payload: choke, unchoke, interested, not_interested."""

    @pytest.mark.parametrize("fn, expected_id", [
        (encode_choke,          MSG_CHOKE),
        (encode_unchoke,        MSG_UNCHOKE),
        (encode_interested,     MSG_INTERESTED),
        (encode_not_interested, MSG_NOT_INTERESTED),
    ])
    def test_length_prefix(self, fn, expected_id):
        data = fn()
        (length,) = struct.unpack("!I", data[:4])
        assert length == 1   # 1 byte for the ID, no payload

    @pytest.mark.parametrize("fn, expected_id", [
        (encode_choke,          MSG_CHOKE),
        (encode_unchoke,        MSG_UNCHOKE),
        (encode_interested,     MSG_INTERESTED),
        (encode_not_interested, MSG_NOT_INTERESTED),
    ])
    def test_message_id(self, fn, expected_id):
        data = fn()
        assert data[4] == expected_id

    @pytest.mark.parametrize("fn, expected_id", [
        (encode_choke,          MSG_CHOKE),
        (encode_unchoke,        MSG_UNCHOKE),
        (encode_interested,     MSG_INTERESTED),
        (encode_not_interested, MSG_NOT_INTERESTED),
    ])
    def test_total_length(self, fn, expected_id):
        assert len(fn()) == 5


class TestEncodeHave:
    def test_message_id(self):
        data = encode_have(7)
        assert data[4] == MSG_HAVE

    def test_payload_length(self):
        data = encode_have(7)
        (length,) = struct.unpack("!I", data[:4])
        assert length == 5   # 1 id + 4 index

    def test_piece_index(self):
        data = encode_have(42)
        (index,) = struct.unpack("!I", data[5:9])
        assert index == 42

    def test_zero_index(self):
        data = encode_have(0)
        (index,) = struct.unpack("!I", data[5:9])
        assert index == 0


class TestEncodeBitfield:
    def test_message_id(self):
        bf = b"\xff\x00"
        data = encode_bitfield(bf)
        assert data[4] == MSG_BITFIELD

    def test_payload_is_bitfield_bytes(self):
        bf = b"\xf0\x0f"
        data = encode_bitfield(bf)
        assert data[5:] == bf

    def test_bytearray_accepted(self):
        bf = bytearray(b"\x80")
        data = encode_bitfield(bf)
        assert data[5:] == b"\x80"


class TestEncodeRequest:
    def test_message_id(self):
        data = encode_request(0, 0, BLOCK_SIZE)
        assert data[4] == MSG_REQUEST

    def test_payload_length(self):
        data = encode_request(0, 0, BLOCK_SIZE)
        (length,) = struct.unpack("!I", data[:4])
        assert length == 13   # 1 id + 12 payload

    def test_fields(self):
        data = encode_request(3, 16384, 16384)
        piece_index, block_offset, block_length = struct.unpack("!III", data[5:])
        assert piece_index   == 3
        assert block_offset  == 16384
        assert block_length  == 16384


class TestEncodePiece:
    def test_message_id(self):
        data = encode_piece(0, 0, b"x" * 16)
        assert data[4] == MSG_PIECE

    def test_fields(self):
        payload_data = b"hello world!!!!!"
        data = encode_piece(5, 32768, payload_data)
        piece_index, block_offset = struct.unpack("!II", data[5:13])
        assert piece_index  == 5
        assert block_offset == 32768
        assert data[13:] == payload_data

    def test_payload_length(self):
        blob = b"\xab" * 100
        data = encode_piece(0, 0, blob)
        (length,) = struct.unpack("!I", data[:4])
        assert length == 1 + 8 + len(blob)


class TestEncodeCancel:
    def test_message_id(self):
        data = encode_cancel(0, 0, BLOCK_SIZE)
        assert data[4] == MSG_CANCEL

    def test_same_structure_as_request(self):
        # Same length prefix and payload, different message ID byte
        assert encode_cancel(1, 2, 3)[:4] == encode_request(1, 2, 3)[:4]  # same length
        assert encode_cancel(1, 2, 3)[5:] == encode_request(1, 2, 3)[5:]  # same payload
        assert encode_cancel(1, 2, 3)[4]  == MSG_CANCEL
        assert encode_request(1, 2, 3)[4] == MSG_REQUEST


# ---------------------------------------------------------------------------
# PeerMessage accessors
# ---------------------------------------------------------------------------

class TestPeerMessageAccessors:
    def test_keepalive_name(self):
        m = PeerMessage(None)
        assert m.name == "KEEP_ALIVE"
        assert m.is_keepalive

    def test_named_messages(self):
        assert PeerMessage(MSG_CHOKE).name == "CHOKE"
        assert PeerMessage(MSG_UNCHOKE).name == "UNCHOKE"
        assert PeerMessage(MSG_PIECE).name == "PIECE"

    def test_unknown_id(self):
        m = PeerMessage(99)
        assert "99" in m.name

    def test_have_index(self):
        m = PeerMessage(MSG_HAVE, struct.pack("!I", 17))
        assert m.have_index() == 17

    def test_have_index_bad_payload(self):
        with pytest.raises(MessageError):
            PeerMessage(MSG_HAVE, b"\x00").have_index()

    def test_request_fields(self):
        payload = struct.pack("!III", 2, 32768, 16384)
        m = PeerMessage(MSG_REQUEST, payload)
        assert m.request_fields() == (2, 32768, 16384)

    def test_request_fields_bad_payload(self):
        with pytest.raises(MessageError):
            PeerMessage(MSG_REQUEST, b"\x00" * 8).request_fields()

    def test_piece_fields(self):
        payload = struct.pack("!II", 3, 16384) + b"data"
        m = PeerMessage(MSG_PIECE, payload)
        idx, offset, data = m.piece_fields()
        assert idx    == 3
        assert offset == 16384
        assert data   == b"data"

    def test_piece_fields_too_short(self):
        with pytest.raises(MessageError):
            PeerMessage(MSG_PIECE, b"\x00" * 4).piece_fields()


# ---------------------------------------------------------------------------
# read_message — async, uses StreamReader.feed_data()
# ---------------------------------------------------------------------------

class TestReadMessage:
    async def test_keepalive(self):
        reader = make_reader(b"\x00\x00\x00\x00")
        msg = await read_message(reader)
        assert msg.is_keepalive
        assert msg.msg_id is None

    async def test_choke(self):
        reader = make_reader(raw_message(MSG_CHOKE))
        msg = await read_message(reader)
        assert msg.msg_id == MSG_CHOKE
        assert msg.payload == b""

    async def test_unchoke(self):
        reader = make_reader(raw_message(MSG_UNCHOKE))
        msg = await read_message(reader)
        assert msg.msg_id == MSG_UNCHOKE

    async def test_have(self):
        payload = struct.pack("!I", 99)
        reader = make_reader(raw_message(MSG_HAVE, payload))
        msg = await read_message(reader)
        assert msg.msg_id == MSG_HAVE
        assert msg.have_index() == 99

    async def test_piece(self):
        payload = struct.pack("!II", 5, 0) + b"\xab" * 16
        reader = make_reader(raw_message(MSG_PIECE, payload))
        msg = await read_message(reader)
        idx, offset, data = msg.piece_fields()
        assert idx    == 5
        assert offset == 0
        assert data   == b"\xab" * 16

    async def test_bitfield(self):
        bf = b"\xff\x80"
        reader = make_reader(raw_message(MSG_BITFIELD, bf))
        msg = await read_message(reader)
        assert msg.msg_id == MSG_BITFIELD
        assert msg.payload == bf

    async def test_multiple_messages_sequential(self):
        data = raw_message(MSG_CHOKE) + raw_message(MSG_UNCHOKE)
        reader = make_reader(data)
        m1 = await read_message(reader)
        m2 = await read_message(reader)
        assert m1.msg_id == MSG_CHOKE
        assert m2.msg_id == MSG_UNCHOKE

    async def test_eof_raises(self):
        reader = make_reader(b"")   # empty — immediately EOF
        with pytest.raises(EOFError):
            await read_message(reader)

    async def test_partial_length_raises(self):
        reader = make_reader(b"\x00\x00")  # only 2 of 4 length bytes
        with pytest.raises(EOFError):
            await read_message(reader)

    async def test_partial_body_raises(self):
        # length says 5 bytes but only 2 bytes of body follow
        data = struct.pack("!I", 5) + b"\x07\x00"
        reader = make_reader(data)
        with pytest.raises(EOFError):
            await read_message(reader)

    async def test_roundtrip_request(self):
        encoded = encode_request(3, 49152, BLOCK_SIZE)
        reader = make_reader(encoded)
        msg = await read_message(reader)
        assert msg.msg_id == MSG_REQUEST
        assert msg.request_fields() == (3, 49152, BLOCK_SIZE)

    async def test_roundtrip_piece(self):
        blob = b"\xde\xad\xbe\xef" * 4
        encoded = encode_piece(1, 16384, blob)
        reader = make_reader(encoded)
        msg = await read_message(reader)
        idx, offset, data = msg.piece_fields()
        assert idx    == 1
        assert offset == 16384
        assert data   == blob
