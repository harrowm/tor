"""
Tests for bittorrent.peer — PeerConnection handshake and piece download.

No real network connections. We inject asyncio.StreamReader (fed with
pre-canned bytes) and a MockWriter (captures written bytes) directly into
PeerConnection._from_streams().
"""

import asyncio
import hashlib
import struct
import pytest

from bittorrent.messages import (
    BLOCK_SIZE,
    MSG_BITFIELD,
    MSG_CHOKE,
    MSG_HAVE,
    MSG_INTERESTED,
    MSG_PIECE,
    MSG_REQUEST,
    MSG_UNCHOKE,
    encode_bitfield,
    encode_choke,
    encode_handshake,
    encode_have,
    encode_interested,
    encode_piece,
    encode_unchoke,
)
from bittorrent.peer import PeerConnection, PeerError, _block_spans


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class MockWriter:
    """Captures bytes written by PeerConnection without a real socket."""

    def __init__(self):
        self.buffer = bytearray()
        self.closed = False

    def write(self, data: bytes) -> None:
        self.buffer.extend(data)

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass

    # ---- inspection helpers ----

    def messages_sent(self) -> list[tuple[int | None, bytes]]:
        """Parse the write buffer into (msg_id, payload) tuples."""
        result = []
        buf = bytes(self.buffer)
        pos = 0
        while pos + 4 <= len(buf):
            (length,) = struct.unpack("!I", buf[pos:pos + 4])
            pos += 4
            if length == 0:
                result.append((None, b""))
                continue
            if pos + length > len(buf):
                break
            msg_id  = buf[pos]
            payload = buf[pos + 1: pos + length]
            result.append((msg_id, payload))
            pos += length
        return result

    def sent_ids(self) -> list[int | None]:
        return [m[0] for m in self.messages_sent()]


def make_reader(*chunks: bytes) -> asyncio.StreamReader:
    """Return a StreamReader pre-loaded with the concatenation of *chunks*."""
    reader = asyncio.StreamReader()
    for chunk in chunks:
        reader.feed_data(chunk)
    reader.feed_eof()
    return reader


def make_peer(reader: asyncio.StreamReader, writer: MockWriter) -> PeerConnection:
    return PeerConnection._from_streams("1.2.3.4", 6881, reader, writer)


INFO_HASH = bytes(range(20))
PEER_ID   = b"-BC0001-" + b"X" * 12
THEIR_ID  = b"-QT0000-" + b"Y" * 12


# ---------------------------------------------------------------------------
# _block_spans helper
# ---------------------------------------------------------------------------

class TestBlockSpans:
    def test_exact_one_block(self):
        spans = _block_spans(BLOCK_SIZE)
        assert spans == [(0, BLOCK_SIZE)]

    def test_two_full_blocks(self):
        spans = _block_spans(BLOCK_SIZE * 2)
        assert spans == [(0, BLOCK_SIZE), (BLOCK_SIZE, BLOCK_SIZE)]

    def test_partial_last_block(self):
        spans = _block_spans(BLOCK_SIZE + 100)
        assert spans == [(0, BLOCK_SIZE), (BLOCK_SIZE, 100)]

    def test_smaller_than_block(self):
        spans = _block_spans(512)
        assert spans == [(0, 512)]

    def test_covers_full_length(self):
        piece_len = BLOCK_SIZE * 3 + 777
        spans = _block_spans(piece_len)
        assert sum(l for _, l in spans) == piece_len

    def test_offsets_are_contiguous(self):
        spans = _block_spans(BLOCK_SIZE * 4)
        for i, (offset, length) in enumerate(spans):
            assert offset == i * BLOCK_SIZE


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------

class TestHandshake:
    async def test_sends_our_handshake(self):
        their_hs = encode_handshake(INFO_HASH, THEIR_ID)
        writer = MockWriter()
        peer = make_peer(make_reader(their_hs), writer)
        await peer._handshake(INFO_HASH, PEER_ID)

        sent = bytes(writer.buffer)
        # After handshake (68 bytes) we immediately send INTERESTED (5 bytes)
        assert len(sent) == 68 + 5
        assert sent[1:20] == b"BitTorrent protocol"
        assert sent[28:48] == INFO_HASH
        assert sent[48:68] == PEER_ID

    async def test_stores_remote_peer_id(self):
        their_hs = encode_handshake(INFO_HASH, THEIR_ID)
        writer = MockWriter()
        peer = make_peer(make_reader(their_hs), writer)
        await peer._handshake(INFO_HASH, PEER_ID)
        assert peer.remote_peer_id == THEIR_ID

    async def test_wrong_info_hash_raises(self):
        wrong_hash = bytes(reversed(range(20)))
        their_hs = encode_handshake(wrong_hash, THEIR_ID)
        writer = MockWriter()
        peer = make_peer(make_reader(their_hs), writer)
        with pytest.raises(PeerError, match="info_hash mismatch"):
            await peer._handshake(INFO_HASH, PEER_ID)

    async def test_truncated_handshake_raises(self):
        writer = MockWriter()
        peer = make_peer(make_reader(b"\x13BitTorrent protocol"), writer)
        with pytest.raises(PeerError):
            await peer._handshake(INFO_HASH, PEER_ID)

    async def test_connection_reset_during_handshake_raises_peer_error(self):
        """ConnectionResetError (OSError subclass) during handshake → PeerError."""
        class ResetReader:
            async def readexactly(self, n):
                raise ConnectionResetError(54, "Connection reset by peer")

        writer = MockWriter()
        peer = PeerConnection("1.2.3.4", 6881)
        peer._reader = ResetReader()
        peer._writer = writer
        with pytest.raises(PeerError, match="Handshake failed"):
            await peer._handshake(INFO_HASH, PEER_ID)

    async def test_reads_bitfield_after_handshake(self):
        their_hs = encode_handshake(INFO_HASH, THEIR_ID)
        bf_bytes  = b"\xff\x80"
        bf_msg    = encode_bitfield(bf_bytes)
        writer = MockWriter()
        peer = make_peer(make_reader(their_hs + bf_msg), writer)
        await peer._handshake(INFO_HASH, PEER_ID)
        assert peer.bitfield == bytearray(bf_bytes)

    async def test_no_bitfield_is_fine(self):
        their_hs = encode_handshake(INFO_HASH, THEIR_ID)
        writer = MockWriter()
        peer = make_peer(make_reader(their_hs), writer)
        await peer._handshake(INFO_HASH, PEER_ID)   # should not raise
        assert peer.bitfield == bytearray()

    async def test_non_bitfield_message_preserved(self):
        """A HAVE message after handshake must not be swallowed."""
        their_hs = encode_handshake(INFO_HASH, THEIR_ID)
        have_msg = encode_have(5)
        writer = MockWriter()
        peer = make_peer(make_reader(their_hs + have_msg), writer)
        await peer._handshake(INFO_HASH, PEER_ID)
        # Non-BITFIELD messages go into _pending; _read_next() drains it first
        msg = await peer._read_next()
        assert msg.msg_id == MSG_HAVE


# ---------------------------------------------------------------------------
# has_piece / bitfield
# ---------------------------------------------------------------------------

class TestHasPiece:
    def _peer_with_bitfield(self, bf: bytes) -> PeerConnection:
        peer = PeerConnection("h", 1)
        peer.bitfield = bytearray(bf)
        return peer

    def test_bit_set(self):
        # \x80 = 10000000 → piece 0 is set
        peer = self._peer_with_bitfield(b"\x80")
        assert peer.has_piece(0) is True

    def test_bit_clear(self):
        peer = self._peer_with_bitfield(b"\x80")
        assert peer.has_piece(1) is False

    def test_all_set(self):
        peer = self._peer_with_bitfield(b"\xff\xff")
        for i in range(16):
            assert peer.has_piece(i) is True

    def test_none_set(self):
        peer = self._peer_with_bitfield(b"\x00\x00")
        for i in range(16):
            assert peer.has_piece(i) is False

    def test_empty_bitfield(self):
        peer = self._peer_with_bitfield(b"")
        assert peer.has_piece(0) is False

    def test_out_of_range(self):
        peer = self._peer_with_bitfield(b"\xff")
        assert peer.has_piece(100) is False

    def test_second_byte_first_bit(self):
        # \x00\x80 → piece 8 set, piece 0 clear
        peer = self._peer_with_bitfield(b"\x00\x80")
        assert peer.has_piece(0) is False
        assert peer.has_piece(8) is True


# ---------------------------------------------------------------------------
# download_piece
# ---------------------------------------------------------------------------

def make_piece_response(piece_index: int, piece_length: int, data: bytes) -> bytes:
    """Build all the PIECE messages for a piece (one per block)."""
    result = b""
    spans = _block_spans(piece_length)
    offset_in_data = 0
    for block_offset, block_length in spans:
        block_data = data[offset_in_data: offset_in_data + block_length]
        result += encode_piece(piece_index, block_offset, block_data)
        offset_in_data += block_length
    return result


def make_valid_piece(length: int) -> tuple[bytes, bytes]:
    """Return (piece_data, sha1_hash) for a piece of *length* bytes."""
    data = bytes(range(256)) * (length // 256 + 1)
    data = data[:length]
    return data, hashlib.sha1(data).digest()


class TestDownloadPiece:
    async def _download(
        self,
        peer_responses: bytes,
        piece_index: int = 0,
        piece_length: int = BLOCK_SIZE,
        piece_hash: bytes | None = None,
    ) -> tuple[bytes, MockWriter]:
        data, valid_hash = make_valid_piece(piece_length)
        if piece_hash is None:
            piece_hash = valid_hash

        unchoke = encode_unchoke()
        reader  = make_reader(unchoke + peer_responses)
        writer  = MockWriter()
        peer    = make_peer(reader, writer)
        result  = await peer.download_piece(piece_index, piece_length, piece_hash)
        return result, writer

    async def test_single_block_piece(self):
        data, h = make_valid_piece(BLOCK_SIZE)
        responses = encode_piece(0, 0, data)
        result, _ = await self._download(responses, piece_length=BLOCK_SIZE, piece_hash=h)
        assert result == data

    async def test_multi_block_piece(self):
        piece_len = BLOCK_SIZE * 3
        data, h = make_valid_piece(piece_len)
        responses = make_piece_response(0, piece_len, data)
        result, _ = await self._download(responses, piece_length=piece_len, piece_hash=h)
        assert result == data

    async def test_partial_last_block(self):
        piece_len = BLOCK_SIZE + 500
        data, h = make_valid_piece(piece_len)
        responses = make_piece_response(0, piece_len, data)
        result, _ = await self._download(responses, piece_length=piece_len, piece_hash=h)
        assert result == data

    async def test_sends_interested_first(self):
        data, h = make_valid_piece(BLOCK_SIZE)
        responses = encode_piece(0, 0, data)
        _, writer = await self._download(responses)
        assert writer.sent_ids()[0] == MSG_INTERESTED

    async def test_sends_request_per_block(self):
        piece_len = BLOCK_SIZE * 2
        data, h = make_valid_piece(piece_len)
        responses = make_piece_response(0, piece_len, data)
        _, writer = await self._download(
            responses, piece_length=piece_len, piece_hash=h
        )
        ids = writer.sent_ids()
        request_count = ids.count(MSG_REQUEST)
        assert request_count == 2

    async def test_request_fields_are_correct(self):
        piece_len = BLOCK_SIZE * 2
        data, h = make_valid_piece(piece_len)
        responses = make_piece_response(3, piece_len, data)   # piece_index must match
        _, writer = await self._download(
            responses, piece_index=3, piece_length=piece_len, piece_hash=h
        )
        msgs = writer.messages_sent()
        requests = [(m_id, payload) for m_id, payload in msgs if m_id == MSG_REQUEST]
        # First request: piece 3, offset 0, BLOCK_SIZE
        idx, off, length = struct.unpack("!III", requests[0][1])
        assert idx    == 3
        assert off    == 0
        assert length == BLOCK_SIZE
        # Second request: piece 3, offset BLOCK_SIZE, BLOCK_SIZE
        idx, off, length = struct.unpack("!III", requests[1][1])
        assert idx    == 3
        assert off    == BLOCK_SIZE
        assert length == BLOCK_SIZE

    async def test_hash_mismatch_raises(self):
        data = b"\x00" * BLOCK_SIZE
        wrong_hash = b"\xff" * 20
        responses = encode_piece(0, 0, data)
        unchoke = encode_unchoke()
        reader  = make_reader(unchoke + responses)
        writer  = MockWriter()
        peer    = make_peer(reader, writer)
        with pytest.raises(PeerError, match="hash mismatch"):
            await peer.download_piece(0, BLOCK_SIZE, wrong_hash)

    async def test_choke_during_download_raises(self):
        # Peer sends unchoke then immediately chokes us
        choke = encode_choke()
        unchoke = encode_unchoke()
        reader  = make_reader(unchoke + choke)
        writer  = MockWriter()
        peer    = make_peer(reader, writer)
        _, h = make_valid_piece(BLOCK_SIZE)
        with pytest.raises(PeerError, match="choked"):
            await peer.download_piece(0, BLOCK_SIZE, h)

    async def test_out_of_order_blocks_assembled_correctly(self):
        """Blocks arriving out of order must still produce the right piece."""
        piece_len = BLOCK_SIZE * 3
        data, h = make_valid_piece(piece_len)

        # Send blocks in reverse order: 2, 1, 0
        responses = (
            encode_piece(0, BLOCK_SIZE * 2, data[BLOCK_SIZE * 2:])
            + encode_piece(0, BLOCK_SIZE * 1, data[BLOCK_SIZE: BLOCK_SIZE * 2])
            + encode_piece(0, 0,              data[:BLOCK_SIZE])
        )
        unchoke = encode_unchoke()
        reader  = make_reader(unchoke + responses)
        writer  = MockWriter()
        peer    = make_peer(reader, writer)
        result  = await peer.download_piece(0, piece_len, h)
        assert result == data

    async def test_have_messages_ignored_during_download(self):
        """HAVE messages interspersed with PIECE messages must not stall download."""
        data, h = make_valid_piece(BLOCK_SIZE)
        have    = encode_have(7)
        piece   = encode_piece(0, 0, data)
        unchoke = encode_unchoke()
        reader  = make_reader(unchoke + have + piece)
        writer  = MockWriter()
        peer    = make_peer(reader, writer)
        result  = await peer.download_piece(0, BLOCK_SIZE, h)
        assert result == data

    async def test_eof_during_block_receive_raises_peer_error(self):
        """If the connection closes while waiting for block data, PeerError is raised."""
        _, h = make_valid_piece(BLOCK_SIZE)
        unchoke = encode_unchoke()
        # EOF immediately after unchoke — no block data arrives
        reader = make_reader(unchoke)
        writer = MockWriter()
        peer   = make_peer(reader, writer)
        with pytest.raises(PeerError):
            await peer.download_piece(0, BLOCK_SIZE, h)

    async def test_block_timeout_raises_peer_error(self):
        """A peer that unchokes but then sends no block data raises PeerError."""
        _, h = make_valid_piece(BLOCK_SIZE)
        unchoke = encode_unchoke()
        # Feed unchoke but never feed block data or EOF — simulates a stalled peer
        reader = asyncio.StreamReader()
        reader.feed_data(unchoke)
        writer = MockWriter()
        peer   = make_peer(reader, writer)
        with pytest.raises(PeerError, match="[Tt]imed"):
            await peer.download_piece(0, BLOCK_SIZE, h, block_timeout=0.05)

    async def test_block_timeout_message_includes_piece_index(self):
        """The PeerError message from a timeout should name the piece index."""
        _, h = make_valid_piece(BLOCK_SIZE)
        unchoke = encode_unchoke()
        reader = asyncio.StreamReader()
        reader.feed_data(unchoke)
        writer = MockWriter()
        peer   = make_peer(reader, writer)
        with pytest.raises(PeerError, match="piece 7"):
            await peer.download_piece(7, BLOCK_SIZE, h, block_timeout=0.05)

    async def test_normal_download_unaffected_by_timeout_param(self):
        """Passing a custom block_timeout doesn't break a normal fast download."""
        data, h = make_valid_piece(BLOCK_SIZE)
        responses = encode_piece(0, 0, data)
        unchoke   = encode_unchoke()
        reader    = make_reader(unchoke + responses)
        writer    = MockWriter()
        peer      = make_peer(reader, writer)
        result    = await peer.download_piece(0, BLOCK_SIZE, h, block_timeout=5.0)
        assert result == data


# ---------------------------------------------------------------------------
# CANCEL in end-game (completion_check)
# ---------------------------------------------------------------------------

class TestCancelEndGame:
    async def test_no_cancel_when_no_completion_check(self):
        """Without completion_check, download proceeds normally."""
        data, h = make_valid_piece(BLOCK_SIZE)
        reader = make_reader(encode_unchoke() + encode_piece(0, 0, data))
        writer = MockWriter()
        peer   = make_peer(reader, writer)
        result = await peer.download_piece(0, BLOCK_SIZE, h)
        assert result == data
        assert MSG_REQUEST in writer.sent_ids()
        # No CANCEL
        from bittorrent.messages import MSG_CANCEL
        assert MSG_CANCEL not in writer.sent_ids()

    async def test_cancel_sent_when_completion_check_returns_true(self):
        """When completion_check fires, CANCEL is sent for the pending block."""
        from bittorrent.messages import MSG_CANCEL, encode_cancel
        data, h = make_valid_piece(BLOCK_SIZE)
        # Peer sends unchoke only — no PIECE response (it's slow)
        reader = asyncio.StreamReader()
        reader.feed_data(encode_unchoke())
        # Piece will never arrive, but completion_check fires first
        writer = MockWriter()
        peer   = make_peer(reader, writer)

        # completion_check returns True immediately
        with pytest.raises(PeerError, match="already complete"):
            await peer.download_piece(
                0, BLOCK_SIZE, h,
                block_timeout=1.0,
                completion_check=lambda idx: True,
            )

        # CANCEL must have been sent for the one block
        assert MSG_CANCEL in writer.sent_ids()
        cancel_payloads = [
            p for mid, p in writer.messages_sent() if mid == MSG_CANCEL
        ]
        assert len(cancel_payloads) == 1
        pi, off, length = struct.unpack("!III", cancel_payloads[0])
        assert pi == 0
        assert off == 0
        assert length == BLOCK_SIZE

    async def test_cancel_sent_for_all_unreceived_blocks_multi_block(self):
        """For a multi-block piece, CANCELs cover only blocks not yet received."""
        from bittorrent.messages import MSG_CANCEL
        piece_size = BLOCK_SIZE * 3
        piece_data = bytes(range(256)) * (piece_size // 256)
        h = hashlib.sha1(piece_data).digest()

        # First block arrives, then completion_check fires before blocks 1 and 2
        block0 = encode_piece(0, 0, piece_data[:BLOCK_SIZE])
        reader = asyncio.StreamReader()
        reader.feed_data(encode_unchoke() + block0)

        writer = MockWriter()
        peer   = make_peer(reader, writer)

        call_count = [0]
        def check(idx):
            call_count[0] += 1
            # False on first call (block 0 arrives), True on second
            return call_count[0] > 1

        with pytest.raises(PeerError, match="already complete"):
            await peer.download_piece(
                0, piece_size, h,
                block_timeout=1.0,
                completion_check=check,
            )

        cancel_msgs = [p for mid, p in writer.messages_sent() if mid == MSG_CANCEL]
        # Blocks 1 and 2 should be cancelled, block 0 was received
        assert len(cancel_msgs) == 2
        cancelled_offsets = sorted(
            struct.unpack("!III", p)[1] for p in cancel_msgs
        )
        assert cancelled_offsets == [BLOCK_SIZE, BLOCK_SIZE * 2]

    async def test_completion_check_not_triggered_on_normal_download(self):
        """If completion_check always returns False, download completes normally."""
        data, h = make_valid_piece(BLOCK_SIZE)
        reader = make_reader(encode_unchoke() + encode_piece(0, 0, data))
        writer = MockWriter()
        peer   = make_peer(reader, writer)
        result = await peer.download_piece(
            0, BLOCK_SIZE, h,
            completion_check=lambda idx: False,
        )
        assert result == data


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------

class TestClose:
    async def test_close_marks_writer_closed(self):
        reader = make_reader(b"")
        writer = MockWriter()
        peer   = make_peer(reader, writer)
        await peer.close()
        assert writer.closed


# ---------------------------------------------------------------------------
# Upload / seeding methods
# ---------------------------------------------------------------------------

class TestUploadMethods:
    """Tests for the seeder-side PeerConnection methods."""

    async def test_send_have(self):
        reader = make_reader(b"")
        writer = MockWriter()
        peer   = make_peer(reader, writer)
        await peer.send_have(42)
        ids = writer.sent_ids()
        assert MSG_HAVE in ids
        # payload should encode piece index 42
        msgs = writer.messages_sent()
        have_payloads = [p for mid, p in msgs if mid == MSG_HAVE]
        assert struct.unpack("!I", have_payloads[0])[0] == 42

    async def test_send_bitfield(self):
        reader = make_reader(b"")
        writer = MockWriter()
        peer   = make_peer(reader, writer)
        bf = bytearray(b"\xff\x00")
        await peer.send_bitfield(bf)
        msgs = writer.messages_sent()
        assert any(mid == MSG_BITFIELD for mid, _ in msgs)

    async def test_send_choke_sets_am_choking(self):
        reader = make_reader(b"")
        writer = MockWriter()
        peer   = make_peer(reader, writer)
        peer.am_choking = False
        await peer.send_choke()
        assert peer.am_choking is True
        assert MSG_CHOKE in writer.sent_ids()

    async def test_send_unchoke_clears_am_choking(self):
        reader = make_reader(b"")
        writer = MockWriter()
        peer   = make_peer(reader, writer)
        await peer.send_unchoke()
        assert peer.am_choking is False
        assert MSG_UNCHOKE in writer.sent_ids()

    async def test_read_request_returns_fields(self):
        from bittorrent.messages import encode_request, MSG_INTERESTED
        # INTERESTED first, then a REQUEST
        msgs = (
            encode_interested()    # sets peer_interested
            + encode_request(5, 0, 16384)
        )
        reader = make_reader(msgs)
        writer = MockWriter()
        peer   = make_peer(reader, writer)
        piece, offset, length = await peer.read_request()
        assert piece == 5
        assert offset == 0
        assert length == 16384
        assert peer.peer_interested is True

    async def test_read_request_skips_non_request_messages(self):
        from bittorrent.messages import encode_request, encode_have
        msgs = encode_have(7) + encode_request(3, 16384, 16384)
        reader = make_reader(msgs)
        writer = MockWriter()
        peer   = make_peer(reader, writer)
        piece, offset, _ = await peer.read_request()
        assert piece == 3
        assert offset == 16384

    async def test_send_piece_block(self):
        from bittorrent.messages import MSG_PIECE
        reader = make_reader(b"")
        writer = MockWriter()
        peer   = make_peer(reader, writer)
        data   = b"X" * 16384
        await peer.send_piece_block(0, 0, data)
        assert MSG_PIECE in writer.sent_ids()


# ---------------------------------------------------------------------------
# Incoming handshake (accept)
# ---------------------------------------------------------------------------

class TestAccept:
    """Tests for PeerConnection.accept() — the seeder-side handshake."""

    def _make_incoming_writer(self, peername=("5.6.7.8", 12345)):
        """A MockWriter that also supports get_extra_info('peername')."""
        writer = MockWriter()
        writer.get_extra_info = lambda key, default=None: (
            peername if key == "peername" else default
        )
        return writer

    async def test_accept_validates_info_hash(self):
        their_hs = encode_handshake(INFO_HASH, THEIR_ID)
        reader   = make_reader(their_hs)
        writer   = self._make_incoming_writer()
        conn     = await PeerConnection.accept(reader, writer, INFO_HASH, PEER_ID)
        assert conn.remote_peer_id == THEIR_ID

    async def test_accept_sends_our_handshake_back(self):
        their_hs = encode_handshake(INFO_HASH, THEIR_ID)
        reader   = make_reader(their_hs)
        writer   = self._make_incoming_writer()
        await PeerConnection.accept(reader, writer, INFO_HASH, PEER_ID)
        # First 68 bytes written should be our handshake with correct info_hash
        from bittorrent.messages import decode_handshake
        sent = bytes(writer.buffer[:68])
        got_hash, got_id = decode_handshake(sent)
        assert got_hash == INFO_HASH
        assert got_id == PEER_ID

    async def test_accept_wrong_info_hash_raises(self):
        wrong_hash = bytes(reversed(range(20)))
        their_hs   = encode_handshake(wrong_hash, THEIR_ID)
        reader     = make_reader(their_hs)
        writer     = self._make_incoming_writer()
        with pytest.raises(PeerError, match="info_hash mismatch"):
            await PeerConnection.accept(reader, writer, INFO_HASH, PEER_ID)

    async def test_accept_truncated_handshake_raises(self):
        reader = make_reader(b"\x13BitTorrent prot")   # truncated
        writer = self._make_incoming_writer()
        with pytest.raises(PeerError):
            await PeerConnection.accept(reader, writer, INFO_HASH, PEER_ID)

    async def test_accept_stores_host_port_from_peername(self):
        their_hs = encode_handshake(INFO_HASH, THEIR_ID)
        reader   = make_reader(their_hs)
        writer   = self._make_incoming_writer(("9.8.7.6", 54321))
        conn     = await PeerConnection.accept(reader, writer, INFO_HASH, PEER_ID)
        assert conn.host == "9.8.7.6"
        assert conn.port == 54321
