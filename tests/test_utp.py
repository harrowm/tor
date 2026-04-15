"""
Tests for bittorrent.utp — BEP 29 Micro Transport Protocol.

Tests cover:
  - Pure helpers: pack_header, parse_packet, seq16_le
  - UTPPacket dataclass
  - UTPWriter (StreamWriter-compatible wrapper)
  - UTPConnection state machine (using localhost loopback for transport tests)
"""

from __future__ import annotations

import asyncio
import socket
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bittorrent.utp import (
    HEADER_SIZE,
    MAX_PAYLOAD,
    ST_DATA,
    ST_FIN,
    ST_RESET,
    ST_STATE,
    ST_SYN,
    UTPConnection,
    UTPError,
    UTPPacket,
    UTPWriter,
    _UDPProtocol,
    _now_us,
    pack_header,
    parse_packet,
    seq16_le,
)


# ---------------------------------------------------------------------------
# pack_header / parse_packet — round-trip tests
# ---------------------------------------------------------------------------

class TestPackHeader:
    def test_returns_20_bytes(self):
        raw = pack_header(ST_DATA, 1234, 0, 65536, 1, 0)
        assert len(raw) == HEADER_SIZE

    def test_type_encoded_in_high_nibble(self):
        for pkt_type in (ST_DATA, ST_FIN, ST_STATE, ST_RESET, ST_SYN):
            raw = pack_header(pkt_type, 0, 0, 0, 0, 0)
            assert (raw[0] >> 4) == pkt_type

    def test_version_encoded_in_low_nibble(self):
        raw = pack_header(ST_DATA, 0, 0, 0, 0, 0)
        assert (raw[0] & 0x0F) == 1

    def test_connection_id_big_endian(self):
        raw = pack_header(ST_DATA, 0x1234, 0, 0, 0, 0)
        assert struct.unpack_from("!H", raw, 2)[0] == 0x1234

    def test_seq_nr_big_endian(self):
        raw = pack_header(ST_DATA, 0, 0, 0, 0xABCD, 0)
        assert struct.unpack_from("!H", raw, 16)[0] == 0xABCD

    def test_ack_nr_big_endian(self):
        raw = pack_header(ST_DATA, 0, 0, 0, 0, 0xBEEF)
        assert struct.unpack_from("!H", raw, 18)[0] == 0xBEEF

    def test_wnd_size_big_endian(self):
        raw = pack_header(ST_DATA, 0, 0, 0x00_01_00_00, 0, 0)
        assert struct.unpack_from("!I", raw, 12)[0] == 0x00_01_00_00

    def test_uint16_wrapping_seq(self):
        """seq_nr > 0xFFFF wraps to 16 bits."""
        raw = pack_header(ST_DATA, 0, 0, 0, 0x1_0001, 0)
        assert struct.unpack_from("!H", raw, 16)[0] == 1

    def test_uint16_wrapping_conn_id(self):
        raw = pack_header(ST_DATA, 0x1_0042, 0, 0, 0, 0)
        assert struct.unpack_from("!H", raw, 2)[0] == 0x42


class TestParsePacket:
    def _make(self, pkt_type, conn_id=1, ts_diff=0, wnd=65536, seq=1, ack=0, payload=b""):
        return pack_header(pkt_type, conn_id, ts_diff, wnd, seq, ack) + payload

    def test_parses_valid_syn(self):
        raw = self._make(ST_SYN, conn_id=999, seq=42)
        pkt = parse_packet(raw)
        assert pkt is not None
        assert pkt.pkt_type == ST_SYN
        assert pkt.conn_id == 999
        assert pkt.seq_nr == 42

    def test_parses_data_with_payload(self):
        raw = self._make(ST_DATA, seq=7, ack=3, payload=b"hello world")
        pkt = parse_packet(raw)
        assert pkt is not None
        assert pkt.pkt_type == ST_DATA
        assert pkt.payload == b"hello world"
        assert pkt.seq_nr == 7
        assert pkt.ack_nr == 3

    def test_returns_none_if_too_short(self):
        assert parse_packet(b"\x00" * (HEADER_SIZE - 1)) is None

    def test_returns_none_for_wrong_version(self):
        raw = bytearray(self._make(ST_DATA))
        raw[0] = (ST_DATA << 4) | 2   # version=2 → invalid
        assert parse_packet(bytes(raw)) is None

    def test_returns_none_for_unknown_type(self):
        raw = bytearray(self._make(ST_DATA))
        raw[0] = (6 << 4) | 1   # type=6 → > ST_SYN
        assert parse_packet(bytes(raw)) is None

    def test_all_packet_types_parse(self):
        for t in (ST_DATA, ST_FIN, ST_STATE, ST_RESET, ST_SYN):
            assert parse_packet(self._make(t)) is not None

    def test_empty_payload_ok(self):
        raw = self._make(ST_STATE)
        pkt = parse_packet(raw)
        assert pkt is not None
        assert pkt.payload == b""

    def test_roundtrip_preserves_fields(self):
        raw = pack_header(ST_DATA, 0xABCD, 12345, 0xFFFF, 0x100, 0x200) + b"\xDE\xAD"
        pkt = parse_packet(raw)
        assert pkt.pkt_type == ST_DATA
        assert pkt.conn_id  == 0xABCD
        assert pkt.ts_diff  == 12345
        assert pkt.wnd_size == 0xFFFF
        assert pkt.seq_nr   == 0x100
        assert pkt.ack_nr   == 0x200
        assert pkt.payload  == b"\xDE\xAD"


# ---------------------------------------------------------------------------
# seq16_le — wraparound comparisons
# ---------------------------------------------------------------------------

class TestSeq16Le:
    def test_equal_is_le(self):
        assert seq16_le(5, 5) is True

    def test_smaller_is_le(self):
        assert seq16_le(3, 7) is True

    def test_larger_is_not_le(self):
        assert seq16_le(7, 3) is False

    def test_wraparound_le(self):
        # 0xFFFE < 0x0001 with wraparound
        assert seq16_le(0xFFFE, 0x0001) is True

    def test_wraparound_not_le(self):
        assert seq16_le(0x0001, 0xFFFE) is False

    def test_boundaries(self):
        # Exactly 0x7FFF apart — le is True
        assert seq16_le(0, 0x7FFF) is True
        # 0x8000 apart: ((0x8000 - 0) & 0xFFFF) = 0x8000 > 0x7FFF → not le
        assert seq16_le(0, 0x8000) is False


# ---------------------------------------------------------------------------
# UTPWriter
# ---------------------------------------------------------------------------

class TestUTPWriter:
    def _make_writer(self):
        """Return a UTPWriter with a mock UTPConnection."""
        mock_conn = MagicMock()
        mock_conn.send_data = AsyncMock()
        return UTPWriter(mock_conn), mock_conn

    def test_write_buffers_data(self):
        writer, _ = self._make_writer()
        writer.write(b"hello")
        assert bytes(writer._buf) == b"hello"

    def test_write_accumulates(self):
        writer, _ = self._make_writer()
        writer.write(b"foo")
        writer.write(b"bar")
        assert bytes(writer._buf) == b"foobar"

    async def test_drain_sends_and_clears_buffer(self):
        writer, mock_conn = self._make_writer()
        writer.write(b"hello world")
        await writer.drain()
        mock_conn.send_data.assert_awaited_once_with(b"hello world")
        assert bytes(writer._buf) == b""

    async def test_drain_empty_buffer_no_send(self):
        writer, mock_conn = self._make_writer()
        await writer.drain()
        mock_conn.send_data.assert_not_called()

    async def test_drain_multiple_writes_sends_once(self):
        writer, mock_conn = self._make_writer()
        writer.write(b"part1")
        writer.write(b"part2")
        await writer.drain()
        mock_conn.send_data.assert_awaited_once_with(b"part1part2")

    def test_close_is_noop(self):
        writer, mock_conn = self._make_writer()
        writer.close()  # must not raise

    async def test_wait_closed_calls_conn_close(self):
        writer, mock_conn = self._make_writer()
        mock_conn.close = AsyncMock()
        await writer.wait_closed()
        mock_conn.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# UTPConnection._handle_packet — state machine unit tests
# ---------------------------------------------------------------------------

class TestUTPConnectionStateMachine:
    def _make_conn(self):
        """Return a UTPConnection with a mock protocol (no real UDP)."""
        conn = UTPConnection()
        conn._recv_id = 100
        conn._send_id = 101
        conn._seq_nr  = 5
        conn._ack_nr  = 0
        mock_proto = MagicMock()
        mock_proto.send = MagicMock()
        conn._proto = mock_proto
        return conn

    def _synack(self, conn, seq_nr=1, ack_nr=None):
        """Return a ST_STATE packet that acts as a SYNACK for conn."""
        if ack_nr is None:
            ack_nr = conn._seq_nr
        return UTPPacket(
            pkt_type=ST_STATE,
            conn_id=conn._recv_id,
            timestamp=_now_us(),
            ts_diff=0,
            wnd_size=65536,
            seq_nr=seq_nr,
            ack_nr=ack_nr,
        )

    def test_synack_sets_connected(self):
        conn = self._make_conn()
        assert not conn._connected.is_set()
        conn._handle_packet(self._synack(conn))
        assert conn._connected.is_set()

    def test_synack_increments_seq_nr(self):
        conn = self._make_conn()
        old_seq = conn._seq_nr
        conn._handle_packet(self._synack(conn))
        assert conn._seq_nr == (old_seq + 1) & 0xFFFF

    def test_synack_sets_ack_nr(self):
        conn = self._make_conn()
        conn._handle_packet(self._synack(conn, seq_nr=42))
        assert conn._ack_nr == 42

    def test_data_packet_feeds_reader(self):
        conn = self._make_conn()
        conn._connected.set()
        conn._ack_nr = 10

        pkt = UTPPacket(
            pkt_type=ST_DATA, conn_id=conn._recv_id,
            timestamp=_now_us(), ts_diff=0, wnd_size=65536,
            seq_nr=11, ack_nr=conn._seq_nr,
            payload=b"hello",
        )
        conn._handle_packet(pkt)
        assert conn._reader._buffer == bytearray(b"hello")

    def test_out_of_order_data_dropped(self):
        conn = self._make_conn()
        conn._connected.set()
        conn._ack_nr = 10

        pkt = UTPPacket(
            pkt_type=ST_DATA, conn_id=conn._recv_id,
            timestamp=_now_us(), ts_diff=0, wnd_size=65536,
            seq_nr=12, ack_nr=conn._seq_nr,   # gap: 11 missing
            payload=b"late",
        )
        conn._handle_packet(pkt)
        assert conn._reader._buffer == bytearray(b"")  # not fed

    def test_reset_sets_reader_exception(self):
        conn = self._make_conn()
        pkt = UTPPacket(
            pkt_type=ST_RESET, conn_id=conn._recv_id,
            timestamp=_now_us(), ts_diff=0, wnd_size=0,
            seq_nr=1, ack_nr=0,
        )
        conn._handle_packet(pkt)
        assert isinstance(conn._reader.exception(), UTPError)

    def test_fin_feeds_eof(self):
        conn = self._make_conn()
        pkt = UTPPacket(
            pkt_type=ST_FIN, conn_id=conn._recv_id,
            timestamp=_now_us(), ts_diff=0, wnd_size=0,
            seq_nr=1, ack_nr=0,
        )
        conn._handle_packet(pkt)
        # EOF is indicated by feeding b"" (eof_received)
        assert conn._reader.at_eof() or conn._connected.is_set()

    def test_process_ack_removes_inflight(self):
        conn = self._make_conn()
        conn._inflight = {1: b"pkt1", 2: b"pkt2", 3: b"pkt3"}
        conn._process_ack(2)
        assert 1 not in conn._inflight
        assert 2 not in conn._inflight
        assert 3 in conn._inflight

    def test_process_ack_wraparound(self):
        conn = self._make_conn()
        conn._inflight = {0xFFFE: b"pkt", 0xFFFF: b"pkt", 0x0000: b"pkt"}
        conn._process_ack(0x0000)
        assert len(conn._inflight) == 0

    def test_acked_event_set_on_state_packet(self):
        conn = self._make_conn()
        conn._connected.set()
        assert not conn._acked.is_set()
        pkt = UTPPacket(
            pkt_type=ST_STATE, conn_id=conn._recv_id,
            timestamp=_now_us(), ts_diff=0, wnd_size=65536,
            seq_nr=1, ack_nr=conn._seq_nr,
        )
        conn._handle_packet(pkt)
        assert conn._acked.is_set()

    def test_send_state_calls_proto_send(self):
        conn = self._make_conn()
        conn._send_state()
        assert conn._proto.send.called


# ---------------------------------------------------------------------------
# UTPConnection — localhost loopback integration tests
# ---------------------------------------------------------------------------

class TestUTPLoopback:
    """Integration tests using two real UDP endpoints on localhost."""

    async def test_handshake_completes(self):
        """SYNACK received via _handle_packet sets connected state correctly."""
        # Test state machine directly without real UDP (unit test of the handshake path)
        conn = UTPConnection()
        conn._recv_id = 42
        conn._send_id = 43
        conn._seq_nr  = 100

        mock_proto = MagicMock()
        mock_proto.send = MagicMock()
        conn._proto = mock_proto

        assert not conn._connected.is_set()

        # Simulate receiving a SYNACK (ST_STATE acking our SYN)
        synack = UTPPacket(
            pkt_type=ST_STATE,
            conn_id=conn._recv_id,
            timestamp=_now_us(),
            ts_diff=0,
            wnd_size=65536,
            seq_nr=50,       # responder's seq_nr
            ack_nr=100,      # acks our SYN seq_nr
        )
        conn._handle_packet(synack)

        assert conn._connected.is_set()
        assert conn._seq_nr == 101    # incremented after SYNACK
        assert conn._ack_nr == 50     # set to responder's seq_nr

    async def test_data_transfer_loopback(self):
        """Send data from initiator to a UDP server via loopback and verify receipt."""
        loop = asyncio.get_running_loop()

        # Set up a "responder" that auto-ACKs incoming DATA packets
        class AutoAckProtocol(asyncio.DatagramProtocol):
            def __init__(self):
                self._transport: asyncio.DatagramTransport | None = None
                self.received = bytearray()

            def connection_made(self, transport):
                self._transport = transport

            def datagram_received(self, data, addr):
                pkt = parse_packet(data)
                if pkt is None:
                    return
                self.received.extend(pkt.payload)
                # Send ACK with conn_id = initiator's send_id
                ack = pack_header(
                    ST_STATE,
                    (pkt.conn_id + 1) & 0xFFFF,
                    0, 65536,
                    50, pkt.seq_nr,
                )
                self._transport.sendto(ack, addr)

            def error_received(self, exc): pass
            def connection_lost(self, exc): pass

        auto_proto = AutoAckProtocol()
        auto_transport, _ = await loop.create_datagram_endpoint(
            lambda: auto_proto,
            local_addr=("127.0.0.1", 0),
        )
        _, port = auto_transport.get_extra_info("sockname")

        # Set up initiator already in "connected" state (skip handshake)
        conn = UTPConnection()
        conn._recv_id = 100
        conn._send_id = 101
        conn._seq_nr  = 0
        conn._ack_nr  = 0
        conn._connected.set()

        init_transport, init_protocol = await loop.create_datagram_endpoint(
            lambda: _UDPProtocol(conn),
            remote_addr=("127.0.0.1", port),
        )
        conn._proto = init_protocol

        payload = b"hello uTP world"
        await conn.send_data(payload)

        assert payload in bytes(auto_proto.received)

        init_transport.close()
        auto_transport.close()


# ---------------------------------------------------------------------------
# open_utp_connection — error path
# ---------------------------------------------------------------------------

class TestOpenUTPConnection:
    async def test_raises_on_connection_failure(self):
        """Connecting to an unreachable port raises UTPError."""
        from bittorrent.utp import open_utp_connection

        # Port 1 is almost certainly not open; uTP SYN will time out.
        # We don't want the test to wait 10 seconds so we patch _do_syn.
        with patch.object(UTPConnection, "_do_syn", side_effect=UTPError("timeout")):
            with pytest.raises(UTPError):
                await open_utp_connection("127.0.0.1", 1, timeout=0.1)
