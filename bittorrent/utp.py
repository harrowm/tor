"""
BEP 29 — uTP (Micro Transport Protocol) over UDP.

uTP provides TCP-like reliable ordered delivery over UDP, with LEDBAT
congestion control (we use a simplified fixed window for clarity).

Architecture:
  UTPConnection manages one connection using asyncio DatagramProtocol.
  It exposes a StreamReader for incoming data and a UTPWriter for outgoing
  data, making it drop-in compatible with PeerConnection._from_streams().

20-byte header layout (all big-endian):
  Byte 0:      (type << 4) | version  — version is always 1
  Byte 1:      extension              — 0 = none
  Bytes 2-3:   connection_id
  Bytes 4-7:   timestamp_microseconds
  Bytes 8-11:  timestamp_difference
  Bytes 12-15: wnd_size (advertised receive window)
  Bytes 16-17: seq_nr
  Bytes 18-19: ack_nr

Connection setup (as initiator):
  1. Choose recv_id (random).  send_id = recv_id + 1.
  2. Send ST_SYN  with connection_id = recv_id.
  3. Peer responds ST_STATE with connection_id = recv_id, ack_nr = our seq_nr.
  4. seq_nr is incremented (SYN occupies one slot).
"""

from __future__ import annotations

import asyncio
import random
import struct
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Packet type constants
# ---------------------------------------------------------------------------

ST_DATA  = 0   # carries payload data
ST_FIN   = 1   # graceful close
ST_STATE = 2   # ACK only (no payload)
ST_RESET = 3   # hard reset
ST_SYN   = 4   # connection initiation

VERSION     = 1
HEADER_SIZE = 20
MAX_PAYLOAD = 1200       # safe below typical 1500-byte MTU (IP + UDP overhead)
RECV_WINDOW = 1 << 18   # 256 KB advertised receive window
MAX_RETRIES = 6


class UTPError(Exception):
    """Raised on uTP protocol errors (timeout, reset, malformed packet)."""


# ---------------------------------------------------------------------------
# Packet encode / decode
# ---------------------------------------------------------------------------

def _now_us() -> int:
    """Current monotonic time in microseconds, truncated to uint32."""
    return int(time.monotonic() * 1_000_000) & 0xFFFF_FFFF


def pack_header(
    pkt_type: int,
    conn_id: int,
    ts_diff: int,
    wnd_size: int,
    seq_nr: int,
    ack_nr: int,
) -> bytes:
    """Return a 20-byte uTP header.

    Args:
        pkt_type:  One of ST_DATA, ST_FIN, ST_STATE, ST_RESET, ST_SYN.
        conn_id:   Connection ID for this packet (uint16).
        ts_diff:   Timestamp difference from peer's last packet (uint32).
        wnd_size:  Advertised receive window in bytes (uint32).
        seq_nr:    Sequence number (uint16).
        ack_nr:    Acknowledgement number (uint16).
    """
    b0 = (pkt_type << 4) | VERSION
    ts = _now_us()
    return struct.pack(
        "!BBHIIIHH",
        b0, 0,
        conn_id  & 0xFFFF,
        ts,
        ts_diff  & 0xFFFF_FFFF,
        wnd_size,
        seq_nr   & 0xFFFF,
        ack_nr   & 0xFFFF,
    )


@dataclass
class UTPPacket:
    """Parsed uTP packet."""
    pkt_type: int
    conn_id:  int
    timestamp: int
    ts_diff:  int
    wnd_size: int
    seq_nr:   int
    ack_nr:   int
    payload:  bytes = b""


def parse_packet(data: bytes) -> UTPPacket | None:
    """Parse raw UDP payload bytes into a UTPPacket.

    Returns None if the data is too short, has the wrong version, or an
    unrecognised packet type.
    """
    if len(data) < HEADER_SIZE:
        return None
    b0, _ext, conn_id, ts, ts_diff, wnd_size, seq_nr, ack_nr = struct.unpack_from(
        "!BBHIIIHH", data
    )
    pkt_type = (b0 >> 4) & 0xF
    version  = b0 & 0xF
    if version != VERSION or pkt_type > ST_SYN:
        return None
    return UTPPacket(
        pkt_type=pkt_type,
        conn_id=conn_id,
        timestamp=ts,
        ts_diff=ts_diff,
        wnd_size=wnd_size,
        seq_nr=seq_nr,
        ack_nr=ack_nr,
        payload=data[HEADER_SIZE:],
    )


def seq16_le(a: int, b: int) -> bool:
    """Return True if sequence number *a* ≤ *b* with uint16 wraparound."""
    return ((b - a) & 0xFFFF) <= 0x7FFF


# ---------------------------------------------------------------------------
# asyncio DatagramProtocol
# ---------------------------------------------------------------------------

class _UDPProtocol(asyncio.DatagramProtocol):
    """UDP datagram protocol that feeds parsed packets into a UTPConnection."""

    def __init__(self, conn: "UTPConnection") -> None:
        self._conn      = conn
        self._transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        self._transport = transport

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        pkt = parse_packet(data)
        if pkt is not None:
            self._conn._handle_packet(pkt)

    def error_received(self, exc: Exception) -> None:
        pass  # ICMP errors — treated as transient

    def connection_lost(self, exc: Exception | None) -> None:
        if exc:
            self._conn._reader.set_exception(UTPError(str(exc)))
        else:
            self._conn._reader.feed_eof()

    def send(self, data: bytes) -> None:
        if self._transport is not None:
            self._transport.sendto(data)

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None


# ---------------------------------------------------------------------------
# UTPConnection — the main class
# ---------------------------------------------------------------------------

class UTPConnection:
    """One uTP connection over UDP.

    Create via ``UTPConnection.connect()``, then use ``reader`` and
    ``make_writer()`` to pass into ``PeerConnection._from_streams()``.
    """

    def __init__(self) -> None:
        self._proto: _UDPProtocol | None = None
        self._reader = asyncio.StreamReader(limit=1 << 24)

        # Connection IDs
        self._recv_id: int = 0   # ID we expect in incoming packets
        self._send_id: int = 0   # ID we put in outgoing packets

        # Sequence number tracking
        self._seq_nr:  int = 0   # last seq_nr we sent
        self._ack_nr:  int = 0   # last in-order seq_nr we received

        # Timestamp tracking for LEDBAT (simplified)
        self._ts_diff: int = 0

        # In-flight: seq_nr → raw packet bytes awaiting ACK
        self._inflight: dict[int, bytes] = {}

        # Synchronisation
        self._connected = asyncio.Event()
        self._acked     = asyncio.Event()
        self._closed    = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def connect(
        cls,
        host: str,
        port: int,
        *,
        timeout: float = 10.0,
    ) -> "UTPConnection":
        """Open a uTP connection to *host*:*port*.

        Returns the connected UTPConnection.
        Raises UTPError on connection failure.
        """
        conn = cls()
        loop = asyncio.get_running_loop()
        try:
            _, proto = await loop.create_datagram_endpoint(
                lambda: _UDPProtocol(conn),
                remote_addr=(host, port),
            )
        except OSError as exc:
            raise UTPError(f"UDP socket creation failed: {exc}") from exc

        conn._proto   = proto    # type: ignore[assignment]
        conn._recv_id = random.getrandbits(16)
        conn._send_id = (conn._recv_id + 1) & 0xFFFF
        conn._seq_nr  = random.getrandbits(16)

        try:
            await conn._do_syn(timeout)
        except Exception as exc:
            conn._proto.close()    # type: ignore[union-attr]
            raise UTPError(f"uTP handshake failed: {exc}") from exc

        return conn

    async def _do_syn(self, timeout: float) -> None:
        """Send ST_SYN with exponential-backoff retries until SYNACK arrives."""
        syn = pack_header(
            ST_SYN, self._recv_id, 0, RECV_WINDOW,
            self._seq_nr, 0,
        )
        delay = min(timeout / 4, 1.0)
        for _ in range(MAX_RETRIES):
            self._proto.send(syn)   # type: ignore[union-attr]
            try:
                await asyncio.wait_for(self._connected.wait(), timeout=delay)
                return
            except asyncio.TimeoutError:
                delay = min(delay * 2, timeout)
        raise UTPError("SYN timed out after retries")

    # ------------------------------------------------------------------
    # Incoming packet dispatch
    # ------------------------------------------------------------------

    def _handle_packet(self, pkt: UTPPacket) -> None:
        """Called by _UDPProtocol on every valid incoming packet."""
        self._ts_diff = (_now_us() - pkt.timestamp) & 0xFFFF_FFFF

        if pkt.pkt_type == ST_RESET:
            exc = UTPError("Connection reset by peer")
            if not self._reader.exception():
                self._reader.set_exception(exc)
            self._connected.set()
            return

        if pkt.pkt_type == ST_FIN:
            self._reader.feed_eof()
            self._connected.set()
            return

        # ST_STATE or ST_DATA both carry an ACK
        if pkt.pkt_type in (ST_STATE, ST_DATA):
            self._process_ack(pkt.ack_nr)
            self._acked.set()

            if not self._connected.is_set():
                # First SYNACK: SYN occupies one seq slot
                self._ack_nr = pkt.seq_nr
                self._seq_nr = (self._seq_nr + 1) & 0xFFFF
                self._connected.set()

            if pkt.pkt_type == ST_DATA and pkt.payload:
                expected = (self._ack_nr + 1) & 0xFFFF
                if pkt.seq_nr == expected:
                    self._ack_nr = pkt.seq_nr
                    self._reader.feed_data(pkt.payload)
                # Out-of-order: drop; peer will retransmit on missing ACK
                self._send_state()

    def _process_ack(self, ack_nr: int) -> None:
        """Remove all inflight packets whose seq_nr ≤ ack_nr."""
        done = [s for s in self._inflight if seq16_le(s, ack_nr)]
        for s in done:
            del self._inflight[s]

    def _send_state(self) -> None:
        """Emit a pure ACK (ST_STATE) for current _ack_nr."""
        pkt = pack_header(
            ST_STATE, self._send_id, self._ts_diff, RECV_WINDOW,
            self._seq_nr, self._ack_nr,
        )
        self._proto.send(pkt)   # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Sending data
    # ------------------------------------------------------------------

    async def send_data(self, data: bytes) -> None:
        """Send *data* reliably over uTP.

        Chunks *data* into MAX_PAYLOAD-sized ST_DATA packets and waits for
        each to be acknowledged before sending the next.

        Raises UTPError on repeated timeout.
        """
        offset = 0
        while offset < len(data):
            chunk = data[offset : offset + MAX_PAYLOAD]
            seq   = (self._seq_nr + 1) & 0xFFFF
            self._seq_nr = seq

            raw = pack_header(
                ST_DATA, self._send_id, self._ts_diff, RECV_WINDOW,
                seq, self._ack_nr,
            ) + chunk
            self._inflight[seq] = raw
            self._proto.send(raw)   # type: ignore[union-attr]

            delay = 1.0
            for _ in range(MAX_RETRIES):
                self._acked.clear()
                try:
                    await asyncio.wait_for(self._acked.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                if seq not in self._inflight:
                    break
                self._proto.send(raw)   # retransmit  # type: ignore[union-attr]
                delay = min(delay * 2, 30.0)
            else:
                raise UTPError(f"Data ACK timeout (seq={seq})")

            offset += len(chunk)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Send ST_FIN and close the UDP socket."""
        if self._closed or self._proto is None:
            return
        self._closed = True
        self._seq_nr = (self._seq_nr + 1) & 0xFFFF
        fin = pack_header(
            ST_FIN, self._send_id, self._ts_diff, RECV_WINDOW,
            self._seq_nr, self._ack_nr,
        )
        self._proto.send(fin)
        await asyncio.sleep(0.05)  # give FIN time to reach peer
        self._proto.close()

    # ------------------------------------------------------------------
    # Integration helpers
    # ------------------------------------------------------------------

    @property
    def reader(self) -> asyncio.StreamReader:
        """asyncio.StreamReader fed with incoming uTP data."""
        return self._reader

    def make_writer(self) -> "UTPWriter":
        """Return a StreamWriter-compatible UTPWriter for this connection."""
        return UTPWriter(self)


# ---------------------------------------------------------------------------
# UTPWriter — StreamWriter-compatible write side
# ---------------------------------------------------------------------------

class UTPWriter:
    """Wraps UTPConnection.send_data() to look like asyncio.StreamWriter.

    PeerConnection calls ``writer.write(data)`` followed by
    ``await writer.drain()``.  We buffer in ``write()`` and flush in
    ``drain()``.
    """

    def __init__(self, conn: UTPConnection) -> None:
        self._conn = conn
        self._buf  = bytearray()

    def write(self, data: bytes) -> None:
        self._buf.extend(data)

    async def drain(self) -> None:
        if self._buf:
            await self._conn.send_data(bytes(self._buf))
            self._buf.clear()

    def close(self) -> None:
        pass  # actual close via wait_closed()

    async def wait_closed(self) -> None:
        await self._conn.close()


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------

async def open_utp_connection(
    host: str,
    port: int,
    *,
    timeout: float = 10.0,
) -> tuple[asyncio.StreamReader, UTPWriter]:
    """Open a uTP connection and return (reader, writer).

    The returned pair is compatible with ``PeerConnection._from_streams()``.
    Raises UTPError on connection failure.
    """
    conn = await UTPConnection.connect(host, port, timeout=timeout)
    return conn.reader, conn.make_writer()
