"""
Single peer connection — handshake, message loop, piece download.

A PeerConnection manages one TCP connection to one peer. It is responsible
for the BitTorrent handshake, receiving the peer's bitfield, sending
interested/request messages, and assembling + verifying one piece at a time.

Design notes:
  - Uses asyncio streams throughout (no threading).
  - PeerConnection.open() is the normal entry point; it connects and
    performs the handshake in one call.
  - For testing, construct with _from_streams() to inject fake reader/writer.
  - download_piece() pipelines all block requests for a piece before waiting
    for piece messages, which is more efficient than request-wait-request.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING

from bittorrent.messages import (
    BLOCK_SIZE,
    HANDSHAKE_LEN,
    MSG_BITFIELD,
    MSG_CHOKE,
    MSG_HAVE,
    MSG_PIECE,
    MSG_UNCHOKE,
    PeerMessage,
    decode_handshake,
    encode_handshake,
    encode_interested,
    encode_request,
    read_message,
)

log = logging.getLogger(__name__)


class PeerError(Exception):
    """Raised when a peer behaves unexpectedly or sends bad data."""


class PeerConnection:
    """Async BitTorrent peer connection."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

        # State set after handshake / initial message exchange
        self.remote_peer_id: bytes | None = None
        self.am_choked: bool = True
        self.bitfield: bytearray = bytearray()   # one bit per piece
        self._pending: list[PeerMessage] = []    # messages read ahead, not yet consumed

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    async def open(
        cls,
        host: str,
        port: int,
        info_hash: bytes,
        peer_id: bytes,
        *,
        timeout: float = 10.0,
    ) -> "PeerConnection":
        """Connect to *host*:*port* and complete the handshake.

        Raises PeerError on connection failure or handshake mismatch.
        """
        conn = cls(host, port)
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            raise PeerError(f"Cannot connect to {host}:{port}: {exc}") from exc

        conn._reader = reader
        conn._writer = writer
        await conn._handshake(info_hash, peer_id)
        return conn

    @classmethod
    def _from_streams(
        cls,
        host: str,
        port: int,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> "PeerConnection":
        """Create a PeerConnection from existing streams (for tests)."""
        conn = cls(host, port)
        conn._reader = reader
        conn._writer = writer
        return conn

    # ------------------------------------------------------------------
    # Handshake
    # ------------------------------------------------------------------

    async def _handshake(self, info_hash: bytes, peer_id: bytes) -> None:
        """Send our handshake and validate the peer's response."""
        await self._send_raw(encode_handshake(info_hash, peer_id))

        try:
            data = await asyncio.wait_for(
                self._reader.readexactly(HANDSHAKE_LEN),
                timeout=10.0,
            )
        except asyncio.IncompleteReadError as exc:
            raise PeerError("Peer closed connection during handshake") from exc
        except asyncio.TimeoutError:
            raise PeerError("Handshake timed out")

        try:
            their_hash, their_id = decode_handshake(data)
        except Exception as exc:
            raise PeerError(f"Bad handshake from peer: {exc}") from exc

        if their_hash != info_hash:
            raise PeerError(
                f"info_hash mismatch: expected {info_hash.hex()}, "
                f"got {their_hash.hex()}"
            )

        self.remote_peer_id = their_id
        log.debug("Handshake OK with %s:%s peer_id=%s", self.host, self.port,
                  their_id.hex())

        # Peers often send a BITFIELD immediately after their handshake.
        # Read it if present (it's optional per BEP 3).
        await self._maybe_read_bitfield()

    async def _maybe_read_bitfield(self) -> None:
        """If the next message is a BITFIELD, read and store it.

        Non-BITFIELD messages are placed in self._pending so _read_next()
        can return them later — we never push data back onto the stream reader.
        """
        try:
            msg = await asyncio.wait_for(read_message(self._reader), timeout=3.0)
        except (asyncio.TimeoutError, EOFError):
            return  # no message waiting or stream already closed — fine

        if msg.msg_id == MSG_BITFIELD:
            self.bitfield = bytearray(msg.payload)
            log.debug("Got BITFIELD from %s:%s (%d bytes)", self.host, self.port,
                      len(msg.payload))
        elif not msg.is_keepalive:
            # Save it so the next _read_next() call returns it.
            self._pending.append(msg)

    # ------------------------------------------------------------------
    # Download a single piece
    # ------------------------------------------------------------------

    async def download_piece(
        self,
        piece_index: int,
        piece_length: int,
        expected_hash: bytes,
    ) -> bytes:
        """Download one piece from this peer and verify its SHA-1 hash.

        Sends INTERESTED, waits for UNCHOKE, pipelines all block REQUESTs,
        collects PIECE messages, assembles and hash-verifies the result.

        Raises PeerError on choke, hash mismatch, or unexpected EOF.
        """
        # Tell the peer we're interested
        await self._send_raw(encode_interested())

        # Wait for UNCHOKE (skip HAVE/BITFIELD messages that may arrive first)
        await self._wait_for_unchoke()

        # Calculate block spans for this piece
        blocks = _block_spans(piece_length)

        # Pipeline all requests
        for offset, length in blocks:
            await self._send_raw(encode_request(piece_index, offset, length))

        # Collect PIECE responses
        received: dict[int, bytes] = {}   # block_offset -> data
        while len(received) < len(blocks):
            msg = await self._read_next()
            if msg.msg_id == MSG_PIECE:
                idx, block_offset, data = msg.piece_fields()
                if idx == piece_index:
                    received[block_offset] = data
                # Silently drop PIECE messages for other piece indices
            elif msg.msg_id == MSG_CHOKE:
                self.am_choked = True
                raise PeerError(f"Peer choked us while downloading piece {piece_index}")
            # Ignore HAVE, UNCHOKE, BITFIELD during download

        # Assemble in order
        piece_data = b"".join(received[offset] for offset, _ in blocks)

        # Verify
        actual = hashlib.sha1(piece_data).digest()
        if actual != expected_hash:
            raise PeerError(
                f"Piece {piece_index} hash mismatch: "
                f"expected {expected_hash.hex()}, got {actual.hex()}"
            )

        log.debug("Piece %d verified OK (%d bytes)", piece_index, len(piece_data))
        return piece_data

    async def _wait_for_unchoke(self, timeout: float = 30.0) -> None:
        """Read messages until UNCHOKE arrives (or timeout)."""
        async def _loop() -> None:
            while True:
                msg = await self._read_next()
                if msg.msg_id == MSG_UNCHOKE:
                    self.am_choked = False
                    return
                if msg.msg_id == MSG_BITFIELD:
                    self.bitfield = bytearray(msg.payload)
                if msg.msg_id == MSG_HAVE:
                    self._apply_have(msg)

        try:
            await asyncio.wait_for(_loop(), timeout=timeout)
        except asyncio.TimeoutError:
            raise PeerError("Timed out waiting for UNCHOKE")

    # ------------------------------------------------------------------
    # Low-level I/O
    # ------------------------------------------------------------------

    async def _send_raw(self, data: bytes) -> None:
        self._writer.write(data)
        await self._writer.drain()

    async def _read_next(self) -> PeerMessage:
        if self._pending:
            return self._pending.pop(0)
        try:
            return await read_message(self._reader)
        except EOFError as exc:
            raise PeerError(f"Connection to {self.host}:{self.port} closed: {exc}") from exc

    def _apply_have(self, msg: PeerMessage) -> None:
        """Mark a piece as available in the peer's bitfield."""
        try:
            piece_index = msg.have_index()
        except Exception:
            return
        byte_index  = piece_index // 8
        bit_offset  = 7 - (piece_index % 8)
        if byte_index < len(self.bitfield):
            self.bitfield[byte_index] |= (1 << bit_offset)

    def has_piece(self, piece_index: int) -> bool:
        """Return True if the peer's bitfield indicates they have this piece."""
        if not self.bitfield:
            return False
        byte_index = piece_index // 8
        bit_offset = 7 - (piece_index % 8)
        if byte_index >= len(self.bitfield):
            return False
        return bool(self.bitfield[byte_index] & (1 << bit_offset))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying TCP connection."""
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _block_spans(piece_length: int) -> list[tuple[int, int]]:
    """Return [(offset, length), ...] covering *piece_length* bytes in BLOCK_SIZE chunks."""
    spans = []
    offset = 0
    while offset < piece_length:
        length = min(BLOCK_SIZE, piece_length - offset)
        spans.append((offset, length))
        offset += length
    return spans
