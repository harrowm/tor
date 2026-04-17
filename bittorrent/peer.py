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
    EXT_AND_FAST_RESERVED,
    EXT_PROTOCOL_RESERVED,
    HANDSHAKE_LEN,
    MSG_BITFIELD,
    MSG_CHOKE,
    MSG_EXTENDED,
    MSG_HAVE,
    MSG_HAVE_ALL,
    MSG_HAVE_NONE,
    MSG_INTERESTED,
    MSG_NOT_INTERESTED,
    MSG_PIECE,
    MSG_REJECT_REQUEST,
    MSG_REQUEST,
    MSG_UNCHOKE,
    PeerMessage,
    decode_handshake_full,
    encode_bitfield,
    encode_cancel,
    encode_choke,
    encode_extended,
    encode_handshake,
    encode_have,
    encode_have_all,
    encode_have_none,
    encode_interested,
    encode_piece,
    encode_reject_request,
    encode_request,
    encode_unchoke,
    read_message,
    supports_extension_protocol,
    supports_fast_extension,
)

log = logging.getLogger(__name__)

BLOCK_TIMEOUT = 30.0  # seconds to wait for a single block response


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
        self.am_choked: bool = True          # we are choked by the remote peer
        self.am_choking: bool = True         # we are choking the remote peer
        self.peer_interested: bool = False   # remote peer has expressed interest in us
        self.bitfield: bytearray = bytearray()   # one bit per piece
        self._pending: list[PeerMessage] = []    # messages read ahead, not yet consumed

        # BEP 10 extension protocol state (populated after do_extension_handshake)
        self.remote_supports_extensions: bool = False
        self._peer_ext_ids: dict[bytes, int] = {}  # extension name -> peer's msg id
        self.metadata_size: int = 0                # peer-reported info dict size

        # BEP 6 Fast Extension state
        self.remote_supports_fast: bool = False
        self.num_pieces: int = 0   # set from piece_manager; used for HAVE_ALL bitfield

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
        extension_protocol: bool = False,
        num_pieces: int = 0,
    ) -> "PeerConnection":
        """Connect to *host*:*port* and complete the handshake.

        Args:
            extension_protocol: Advertise BEP 10 extension protocol support
                                 in the reserved bytes.  Check
                                 ``conn.remote_supports_extensions`` afterwards,
                                 then call ``conn.do_extension_handshake()`` to
                                 negotiate specific extensions.
            num_pieces:         Total number of pieces in the torrent.  Required
                                 so that a BEP 6 HAVE_ALL message can be
                                 converted to a full bitfield.

        Raises PeerError on connection failure or handshake mismatch.
        """
        conn = cls(host, port)
        conn.num_pieces = num_pieces
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            raise PeerError(f"Cannot connect to {host}:{port}: {exc}") from exc

        conn._reader = reader
        conn._writer = writer
        await conn._handshake(info_hash, peer_id, extension_protocol=extension_protocol)
        return conn

    @classmethod
    def _from_streams(
        cls,
        host: str,
        port: int,
        reader: asyncio.StreamReader,
        writer,  # asyncio.StreamWriter or UTPWriter
    ) -> "PeerConnection":
        """Create a PeerConnection from existing streams (for tests / uTP)."""
        conn = cls(host, port)
        conn._reader = reader
        conn._writer = writer
        return conn

    @classmethod
    async def open_utp(
        cls,
        host: str,
        port: int,
        info_hash: bytes,
        peer_id: bytes,
        *,
        timeout: float = 10.0,
        extension_protocol: bool = False,
        num_pieces: int = 0,
    ) -> "PeerConnection":
        """Connect to *host*:*port* via uTP (BEP 29) and complete the handshake.

        Raises PeerError on connection failure or handshake mismatch.
        """
        from bittorrent.utp import UTPError, open_utp_connection

        try:
            reader, writer = await open_utp_connection(host, port, timeout=timeout)
        except UTPError as exc:
            raise PeerError(f"uTP connect to {host}:{port} failed: {exc}") from exc
        except OSError as exc:
            raise PeerError(f"uTP socket error connecting to {host}:{port}: {exc}") from exc

        conn = cls._from_streams(host, port, reader, writer)
        conn.num_pieces = num_pieces
        await conn._handshake(info_hash, peer_id, extension_protocol=extension_protocol)
        return conn

    @classmethod
    async def accept(
        cls,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        info_hash: bytes,
        peer_id: bytes,
        *,
        extension_protocol: bool = False,
    ) -> "PeerConnection":
        """Accept an incoming peer connection.

        Reads the remote handshake first, validates the info_hash, then sends
        our handshake back.  This is the mirror image of ``open()``.

        Raises PeerError if the remote handshake is invalid or info_hash mismatches.
        """
        try:
            data = await asyncio.wait_for(
                reader.readexactly(HANDSHAKE_LEN),
                timeout=10.0,
            )
        except asyncio.IncompleteReadError as exc:
            raise PeerError("Peer closed connection during handshake") from exc
        except asyncio.TimeoutError:
            raise PeerError("Incoming handshake timed out")

        try:
            their_hash, their_id, their_reserved = decode_handshake_full(data)
        except Exception as exc:
            raise PeerError(f"Bad incoming handshake: {exc}") from exc

        if their_hash != info_hash:
            raise PeerError(
                f"info_hash mismatch from incoming peer: "
                f"expected {info_hash.hex()}, got {their_hash.hex()}"
            )

        peername = writer.get_extra_info("peername", ("unknown", 0))
        host, port = peername[0], int(peername[1])

        conn = cls(host, port)
        conn._reader = reader
        conn._writer = writer
        conn.remote_peer_id = their_id
        conn.remote_supports_extensions = supports_extension_protocol(their_reserved)
        conn.remote_supports_fast = supports_fast_extension(their_reserved)

        if extension_protocol:
            await conn._send_raw(
                encode_handshake(info_hash, peer_id, reserved=EXT_AND_FAST_RESERVED)
            )
        else:
            await conn._send_raw(encode_handshake(info_hash, peer_id))

        return conn

    # ------------------------------------------------------------------
    # Upload / seeding methods
    # ------------------------------------------------------------------

    async def send_have(self, piece_index: int) -> None:
        """Send a HAVE message to notify the peer we now have *piece_index*."""
        await self._send_raw(encode_have(piece_index))

    async def send_bitfield(self, bitfield: bytes | bytearray) -> None:
        """Send our BITFIELD to the peer."""
        await self._send_raw(encode_bitfield(bitfield))

    async def send_have_all(self) -> None:
        """BEP 6: send HAVE_ALL (we are a seeder with all pieces)."""
        await self._send_raw(encode_have_all())

    async def send_have_none(self) -> None:
        """BEP 6: send HAVE_NONE (we have no pieces yet)."""
        await self._send_raw(encode_have_none())

    async def send_reject_request(
        self, piece_index: int, block_offset: int, block_length: int
    ) -> None:
        """BEP 6: send REJECT_REQUEST to explicitly decline a block request."""
        await self._send_raw(encode_reject_request(piece_index, block_offset, block_length))

    async def send_choke(self) -> None:
        """Choke the remote peer (stop serving their requests)."""
        self.am_choking = True
        await self._send_raw(encode_choke())

    async def send_unchoke(self) -> None:
        """Unchoke the remote peer (allow them to request blocks)."""
        self.am_choking = False
        await self._send_raw(encode_unchoke())

    async def send_piece_block(
        self,
        piece_index: int,
        block_offset: int,
        data: bytes,
    ) -> None:
        """Send a PIECE block in response to a REQUEST."""
        await self._send_raw(encode_piece(piece_index, block_offset, data))

    async def read_request(self) -> tuple[int, int, int]:
        """Read the next REQUEST from this peer, skipping non-request messages.

        Updates ``peer_interested`` when INTERESTED / NOT_INTERESTED arrives.
        Returns ``(piece_index, block_offset, block_length)``.
        Raises PeerError on connection close or error.
        """
        while True:
            msg = await self._read_next()
            if msg.msg_id == MSG_REQUEST:
                return msg.request_fields()
            elif msg.msg_id == MSG_INTERESTED:
                self.peer_interested = True
            elif msg.msg_id == MSG_NOT_INTERESTED:
                self.peer_interested = False
            elif msg.msg_id == MSG_HAVE:
                self._apply_have(msg)
            elif msg.msg_id == MSG_CHOKE:
                self.am_choked = True
            elif msg.msg_id == MSG_UNCHOKE:
                self.am_choked = False
            # keep-alives, BITFIELD updates, etc. are silently skipped

    # ------------------------------------------------------------------
    # Handshake
    # ------------------------------------------------------------------

    async def _handshake(
        self,
        info_hash: bytes,
        peer_id: bytes,
        *,
        extension_protocol: bool = False,
    ) -> None:
        """Send our handshake and validate the peer's response."""
        if extension_protocol:
            # Always advertise both BEP 10 and BEP 6 together
            await self._send_raw(
                encode_handshake(info_hash, peer_id, reserved=EXT_AND_FAST_RESERVED)
            )
        else:
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
        except OSError as exc:
            raise PeerError(f"Handshake failed: {exc}") from exc

        try:
            their_hash, their_id, their_reserved = decode_handshake_full(data)
        except Exception as exc:
            raise PeerError(f"Bad handshake from peer: {exc}") from exc

        if their_hash != info_hash:
            raise PeerError(
                f"info_hash mismatch: expected {info_hash.hex()}, "
                f"got {their_hash.hex()}"
            )

        self.remote_peer_id = their_id
        self.remote_supports_extensions = supports_extension_protocol(their_reserved)
        self.remote_supports_fast = supports_fast_extension(their_reserved)
        log.debug("Handshake OK with %s:%s peer_id=%s ext=%s fast=%s",
                  self.host, self.port, their_id.hex(),
                  self.remote_supports_extensions, self.remote_supports_fast)

        # Peers often send a BITFIELD immediately after their handshake.
        # Read it if present (it's optional per BEP 3).
        await self._maybe_read_bitfield()

        # Tell the peer we're interested right away.  Seeders close idle
        # connections within a second or two of the handshake; delaying
        # INTERESTED until download_piece() is called means we wait the full
        # extension-handshake round-trip before the peer even considers
        # unchocking us.
        await self._send_raw(encode_interested())

    async def _maybe_read_bitfield(self) -> None:
        """If the next message is a BITFIELD/HAVE_ALL/HAVE_NONE, read and store it.

        Non-BITFIELD messages are placed in self._pending so _read_next()
        can return them later — we never push data back onto the stream reader.
        BEP 6: HAVE_ALL and HAVE_NONE replace BITFIELD for Fast Extension peers.
        """
        try:
            msg = await asyncio.wait_for(read_message(self._reader), timeout=3.0)
        except (asyncio.TimeoutError, EOFError):
            return  # no message waiting or stream already closed — fine

        if msg.msg_id == MSG_BITFIELD:
            self.bitfield = bytearray(msg.payload)
            log.debug("Got BITFIELD from %s:%s (%d bytes)", self.host, self.port,
                      len(msg.payload))
        elif msg.msg_id == MSG_HAVE_ALL:
            # BEP 6: peer has all pieces — synthesize a full bitfield
            if self.num_pieces > 0:
                n_bytes = (self.num_pieces + 7) // 8
                self.bitfield = bytearray(b"\xff" * n_bytes)
            log.debug("Got HAVE_ALL from %s:%s", self.host, self.port)
        elif msg.msg_id == MSG_HAVE_NONE:
            # BEP 6: peer has no pieces
            self.bitfield = bytearray()
            log.debug("Got HAVE_NONE from %s:%s", self.host, self.port)
        elif msg.msg_id == MSG_UNCHOKE:
            # Seeder may pre-emptively unchoke us right after handshake.
            self.am_choked = False
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
        *,
        block_timeout: float = BLOCK_TIMEOUT,
        completion_check: "callable[[int], bool] | None" = None,
    ) -> bytes:
        """Download one piece from this peer and verify its SHA-1 hash.

        Sends INTERESTED, waits for UNCHOKE, pipelines all block REQUESTs,
        collects PIECE messages, assembles and hash-verifies the result.

        Args:
            block_timeout:    Seconds to wait for each individual block response.
                              Raises PeerError if a peer stalls mid-piece.
            completion_check: Optional callable(piece_index) -> bool.  Called
                              after each block arrives; if it returns True the
                              piece was completed by another peer (end-game race).
                              CANCEL is sent for remaining blocks and PeerError
                              is raised with the message "piece already complete".

        Raises PeerError on choke, hash mismatch, timeout, unexpected EOF, or
        when another peer wins the end-game race (completion_check returns True).
        """
        # Send INTERESTED and wait for UNCHOKE only when still choked.
        # The handshake path sends INTERESTED early; by the time
        # download_piece() is called we may already be unchoked.
        if self.am_choked:
            await self._send_raw(encode_interested())
            log.info("Waiting for unchoke from %s:%s", self.host, self.port)
            await self._wait_for_unchoke()
            log.info("Unchoked by %s:%s", self.host, self.port)

        # Calculate block spans for this piece
        blocks = _block_spans(piece_length)

        # Pipeline all requests
        for offset, length in blocks:
            await self._send_raw(encode_request(piece_index, offset, length))

        # Collect PIECE responses
        received: dict[int, bytes] = {}   # block_offset -> data
        while len(received) < len(blocks):
            # End-game: another worker finished this piece — cancel our requests.
            if completion_check and completion_check(piece_index):
                for offset, length in blocks:
                    if offset not in received:
                        try:
                            await self._send_raw(
                                encode_cancel(piece_index, offset, length)
                            )
                        except (PeerError, OSError):
                            pass
                raise PeerError(
                    f"piece {piece_index} already complete (end-game cancel)"
                )

            try:
                msg = await asyncio.wait_for(self._read_next(), timeout=block_timeout)
            except asyncio.TimeoutError:
                raise PeerError(
                    f"Timed out waiting for block from piece {piece_index}"
                )
            if msg.msg_id == MSG_PIECE:
                idx, block_offset, data = msg.piece_fields()
                if idx == piece_index:
                    received[block_offset] = data
                # Silently drop PIECE messages for other piece indices
            elif msg.msg_id == MSG_CHOKE:
                self.am_choked = True
                raise PeerError(f"Peer choked us while downloading piece {piece_index}")
            elif msg.msg_id == MSG_REJECT_REQUEST:
                # BEP 6: peer explicitly declined this specific request.
                # Treat it like a choke so the caller can wait for unchoke.
                raise PeerError(
                    f"Peer rejected request for piece {piece_index} (BEP 6)"
                )
            elif msg.msg_id == MSG_EXTENDED:
                # Buffer extended messages (e.g. PEX) for processing between pieces
                self._pending.append(msg)
            # Ignore HAVE, UNCHOKE, BITFIELD, ALLOWED_FAST during download

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

    async def _wait_for_unchoke(self, timeout: float = 90.0) -> None:
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
        try:
            self._writer.write(data)
            await self._writer.drain()
        except OSError as exc:
            raise PeerError(f"Write failed to {self.host}:{self.port}: {exc}") from exc

    async def _read_next(self) -> PeerMessage:
        if self._pending:
            return self._pending.pop(0)
        try:
            return await read_message(self._reader)
        except (EOFError, OSError) as exc:
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
    # BEP 10 extension protocol
    # ------------------------------------------------------------------

    async def do_extension_handshake(
        self,
        extensions: dict[bytes, int],
        *,
        timeout: float = 15.0,
    ) -> None:
        """Send and receive the BEP 10 extension handshake.

        Sends our extension capabilities and parses the peer's response.
        After this call, ``peer_ext_id(name)`` and ``self.metadata_size``
        are populated.

        Args:
            extensions: Extension names we support, e.g. ``{b"ut_metadata": 1}``.
                        The value is the local message ID we assign.
            timeout:    Seconds to wait for the peer's extension handshake.
        """
        from bittorrent.bencode import encode as _bencode, decode as _bdecode

        payload = _bencode({b"m": extensions})
        await self._send_raw(encode_extended(0, payload))

        deferred: list[PeerMessage] = []
        deadline = asyncio.get_event_loop().time() + timeout
        try:
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise PeerError("Extension handshake timed out")
                try:
                    msg = await asyncio.wait_for(self._read_next(), timeout=remaining)
                except asyncio.TimeoutError:
                    raise PeerError("Extension handshake timed out")

                if (msg.msg_id == MSG_EXTENDED
                        and msg.payload
                        and msg.payload[0] == 0):
                    # Extension handshake response
                    try:
                        d = _bdecode(msg.payload[1:])
                    except Exception as exc:
                        raise PeerError(
                            f"Cannot decode extension handshake: {exc}"
                        ) from exc
                    if isinstance(d, dict):
                        m = d.get(b"m")
                        if isinstance(m, dict):
                            self._peer_ext_ids = {
                                k: v for k, v in m.items()
                                if isinstance(k, bytes) and isinstance(v, int)
                            }
                        sz = d.get(b"metadata_size")
                        if isinstance(sz, int):
                            self.metadata_size = sz
                    return
                elif msg.msg_id == MSG_BITFIELD:
                    self.bitfield = bytearray(msg.payload)
                elif msg.msg_id == MSG_UNCHOKE:
                    # Peer unchocked us during extension handshake — apply now.
                    self.am_choked = False
                elif msg.msg_id == MSG_CHOKE:
                    self.am_choked = True
                else:
                    deferred.append(msg)
        finally:
            # Restore any messages we skipped over
            self._pending = deferred + self._pending

    def peer_ext_id(self, name: bytes) -> int | None:
        """Return the peer's message ID for *name*, or None if not supported."""
        val = self._peer_ext_ids.get(name)
        return val if isinstance(val, int) else None

    async def send_extension(self, ext_id: int, payload: bytes) -> None:
        """Send a BEP 10 extended message with *ext_id* and *payload*."""
        await self._send_raw(encode_extended(ext_id, payload))

    async def read_extension_payload(self) -> tuple[int, bytes]:
        """Read the next MSG_EXTENDED from this peer, skipping other messages.

        Returns (ext_id, raw_payload) where raw_payload follows the ext_id byte.
        Raises PeerError on connection close.
        """
        while True:
            msg = await self._read_next()
            if msg.msg_id == MSG_EXTENDED:
                if not msg.payload:
                    raise PeerError("Empty extended message")
                return msg.payload[0], msg.payload[1:]
            # Skip HAVE, UNCHOKE, keepalives, etc.

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying TCP connection."""
        if self._writer:
            try:
                self._writer.close()
            except Exception:
                pass
            # wait_closed() can re-raise transport-level OSError in Python 3.12
            # (e.g. ConnectionResetError) even when the close was intentional.
            # We shield it and swallow all errors — the OS will clean up the fd.
            try:
                await asyncio.wait_for(self._writer.wait_closed(), timeout=2.0)
            except Exception:
                pass
            self._writer = None


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
