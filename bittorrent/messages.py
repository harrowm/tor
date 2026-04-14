"""
Peer wire protocol message encoding and decoding (BEP 3).

Wire format:
  Keep-alive:  \x00\x00\x00\x00                     (4 bytes, length=0)
  Message:     <length:4> <id:1> <payload:length-1>

Handshake (special — sent before the message loop):
  \x13 + "BitTorrent protocol" + <reserved:8> + <info_hash:20> + <peer_id:20>
  = 68 bytes total

Block size: always 2^14 = 16,384 bytes per request (except possibly the last
block of a piece, which may be smaller).
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK_SIZE: int = 16_384          # 2^14 — standard request block size

# Message IDs
MSG_CHOKE          = 0
MSG_UNCHOKE        = 1
MSG_INTERESTED     = 2
MSG_NOT_INTERESTED = 3
MSG_HAVE           = 4
MSG_BITFIELD       = 5
MSG_REQUEST        = 6
MSG_PIECE          = 7
MSG_CANCEL         = 8

_MSG_NAMES = {
    0: "CHOKE",
    1: "UNCHOKE",
    2: "INTERESTED",
    3: "NOT_INTERESTED",
    4: "HAVE",
    5: "BITFIELD",
    6: "REQUEST",
    7: "PIECE",
    8: "CANCEL",
}

# Handshake
_PROTOCOL_STRING = b"BitTorrent protocol"
_HANDSHAKE_PREFIX = bytes([len(_PROTOCOL_STRING)]) + _PROTOCOL_STRING
_RESERVED_BYTES   = b"\x00" * 8
HANDSHAKE_LEN     = 68   # 1 + 19 + 8 + 20 + 20


class MessageError(Exception):
    """Raised when a peer message cannot be decoded."""


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------

def encode_handshake(info_hash: bytes, peer_id: bytes) -> bytes:
    """Return the 68-byte handshake payload."""
    if len(info_hash) != 20:
        raise ValueError(f"info_hash must be 20 bytes, got {len(info_hash)}")
    if len(peer_id) != 20:
        raise ValueError(f"peer_id must be 20 bytes, got {len(peer_id)}")
    return _HANDSHAKE_PREFIX + _RESERVED_BYTES + info_hash + peer_id


def decode_handshake(data: bytes) -> tuple[bytes, bytes]:
    """Parse a 68-byte handshake payload.

    Returns (info_hash, peer_id).
    Raises MessageError if the data is malformed.
    """
    if len(data) < HANDSHAKE_LEN:
        raise MessageError(
            f"Handshake too short: expected {HANDSHAKE_LEN} bytes, got {len(data)}"
        )
    pstrlen = data[0]
    if pstrlen != 19:
        raise MessageError(f"Unexpected pstrlen {pstrlen}, expected 19")
    pstr = data[1:20]
    if pstr != _PROTOCOL_STRING:
        raise MessageError(f"Unexpected protocol string: {pstr!r}")
    # bytes 20-27: reserved (ignored for now)
    info_hash = data[28:48]
    peer_id   = data[48:68]
    return info_hash, peer_id


# ---------------------------------------------------------------------------
# Message container
# ---------------------------------------------------------------------------

@dataclass
class PeerMessage:
    """A decoded peer wire protocol message."""
    msg_id: int | None     # None = keep-alive
    payload: bytes = b""

    @property
    def is_keepalive(self) -> bool:
        return self.msg_id is None

    @property
    def name(self) -> str:
        if self.msg_id is None:
            return "KEEP_ALIVE"
        return _MSG_NAMES.get(self.msg_id, f"UNKNOWN({self.msg_id})")

    # --- typed payload accessors ---

    def have_index(self) -> int:
        """For HAVE: piece index."""
        if len(self.payload) != 4:
            raise MessageError(f"HAVE payload must be 4 bytes, got {len(self.payload)}")
        return struct.unpack("!I", self.payload)[0]

    def request_fields(self) -> tuple[int, int, int]:
        """For REQUEST / CANCEL: (piece_index, block_offset, block_length)."""
        if len(self.payload) != 12:
            raise MessageError(f"REQUEST payload must be 12 bytes, got {len(self.payload)}")
        return struct.unpack("!III", self.payload)

    def piece_fields(self) -> tuple[int, int, bytes]:
        """For PIECE: (piece_index, block_offset, data)."""
        if len(self.payload) < 8:
            raise MessageError(f"PIECE payload too short: {len(self.payload)} bytes")
        piece_index, block_offset = struct.unpack("!II", self.payload[:8])
        return piece_index, block_offset, self.payload[8:]


# ---------------------------------------------------------------------------
# Encoders — one function per message type for clarity
# ---------------------------------------------------------------------------

def _encode(msg_id: int, payload: bytes = b"") -> bytes:
    """Low-level: pack length + id + payload."""
    length = 1 + len(payload)
    return struct.pack("!I", length) + bytes([msg_id]) + payload


def encode_keepalive() -> bytes:
    return b"\x00\x00\x00\x00"


def encode_choke() -> bytes:
    return _encode(MSG_CHOKE)


def encode_unchoke() -> bytes:
    return _encode(MSG_UNCHOKE)


def encode_interested() -> bytes:
    return _encode(MSG_INTERESTED)


def encode_not_interested() -> bytes:
    return _encode(MSG_NOT_INTERESTED)


def encode_have(piece_index: int) -> bytes:
    return _encode(MSG_HAVE, struct.pack("!I", piece_index))


def encode_bitfield(bitfield: bytes | bytearray) -> bytes:
    return _encode(MSG_BITFIELD, bytes(bitfield))


def encode_request(piece_index: int, block_offset: int, block_length: int) -> bytes:
    payload = struct.pack("!III", piece_index, block_offset, block_length)
    return _encode(MSG_REQUEST, payload)


def encode_piece(piece_index: int, block_offset: int, data: bytes) -> bytes:
    header = struct.pack("!II", piece_index, block_offset)
    return _encode(MSG_PIECE, header + data)


def encode_cancel(piece_index: int, block_offset: int, block_length: int) -> bytes:
    payload = struct.pack("!III", piece_index, block_offset, block_length)
    return _encode(MSG_CANCEL, payload)


# ---------------------------------------------------------------------------
# Async reader
# ---------------------------------------------------------------------------

async def read_message(reader: asyncio.StreamReader) -> PeerMessage:
    """Read one message from *reader* and return a PeerMessage.

    Raises MessageError on malformed data, EOFError on connection close.
    """
    try:
        length_bytes = await reader.readexactly(4)
    except asyncio.IncompleteReadError as exc:
        raise EOFError("Connection closed while reading message length") from exc

    (length,) = struct.unpack("!I", length_bytes)

    if length == 0:
        return PeerMessage(None)   # keep-alive

    try:
        data = await reader.readexactly(length)
    except asyncio.IncompleteReadError as exc:
        raise EOFError(
            f"Connection closed while reading message body (expected {length} bytes)"
        ) from exc

    if not data:
        raise MessageError("Message data is empty after non-zero length")

    msg_id  = data[0]
    payload = data[1:]
    return PeerMessage(msg_id, payload)
