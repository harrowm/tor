"""
Peer wire protocol message encoding and decoding (BEP 3 + BEP 6).

Wire format:
  Keep-alive:  \x00\x00\x00\x00                     (4 bytes, length=0)
  Message:     <length:4> <id:1> <payload:length-1>

Handshake (special — sent before the message loop):
  \x13 + "BitTorrent protocol" + <reserved:8> + <info_hash:20> + <peer_id:20>
  = 68 bytes total

Block size: always 2^14 = 16,384 bytes per request (except possibly the last
block of a piece, which may be smaller).

BEP 6 — Fast Extension:
  Adds HAVE_ALL (0x04 in reserved[7]), HAVE_NONE, SUGGEST_PIECE,
  REJECT_REQUEST, and ALLOWED_FAST messages.  These allow seeders to replace
  the BITFIELD with a single HAVE_ALL message and let a peer explicitly
  reject a request rather than silently ignoring it.
"""

from __future__ import annotations

import asyncio
import socket
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
# BEP 6 — Fast Extension
MSG_SUGGEST_PIECE  = 13
MSG_HAVE_ALL       = 14
MSG_HAVE_NONE      = 15
MSG_REJECT_REQUEST = 16
MSG_ALLOWED_FAST   = 17
MSG_EXTENDED       = 20   # BEP 10 extension protocol

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
    13: "SUGGEST_PIECE",
    14: "HAVE_ALL",
    15: "HAVE_NONE",
    16: "REJECT_REQUEST",
    17: "ALLOWED_FAST",
    20: "EXTENDED",
}

# BEP 10: extension protocol bit is bit 20 from the right in the 8-byte
# reserved field, which maps to reserved[5] & 0x10.
EXT_PROTOCOL_RESERVED = b"\x00\x00\x00\x00\x00\x10\x00\x00"

# BEP 6: Fast Extension bit is the LSB of reserved[7] (bit 2 from the right
# of the full 64-bit reserved field).
FAST_EXT_RESERVED = b"\x00\x00\x00\x00\x00\x00\x00\x04"

# Combined reserved bytes: BEP 10 extension protocol + BEP 6 Fast Extension
EXT_AND_FAST_RESERVED = b"\x00\x00\x00\x00\x00\x10\x00\x04"

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

def encode_handshake(
    info_hash: bytes,
    peer_id: bytes,
    *,
    reserved: bytes = _RESERVED_BYTES,
) -> bytes:
    """Return the 68-byte handshake payload.

    Args:
        reserved: 8-byte reserved field (default: all zeros).
                  Pass EXT_PROTOCOL_RESERVED to advertise BEP 10 support.
    """
    if len(info_hash) != 20:
        raise ValueError(f"info_hash must be 20 bytes, got {len(info_hash)}")
    if len(peer_id) != 20:
        raise ValueError(f"peer_id must be 20 bytes, got {len(peer_id)}")
    if len(reserved) != 8:
        raise ValueError(f"reserved must be 8 bytes, got {len(reserved)}")
    return _HANDSHAKE_PREFIX + reserved + info_hash + peer_id


def decode_handshake(data: bytes) -> tuple[bytes, bytes]:
    """Parse a 68-byte handshake payload.

    Returns (info_hash, peer_id).  Use decode_handshake_full() to also get
    the reserved bytes (needed to detect BEP 10 support).
    Raises MessageError if the data is malformed.
    """
    info_hash, peer_id, _ = decode_handshake_full(data)
    return info_hash, peer_id


def decode_handshake_full(data: bytes) -> tuple[bytes, bytes, bytes]:
    """Parse a 68-byte handshake payload.

    Returns (info_hash, peer_id, reserved).
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
    reserved  = data[20:28]
    info_hash = data[28:48]
    peer_id   = data[48:68]
    return info_hash, peer_id, reserved


def supports_extension_protocol(reserved: bytes) -> bool:
    """Return True if the reserved bytes indicate BEP 10 extension protocol support."""
    return len(reserved) >= 6 and bool(reserved[5] & 0x10)


def supports_fast_extension(reserved: bytes) -> bool:
    """Return True if the reserved bytes indicate BEP 6 Fast Extension support."""
    return len(reserved) >= 8 and bool(reserved[7] & 0x04)


def encode_extended(ext_id: int, payload: bytes) -> bytes:
    """Encode a BEP 10 extended message (message ID 20).

    Args:
        ext_id:  Extension message ID (0 for the extension handshake).
        payload: Raw bytes (typically a bencoded dict for the handshake,
                 or bencoded dict + raw data for ut_metadata).
    """
    return _encode(MSG_EXTENDED, bytes([ext_id]) + payload)


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
# BEP 6 — Fast Extension encoders
# ---------------------------------------------------------------------------

def encode_have_all() -> bytes:
    """HAVE_ALL: peer has every piece (replaces full BITFIELD for seeders)."""
    return _encode(MSG_HAVE_ALL)


def encode_have_none() -> bytes:
    """HAVE_NONE: peer has no pieces (replaces empty BITFIELD for new leechers)."""
    return _encode(MSG_HAVE_NONE)


def encode_suggest_piece(piece_index: int) -> bytes:
    """SUGGEST_PIECE: advisory hint that this piece would be useful to request."""
    return _encode(MSG_SUGGEST_PIECE, struct.pack("!I", piece_index))


def encode_reject_request(piece_index: int, block_offset: int, block_length: int) -> bytes:
    """REJECT_REQUEST: explicit refusal of a REQUEST (e.g. while choked)."""
    payload = struct.pack("!III", piece_index, block_offset, block_length)
    return _encode(MSG_REJECT_REQUEST, payload)


def encode_allowed_fast(piece_index: int) -> bytes:
    """ALLOWED_FAST: tell the peer it may request this piece even while choked."""
    return _encode(MSG_ALLOWED_FAST, struct.pack("!I", piece_index))


# ---------------------------------------------------------------------------
# Async reader
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# BEP 11 — Peer Exchange (PEX)
# ---------------------------------------------------------------------------

# Local extension message ID we assign to ut_pex in our extension handshake.
# When we advertise {"m": {"ut_pex": PEX_LOCAL_ID}}, the remote peer uses this
# ID to send us PEX messages.
PEX_LOCAL_ID: int = 1


def decode_pex_peers(compact: bytes) -> list[tuple[str, int]]:
    """Decode a compact IPv4 peer list from a PEX 'added' field.

    Each entry is 6 bytes: 4 bytes big-endian IPv4 + 2 bytes big-endian port.
    Trailing partial entries are silently ignored.
    """
    peers: list[tuple[str, int]] = []
    for i in range(0, len(compact) - 5, 6):
        ip = socket.inet_ntoa(compact[i : i + 4])
        (port,) = struct.unpack("!H", compact[i + 4 : i + 6])
        if port > 0:
            peers.append((ip, port))
    return peers


def encode_pex_peers(peers: list[tuple[str, int]]) -> bytes:
    """Encode a list of (ip, port) peers in compact IPv4 format for PEX."""
    result = bytearray()
    for ip, port in peers:
        result += socket.inet_aton(ip)
        result += struct.pack("!H", port)
    return bytes(result)


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
