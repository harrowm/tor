"""
BEP 9 metadata exchange (ut_metadata extension).

Fetches the info dict from a peer that supports ut_metadata via BEP 10.
The caller is responsible for the extension handshake; this module only
handles the metadata piece request/response loop.

Protocol (BEP 9):
  - Metadata is split into 16 KiB pieces.
  - Request:  bencoded {msg_type: 0, piece: N}
  - Data:     bencoded {msg_type: 1, piece: N, total_size: X} + raw bytes
  - Reject:   bencoded {msg_type: 2, piece: N}
"""

from __future__ import annotations

import hashlib
import math

from bittorrent.bencode import _decode_next, DecodeError, encode as bencode
from bittorrent.peer import PeerConnection, PeerError

_UT_METADATA  = b"ut_metadata"
_PIECE_SIZE   = 16_384   # BEP 9: metadata pieces are 16 KiB

# BEP 9 msg_type values
_MSG_REQUEST = 0
_MSG_DATA    = 1
_MSG_REJECT  = 2


async def fetch_metadata(
    conn: PeerConnection,
    info_hash: bytes,
    *,
    timeout: float = 60.0,
) -> bytes:
    """Fetch and SHA-1-verify the info dict via ut_metadata.

    *conn* must already have completed ``do_extension_handshake()`` so that
    ``conn.peer_ext_id(b"ut_metadata")`` and ``conn.metadata_size`` are set.

    Returns the raw bencoded info bytes on success.

    Raises:
        PeerError: peer doesn't support ut_metadata, rejects a piece,
                   the hash doesn't match, or the fetch times out.
    """
    import asyncio

    ut_id = conn.peer_ext_id(_UT_METADATA)
    if ut_id is None:
        raise PeerError("Peer does not support ut_metadata")
    if not conn.metadata_size:
        raise PeerError("Peer did not advertise metadata_size")

    num_pieces = math.ceil(conn.metadata_size / _PIECE_SIZE)
    pieces: dict[int, bytes] = {}

    # Pipeline: request all pieces before waiting for any response
    for i in range(num_pieces):
        await conn.send_extension(
            ut_id,
            bencode({b"msg_type": _MSG_REQUEST, b"piece": i}),
        )

    # Collect responses
    deadline = asyncio.get_event_loop().time() + timeout
    while len(pieces) < num_pieces:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise PeerError("Metadata fetch timed out")
        ext_id, raw = await asyncio.wait_for(
            conn.read_extension_payload(), timeout=remaining
        )
        if ext_id != ut_id:
            continue  # not our extension; skip

        # Payload: bencoded dict immediately followed by raw piece bytes
        try:
            d, end = _decode_next(raw, 0)
        except DecodeError as exc:
            raise PeerError(f"Cannot decode metadata message: {exc}") from exc

        if not isinstance(d, dict):
            continue

        msg_type  = d.get(b"msg_type")
        piece_idx = d.get(b"piece")

        if msg_type == _MSG_REJECT:
            raise PeerError(f"Peer rejected metadata piece {piece_idx}")
        if msg_type == _MSG_DATA and isinstance(piece_idx, int):
            pieces[piece_idx] = raw[end:]

    # Assemble and trim to declared size (last piece may be shorter)
    result = b"".join(pieces[i] for i in range(num_pieces))
    result = result[:conn.metadata_size]

    # Verify SHA-1 against the known info_hash
    if hashlib.sha1(result).digest() != info_hash:
        raise PeerError(
            "Metadata SHA-1 mismatch — downloaded info dict is corrupt"
        )

    return result
