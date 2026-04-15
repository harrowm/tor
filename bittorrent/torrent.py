"""
.torrent file parser.

Parses a .torrent file (bencoded dict) and exposes typed fields.

Critical: info_hash is computed by SHA-1 hashing the *raw bencoded bytes*
of the 'info' dict, not the decoded dict or the whole file. We locate those
bytes by tracking byte offsets during decode.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

from bittorrent.bencode import DecodeError, _decode_next


@dataclass
class FileInfo:
    """Represents one file inside a multi-file torrent."""
    path: list[str]   # path components, e.g. ["dir", "file.txt"]
    length: int       # file size in bytes


@dataclass
class Torrent:
    """Parsed representation of a .torrent file."""

    # --- tracker ---
    announce: str
    announce_list: list[list[str]] = field(default_factory=list)

    # --- identity ---
    info_hash: bytes = b""        # 20-byte SHA-1 of bencoded info dict
    info_hash_hex: str = ""       # hex string for display/URLs

    # --- content ---
    name: str = ""
    piece_length: int = 0         # bytes per piece
    piece_hashes: list[bytes] = field(default_factory=list)  # 20-byte SHA-1 each

    # Single-file torrents
    length: int = 0               # 0 for multi-file

    # Multi-file torrents (empty for single-file)
    files: list[FileInfo] = field(default_factory=list)

    # --- derived ---
    @property
    def total_length(self) -> int:
        """Total download size in bytes."""
        if self.files:
            return sum(f.length for f in self.files)
        return self.length

    @property
    def num_pieces(self) -> int:
        return math.ceil(self.total_length / self.piece_length)

    @property
    def is_multi_file(self) -> bool:
        return bool(self.files)


class ParseError(Exception):
    """Raised when a .torrent file cannot be parsed."""


def load(path: str | Path) -> Torrent:
    """Parse a .torrent file from disk and return a Torrent."""
    data = Path(path).read_bytes()
    return parse(data)


def parse(data: bytes) -> Torrent:
    """Parse raw bencoded .torrent bytes and return a Torrent.

    The info_hash is computed from the exact bytes of the bencoded 'info'
    dict as they appear in *data* — this is the canonical torrent identity.
    """
    try:
        meta, _ = _decode_next(data, 0)
    except (DecodeError, ValueError) as exc:
        raise ParseError(f"Cannot decode torrent file: {exc}") from exc

    if not isinstance(meta, dict):
        raise ParseError("Torrent file must be a bencoded dict at the top level")

    # --- announce (optional per BEP 12 — may be absent in trackerless torrents) ---
    announce = ""
    if b"announce" in meta:
        announce_raw = meta[b"announce"]
        if not isinstance(announce_raw, bytes):
            raise ParseError("'announce' must be a byte string")
        try:
            announce = announce_raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ParseError("announce URL is not valid UTF-8")

    # --- announce-list (BEP 12, optional) ---
    announce_list: list[list[str]] = []
    if b"announce-list" in meta:
        raw_list = meta[b"announce-list"]
        if not isinstance(raw_list, list):
            raise ParseError("announce-list must be a list")
        for tier in raw_list:
            if not isinstance(tier, list):
                raise ParseError("Each announce-list tier must be a list")
            decoded_tier = []
            for url in tier:
                if not isinstance(url, bytes):
                    raise ParseError("announce-list URLs must be byte strings")
                decoded_tier.append(url.decode("utf-8"))
            announce_list.append(decoded_tier)

    # If announce is absent but announce-list is present, use the first entry
    # so the rest of the code always has a primary tracker URL to work with.
    if not announce and announce_list:
        announce = announce_list[0][0] if announce_list[0] else ""

    # --- info dict ---
    if b"info" not in meta:
        raise ParseError("Missing 'info' key")
    info = meta[b"info"]
    if not isinstance(info, dict):
        raise ParseError("'info' must be a dict")

    # Compute info_hash from the raw bencoded bytes of the info dict.
    # We find those bytes by locating 'info' key inside the file.
    info_bytes = _extract_info_bytes(data)
    info_hash = hashlib.sha1(info_bytes).digest()

    # --- name ---
    name_raw = _require(info, b"name", bytes, "info.name")
    name = name_raw.decode("utf-8", errors="replace")

    # --- piece length ---
    piece_length = _require(info, b"piece length", int, "info.piece length")
    if piece_length <= 0:
        raise ParseError("piece length must be positive")

    # --- pieces (concatenated 20-byte SHA-1 hashes) ---
    pieces_raw = _require(info, b"pieces", bytes, "info.pieces")
    if len(pieces_raw) % 20 != 0:
        raise ParseError(
            f"info.pieces length {len(pieces_raw)} is not a multiple of 20"
        )
    piece_hashes = [pieces_raw[i : i + 20] for i in range(0, len(pieces_raw), 20)]

    # --- single vs multi file ---
    length = 0
    files: list[FileInfo] = []

    if b"files" in info:
        # Multi-file mode
        raw_files = info[b"files"]
        if not isinstance(raw_files, list):
            raise ParseError("info.files must be a list")
        for entry in raw_files:
            if not isinstance(entry, dict):
                raise ParseError("Each entry in info.files must be a dict")
            file_length = _require(entry, b"length", int, "file entry length")
            path_raw = _require(entry, b"path", list, "file entry path")
            path_parts = []
            for part in path_raw:
                if not isinstance(part, bytes):
                    raise ParseError("File path components must be byte strings")
                path_parts.append(part.decode("utf-8", errors="replace"))
            files.append(FileInfo(path=path_parts, length=file_length))
    else:
        # Single-file mode
        length = _require(info, b"length", int, "info.length")

    return Torrent(
        announce=announce,
        announce_list=announce_list,
        info_hash=info_hash,
        info_hash_hex=info_hash.hex(),
        name=name,
        piece_length=piece_length,
        piece_hashes=piece_hashes,
        length=length,
        files=files,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require(d: dict, key: bytes, typ: type, label: str) -> object:
    """Get a required key from a decoded dict and type-check it."""
    if key not in d:
        raise ParseError(f"Missing required field: {label!r}")
    val = d[key]
    if not isinstance(val, typ):
        raise ParseError(
            f"Field {label!r} must be {typ.__name__}, "
            f"got {type(val).__name__}"
        )
    return val


def _extract_info_bytes(data: bytes) -> bytes:
    """Return the exact raw bencoded bytes of the 'info' value in *data*.

    The .torrent file is a bencoded dict. We scan through the dict keys
    looking for '4:info', then capture everything from the start of its
    value to the end of that value using the decoder's position tracking.
    """
    # Top-level dict: d ... e
    # Keys are sorted, so we scan for b'4:info' followed by the value.
    # We use _decode_next to find the exact span of the info value.
    key = b"4:info"
    idx = data.find(key)
    if idx == -1:
        raise ParseError("Cannot locate 'info' key in raw torrent bytes")

    # The info value starts right after '4:info'
    value_start = idx + len(key)
    _, value_end = _decode_next(data, value_start)
    return data[value_start:value_end]
