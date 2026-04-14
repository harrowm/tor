"""
Bencoding encoder and decoder.

Bencode is the serialization format used by BitTorrent (BEP 3).

Four data types:
  - Integers:     i<decimal>e          e.g. i42e, i-7e
  - Byte strings: <length>:<data>      e.g. 4:spam
  - Lists:        l<items>e            e.g. li1e4:spame
  - Dicts:        d<key><value>...e    e.g. d4:spami42ee  (keys must be byte strings, sorted)

All decoded strings are returned as bytes. Callers should decode to str where needed.
"""

from __future__ import annotations


class DecodeError(Exception):
    """Raised when input cannot be decoded as valid bencode."""


class EncodeError(Exception):
    """Raised when a value cannot be encoded as bencode."""


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def decode(data: bytes) -> object:
    """Decode a bencoded byte string and return the Python value.

    Raises DecodeError on malformed input.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise DecodeError(f"Expected bytes, got {type(data).__name__}")
    value, index = _decode_next(data, 0)
    if index != len(data):
        raise DecodeError(
            f"Trailing data after decoded value at index {index}"
        )
    return value


def _decode_next(data: bytes, index: int) -> tuple[object, int]:
    """Decode one value starting at *index*, return (value, next_index)."""
    if index >= len(data):
        raise DecodeError("Unexpected end of data")

    ch = data[index : index + 1]

    if ch == b"i":
        return _decode_int(data, index)
    if ch == b"l":
        return _decode_list(data, index)
    if ch == b"d":
        return _decode_dict(data, index)
    if ch.isdigit():
        return _decode_string(data, index)

    raise DecodeError(
        f"Invalid token {ch!r} at index {index}"
    )


def _decode_int(data: bytes, index: int) -> tuple[int, int]:
    """Decode i<decimal>e starting at *index*."""
    try:
        end = data.index(b"e", index + 1)
    except ValueError:
        raise DecodeError(f"Unterminated integer at index {index}")
    raw = data[index + 1 : end]
    if not raw:
        raise DecodeError(f"Empty integer at index {index}")
    # Leading zeros and negative zero are illegal in bencode.
    if (raw.startswith(b"0") and len(raw) > 1) or raw == b"-0":
        raise DecodeError(f"Invalid integer encoding {raw!r} at index {index}")
    try:
        return int(raw), end + 1
    except ValueError:
        raise DecodeError(f"Cannot parse integer {raw!r} at index {index}")


def _decode_string(data: bytes, index: int) -> tuple[bytes, int]:
    """Decode <length>:<data> starting at *index*."""
    try:
        colon = data.index(b":", index)
    except ValueError:
        raise DecodeError(f"Missing colon in string at index {index}")
    length_bytes = data[index:colon]
    if not length_bytes.isdigit():
        raise DecodeError(
            f"Invalid string length {length_bytes!r} at index {index}"
        )
    length = int(length_bytes)
    start = colon + 1
    end = start + length
    if end > len(data):
        raise DecodeError(
            f"String length {length} exceeds available data at index {index}"
        )
    return data[start:end], end


def _decode_list(data: bytes, index: int) -> tuple[list, int]:
    """Decode l<items>e starting at *index*."""
    result = []
    index += 1  # skip 'l'
    while index < len(data) and data[index : index + 1] != b"e":
        value, index = _decode_next(data, index)
        result.append(value)
    if index >= len(data):
        raise DecodeError("Unterminated list")
    return result, index + 1  # skip 'e'


def _decode_dict(data: bytes, index: int) -> tuple[dict, int]:
    """Decode d<key><value>...e starting at *index*.

    Keys must be byte strings and must appear in sorted order (BEP 3).
    We enforce sorted order on decode so we catch malformed torrents early.
    """
    result: dict[bytes, object] = {}
    index += 1  # skip 'd'
    last_key: bytes | None = None

    while index < len(data) and data[index : index + 1] != b"e":
        # Keys must be byte strings
        if not data[index : index + 1].isdigit():
            raise DecodeError(
                f"Dict key must be a byte string at index {index}"
            )
        key, index = _decode_string(data, index)
        if last_key is not None and key <= last_key:
            raise DecodeError(
                f"Dict keys not in sorted order: {key!r} after {last_key!r}"
            )
        last_key = key
        value, index = _decode_next(data, index)
        result[key] = value

    if index >= len(data):
        raise DecodeError("Unterminated dict")
    return result, index + 1  # skip 'e'


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def encode(value: object) -> bytes:
    """Encode a Python value as bencode bytes.

    Supported types: int, bytes, list, dict.
    str values are encoded as UTF-8 bytes.
    Dict keys must be bytes or str; they will be sorted before encoding.

    Raises EncodeError for unsupported types.
    """
    if isinstance(value, bool):
        # bool is a subclass of int in Python — reject it explicitly because
        # bencode has no boolean type and silently encoding True as i1e is
        # almost certainly a caller bug.
        raise EncodeError("bool is not a supported bencode type; use int")
    if isinstance(value, int):
        return b"i" + str(value).encode() + b"e"
    if isinstance(value, bytes):
        return str(len(value)).encode() + b":" + value
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        return str(len(encoded)).encode() + b":" + encoded
    if isinstance(value, (list, tuple)):
        return b"l" + b"".join(encode(item) for item in value) + b"e"
    if isinstance(value, dict):
        # Keys must be bytes or str; sort as bytes
        items = {}
        for k, v in value.items():
            if isinstance(k, str):
                k = k.encode("utf-8")
            elif not isinstance(k, bytes):
                raise EncodeError(
                    f"Dict keys must be str or bytes, got {type(k).__name__}"
                )
            items[k] = v
        sorted_pairs = sorted(items.items())
        return (
            b"d"
            + b"".join(encode(k) + encode(v) for k, v in sorted_pairs)
            + b"e"
        )
    raise EncodeError(f"Cannot encode type {type(value).__name__}")
