"""
Tests for bittorrent.bencode — decoder and encoder.

Coverage targets:
  - All four bencode types (int, string, list, dict)
  - Round-trip: encode(decode(x)) == x and decode(encode(x)) == x
  - Edge cases: zero, negative ints, empty string, empty list, empty dict
  - Nested structures
  - Error cases: malformed input, illegal encodings, unsupported types
"""

import pytest
from bittorrent.bencode import decode, encode, DecodeError, EncodeError


# ---------------------------------------------------------------------------
# Decode — integers
# ---------------------------------------------------------------------------

class TestDecodeInt:
    def test_positive(self):
        assert decode(b"i42e") == 42

    def test_zero(self):
        assert decode(b"i0e") == 0

    def test_negative(self):
        assert decode(b"i-7e") == -7

    def test_large(self):
        assert decode(b"i1000000000000e") == 1_000_000_000_000

    def test_negative_simple(self):
        assert decode(b"i-1e") == -1

    def test_leading_zero_rejected(self):
        with pytest.raises(DecodeError):
            decode(b"i01e")

    def test_negative_zero_rejected(self):
        with pytest.raises(DecodeError):
            decode(b"i-0e")

    def test_empty_integer_rejected(self):
        with pytest.raises(DecodeError):
            decode(b"ie")

    def test_unterminated_rejected(self):
        with pytest.raises(DecodeError):
            decode(b"i42")

    def test_non_numeric_rejected(self):
        with pytest.raises(DecodeError):
            decode(b"ixe")


# ---------------------------------------------------------------------------
# Decode — byte strings
# ---------------------------------------------------------------------------

class TestDecodeString:
    def test_simple(self):
        assert decode(b"4:spam") == b"spam"

    def test_empty(self):
        assert decode(b"0:") == b""

    def test_binary_content(self):
        assert decode(b"3:\x00\x01\x02") == b"\x00\x01\x02"

    def test_length_mismatch_rejected(self):
        with pytest.raises(DecodeError):
            decode(b"10:short")

    def test_no_colon_rejected(self):
        with pytest.raises(DecodeError):
            decode(b"4spam")


# ---------------------------------------------------------------------------
# Decode — lists
# ---------------------------------------------------------------------------

class TestDecodeList:
    def test_empty(self):
        assert decode(b"le") == []

    def test_int_list(self):
        assert decode(b"li1ei2ei3ee") == [1, 2, 3]

    def test_string_list(self):
        assert decode(b"l4:spam3:fooe") == [b"spam", b"foo"]

    def test_mixed(self):
        assert decode(b"li42e4:teste") == [42, b"test"]

    def test_nested(self):
        assert decode(b"lli1ei2eeli3eee") == [[1, 2], [3]]

    def test_unterminated_rejected(self):
        with pytest.raises(DecodeError):
            decode(b"li1e")


# ---------------------------------------------------------------------------
# Decode — dicts
# ---------------------------------------------------------------------------

class TestDecodeDict:
    def test_empty(self):
        assert decode(b"de") == {}

    def test_single_entry(self):
        assert decode(b"d4:spami42ee") == {b"spam": 42}

    def test_multiple_entries(self):
        result = decode(b"d3:bar4:spam3:fooi42ee")
        assert result == {b"bar": b"spam", b"foo": 42}

    def test_nested_dict(self):
        result = decode(b"d4:infod6:lengthi100eee")
        assert result == {b"info": {b"length": 100}}

    def test_unsorted_keys_rejected(self):
        # 'z' before 'a' — violates sorted key requirement
        with pytest.raises(DecodeError):
            decode(b"d1:zi1e1:ai2ee")

    def test_duplicate_keys_rejected(self):
        with pytest.raises(DecodeError):
            decode(b"d3:fooi1e3:fooi2ee")

    def test_non_string_key_rejected(self):
        with pytest.raises(DecodeError):
            decode(b"di1e4:spame")

    def test_unterminated_rejected(self):
        with pytest.raises(DecodeError):
            decode(b"d4:spami42e")


# ---------------------------------------------------------------------------
# Decode — structural / top-level errors
# ---------------------------------------------------------------------------

class TestDecodeErrors:
    def test_empty_input(self):
        with pytest.raises(DecodeError):
            decode(b"")

    def test_trailing_data(self):
        with pytest.raises(DecodeError):
            decode(b"i1ei2e")

    def test_invalid_type_token(self):
        with pytest.raises(DecodeError):
            decode(b"x")

    def test_wrong_input_type(self):
        with pytest.raises(DecodeError):
            decode("i42e")  # str instead of bytes

    def test_bytearray_accepted(self):
        # bytearray should work the same as bytes
        assert decode(bytearray(b"i42e")) == 42


# ---------------------------------------------------------------------------
# Encode — integers
# ---------------------------------------------------------------------------

class TestEncodeInt:
    def test_positive(self):
        assert encode(42) == b"i42e"

    def test_zero(self):
        assert encode(0) == b"i0e"

    def test_negative(self):
        assert encode(-7) == b"i-7e"

    def test_large(self):
        assert encode(1_000_000_000_000) == b"i1000000000000e"

    def test_bool_rejected(self):
        with pytest.raises(EncodeError):
            encode(True)
        with pytest.raises(EncodeError):
            encode(False)


# ---------------------------------------------------------------------------
# Encode — strings
# ---------------------------------------------------------------------------

class TestEncodeString:
    def test_bytes(self):
        assert encode(b"spam") == b"4:spam"

    def test_empty_bytes(self):
        assert encode(b"") == b"0:"

    def test_str_encoded_as_utf8(self):
        assert encode("spam") == b"4:spam"

    def test_unicode_str(self):
        # "café" is 5 bytes in UTF-8 (é = 2 bytes)
        assert encode("café") == b"5:caf\xc3\xa9"

    def test_binary(self):
        assert encode(b"\x00\x01\x02") == b"3:\x00\x01\x02"


# ---------------------------------------------------------------------------
# Encode — lists
# ---------------------------------------------------------------------------

class TestEncodeList:
    def test_empty(self):
        assert encode([]) == b"le"

    def test_int_list(self):
        assert encode([1, 2, 3]) == b"li1ei2ei3ee"

    def test_mixed(self):
        assert encode([42, b"test"]) == b"li42e4:teste"

    def test_nested(self):
        assert encode([[1, 2], [3]]) == b"lli1ei2eeli3eee"

    def test_tuple_treated_as_list(self):
        assert encode((1, 2)) == b"li1ei2ee"


# ---------------------------------------------------------------------------
# Encode — dicts
# ---------------------------------------------------------------------------

class TestEncodeDict:
    def test_empty(self):
        assert encode({}) == b"de"

    def test_single(self):
        assert encode({b"spam": 42}) == b"d4:spami42ee"

    def test_keys_sorted(self):
        # Supply in reverse order — encoder must sort them
        result = encode({b"z": 1, b"a": 2})
        assert result == b"d1:ai2e1:zi1ee"

    def test_str_keys_accepted(self):
        assert encode({"spam": 42}) == b"d4:spami42ee"

    def test_int_key_rejected(self):
        with pytest.raises(EncodeError):
            encode({1: b"val"})

    def test_nested(self):
        result = encode({b"info": {b"length": 100}})
        assert result == b"d4:infod6:lengthi100eee"


# ---------------------------------------------------------------------------
# Encode — unsupported types
# ---------------------------------------------------------------------------

class TestEncodeUnsupported:
    def test_none_rejected(self):
        with pytest.raises(EncodeError):
            encode(None)

    def test_float_rejected(self):
        with pytest.raises(EncodeError):
            encode(3.14)

    def test_set_rejected(self):
        with pytest.raises(EncodeError):
            encode({1, 2, 3})


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """encode(decode(x)) == x and decode(encode(x)) == x for valid inputs."""

    CASES = [
        (b"i0e", 0),
        (b"i-99e", -99),
        (b"4:spam", b"spam"),
        (b"0:", b""),
        (b"le", []),
        (b"li1ei2ee", [1, 2]),
        (b"de", {}),
        (b"d3:fooi1ee", {b"foo": 1}),
        (b"d4:infod6:lengthi999eee", {b"info": {b"length": 999}}),
    ]

    @pytest.mark.parametrize("encoded,decoded", CASES)
    def test_decode_then_encode(self, encoded, decoded):
        assert encode(decode(encoded)) == encoded

    @pytest.mark.parametrize("encoded,decoded", CASES)
    def test_encode_then_decode(self, encoded, decoded):
        assert decode(encode(decoded)) == decoded
