"""
Tests for bittorrent.tracker.

Strategy:
  - _parse_compact_peers, _parse_dict_peers, _parse_response, _build_url
    are pure functions tested directly with no mocking.
  - announce() requires aiohttp; we mock ClientSession with lightweight
    async context manager stubs — no extra libraries needed.
"""

import struct
import urllib.parse
import pytest

from bittorrent.bencode import encode
from bittorrent.tracker import (
    TrackerError,
    TrackerResponse,
    _build_url,
    _parse_compact_peers,
    _parse_dict_peers,
    _parse_response,
    announce,
    generate_peer_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INFO_HASH = bytes(range(20))          # fake 20-byte info_hash
PEER_ID   = b"-BC0001-" + b"X" * 12  # fake 20-byte peer_id


def compact_peer(ip: str, port: int) -> bytes:
    """Encode one peer as 6 compact bytes."""
    parts = [int(p) for p in ip.split(".")]
    return bytes(parts) + struct.pack("!H", port)


def make_tracker_response(
    interval: int = 1800,
    peers: bytes | list = b"",
    *,
    min_interval: int = 0,
    complete: int = 0,
    incomplete: int = 0,
) -> bytes:
    """Build a bencoded tracker response."""
    d: dict = {
        b"complete": complete,
        b"incomplete": incomplete,
        b"interval": interval,
        b"peers": peers,
    }
    if min_interval:
        d[b"min interval"] = min_interval
    return encode(d)


# ---------------------------------------------------------------------------
# generate_peer_id
# ---------------------------------------------------------------------------

class TestGeneratePeerId:
    def test_length(self):
        assert len(generate_peer_id()) == 20

    def test_prefix(self):
        pid = generate_peer_id()
        assert pid.startswith(b"-BC0001-")

    def test_unique(self):
        assert generate_peer_id() != generate_peer_id()


# ---------------------------------------------------------------------------
# _parse_compact_peers
# ---------------------------------------------------------------------------

class TestParseCompactPeers:
    def test_empty(self):
        assert _parse_compact_peers(b"") == []

    def test_single_peer(self):
        data = compact_peer("192.168.1.1", 6881)
        peers = _parse_compact_peers(data)
        assert peers == [("192.168.1.1", 6881)]

    def test_multiple_peers(self):
        data = compact_peer("1.2.3.4", 1000) + compact_peer("5.6.7.8", 2000)
        peers = _parse_compact_peers(data)
        assert peers == [("1.2.3.4", 1000), ("5.6.7.8", 2000)]

    def test_boundary_ips(self):
        data = compact_peer("0.0.0.0", 0) + compact_peer("255.255.255.255", 65535)
        peers = _parse_compact_peers(data)
        assert peers == [("0.0.0.0", 0), ("255.255.255.255", 65535)]

    def test_port_byte_order(self):
        # Port 6881 = 0x1AE1; in big-endian: 0x1A, 0xE1
        data = b"\x01\x02\x03\x04\x1a\xe1"
        peers = _parse_compact_peers(data)
        assert peers[0][1] == 6881

    def test_not_multiple_of_6_raises(self):
        with pytest.raises(TrackerError, match="multiple of 6"):
            _parse_compact_peers(b"\x01\x02\x03")

    def test_five_bytes_raises(self):
        with pytest.raises(TrackerError):
            _parse_compact_peers(b"\x00" * 5)

    def test_seven_bytes_raises(self):
        with pytest.raises(TrackerError):
            _parse_compact_peers(b"\x00" * 7)


# ---------------------------------------------------------------------------
# _parse_dict_peers
# ---------------------------------------------------------------------------

class TestParseDictPeers:
    def test_empty(self):
        assert _parse_dict_peers([]) == []

    def test_single(self):
        peers = _parse_dict_peers([
            {b"ip": b"1.2.3.4", b"port": 6881, b"peer id": b"x" * 20}
        ])
        assert peers == [("1.2.3.4", 6881)]

    def test_multiple(self):
        entries = [
            {b"ip": b"10.0.0.1", b"port": 1234},
            {b"ip": b"10.0.0.2", b"port": 5678},
        ]
        assert _parse_dict_peers(entries) == [("10.0.0.1", 1234), ("10.0.0.2", 5678)]

    def test_non_dict_entry_raises(self):
        with pytest.raises(TrackerError, match="dict"):
            _parse_dict_peers([b"not a dict"])

    def test_invalid_ip_type_raises(self):
        with pytest.raises(TrackerError, match="ip"):
            _parse_dict_peers([{b"ip": 12345, b"port": 6881}])

    def test_invalid_port_type_raises(self):
        with pytest.raises(TrackerError, match="port"):
            _parse_dict_peers([{b"ip": b"1.2.3.4", b"port": b"6881"}])


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_compact_peers_parsed(self):
        data = compact_peer("10.0.0.1", 6881) + compact_peer("10.0.0.2", 6882)
        body = make_tracker_response(interval=900, peers=data)
        result = _parse_response(body)
        assert result.interval == 900
        assert ("10.0.0.1", 6881) in result.peers
        assert ("10.0.0.2", 6882) in result.peers

    def test_dict_peers_parsed(self):
        peers_list = [
            {b"ip": b"1.2.3.4", b"port": 9999},
        ]
        body = encode({
            b"complete": 0,
            b"incomplete": 0,
            b"interval": 600,
            b"peers": peers_list,
        })
        result = _parse_response(body)
        assert result.peers == [("1.2.3.4", 9999)]

    def test_interval(self):
        body = make_tracker_response(interval=1800)
        assert _parse_response(body).interval == 1800

    def test_min_interval(self):
        body = make_tracker_response(interval=1800, min_interval=60)
        result = _parse_response(body)
        assert result.min_interval == 60

    def test_complete_incomplete(self):
        body = make_tracker_response(complete=10, incomplete=3)
        result = _parse_response(body)
        assert result.complete == 10
        assert result.incomplete == 3

    def test_empty_peer_list(self):
        body = make_tracker_response(peers=b"")
        result = _parse_response(body)
        assert result.peers == []

    def test_failure_reason_raises(self):
        body = encode({b"failure reason": b"torrent not registered"})
        with pytest.raises(TrackerError, match="torrent not registered"):
            _parse_response(body)

    def test_not_a_dict_raises(self):
        with pytest.raises(TrackerError, match="dict"):
            _parse_response(encode([1, 2, 3]))

    def test_invalid_bencode_raises(self):
        with pytest.raises(TrackerError, match="decode"):
            _parse_response(b"not bencode!!!")

    def test_missing_peers_defaults_to_empty(self):
        body = encode({b"complete": 0, b"incomplete": 0, b"interval": 60})
        result = _parse_response(body)
        assert result.peers == []

    def test_missing_interval_defaults_to_zero(self):
        body = encode({b"complete": 0, b"incomplete": 0, b"peers": b""})
        result = _parse_response(body)
        assert result.interval == 0


# ---------------------------------------------------------------------------
# _build_url
# ---------------------------------------------------------------------------

class TestBuildUrl:
    def test_contains_info_hash(self):
        url = _build_url("http://t.com/announce", INFO_HASH, PEER_ID, 6881)
        assert "info_hash=" in url

    def test_contains_peer_id(self):
        url = _build_url("http://t.com/announce", INFO_HASH, PEER_ID, 6881)
        assert "peer_id=" in url

    def test_compact_flag(self):
        url = _build_url("http://t.com/announce", INFO_HASH, PEER_ID, 6881)
        assert "compact=1" in url

    def test_port(self):
        url = _build_url("http://t.com/announce", INFO_HASH, PEER_ID, 6881)
        assert "port=6881" in url

    def test_uses_ampersand_when_query_exists(self):
        url = _build_url("http://t.com/ann?foo=bar", INFO_HASH, PEER_ID, 6881)
        assert url.startswith("http://t.com/ann?foo=bar&")

    def test_uses_question_mark_when_no_query(self):
        url = _build_url("http://t.com/announce", INFO_HASH, PEER_ID, 6881)
        assert "?" in url
        assert url.index("?") == len("http://t.com/announce")

    def test_event_included_when_provided(self):
        url = _build_url("http://t.com/announce", INFO_HASH, PEER_ID, 6881, event="started")
        assert "event=started" in url

    def test_event_omitted_when_empty(self):
        url = _build_url("http://t.com/announce", INFO_HASH, PEER_ID, 6881, event="")
        assert "event" not in url

    def test_info_hash_percent_encoded(self):
        # All-zeros info_hash should be encoded as %00%00...
        zeros = b"\x00" * 20
        url = _build_url("http://t.com/announce", zeros, PEER_ID, 6881)
        assert "%00" in url

    def test_non_ascii_bytes_encoded(self):
        # Bytes like \xff must appear percent-encoded, not raw
        tricky = b"\xff" * 20
        url = _build_url("http://t.com/announce", tricky, PEER_ID, 6881)
        assert "%FF" in url.upper()

    def test_url_decodeable(self):
        # The produced URL must be valid enough to parse
        url = _build_url("http://t.com/announce", INFO_HASH, PEER_ID, 6881)
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == "http"
        assert parsed.query != ""

    def test_numwant_default(self):
        url = _build_url("http://t.com/announce", INFO_HASH, PEER_ID, 6881)
        assert "numwant=200" in url

    def test_numwant_custom(self):
        url = _build_url("http://t.com/announce", INFO_HASH, PEER_ID, 6881, numwant=100)
        assert "numwant=100" in url


# ---------------------------------------------------------------------------
# announce() — mocked aiohttp
# ---------------------------------------------------------------------------

class _MockResponse:
    """Async context manager that returns a fake HTTP response."""
    def __init__(self, body: bytes, status: int = 200):
        self.status = status
        self._body = body

    async def read(self) -> bytes:
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass


class _MockSession:
    """Async context manager that stands in for aiohttp.ClientSession."""
    def __init__(self, response: _MockResponse):
        self._response = response
        self.last_url: str = ""

    def get(self, url, **kwargs):
        self.last_url = url
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass


@pytest.fixture
def patch_session(monkeypatch):
    """Return a factory that installs a mock session returning *body*."""
    def _install(body: bytes, status: int = 200):
        session = _MockSession(_MockResponse(body, status=status))
        import aiohttp
        monkeypatch.setattr(aiohttp, "ClientSession", lambda: session)
        return session
    return _install


class TestAnnounce:
    async def test_returns_tracker_response(self, patch_session):
        peers = compact_peer("1.2.3.4", 6881)
        patch_session(make_tracker_response(interval=900, peers=peers))
        result = await announce(
            "http://tracker.example.com/announce",
            INFO_HASH, PEER_ID, 6881,
        )
        assert isinstance(result, TrackerResponse)
        assert result.interval == 900
        assert ("1.2.3.4", 6881) in result.peers

    async def test_http_error_raises_tracker_error(self, patch_session):
        patch_session(b"", status=404)
        with pytest.raises(TrackerError, match="HTTP 404"):
            await announce("http://tracker.example.com/announce",
                           INFO_HASH, PEER_ID, 6881)

    async def test_failure_reason_raises(self, patch_session):
        body = encode({b"failure reason": b"not registered"})
        patch_session(body)
        with pytest.raises(TrackerError, match="not registered"):
            await announce("http://tracker.example.com/announce",
                           INFO_HASH, PEER_ID, 6881)

    async def test_url_sent_to_session(self, patch_session):
        peers = compact_peer("1.2.3.4", 6881)
        session = patch_session(make_tracker_response(peers=peers))
        await announce("http://tracker.example.com/announce",
                       INFO_HASH, PEER_ID, 6881, event="started")
        assert "info_hash=" in session.last_url
        assert "event=started" in session.last_url
        assert "compact=1" in session.last_url

    async def test_empty_peer_list(self, patch_session):
        patch_session(make_tracker_response(peers=b""))
        result = await announce("http://tracker.example.com/announce",
                                INFO_HASH, PEER_ID, 6881)
        assert result.peers == []

    async def test_multiple_peers_returned(self, patch_session):
        data = (
            compact_peer("10.0.0.1", 6881)
            + compact_peer("10.0.0.2", 6882)
            + compact_peer("10.0.0.3", 6883)
        )
        patch_session(make_tracker_response(peers=data))
        result = await announce("http://tracker.example.com/announce",
                                INFO_HASH, PEER_ID, 6881)
        assert len(result.peers) == 3

    async def test_network_error_raises_tracker_error(self, monkeypatch):
        import aiohttp

        class _FailSession:
            def get(self, *a, **kw):
                raise aiohttp.ClientConnectionError("connection refused")
            async def __aenter__(self): return self
            async def __aexit__(self, *_): pass

        monkeypatch.setattr(aiohttp, "ClientSession", lambda: _FailSession())
        with pytest.raises(TrackerError, match="HTTP request failed"):
            await announce("http://tracker.example.com/announce",
                           INFO_HASH, PEER_ID, 6881)
