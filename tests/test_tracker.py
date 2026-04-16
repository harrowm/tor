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
from unittest.mock import AsyncMock, patch

from bittorrent.bencode import encode
from bittorrent.tracker import (
    TrackerError,
    TrackerResponse,
    _UDP_ANNOUNCE,
    _UDP_CONNECT,
    _UDP_ERROR,
    _UDP_MAGIC,
    _announce_udp,
    _build_url,
    _decode_announce_response,
    _decode_connect_response,
    _encode_announce_request,
    _encode_connect_request,
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

    async def test_udp_url_routes_to_udp(self, monkeypatch):
        """announce() dispatches UDP URLs to _announce_udp."""
        expected = TrackerResponse(interval=900, peers=[("1.2.3.4", 6881)])
        mock_udp = AsyncMock(return_value=expected)
        monkeypatch.setattr("bittorrent.tracker._announce_udp", mock_udp)
        result = await announce("udp://tracker.example.com:1337/announce",
                                INFO_HASH, PEER_ID, 6881)
        assert result is expected
        mock_udp.assert_called_once()

    async def test_http_url_does_not_route_to_udp(self, patch_session):
        """announce() does NOT call _announce_udp for http:// URLs."""
        peers = compact_peer("1.2.3.4", 6881)
        patch_session(make_tracker_response(peers=peers))
        # If _announce_udp were called it would fail (no mock), so this just
        # verifies the HTTP path executes cleanly.
        result = await announce("http://tracker.example.com/announce",
                                INFO_HASH, PEER_ID, 6881)
        assert isinstance(result, TrackerResponse)

    async def test_https_url_routes_to_http_handler(self, patch_session):
        """https:// URLs use the HTTP handler (aiohttp supports TLS natively)."""
        peers = compact_peer("5.6.7.8", 6881)
        session = patch_session(make_tracker_response(peers=peers))
        result = await announce("https://tracker.example.com/announce",
                                INFO_HASH, PEER_ID, 6881)
        assert isinstance(result, TrackerResponse)
        assert ("5.6.7.8", 6881) in result.peers
        # Verify the https URL was passed through unchanged
        assert session.last_url.startswith("https://")

    async def test_https_passes_event_parameter(self, patch_session):
        """event= is included in HTTPS announce URLs just like HTTP."""
        peers = compact_peer("1.2.3.4", 6881)
        session = patch_session(make_tracker_response(peers=peers))
        await announce("https://tracker.example.com/announce",
                       INFO_HASH, PEER_ID, 6881, event="started")
        assert "event=started" in session.last_url


# ---------------------------------------------------------------------------
# UDP packet encoding / decoding — pure functions
# ---------------------------------------------------------------------------

TX_ID       = 0xDEADBEEF
CONN_ID     = 0x123456789ABCDEF0


class TestEncodeConnectRequest:
    def test_length(self):
        pkt = _encode_connect_request(TX_ID)
        assert len(pkt) == 16

    def test_magic(self):
        pkt = _encode_connect_request(TX_ID)
        magic, = struct.unpack_from("!Q", pkt, 0)
        assert magic == _UDP_MAGIC

    def test_action_zero(self):
        pkt = _encode_connect_request(TX_ID)
        action, = struct.unpack_from("!I", pkt, 8)
        assert action == _UDP_CONNECT

    def test_transaction_id(self):
        pkt = _encode_connect_request(TX_ID)
        txid, = struct.unpack_from("!I", pkt, 12)
        assert txid == TX_ID


class TestDecodeConnectResponse:
    def _make_response(self, action=_UDP_CONNECT, txid=TX_ID, conn_id=CONN_ID):
        return struct.pack("!IIQ", action, txid, conn_id)

    def test_returns_connection_id(self):
        data = self._make_response()
        assert _decode_connect_response(data, TX_ID) == CONN_ID

    def test_short_data_raises(self):
        with pytest.raises(TrackerError, match="too short"):
            _decode_connect_response(b"\x00" * 15, TX_ID)

    def test_wrong_action_raises(self):
        data = self._make_response(action=99)
        with pytest.raises(TrackerError, match="action"):
            _decode_connect_response(data, TX_ID)

    def test_txid_mismatch_raises(self):
        data = self._make_response(txid=0xCAFEBABE)
        with pytest.raises(TrackerError, match="transaction ID"):
            _decode_connect_response(data, TX_ID)

    def test_extra_bytes_ignored(self):
        data = self._make_response() + b"\xff" * 10
        assert _decode_connect_response(data, TX_ID) == CONN_ID


class TestEncodeAnnounceRequest:
    def _make(self, **kwargs):
        return _encode_announce_request(
            CONN_ID, TX_ID, INFO_HASH, PEER_ID, **kwargs
        )

    def test_length(self):
        assert len(self._make()) == 98

    def test_connection_id(self):
        pkt = self._make()
        conn_id, = struct.unpack_from("!Q", pkt, 0)
        assert conn_id == CONN_ID

    def test_action_one(self):
        pkt = self._make()
        action, = struct.unpack_from("!I", pkt, 8)
        assert action == _UDP_ANNOUNCE

    def test_transaction_id(self):
        pkt = self._make()
        txid, = struct.unpack_from("!I", pkt, 12)
        assert txid == TX_ID

    def test_info_hash_at_offset_16(self):
        pkt = self._make()
        assert pkt[16:36] == INFO_HASH

    def test_peer_id_at_offset_36(self):
        pkt = self._make()
        assert pkt[36:56] == PEER_ID

    def test_event_started_is_2(self):
        pkt = self._make(event="started")
        event_id, = struct.unpack_from("!I", pkt, 80)
        assert event_id == 2

    def test_event_stopped_is_3(self):
        pkt = self._make(event="stopped")
        event_id, = struct.unpack_from("!I", pkt, 80)
        assert event_id == 3

    def test_event_completed_is_1(self):
        pkt = self._make(event="completed")
        event_id, = struct.unpack_from("!I", pkt, 80)
        assert event_id == 1

    def test_event_empty_is_0(self):
        pkt = self._make(event="")
        event_id, = struct.unpack_from("!I", pkt, 80)
        assert event_id == 0

    def test_port_at_end(self):
        pkt = self._make(port=51413)
        port, = struct.unpack_from("!H", pkt, 96)
        assert port == 51413


class TestDecodeAnnounceResponse:
    def _make_response(
        self, action=_UDP_ANNOUNCE, txid=TX_ID, interval=1800,
        leechers=5, seeders=10, peers=b""
    ):
        header = struct.pack("!IIIII", action, txid, interval, leechers, seeders)
        return header + peers

    def test_returns_tracker_response(self):
        data = self._make_response()
        result = _decode_announce_response(data, TX_ID)
        assert isinstance(result, TrackerResponse)

    def test_interval(self):
        data = self._make_response(interval=900)
        assert _decode_announce_response(data, TX_ID).interval == 900

    def test_seeders_and_leechers(self):
        data = self._make_response(leechers=3, seeders=7)
        result = _decode_announce_response(data, TX_ID)
        assert result.complete == 7
        assert result.incomplete == 3

    def test_peers_parsed(self):
        peers_bytes = compact_peer("10.0.0.1", 6881)
        data = self._make_response(peers=peers_bytes)
        result = _decode_announce_response(data, TX_ID)
        assert ("10.0.0.1", 6881) in result.peers

    def test_multiple_peers(self):
        peers_bytes = compact_peer("1.1.1.1", 1000) + compact_peer("2.2.2.2", 2000)
        data = self._make_response(peers=peers_bytes)
        result = _decode_announce_response(data, TX_ID)
        assert len(result.peers) == 2

    def test_short_data_raises(self):
        with pytest.raises(TrackerError, match="too short"):
            _decode_announce_response(b"\x00" * 19, TX_ID)

    def test_wrong_action_raises(self):
        data = self._make_response(action=99)
        with pytest.raises(TrackerError, match="action"):
            _decode_announce_response(data, TX_ID)

    def test_error_action_raises(self):
        # action=3 is UDP error response
        header = struct.pack("!II", _UDP_ERROR, TX_ID)
        data = header + b"torrent banned"
        with pytest.raises(TrackerError, match="UDP tracker error"):
            _decode_announce_response(data, TX_ID)

    def test_txid_mismatch_raises(self):
        data = self._make_response(txid=0x11111111)
        with pytest.raises(TrackerError, match="transaction ID"):
            _decode_announce_response(data, TX_ID)


# ---------------------------------------------------------------------------
# _announce_udp — integration (mocked _udp_transact)
# ---------------------------------------------------------------------------

class TestAnnounceUdp:
    """Test the full connect+announce flow with _udp_transact mocked out."""

    def _connect_resp(self, txid=None, conn_id=CONN_ID):
        """Build a valid connect response (txid filled in at call time)."""
        return lambda t: struct.pack("!IIQ", _UDP_CONNECT, t, conn_id)

    def _announce_resp(self, txid=None, interval=1800, leechers=2, seeders=5, peers=b""):
        header = struct.pack("!IIIII", _UDP_ANNOUNCE, txid or TX_ID,
                             interval, leechers, seeders)
        return header + peers

    async def test_full_flow_returns_tracker_response(self, monkeypatch):
        call_count = 0
        stored_tx = {}

        async def fake_transact(host, port, request, *, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Connect phase: extract transaction_id from request
                txid, = struct.unpack_from("!I", request, 12)
                stored_tx["connect"] = txid
                return struct.pack("!IIQ", _UDP_CONNECT, txid, CONN_ID)
            else:
                # Announce phase: extract transaction_id
                txid, = struct.unpack_from("!I", request, 12)
                return struct.pack(
                    "!IIIII", _UDP_ANNOUNCE, txid, 1800, 2, 5
                ) + compact_peer("1.2.3.4", 6881)

        monkeypatch.setattr("bittorrent.tracker._udp_transact", fake_transact)
        result = await _announce_udp(
            "udp://tracker.example.com:1337/announce",
            INFO_HASH, PEER_ID, 6881,
        )
        assert isinstance(result, TrackerResponse)
        assert result.interval == 1800
        assert ("1.2.3.4", 6881) in result.peers
        assert call_count == 2

    async def test_invalid_url_raises(self):
        with pytest.raises(TrackerError, match="Invalid UDP tracker URL"):
            await _announce_udp("udp:///no-host", INFO_HASH, PEER_ID, 6881)

    async def test_url_missing_port_raises(self):
        with pytest.raises(TrackerError, match="Invalid UDP tracker URL"):
            await _announce_udp("udp://tracker.example.com/announce",
                                INFO_HASH, PEER_ID, 6881)

    async def test_transact_error_propagates(self, monkeypatch):
        async def fail_transact(*a, **kw):
            raise TrackerError("DNS failure")

        monkeypatch.setattr("bittorrent.tracker._udp_transact", fail_transact)
        with pytest.raises(TrackerError, match="DNS failure"):
            await _announce_udp(
                "udp://tracker.example.com:1337/announce",
                INFO_HASH, PEER_ID, 6881,
            )

    async def test_correct_host_and_port_used(self, monkeypatch):
        calls = []

        async def capturing_transact(host, port, request, *, timeout):
            calls.append((host, port))
            txid, = struct.unpack_from("!I", request, 12)
            if len(calls) == 1:
                return struct.pack("!IIQ", _UDP_CONNECT, txid, CONN_ID)
            return struct.pack("!IIIII", _UDP_ANNOUNCE, txid, 900, 0, 0)

        monkeypatch.setattr("bittorrent.tracker._udp_transact", capturing_transact)
        await _announce_udp(
            "udp://opentracker.example.org:6969/announce",
            INFO_HASH, PEER_ID, 6881,
        )
        assert all(h == "opentracker.example.org" for h, _ in calls)
        assert all(p == 6969 for _, p in calls)


# ---------------------------------------------------------------------------
# IPv6 compact peers (BEP 7)
# ---------------------------------------------------------------------------

class TestParseCompactPeers6:
    from bittorrent.tracker import _parse_compact_peers6

    def test_single_ipv6_peer(self):
        import socket
        from bittorrent.tracker import _parse_compact_peers6
        # ::1 (loopback) port 6881
        addr_bytes = socket.inet_pton(socket.AF_INET6, "::1")
        port_bytes = struct.pack("!H", 6881)
        peers = _parse_compact_peers6(addr_bytes + port_bytes)
        assert len(peers) == 1
        assert peers[0] == ("::1", 6881)

    def test_multiple_ipv6_peers(self):
        import socket
        from bittorrent.tracker import _parse_compact_peers6
        addrs = ["::1", "::2", "2001:db8::1"]
        ports = [6881, 6882, 6883]
        data = b""
        for addr, port in zip(addrs, ports):
            data += socket.inet_pton(socket.AF_INET6, addr) + struct.pack("!H", port)
        peers = _parse_compact_peers6(data)
        assert len(peers) == 3
        for i, (addr, port) in enumerate(zip(addrs, ports)):
            # inet_ntop may normalise address representation
            assert peers[i][1] == ports[i]

    def test_empty_data_returns_empty(self):
        from bittorrent.tracker import _parse_compact_peers6
        assert _parse_compact_peers6(b"") == []

    def test_non_multiple_of_18_raises(self):
        from bittorrent.tracker import _parse_compact_peers6, TrackerError
        with pytest.raises(TrackerError):
            _parse_compact_peers6(b"\x00" * 17)

    def test_peers6_in_tracker_response(self):
        """HTTP tracker response containing peers6 key adds IPv6 peers."""
        import socket
        from bittorrent.bencode import encode as bencode
        from bittorrent.tracker import _parse_response

        # Build a compact IPv4 peers blob (one peer)
        ipv4_peer = struct.pack("!BBBBH", 1, 2, 3, 4, 6881)
        # Build a compact IPv6 peers6 blob (one peer)
        ipv6_addr = socket.inet_pton(socket.AF_INET6, "::1")
        ipv6_peer = ipv6_addr + struct.pack("!H", 6882)

        body = bencode({
            b"interval": 1800,
            b"peers":    ipv4_peer,
            b"peers6":   ipv6_peer,
        })
        resp = _parse_response(body)
        ips = [ip for ip, _ in resp.peers]
        assert "1.2.3.4" in ips
        assert "::1" in ips
        assert len(resp.peers) == 2
