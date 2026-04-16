"""
Tests for bittorrent.lsd — BEP 14 Local Service Discovery.

We test the packet encoding/decoding functions directly (no network required),
and the peer-discovery callback logic by injecting fake datagrams.
"""

import asyncio
import pytest

from bittorrent.lsd import (
    LSD_MCAST_ADDR,
    LSD_PORT,
    LSDService,
    _make_announce,
    _parse_announce,
)


INFO_HASH = bytes(range(20))
INFO_HASH_HEX = INFO_HASH.hex()


# ---------------------------------------------------------------------------
# Packet encoding
# ---------------------------------------------------------------------------

class TestMakeAnnounce:
    def test_starts_with_bt_search(self):
        pkt = _make_announce(INFO_HASH_HEX, 6881, "abc123")
        assert pkt.startswith(b"BT-SEARCH")

    def test_contains_infohash(self):
        pkt = _make_announce(INFO_HASH_HEX, 6881, "abc123")
        assert INFO_HASH_HEX.encode() in pkt

    def test_contains_port(self):
        pkt = _make_announce(INFO_HASH_HEX, 6881, "abc123")
        assert b"Port: 6881" in pkt

    def test_contains_host(self):
        pkt = _make_announce(INFO_HASH_HEX, 6881, "abc123")
        assert LSD_MCAST_ADDR.encode() in pkt

    def test_contains_cookie(self):
        pkt = _make_announce(INFO_HASH_HEX, 6881, "mycookie")
        assert b"mycookie" in pkt

    def test_ends_with_double_crlf(self):
        pkt = _make_announce(INFO_HASH_HEX, 6881, "x")
        assert pkt.endswith(b"\r\n\r\n")

    def test_different_ports(self):
        pkt = _make_announce(INFO_HASH_HEX, 9999, "x")
        assert b"Port: 9999" in pkt


# ---------------------------------------------------------------------------
# Packet parsing
# ---------------------------------------------------------------------------

class TestParseAnnounce:
    def test_roundtrip(self):
        pkt = _make_announce(INFO_HASH_HEX, 6881, "cookie1")
        ih, port, cookie = _parse_announce(pkt)
        assert ih == INFO_HASH_HEX
        assert port == 6881
        assert cookie == "cookie1"

    def test_returns_none_for_non_bt_search(self):
        ih, port, cookie = _parse_announce(b"HTTP/1.1 200 OK\r\n")
        assert ih is None
        assert port is None

    def test_returns_none_for_empty(self):
        ih, port, cookie = _parse_announce(b"")
        assert ih is None

    def test_returns_none_for_garbage(self):
        ih, port, cookie = _parse_announce(b"\xff\xfe garbage data")
        assert ih is None

    def test_infohash_lowercased(self):
        pkt = _make_announce(INFO_HASH_HEX.upper(), 6881, "x")
        ih, _, _ = _parse_announce(pkt)
        assert ih == INFO_HASH_HEX.lower()

    def test_missing_port_returns_none_port(self):
        data = b"BT-SEARCH * HTTP/1.1\r\nInfohash: " + INFO_HASH_HEX.encode() + b"\r\n\r\n"
        ih, port, _ = _parse_announce(data)
        assert ih == INFO_HASH_HEX
        assert port is None

    def test_invalid_port_returns_none_port(self):
        data = (
            b"BT-SEARCH * HTTP/1.1\r\n"
            b"Infohash: " + INFO_HASH_HEX.encode() + b"\r\n"
            b"Port: notanumber\r\n\r\n"
        )
        ih, port, _ = _parse_announce(data)
        assert port is None


# ---------------------------------------------------------------------------
# LSDService logic (no real UDP socket)
# ---------------------------------------------------------------------------

class TestLSDServiceHandleDatagram:
    """Test the datagram handler without opening real sockets."""

    def _make_service(self, info_hash=INFO_HASH, port=6881):
        svc = LSDService(info_hash, port)
        # Don't call start() — we inject datagrams directly
        return svc

    def test_discovers_valid_peer(self):
        svc = self._make_service()
        pkt = _make_announce(INFO_HASH_HEX, 6882, "othercookie")
        svc._handle_datagram(pkt, ("192.168.1.5", 6771))
        assert ("192.168.1.5", 6882) in svc._discovered

    def test_ignores_own_cookie(self):
        svc = self._make_service()
        pkt = _make_announce(INFO_HASH_HEX, 6881, svc._cookie)
        svc._handle_datagram(pkt, ("192.168.1.5", 6771))
        assert svc._discovered == []

    def test_ignores_different_infohash(self):
        svc = self._make_service()
        other_hash = bytes(reversed(range(20))).hex()
        pkt = _make_announce(other_hash, 6882, "othercookie")
        svc._handle_datagram(pkt, ("192.168.1.5", 6771))
        assert svc._discovered == []

    def test_ignores_garbage(self):
        svc = self._make_service()
        svc._handle_datagram(b"\xff\xfe garbage", ("192.168.1.5", 6771))
        assert svc._discovered == []

    def test_deduplicates_peers(self):
        svc = self._make_service()
        pkt = _make_announce(INFO_HASH_HEX, 6882, "othercookie")
        svc._handle_datagram(pkt, ("192.168.1.5", 6771))
        svc._handle_datagram(pkt, ("192.168.1.5", 6771))
        assert svc._discovered.count(("192.168.1.5", 6882)) == 1

    def test_callback_called_on_discovery(self):
        calls = []
        svc = LSDService(INFO_HASH, 6881, on_peer=lambda h, p: calls.append((h, p)))
        pkt = _make_announce(INFO_HASH_HEX, 6882, "othercookie")
        svc._handle_datagram(pkt, ("10.0.0.1", 6771))
        assert calls == [("10.0.0.1", 6882)]

    def test_callback_not_called_for_own_packet(self):
        calls = []
        svc = LSDService(INFO_HASH, 6881, on_peer=lambda h, p: calls.append((h, p)))
        pkt = _make_announce(INFO_HASH_HEX, 6881, svc._cookie)
        svc._handle_datagram(pkt, ("10.0.0.1", 6771))
        assert calls == []

    def test_multiple_peers_from_different_sources(self):
        svc = self._make_service()
        pkt1 = _make_announce(INFO_HASH_HEX, 6882, "c1")
        pkt2 = _make_announce(INFO_HASH_HEX, 6883, "c2")
        svc._handle_datagram(pkt1, ("10.0.0.1", 6771))
        svc._handle_datagram(pkt2, ("10.0.0.2", 6771))
        assert ("10.0.0.1", 6882) in svc._discovered
        assert ("10.0.0.2", 6883) in svc._discovered


# ---------------------------------------------------------------------------
# announce_once (mocked transport)
# ---------------------------------------------------------------------------

class TestAnnounceOnce:
    async def test_sends_to_mcast_address(self):
        """announce_once sends to the LSD multicast address and port."""
        sent = []

        class FakeTransport:
            def sendto(self, data, addr):
                sent.append((data, addr))
            def close(self):
                pass

        svc = LSDService(INFO_HASH, 6881)
        svc._transport = FakeTransport()
        await svc.announce_once()

        assert len(sent) == 1
        data, addr = sent[0]
        assert addr == (LSD_MCAST_ADDR, LSD_PORT)
        assert INFO_HASH_HEX.encode() in data

    async def test_no_send_without_transport(self):
        """announce_once is a no-op if transport is not initialised."""
        svc = LSDService(INFO_HASH, 6881)
        # No exception — just silently skips
        await svc.announce_once()
