"""
Tests for bittorrent.dht — BEP 5 DHT implementation.

Tests are split into:
  - Pure functions: distance, compact encode/decode, message encode/decode
  - Data structures: KBucket, RoutingTable
  - Transport (mocked UDP): request/response correlation, timeout
  - DHTClient (mocked transport): bootstrap, get_peers iterative lookup
"""

from __future__ import annotations

import asyncio
import struct
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from bittorrent.bencode import encode
from bittorrent.dht import (
    ALPHA,
    K,
    DHTClient,
    DHTNode,
    DHTTransport,
    KBucket,
    RoutingTable,
    decode_compact_nodes,
    decode_compact_nodes6,
    decode_compact_peers,
    decode_compact_peers6,
    decode_response,
    encode_compact_nodes,
    encode_find_node,
    encode_get_peers,
    encode_ping,
    is_error,
    is_response,
    xor_distance,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_id(byte_val: int) -> bytes:
    """20-byte node ID filled with *byte_val*."""
    return bytes([byte_val]) * 20


def make_node(byte_val: int, host: str = "1.2.3.4", port: int = 6881) -> DHTNode:
    return DHTNode(id=make_id(byte_val), host=host, port=port)


def compact_node(node_id: bytes, ip: str, port: int) -> bytes:
    parts = [int(p) for p in ip.split(".")]
    return node_id + bytes(parts) + struct.pack("!H", port)


def compact_peer(ip: str, port: int) -> bytes:
    parts = [int(p) for p in ip.split(".")]
    return bytes(parts) + struct.pack("!H", port)


# ---------------------------------------------------------------------------
# xor_distance
# ---------------------------------------------------------------------------

class TestXorDistance:
    def test_identical_ids_have_zero_distance(self):
        a = make_id(0xAB)
        assert xor_distance(a, a) == 0

    def test_distance_is_symmetric(self):
        a = make_id(0x01)
        b = make_id(0x02)
        assert xor_distance(a, b) == xor_distance(b, a)

    def test_known_distance(self):
        a = b"\x00" * 20
        b = b"\x00" * 19 + b"\x01"
        # XOR of last byte only: distance == 1 (big-endian)
        assert xor_distance(a, b) == 1

    def test_max_distance(self):
        a = b"\x00" * 20
        b = b"\xff" * 20
        assert xor_distance(a, b) == (1 << 160) - 1

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            xor_distance(b"\x00" * 19, b"\x00" * 20)

    def test_triangle_inequality(self):
        # XOR satisfies ultra-metric: d(a,c) <= max(d(a,b), d(b,c))
        a = make_id(0x10)
        b = make_id(0x20)
        c = make_id(0x30)
        assert xor_distance(a, c) <= max(xor_distance(a, b), xor_distance(b, c))


# ---------------------------------------------------------------------------
# decode_compact_nodes / encode_compact_nodes
# ---------------------------------------------------------------------------

class TestCompactNodes:
    def test_decode_single(self):
        node_id = bytes(range(20))
        data = compact_node(node_id, "1.2.3.4", 6881)
        nodes = decode_compact_nodes(data)
        assert len(nodes) == 1
        assert nodes[0].id == node_id
        assert nodes[0].host == "1.2.3.4"
        assert nodes[0].port == 6881

    def test_decode_multiple(self):
        node1 = compact_node(make_id(0x01), "10.0.0.1", 6881)
        node2 = compact_node(make_id(0x02), "10.0.0.2", 1234)
        nodes = decode_compact_nodes(node1 + node2)
        assert len(nodes) == 2
        assert nodes[0].host == "10.0.0.1"
        assert nodes[1].host == "10.0.0.2"
        assert nodes[1].port == 1234

    def test_decode_empty(self):
        assert decode_compact_nodes(b"") == []

    def test_decode_wrong_length_raises(self):
        with pytest.raises(ValueError, match="multiple of 26"):
            decode_compact_nodes(b"\x00" * 25)

    def test_encode_roundtrip(self):
        nodes = [
            DHTNode(id=make_id(0xAA), host="192.168.1.1", port=6881),
            DHTNode(id=make_id(0xBB), host="10.0.0.5",   port=9999),
        ]
        encoded = encode_compact_nodes(nodes)
        assert len(encoded) == 52
        decoded = decode_compact_nodes(encoded)
        assert decoded[0].id   == nodes[0].id
        assert decoded[0].host == nodes[0].host
        assert decoded[0].port == nodes[0].port
        assert decoded[1].id   == nodes[1].id
        assert decoded[1].host == nodes[1].host
        assert decoded[1].port == nodes[1].port

    def test_encode_invalid_ip_raises(self):
        node = DHTNode(id=make_id(0x01), host="not.an.ip", port=6881)
        with pytest.raises((ValueError, Exception)):
            encode_compact_nodes([node])


# ---------------------------------------------------------------------------
# decode_compact_peers
# ---------------------------------------------------------------------------

class TestCompactPeers:
    def test_single_peer(self):
        data = compact_peer("1.2.3.4", 6881)
        peers = decode_compact_peers(data)
        assert peers == [("1.2.3.4", 6881)]

    def test_multiple_peers(self):
        data = compact_peer("10.0.0.1", 100) + compact_peer("10.0.0.2", 200)
        peers = decode_compact_peers(data)
        assert peers == [("10.0.0.1", 100), ("10.0.0.2", 200)]

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="multiple of 6"):
            decode_compact_peers(b"\x00" * 5)

    def test_empty(self):
        assert decode_compact_peers(b"") == []


# ---------------------------------------------------------------------------
# Message encoding
# ---------------------------------------------------------------------------

class TestEncodeMessages:
    def test_ping_structure(self):
        msg = encode_ping(b"aa", make_id(0x01))
        from bittorrent.bencode import decode
        d = decode(msg)
        assert d[b"y"] == b"q"
        assert d[b"q"] == b"ping"
        assert d[b"t"] == b"aa"
        assert d[b"a"][b"id"] == make_id(0x01)

    def test_find_node_structure(self):
        msg = encode_find_node(b"bb", make_id(0x01), make_id(0x02))
        from bittorrent.bencode import decode
        d = decode(msg)
        assert d[b"y"] == b"q"
        assert d[b"q"] == b"find_node"
        assert d[b"a"][b"target"] == make_id(0x02)

    def test_get_peers_structure(self):
        info_hash = bytes(range(20))
        msg = encode_get_peers(b"cc", make_id(0x01), info_hash)
        from bittorrent.bencode import decode
        d = decode(msg)
        assert d[b"y"] == b"q"
        assert d[b"q"] == b"get_peers"
        assert d[b"a"][b"info_hash"] == info_hash


class TestDecodeResponse:
    def test_valid_response(self):
        data = encode({b"t": b"aa", b"y": b"r", b"r": {b"id": make_id(0x01)}})
        msg = decode_response(data)
        assert msg[b"y"] == b"r"

    def test_error_response(self):
        data = encode({b"t": b"aa", b"y": b"e", b"e": [201, b"Generic Error"]})
        msg = decode_response(data)
        assert is_error(msg)

    def test_not_a_dict_raises(self):
        from bittorrent.bencode import encode as bencode_encode
        data = bencode_encode([1, 2, 3])
        with pytest.raises(ValueError):
            decode_response(data)

    def test_invalid_bencode_raises(self):
        with pytest.raises(ValueError):
            decode_response(b"not bencode")

    def test_is_response(self):
        msg = {b"y": b"r"}
        assert is_response(msg)
        assert not is_error(msg)

    def test_is_error(self):
        msg = {b"y": b"e"}
        assert is_error(msg)
        assert not is_response(msg)


# ---------------------------------------------------------------------------
# KBucket
# ---------------------------------------------------------------------------

class TestKBucket:
    def test_add_up_to_k_nodes(self):
        bucket = KBucket(k=3)
        for i in range(3):
            bucket.add(make_node(i))
        assert len(bucket) == 3

    def test_overflow_goes_to_pending(self):
        bucket = KBucket(k=2)
        bucket.add(make_node(0))
        bucket.add(make_node(1))
        bucket.add(make_node(2))  # overflow
        assert len(bucket) == 2
        assert len(bucket.pending) == 1

    def test_update_existing_node(self):
        bucket = KBucket(k=3)
        n = make_node(0)
        bucket.add(n)
        n2 = DHTNode(id=make_id(0), host="9.9.9.9", port=9999)
        bucket.add(n2)
        assert len(bucket) == 1
        assert bucket.nodes[0].host == "9.9.9.9"

    def test_remove_promotes_pending(self):
        bucket = KBucket(k=2)
        bucket.add(make_node(0))
        bucket.add(make_node(1))
        bucket.add(make_node(2))  # goes to pending
        bucket.remove(make_id(0))
        assert len(bucket) == 2
        assert any(n.id == make_id(2) for n in bucket.nodes)

    def test_remove_nonexistent_is_noop(self):
        bucket = KBucket(k=3)
        bucket.add(make_node(0))
        bucket.remove(make_id(99))
        assert len(bucket) == 1

    def test_get_nodes_returns_copy(self):
        bucket = KBucket(k=3)
        bucket.add(make_node(0))
        nodes = bucket.get_nodes()
        nodes.clear()
        assert len(bucket) == 1

    def test_duplicate_in_pending_updated(self):
        bucket = KBucket(k=1)
        bucket.add(make_node(0))
        bucket.add(make_node(1))  # pending
        n2 = DHTNode(id=make_id(1), host="5.5.5.5", port=5555)
        bucket.add(n2)
        assert len(bucket.pending) == 1
        assert bucket.pending[0].host == "5.5.5.5"


# ---------------------------------------------------------------------------
# RoutingTable
# ---------------------------------------------------------------------------

class TestRoutingTable:
    def test_own_id_not_added(self):
        own = make_id(0x01)
        rt = RoutingTable(own)
        rt.add(DHTNode(id=own, host="1.1.1.1", port=1))
        assert rt.size() == 0

    def test_add_and_find_closest(self):
        own = b"\x00" * 20
        rt = RoutingTable(own)
        for i in range(1, 6):
            rt.add(make_node(i, host=f"10.0.0.{i}"))
        closest = rt.find_closest(make_id(0x01), n=3)
        assert len(closest) == 3

    def test_find_closest_returns_sorted_by_distance(self):
        own = b"\x00" * 20
        rt = RoutingTable(own)
        # id 0x01 is closer to target 0x01 than id 0x0F
        rt.add(make_node(0x01, host="close.host"))
        rt.add(make_node(0x0F, host="far.host"))
        closest = rt.find_closest(make_id(0x01), n=2)
        assert closest[0].host == "close.host"

    def test_find_closest_empty_table(self):
        rt = RoutingTable(make_id(0x00))
        assert rt.find_closest(make_id(0x01)) == []

    def test_find_closest_n_larger_than_table(self):
        rt = RoutingTable(make_id(0x00))
        rt.add(make_node(0x01))
        result = rt.find_closest(make_id(0x01), n=100)
        assert len(result) == 1

    def test_remove(self):
        own = b"\x00" * 20
        rt = RoutingTable(own)
        rt.add(make_node(0x01))
        assert rt.size() == 1
        rt.remove(make_id(0x01))
        assert rt.size() == 0

    def test_size(self):
        rt = RoutingTable(b"\x00" * 20)
        for i in range(1, 11):
            rt.add(make_node(i))
        assert rt.size() == 10


# ---------------------------------------------------------------------------
# DHTTransport (mocked asyncio UDP)
# ---------------------------------------------------------------------------

class TestDHTTransport:
    def test_datagram_received_resolves_future(self):
        transport = DHTTransport()
        loop = asyncio.new_event_loop()
        fut = loop.create_future()
        transport._pending[b"\x00\x01"] = fut

        response_data = encode({b"t": b"\x00\x01", b"y": b"r", b"r": {b"id": make_id(0)}})
        transport.datagram_received(response_data, ("1.2.3.4", 6881))

        assert fut.done()
        msg, addr = fut.result()
        assert msg[b"y"] == b"r"
        loop.close()

    def test_unknown_txid_not_resolved(self):
        transport = DHTTransport()
        loop = asyncio.new_event_loop()
        fut = loop.create_future()
        transport._pending[b"\x00\x01"] = fut

        # Different txid
        response_data = encode({b"t": b"\x00\x02", b"y": b"r", b"r": {b"id": make_id(0)}})
        transport.datagram_received(response_data, ("1.2.3.4", 6881))

        assert not fut.done()
        loop.close()

    def test_malformed_data_ignored(self):
        transport = DHTTransport()
        # Should not raise
        transport.datagram_received(b"not bencode!!!", ("1.2.3.4", 6881))

    def test_connection_lost_fails_futures(self):
        transport = DHTTransport()
        loop = asyncio.new_event_loop()
        fut = loop.create_future()
        transport._pending[b"\x00\x01"] = fut
        transport.connection_lost(None)
        assert fut.done()
        assert isinstance(fut.exception(), ConnectionError)
        loop.close()

    def test_send_calls_transport(self):
        transport = DHTTransport()
        mock_udp = MagicMock()
        transport._transport = mock_udp
        transport.send(b"hello", ("1.2.3.4", 6881))
        mock_udp.sendto.assert_called_once_with(b"hello", ("1.2.3.4", 6881))

    def test_send_no_transport_is_noop(self):
        transport = DHTTransport()
        transport._transport = None
        transport.send(b"hello", ("1.2.3.4", 6881))  # should not raise

    @pytest.mark.asyncio
    async def test_request_timeout(self):
        transport = DHTTransport()
        mock_udp = MagicMock()
        transport._transport = mock_udp

        with pytest.raises(asyncio.TimeoutError):
            await transport.request(b"data", ("1.2.3.4", 6881), b"\x00\x01", timeout=0.01)

    @pytest.mark.asyncio
    async def test_request_success(self):
        transport = DHTTransport()
        mock_udp = MagicMock()
        transport._transport = mock_udp

        async def _deliver():
            await asyncio.sleep(0.01)
            response = encode({b"t": b"\x00\x01", b"y": b"r", b"r": {b"id": make_id(0)}})
            transport.datagram_received(response, ("1.2.3.4", 6881))

        asyncio.create_task(_deliver())
        msg, addr = await transport.request(b"data", ("1.2.3.4", 6881), b"\x00\x01")
        assert msg[b"y"] == b"r"


# ---------------------------------------------------------------------------
# DHTClient (mocked transport)
# ---------------------------------------------------------------------------

def _ping_response(txid: bytes, node_id: bytes) -> bytes:
    return encode({b"t": txid, b"y": b"r", b"r": {b"id": node_id}})


def _find_node_response(txid: bytes, node_id: bytes, nodes: list[DHTNode]) -> bytes:
    return encode({
        b"t": txid,
        b"y": b"r",
        b"r": {b"id": node_id, b"nodes": encode_compact_nodes(nodes)},
    })


def _get_peers_with_values(txid: bytes, node_id: bytes, peers: list[tuple[str,int]]) -> bytes:
    values = [compact_peer(ip, port) for ip, port in peers]
    return encode({
        b"t": txid,
        b"y": b"r",
        b"r": {b"id": node_id, b"token": b"token123", b"values": values},
    })


def _get_peers_with_nodes(txid: bytes, node_id: bytes, nodes: list[DHTNode]) -> bytes:
    return encode({
        b"t": txid,
        b"y": b"r",
        b"r": {b"id": node_id, b"token": b"token123", b"nodes": encode_compact_nodes(nodes)},
    })


class TestDHTClientBootstrap:
    @pytest.mark.asyncio
    async def test_bootstrap_queries_provided_nodes(self):
        """bootstrap() should ping each provided node."""
        client = DHTClient(node_id=make_id(0x00))
        mock_transport = MagicMock(spec=DHTTransport)

        # Respond to ping with a distinct node ID per address
        async def fake_request(data, addr, txid, timeout=5.0):
            remote_id = make_id(0xAA) if addr[0] == "10.0.0.1" else make_id(0xBB)
            return (decode_response(_ping_response(txid, remote_id)), addr)

        mock_transport.request = fake_request
        client._transport = mock_transport

        # Override find_node_iterative to avoid further queries
        client._find_node_iterative = AsyncMock(return_value=[])

        nodes = [("10.0.0.1", 6881), ("10.0.0.2", 6881)]
        count = await client.bootstrap(nodes=nodes)
        assert count >= 2  # at least the two bootstrap nodes added

    @pytest.mark.asyncio
    async def test_bootstrap_tolerates_timeout(self):
        """Timeout on one bootstrap node should not prevent others."""
        client = DHTClient(node_id=make_id(0x00))
        mock_transport = MagicMock(spec=DHTTransport)
        call_count = 0

        async def fake_request(data, addr, txid, timeout=5.0):
            nonlocal call_count
            call_count += 1
            if addr[0] == "10.0.0.1":
                raise asyncio.TimeoutError()
            return (decode_response(_ping_response(txid, make_id(0xAA))), addr)

        mock_transport.request = fake_request
        client._transport = mock_transport
        client._find_node_iterative = AsyncMock(return_value=[])

        await client.bootstrap(nodes=[("10.0.0.1", 6881), ("10.0.0.2", 6881)])
        assert call_count == 2
        assert client._table.size() == 1  # only second node added


class TestDHTClientGetPeers:
    @pytest.mark.asyncio
    async def test_get_peers_returns_peers_from_response(self):
        """get_peers() should collect peers from nodes that return values."""
        own_id = make_id(0x00)
        info_hash = make_id(0xFF)
        client = DHTClient(node_id=own_id)

        # Pre-populate routing table
        seed_node = DHTNode(id=make_id(0xAA), host="10.0.0.1", port=6881)
        client._table.add(seed_node)

        mock_transport = MagicMock(spec=DHTTransport)

        async def fake_request(data, addr, txid, timeout=5.0):
            resp = _get_peers_with_values(
                txid, make_id(0xAA),
                [("1.2.3.4", 6881), ("5.6.7.8", 9999)],
            )
            return (decode_response(resp), addr)

        mock_transport.request = fake_request
        client._transport = mock_transport

        peers = await client.get_peers(info_hash)
        assert ("1.2.3.4", 6881) in peers
        assert ("5.6.7.8", 9999) in peers

    @pytest.mark.asyncio
    async def test_get_peers_follows_closer_nodes(self):
        """When nodes respond with closer nodes instead of peers, follow them."""
        own_id   = make_id(0x00)
        info_hash = make_id(0xFF)
        client = DHTClient(node_id=own_id)

        seed_node   = DHTNode(id=make_id(0xAA), host="10.0.0.1", port=6881)
        closer_node = DHTNode(id=make_id(0xFE), host="10.0.0.2", port=6882)
        client._table.add(seed_node)

        mock_transport = MagicMock(spec=DHTTransport)
        calls: list[tuple] = []

        async def fake_request(data, addr, txid, timeout=5.0):
            calls.append(addr)
            if addr == ("10.0.0.1", 6881):
                # Return closer nodes
                resp = _get_peers_with_nodes(txid, make_id(0xAA), [closer_node])
            else:
                # The closer node returns peers
                resp = _get_peers_with_values(txid, make_id(0xFE), [("9.8.7.6", 4321)])
            return (decode_response(resp), addr)

        mock_transport.request = fake_request
        client._transport = mock_transport

        peers = await client.get_peers(info_hash)
        assert ("10.0.0.2", 6882) in [a for a in calls]
        assert ("9.8.7.6", 4321) in peers

    @pytest.mark.asyncio
    async def test_get_peers_empty_routing_table(self):
        """When routing table is empty, get_peers returns [] immediately."""
        client = DHTClient(node_id=make_id(0x00))
        mock_transport = MagicMock(spec=DHTTransport)
        client._transport = mock_transport

        peers = await client.get_peers(make_id(0xFF))
        assert peers == []
        mock_transport.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_peers_timeout_returns_empty(self):
        """If no peers found within timeout, return []."""
        client = DHTClient(node_id=make_id(0x00))
        client._table.add(DHTNode(id=make_id(0xAA), host="10.0.0.1", port=6881))

        mock_transport = MagicMock(spec=DHTTransport)

        async def slow_request(data, addr, txid, timeout=5.0):
            await asyncio.sleep(10)

        mock_transport.request = slow_request
        client._transport = mock_transport

        peers = await client.get_peers(make_id(0xFF), timeout=0.05)
        assert peers == []

    @pytest.mark.asyncio
    async def test_get_peers_deduplicates(self):
        """Same peer returned by multiple nodes should appear once."""
        own_id    = make_id(0x00)
        info_hash = make_id(0xFF)
        client = DHTClient(node_id=own_id)

        for i in range(1, 4):
            client._table.add(DHTNode(id=make_id(i), host=f"10.0.0.{i}", port=6881))

        mock_transport = MagicMock(spec=DHTTransport)

        async def fake_request(data, addr, txid, timeout=5.0):
            resp = _get_peers_with_values(txid, make_id(0xAA), [("1.2.3.4", 6881)])
            return (decode_response(resp), addr)

        mock_transport.request = fake_request
        client._transport = mock_transport

        peers = await client.get_peers(info_hash)
        assert peers.count(("1.2.3.4", 6881)) == 1

    @pytest.mark.asyncio
    async def test_get_peers_handles_node_timeout(self):
        """Timeout on one node should not abort the whole lookup."""
        own_id    = make_id(0x00)
        info_hash = make_id(0xFF)
        client = DHTClient(node_id=own_id)

        client._table.add(DHTNode(id=make_id(0x01), host="10.0.0.1", port=6881))
        client._table.add(DHTNode(id=make_id(0x02), host="10.0.0.2", port=6882))

        mock_transport = MagicMock(spec=DHTTransport)

        async def fake_request(data, addr, txid, timeout=5.0):
            if addr[0] == "10.0.0.1":
                raise asyncio.TimeoutError()
            resp = _get_peers_with_values(txid, make_id(0x02), [("3.3.3.3", 3333)])
            return (decode_response(resp), addr)

        mock_transport.request = fake_request
        client._transport = mock_transport

        peers = await client.get_peers(info_hash)
        assert ("3.3.3.3", 3333) in peers

    @pytest.mark.asyncio
    async def test_get_peers_values_as_bytes(self):
        """Some implementations return values as a single bytes object instead of list."""
        own_id    = make_id(0x00)
        info_hash = make_id(0xFF)
        client = DHTClient(node_id=own_id)
        client._table.add(DHTNode(id=make_id(0xAA), host="10.0.0.1", port=6881))

        mock_transport = MagicMock(spec=DHTTransport)

        async def fake_request(data, addr, txid, timeout=5.0):
            # Values as a flat bytes string (non-standard but seen in practice)
            raw_peers = compact_peer("9.9.9.9", 9999)
            resp = encode({
                b"t": txid,
                b"y": b"r",
                b"r": {b"id": make_id(0xAA), b"token": b"t", b"values": raw_peers},
            })
            return (decode_response(resp), addr)

        mock_transport.request = fake_request
        client._transport = mock_transport

        peers = await client.get_peers(info_hash)
        assert ("9.9.9.9", 9999) in peers


# ---------------------------------------------------------------------------
# DHTClient context manager
# ---------------------------------------------------------------------------

class TestDHTClientContextManager:
    @pytest.mark.asyncio
    async def test_context_manager_creates_transport(self):
        with patch("bittorrent.dht.create_dht_transport") as mock_create:
            mock_transport = MagicMock(spec=DHTTransport)
            mock_create.return_value = mock_transport

            async with DHTClient() as dht:
                assert dht._transport is mock_transport

            mock_transport.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_closes_on_exit(self):
        with patch("bittorrent.dht.create_dht_transport") as mock_create:
            mock_transport = MagicMock(spec=DHTTransport)
            mock_create.return_value = mock_transport

            async with DHTClient() as dht:
                pass

            assert dht._transport is None


# ---------------------------------------------------------------------------
# BEP 32 — DHT IPv6
# ---------------------------------------------------------------------------

class TestDHTIPv6:
    """Tests for BEP 32 IPv6 compact node/peer formats."""

    def _compact_node6(self, node_id: bytes, ip6: str, port: int) -> bytes:
        import socket
        return node_id + socket.inet_pton(socket.AF_INET6, ip6) + struct.pack("!H", port)

    def _compact_peer6(self, ip6: str, port: int) -> bytes:
        import socket
        return socket.inet_pton(socket.AF_INET6, ip6) + struct.pack("!H", port)

    def test_decode_compact_nodes6_single(self):
        node_id = bytes(range(20))
        raw = self._compact_node6(node_id, "::1", 6881)
        nodes = decode_compact_nodes6(raw)
        assert len(nodes) == 1
        assert nodes[0].id == node_id
        assert nodes[0].port == 6881
        assert "::1" in nodes[0].host or nodes[0].host == "::1"

    def test_decode_compact_nodes6_multiple(self):
        id1 = bytes([1]) * 20
        id2 = bytes([2]) * 20
        raw = (
            self._compact_node6(id1, "2001:db8::1", 6881) +
            self._compact_node6(id2, "2001:db8::2", 6882)
        )
        nodes = decode_compact_nodes6(raw)
        assert len(nodes) == 2
        assert nodes[0].id == id1
        assert nodes[1].id == id2

    def test_decode_compact_nodes6_wrong_length_raises(self):
        with pytest.raises(ValueError, match="38"):
            decode_compact_nodes6(b"\x00" * 37)

    def test_decode_compact_peers6_single(self):
        raw = self._compact_peer6("::1", 6881)
        peers = decode_compact_peers6(raw)
        assert len(peers) == 1
        assert peers[0][1] == 6881

    def test_decode_compact_peers6_multiple(self):
        raw = (
            self._compact_peer6("2001:db8::1", 1111) +
            self._compact_peer6("2001:db8::2", 2222)
        )
        peers = decode_compact_peers6(raw)
        assert len(peers) == 2
        assert peers[0][1] == 1111
        assert peers[1][1] == 2222

    def test_decode_compact_peers6_wrong_length_raises(self):
        with pytest.raises(ValueError, match="18"):
            decode_compact_peers6(b"\x00" * 17)

    def test_decode_compact_peers6_empty(self):
        assert decode_compact_peers6(b"") == []

    async def test_get_peers_reads_values6(self):
        """get_peers includes IPv6 peers from the 'values6' field."""
        import socket
        from unittest.mock import AsyncMock, MagicMock, patch
        from bittorrent.bencode import encode
        from bittorrent.dht import DHTClient, DHTNode, DHTTransport, decode_compact_nodes

        node_id = bytes(range(20))
        our_id  = bytes([0]) * 20

        ip6    = "2001:db8::42"
        port   = 6882
        peer6  = socket.inet_pton(socket.AF_INET6, ip6) + struct.pack("!H", port)

        # Build a get_peers response with values6 and no values
        resp = encode({
            b"t": b"\x00\x01",
            b"y": b"r",
            b"r": {
                b"id": node_id,
                b"token": b"tok",
                b"values6": peer6,
            },
        })

        mock_transport = MagicMock(spec=DHTTransport)
        async def fake_request(data, addr, txid, *, timeout=3.0):
            import bittorrent.bencode as _bc
            msg = _bc.decode(data)
            return _bc.decode(resp), addr
        mock_transport.request = fake_request

        with patch("bittorrent.dht.create_dht_transport", return_value=mock_transport):
            async with DHTClient(node_id=our_id) as dht:
                seed = DHTNode(id=node_id, host="1.2.3.4", port=6881)
                dht._table.add(seed)
                peers = await dht.get_peers(bytes([0xff]) * 20, timeout=5.0)

        assert any(p[1] == port for p in peers), f"IPv6 peer port {port} not in {peers}"

    async def test_get_peers_reads_nodes6(self):
        """Nodes from 'nodes6' are added to the routing table during lookup."""
        import socket
        from unittest.mock import MagicMock, patch
        from bittorrent.bencode import encode
        from bittorrent.dht import DHTClient, DHTNode, DHTTransport

        seed_id  = bytes([0xaa]) * 20
        node6_id = bytes([0xbb]) * 20
        our_id   = bytes([0x00]) * 20

        nodes6_raw = node6_id + socket.inet_pton(socket.AF_INET6, "::1") + struct.pack("!H", 9999)

        resp = encode({
            b"t": b"\x00\x01",
            b"y": b"r",
            b"r": {
                b"id": seed_id,
                b"token": b"tok",
                b"nodes6": nodes6_raw,
            },
        })

        call_count = [0]
        async def fake_request(data, addr, txid, *, timeout=3.0):
            import bittorrent.bencode as _bc
            call_count[0] += 1
            return _bc.decode(resp), addr

        mock_transport = MagicMock(spec=DHTTransport)
        mock_transport.request = fake_request

        with patch("bittorrent.dht.create_dht_transport", return_value=mock_transport):
            async with DHTClient(node_id=our_id) as dht:
                seed = DHTNode(id=seed_id, host="1.2.3.4", port=6881)
                dht._table.add(seed)
                await dht.get_peers(bytes([0xff]) * 20, timeout=5.0)

        # The node discovered via nodes6 should be in the routing table
        all_nodes = dht._table.find_closest(bytes([0xbb]) * 20, 20)
        assert any(n.id == node6_id for n in all_nodes), \
            f"nodes6 node {node6_id.hex()} not in routing table"
