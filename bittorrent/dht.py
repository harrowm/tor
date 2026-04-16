"""
DHT peer discovery — BEP 5 (Kademlia-based).

Implements a minimal DHT client that can:
  1. Bootstrap from well-known nodes.
  2. Perform iterative get_peers lookups to find peers for an info_hash.

Architecture:
  - DHTNode: a node in the DHT (id, host, port).
  - KBucket: holds up to K=8 nodes sorted by last-seen time.
  - RoutingTable: 160 k-buckets partitioned by XOR distance.
  - DHTTransport: UDP socket wrapper with request/response correlation.
  - DHTClient: high-level API (bootstrap, get_peers).

Usage::

    async with DHTClient() as dht:
        await dht.bootstrap()
        peers = await dht.get_peers(info_hash)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Callable

from bittorrent.bencode import DecodeError, _decode_next, encode

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K        = 8      # max nodes per bucket
ALPHA    = 8      # parallel lookup concurrency (more = faster convergence)
MAX_HOPS = 20     # maximum iterative lookup rounds

BOOTSTRAP_NODES: list[tuple[str, int]] = [
    ("router.bittorrent.com",  6881),
    ("router.utorrent.com",    6881),
    ("dht.transmissionbt.com", 6881),
]

REQUEST_TIMEOUT = 3.0   # seconds to wait for a response
FIND_TIMEOUT    = 30.0  # seconds for a complete get_peers lookup


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def xor_distance(a: bytes, b: bytes) -> int:
    """XOR metric between two 20-byte node IDs; lower is closer."""
    if len(a) != 20 or len(b) != 20:
        raise ValueError(f"Node IDs must be 20 bytes, got {len(a)} and {len(b)}")
    # Fast XOR via int
    return int.from_bytes(a, "big") ^ int.from_bytes(b, "big")


def random_node_id() -> bytes:
    """Generate a random 20-byte DHT node ID."""
    return os.urandom(20)


def decode_compact_nodes(data: bytes) -> list["DHTNode"]:
    """Decode BEP 5 compact node list (26 bytes per node: 20-byte ID + 4-byte IP + 2-byte port)."""
    if len(data) % 26 != 0:
        raise ValueError(f"Compact nodes data length {len(data)} is not a multiple of 26")
    nodes = []
    for i in range(0, len(data), 26):
        chunk = data[i : i + 26]
        node_id = chunk[:20]
        ip_bytes = chunk[20:24]
        port = struct.unpack("!H", chunk[24:26])[0]
        host = ".".join(str(b) for b in ip_bytes)
        nodes.append(DHTNode(id=node_id, host=host, port=port))
    return nodes


def encode_compact_nodes(nodes: list["DHTNode"]) -> bytes:
    """Encode a list of DHTNode to BEP 5 compact format."""
    result = bytearray()
    for node in nodes:
        result += node.id
        parts = node.host.split(".")
        if len(parts) != 4:
            raise ValueError(f"Invalid IPv4 host: {node.host!r}")
        for part in parts:
            result.append(int(part))
        result += struct.pack("!H", node.port)
    return bytes(result)


def decode_compact_peers(data: bytes) -> list[tuple[str, int]]:
    """Decode BEP 23 compact peer list (6 bytes per peer: 4-byte IP + 2-byte port)."""
    if len(data) % 6 != 0:
        raise ValueError(f"Compact peers length {len(data)} is not a multiple of 6")
    peers = []
    for i in range(0, len(data), 6):
        chunk = data[i : i + 6]
        host = ".".join(str(b) for b in chunk[:4])
        port = struct.unpack("!H", chunk[4:6])[0]
        peers.append((host, port))
    return peers


def decode_compact_nodes6(data: bytes) -> list["DHTNode"]:
    """BEP 32: decode compact IPv6 node list (38 bytes per node: 20-byte ID + 16-byte IPv6 + 2-byte port)."""
    if len(data) % 38 != 0:
        raise ValueError(f"Compact nodes6 length {len(data)} is not a multiple of 38")
    nodes = []
    for i in range(0, len(data), 38):
        chunk   = data[i : i + 38]
        node_id = chunk[:20]
        host    = socket.inet_ntop(socket.AF_INET6, chunk[20:36])
        port    = struct.unpack("!H", chunk[36:38])[0]
        nodes.append(DHTNode(id=node_id, host=host, port=port))
    return nodes


def decode_compact_peers6(data: bytes) -> list[tuple[str, int]]:
    """BEP 32: decode compact IPv6 peer list (18 bytes per peer: 16-byte IPv6 + 2-byte port)."""
    if len(data) % 18 != 0:
        raise ValueError(f"Compact peers6 length {len(data)} is not a multiple of 18")
    peers = []
    for i in range(0, len(data), 18):
        chunk = data[i : i + 18]
        host  = socket.inet_ntop(socket.AF_INET6, chunk[:16])
        port  = struct.unpack("!H", chunk[16:18])[0]
        peers.append((host, port))
    return peers


# ---------------------------------------------------------------------------
# DHT message encoding / decoding
# ---------------------------------------------------------------------------

def encode_ping(txid: bytes, node_id: bytes) -> bytes:
    """Encode a DHT ping query."""
    return encode({
        b"t": txid,
        b"y": b"q",
        b"q": b"ping",
        b"a": {b"id": node_id},
    })


def encode_find_node(txid: bytes, node_id: bytes, target: bytes) -> bytes:
    """Encode a find_node query."""
    return encode({
        b"t": txid,
        b"y": b"q",
        b"q": b"find_node",
        b"a": {b"id": node_id, b"target": target},
    })


def encode_get_peers(txid: bytes, node_id: bytes, info_hash: bytes) -> bytes:
    """Encode a get_peers query."""
    return encode({
        b"t": txid,
        b"y": b"q",
        b"q": b"get_peers",
        b"a": {b"id": node_id, b"info_hash": info_hash},
    })


def decode_response(data: bytes) -> dict:
    """Decode a DHT response message.

    Returns the decoded dict. Raises ValueError if the message is not a
    valid response (missing keys, wrong type, etc.).
    """
    try:
        msg, _ = _decode_next(data, 0)
    except (DecodeError, ValueError) as exc:
        raise ValueError(f"Cannot decode DHT message: {exc}") from exc

    if not isinstance(msg, dict):
        raise ValueError("DHT message is not a dict")
    return msg


def is_response(msg: dict) -> bool:
    return msg.get(b"y") == b"r"


def is_error(msg: dict) -> bool:
    return msg.get(b"y") == b"e"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DHTNode:
    """A node in the DHT network."""
    id: bytes          # 20-byte node ID
    host: str
    port: int
    last_seen: float = field(default_factory=time.monotonic)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DHTNode):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


class KBucket:
    """Holds up to K nodes for a particular XOR distance range.

    Replacement cache (pending list) stores overflow candidates.
    When a slot opens, the first pending node is promoted.
    """

    def __init__(self, k: int = K) -> None:
        self.k    = k
        self.nodes: list[DHTNode] = []
        self.pending: list[DHTNode] = []

    def __len__(self) -> int:
        return len(self.nodes)

    def add(self, node: DHTNode) -> None:
        """Insert or refresh *node*. Returns True if added/updated."""
        # Update last_seen if already present
        for i, existing in enumerate(self.nodes):
            if existing.id == node.id:
                self.nodes[i] = node
                return

        if len(self.nodes) < self.k:
            self.nodes.append(node)
        else:
            # Bucket full — stash in pending (BEP 5 replacement cache)
            for i, p in enumerate(self.pending):
                if p.id == node.id:
                    self.pending[i] = node
                    return
            self.pending.append(node)

    def remove(self, node_id: bytes) -> None:
        """Remove a node (e.g. after timeout). Promote first pending."""
        self.nodes = [n for n in self.nodes if n.id != node_id]
        if self.pending:
            self.nodes.append(self.pending.pop(0))

    def get_nodes(self) -> list[DHTNode]:
        return list(self.nodes)


class RoutingTable:
    """160 k-buckets indexed by leading bit of XOR distance."""

    def __init__(self, own_id: bytes) -> None:
        self.own_id = own_id
        self._buckets: list[KBucket] = [KBucket() for _ in range(160)]

    def _bucket_index(self, node_id: bytes) -> int:
        dist = xor_distance(self.own_id, node_id)
        if dist == 0:
            return 159  # same node — put in farthest bucket (shouldn't happen)
        # Leading zero bits = 159 − floor(log2(dist))
        return 159 - dist.bit_length() + 1

    def add(self, node: DHTNode) -> None:
        if node.id == self.own_id:
            return
        idx = self._bucket_index(node.id)
        self._buckets[idx].add(node)

    def remove(self, node_id: bytes) -> None:
        idx = self._bucket_index(node_id)
        self._buckets[idx].remove(node_id)

    def find_closest(self, target: bytes, n: int = K) -> list[DHTNode]:
        """Return the *n* nodes closest to *target* by XOR distance."""
        all_nodes: list[DHTNode] = []
        for bucket in self._buckets:
            all_nodes.extend(bucket.get_nodes())
        all_nodes.sort(key=lambda nd: xor_distance(nd.id, target))
        return all_nodes[:n]

    def size(self) -> int:
        return sum(len(b) for b in self._buckets)


# ---------------------------------------------------------------------------
# UDP transport
# ---------------------------------------------------------------------------

class DHTTransport(asyncio.DatagramProtocol):
    """asyncio UDP transport for DHT messages.

    Sends encoded query datagrams and correlates responses via the
    transaction ID (b"t" field).  Each outstanding request has an
    asyncio.Future stored in ``_pending``.
    """

    def __init__(self) -> None:
        self._transport: asyncio.DatagramTransport | None = None
        self._pending: dict[bytes, asyncio.Future] = {}
        self._query_handler: Callable[[bytes, tuple[str, int]], None] | None = None

    # ------------------------------------------------------------------
    # asyncio.DatagramProtocol interface
    # ------------------------------------------------------------------

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self._transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            msg = decode_response(data)
        except ValueError:
            return

        txid = msg.get(b"t")
        if txid and txid in self._pending:
            fut = self._pending.pop(txid)
            if not fut.done():
                fut.set_result((msg, addr))
        elif self._query_handler and msg.get(b"y") == b"q":
            self._query_handler(data, addr)

    def error_received(self, exc: Exception) -> None:
        log.debug("DHT UDP error: %s", exc)

    def connection_lost(self, exc: Exception | None) -> None:
        # Fail all pending futures and immediately consume the exception to
        # prevent asyncio from logging "Future exception was never retrieved".
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("DHT transport closed"))
                fut.exception()  # mark as retrieved
        self._pending.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, data: bytes, addr: tuple[str, int]) -> None:
        if self._transport:
            self._transport.sendto(data, addr)

    async def request(
        self,
        data: bytes,
        addr: tuple[str, int],
        txid: bytes,
        *,
        timeout: float = REQUEST_TIMEOUT,
    ) -> tuple[dict, tuple[str, int]]:
        """Send *data* to *addr* and wait for the matching response.

        Returns (response_dict, sender_addr).
        Raises asyncio.TimeoutError if no response arrives within *timeout*.
        """
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[txid] = fut
        self.send(data, addr)
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(txid, None)
            raise
        except Exception:
            self._pending.pop(txid, None)
            raise

    def close(self) -> None:
        if self._transport:
            self._transport.close()
            self._transport = None


async def create_dht_transport(bind_port: int = 0) -> DHTTransport:
    """Create and bind a DHT UDP transport. Returns the protocol object."""
    loop = asyncio.get_event_loop()
    _, protocol = await loop.create_datagram_endpoint(
        DHTTransport,
        local_addr=("0.0.0.0", bind_port),
    )
    return protocol  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# DHT client
# ---------------------------------------------------------------------------

class DHTClient:
    """High-level DHT client: bootstrap + get_peers.

    Use as an async context manager::

        async with DHTClient() as dht:
            await dht.bootstrap()
            peers = await dht.get_peers(info_hash)
    """

    def __init__(self, node_id: bytes | None = None, bind_port: int = 0) -> None:
        self._node_id   = node_id or random_node_id()
        self._bind_port = bind_port
        self._transport: DHTTransport | None = None
        self._table     = RoutingTable(self._node_id)
        self._txid_ctr  = 0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "DHTClient":
        self._transport = await create_dht_transport(self._bind_port)
        return self

    async def __aexit__(self, *_) -> None:
        if self._transport:
            self._transport.close()
            self._transport = None

    # ------------------------------------------------------------------
    # Transaction ID
    # ------------------------------------------------------------------

    def _next_txid(self) -> bytes:
        self._txid_ctr = (self._txid_ctr + 1) % 65536
        return self._txid_ctr.to_bytes(2, "big")

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    async def bootstrap(
        self,
        nodes: list[tuple[str, int]] | None = None,
    ) -> int:
        """Ping bootstrap nodes and do a find_node for our own ID.

        Returns the number of nodes added to the routing table.
        *nodes* defaults to BOOTSTRAP_NODES.
        """
        if nodes is None:
            nodes = BOOTSTRAP_NODES

        tasks = [self._bootstrap_node(host, port) for host, port in nodes]
        await asyncio.gather(*tasks, return_exceptions=True)

        if self._table.size() == 0:
            log.warning("DHT bootstrap: no nodes responded — DHT lookup will fail")
            return 0

        # Self-lookup to fill nearby buckets
        await self._find_node_iterative(self._node_id)
        return self._table.size()

    async def _bootstrap_node(self, host: str, port: int) -> None:
        # Resolve hostname asynchronously — passing a hostname directly to
        # sendto() triggers a blocking getaddrinfo() call that stalls the loop.
        try:
            loop = asyncio.get_event_loop()
            infos = await asyncio.wait_for(
                loop.getaddrinfo(host, port, type=socket.SOCK_DGRAM),
                timeout=5.0,
            )
            ip = infos[0][4][0]
        except Exception as exc:
            log.debug("Bootstrap DNS %s failed: %s", host, exc)
            return

        txid = self._next_txid()
        data = encode_ping(txid, self._node_id)
        try:
            resp, _ = await self._transport.request(data, (ip, port), txid)
            r = resp.get(b"r", {})
            node_id = r.get(b"id", b"")
            if len(node_id) == 20:
                self._table.add(DHTNode(id=node_id, host=ip, port=port))
                log.debug("Bootstrap node %s (%s):%d added (id=%s)", host, ip, port, node_id.hex()[:8])
        except (asyncio.TimeoutError, OSError) as exc:
            log.debug("Bootstrap %s (%s):%d failed: %s", host, ip, port, exc)

    # ------------------------------------------------------------------
    # find_node (iterative — fills routing table)
    # ------------------------------------------------------------------

    async def _find_node_iterative(self, target: bytes) -> list[DHTNode]:
        """Iterative find_node to fill the routing table near *target*."""
        queried:    set[bytes]               = set()
        discovered: dict[bytes, DHTNode]     = {
            n.id: n for n in self._table.find_closest(target, K * 2)
        }

        if not discovered:
            return []

        def _closest_k() -> list[DHTNode]:
            return sorted(
                discovered.values(),
                key=lambda n: xor_distance(n.id, target),
            )[:K]

        for _ in range(MAX_HOPS):
            candidates = sorted(
                [n for n in discovered.values() if n.id not in queried],
                key=lambda n: xor_distance(n.id, target),
            )[:ALPHA]

            if not candidates:
                break

            tasks = [self._query_find_node(n, target) for n in candidates]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for n, result in zip(candidates, results):
                queried.add(n.id)
                if isinstance(result, list):
                    for new_node in result:
                        if new_node.id not in discovered:
                            discovered[new_node.id] = new_node
                        self._table.add(new_node)

            if all(n.id in queried for n in _closest_k()):
                break

        return self._table.find_closest(target, K)

    async def _query_find_node(
        self, node: DHTNode, target: bytes
    ) -> list[DHTNode]:
        txid = self._next_txid()
        data = encode_find_node(txid, self._node_id, target)
        try:
            resp, _ = await self._transport.request(data, (node.host, node.port), txid)
        except (asyncio.TimeoutError, OSError) as exc:
            log.debug("find_node %s:%d timeout: %s", node.host, node.port, exc)
            return []

        r = resp.get(b"r", {})
        node_id = r.get(b"id", b"")
        if len(node_id) == 20:
            self._table.add(DHTNode(id=node_id, host=node.host, port=node.port))

        nodes: list[DHTNode] = []
        raw_nodes = r.get(b"nodes", b"")
        if isinstance(raw_nodes, bytes) and raw_nodes:
            try:
                nodes.extend(decode_compact_nodes(raw_nodes))
            except ValueError:
                pass
        # BEP 32: also handle IPv6 nodes in find_node responses
        raw_nodes6 = r.get(b"nodes6", b"")
        if isinstance(raw_nodes6, bytes) and raw_nodes6 and len(raw_nodes6) % 38 == 0:
            try:
                nodes.extend(decode_compact_nodes6(raw_nodes6))
            except ValueError:
                pass
        return nodes

    # ------------------------------------------------------------------
    # get_peers (the main API)
    # ------------------------------------------------------------------

    async def get_peers(
        self,
        info_hash: bytes,
        *,
        timeout: float = FIND_TIMEOUT,
    ) -> list[tuple[str, int]]:
        """Find peers for *info_hash* using an iterative get_peers lookup.

        Returns a list of (ip, port) tuples.
        Returns [] if no peers found within *timeout* seconds.
        """
        try:
            return await asyncio.wait_for(
                self._get_peers_iterative(info_hash),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            log.debug("get_peers timed out after %.1fs", timeout)
            return []

    async def _get_peers_iterative(self, info_hash: bytes) -> list[tuple[str, int]]:
        """Proper Kademlia iterative get_peers.

        Maintains a *local* set of all discovered nodes so that timed-out
        queries don't cause premature convergence (the old bug: when all ALPHA
        queries timed out, the routing-table closest set was unchanged → we
        broke out after a single hop).

        Stopping condition: all K closest *seen* nodes have been queried.
        """
        queried:    set[bytes]               = set()
        all_peers:  list[tuple[str, int]]    = []
        seen_peers: set[tuple[str, int]]     = set()

        # All discovered nodes keyed by ID.  Start from routing table.
        discovered: dict[bytes, DHTNode] = {
            n.id: n for n in self._table.find_closest(info_hash, K * 2)
        }

        if not discovered:
            log.debug("get_peers: routing table empty, no nodes to query")
            return []

        def _closest_k() -> list[DHTNode]:
            return sorted(
                discovered.values(),
                key=lambda n: xor_distance(n.id, info_hash),
            )[:K]

        for _ in range(MAX_HOPS):
            # Pick ALPHA closest unqueried nodes from *all* discovered nodes.
            candidates = sorted(
                [n for n in discovered.values() if n.id not in queried],
                key=lambda n: xor_distance(n.id, info_hash),
            )[:ALPHA]

            if not candidates:
                break

            tasks = [self._query_get_peers(n, info_hash) for n in candidates]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for node, result in zip(candidates, results):
                queried.add(node.id)
                if isinstance(result, tuple):
                    peers, new_nodes = result
                    for p in peers:
                        if p not in seen_peers:
                            seen_peers.add(p)
                            all_peers.append(p)
                    for n in new_nodes:
                        if n.id not in discovered:
                            discovered[n.id] = n
                        self._table.add(n)

            # Stop when the K closest discovered nodes are all queried.
            if all(n.id in queried for n in _closest_k()):
                break

        return all_peers

    async def _query_get_peers(
        self, node: DHTNode, info_hash: bytes
    ) -> tuple[list[tuple[str, int]], list[DHTNode]]:
        """Query one node. Returns (peers, closer_nodes)."""
        txid = self._next_txid()
        data = encode_get_peers(txid, self._node_id, info_hash)
        try:
            resp, _ = await self._transport.request(data, (node.host, node.port), txid)
        except (asyncio.TimeoutError, OSError) as exc:
            log.debug("get_peers %s:%d timeout: %s", node.host, node.port, exc)
            return [], []

        r = resp.get(b"r", {})
        node_id = r.get(b"id", b"")
        if len(node_id) == 20:
            self._table.add(DHTNode(id=node_id, host=node.host, port=node.port))

        peers: list[tuple[str, int]] = []
        if b"values" in r:
            raw_values = r[b"values"]
            if isinstance(raw_values, list):
                for item in raw_values:
                    if isinstance(item, bytes) and len(item) == 6:
                        try:
                            peers.extend(decode_compact_peers(item))
                        except ValueError:
                            pass
                    elif isinstance(item, bytes) and len(item) == 18:
                        try:
                            peers.extend(decode_compact_peers6(item))
                        except ValueError:
                            pass
            elif isinstance(raw_values, bytes):
                try:
                    peers.extend(decode_compact_peers(raw_values))
                except ValueError:
                    pass

        # BEP 32: also read IPv6 peers
        if b"values6" in r:
            raw6 = r[b"values6"]
            if isinstance(raw6, bytes) and raw6 and len(raw6) % 18 == 0:
                try:
                    peers.extend(decode_compact_peers6(raw6))
                except ValueError:
                    pass

        nodes: list[DHTNode] = []
        raw_nodes = r.get(b"nodes", b"")
        if isinstance(raw_nodes, bytes) and raw_nodes:
            try:
                nodes = decode_compact_nodes(raw_nodes)
            except ValueError:
                pass

        # BEP 32: also read IPv6 nodes from get_peers response
        raw_nodes6 = r.get(b"nodes6", b"")
        if isinstance(raw_nodes6, bytes) and raw_nodes6 and len(raw_nodes6) % 38 == 0:
            try:
                nodes.extend(decode_compact_nodes6(raw_nodes6))
            except ValueError:
                pass

        return peers, nodes
