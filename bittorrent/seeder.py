"""
BEP 3 — Seeding (upload) support.

After a download completes, the client can stay up as a seeder to share
pieces with other peers who are still downloading.

Architecture:
  - Seeder starts a TCP server on the configured port.
  - Each incoming connection gets its own ``_UploadPeer`` task.
  - A rechoke loop runs every RECHOKE_INTERVAL seconds, keeping at most
    MAX_UPLOAD_SLOTS peers unchoked (a simplified tit-for-tat).
  - When a new piece arrives during hybrid download+seed, broadcast_have()
    notifies all connected upload peers.

Wire protocol (upload side):
  1. Accept incoming handshake; validate info_hash.
  2. Send our handshake back.
  3. Send BITFIELD with all pieces we have.
  4. Unchoke the peer (up to MAX_UPLOAD_SLOTS slots).
  5. Loop reading REQUEST messages; read data from storage and send PIECE.
  6. Respect CHOKE/UNCHOKE state; wait when we've choked the peer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from bittorrent.peer import PeerConnection, PeerError

if TYPE_CHECKING:
    from bittorrent.piece_manager import PieceManager
    from bittorrent.storage import Storage
    from bittorrent.torrent import Torrent

log = logging.getLogger(__name__)

MAX_UPLOAD_SLOTS  = 4      # max simultaneous unchoked upload peers
RECHOKE_INTERVAL  = 10.0   # seconds between rechoke passes
KEEPALIVE_INTERVAL = 90.0  # seconds between keep-alives to upload peers


class Seeder:
    """Accepts incoming peer connections and serves pieces from local storage.

    Args:
        torrent:       The parsed Torrent metadata.
        storage:       Storage object to read piece data from.
        piece_manager: PieceManager that tracks which pieces are complete.
        info_hash:     20-byte info hash.
        peer_id:       Our 20-byte peer ID.
        port:          TCP port to listen on.
    """

    def __init__(
        self,
        torrent: "Torrent",
        storage: "Storage",
        piece_manager: "PieceManager",
        *,
        info_hash: bytes,
        peer_id: bytes,
        port: int = 6881,
    ) -> None:
        self._torrent  = torrent
        self._storage  = storage
        self._pm       = piece_manager
        self._info_hash = info_hash
        self._peer_id   = peer_id
        self._port      = port
        self._server: asyncio.AbstractServer | None = None
        self._peers: list[_UploadPeer] = []
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start the TCP server and serve peers until cancelled."""
        self._server = await asyncio.start_server(
            self._handle_connection, "0.0.0.0", self._port
        )
        rechoke_task = asyncio.create_task(self._rechoke_loop())
        self._tasks.add(rechoke_task)
        try:
            async with self._server:
                await self._server.serve_forever()
        finally:
            rechoke_task.cancel()
            try:
                await rechoke_task
            except asyncio.CancelledError:
                pass

    async def stop(self) -> None:
        """Shut down the server and close all upload peer connections."""
        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            self._server = None

        # Cancel all in-flight peer tasks
        for t in list(self._tasks):
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        for peer in list(self._peers):
            await peer.close()
        self._peers.clear()

    def broadcast_have(self, piece_index: int) -> None:
        """Notify all connected upload peers that we now have *piece_index*."""
        for peer in list(self._peers):
            if not peer.closed:
                asyncio.create_task(peer.send_have_safe(piece_index))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = _UploadPeer(self._torrent, self._storage, self._pm)
        self._peers.append(peer)

        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)

        try:
            await peer.serve(
                reader, writer,
                self._info_hash, self._peer_id,
                unchoke=(len([p for p in self._peers if not p.am_choking])
                         < MAX_UPLOAD_SLOTS),
            )
        except (PeerError, OSError) as exc:
            log.debug("Upload peer error: %s", exc)
        finally:
            await peer.close()
            self._peers = [p for p in self._peers if p is not peer]
            if task is not None:
                self._tasks.discard(task)

    async def _rechoke_loop(self) -> None:
        """Periodically rechoke to enforce MAX_UPLOAD_SLOTS."""
        while True:
            await asyncio.sleep(RECHOKE_INTERVAL)
            self._rechoke()

    def _rechoke(self) -> None:
        """Unchoke up to MAX_UPLOAD_SLOTS interested peers; choke the rest."""
        active = [p for p in self._peers if not p.closed and p.peer_interested]
        for i, peer in enumerate(active):
            if i < MAX_UPLOAD_SLOTS:
                if peer.am_choking:
                    asyncio.create_task(peer.send_unchoke_safe())
            else:
                if not peer.am_choking:
                    asyncio.create_task(peer.send_choke_safe())


class _UploadPeer:
    """State and I/O for one upload (seeding) connection."""

    def __init__(
        self,
        torrent: "Torrent",
        storage: "Storage",
        piece_manager: "PieceManager",
    ) -> None:
        self._torrent = torrent
        self._storage = storage
        self._pm      = piece_manager
        self._conn: PeerConnection | None = None
        self.closed: bool = False

    @property
    def am_choking(self) -> bool:
        return self._conn is None or self._conn.am_choking

    @property
    def peer_interested(self) -> bool:
        return self._conn is not None and self._conn.peer_interested

    async def serve(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        info_hash: bytes,
        peer_id: bytes,
        *,
        unchoke: bool = True,
    ) -> None:
        """Full lifecycle for one upload peer connection."""
        self._conn = await PeerConnection.accept(
            reader, writer, info_hash, peer_id
        )
        conn = self._conn

        # Send our bitfield so the peer knows what we have.
        bitfield = self._pm.bitfield_bytes()
        if any(b != 0 for b in bitfield):
            await conn.send_bitfield(bitfield)

        # Unchoke immediately if we have a slot.
        if unchoke:
            await conn.send_unchoke()

        ka_task = asyncio.create_task(self._keepalive_loop(conn))
        try:
            await self._request_loop(conn)
        finally:
            ka_task.cancel()
            try:
                await ka_task
            except asyncio.CancelledError:
                pass

    async def _keepalive_loop(self, conn: PeerConnection) -> None:
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL)
            try:
                from bittorrent.messages import encode_keepalive
                await conn._send_raw(encode_keepalive())
            except (PeerError, OSError):
                return

    async def _request_loop(self, conn: PeerConnection) -> None:
        """Read REQUEST messages and serve blocks until the peer disconnects."""
        while True:
            piece_index, block_offset, block_length = await conn.read_request()

            if conn.am_choking:
                # We choked this peer; ignore their request
                log.debug(
                    "Ignoring request from choked peer %s:%s",
                    conn.host, conn.port,
                )
                continue

            if not self._pm.is_complete_piece(piece_index):
                log.debug(
                    "Peer %s:%s requested piece %d we don't have",
                    conn.host, conn.port, piece_index,
                )
                continue

            try:
                piece_data = self._storage.read_piece(piece_index)
                block_data = piece_data[block_offset : block_offset + block_length]
                await conn.send_piece_block(piece_index, block_offset, block_data)
                log.debug(
                    "Served piece %d offset %d length %d to %s:%s",
                    piece_index, block_offset, block_length,
                    conn.host, conn.port,
                )
            except OSError as exc:
                raise PeerError(f"Storage read failed: {exc}") from exc

    async def send_have_safe(self, piece_index: int) -> None:
        if self._conn and not self.closed:
            try:
                await self._conn.send_have(piece_index)
            except (PeerError, OSError):
                pass

    async def send_unchoke_safe(self) -> None:
        if self._conn and not self.closed:
            try:
                await self._conn.send_unchoke()
            except (PeerError, OSError):
                pass

    async def send_choke_safe(self) -> None:
        if self._conn and not self.closed:
            try:
                await self._conn.send_choke()
            except (PeerError, OSError):
                pass

    async def close(self) -> None:
        self.closed = True
        if self._conn:
            await self._conn.close()
            self._conn = None
