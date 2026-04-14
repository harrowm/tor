"""
Peer manager — orchestrates concurrent connections and piece downloads.

PeerManager opens connections to multiple peers and runs one asyncio task
per peer. Each task loops: ask PieceManager for the next piece this peer has,
download it, verify it, write it to Storage, mark it complete.

Concurrency model:
  - One asyncio Task per peer (MAX_PEERS tasks max).
  - PieceManager is accessed only from the event loop (no locks needed).
  - Storage writes are synchronous but short-lived (seek + write).
  - Failed peers (disconnect, hash mismatch, timeout) are dropped quietly;
    their in-progress piece is returned to MISSING.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from bittorrent.peer import PeerConnection, PeerError
from bittorrent.piece_manager import PieceManager
from bittorrent.storage import Storage
from bittorrent.torrent import Torrent

log = logging.getLogger(__name__)

MAX_PEERS       = 30   # maximum simultaneous connections
CONNECT_TIMEOUT = 10.0  # seconds to wait for a TCP connection
STALL_WAIT      = 0.1   # seconds to wait when all remaining pieces are in-flight


@dataclass
class DownloadStats:
    """Live download statistics."""
    pieces_complete: int = 0
    pieces_total:    int = 0
    peers_active:    int = 0
    bytes_downloaded: int = 0

    @property
    def percent(self) -> float:
        if self.pieces_total == 0:
            return 0.0
        return 100.0 * self.pieces_complete / self.pieces_total


class PeerManager:
    """Manages a pool of peer connections and drives the download to completion."""

    def __init__(
        self,
        torrent: Torrent,
        piece_manager: PieceManager,
        storage: Storage,
        info_hash: bytes,
        peer_id: bytes,
    ) -> None:
        self._torrent       = torrent
        self._pm            = piece_manager
        self._storage       = storage
        self._info_hash     = info_hash
        self._peer_id       = peer_id
        self._active_tasks: set[asyncio.Task] = set()
        self._stats         = DownloadStats(pieces_total=torrent.num_pieces)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def stats(self) -> DownloadStats:
        return self._stats

    async def run(
        self,
        peers: list[tuple[str, int]],
        *,
        on_progress: callable | None = None,
        reannounce: callable | None = None,
        reannounce_interval: int = 1800,
    ) -> None:
        """Download the torrent by connecting to *peers* and fetching all pieces.

        Args:
            peers:               List of (ip, port) tuples from the tracker.
            on_progress:         Optional async callable(stats) called after each piece.
            reannounce:          Optional async callable(*, downloaded, left) that
                                 returns a TrackerResponse with fresh peers.
            reannounce_interval: Seconds between re-announce calls (default 1800).
        """
        self._storage.allocate()

        peer_queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
        for peer in peers:
            await peer_queue.put(peer)

        # _all_tasks grows over time as re-announce spawns fresh workers.
        _all_tasks: set[asyncio.Task] = set()

        def _spawn_workers() -> None:
            alive    = sum(1 for t in _all_tasks if not t.done())
            to_spawn = min(peer_queue.qsize(), MAX_PEERS - alive)
            for _ in range(to_spawn):
                t = asyncio.create_task(self._worker(peer_queue, on_progress))
                _all_tasks.add(t)

        _spawn_workers()

        if reannounce is not None:
            async def _reannounce_loop() -> None:
                while not self._pm.is_complete():
                    await asyncio.sleep(reannounce_interval)
                    if self._pm.is_complete():
                        return
                    try:
                        left = max(
                            0, self._torrent.total_length - self._stats.bytes_downloaded
                        )
                        resp = await reannounce(
                            downloaded=self._stats.bytes_downloaded,
                            left=left,
                        )
                        for peer in resp.peers:
                            await peer_queue.put(peer)
                        log.info("Re-announce: %d peers added", len(resp.peers))
                        _spawn_workers()
                    except Exception as exc:
                        log.warning("Re-announce failed: %s", exc)

            _all_tasks.add(asyncio.create_task(_reannounce_loop()))

        # Event loop: re-check _all_tasks each iteration so newly-spawned
        # workers (added by _spawn_workers inside _reannounce_loop) are awaited.
        while True:
            active = {t for t in _all_tasks if not t.done()}
            if not active:
                break
            if self._pm.is_complete():
                for t in active:
                    t.cancel()
                await asyncio.gather(*active, return_exceptions=True)
                break
            await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)

        if not self._pm.is_complete():
            done, total = self._pm.progress()
            raise RuntimeError(
                f"Download incomplete: {done}/{total} pieces after exhausting peers"
            )

        log.info("Download complete: %d/%d pieces", *self._pm.progress())

    async def _worker(
        self,
        peer_queue: asyncio.Queue,
        on_progress: callable | None,
    ) -> None:
        """One worker: repeatedly pull a peer from the queue and download pieces."""
        while not self._pm.is_complete():
            try:
                host, port = peer_queue.get_nowait()
            except asyncio.QueueEmpty:
                return  # no more peers to try

            await self._download_from_peer(host, port, on_progress)

    async def _download_from_peer(
        self,
        host: str,
        port: int,
        on_progress: callable | None,
    ) -> None:
        """Connect to one peer and download as many pieces as possible from it."""
        try:
            conn = await PeerConnection.open(
                host, port,
                self._info_hash, self._peer_id,
                timeout=CONNECT_TIMEOUT,
            )
        except PeerError as exc:
            log.debug("Cannot connect to %s:%s: %s", host, port, exc)
            return

        log.info("Connected to %s:%s", host, port)
        self._stats.peers_active += 1

        # Inform piece manager of peer's availability
        if conn.bitfield:
            self._pm.record_bitfield(conn.bitfield)

        try:
            await self._download_loop(conn, on_progress)
        except PeerError as exc:
            log.debug("Peer %s:%s error: %s", host, port, exc)
        finally:
            self._stats.peers_active -= 1
            await conn.close()

    async def _download_loop(
        self,
        conn: PeerConnection,
        on_progress: callable | None,
    ) -> None:
        """Keep downloading pieces from *conn* until it has nothing we need.

        When ``next_piece`` returns None but pieces are still IN_PROGRESS we
        wait briefly and retry.  This prevents a worker from exiting and
        consuming its peer just before another peer's in-flight piece fails and
        returns to MISSING — which would leave no worker alive to pick it up.
        """
        while not self._pm.is_complete():
            piece_index = self._pm.next_piece(conn.bitfield)
            if piece_index is None:
                if self._pm.num_in_progress == 0:
                    # No MISSING pieces for this peer and nothing in-flight.
                    log.debug("Peer %s:%s has nothing we need", conn.host, conn.port)
                    return
                # Pieces are in-flight; wait in case one fails back to MISSING.
                await asyncio.sleep(STALL_WAIT)
                continue

            self._pm.mark_in_progress(piece_index)
            piece_size = self._pm.piece_size(piece_index)
            expected_hash = self._torrent.piece_hashes[piece_index]

            try:
                data = await conn.download_piece(piece_index, piece_size, expected_hash)
            except PeerError as exc:
                self._pm.mark_missing(piece_index)
                raise  # let _download_from_peer log it and close connection

            # In end-game mode two workers may race on the same piece.
            # Check before marking complete so stats are only updated once.
            # This is safe: no await between the check and mark_complete, so
            # no other coroutine can slip in between them.
            already_complete = self._pm.is_complete_piece(piece_index)
            self._storage.write_piece(piece_index, data)
            self._pm.mark_complete(piece_index)

            if not already_complete:
                self._stats.pieces_complete += 1
                self._stats.bytes_downloaded += len(data)
                done, total = self._pm.progress()
                log.info("Piece %d/%d complete (%.1f%%)", done, total,
                         self._stats.percent)

                if on_progress:
                    await on_progress(self._stats)
