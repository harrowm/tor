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

from bittorrent.messages import MSG_EXTENDED, PEX_LOCAL_ID, decode_pex_peers, encode_keepalive
from bittorrent.peer import PeerConnection, PeerError
from bittorrent.piece_manager import PieceManager
from bittorrent.storage import Storage
from bittorrent.torrent import Torrent
from bittorrent.webseed import WebSeedError, build_webseed_clients

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
        *,
        use_utp: bool = False,
        on_piece_complete: "callable[[int], None] | None" = None,
    ) -> None:
        self._torrent           = torrent
        self._pm                = piece_manager
        self._storage           = storage
        self._info_hash         = info_hash
        self._peer_id           = peer_id
        self._use_utp           = use_utp
        self._on_piece_complete = on_piece_complete
        self._active_tasks: set[asyncio.Task] = set()
        # Pre-seed stats with any pieces already complete (resume case).
        already_done = piece_manager.progress()[0]
        self._stats  = DownloadStats(
            pieces_complete=already_done,
            pieces_total=torrent.num_pieces,
            bytes_downloaded=already_done * torrent.piece_length,
        )

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
        allocate: bool = True,
    ) -> None:
        """Download the torrent by connecting to *peers* and fetching all pieces.

        Args:
            peers:               List of (ip, port) tuples from the tracker.
            on_progress:         Optional async callable(stats) called after each piece.
            reannounce:          Optional async callable(*, downloaded, left) that
                                 returns a TrackerResponse with fresh peers.
            reannounce_interval: Seconds between re-announce calls (default 1800).
            allocate:            If False, skip storage.allocate() (caller already did it).
        """
        if allocate:
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

        # BEP 19: spawn one web-seed worker per URL (if any)
        webseed_clients = build_webseed_clients(self._torrent)
        _webseed_session = None
        if webseed_clients:
            _webseed_session = self._create_http_session()
            for client in webseed_clients:
                t = asyncio.create_task(
                    self._webseed_worker(client, _webseed_session, on_progress)
                )
                _all_tasks.add(t)

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

        if _webseed_session is not None:
            await _webseed_session.close()

        if not self._pm.is_complete():
            done, total = self._pm.progress()
            raise RuntimeError(
                f"Download incomplete: {done}/{total} pieces after exhausting peers"
            )

        log.info("Download complete: %d/%d pieces", *self._pm.progress())

    def _create_http_session(self) -> "aiohttp.ClientSession":
        """Create an aiohttp.ClientSession for web seed requests.

        Isolated into its own method so tests can easily mock it without
        needing to patch module-level imports.
        """
        import aiohttp
        return aiohttp.ClientSession()

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

            await self._download_from_peer(host, port, on_progress, peer_queue)

    async def _webseed_worker(
        self,
        client: "WebSeedClient",
        session: "aiohttp.ClientSession",
        on_progress: callable | None,
    ) -> None:
        """Download pieces from a BEP 19 web seed until the torrent is complete.

        Pieces are fetched sequentially from the web seed.  Any piece that is
        already COMPLETE or IN_PROGRESS is skipped (peers take priority).
        Web-seed errors are logged and retried on the next piece.
        """
        from bittorrent.webseed import WebSeedClient

        while not self._pm.is_complete():
            # Ask for any missing piece (no bitfield restriction — HTTP serves all)
            piece_index = self._pm.next_piece(None)
            if piece_index is None:
                if self._pm.num_in_progress == 0:
                    return  # nothing left for us to do
                await asyncio.sleep(STALL_WAIT)
                continue

            self._pm.mark_in_progress(piece_index)

            try:
                data = await client.fetch_piece(session, piece_index)
            except WebSeedError as exc:
                self._pm.mark_missing(piece_index)
                log.debug("Web seed error on piece %d: %s", piece_index, exc)
                await asyncio.sleep(1.0)   # back off briefly before retrying
                continue

            already_complete = self._pm.is_complete_piece(piece_index)
            self._storage.write_piece(piece_index, data)
            self._pm.mark_complete(piece_index)

            if not already_complete:
                self._stats.pieces_complete += 1
                self._stats.bytes_downloaded += len(data)
                done, total = self._pm.progress()
                log.info(
                    "Web seed piece %d/%d complete (%.1f%%)",
                    done, total, self._stats.percent,
                )
                if on_progress:
                    await on_progress(self._stats)

    async def _open_connection(
        self,
        host: str,
        port: int,
    ) -> "PeerConnection | None":
        """Try TCP first, then uTP (BEP 29) as fallback if enabled.

        Returns a connected PeerConnection, or None if all transports fail.
        """
        # Try TCP
        try:
            return await PeerConnection.open(
                host, port,
                self._info_hash, self._peer_id,
                timeout=CONNECT_TIMEOUT,
                extension_protocol=True,
            )
        except (PeerError, OSError) as exc:
            if not self._use_utp:
                log.debug("Cannot connect to %s:%s: %s", host, port, exc)
                return None
            log.debug("TCP to %s:%s failed: %s — trying uTP", host, port, exc)

        # Fallback: uTP (only when explicitly enabled)
        from bittorrent.utp import UTPError
        try:
            return await PeerConnection.open_utp(
                host, port,
                self._info_hash, self._peer_id,
                timeout=CONNECT_TIMEOUT,
                extension_protocol=True,
            )
        except (PeerError, UTPError, OSError) as exc:
            log.debug("uTP to %s:%s also failed: %s", host, port, exc)

        return None

    async def _download_from_peer(
        self,
        host: str,
        port: int,
        on_progress: callable | None,
        peer_queue: asyncio.Queue,
    ) -> None:
        """Connect to one peer and download as many pieces as possible from it."""
        conn = await self._open_connection(host, port)
        if conn is None:
            return

        log.info("Connected to %s:%s", host, port)
        self._stats.peers_active += 1

        # Inform piece manager of peer's availability
        if conn.bitfield:
            self._pm.record_bitfield(conn.bitfield)

        # BEP 10 + BEP 11: attempt extension handshake to negotiate ut_pex
        if conn.remote_supports_extensions:
            try:
                await conn.do_extension_handshake(
                    {b"ut_pex": PEX_LOCAL_ID}, timeout=5.0
                )
                log.debug(
                    "Extension handshake OK with %s:%s, ut_pex=%s",
                    host, port,
                    conn.peer_ext_id(b"ut_pex"),
                )
            except (PeerError, OSError) as exc:
                log.debug("Extension handshake failed with %s:%s: %s", host, port, exc)
                # Non-fatal — continue without PEX

        async def _keepalive_loop() -> None:
            """Send a keep-alive every 90 s so the peer doesn't time us out."""
            while True:
                await asyncio.sleep(90)
                try:
                    await conn._send_raw(encode_keepalive())
                except (PeerError, OSError):
                    return

        ka_task = asyncio.create_task(_keepalive_loop())
        try:
            await self._download_loop(conn, on_progress, peer_queue)
        except (PeerError, OSError) as exc:
            log.debug("Peer %s:%s error: %s", host, port, exc)
        finally:
            ka_task.cancel()
            try:
                await ka_task
            except asyncio.CancelledError:
                pass
            self._stats.peers_active -= 1
            await conn.close()

    def _drain_pex(
        self,
        conn: PeerConnection,
        peer_queue: asyncio.Queue,
    ) -> None:
        """Process any PEX messages buffered in conn._pending.

        Extended messages with PEX_LOCAL_ID are decoded and their peers added
        to peer_queue.  All other pending messages are left untouched.
        """
        kept: list = []
        for msg in conn._pending:
            if (
                msg.msg_id == MSG_EXTENDED
                and msg.payload
                and msg.payload[0] == PEX_LOCAL_ID
            ):
                # Parse the bencoded PEX payload
                try:
                    from bittorrent.bencode import decode as _bdecode
                    d = _bdecode(msg.payload[1:])
                    if isinstance(d, dict):
                        added = d.get(b"added", b"")
                        if isinstance(added, bytes):
                            for peer in decode_pex_peers(added):
                                try:
                                    peer_queue.put_nowait(peer)
                                except asyncio.QueueFull:
                                    pass
                            log.debug(
                                "PEX: received %d peer(s)",
                                len(decode_pex_peers(added)),
                            )
                except Exception as exc:
                    log.debug("PEX decode error: %s", exc)
            else:
                kept.append(msg)
        conn._pending = kept

    async def _download_loop(
        self,
        conn: PeerConnection,
        on_progress: callable | None,
        peer_queue: asyncio.Queue,
    ) -> None:
        """Keep downloading pieces from *conn* until it has nothing we need.

        When ``next_piece`` returns None but pieces are still IN_PROGRESS we
        wait briefly and retry.  This prevents a worker from exiting and
        consuming its peer just before another peer's in-flight piece fails and
        returns to MISSING — which would leave no worker alive to pick it up.
        """
        while not self._pm.is_complete():
            # If the peer sent no BITFIELD, treat as "can serve any piece"
            # (common for seeders that skip BITFIELD and just answer requests).
            bitfield = conn.bitfield if conn.bitfield else None
            piece_index = self._pm.next_piece(bitfield)
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
                data = await conn.download_piece(
                    piece_index, piece_size, expected_hash,
                    completion_check=self._pm.is_complete_piece,
                )
            except PeerError as exc:
                if "already complete" in str(exc):
                    # End-game: another peer won the race; CANCEL was sent.
                    # Return the piece to MISSING so stats stay consistent,
                    # then continue looking for another piece to download.
                    self._pm.mark_missing(piece_index)
                    continue
                self._pm.mark_missing(piece_index)
                raise  # let _download_from_peer log it and close connection

            # Process any PEX messages that arrived during the piece download
            self._drain_pex(conn, peer_queue)

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

                # Notify the seeder so it can broadcast HAVE to upload peers.
                if self._on_piece_complete is not None:
                    self._on_piece_complete(piece_index)

                if on_progress:
                    await on_progress(self._stats)
