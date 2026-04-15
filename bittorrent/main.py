"""
CLI entry point.

Usage:
    uv run bittorrent <torrent_file> [--output-dir DIR] [--port PORT]

Downloads the torrent to the current directory (or --output-dir) and exits.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Column
from rich.text import Text

from bittorrent.dht import DHTClient
from bittorrent.magnet import MagnetError, parse_magnet, resolve_magnet
from bittorrent.peer_manager import PeerManager
from bittorrent.piece_manager import PieceManager
from bittorrent.storage import Storage
from bittorrent.torrent import ParseError, load
from bittorrent.tracker import TrackerError, announce, generate_peer_id

# Characters used to render the piece map (low → high completion)
_MAP_CHARS = " ░▒▓█"


def _torrent_paths(torrent: "Torrent", output_dir: Path) -> list[Path]:
    """Return the on-disk paths for all files in *torrent*."""
    if torrent.is_multi_file:
        return [
            output_dir / torrent.name / Path(*f.path)
            for f in torrent.files
        ]
    return [output_dir / torrent.name]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bittorrent",
        description="Minimal BitTorrent client",
    )
    parser.add_argument("source", help=".torrent file or magnet URI to download")
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Directory to write downloaded files (default: current dir)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=6881,
        help="Port to advertise to tracker (default: 6881)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def _piece_map(fractions: list[float], width: int) -> Text:
    """Render a colour piece-map Text from a list of [0,1] fractions."""
    text = Text()
    for f in fractions[:width]:
        idx  = int(f * (len(_MAP_CHARS) - 1))
        char = _MAP_CHARS[idx]
        if f == 0.0:
            text.append(char, style="dim")
        elif f == 1.0:
            text.append(char, style="bold green")
        else:
            text.append(char, style="yellow")
    return text


async def _announce_all(
    torrent: "Torrent",
    peer_id: bytes,
    port: int,
    console: Console,
) -> list[tuple[str, int]]:
    """Announce to every tracker in announce + announce-list concurrently.

    Returns a deduplicated list of (ip, port) peers gathered from all
    trackers that responded successfully.  Tracker failures are logged
    but never fatal — we simply skip that tracker.
    """
    # Build a deduplicated ordered list of all tracker URLs.
    seen_urls: set[str] = set()
    trackers: list[str] = []
    for url in [torrent.announce] + [u for tier in torrent.announce_list for u in tier]:
        if url and url not in seen_urls:
            seen_urls.add(url)
            trackers.append(url)

    if not trackers:
        return []

    console.print(f"\nAnnouncing to {len(trackers)} tracker(s)…")

    all_peers: list[tuple[str, int]] = []
    seen_peers: set[tuple[str, int]] = set()

    async def _try(url: str) -> None:
        try:
            resp = await announce(
                url, torrent.info_hash, peer_id, port,
                left=torrent.total_length, event="started",
            )
            # No await between the two mutations — safe in asyncio.
            new = [p for p in resp.peers if tuple(p) not in seen_peers]
            seen_peers.update(tuple(p) for p in new)
            all_peers.extend(new)
            console.print(
                f"  {url}: [bold]{len(resp.peers)}[/bold] peers"
                f" (seeders={resp.complete}, leechers={resp.incomplete})"
            )
        except TrackerError as exc:
            console.print(f"  [dim]{url}: {exc}[/dim]")

    await asyncio.gather(*[_try(url) for url in trackers])
    console.print(f"Total: [bold]{len(all_peers)}[/bold] unique peers")
    return all_peers


async def _dht_peers(
    info_hash: bytes,
    console: Console,
    *,
    timeout: float = 60.0,
) -> list[tuple[str, int]]:
    """Bootstrap DHT and run get_peers for *info_hash* concurrently.

    Returns a (possibly empty) list of (ip, port) peers.  Never raises.
    """
    try:
        async with DHTClient() as dht:
            # Allow up to 20 s for bootstrap; give the bulk of the budget
            # to the iterative get_peers lookup where the real work happens.
            bootstrap_timeout = min(20.0, timeout * 0.3)
            n = await asyncio.wait_for(dht.bootstrap(), timeout=bootstrap_timeout)
            console.print(f"DHT bootstrapped: [bold]{n}[/bold] nodes")
            peers = await dht.get_peers(info_hash, timeout=timeout - bootstrap_timeout)
            console.print(f"DHT peers: [bold]{len(peers)}[/bold]")
            return peers
    except Exception as exc:
        console.print(f"[dim]DHT: {exc}[/dim]")
        return []


async def _run(args: argparse.Namespace, console: Console | None = None) -> int:
    """Main async body. Returns exit code."""
    log = logging.getLogger(__name__)

    if console is None:
        console = Console(stderr=True)

    # --- Generate peer identity (needed for both .torrent and magnet flows) ---
    peer_id = generate_peer_id()
    log.debug("peer_id: %s", peer_id.hex())

    source = args.source

    # --- Parse torrent or resolve magnet ---
    if source.lower().startswith("magnet:"):
        try:
            magnet = parse_magnet(source)
        except MagnetError as exc:
            print(f"Invalid magnet URI: {exc}", file=sys.stderr)
            return 1
        console.print(f"[bold]Magnet:[/bold]    {magnet.name or '(unknown name)'}")
        console.print(f"[bold]Info hash:[/bold] {magnet.info_hash_hex}")
        console.print(f"[bold]Trackers:[/bold]  {len(magnet.trackers)}")
        console.print("\nFetching metadata from peers…")
        try:
            torrent = await resolve_magnet(magnet, peer_id, args.port)
        except MagnetError as exc:
            console.print(f"[red]Magnet error:[/red] {exc}")
            return 1
        console.print(f"[bold]Name:[/bold]      {torrent.name}")
        console.print(f"[bold]Size:[/bold]      {torrent.total_length:,} bytes")
        console.print(f"[bold]Pieces:[/bold]    {torrent.num_pieces} × {torrent.piece_length:,} bytes")
        peers: list[tuple[str, int]] = []   # will re-announce below
    else:
        try:
            torrent = load(source)
        except (ParseError, OSError) as exc:
            print(f"Error reading torrent file: {exc}", file=sys.stderr)
            return 1
        console.print(f"[bold]Name:[/bold]      {torrent.name}")
        console.print(f"[bold]Size:[/bold]      {torrent.total_length:,} bytes")
        console.print(f"[bold]Pieces:[/bold]    {torrent.num_pieces} × {torrent.piece_length:,} bytes")
        all_trackers = len({torrent.announce} | {u for tier in torrent.announce_list for u in tier} - {""})
        console.print(f"[bold]Info hash:[/bold] {torrent.info_hash_hex}")
        console.print(f"[bold]Trackers:[/bold]  {all_trackers}")
        peers = []  # populated below

    # --- Announce to all trackers (BEP 12) concurrently with DHT lookup ---
    tracker_task = asyncio.create_task(
        _announce_all(torrent, peer_id, args.port, console)
    )
    dht_task = asyncio.create_task(
        _dht_peers(torrent.info_hash, console)
    )
    extra_peers, dht_peers = await asyncio.gather(tracker_task, dht_task)
    all_found = list(peers) + list(extra_peers) + list(dht_peers)
    seen: set[tuple] = set()
    peers = []
    for p in all_found:
        pt = tuple(p)
        if pt not in seen:
            seen.add(pt)
            peers.append(pt)

    if not peers:
        console.print("[red]No peers — cannot download.[/red]")
        return 1

    # --- Set up storage and piece tracking ---
    output_dir = Path(args.output_dir).resolve()
    storage    = Storage(torrent, output_dir)
    pm         = PieceManager(
        torrent.num_pieces,
        torrent.piece_length,
        torrent.total_length,
    )

    # --- Resume: scan existing pieces ---
    storage.allocate()   # ensure files exist before scanning
    if any(p.exists() for p in _torrent_paths(torrent, output_dir)):
        console.print("Scanning existing files for resume…")
        done_count = [0]
        def _scan_cb(i: int, total: int) -> None:
            done_count[0] = i + 1
        good_pieces = storage.scan_pieces(progress_cb=_scan_cb)
        if good_pieces:
            for idx in good_pieces:
                pm.mark_complete(idx)
            console.print(
                f"Resuming: [bold]{len(good_pieces)}[/bold]/{torrent.num_pieces} "
                f"pieces already on disk."
            )
        if pm.is_complete():
            console.print("[green]Already downloaded — nothing to do.[/green]")
            return 0
    else:
        # Fresh download — allocate was already called above.
        pass

    # --- Build Rich progress display ---
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, table_column=Column(ratio=1)),
        TextColumn("{task.percentage:>5.1f}%"),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    )
    # Seed the progress bar from pieces already complete (resume case).
    _resume_bytes = min(pm.progress()[0] * torrent.piece_length, torrent.total_length)
    task_id: TaskID = progress.add_task(
        "Downloading",
        total=torrent.total_length,
        completed=_resume_bytes,
    )

    # Mutable state updated by on_progress
    _state: dict = {
        "peers":    0,
        "fracs":    [0.0] * 80,
        "t_start":  time.monotonic(),
        "bytes_done": 0,
    }
    _map_width = 80

    def _make_status() -> Text:
        elapsed = time.monotonic() - _state["t_start"]
        done, total = pm.progress()
        line = Text()
        line.append(f"Pieces: {done}/{total}  ", style="cyan")
        line.append(f"Peers: {_state['peers']}  ", style="magenta")
        line.append(f"Elapsed: {elapsed:.0f}s", style="dim")
        return line

    def _render() -> object:
        from rich.console import Group as RGroup
        fracs    = pm.piece_fractions(_map_width)
        map_text = _piece_map(fracs, _map_width)
        return RGroup(progress, _make_status(), Text(""), map_text)

    async def on_progress(stats) -> None:
        bytes_done = stats.pieces_complete * torrent.piece_length
        # Last piece may be shorter — don't overshoot total
        bytes_done = min(bytes_done, torrent.total_length)
        _state["peers"]     = stats.peers_active
        _state["bytes_done"] = bytes_done
        progress.update(task_id, completed=bytes_done)

    # --- Download ---
    manager = PeerManager(
        torrent, pm, storage,
        info_hash=torrent.info_hash,
        peer_id=peer_id,
        use_utp=True,   # BEP 29: fall back to uTP when TCP is blocked
    )

    console.print()
    try:
        with Live(_render(), console=console, refresh_per_second=4,
                  transient=False) as live:
            async def _tick() -> None:
                while True:
                    live.update(_render())
                    await asyncio.sleep(0.25)

            tick_task = asyncio.create_task(_tick())
            try:
                await manager.run(peers, on_progress=on_progress, allocate=False)
            finally:
                tick_task.cancel()
                try:
                    await tick_task
                except asyncio.CancelledError:
                    pass
                # Final render — show 100 %
                progress.update(task_id, completed=torrent.total_length)
                live.update(_render())
    except RuntimeError as exc:
        console.print(f"\n[red]Download failed:[/red] {exc}")
        return 1

    console.print(f"\n[bold green]Done![/bold green] Saved to: {output_dir / torrent.name}")
    return 0


def main() -> None:
    args    = _parse_args()
    level   = logging.DEBUG if args.verbose else logging.WARNING
    console = Console(stderr=True)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False)],
    )
    sys.exit(asyncio.run(_run(args, console=console)))


if __name__ == "__main__":
    main()
