# tor

A BitTorrent client built from scratch in Python. I wrote this to actually understand how BitTorrent works — the protocol, the peer wire format, async I/O, all of it. No libtorrent, no shortcuts.

## What it does

- Downloads torrents from `.torrent` files or magnet links
- Talks to HTTP and UDP trackers (BEP 15)
- Connects to peers, does the handshake, downloads pieces in parallel
- Verifies every piece with SHA-1 before writing to disk
- Resumes partial downloads
- Multi-file torrents with pieces that span file boundaries
- Rarest-first piece selection, end-game mode
- DHT peer discovery (BEP 5) — finds peers without a tracker
- Peer Exchange / PEX (BEP 11) — peers share their peer lists
- uTP transport (BEP 29) — falls back to UDP if TCP is blocked
- Web seeds (BEP 19) — fetches pieces from HTTP servers when peers are slow

## Usage

```bash
uv run bittorrent <torrent_file_or_magnet>
uv run bittorrent ubuntu.torrent --output-dir ~/Downloads
uv run bittorrent "magnet:?xt=urn:btih:..." -o ~/Downloads
```

Options:

```
-o, --output-dir DIR   where to save files (default: current directory)
-p, --port PORT        port to advertise to trackers (default: 6881)
-v, --verbose          debug logging
```

## Setup

Requires Python 3.12+. Uses [uv](https://github.com/astral-sh/uv) for package management.

```bash
uv sync
uv run bittorrent ubuntu.torrent
```

## Running tests

```bash
uv sync --group dev
uv run pytest
uv run pytest tests/test_peer.py -v   # run one file
```

640 tests. Everything is async, everything has tests.

## How it's structured

```
bittorrent/
├── main.py          CLI, progress display (Rich)
├── torrent.py       .torrent file parser, info_hash computation
├── tracker.py       HTTP + UDP tracker announce (BEP 15)
├── peer.py          Single peer connection — handshake, piece download
├── peer_manager.py  Download orchestration, peer pool, PEX, web seeds
├── piece_manager.py Piece state machine, rarest-first, end-game
├── storage.py       Pre-allocation, random-access writes, hash verify
├── messages.py      Peer wire protocol encode/decode
├── dht.py           DHT (BEP 5) — Kademlia bootstrap + get_peers
├── magnet.py        Magnet URI parsing + metadata fetch (BEP 9/10)
├── metadata.py      ut_metadata extension
├── utp.py           uTP protocol over UDP (BEP 29)
└── webseed.py       HTTP web seed fetching (BEP 19)
```

## Some notes

The tricky bits:

**info_hash** — you hash the bencoded bytes of the `info` dict, not the decoded dict. Took me a while to get right.

**Block vs piece** — pieces are what you verify (256 KB–4 MB, SHA-1 hashed). Blocks are what you request from peers (always 16,384 bytes). You request blocks, assemble them into a piece, then verify.

**Piece spanning** — in multi-file torrents, a piece can straddle two files. The storage layer handles this by mapping piece byte ranges across the concatenated virtual file space.

**DHT convergence** — the stopping condition for iterative Kademlia lookups is "stop when the K closest nodes you've discovered are all queried", not "stop when the routing table stops changing". Subtle but important.

**INTERESTED timing** — seeders close idle connections 1–2 seconds after the handshake. You need to send INTERESTED immediately, not wait until you're ready to request a piece.

## License

MIT
