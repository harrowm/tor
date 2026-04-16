# BitTorrent Client — Project Guide

## Project Overview

A BitTorrent client built from scratch in Python, following the phased plan below.
Goal: learn networking, protocol design, and async I/O by implementing the protocol directly.

## Tech Stack

- **Python 3.12+**
- **uv** for package management (not pip)
- **asyncio** throughout — no threading
- **aiohttp** for HTTP tracker announces
- **aiofiles** for async file I/O
- **pytest + pytest-asyncio** for tests

## Commands

```bash
# Install / sync dependencies
uv sync --group dev

# Run tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_bencode.py -v

# Run the client (once main.py is implemented)
uv run bittorrent <torrent_file>
```

## Project Structure

```
bittorrent/
├── __init__.py
├── main.py           # CLI entry point (accepts .torrent file or magnet URI)
├── bencode.py        # Bencoding encoder/decoder
├── torrent.py        # .torrent file parser + info_hash computation
├── tracker.py        # HTTP + UDP tracker announce
├── peer.py           # Single peer connection (asyncio streams, BEP 10 extensions)
├── peer_manager.py   # Connection pool, peer selection, PEX, web seed workers
├── piece_manager.py  # Piece/block state machine, rarest-first selection
├── storage.py        # Pre-allocation, random-access writes, hash verify, resume scan
├── messages.py       # Peer wire protocol encode/decode + PEX helpers
├── metadata.py       # BEP 9 ut_metadata fetch
├── magnet.py         # Magnet URI parsing + metadata resolution (BEP 9/10)
├── dht.py            # BEP 5 DHT peer discovery (Kademlia: bootstrap, get_peers)
├── utp.py            # BEP 29 uTP transport over UDP
└── webseed.py        # BEP 19 HTTP web seed fetching

tests/
├── test_bencode.py
├── test_torrent.py
├── test_tracker.py
├── test_peer.py
├── test_peer_manager.py
├── test_piece_manager.py
├── test_storage.py
├── test_main.py
├── test_magnet.py
├── test_dht.py
├── test_utp.py
└── test_webseed.py
```

## Implementation Phases

### Phase 0 — MVP ✓
- [x] Project skeleton + uv setup
- [x] `bencode.py` — decoder and encoder
- [x] `torrent.py` — parse .torrent file, compute info_hash correctly
- [x] `tracker.py` — HTTP + UDP (BEP 15) announce, parse compact peer list
- [x] `messages.py` — peer wire protocol encode/decode
- [x] `peer.py` — handshake, bitfield, piece download with block pipelining
- [x] `storage.py` — pre-alloc, random-access writes, multi-file spanning
- [x] `piece_manager.py` — state machine, sequential + rarest-first + end-game selection
- [x] `peer_manager.py` — async download orchestration, peer pool, parallel download, end-game
- [x] `main.py` — CLI entry point with Rich progress display

### Phase 1 — Parallel peers + robustness ✓
- [x] Multiple concurrent peer connections (up to 30)
- [x] Disconnection/timeout handling — failed pieces returned to MISSING
- [x] Stall-wait logic — workers wait when all pieces are in-flight
- [x] Re-announce to tracker during long downloads

### Phase 2 — End-game + multi-file ✓
- [x] End-game mode — last few pieces requested from multiple peers simultaneously
- [x] Multi-file torrent support with cross-file piece spanning
- [x] Pre-allocation of all files before download

### Phase 3 — Choking + performance ✓
- [x] Tit-for-tat choke/unchoke (BEP 3) — rechoke every 10s by bytes_sent; optimistic unchoke every 30s
- [x] Upload slots — MAX_UPLOAD_SLOTS=4 enforced in Seeder._rechoke()
- [x] Keep-alive messages — keepalive loop every 90s in both download and upload peers

### Phase 4 — Stretch goals ✓
- [x] UDP tracker protocol (BEP 15)
- [x] DHT peer discovery (BEP 5) — Kademlia bootstrap + iterative get_peers
- [x] Magnet links (BEP 9 + BEP 10 extension protocol)
- [x] Resume/partial download — SHA-1 scan of existing pieces on startup
- [x] Peer Exchange / PEX (BEP 11) — peers share peer lists via ut_pex extension
- [x] uTP transport (BEP 29) — reliable delivery over UDP, TCP fallback
- [x] Web seeds (BEP 19) — HTTP Range fetching from url-list servers
- [x] Seeding / uploading to peers — Seeder with tit-for-tat, HAVE broadcast, upload while downloading
- [x] IPv6 peer support — peers6 compact format in tracker responses
- [x] HTTPS tracker support — aiohttp handles TLS natively in announce()
- [x] Tracker completed/stopped events — _announce_event() called after download/shutdown
- [x] End-game CANCEL messages — CANCEL sent to losing peers in end-game mode
- [x] Signal handling — SIGINT/SIGTERM → clean shutdown + stopped announce
- [x] Fast Extension (BEP 6) — HAVE_ALL/HAVE_NONE/REJECT_REQUEST/ALLOWED_FAST/SUGGEST_PIECE
- [x] Local Service Discovery (BEP 14) — LAN peer discovery via UDP multicast
- [x] DHT IPv6 (BEP 32) — nodes6/values6 compact formats, decode_compact_nodes6/peers6

**749 tests passing.**

## Key BEP References

- **BEP 3** — Core protocol (peer wire, piece hashing)
- **BEP 5** — DHT (Kademlia peer discovery)
- **BEP 6** — Fast Extension (HAVE_ALL, HAVE_NONE, REJECT_REQUEST, ALLOWED_FAST)
- **BEP 9** — Metadata exchange (magnet links)
- **BEP 10** — Extension protocol
- **BEP 11** — Peer Exchange (PEX)
- **BEP 12** — Multitracker (announce-list)
- **BEP 14** — Local Service Discovery (LSD)
- **BEP 15** — UDP tracker protocol
- **BEP 19** — Web seeds (HTTP seeding)
- **BEP 23** — Compact tracker response format
- **BEP 29** — uTP (Micro Transport Protocol)
- **BEP 32** — DHT IPv6 extension

## Critical Technical Notes

- **info_hash**: hash the *bencoded bytes* of the `info` dict only, not the whole .torrent file
- **Blocks vs pieces**: pieces are 256KB–4MB (SHA-1 verified); blocks are always exactly 16,384 bytes (2^14) — you request blocks, assemble into pieces, then verify
- **Compact peer format**: tracker returns 6 bytes per peer (4 bytes IP + 2 bytes port), parse with `struct.unpack`
- **Keep-alives**: send a zero-length message to peers every ~2 minutes or they disconnect
- **asyncio streams**: use `asyncio.open_connection()` not raw sockets

## Test Discipline

- Write tests before or alongside each module
- Do not proceed to the next phase until all tests pass
- Use `uv run pytest -v` to run the full suite
- Legal test torrents: Ubuntu ISO, Big Buck Bunny
