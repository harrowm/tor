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
├── peer_manager.py   # Connection pool and peer selection
├── piece_manager.py  # Piece/block state machine, rarest-first selection
├── storage.py        # Pre-allocation, random-access writes, hash verify
├── messages.py       # Peer wire protocol message encode/decode
├── metadata.py       # BEP 9 ut_metadata fetch
└── magnet.py         # Magnet URI parsing + metadata resolution (BEP 9/10)

tests/
├── test_bencode.py
├── test_torrent.py
├── test_tracker.py
└── test_peer.py
```

## Implementation Phases

### Phase 0 — MVP (current)
Goal: parse a .torrent, announce to tracker, download one piece from one peer, verify SHA-1, write to disk.

- [x] Project skeleton + uv setup
- [x] `bencode.py` — decoder and encoder, fully tested (76/76)
- [x] `torrent.py` — parse .torrent file, compute info_hash correctly (29/29)
- [x] `tracker.py` — HTTP + UDP (BEP 15) announce, parse compact peer list (84/84)
- [x] `messages.py` — peer wire protocol encode/decode (64/64)
- [x] `peer.py` — handshake, bitfield, piece download with block pipelining (35/35)
- [x] `storage.py` — pre-alloc, random-access writes, multi-file spanning (23/23)
- [x] `piece_manager.py` — state machine, sequential + rarest-first + end-game selection (60/60)
- [x] `peer_manager.py` — async download orchestration, peer pool, parallel download, end-game (26/26)
- [x] `main.py` — CLI entry point (14/14)

**482/482 tests passing.**

### Phase 0 complete (MVP)
### Phase 1 complete (parallel peers, disconnection/timeout handling)
### Phase 2 end-game complete; multi-file/pre-alloc done in Phase 0

### Phase 3 — Choking + performance
- Tit-for-tat choke/unchoke
- Upload slots

### Phase 4 — Stretch goals
- [x] UDP tracker protocol (BEP 15)
- DHT (BEP 5)
- [x] Magnet links (BEP 9 + BEP 10 extension protocol)
- Seeding

## Key BEP References

- **BEP 3** — Core protocol (peer wire, piece hashing)
- **BEP 23** — Compact tracker response format
- **BEP 10** — Extension protocol (needed for magnet metadata)
- **BEP 9** — Metadata exchange
- **BEP 5** — DHT

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
