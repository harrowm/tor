"""
Piece selection and state tracking.

Tracks which pieces are MISSING, IN_PROGRESS, or COMPLETE and decides which
piece to download next. For the MVP we use sequential selection; the rarest-
first optimisation (counting how many peers have each piece) is layered on top.

Piece states:
  MISSING     — not yet started
  IN_PROGRESS — assigned to a peer, download in flight
  COMPLETE    — downloaded and hash-verified

Thread / task safety: PieceManager is designed for single-threaded asyncio use.
If multiple coroutines call next_piece() concurrently, wrap calls in a lock.
"""

from __future__ import annotations

from enum import Enum, auto


class PieceState(Enum):
    MISSING     = auto()
    IN_PROGRESS = auto()
    COMPLETE    = auto()


class PieceManager:
    """Tracks download state for every piece in a torrent.

    Args:
        num_pieces:   Total number of pieces.
        piece_length: Nominal piece size in bytes (last piece may be shorter).
        total_length: Total torrent size in bytes (used to compute last piece size).
    """

    def __init__(
        self,
        num_pieces: int,
        piece_length: int,
        total_length: int,
        *,
        end_game_threshold: int = 20,
    ) -> None:
        if num_pieces <= 0:
            raise ValueError(f"num_pieces must be positive, got {num_pieces}")
        if piece_length <= 0:
            raise ValueError(f"piece_length must be positive, got {piece_length}")

        self._num_pieces         = num_pieces
        self._piece_length       = piece_length
        self._total_length       = total_length
        self._end_game_threshold = end_game_threshold
        self._state: list[PieceState] = [PieceState.MISSING] * num_pieces

        # peer availability: piece_index -> count of peers that have it
        self._availability: list[int] = [0] * num_pieces

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def num_pieces(self) -> int:
        return self._num_pieces

    @property
    def num_complete(self) -> int:
        return sum(1 for s in self._state if s == PieceState.COMPLETE)

    @property
    def num_missing(self) -> int:
        return sum(1 for s in self._state if s == PieceState.MISSING)

    @property
    def num_in_progress(self) -> int:
        return sum(1 for s in self._state if s == PieceState.IN_PROGRESS)

    def is_complete(self) -> bool:
        """True when every piece is COMPLETE."""
        return all(s == PieceState.COMPLETE for s in self._state)

    def is_end_game(self) -> bool:
        """True when so few pieces remain that we request them from multiple peers.

        End-game activates only after at least one piece is complete AND the
        number of remaining (non-COMPLETE) pieces is within the threshold.
        Requiring at least one complete piece prevents end-game from firing at
        the very start of a download on small-piece-count managers.
        """
        if self._end_game_threshold <= 0:
            return False
        complete  = sum(1 for s in self._state if s == PieceState.COMPLETE)
        remaining = self._num_pieces - complete
        return complete > 0 and 0 < remaining <= self._end_game_threshold

    def piece_state(self, piece_index: int) -> PieceState:
        self._check_index(piece_index)
        return self._state[piece_index]

    def is_missing(self, piece_index: int) -> bool:
        return self._state[piece_index] == PieceState.MISSING

    def is_complete_piece(self, piece_index: int) -> bool:
        return self._state[piece_index] == PieceState.COMPLETE

    # ------------------------------------------------------------------
    # Piece length for a specific index (last piece may differ)
    # ------------------------------------------------------------------

    def piece_size(self, piece_index: int) -> int:
        """Return the byte length of this piece (last piece may be shorter)."""
        self._check_index(piece_index)
        if piece_index == self._num_pieces - 1:
            remainder = self._total_length % self._piece_length
            return remainder if remainder != 0 else self._piece_length
        return self._piece_length

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def mark_in_progress(self, piece_index: int) -> None:
        """Mark a piece as assigned to a peer for downloading."""
        self._check_index(piece_index)
        if self._state[piece_index] == PieceState.COMPLETE:
            raise ValueError(f"Piece {piece_index} is already COMPLETE")
        self._state[piece_index] = PieceState.IN_PROGRESS

    def mark_complete(self, piece_index: int) -> None:
        """Mark a piece as successfully downloaded and verified."""
        self._check_index(piece_index)
        self._state[piece_index] = PieceState.COMPLETE

    def mark_missing(self, piece_index: int) -> None:
        """Return a piece to MISSING (e.g. after a peer disconnects or hash fails)."""
        self._check_index(piece_index)
        if self._state[piece_index] != PieceState.COMPLETE:
            self._state[piece_index] = PieceState.MISSING

    # ------------------------------------------------------------------
    # Peer availability
    # ------------------------------------------------------------------

    def record_bitfield(self, bitfield: bytes | bytearray) -> None:
        """Add one peer's bitfield to availability counts.

        Bit ordering: piece 0 is the MSB (0x80) of byte 0,
        piece 1 is 0x40, ..., piece 8 is 0x80 of byte 1, etc.
        Iterating bit_pos 0..7 per byte gives monotonically increasing
        piece indices, so `break` is safe when we exceed num_pieces.
        """
        for byte_idx, byte_val in enumerate(bitfield):
            for bit_pos in range(8):
                piece_idx = byte_idx * 8 + bit_pos
                if piece_idx >= self._num_pieces:
                    break   # safe: piece_idx only increases from here
                if byte_val & (0x80 >> bit_pos):
                    self._availability[piece_idx] += 1

    def record_have(self, piece_index: int) -> None:
        """A peer just announced it has this piece."""
        if 0 <= piece_index < self._num_pieces:
            self._availability[piece_index] += 1

    def availability(self, piece_index: int) -> int:
        """Return how many peers currently have this piece."""
        self._check_index(piece_index)
        return self._availability[piece_index]

    # ------------------------------------------------------------------
    # Piece selection
    # ------------------------------------------------------------------

    def next_piece(
        self,
        peer_bitfield: bytes | bytearray | None = None,
        *,
        strategy: str = "rarest_first",
    ) -> int | None:
        """Return the index of the next piece to download, or None if done.

        Args:
            peer_bitfield: Only consider pieces this peer has (None = any).
            strategy:      "sequential" or "rarest_first".

        A piece is eligible if it is MISSING and (peer_bitfield is None or the
        peer has it).
        """
        in_end_game = self.is_end_game()
        candidates = [
            i for i in range(self._num_pieces)
            if (self._state[i] == PieceState.MISSING or
                (in_end_game and self._state[i] == PieceState.IN_PROGRESS))
            and (peer_bitfield is None or _peer_has(peer_bitfield, i))
        ]

        if not candidates:
            return None

        if strategy == "sequential":
            return candidates[0]

        # rarest_first: prefer pieces fewest peers have, break ties by index
        return min(candidates, key=lambda i: (self._availability[i], i))

    def progress(self) -> tuple[int, int]:
        """Return (num_complete, num_pieces)."""
        return self.num_complete, self._num_pieces

    def piece_fractions(self, width: int) -> list[float]:
        """Return *width* floats in [0.0, 1.0] for a terminal piece map.

        Each cell represents a proportional slice of the piece list.
        0.0 = all missing, 1.0 = all complete.
        """
        if width <= 0:
            return []
        n = self._num_pieces
        result: list[float] = []
        for i in range(width):
            start = (i * n) // width
            end   = ((i + 1) * n) // width
            end   = max(end, start + 1)
            end   = min(end, n)
            total = end - start
            done  = sum(
                1 for j in range(start, end)
                if self._state[j] == PieceState.COMPLETE
            )
            result.append(done / total if total > 0 else 0.0)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_index(self, piece_index: int) -> None:
        if piece_index < 0 or piece_index >= self._num_pieces:
            raise IndexError(
                f"piece_index {piece_index} out of range [0, {self._num_pieces})"
            )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _peer_has(bitfield: bytes | bytearray, piece_index: int) -> bool:
    """Return True if *bitfield* has the bit for *piece_index* set."""
    byte_idx = piece_index // 8
    bit      = 7 - (piece_index % 8)
    if byte_idx >= len(bitfield):
        return False
    return bool(bitfield[byte_idx] & (1 << bit))
