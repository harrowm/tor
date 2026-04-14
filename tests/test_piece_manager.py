"""
Tests for bittorrent.piece_manager — piece state machine and selection.
"""

import pytest
from bittorrent.piece_manager import PieceManager, PieceState, _peer_has


PIECE_LENGTH  = 512 * 1024    # 512 KB
TOTAL_4_FULL  = PIECE_LENGTH * 4
TOTAL_4_SHORT = PIECE_LENGTH * 3 + 100   # last piece = 100 bytes


def make_pm(num_pieces=4, piece_length=PIECE_LENGTH, total=None) -> PieceManager:
    if total is None:
        total = piece_length * num_pieces
    return PieceManager(num_pieces, piece_length, total)


# ---------------------------------------------------------------------------
# _peer_has helper
# ---------------------------------------------------------------------------

class TestPeerHas:
    def test_first_bit(self):
        assert _peer_has(b"\x80", 0) is True

    def test_second_bit(self):
        assert _peer_has(b"\x40", 1) is True

    def test_bit_not_set(self):
        assert _peer_has(b"\x80", 1) is False

    def test_second_byte(self):
        assert _peer_has(b"\x00\x80", 8) is True

    def test_out_of_range(self):
        assert _peer_has(b"\xff", 100) is False

    def test_all_bits_set(self):
        for i in range(8):
            assert _peer_has(b"\xff", i) is True


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_all_missing_initially(self):
        pm = make_pm(5)
        for i in range(5):
            assert pm.piece_state(i) == PieceState.MISSING

    def test_num_pieces(self):
        pm = make_pm(10)
        assert pm.num_pieces == 10

    def test_zero_pieces_raises(self):
        with pytest.raises(ValueError, match="num_pieces"):
            PieceManager(0, PIECE_LENGTH, 0)

    def test_zero_piece_length_raises(self):
        with pytest.raises(ValueError, match="piece_length"):
            PieceManager(4, 0, PIECE_LENGTH * 4)


# ---------------------------------------------------------------------------
# piece_size
# ---------------------------------------------------------------------------

class TestPieceSize:
    def test_full_piece(self):
        pm = make_pm(4, piece_length=512, total=2048)
        for i in range(3):
            assert pm.piece_size(i) == 512

    def test_last_piece_shorter(self):
        pm = PieceManager(4, 512, 512 * 3 + 100)
        assert pm.piece_size(3) == 100

    def test_last_piece_exactly_full(self):
        pm = PieceManager(4, 512, 512 * 4)
        assert pm.piece_size(3) == 512

    def test_out_of_range_raises(self):
        pm = make_pm(4)
        with pytest.raises(IndexError):
            pm.piece_size(4)


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

class TestStateTransitions:
    def test_mark_in_progress(self):
        pm = make_pm()
        pm.mark_in_progress(0)
        assert pm.piece_state(0) == PieceState.IN_PROGRESS

    def test_mark_complete(self):
        pm = make_pm()
        pm.mark_complete(0)
        assert pm.piece_state(0) == PieceState.COMPLETE

    def test_mark_missing_from_in_progress(self):
        pm = make_pm()
        pm.mark_in_progress(0)
        pm.mark_missing(0)
        assert pm.piece_state(0) == PieceState.MISSING

    def test_mark_missing_does_not_reset_complete(self):
        pm = make_pm()
        pm.mark_complete(0)
        pm.mark_missing(0)   # must be a no-op for COMPLETE pieces
        assert pm.piece_state(0) == PieceState.COMPLETE

    def test_mark_in_progress_on_complete_raises(self):
        pm = make_pm()
        pm.mark_complete(0)
        with pytest.raises(ValueError):
            pm.mark_in_progress(0)

    def test_out_of_range_raises(self):
        pm = make_pm(4)
        with pytest.raises(IndexError):
            pm.mark_complete(99)


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------

class TestCounters:
    def test_initial_all_missing(self):
        pm = make_pm(4)
        assert pm.num_missing == 4
        assert pm.num_complete == 0
        assert pm.num_in_progress == 0

    def test_after_in_progress(self):
        pm = make_pm(4)
        pm.mark_in_progress(0)
        assert pm.num_missing == 3
        assert pm.num_in_progress == 1
        assert pm.num_complete == 0

    def test_after_complete(self):
        pm = make_pm(4)
        pm.mark_complete(0)
        pm.mark_complete(1)
        assert pm.num_complete == 2
        assert pm.num_missing == 2

    def test_is_complete_false(self):
        pm = make_pm(3)
        pm.mark_complete(0)
        assert pm.is_complete() is False

    def test_is_complete_true(self):
        pm = make_pm(3)
        for i in range(3):
            pm.mark_complete(i)
        assert pm.is_complete() is True

    def test_progress(self):
        pm = make_pm(5)
        pm.mark_complete(0)
        pm.mark_complete(2)
        done, total = pm.progress()
        assert done == 2
        assert total == 5


# ---------------------------------------------------------------------------
# Availability / bitfield
# ---------------------------------------------------------------------------

class TestAvailability:
    def test_record_bitfield_all_set(self):
        pm = make_pm(8)
        pm.record_bitfield(b"\xff")
        for i in range(8):
            assert pm.availability(i) == 1

    def test_record_bitfield_none_set(self):
        pm = make_pm(8)
        pm.record_bitfield(b"\x00")
        for i in range(8):
            assert pm.availability(i) == 0

    def test_record_bitfield_partial(self):
        pm = make_pm(8)
        pm.record_bitfield(b"\x80")  # only piece 0
        assert pm.availability(0) == 1
        assert pm.availability(1) == 0

    def test_multiple_peers_accumulate(self):
        pm = make_pm(8)
        pm.record_bitfield(b"\xff")
        pm.record_bitfield(b"\x80")
        assert pm.availability(0) == 2
        assert pm.availability(1) == 1

    def test_record_have(self):
        pm = make_pm(4)
        pm.record_have(2)
        assert pm.availability(2) == 1
        assert pm.availability(0) == 0

    def test_record_have_out_of_range_ignored(self):
        pm = make_pm(4)
        pm.record_have(99)   # must not raise

    def test_bitfield_ignores_extra_bits(self):
        # 3 pieces but 1 full byte (8 bits) — bits 3-7 are padding and must be ignored
        pm = PieceManager(3, 512, 1536)
        pm.record_bitfield(b"\xff")
        for i in range(3):
            assert pm.availability(i) == 1


# ---------------------------------------------------------------------------
# next_piece — sequential
# ---------------------------------------------------------------------------

class TestNextPieceSequential:
    def test_returns_first_missing(self):
        pm = make_pm(4)
        assert pm.next_piece(strategy="sequential") == 0

    def test_skips_in_progress(self):
        pm = make_pm(4)
        pm.mark_in_progress(0)
        assert pm.next_piece(strategy="sequential") == 1

    def test_skips_complete(self):
        pm = make_pm(4)
        pm.mark_complete(0)
        assert pm.next_piece(strategy="sequential") == 1

    def test_returns_none_when_all_complete(self):
        pm = make_pm(3)
        for i in range(3):
            pm.mark_complete(i)
        assert pm.next_piece(strategy="sequential") is None

    def test_returns_none_when_all_in_progress(self):
        pm = make_pm(3)
        for i in range(3):
            pm.mark_in_progress(i)
        assert pm.next_piece(strategy="sequential") is None

    def test_with_peer_bitfield_filters(self):
        pm = make_pm(4)
        # Peer only has piece 2 (bitfield: 0b00100000 = 0x20)
        assert pm.next_piece(b"\x20", strategy="sequential") == 2

    def test_with_peer_bitfield_none_available(self):
        pm = make_pm(4)
        # Peer has nothing
        result = pm.next_piece(b"\x00", strategy="sequential")
        assert result is None

    def test_with_peer_bitfield_none_missing(self):
        pm = make_pm(4)
        pm.mark_complete(0)
        # Peer only has piece 0 which we already have
        result = pm.next_piece(b"\x80", strategy="sequential")
        assert result is None


# ---------------------------------------------------------------------------
# next_piece — rarest first
# ---------------------------------------------------------------------------

class TestNextPieceRarestFirst:
    def test_picks_rarest(self):
        pm = make_pm(4)
        # piece 0: 3 peers, piece 1: 1 peer, piece 2: 2 peers, piece 3: 1 peer
        pm.record_bitfield(b"\xff")   # all 4 pieces, 1st peer
        pm.record_bitfield(b"\xf0")   # pieces 0,1,2,3 = only top 4 bits? No...
        # \xf0 = 1111 0000 → pieces 0,1,2,3 have bits 7,6,5,4 set
        # Let me be explicit: piece 0 = bit 7, piece 1 = bit 6, piece 2 = bit 5, piece 3 = bit 4
        # \x80 = 1000 0000 → piece 0
        # \x40 = 0100 0000 → piece 1
        pm2 = make_pm(4)
        pm2.record_have(0)   # 1 peer has piece 0
        pm2.record_have(0)   # 2 peers have piece 0
        pm2.record_have(1)   # 1 peer has piece 1
        # piece 2, 3: 0 peers
        # rarest = piece 2 or 3 (tied at 0) → picks index 2 first
        result = pm2.next_piece(strategy="rarest_first")
        assert result == 2   # tie broken by index

    def test_tie_broken_by_index(self):
        pm = make_pm(4)
        # All pieces have same (zero) availability — picks lowest index
        assert pm.next_piece(strategy="rarest_first") == 0

    def test_rarest_over_common(self):
        pm = make_pm(4)
        pm.record_have(0)
        pm.record_have(0)
        pm.record_have(0)   # piece 0: 3 peers
        pm.record_have(1)   # piece 1: 1 peer
        # pieces 2,3: 0 peers
        result = pm.next_piece(strategy="rarest_first")
        assert result == 2  # 0 peers, lowest index

    def test_skips_complete(self):
        pm = make_pm(4)
        pm.mark_complete(0)
        pm.mark_complete(1)
        result = pm.next_piece(strategy="rarest_first")
        assert result in (2, 3)

    def test_returns_none_when_all_done(self):
        pm = make_pm(3)
        for i in range(3):
            pm.mark_complete(i)
        assert pm.next_piece(strategy="rarest_first") is None

    def test_with_peer_bitfield(self):
        pm = make_pm(4)
        # Peer only has piece 3 (\x10 = 0001 0000 → piece 3)
        result = pm.next_piece(b"\x10", strategy="rarest_first")
        assert result == 3


# ---------------------------------------------------------------------------
# piece_fractions
# ---------------------------------------------------------------------------

class TestPieceFractions:
    def test_zero_width_returns_empty(self):
        pm = make_pm(4)
        assert pm.piece_fractions(0) == []

    def test_length_equals_width(self):
        pm = make_pm(8)
        result = pm.piece_fractions(10)
        assert len(result) == 10

    def test_all_missing_is_zero(self):
        pm = make_pm(8)
        for f in pm.piece_fractions(8):
            assert f == 0.0

    def test_all_complete_is_one(self):
        pm = make_pm(8)
        for i in range(8):
            pm.mark_complete(i)
        for f in pm.piece_fractions(8):
            assert f == 1.0

    def test_half_complete(self):
        # 4 pieces, first 2 complete, width=2 → each cell covers 2 pieces
        pm = make_pm(4)
        pm.mark_complete(0)
        pm.mark_complete(1)
        fracs = pm.piece_fractions(2)
        assert fracs[0] == 1.0   # pieces 0-1: both complete
        assert fracs[1] == 0.0   # pieces 2-3: both missing

    def test_width_one_partial(self):
        # width=1 → single cell covering all pieces
        pm = make_pm(4)
        pm.mark_complete(0)
        pm.mark_complete(1)
        fracs = pm.piece_fractions(1)
        assert len(fracs) == 1
        assert fracs[0] == pytest.approx(0.5)

    def test_width_equals_num_pieces(self):
        pm = make_pm(4)
        pm.mark_complete(0)
        pm.mark_complete(2)
        fracs = pm.piece_fractions(4)
        assert fracs[0] == 1.0
        assert fracs[1] == 0.0
        assert fracs[2] == 1.0
        assert fracs[3] == 0.0

    def test_values_bounded(self):
        pm = make_pm(10)
        for i in range(5):
            pm.mark_complete(i)
        for f in pm.piece_fractions(20):
            assert 0.0 <= f <= 1.0

    def test_width_larger_than_pieces(self):
        # More cells than pieces — each piece maps to at least one cell
        pm = make_pm(3)
        pm.mark_complete(0)
        fracs = pm.piece_fractions(9)
        assert len(fracs) == 9
        # First 3 cells cover piece 0 (complete), rest cover pieces 1 and 2
        assert fracs[0] == 1.0
