"""
Tests for bots/heuristic_bot.py (Phase 2a).

The heuristic bot is the FROZEN external yardstick for self-play. Every
constant and rule below is pinned on purpose: if the baseline drifts, old
win rates stop being comparable to new ones. To try a stronger heuristic,
write a new class (HeuristicBotV2) and leave this one alone.

Spec summary (see the Phase 2 spec for the reasoning):

    Constants
        RIGHT_POINTS = 3.0, LEFT_POINTS = 2.5
        TRUMP_RANK_POINTS = {"Ace": 2.0, "King": 1.5, "Queen": 1.0, "10": 1.0, "9": 1.0}
        OFF_ACE_POINTS = 1.0
        UP_CARD_WEIGHT = 0.5
        CALL_THRESHOLD = 5.0, ALONE_THRESHOLD = 8.5

    Pure helpers
        card_points(card, trump) -> float
        hand_strength(hand, trump) -> float
        strength_key(card, trump) -> tuple        total order used for "lowest"/"highest"
        lowest_card(cards, trump) -> Card
        highest_card(cards, trump) -> Card
        pick_discard(hand, trump) -> Card         hand = the dealer's six cards
        round1_strength(view: BidView) -> float

    class HeuristicBot(call_threshold=CALL_THRESHOLD, alone_threshold=ALONE_THRESHOLD)
        choose_bid(view, legal_actions) -> BidAction   (one of legal_actions)
        choose_discard(view) -> Card
        choose_play(view) -> Card                      (one of view.legal_cards)
        as_callbacks() -> (choose_bid, choose_discard, choose_play)

Fully deterministic: no rng, and no dependence on the ORDER of view.hand or
view.legal_cards.
"""
import random

import pytest

hb = pytest.importorskip("bots.heuristic_bot", reason="bots/heuristic_bot.py not written yet")

from engine.bidding import BidAction, BidView, DiscardView
from engine.deck import effective_suit
from engine.game import PlayerView, play_hand
from engine.rules import legal_plays
from card_notation import c, cards      # tests/ is on sys.path under pytest
from void_oracle import derive_voids

HeuristicBot = hb.HeuristicBot


# ---------- helpers ----------

def round1_legal(up_suit):
    return [
        BidAction(kind="pass"),
        BidAction(kind="order_up", suit=up_suit, alone=False),
        BidAction(kind="order_up", suit=up_suit, alone=True),
    ]


def round2_legal(up_suit, stuck=False):
    acts = [] if stuck else [BidAction(kind="pass")]
    for s in ["Hearts", "Diamonds", "Clubs", "Spades"]:
        if s != up_suit:
            acts.append(BidAction(kind="call_suit", suit=s, alone=False))
            acts.append(BidAction(kind="call_suit", suit=s, alone=True))
    return acts


def bid_view(hand, up, player_id=0, dealer=3, rnd=1, passes=0):
    return BidView(player_id=player_id, hand=cards(hand), up_card=c(up),
                   dealer=dealer, round=rnd, passes_this_round=passes)


def play_view(hand, played=(), trump="Hearts", player_id=0, caller=1,
              went_alone=False, dealer=3, up="9H"):
    """A PlayerView built the way the engine would build it."""
    hand = cards(hand)
    played = [(pid, c(code)) for pid, code in played]
    active = [0, 1, 2, 3]
    if went_alone:
        active.remove((caller + 2) % 4)
    return PlayerView(
        player_id=player_id,
        hand=hand,
        legal_cards=legal_plays(hand, played, trump),
        trump=trump,
        up_card=c(up),
        caller=caller,
        went_alone=went_alone,
        active_players=active,
        dealer=dealer,
        led_suit=effective_suit(played[0][1], trump) if played else None,
        cards_played_this_trick=played,
        completed_tricks=[],
        tricks_won={p: 0 for p in range(4)},
        void_suits=derive_voids([], played, trump),
    )


# ---------- frozen constants ----------

def test_constants_are_frozen():
    assert hb.RIGHT_POINTS == 3.0
    assert hb.LEFT_POINTS == 2.5
    assert hb.TRUMP_RANK_POINTS == {"Ace": 2.0, "King": 1.5, "Queen": 1.0, "10": 1.0, "9": 1.0}
    assert hb.OFF_ACE_POINTS == 1.0
    assert hb.UP_CARD_WEIGHT == 0.5
    assert hb.CALL_THRESHOLD == 5.0
    assert hb.ALONE_THRESHOLD == 8.5


def test_default_thresholds_come_from_constants():
    bot = HeuristicBot()
    assert bot.call_threshold == hb.CALL_THRESHOLD
    assert bot.alone_threshold == hb.ALONE_THRESHOLD


# ---------- card_points / hand_strength ----------

@pytest.mark.parametrize("code, trump, expected", [
    ("JH", "Hearts", 3.0),     # right bower
    ("JD", "Hearts", 2.5),     # left bower (same color)
    ("AH", "Hearts", 2.0),
    ("KH", "Hearts", 1.5),
    ("QH", "Hearts", 1.0),
    ("10H", "Hearts", 1.0),
    ("9H", "Hearts", 1.0),
    ("AS", "Hearts", 1.0),     # off-suit ace
    ("AD", "Hearts", 1.0),     # next-suit ace is still an off-suit ace
    ("KS", "Hearts", 0.0),
    ("9D", "Hearts", 0.0),
    ("JC", "Hearts", 0.0),     # jack of a cross suit is just a jack
    ("JS", "Spades", 3.0),
    ("JC", "Spades", 2.5),
    ("JH", "Spades", 0.0),
])
def test_card_points(code, trump, expected):
    assert hb.card_points(c(code), trump) == expected


def test_hand_strength_sums_card_points():
    assert hb.hand_strength(cards("JH JD AH AS 9C"), "Hearts") == 8.5


def test_hand_strength_depends_on_trump():
    # Same cards, Diamonds trump: JD right 3, JH left 2.5, AH + AS off aces 1 each.
    assert hb.hand_strength(cards("JH JD AH AS 9C"), "Diamonds") == 7.5


def test_hand_strength_of_junk_is_zero():
    assert hb.hand_strength(cards("9C 10C 9S 10S QD"), "Hearts") == 0.0


# ---------- lowest_card / highest_card ----------

def test_lowest_prefers_off_suit_over_trump():
    assert hb.lowest_card(cards("9H AS"), "Hearts") == c("AS")


def test_left_bower_counts_as_trump_for_lowest():
    assert hb.lowest_card(cards("JD AS"), "Hearts") == c("AS")


def test_lowest_within_a_suit_is_by_rank():
    assert hb.lowest_card(cards("KS 10S QS"), "Hearts") == c("10S")


def test_highest_trump_is_right_bower():
    assert hb.highest_card(cards("AH JD JH"), "Hearts") == c("JH")


def test_any_trump_beats_off_suit_for_highest():
    assert hb.highest_card(cards("AS 9H"), "Hearts") == c("9H")


def test_lowest_tie_goes_to_higher_canonical_index():
    # Trump Hearts: canonical suits [Hearts, Diamonds, Clubs, Spades].
    # 9C is index 17, 9S is index 23 -> the tie goes to 9S.
    assert hb.lowest_card(cards("9C 9S"), "Hearts") == c("9S")
    assert hb.lowest_card(cards("9S 9C"), "Hearts") == c("9S")


def test_highest_tie_goes_to_lower_canonical_index():
    # AD (next, index 7) beats AC (cross1, index 12) on the tie-break.
    assert hb.highest_card(cards("AC AD"), "Hearts") == c("AD")
    assert hb.highest_card(cards("AD AC"), "Hearts") == c("AD")


# ---------- pick_discard ----------

def test_discard_prefers_singleton_to_create_a_void():
    # KS is the only spade: dropping it voids spades, even though 9C is lower.
    assert hb.pick_discard(cards("JH AH KH KS 9C 10C"), "Hearts") == c("KS")


def test_discard_lowest_off_card_when_no_singleton():
    assert hb.pick_discard(cards("JH AH 9C 10C 10S QS"), "Hearts") == c("9C")


def test_discard_keeps_off_aces():
    # AS is a singleton, but aces are never discarded while a non-ace off card exists.
    assert hb.pick_discard(cards("JH AH KH AS 9C 10C"), "Hearts") == c("9C")


def test_discard_an_ace_only_when_every_off_card_is_an_ace():
    assert hb.pick_discard(cards("JH AH KH QH AS AC"), "Hearts") in (c("AS"), c("AC"))


def test_discard_lowest_trump_when_all_trump():
    assert hb.pick_discard(cards("JH JD AH KH 10H 9H"), "Hearts") == c("9H")


def test_discard_counts_suits_by_effective_suit():
    # JD is trump (left bower), so 9D is the only diamond -> singleton.
    # If JD were counted as a diamond, QS would be the only singleton instead.
    assert hb.pick_discard(cards("JH JD AH KH 9D QS"), "Hearts") == c("9D")


# ---------- round1_strength ----------

def test_round1_strength_dealer_uses_best_five_of_six():
    # Six = JH AH KH 9C 10C 9S; the discard rule drops 9S (singleton).
    v = bid_view("JH AH 9C 10C 9S", up="KH", player_id=3, dealer=3)
    assert hb.round1_strength(v) == 3.0 + 2.0 + 1.5


def test_round1_strength_partner_of_dealer_gains_half_the_up_card():
    v = bid_view("AH KH AS 9C 10D", up="JH", player_id=1, dealer=3)
    assert hb.round1_strength(v) == 4.5 + 0.5 * 3.0


@pytest.mark.parametrize("player_id", [0, 2])
def test_round1_strength_opponent_of_dealer_loses_half_the_up_card(player_id):
    v = bid_view("AH KH AS 9C 10D", up="JH", player_id=player_id, dealer=3)
    assert hb.round1_strength(v) == 4.5 - 0.5 * 3.0


# ---------- choose_bid ----------

def test_round1_strong_hand_orders_up():
    v = bid_view("JH JD AH 9C 9S", up="9H", player_id=0, dealer=3)   # 7.5 - 0.5 = 7.0
    assert HeuristicBot().choose_bid(v, round1_legal("Hearts")) == \
        BidAction(kind="order_up", suit="Hearts", alone=False)


def test_round1_very_strong_hand_goes_alone():
    v = bid_view("JH JD AH KH AS", up="9H", player_id=0, dealer=3)   # 10.0 - 0.5 = 9.5
    assert HeuristicBot().choose_bid(v, round1_legal("Hearts")) == \
        BidAction(kind="order_up", suit="Hearts", alone=True)


def test_round1_weak_hand_passes():
    v = bid_view("9C 10C 9S 10S QD", up="9H", player_id=0, dealer=3)
    assert HeuristicBot().choose_bid(v, round1_legal("Hearts")) == BidAction(kind="pass")


def test_round1_threshold_is_inclusive_and_up_card_direction_matters():
    # Base 4.5, up card 9H is worth 1.0.
    # Partner of dealer: 4.5 + 0.5 = 5.0 -> exactly the threshold -> order up.
    # Opponent of dealer: 4.5 - 0.5 = 4.0 -> pass.
    partner = bid_view("AH KH AS 9C 10D", up="9H", player_id=1, dealer=3)
    opponent = bid_view("AH KH AS 9C 10D", up="9H", player_id=0, dealer=3)
    bot = HeuristicBot()
    assert bot.choose_bid(partner, round1_legal("Hearts")).kind == "order_up"
    assert bot.choose_bid(opponent, round1_legal("Hearts")).kind == "pass"


def test_round2_calls_best_suit_never_the_up_suit():
    # Spades: JS 3 + AS 2 + KS 1.5 + AH 1 = 7.5. Hearts is the turned-down suit.
    v = bid_view("JS AS KS AH 9D", up="9H", player_id=0, dealer=3, rnd=2)
    assert HeuristicBot().choose_bid(v, round2_legal("Hearts")) == \
        BidAction(kind="call_suit", suit="Spades", alone=False)


def test_round2_very_strong_hand_goes_alone():
    # Spades: JS 3 + JC 2.5 + AS 2 + AH 1 = 8.5 -> exactly the alone threshold.
    v = bid_view("JS JC AS AH 9D", up="9H", player_id=0, dealer=3, rnd=2)
    assert HeuristicBot().choose_bid(v, round2_legal("Hearts")) == \
        BidAction(kind="call_suit", suit="Spades", alone=True)


def test_round2_weak_hand_passes():
    v = bid_view("9C 10C 9S 10D QD", up="9H", player_id=0, dealer=3, rnd=2)
    assert HeuristicBot().choose_bid(v, round2_legal("Hearts")) == BidAction(kind="pass")


def test_stuck_dealer_calls_best_suit_with_suits_order_tie_break():
    # Diamonds 2.0 and Clubs 2.0 tie; Diamonds comes first in SUITS.
    v = bid_view("9C 10C 9S 10D QD", up="9H", player_id=3, dealer=3, rnd=2, passes=3)
    assert HeuristicBot().choose_bid(v, round2_legal("Hearts", stuck=True)) == \
        BidAction(kind="call_suit", suit="Diamonds", alone=False)


def test_custom_thresholds_are_used():
    v = bid_view("JH JD AH 9C 9S", up="9H", player_id=0, dealer=3)   # 7.0
    legal = round1_legal("Hearts")
    assert HeuristicBot(call_threshold=7.5).choose_bid(v, legal).kind == "pass"
    assert HeuristicBot(alone_threshold=7.0).choose_bid(v, legal).alone is True


# ---------- choose_discard ----------

def discard_view(hand, up, caller=1, went_alone=False):
    return DiscardView(player_id=3, hand=cards(hand), up_card=c(up),
                       trump=c(up).suit, caller=caller, went_alone=went_alone)


def test_choose_discard_uses_pick_discard():
    v = discard_view("JH AH KH KS 9C 10C", up="KH")
    assert HeuristicBot().choose_discard(v) == c("KS")


def test_up_card_can_be_the_discard():
    # Trump Clubs, all six cards are trump (JS is the left bower).
    # The up card 9C is the lowest trump, so it goes back.
    v = discard_view("JC JS AC KC QC 9C", up="9C")
    assert HeuristicBot().choose_discard(v) == c("9C")


# ---------- choose_play: leading ----------

def test_single_legal_card_is_played():
    v = play_view("KS", played=[(3, "AS")])
    assert HeuristicBot().choose_play(v) == c("KS")


def test_caller_with_two_trump_leads_highest_trump():
    v = play_view("JD AH 9C AS KS", caller=0)
    assert HeuristicBot().choose_play(v) == c("JD")


def test_caller_with_one_trump_leads_off_ace():
    v = play_view("JD 9C AS KS 10C", caller=0)
    assert HeuristicBot().choose_play(v) == c("AS")


def test_non_caller_leads_off_ace_before_trump():
    v = play_view("JH AH AS 9C 10C", caller=1)
    assert HeuristicBot().choose_play(v) == c("AS")


def test_no_ace_leads_lowest_off_card():
    v = play_view("JH KS 9C 10C", caller=1)
    assert HeuristicBot().choose_play(v) == c("9C")


def test_only_trump_leads_highest_trump():
    v = play_view("AH 9H KH", caller=1)
    assert HeuristicBot().choose_play(v) == c("AH")


def test_only_trump_counts_the_left_bower_as_trump():
    # JD is a Diamond by card.suit but trump by effective suit, so this hand
    # is all trump and leads its highest trump (the left bower).
    v = play_view("JD AH KH", caller=1)
    assert HeuristicBot().choose_play(v) == c("JD")


# ---------- choose_play: following ----------

def test_partner_winning_play_lowest():
    # Partner (2) led KS and is winning; don't overtake with AS.
    v = play_view("AS 9S JH", played=[(2, "KS"), (3, "10S")])
    assert HeuristicBot().choose_play(v) == c("9S")


def test_opponent_winning_win_with_cheapest_card():
    # Opponent 1 is winning with QS; both AS and KS win -> KS.
    v = play_view("AS KS 9C", played=[(1, "QS"), (2, "9S"), (3, "10S")])
    assert HeuristicBot().choose_play(v) == c("KS")


def test_void_in_led_suit_trumps_in_with_lowest_trump():
    v = play_view("9H JH 9C", played=[(1, "AS")])
    assert HeuristicBot().choose_play(v) == c("9H")


def test_overtrump_with_cheapest_winning_trump():
    # Opponent 3 trumped with 10H. 9H can't beat it; QH can.
    v = play_view("QH 9H 9C", played=[(1, "AS"), (2, "9S"), (3, "10H")])
    assert HeuristicBot().choose_play(v) == c("QH")


def test_cannot_win_throws_lowest_and_keeps_trump():
    # Opponent 3 trumped with KH; QH can't beat it. Throw 9C, keep QH.
    v = play_view("QH 9C KD", played=[(1, "AS"), (2, "9S"), (3, "KH")])
    assert HeuristicBot().choose_play(v) == c("9C")


def test_left_bower_lead_means_trump_was_led():
    # Opponent led JD (left bower, trump). Must follow with trump; AH can't
    # beat the left bower, so play the lowest trump.
    v = play_view("9H AH 9D", played=[(1, "JD")])
    assert HeuristicBot().choose_play(v) == c("9H")


def test_alone_caller_partner_absent_still_plays_legally():
    # Player 0 went alone; partner 2 is out. Opponent 1 led QS.
    v = play_view("AS KS JH", played=[(1, "QS")], caller=0, went_alone=True)
    assert HeuristicBot().choose_play(v) == c("KS")


# ---------- properties over real games ----------

class Recording:
    """Wraps a bot, records every (kind, view, legal, choice) it makes."""
    def __init__(self, bot):
        self.bot = bot
        self.log = []

    def bid(self, view, legal):
        a = self.bot.choose_bid(view, legal)
        self.log.append(("bid", view, legal, a))
        return a

    def discard(self, view):
        d = self.bot.choose_discard(view)
        self.log.append(("discard", view, None, d))
        return d

    def play(self, view):
        p = self.bot.choose_play(view)
        self.log.append(("play", view, None, p))
        return p


@pytest.fixture(scope="module")
def recorded():
    rec = Recording(HeuristicBot())
    rng = random.Random(2026)
    for h in range(400):
        play_hand(h % 4, rng, rec.bid, rec.discard, rec.play)   # engine raises on illegal plays/discards
    return rec.log


def test_bids_are_always_legal(recorded):
    bids = [(legal, a) for kind, _, legal, a in recorded if kind == "bid"]
    assert len(bids) > 400
    for legal, a in bids:
        assert a in legal


def test_all_decision_types_were_exercised(recorded):
    kinds = {k for k, *_ in recorded}
    assert kinds == {"bid", "discard", "play"}
    assert any(k == "bid" and a.alone for k, _, _, a in recorded), "no alone calls in sample"


def _shuffled(view, rng):
    import copy
    v = copy.deepcopy(view)
    rng.shuffle(v.hand)
    if hasattr(v, "legal_cards"):
        rng.shuffle(v.legal_cards)
    return v


def test_decisions_do_not_depend_on_card_order(recorded):
    rng = random.Random(0)
    bot = HeuristicBot()
    for kind, view, legal, choice in recorded:
        v = _shuffled(view, rng)
        if kind == "bid":
            legal = list(legal)
            rng.shuffle(legal)
            assert bot.choose_bid(v, legal) == choice
        elif kind == "discard":
            assert bot.choose_discard(v) == choice
        else:
            assert bot.choose_play(v) == choice


def test_as_callbacks_order():
    bot = HeuristicBot()
    bid, discard, play = bot.as_callbacks()
    assert (bid, discard, play) == (bot.choose_bid, bot.choose_discard, bot.choose_play)
