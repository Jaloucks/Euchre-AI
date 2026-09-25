import random
from dataclasses import dataclass, field

from engine.deck import Card, create_deck, effective_suit, deal_hands
from engine.rules import legal_plays, trick_winner
from engine.bidding import run_bidding, resolve_discard, BidResult
from engine.scoring import score_hand


@dataclass
class HandContext:
    """Fixed facts about the hand — set once after bidding, never mutated."""
    trump: str
    up_card: Card
    caller: int              # player who called trump
    went_alone: bool
    active_players: list[int]
    dealer: int


@dataclass
class PlayerView:
    """
    Exactly what one player legitimately knows at their moment of decision.
    This is the observation space for the play-net — nothing here is hidden
    information, and nothing hidden gets in.

    NOTE: completed_tricks gives perfect recall of every card played this
    hand. That is public information, but it is a deliberate choice: the
    agent will track cards better than a typical human.
    """
    player_id: int
    hand: list[Card]                                # own cards only
    legal_cards: list[Card]
    trump: str
    up_card: Card
    caller: int
    went_alone: bool
    active_players: list[int]
    dealer: int
    led_suit: str | None                            # None if leading
    cards_played_this_trick: list[tuple[int, Card]]
    completed_tricks: list[list[tuple[int, Card]]]  # public history
    tricks_won: dict[int, int]
    void_suits: dict[int, set[str]] = field(default_factory=dict)
    # player_id -> suits that player has shown out of (failed to follow).
    # Real suit names, effective suits (a left-bower lead counts as trump).
    # Public information: derived only from cards everyone has seen.


@dataclass
class HandResult:
    """Everything that happened in one hand — scores plus trajectory for RL."""
    scores: dict[int, int]                          # team_id -> points
    tricks_won: dict[int, int]                      # player_id -> tricks
    trump: str
    caller: int
    went_alone: bool
    up_card: Card
    dealer: int
    completed_tricks: list[list[tuple[int, Card]]]
    bid_result: BidResult
    discarded: Card | None = None                   # only if dealer picked up


@dataclass
class GameResult:
    """A full game played to `target` points."""
    winning_team: int
    final_scores: dict[int, int]
    hands: list[HandResult]


def run_trick(
    hands: dict[int, list[Card]],
    leader: int,
    ctx: HandContext,
    completed_tricks: list[list[tuple[int, Card]]],
    tricks_won: dict[int, int],
    void_suits: dict[int, set[str]],
    choose_play_fn,
) -> tuple[int, list[tuple[int, Card]]]:
    """
    Play one trick. MUTATES `hands` (played cards are removed) and
    `void_suits` (a player who fails to follow the led suit is marked void in it).
    Returns (winning_player, cards_played).
    """
    order = sorted(ctx.active_players, key=lambda p: (p - leader) % 4)
    cards_played: list[tuple[int, Card]] = []

    for player_id in order:
        hand = hands[player_id]
        legal_cards = legal_plays(hand, cards_played, ctx.trump)

        led_suit = (
            effective_suit(cards_played[0][1], ctx.trump) if cards_played else None
        )

        view = PlayerView(
            player_id=player_id,
            hand=list(hand),                    # copy — callback can't mutate ours
            legal_cards=list(legal_cards),
            trump=ctx.trump,
            up_card=ctx.up_card,
            caller=ctx.caller,
            went_alone=ctx.went_alone,
            active_players=list(ctx.active_players),
            dealer=ctx.dealer,
            led_suit=led_suit,
            cards_played_this_trick=list(cards_played),
            completed_tricks=[list(t) for t in completed_tricks],
            tricks_won=dict(tricks_won),
            void_suits={p: set(suits) for p, suits in void_suits.items()},  # copy
        )

        chosen_card = choose_play_fn(view)

        if chosen_card not in legal_cards:
            raise ValueError(
                f"Player {player_id} chose an illegal card: {chosen_card} "
                f"(legal: {legal_cards})"
            )

        hands[player_id].remove(chosen_card)
        cards_played.append((player_id, chosen_card))
        # The leader (led_suit is None) can play anything, so leading reveals nothing.
        if led_suit is not None and effective_suit(chosen_card, ctx.trump) != led_suit:
            void_suits[player_id].add(led_suit)

    winner = trick_winner(cards_played, ctx.trump)
    return winner, cards_played


def run_tricks(
    hands: dict[int, list[Card]],
    ctx: HandContext,
    choose_play_fn,
) -> tuple[dict[int, int], list[list[tuple[int, Card]]]]:
    """
    Play all 5 tricks. MUTATES `hands` — played cards are removed.
    Returns (tricks_won by player_id, completed_tricks).
    """
    completed_tricks: list[list[tuple[int, Card]]] = []
    tricks_won: dict[int, int] = {p: 0 for p in range(4)}
    void_suits: dict[int, set[str]] = {p: set() for p in range(4)}

    leader = (ctx.dealer + 1) % 4
    # If the leader is sitting out (partner of an alone caller), advance
    # to the next active player in clockwise order.
    while leader not in ctx.active_players:
        leader = (leader + 1) % 4

    for _ in range(5):
        winner, cards_played = run_trick(
            hands, leader, ctx, completed_tricks, tricks_won, void_suits, choose_play_fn
        )
        completed_tricks.append(cards_played)
        tricks_won[winner] += 1
        leader = winner

    return tricks_won, completed_tricks


def play_hand(
    dealer: int,
    rng: random.Random,
    choose_bid_fn,
    choose_discard_fn,
    choose_play_fn,
) -> HandResult:
    """Deal, bid, discard if needed, play 5 tricks, score. One complete hand."""

    # 1. Deal
    hand_list, up_card, kitty = deal_hands(create_deck(), rng)
    hands = {p: list(hand_list[p]) for p in range(4)}

    # 2. Bid
    bid_result = run_bidding(
        dealer=dealer,
        up_card=up_card,
        hands=hands,
        choose_action_fn=choose_bid_fn,
    )
    caller = bid_result.winning_player
    trump = bid_result.winning_bid.suit
    went_alone = bid_result.winning_bid.alone

    # 3. Dealer picks up the up-card only if it was ordered up in round 1
    discarded = None
    if bid_result.winning_bid.kind == "order_up" and not ((caller + 2) % 4 == dealer and went_alone):
        hands[dealer], discarded = resolve_discard(
            hands[dealer], up_card, dealer, caller, went_alone, choose_discard_fn
        )

    # 4. Determine who actually plays
    if went_alone:
        partner = (caller + 2) % 4
        active_players = [p for p in range(4) if p != partner]
    else:
        active_players = list(range(4))

    ctx = HandContext(
        trump=trump,
        up_card=up_card,
        caller=caller,
        went_alone=went_alone,
        active_players=active_players,
        dealer=dealer,
    )

    # 5. Play the tricks
    tricks_won, completed_tricks = run_tricks(hands, ctx, choose_play_fn)

    # 6. Score
    scores = score_hand(
        tricks_won=tricks_won,
        calling_team=caller % 2,
        went_alone=went_alone,
    )

    return HandResult(
        scores=scores,
        tricks_won=tricks_won,
        trump=trump,
        caller=caller,
        went_alone=went_alone,
        up_card=up_card,
        dealer=dealer,
        completed_tricks=completed_tricks,
        bid_result=bid_result,
        discarded=discarded,
    )


def play_game(
    rng: random.Random,
    choose_bid_fn,
    choose_discard_fn,
    choose_play_fn,
    target: int = 10,
    first_dealer: int = 0,
    max_hands: int = 200,
) -> GameResult:
    """
    Play hands until a team reaches `target` points. Dealer rotates clockwise.
    `max_hands` is a safety valve — a correct engine should never hit it,
    since every hand awards at least 1 point to somebody.
    """
    scores = {0: 0, 1: 0}
    dealer = first_dealer
    hand_results: list[HandResult] = []

    while max(scores.values()) < target:
        if len(hand_results) >= max_hands:
            raise RuntimeError(
                f"Game exceeded {max_hands} hands without reaching {target} "
                f"points (scores={scores}) — likely a scoring bug."
            )

        result = play_hand(
            dealer, rng, choose_bid_fn, choose_discard_fn, choose_play_fn
        )
        for team, pts in result.scores.items():
            scores[team] += pts
        hand_results.append(result)
        dealer = (dealer + 1) % 4

    winning_team = 0 if scores[0] >= target else 1
    return GameResult(
        winning_team=winning_team,
        final_scores=scores,
        hands=hand_results,
    )