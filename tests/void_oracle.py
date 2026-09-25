"""
Independent reference for void tracking, used only by tests.

A player is void in a suit once they fail to follow it. Suits are EFFECTIVE
suits (a left bower belongs to trump). The trick leader reveals nothing.
"""
from engine.deck import effective_suit


def derive_voids(completed_tricks, current_trick, trump) -> dict[int, set[str]]:
    voids = {p: set() for p in range(4)}
    tricks = list(completed_tricks) + ([current_trick] if current_trick else [])
    for trick in tricks:
        led = effective_suit(trick[0][1], trump)
        for player, card in trick[1:]:
            if effective_suit(card, trump) != led:
                voids[player].add(led)
    return voids
