TEAMS = {0: [0, 2], 1: [1, 3]}

def score_hand(tricks_won: dict[int, int], calling_team: int, went_alone: bool) -> dict[int, int]:
    scores = {0: 0, 1: 0}
    calling_team_tricks = sum(tricks_won[p] for p in TEAMS[calling_team])
    other_team = 1 - calling_team

    if calling_team_tricks < 3:
        scores[other_team] += 2  # euchred
    elif calling_team_tricks == 5:
        scores[calling_team] += 4 if went_alone else 2  # march
    else:
        scores[calling_team] += 1  # made it, 3-4 tricks

    return scores