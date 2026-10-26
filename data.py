"""Synthetic, football-realistic match event generator.

No real Premier League data is used or redistributed. Clubs and players are
fictional, and every match is generated deterministically from a seed.
"""
from __future__ import annotations

import random

SQUADS = {
    "Northbridge FC": ["A. Keller", "M. Osei", "J. Duarte", "L. Varga", "T. Okafor",
                       "R. Lindqvist", "S. Moreau", "D. Haas", "K. Adeyemi", "P. Novak", "E. Castillo"],
    "Eastport United": ["B. Santoro", "C. Mbeki", "H. Tanaka", "F. Lemaire", "Y. Petrov",
                        "G. Alvarez", "N. Hassan", "O. Brandt", "V. Rossi", "I. Kowalski", "Z. Mensah"],
    "Riverside Athletic": ["W. Fraser", "U. Demir", "A. Costa", "J. Nakamura", "L. Dubois",
                           "M. Eriksen", "T. Abara", "S. Kovac", "R. Silva", "D. Ortega", "P. Lindahl"],
    "Highvale Rovers": ["C. Murphy", "F. Adeleke", "K. Jansen", "E. Vidal", "B. Moretti",
                        "H. Sato", "N. Popescu", "G. Ferreira", "O. Nilsen", "Q. Bailey", "V. Torres"],
}

EVENT_WEIGHTS = [
    ("pass_into_box", 0.28), ("tackle", 0.22), ("foul", 0.15),
    ("corner", 0.09), ("shot", 0.23), ("card", 0.03),
]


def generate_match(home: str, away: str, seed: int = 7) -> dict:
    """Return a synthetic match: {home, away, events: [...]}."""
    rng = random.Random(seed)
    momentum = 0.0  # -1 (away on top) .. +1 (home on top)
    events: list[dict] = []
    score = {home: 0, away: 0}
    types, weights = zip(*EVENT_WEIGHTS)
    stoppage = rng.randint(2, 5)

    for minute in range(1, 90 + stoppage + 1):
        momentum = max(-1.0, min(1.0, momentum * 0.9 + rng.uniform(-0.35, 0.35)))
        for _ in range(rng.choice([0, 1, 1, 1, 2])):
            p_home = min(0.85, max(0.15, 0.53 + 0.30 * momentum))
            team = home if rng.random() < p_home else away
            etype = rng.choices(types, weights)[0]
            player = rng.choice(SQUADS[team])
            ev = {"minute": minute, "team": team, "player": player, "type": etype,
                  "xg": 0.0, "outcome": ""}
            if etype == "shot":
                ev["xg"] = round(rng.betavariate(1.0, 10) + 0.02, 2)
                roll = rng.random()
                if roll < ev["xg"]:
                    ev["outcome"] = "goal"
                    score[team] += 1
                elif roll < ev["xg"] + 0.30:
                    ev["outcome"] = "saved"
                elif roll < ev["xg"] + 0.45:
                    ev["outcome"] = "blocked"
                else:
                    ev["outcome"] = "off_target"
            elif etype == "card":
                ev["outcome"] = "red" if rng.random() < 0.06 else "yellow"
                if ev["outcome"] == "red":
                    momentum += -0.5 if team == home else 0.5
            ev["score_after"] = f"{score[home]}-{score[away]}"
            events.append(ev)
    return {"home": home, "away": away, "seed": seed, "events": events}
