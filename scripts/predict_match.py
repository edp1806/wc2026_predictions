# -*- coding: utf-8 -*-
"""Prédit un match précis à partir de l'état sauvegardé.

Usage :
    python -m scripts.predict_match France Brazil
    python -m scripts.predict_match FRA BRA
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, elo, state
from src.simulate import match_probabilities, simulate_match


def find_team(query: str) -> str | None:
    q = query.strip().lower()
    for _, name, code, _ in config.WC2026_TEAMS:
        if name.lower() == q or code.lower() == q:
            return name
    for _, name, _, _ in config.WC2026_TEAMS:
        if q in name.lower():
            return name
    return None


def main():
    parser = argparse.ArgumentParser(description="Prédit un match WC2026 (nom d'équipe ou code ISO).")
    parser.add_argument("team_a")
    parser.add_argument("team_b")
    parser.add_argument(
        "--draws", type=int, default=0,
        help="Tire N scores simulés (Poisson) en plus des probabilités, ex. --draws 5",
    )
    args = parser.parse_args()

    tracker = state.load_state()
    if tracker is None:
        print("Aucun état sauvegardé. Lance d'abord : python -m scripts.run_pipeline")
        return

    a, b = find_team(args.team_a), find_team(args.team_b)
    if a is None or b is None:
        print(f"Équipe non trouvée : {args.team_a if a is None else args.team_b}")
        return

    elo_a, elo_b = tracker.ratings.get(a, config.ELO_BASE), tracker.ratings.get(b, config.ELO_BASE)
    form_a, form_b = elo.get_form(tracker.form, a), elo.get_form(tracker.form, b)
    h2h_edge = elo.get_h2h_edge(tracker.h2h, a, b)
    r = match_probabilities(tracker.model, tracker.ratings, a, b, form=tracker.form, h2h=tracker.h2h, competition_weight=1.0)

    h2h_matches = len(tracker.h2h.get(elo.h2h_key(a, b), []))
    h2h_note = f", h2h {h2h_edge:+.2f} ({h2h_matches} confront.)" if h2h_matches > 0 else ""
    print(f"\n{a} (Elo {round(elo_a)}, forme {form_a:.0%})  vs  {b} (Elo {round(elo_b)}, forme {form_b:.0%}){h2h_note}")
    print(f"  {a} gagne   : {r['p_win_a']:.1%}")
    print(f"  Match nul  : {r['p_draw']:.1%}")
    print(f"  {b} gagne   : {r['p_win_b']:.1%}")
    print(f"  xG         : {a} {r['xg_a']} – {r['xg_b']} {b}")
    print(f"  Score le plus probable : {r['most_likely_score']}")

    if args.draws > 0:
        print(f"\n{args.draws} score(s) simulé(s) :")
        for _ in range(args.draws):
            ga, gb = simulate_match(tracker.model, tracker.ratings, a, b, form=tracker.form, h2h=tracker.h2h)
            print(f"  {a} {ga} - {gb} {b}")


if __name__ == "__main__":
    main()
