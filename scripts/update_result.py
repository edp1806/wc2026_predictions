# -*- coding: utf-8 -*-
"""Intègre un résultat réel dans le modèle : met à jour l'ELO, compare à
la prédiction loggée (si elle existe), réentraîne la régression de Poisson,
et sauvegarde le nouvel état.

Usage :
    # 1. Avant le match, on fige ce que le modèle pensait :
    python -m scripts.update_result --log France Brazil

    # 2. Après le match, on intègre le vrai score :
    python -m scripts.update_result France Brazil 2 1 --neutral
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import state


def main():
    parser = argparse.ArgumentParser(description="Log ou intègre un résultat de match WC2026.")
    parser.add_argument("home")
    parser.add_argument("away")
    parser.add_argument("home_score", type=int, nargs="?")
    parser.add_argument("away_score", type=int, nargs="?")
    parser.add_argument("--log", action="store_true", help="Log seulement la prédiction (avant le match)")
    parser.add_argument("--neutral", action="store_true", default=True)
    parser.add_argument("--tournament", default="FIFA World Cup")
    parser.add_argument("--retrain", action="store_true", help="Réentraîne la régression de Poisson après")
    args = parser.parse_args()

    tracker = state.load_state()
    if tracker is None:
        print("Aucun état sauvegardé. Lance d'abord : python -m scripts.run_pipeline")
        return

    if args.log:
        entry = tracker.log_prediction(args.home, args.away)
        print(f"Prédiction enregistrée : {args.home} {entry['pred_xg_home']} – {entry['pred_xg_away']} {args.away}")
        print(f"  ({entry['pred_p_home']:.0%} / {entry['pred_p_draw']:.0%} / {entry['pred_p_away']:.0%})")
        state.save_state(tracker)
        return

    if args.home_score is None or args.away_score is None:
        parser.error("home_score et away_score sont requis sauf avec --log")

    old_home = round(tracker.ratings.get(args.home, 1000.0))
    old_away = round(tracker.ratings.get(args.away, 1000.0))

    comparison = tracker.update_with_result(
        args.home, args.away, args.home_score, args.away_score,
        tournament=args.tournament, neutral=args.neutral,
    )

    if comparison:
        print(f"{args.home} {args.home_score}-{args.away_score} {args.away}")
        print(f"  Probabilité que le modèle avait donnée à ce résultat : {comparison['proba_assigned_to_actual']:.1%}")
        print(f"  Erreur xG moyenne : {comparison['xg_error']} but(s)")
    else:
        print(f"Pas de prédiction enregistrée pour {args.home} vs {args.away} — ajout direct sans comparaison")

    new_home = round(tracker.ratings[args.home])
    new_away = round(tracker.ratings[args.away])
    print(f"  ELO {args.home}: {old_home} → {new_home} ({new_home - old_home:+d})")
    print(f"  ELO {args.away}: {old_away} → {new_away} ({new_away - old_away:+d})")

    if args.retrain:
        tracker.retrain()
        print(f"\nModèle réentraîné sur {len(tracker.history):,} matchs")
        print(tracker.model.summary())

    state.save_state(tracker)
    print("\nÉtat sauvegardé.")


if __name__ == "__main__":
    main()
