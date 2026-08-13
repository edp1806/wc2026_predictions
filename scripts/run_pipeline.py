# -*- coding: utf-8 -*-
"""Pipeline complet : charge les données -> ELO -> Poisson -> Monte Carlo.

Usage :
    python -m scripts.run_pipeline
    python -m scripts.run_pipeline --simulations 20000 --fresh
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, data_loader, elo, poisson_model, simulate, state, visualize
from src.tracking import PredictionTracker


def main():
    parser = argparse.ArgumentParser(description="Entraîne le modèle WC2026 et lance Monte Carlo.")
    parser.add_argument("--simulations", type=int, default=config.N_SIMULATIONS)
    parser.add_argument("--fresh", action="store_true", help="Ignore l'état sauvegardé, recalcule depuis le CSV")
    parser.add_argument("--top", type=int, default=20, help="Nombre d'équipes affichées dans le classement")
    parser.add_argument(
        "--years", type=int, default=config.TRAINING_YEARS,
        help=f"N'entraîne que sur les N dernières années (défaut {config.TRAINING_YEARS}). "
             f"0 ou négatif = tout l'historique.",
    )
    parser.add_argument(
        "--half-life", type=int, default=config.RECENCY_HALF_LIFE_DAYS,
        help=f"Demi-vie en jours de la pondération temporelle (défaut {config.RECENCY_HALF_LIFE_DAYS}). "
             f"0 ou négatif = pas de décroissance, tous les matchs de la fenêtre pèsent pareil.",
    )
    parser.add_argument(
        "--from-rd32", action="store_true",
        help="Simule depuis le RD32 (groupes déjà joués) en gelant config.KNOWN_RD32_WINNERS, "
             "plutôt que de resimuler un tournoi complet depuis les groupes.",
    )
    args = parser.parse_args()

    years = args.years if args.years and args.years > 0 else None
    half_life = args.half_life if args.half_life and args.half_life > 0 else None

    tracker = None if args.fresh else state.load_state()

    if tracker is not None:
        print(f"État rechargé : {len(tracker.ratings)} équipes, {len(tracker.history):,} matchs, "
              f"{len(tracker.prediction_log)} prédictions loggées")
        print("(--years / --half-life ignorés : ils ne s'appliquent qu'au calcul initial avec --fresh)")
    else:
        print("Aucun état sauvegardé — calcul depuis zéro.")
        print(f"Fenêtre d'entraînement : {'toutes les années' if years is None else f'{years} dernières années'}")
        print(f"Décroissance temporelle : {'aucune' if half_life is None else f'demi-vie {half_life} jours'}")

        results = data_loader.load_results(years=years)
        print(f"Période couverte : {results['date'].min().date()} → {results['date'].max().date()}")
        print(f"Matchs : {len(results):,}")

        ratings, form_history, h2h_history, history = elo.compute_elo_history(results, recency_half_life_days=half_life)
        print(f"ELO calculé sur {len(results):,} matchs, {len(ratings)} équipes")

        model = poisson_model.train_model(history)
        print("\nModèle entraîné :")
        print(model.summary())

        tracker = PredictionTracker(ratings=ratings, history=history, model=model, form=form_history, h2h=h2h_history)
        state.save_state(tracker)

    print("\n=== TOP ELO MONDIAL ===")
    print(elo.elo_ranking(tracker.ratings).head(args.top).to_string())

    print(f"\nLancement de {args.simulations:,} simulations Monte Carlo...")
    known_winners = config.KNOWN_RD32_WINNERS if args.from_rd32 else None
    known_r16_winners = config.KNOWN_R16_WINNERS if args.from_rd32 else None
    known_qf_winners = config.KNOWN_QF_WINNERS if args.from_rd32 else None
    if args.from_rd32:
        n_known = len(known_winners)
        n_known_r16 = len(known_r16_winners)
        n_known_qf = len(known_qf_winners)
        print(f"Mode --from-rd32 : {n_known}/16 résultats RD32 gelés, {n_known_r16}/8 résultats R16 gelés, "
              f"{n_known_qf}/4 résultats QF gelés, le reste est simulé.")
    counts = simulate.monte_carlo(
        tracker.model, tracker.ratings, n_simulations=args.simulations, known_rd32_winners=known_winners,
        known_r16_winners=known_r16_winners, known_qf_winners=known_qf_winners, form=tracker.form, h2h=tracker.h2h,
    )

    rows = sorted(counts.items(), key=lambda kv: kv[1].get("champion", 0), reverse=True)
    team_elo = {name: round(tracker.ratings.get(name, config.ELO_BASE)) for _, name, _, _ in config.WC2026_TEAMS}

    print("\n" + "=" * 70)
    print(f"  PROBABILITÉS — COUPE DU MONDE 2026 ({args.simulations:,} simulations)")
    print("=" * 70)
    print(f"  {'#':>2}  {'Équipe':<18} {'Elo':>5}  {'Champion':>9}  {'Finale':>7}  {'Demi':>7}")
    print("  " + "-" * 66)
    for i, (name, c) in enumerate(rows[: args.top], 1):
        print(
            f"  {i:>2}  {name:<18} {team_elo.get(name, 1000):>5}  "
            f"{c.get('champion', 0) / args.simulations:>8.1%}  "
            f"{c.get('final', 0) / args.simulations:>7.1%}  "
            f"{c.get('sf', 0) / args.simulations:>7.1%}"
        )

    png_path = visualize.plot_title_probabilities(counts, args.simulations)
    print(f"\nGraphique enregistré : {png_path}")


if __name__ == "__main__":
    main()
