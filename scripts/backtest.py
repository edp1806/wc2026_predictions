# -*- coding: utf-8 -*-
"""Backtest walk-forward : compare le modèle de base (ELO + domicile
seulement, l'équivalent du notebook Colab d'origine) au modèle complet
(+ forme récente + enjeu tournoi + h2h + Dixon-Coles attaque/défense),
sur les mêmes matchs de test.

Principe : jamais d'information du futur. Pour chaque match de test, on
prédit avec un modèle entraîné uniquement sur ce qui le précède
chronologiquement, puis on intègre le résultat réel avant de passer au
match suivant.

Métriques :
- probabilité moyenne donnée par le modèle au résultat réel (33% = hasard
  pur, 100% = modèle parfait)
- erreur xG moyenne (écart absolu moyen entre buts prédits et buts réels)

Usage :
    python -m scripts.backtest
    python -m scripts.backtest --test-days 365 --trials 1500
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, data_loader, elo, poisson_model
from src.simulate import match_probabilities


def _proba_and_error(model, ratings, home, away, home_score, away_score, trials, form, h2h):
    r = match_probabilities(model, ratings, home, away, trials=trials, form=form, h2h=h2h)
    if home_score > away_score:
        p = r["p_win_a"]
    elif home_score < away_score:
        p = r["p_win_b"]
    else:
        p = r["p_draw"]
    xg_error = (abs(r["xg_a"] - home_score) + abs(r["xg_b"] - away_score)) / 2
    return p, xg_error


def run_backtest(
    test_days: int,
    retrain_every: int,
    trials: int,
    label: str,
    years: int | None = config.TRAINING_YEARS,
    half_life: float | None = config.RECENCY_HALF_LIFE_DAYS,
    use_form: bool = True,
    use_competition: bool = True,
    use_h2h: bool = True,
    use_dixon_coles: bool = True,
) -> dict:
    """Charge les données, réserve les `test_days` derniers jours comme
    jeu de test, et fait tourner l'évaluation walk-forward dessus. Les
    flags `use_*` permettent de désactiver une feature pour comparer
    objectivement avec/sans (voir main())."""
    results = data_loader.load_results(years=years)
    reference_date = results["date"].max()

    cutoff = reference_date - pd.Timedelta(days=test_days)
    train_df = results[results["date"] < cutoff].reset_index(drop=True)
    test_df = results[results["date"] >= cutoff].reset_index(drop=True)

    ratings, form_history, h2h_history, history = elo.compute_elo_history(
        train_df, recency_half_life_days=half_life, reference_date=reference_date
    )
    model = poisson_model.train_model(
        history, use_form=use_form, use_competition=use_competition, use_h2h=use_h2h, use_dixon_coles=use_dixon_coles
    )

    per_match = []
    for i, r in test_df.iterrows():
        home, away = r["home_team"], r["away_team"]

        p, xg_error = _proba_and_error(
            model, ratings, home, away, r["home_score"], r["away_score"], trials, form_history, h2h_history
        )
        per_match.append(
            {
                "date": r["date"].date(),
                "home": home,
                "away": away,
                "score": f"{int(r['home_score'])}-{int(r['away_score'])}",
                "proba_actual": p,
                "xg_error": xg_error,
            }
        )

        weight = elo.recency_weight(r["date"], reference_date, half_life)
        home_form_pre, away_form_pre = elo.get_form(form_history, home), elo.get_form(form_history, away)
        h2h_edge_pre = elo.get_h2h_edge(h2h_history, home, away)

        elo.apply_result(ratings, home, away, r["home_score"], r["away_score"], r["tournament"], r["neutral"], weight=weight)
        history.append(
            {
                "home_team": home, "away_team": away,
                "home_elo": ratings.get(home, config.ELO_BASE), "away_elo": ratings.get(away, config.ELO_BASE),
                "home_form": home_form_pre, "away_form": away_form_pre, "h2h_edge": h2h_edge_pre,
                "neutral": r["neutral"], "home_score": r["home_score"], "away_score": r["away_score"],
                "weight": weight, "date": r["date"], "tournament": r["tournament"],
            }
        )
        result_home, result_away = elo.match_result_value(r["home_score"], r["away_score"])
        elo.update_form(form_history, home, result_home)
        elo.update_form(form_history, away, result_away)
        elo.update_h2h(h2h_history, home, away, result_home)

        if (i + 1) % retrain_every == 0:
            model = poisson_model.train_model(
                history, use_form=use_form, use_competition=use_competition, use_h2h=use_h2h, use_dixon_coles=use_dixon_coles
            )

    df = pd.DataFrame(per_match)
    return {
        "label": label,
        "n_matches": len(df),
        "avg_proba_true": df["proba_actual"].mean() if len(df) else float("nan"),
        "avg_xg_error": df["xg_error"].mean() if len(df) else float("nan"),
        "per_match": df,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare le modèle de base au modèle complet, en walk-forward.")
    parser.add_argument("--test-days", type=int, default=180, help="Taille du jeu de test, en jours (défaut 180)")
    parser.add_argument("--retrain-every", type=int, default=10, help="Réentraîne le Poisson tous les N matchs")
    parser.add_argument("--trials", type=int, default=3000, help="Tirages Monte Carlo par match testé")
    parser.add_argument("--verbose", action="store_true", help="Affiche le détail match par match")
    args = parser.parse_args()

    print(f"Backtest walk-forward sur les {args.test_days} derniers jours "
          f"(réentraînement tous les {args.retrain_every} matchs, {args.trials} tirages/match)\n")

    common = dict(test_days=args.test_days, retrain_every=args.retrain_every, trials=args.trials)

    base = run_backtest(
        **common, use_form=False, use_competition=False, use_h2h=False, use_dixon_coles=False,
        label="BASE     — ELO + domicile seulement (équivalent notebook Colab d'origine)",
    )
    full = run_backtest(
        **common, use_form=True, use_competition=True, use_h2h=True, use_dixon_coles=True,
        label="COMPLET  — + forme + enjeu tournoi + h2h + Dixon-Coles",
    )

    print("=" * 82)
    for res in (base, full):
        print(res["label"])
        print(f"  Matchs testés                                : {res['n_matches']}")
        print(f"  Probabilité moyenne donnée au résultat réel  : {res['avg_proba_true']:.1%}  (hasard pur = 33%)")
        print(f"  Erreur xG moyenne                            : {res['avg_xg_error']:.2f} but(s)/équipe")
        print("-" * 82)

    delta_p = full["avg_proba_true"] - base["avg_proba_true"]
    delta_xg = full["avg_xg_error"] - base["avg_xg_error"]
    verdict = "meilleur" if delta_p > 0 else ("moins bon" if delta_p < 0 else "identique")
    print(f"\nÉcart (complet - base) : {delta_p:+.1%} sur la probabilité donnée au résultat réel, "
          f"{delta_xg:+.2f} but(s) sur l'erreur xG moyenne")
    print(f"=> le modèle complet est {verdict} sur cette fenêtre de test.")

    if args.verbose:
        print("\n=== Détail BASE ===")
        print(base["per_match"].to_string(index=False))
        print("\n=== Détail COMPLET ===")
        print(full["per_match"].to_string(index=False))


if __name__ == "__main__":
    main()
