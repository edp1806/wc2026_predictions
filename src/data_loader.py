# -*- coding: utf-8 -*-
"""Chargement et normalisation des données historiques (results.csv,
former_names.csv).

Isolé dans son propre module pour que le reste du pipeline ne dépende
jamais du format brut des CSV — si la source de données change demain
(nouveau dump, nouvelle colonne), c'est le seul fichier à toucher.
"""

import pandas as pd

from . import config


def load_former_names(path=None) -> dict:
    """Charge la table des anciens noms de pays (Dahomey -> Benin, etc.)."""
    path = path or config.FORMER_NAMES_CSV
    former = pd.read_csv(path)
    return {row["former"]: row["current"] for _, row in former.iterrows()}


def build_name_map(path=None) -> dict:
    """former_names.csv + les cas manuels non couverts (Yougoslavie, etc.)."""
    name_map = load_former_names(path)
    return {**name_map, **config.EXTRA_NAME_MAP}


def load_results(
    results_path=None, former_names_path=None, years: int | None = config.TRAINING_YEARS
) -> pd.DataFrame:
    """Charge results.csv, normalise les noms d'équipes, trie par date.

    Ne garde que les matchs avec un score connu (les lignes de matchs
    futurs contiennent des NaN dans home_score/away_score).

    `years` : ne garde que les `years` dernières années de matchs (par
    rapport à la date la plus récente du fichier). None = tout l'historique.
    Passe `years=None` explicitement pour désactiver la fenêtre, y compris
    si `config.TRAINING_YEARS` est défini.
    """
    results_path = results_path or config.RESULTS_CSV
    results = pd.read_csv(results_path)
    results["date"] = pd.to_datetime(results["date"], format="%Y-%m-%d")

    results = results.dropna(subset=["home_score", "away_score"]).copy()
    results = results.sort_values("date").reset_index(drop=True)

    if years is not None:
        cutoff = results["date"].max() - pd.DateOffset(years=years)
        results = results[results["date"] >= cutoff].reset_index(drop=True)

    full_name_map = build_name_map(former_names_path)
    results["home_team"] = results["home_team"].map(lambda x: full_name_map.get(x, x))
    results["away_team"] = results["away_team"].map(lambda x: full_name_map.get(x, x))

    return results
