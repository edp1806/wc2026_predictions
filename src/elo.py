# -*- coding: utf-8 -*-
"""Système de classement ELO.

- Base ELO = 1000
- K-factor par type de tournoi (voir config.k_factor)
- Multiplicateur d'écart de buts plafonné : ≤1 but -> x1.0, 2 buts -> x1.5, 3+ -> x1.75
- Avantage du terrain : +75 ELO si le match n'est pas neutre

En plus du rating final de chaque équipe, on conserve un "snapshot" de
l'ELO de chaque équipe juste AVANT chaque match : c'est exactement ce que
la régression de Poisson (poisson_model.py) utilise comme feature
d'entraînement.
"""

import pandas as pd

from . import config


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probabilité théorique de victoire de A selon l'écart ELO."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def get_rating(ratings: dict, team: str) -> float:
    return ratings.setdefault(team, config.ELO_BASE)


def recency_weight(match_date, reference_date, half_life_days: float | None) -> float:
    """Poids de décroissance exponentielle : 1.0 pour un match à
    `reference_date`, 0.5 pour un match vieux de `half_life_days`, 0.25
    pour 2x cette durée, etc. `half_life_days=None` -> poids uniforme (1.0)."""
    if half_life_days is None:
        return 1.0
    days_ago = max((reference_date - match_date).days, 0)
    return 0.5 ** (days_ago / half_life_days)


def apply_result(
    ratings: dict,
    home: str,
    away: str,
    home_score: float,
    away_score: float,
    tournament: str,
    neutral: bool,
    weight: float = 1.0,
) -> None:
    """Met à jour `ratings` (in place) avec le résultat d'un match.

    `weight` multiplie le K-factor : un match récent (weight proche de 1)
    déplace le rating plus qu'un match ancien (weight proche de 0) au sein
    de la fenêtre d'entraînement — voir `recency_weight`.
    """
    home_elo = get_rating(ratings, home)
    away_elo = get_rating(ratings, away)

    home_adv = 0 if neutral else config.HOME_ADVANTAGE
    ea = expected_score(home_elo + home_adv, away_elo)
    eb = 1 - ea

    if home_score > away_score:
        aa, ab = 1.0, 0.0
    elif home_score < away_score:
        aa, ab = 0.0, 1.0
    else:
        aa, ab = 0.5, 0.5

    k = config.k_factor(tournament) * weight
    gd = abs(home_score - away_score)
    gd_mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else 1.75)

    ratings[home] = home_elo + k * gd_mult * (aa - ea)
    ratings[away] = away_elo + k * gd_mult * (ab - eb)


def match_result_value(home_score: float, away_score: float) -> tuple[float, float]:
    """1.0 / 0.5 / 0.0 (victoire / nul / défaite) du point de vue de
    chaque équipe — utilisé à la fois par apply_result (ELO) et par le
    calcul de forme récente."""
    if home_score > away_score:
        return 1.0, 0.0
    if home_score < away_score:
        return 0.0, 1.0
    return 0.5, 0.5


def get_form(form_history: dict, team: str) -> float:
    """Moyenne des FORM_WINDOW derniers résultats de `team`. 0.5 (neutre)
    si l'équipe n'a pas encore de match connu — ne pas la pénaliser ni
    l'avantager par défaut."""
    results = form_history.get(team, [])
    if not results:
        return 0.5
    return sum(results) / len(results)


def update_form(form_history: dict, team: str, result: float) -> None:
    """Ajoute `result` à l'historique glissant de `team` (in place),
    tronqué à FORM_WINDOW matchs."""
    results = form_history.setdefault(team, [])
    results.append(result)
    if len(results) > config.FORM_WINDOW:
        results.pop(0)


def h2h_key(team_a: str, team_b: str) -> tuple[str, str]:
    """Clé symétrique pour une paire d'équipes, indépendante de l'ordre
    home/away."""
    return tuple(sorted([team_a, team_b]))


def get_h2h_edge(h2h_history: dict, team_a: str, team_b: str, min_matches: int = config.H2H_MIN_MATCHES) -> float:
    """Avantage historique de `team_a` face à `team_b` spécifiquement
    (pas sa force générale) : +1 si team_a a toujours gagné leurs
    confrontations directes, -1 si toujours perdu, 0 si équilibré ou pas
    assez de confrontations connues (< `min_matches`)."""
    key = h2h_key(team_a, team_b)
    matches = h2h_history.get(key, [])
    if len(matches) < min_matches:
        return 0.0
    ref_team = key[0]  # équipe alphabétiquement première = référence de stockage
    avg_result_ref = sum(matches) / len(matches)
    edge_for_ref = 2 * avg_result_ref - 1  # [-1, 1]
    return edge_for_ref if team_a == ref_team else -edge_for_ref


def update_h2h(h2h_history: dict, team_a: str, team_b: str, result_a: float) -> None:
    """`result_a` : résultat de `team_a` dans CE match (1/0.5/0). Stocké
    du point de vue de l'équipe de référence (alphabétiquement première)
    pour que la clé reste symétrique quel que soit qui reçoit."""
    key = h2h_key(team_a, team_b)
    ref_team = key[0]
    result_ref = result_a if team_a == ref_team else 1 - result_a
    h2h_history.setdefault(key, []).append(result_ref)


def compute_elo_history(
    results: pd.DataFrame,
    recency_half_life_days: float | None = config.RECENCY_HALF_LIFE_DAYS,
    reference_date=None,
) -> tuple[dict, dict, dict, list[dict]]:
    """Calcule l'ELO, la forme récente et l'historique face-à-face sur
    l'historique fourni (déjà filtré à la bonne fenêtre par
    data_loader.load_results si besoin).

    `recency_half_life_days` : décroissance temporelle du K-factor au sein
    de cet historique (voir `recency_weight`). `reference_date=None` prend
    la date du match le plus récent du DataFrame comme "aujourd'hui".

    Retourne (ratings, form_history, h2h_history, history) :
    - `ratings` : Elo actuel de chaque équipe
    - `form_history` : les FORM_WINDOW derniers résultats de chaque équipe
    - `h2h_history` : confrontations directes connues par paire d'équipes
    - `history` : snapshots pré-match (Elo, forme, h2h, identités
      d'équipes, poids) utilisés pour entraîner la régression de Poisson
      (features + dummies attaque/défense Dixon-Coles)
    """
    ratings: dict[str, float] = {}
    form_history: dict[str, list[float]] = {}
    h2h_history: dict[tuple[str, str], list[float]] = {}
    history: list[dict] = []

    reference_date = reference_date or results["date"].max()

    for _, r in results.iterrows():
        home, away = r["home_team"], r["away_team"]
        home_elo = get_rating(ratings, home)
        away_elo = get_rating(ratings, away)
        home_form = get_form(form_history, home)
        away_form = get_form(form_history, away)
        h2h_edge = get_h2h_edge(h2h_history, home, away)
        weight = recency_weight(r["date"], reference_date, recency_half_life_days)

        history.append(
            {
                "home_team": home,
                "away_team": away,
                "home_elo": home_elo,
                "away_elo": away_elo,
                "home_form": home_form,
                "away_form": away_form,
                "h2h_edge": h2h_edge,
                "neutral": r["neutral"],
                "home_score": r["home_score"],
                "away_score": r["away_score"],
                "weight": weight,
                "date": r["date"],
                "tournament": r["tournament"],
            }
        )

        apply_result(ratings, home, away, r["home_score"], r["away_score"], r["tournament"], r["neutral"], weight=weight)

        result_home, result_away = match_result_value(r["home_score"], r["away_score"])
        update_form(form_history, home, result_home)
        update_form(form_history, away, result_away)
        update_h2h(h2h_history, home, away, result_home)

    return ratings, form_history, h2h_history, history


def elo_ranking(ratings: dict) -> pd.DataFrame:
    """Classement ELO trié, sous forme de DataFrame lisible."""
    df = pd.DataFrame(sorted(ratings.items(), key=lambda x: -x[1]), columns=["Équipe", "ELO"])
    df["ELO"] = df["ELO"].round(0).astype(int)
    df.index = range(1, len(df) + 1)
    return df
