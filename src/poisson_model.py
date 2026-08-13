# -*- coding: utf-8 -*-
"""Régression de Poisson : prédit le nombre de buts attendus (xG) de
chaque équipe.

    log(buts attendus) = β0 + β1*(Δ ELO/100) + β2*domicile + β3*Δforme
                          + β4*enjeu_tournoi + β5*h2h_edge
                          + attaque[équipe] + défense[adversaire]

Les deux derniers termes (attaque/défense par équipe) sont l'approche
Dixon-Coles : au lieu de ne dépendre que de l'écart ELO agrégé, chaque
équipe a une force offensive et une force défensive propres, apprises
comme des "dummies" (indicatrices one-hot) dans la même régression de
Poisson — pas besoin d'un second modèle séparé.

Les coefficients sont appris par maximum de vraisemblance sur ~49 000
matchs historiques (2 lignes d'entraînement par match — une par équipe,
pour que le modèle apprenne une seule relation valable côté favori comme
côté outsider).
"""

from dataclasses import dataclass, field

import math
import numpy as np
from scipy import sparse
from sklearn.linear_model import PoissonRegressor

from . import config

N_BASE_FEATURES = 5  # elo_diff, host_flag, form_diff, competition_weight, h2h_edge


@dataclass
class PoissonGoalModel:
    """Petit wrapper autour du PoissonRegressor + ses coefficients bruts.

    On garde les coefficients à part (et pas seulement l'objet sklearn)
    car c'est ce format qui est sérialisé dans wc2026_model_state.pkl
    pour la persistance entre sessions. `attack`/`defense` : dict
    équipe -> coefficient Dixon-Coles (0.0 pour une équipe inconnue du
    dernier entraînement).
    """

    intercept: float
    elo_coef: float
    home_coef: float
    form_coef: float
    competition_coef: float
    h2h_coef: float
    attack: dict = field(default_factory=dict)
    defense: dict = field(default_factory=dict)
    sklearn_model: PoissonRegressor | None = None

    def expected_goals(
        self,
        elo_a: float,
        elo_b: float,
        team_a: str | None = None,
        team_b: str | None = None,
        a_home: bool = False,
        b_home: bool = False,
        form_a: float = 0.5,
        form_b: float = 0.5,
        competition_weight: float = 1.0,
        h2h_edge: float = 0.0,
    ) -> tuple[float, float]:
        """xG des deux équipes.

        `team_a`/`team_b` : noms d'équipe pour les termes Dixon-Coles
        attaque/défense — None ou équipe inconnue -> traité comme
        neutre (0.0). `form_a`/`form_b` : forme récente (0 à 1, 0.5 =
        neutre). `h2h_edge` : avantage historique de `team_a` sur
        `team_b` dans leurs confrontations directes (-1 à 1, 0 = neutre).
        `competition_weight` : enjeu du tournoi (0.33 à 1.0). Pour un
        match Coupe du Monde, laisser competition_weight=1.0 (défaut)."""
        attack_a = self.attack.get(team_a, 0.0)
        defense_a = self.defense.get(team_a, 0.0)
        attack_b = self.attack.get(team_b, 0.0)
        defense_b = self.defense.get(team_b, 0.0)

        log_xg_a = (
            self.intercept
            + self.elo_coef * (elo_a - elo_b) / config.ELO_DIFF_SCALE
            + self.home_coef * (1.0 if a_home else 0.0)
            + self.form_coef * (form_a - form_b)
            + self.competition_coef * competition_weight
            + self.h2h_coef * h2h_edge
            + attack_a
            + defense_b
        )
        log_xg_b = (
            self.intercept
            + self.elo_coef * (elo_b - elo_a) / config.ELO_DIFF_SCALE
            + self.home_coef * (1.0 if b_home else 0.0)
            + self.form_coef * (form_b - form_a)
            + self.competition_coef * competition_weight
            + self.h2h_coef * (-h2h_edge)
            + attack_b
            + defense_a
        )
        return math.exp(log_xg_a), math.exp(log_xg_b)

    def summary(self, top_n: int = 5) -> str:
        top_attack = sorted(self.attack.items(), key=lambda kv: -kv[1])[:top_n]
        # défense la plus solide = coefficient le plus négatif (fait le moins marquer l'adversaire)
        top_defense = sorted(self.defense.items(), key=lambda kv: kv[1])[:top_n]

        lines = [
            f"Intercept (β0)         : {self.intercept:.4f}",
            f"Coef ELO (β1)          : {self.elo_coef:.4f}",
            f"Coef domicile (β2)     : {self.home_coef:.4f}",
            f"Coef forme (β3)        : {self.form_coef:.4f}",
            f"Coef enjeu tournoi (β4): {self.competition_coef:.4f}",
            f"Coef h2h (β5)          : {self.h2h_coef:.4f}",
            f"Match neutre, équipes égales/neutres -> {math.exp(self.intercept + self.competition_coef):.2f} buts attendus/équipe (WC)",
            f"Jouer à domicile multiplie les buts par x{math.exp(self.home_coef):.2f}",
            f"Chaque +100 ELO multiplie les buts attendus par x{math.exp(self.elo_coef):.2f}",
            "",
            f"Top {top_n} attaques (Dixon-Coles) :",
        ]
        for t, c in top_attack:
            lines.append(f"  {t:<20} +{c:.3f}  (x{math.exp(c):.2f} buts marqués)")
        lines.append(f"Top {top_n} défenses (Dixon-Coles) :")
        for t, c in top_defense:
            lines.append(f"  {t:<20} {c:.3f}  (x{math.exp(c):.2f} buts encaissés)")
        return "\n".join(lines)


def build_training_matrix(
    history: list[dict],
    use_form: bool = config.USE_FORM,
    use_competition: bool = config.USE_COMPETITION_WEIGHT,
    use_h2h: bool = config.USE_H2H,
    use_dixon_coles: bool = config.USE_DIXON_COLES,
):
    """Empile les 2 perspectives (domicile / extérieur) de chaque match du
    `history` calculé par elo.compute_elo_history.

    5 features de base (écart ELO, domicile, diff. forme, enjeu tournoi,
    h2h) + 2×N_équipes colonnes one-hot creuses (attaque/défense
    Dixon-Coles). Retourne une matrice sparse (scipy.sparse.csr_matrix)
    pour rester gérable en mémoire avec ~330 équipes.

    Les flags `use_*` permettent de désactiver une feature (mise à 0) —
    utile pour un backtest comparatif "avec/sans" (voir scripts/backtest.py).

    Retourne (X, y, sample_weight, teams) où `teams` est la liste triée
    des équipes (ordre des colonnes one-hot, nécessaire pour relire les
    coefficients après entraînement) — vide si use_dixon_coles=False."""
    teams = []
    if use_dixon_coles:
        match_count: dict[str, int] = {}
        for m in history:
            for t in (m.get("home_team"), m.get("away_team")):
                if t:
                    match_count[t] = match_count.get(t, 0) + 1
        teams = sorted(t for t, n in match_count.items() if n >= config.DIXON_COLES_MIN_MATCHES)
    team_index = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    base_rows, y, w = [], [], []
    attack_rows, attack_cols = [], []
    defense_rows, defense_cols = [], []
    row = 0

    for m in history:
        host_flag = 0.0 if m["neutral"] else 1.0
        weight = m.get("weight", 1.0)
        home_form = m.get("home_form", 0.5) if use_form else 0.5
        away_form = m.get("away_form", 0.5) if use_form else 0.5
        comp_weight = config.competition_weight(m.get("tournament", "Friendly")) if use_competition else 0.0
        h2h_edge = m.get("h2h_edge", 0.0) if use_h2h else 0.0
        home_team, away_team = m.get("home_team"), m.get("away_team")

        # perspective domicile
        base_rows.append(
            [(m["home_elo"] - m["away_elo"]) / config.ELO_DIFF_SCALE, host_flag, home_form - away_form, comp_weight, h2h_edge]
        )
        y.append(m["home_score"])
        w.append(weight)
        if use_dixon_coles:
            if home_team in team_index:
                attack_rows.append(row)
                attack_cols.append(team_index[home_team])
            if away_team in team_index:
                defense_rows.append(row)
                defense_cols.append(team_index[away_team])
        row += 1

        # perspective extérieur
        base_rows.append(
            [(m["away_elo"] - m["home_elo"]) / config.ELO_DIFF_SCALE, 0.0, away_form - home_form, comp_weight, -h2h_edge]
        )
        y.append(m["away_score"])
        w.append(weight)
        if use_dixon_coles:
            if away_team in team_index:
                attack_rows.append(row)
                attack_cols.append(team_index[away_team])
            if home_team in team_index:
                defense_rows.append(row)
                defense_cols.append(team_index[home_team])
        row += 1

    n_rows = row
    X_base = sparse.csr_matrix(np.array(base_rows))
    if use_dixon_coles:
        attack_onehot = sparse.csr_matrix(
            (np.ones(len(attack_rows)), (attack_rows, attack_cols)), shape=(n_rows, n_teams)
        )
        defense_onehot = sparse.csr_matrix(
            (np.ones(len(defense_rows)), (defense_rows, defense_cols)), shape=(n_rows, n_teams)
        )
        X = sparse.hstack([X_base, attack_onehot, defense_onehot], format="csr")
    else:
        X = X_base

    return X, np.array(y), np.array(w), teams


def train_model(
    history: list[dict],
    use_form: bool = config.USE_FORM,
    use_competition: bool = config.USE_COMPETITION_WEIGHT,
    use_h2h: bool = config.USE_H2H,
    use_dixon_coles: bool = config.USE_DIXON_COLES,
) -> PoissonGoalModel:
    """Entraîne (ou réentraîne) la régression de Poisson sur `history`,
    pondérée par la récence de chaque match (voir build_training_matrix).

    Régularisation plus forte que les features de base seules
    (config.DIXON_COLES_ALPHA) si use_dixon_coles=True, sinon la
    régularisation quasi nulle d'origine (1e-8) — inutile de pénaliser 5
    features de base bien identifiées sur 49 000 matchs."""
    X, y, sample_weight, teams = build_training_matrix(
        history, use_form=use_form, use_competition=use_competition, use_h2h=use_h2h, use_dixon_coles=use_dixon_coles
    )

    alpha = config.DIXON_COLES_ALPHA if use_dixon_coles else 1e-8
    sklearn_model = PoissonRegressor(alpha=alpha, max_iter=3000)
    sklearn_model.fit(X, y, sample_weight=sample_weight)

    coefs = sklearn_model.coef_
    intercept = float(sklearn_model.intercept_)
    elo_coef, home_coef, form_coef, competition_coef, h2h_coef = (float(c) for c in coefs[:N_BASE_FEATURES])

    n_teams = len(teams)
    if use_dixon_coles:
        attack_coefs = coefs[N_BASE_FEATURES : N_BASE_FEATURES + n_teams]
        defense_coefs = coefs[N_BASE_FEATURES + n_teams : N_BASE_FEATURES + 2 * n_teams]
        attack = {t: float(c) for t, c in zip(teams, attack_coefs)}
        defense = {t: float(c) for t, c in zip(teams, defense_coefs)}
    else:
        attack, defense = {}, {}

    return PoissonGoalModel(
        intercept=intercept,
        elo_coef=elo_coef,
        home_coef=home_coef,
        form_coef=form_coef,
        competition_coef=competition_coef,
        h2h_coef=h2h_coef,
        attack=attack,
        defense=defense,
        sklearn_model=sklearn_model,
    )
