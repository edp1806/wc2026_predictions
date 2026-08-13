# -*- coding: utf-8 -*-
"""Boucle d'amélioration continue.

Workflow recommandé pendant le tournoi :
1. Avant un match  -> tracker.log_prediction(home, away)     (fige la prédiction)
2. Après le match   -> tracker.update_with_result(...)        (met à jour l'ELO, mesure l'erreur)
3. Tous les 4-5 matchs -> poisson_model.train_model(tracker.history) (réentraîne)
4. Relancer simulate.monte_carlo() pour des probabilités de titre à jour
5. tracker.performance_dashboard() pour suivre la précision au fil du tournoi
"""

from dataclasses import dataclass, field

import pandas as pd

from . import config, elo
from .poisson_model import PoissonGoalModel
from .simulate import match_probabilities


@dataclass
class PredictionTracker:
    """Regroupe l'état mutable du pipeline (ratings, forme, h2h, history,
    log de prédictions) pour qu'il puisse être sérialisé tel quel (voir
    state.py) et réutilisé d'une session à l'autre."""

    ratings: dict
    history: list
    model: PoissonGoalModel
    form: dict = field(default_factory=dict)
    h2h: dict = field(default_factory=dict)
    prediction_log: list = field(default_factory=list)

    def log_prediction(self, home: str, away: str) -> dict:
        """Capture la prédiction du modèle AVANT de connaître le résultat réel."""
        elo_home = self.ratings.get(home, config.ELO_BASE)
        elo_away = self.ratings.get(away, config.ELO_BASE)
        form_home = elo.get_form(self.form, home)
        form_away = elo.get_form(self.form, away)
        h2h_edge = elo.get_h2h_edge(self.h2h, home, away)
        pred = match_probabilities(
            self.model, self.ratings, home, away, trials=20_000,
            form=self.form, h2h=self.h2h, competition_weight=1.0,
        )
        entry = {
            "home": home,
            "away": away,
            "elo_home_pre": round(elo_home),
            "elo_away_pre": round(elo_away),
            "form_home_pre": round(form_home, 2),
            "form_away_pre": round(form_away, 2),
            "h2h_edge_pre": round(h2h_edge, 2),
            "pred_p_home": pred["p_win_a"],
            "pred_p_draw": pred["p_draw"],
            "pred_p_away": pred["p_win_b"],
            "pred_xg_home": pred["xg_a"],
            "pred_xg_away": pred["xg_b"],
        }
        self.prediction_log.append(entry)
        return entry

    def update_with_result(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        tournament: str = "FIFA World Cup",
        neutral: bool = True,
    ) -> dict | None:
        """Intègre un résultat réel : met à jour l'ELO, mesure l'erreur de
        la dernière prédiction loggée pour ce match (si elle existe), et
        ajoute le match à `history` pour le prochain réentraînement.

        Retourne le détail de la comparaison prédiction/réalité, ou None
        si aucune prédiction n'avait été loggée pour ce match.
        """
        match_pred = next(
            (p for p in reversed(self.prediction_log) if p["home"] == home and p["away"] == away), None
        )

        comparison = None
        if match_pred is not None:
            actual_result = "home" if home_score > away_score else ("away" if away_score > home_score else "draw")
            predicted_p = {
                "home": match_pred["pred_p_home"],
                "draw": match_pred["pred_p_draw"],
                "away": match_pred["pred_p_away"],
            }[actual_result]
            xg_error_home = abs(match_pred["pred_xg_home"] - home_score)
            xg_error_away = abs(match_pred["pred_xg_away"] - away_score)

            comparison = dict(match_pred)
            comparison["actual_home_score"] = home_score
            comparison["actual_away_score"] = away_score
            comparison["actual_result"] = actual_result
            comparison["proba_assigned_to_actual"] = predicted_p
            comparison["xg_error"] = round((xg_error_home + xg_error_away) / 2, 2)
            match_pred["actual_home_score"] = home_score
            match_pred["actual_away_score"] = away_score
            match_pred["actual_result"] = actual_result
            match_pred["proba_assigned_to_actual"] = predicted_p
            match_pred["xg_error"] = comparison["xg_error"]

        old_home, old_away = round(self.ratings.get(home, config.ELO_BASE)), round(
            self.ratings.get(away, config.ELO_BASE)
        )
        home_form_pre = elo.get_form(self.form, home)
        away_form_pre = elo.get_form(self.form, away)
        h2h_edge_pre = elo.get_h2h_edge(self.h2h, home, away)

        # weight=1.0 : un résultat intégré en direct est par définition le
        # match le plus récent du dataset, donc pas de décroissance à lui
        # appliquer (voir elo.recency_weight).
        elo.apply_result(self.ratings, home, away, home_score, away_score, tournament, neutral, weight=1.0)

        self.history.append(
            {
                "home_team": home,
                "away_team": away,
                "home_elo": old_home,
                "away_elo": old_away,
                "home_form": home_form_pre,
                "away_form": away_form_pre,
                "h2h_edge": h2h_edge_pre,
                "neutral": neutral,
                "home_score": home_score,
                "away_score": away_score,
                "weight": 1.0,
                "date": pd.Timestamp.now(),
                "tournament": tournament,
            }
        )

        result_home, result_away = elo.match_result_value(home_score, away_score)
        elo.update_form(self.form, home, result_home)
        elo.update_form(self.form, away, result_away)
        elo.update_h2h(self.h2h, home, away, result_home)

        return comparison

    def retrain(self) -> None:
        """Réentraîne la régression de Poisson sur `history` mis à jour.
        À faire après plusieurs résultats (pas nécessairement à chaque match)."""
        from .poisson_model import train_model

        self.model = train_model(self.history)

    def performance_dashboard(self) -> pd.DataFrame | None:
        """Tableau des matchs déjà comparés (prédiction vs réalité)."""
        evaluated = [p for p in self.prediction_log if "actual_result" in p]
        if not evaluated:
            return None

        df = pd.DataFrame(evaluated)[
            [
                "home",
                "away",
                "actual_home_score",
                "actual_away_score",
                "pred_xg_home",
                "pred_xg_away",
                "proba_assigned_to_actual",
                "xg_error",
            ]
        ]
        df.columns = [
            "Domicile",
            "Extérieur",
            "Score réel D",
            "Score réel E",
            "xG prédit D",
            "xG prédit E",
            "Proba donnée au résultat réel",
            "Erreur xG moy.",
        ]
        return df
