# -*- coding: utf-8 -*-
"""Sauvegarde / rechargement de l'état du modèle entre deux sessions.

Dans le notebook Colab original, l'état était pickled sur Google Drive.
Ici, il est simplement pickled dans le repo (`wc2026_model_state.pkl`,
à ajouter au .gitignore si tu ne veux pas versionner l'état).

⚠️ Le CSV `results.csv` ne sert qu'à l'initialisation (première ELO calculée
depuis zéro). Une fois qu'un état existe, ne le régénère pas depuis le CSV :
charge-le avec `load_state()` pour ne pas perdre les résultats déjà intégrés
via `PredictionTracker.update_with_result()`.
"""

import pickle
from pathlib import Path

from . import config
from .poisson_model import PoissonGoalModel
from .tracking import PredictionTracker


def save_state(tracker: PredictionTracker, path: Path = config.STATE_PATH) -> None:
    state = {
        "ratings": tracker.ratings,
        "form": tracker.form,
        "h2h": tracker.h2h,
        "history": tracker.history,
        "prediction_log": tracker.prediction_log,
        "model_intercept": tracker.model.intercept,
        "model_elo_coef": tracker.model.elo_coef,
        "model_home_coef": tracker.model.home_coef,
        "model_form_coef": tracker.model.form_coef,
        "model_competition_coef": tracker.model.competition_coef,
        "model_h2h_coef": tracker.model.h2h_coef,
        "model_attack": tracker.model.attack,
        "model_defense": tracker.model.defense,
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_state(path: Path = config.STATE_PATH) -> PredictionTracker | None:
    """Retourne un PredictionTracker reconstruit depuis le pickle, ou None
    si aucun état sauvegardé n'existe."""
    if not path.exists():
        return None

    with open(path, "rb") as f:
        state = pickle.load(f)

    model = PoissonGoalModel(
        intercept=state["model_intercept"],
        elo_coef=state["model_elo_coef"],
        home_coef=state["model_home_coef"],
        form_coef=state.get("model_form_coef", 0.0),
        competition_coef=state.get("model_competition_coef", 0.0),
        h2h_coef=state.get("model_h2h_coef", 0.0),
        attack=state.get("model_attack", {}),
        defense=state.get("model_defense", {}),
        sklearn_model=None,  # non sérialisé — reconstruit via retrain() si besoin
    )
    return PredictionTracker(
        ratings=state["ratings"],
        history=state["history"],
        model=model,
        form=state.get("form", {}),
        h2h=state.get("h2h", {}),
        prediction_log=state["prediction_log"],
    )
