# -*- coding: utf-8 -*-
"""Génère le graphique du classement ELO actuel à partir de l'état sauvegardé.

Usage :
    python -m scripts.show_elo
    python -m scripts.show_elo --top 15
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import state, visualize


def main():
    parser = argparse.ArgumentParser(description="Génère le graphique du classement ELO.")
    parser.add_argument("--top", type=int, default=20, help="Nombre d'équipes affichées (défaut 20)")
    args = parser.parse_args()

    tracker = state.load_state()
    if tracker is None:
        print("Aucun état sauvegardé. Lance d'abord : python -m scripts.run_pipeline")
        return

    path = visualize.plot_elo_ranking(tracker.ratings, top_n=args.top)
    print(f"Graphique enregistré : {path}")


if __name__ == "__main__":
    main()
