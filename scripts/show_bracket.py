# -*- coding: utf-8 -*-
"""Génère le bracket visuel du tournoi (HTML) à partir de l'état sauvegardé
et des résultats connus (config.KNOWN_RD32_WINNERS / KNOWN_R16_WINNERS /
KNOWN_QF_WINNERS), et l'enregistre dans reports/bracket.html.

Usage :
    python -m scripts.show_bracket
"""

import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, state, visualize


def main():
    tracker = state.load_state()
    if tracker is None:
        print("Aucun état sauvegardé. Lance d'abord : python -m scripts.run_pipeline")
        return

    known_winners = visualize.known_winners_positional(
        config.KNOWN_RD32_WINNERS,
        config.KNOWN_R16_WINNERS,
        config.KNOWN_QF_WINNERS,
    )

    bracket_html = visualize.render_bracket_html(
        tracker.model, tracker.ratings, known_winners, form=tracker.form, h2h=tracker.h2h
    )

    full_html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>WC2026 — Bracket</title>
</head>
<body style="margin:0; padding:24px; background:#0E3A8C;">
{bracket_html}
</body>
</html>"""

    out_path = config.REPORTS_DIR / "bracket.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Bracket généré : {out_path}")

    try:
        webbrowser.open(f"file://{out_path.resolve()}")
    except Exception:
        pass
    print(f"Si le navigateur ne s'est pas ouvert automatiquement, ouvre ce lien à la main :")
    print(f"  file://{out_path.resolve()}")


if __name__ == "__main__":
    main()
