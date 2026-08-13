# -*- coding: utf-8 -*-
"""Graphiques et rendu HTML du bracket.

Toutes les fonctions ici écrivent dans `reports/` plutôt que d'appeler
`plt.show()`, pour fonctionner aussi bien en script qu'en notebook.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import config
from .simulate import match_probabilities

COLORS_MAP = {
    "France": "#002395", "Argentina": "#75AADB", "Spain": "#AA151B",
    "England": "#CF081F", "Brazil": "#009C3B", "Germany": "#000000",
    "Netherlands": "#FF6600", "Portugal": "#006600", "Colombia": "#FCD116",
}


def plot_title_probabilities(
    counts: dict, n_simulations: int, top_n: int = 12, path: Path = None
) -> Path:
    """Barres horizontales des probabilités de titre (top N équipes)."""
    rows = sorted(counts.items(), key=lambda kv: kv[1].get("champion", 0), reverse=True)[:top_n]
    teams = [name for name, _ in rows][::-1]
    probs = [c.get("champion", 0) / n_simulations * 100 for _, c in rows][::-1]
    bar_colors = [COLORS_MAP.get(t, "#1a73e8") for t in teams]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(teams, probs, color=bar_colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, probs):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
            va="center", fontweight="bold", fontsize=10,
        )
    ax.set_xlabel("Probabilité de titre (%)", fontsize=12)
    ax.set_title(
        f"Probabilités Champion du Monde 2026\n(ELO + Poisson + Monte Carlo, {n_simulations:,} simulations)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, max(probs) * 1.25 if probs else 1)
    plt.tight_layout()

    path = path or (config.REPORTS_DIR / "title_probabilities.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def plot_elo_ranking(ratings: dict, top_n: int = 20, path: Path = None) -> Path:
    """Barres horizontales du classement ELO actuel (top N équipes)."""
    rows = sorted(ratings.items(), key=lambda kv: -kv[1])[:top_n]
    teams = [name for name, _ in rows][::-1]
    elos = [round(elo) for _, elo in rows][::-1]
    bar_colors = [COLORS_MAP.get(t, "#1a73e8") for t in teams]

    fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.35)))
    bars = ax.barh(teams, elos, color=bar_colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, elos):
        ax.text(
            bar.get_width() + 5, bar.get_y() + bar.get_height() / 2, f"{val}",
            va="center", fontweight="bold", fontsize=10,
        )
    ax.set_xlabel("ELO", fontsize=12)
    ax.set_title(f"Classement ELO — WC2026 (top {top_n})", fontsize=13, fontweight="bold")
    ax.set_xlim(min(elos) - 60, max(elos) * 1.05)
    plt.tight_layout()

    path = path or (config.REPORTS_DIR / "elo_ranking.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_prediction_accuracy(evaluated_df: pd.DataFrame, path: Path = None) -> Path:
    """Barres : probabilité donnée par le modèle au résultat réellement
    survenu, match par match (référence : 33% = hasard pur)."""
    match_labels = [f"{h[:3]}-{a[:3]}" for h, a in zip(evaluated_df["Domicile"], evaluated_df["Extérieur"])]
    probas = evaluated_df["Proba donnée au résultat réel"] * 100

    colors = ["#39FF6A" if p >= 50 else ("#FFD23F" if p >= 33 else "#FF4757") for p in probas]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(match_labels, probas, color=colors, edgecolor="white")
    ax.axhline(y=33.3, color="gray", linestyle="--", alpha=0.6, label="Hasard pur (33%)")
    ax.set_ylabel("Probabilité donnée au résultat réel (%)")
    ax.set_title("Précision du modèle, match par match")
    ax.set_ylim(0, 100)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = path or (config.REPORTS_DIR / "prediction_accuracy.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def known_winners_positional(
    known_rd32_winners: dict,
    known_r16_winners: dict | None = None,
    known_qf_winners: dict | None = None,
    known_final_winner: str | None = None,
) -> dict:
    """Convertit nos dicts KNOWN_*_WINNERS (clé "Team1 vs Team2" -> vainqueur,
    utilisés par simulate.simulate_from_rd32) vers le format positionnel
    attendu par render_bracket_html (clé "L-r32-0", "L-r16-0", etc.),
    en suivant l'ordre de config.BRACKET_RD32."""
    known_r16_winners = known_r16_winners or {}
    known_qf_winners = known_qf_winners or {}
    positional: dict[str, str] = {}

    def fill_side(pairs, prefix):
        for i, (home, away) in enumerate(pairs):
            winner = known_rd32_winners.get(f"{home} vs {away}")
            if winner:
                positional[f"{prefix}-r32-{i}"] = winner

        r16_pairs = []
        for i in range(0, len(pairs), 2):
            r16_pairs.append((positional.get(f"{prefix}-r32-{i}"), positional.get(f"{prefix}-r32-{i + 1}")))
        for i, (a, b) in enumerate(r16_pairs):
            if a and b:
                winner = known_r16_winners.get(f"{a} vs {b}") or known_r16_winners.get(f"{b} vs {a}")
                if winner:
                    positional[f"{prefix}-r16-{i}"] = winner

        qf_pairs = []
        for i in range(0, len(r16_pairs), 2):
            qf_pairs.append((positional.get(f"{prefix}-r16-{i}"), positional.get(f"{prefix}-r16-{i + 1}")))
        for i, (a, b) in enumerate(qf_pairs):
            if a and b:
                winner = known_qf_winners.get(f"{a} vs {b}") or known_qf_winners.get(f"{b} vs {a}")
                if winner:
                    positional[f"{prefix}-qf-{i}"] = winner

    fill_side(config.BRACKET_RD32[:8], "L")
    fill_side(config.BRACKET_RD32[8:], "R")

    if known_final_winner:
        positional["FINAL"] = known_final_winner

    return positional


def _flag_img(name: str) -> str:
    code = config.TEAM_ISO.get(name, "")
    if not code:
        return ""
    return (
        f'<img src="https://flagcdn.com/w20/{code}.png" '
        f'style="width:20px;height:14px;object-fit:cover;border-radius:2px;'
        f'margin-right:6px;vertical-align:middle">'
    )


def render_bracket_html(
    model, ratings: dict, known_winners: dict, champion: str | None = None,
    form: dict | None = None, h2h: dict | None = None,
) -> str:
    """Génère le HTML du bracket interactif (utilisable dans un notebook
    via IPython.display.HTML, ou écrit sur disque comme fragment web).

    `known_winners` : dict {match_id: équipe gagnante}, ex. "L-r32-0": "France".
    `form`/`h2h` : mêmes snapshots que ceux utilisés par predict_match.py —
    sans eux, les probabilités affichées ici ne correspondraient pas à
    celles du terminal.
    """
    left_r32 = [list(pair) for pair in config.BRACKET_RD32[:8]]
    right_r32 = [list(pair) for pair in config.BRACKET_RD32[8:]]

    def build_rounds(r32, prefix):
        rounds = {"r32": [], "r16": [], "qf": [], "sf": None}
        rounds["r32"] = [{"id": f"{prefix}-r32-{i}", "teams": pair} for i, pair in enumerate(r32)]
        for i in range(0, len(rounds["r32"]), 2):
            a = known_winners.get(rounds["r32"][i]["id"])
            b = known_winners.get(rounds["r32"][i + 1]["id"])
            rounds["r16"].append({"id": f"{prefix}-r16-{i // 2}", "teams": [a, b]})
        for i in range(0, len(rounds["r16"]), 2):
            a = known_winners.get(rounds["r16"][i]["id"])
            b = known_winners.get(rounds["r16"][i + 1]["id"])
            rounds["qf"].append({"id": f"{prefix}-qf-{i // 2}", "teams": [a, b]})
        sf_a = known_winners.get(rounds["qf"][0]["id"])
        sf_b = known_winners.get(rounds["qf"][1]["id"])
        rounds["sf"] = {"id": f"{prefix}-sf", "teams": [sf_a, sf_b]}
        return rounds

    left = build_rounds(left_r32, "L")
    right = build_rounds(right_r32, "R")

    def get_probs(a, b):
        """Probabilité de QUALIFICATION (pas juste de victoire) : le nul
        est résolu par tirs au but avec le même léger avantage ELO que
        simulate.simulate_knockout, pour que les deux chiffres somment à
        100% comme dans un vrai match à élimination directe."""
        if not a or not b:
            return None
        try:
            r = match_probabilities(model, ratings, a, b, trials=5000, form=form, h2h=h2h)
            elo_a, elo_b = ratings.get(a, config.ELO_BASE), ratings.get(b, config.ELO_BASE)
            pen_edge = min(0.6, 0.5 + (elo_a - elo_b) / 2000)
            qual_a = r["p_win_a"] + r["p_draw"] * pen_edge
            qual_b = r["p_win_b"] + r["p_draw"] * (1 - pen_edge)
            return [round(qual_a, 2), round(qual_b, 2)]
        except Exception:
            return None

    def match_html(match_id, teams, probs):
        a, b = teams
        winner = known_winners.get(match_id)

        def slot(team, prob):
            if not team:
                return '<div style="padding:6px 10px;color:#999;font-size:12px;">?</div>'
            bold = "font-weight:700;" if team == winner else ""
            prob_txt = f'<span style="float:right;color:#888;font-size:11px;">{prob:.0%}</span>' if prob else ""
            return f'<div style="padding:6px 10px;font-size:12px;{bold}">{_flag_img(team)}{team}{prob_txt}</div>'

        border = "2px solid #FFD23F" if winner else "1px solid #ddd"
        prob_a = probs[0] if probs else None
        prob_b = probs[1] if probs else None
        return (
            f'<div style="border:{border};border-radius:6px;overflow:hidden;width:180px;'
            f'background:white;box-shadow:0 1px 4px rgba(0,0,0,0.12);margin:4px auto;">'
            f'{slot(a, prob_a)}<div style="border-top:1px solid #eee;">{slot(b, prob_b)}</div></div>'
        )

    def col_html(matches, label):
        cells = ""
        for m in matches:
            probs = None if known_winners.get(m["id"]) else get_probs(m["teams"][0], m["teams"][1])
            cells += f'<div style="margin:8px 0;">{match_html(m["id"], m["teams"], probs)}</div>'
        return (
            '<div style="display:flex;flex-direction:column;justify-content:space-around;min-height:100%;">'
            f'<div style="text-align:center;font-size:10px;font-weight:700;color:rgba(255,255,255,0.6);'
            f'text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">{label}</div>{cells}</div>'
        )

    left_html = (
        col_html(left["r32"], "RD32") + col_html(left["r16"], "RD16")
        + col_html(left["qf"], "QF") + col_html([left["sf"]], "SF")
    )
    right_html = (
        col_html([right["sf"]], "SF") + col_html(right["qf"], "QF")
        + col_html(right["r16"], "RD16") + col_html(right["r32"], "RD32")
    )

    champ_name = champion or "?"
    champ_flag = _flag_img(champion) if champion else ""
    champ_bg = "#FFD23F" if champion else "rgba(255,255,255,0.1)"
    champ_color = "#0E2A5C" if champion else "rgba(255,255,255,0.4)"

    final_sf_a = known_winners.get(left["sf"]["id"])
    final_sf_b = known_winners.get(right["sf"]["id"])
    final_probs = None if known_winners.get("FINAL") else get_probs(final_sf_a, final_sf_b)
    final_html = match_html("FINAL", [final_sf_a, final_sf_b], final_probs)

    center_html = f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:0 24px;gap:16px;">
      <div style="font-size:56px;filter:drop-shadow(0 4px 12px rgba(0,0,0,0.4));">🏆</div>
      {final_html}
      <div style="background:{champ_bg};color:{champ_color};padding:10px 20px;border-radius:8px;
                  font-weight:700;font-size:14px;text-transform:uppercase;letter-spacing:2px;
                  text-align:center;min-width:160px;">{champ_flag}{champ_name}</div>
    </div>"""

    return f"""
    <div style="background:#0E3A8C;border-radius:16px;padding:28px;font-family:Oswald,Arial,sans-serif;overflow-x:auto;">
      <h2 style="color:white;text-align:center;font-size:18px;font-weight:700;text-transform:uppercase;
                 letter-spacing:4px;margin-bottom:24px;">🏆 Coupe du Monde 2026</h2>
      <div style="display:flex;align-items:center;justify-content:center;gap:0;min-width:1400px;">
        <div style="display:flex;gap:12px;align-items:center;">{left_html}</div>
        {center_html}
        <div style="display:flex;gap:12px;align-items:center;">{right_html}</div>
      </div>
    </div>
    <link href="https://fonts.googleapis.com/css2?family=Oswald:wght@600&display=swap" rel="stylesheet">
    """
