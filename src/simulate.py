# -*- coding: utf-8 -*-
"""Simulation de matchs et de tournois.

Chaîne complète : xG (poisson_model, avec Dixon-Coles + h2h) -> tirage
Poisson -> score réel -> phase de groupes -> bracket FIFA -> Monte Carlo
sur N tournois.
"""

import math
import random

from . import config, elo
from .poisson_model import PoissonGoalModel


def poisson_sample(lam: float) -> int:
    """Algorithme de Knuth : tire un entier selon une loi de Poisson(lam)."""
    if lam <= 0:
        return 0
    target = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= target:
            break
    return k - 1


def simulate_match(
    model: PoissonGoalModel,
    ratings: dict,
    team_a: str,
    team_b: str,
    a_home: bool = False,
    b_home: bool = False,
    form: dict | None = None,
    h2h: dict | None = None,
    competition_weight: float = 1.0,
) -> tuple[int, int]:
    """Simule un score complet (ex: 2-1) entre deux équipes nommées, en
    tenant compte de l'ELO, de la forme récente, du h2h et des forces
    attaque/défense Dixon-Coles (via team_a/team_b, résolues par le modèle)."""
    form = form or {}
    h2h = h2h or {}
    elo_a, elo_b = ratings.get(team_a, config.ELO_BASE), ratings.get(team_b, config.ELO_BASE)
    form_a, form_b = elo.get_form(form, team_a), elo.get_form(form, team_b)
    h2h_edge = elo.get_h2h_edge(h2h, team_a, team_b)
    xg_a, xg_b = model.expected_goals(
        elo_a, elo_b, team_a, team_b, a_home, b_home, form_a, form_b, competition_weight, h2h_edge
    )
    return poisson_sample(xg_a), poisson_sample(xg_b)


def match_probabilities(
    model: PoissonGoalModel,
    ratings: dict,
    team_a: str,
    team_b: str,
    trials: int = 50_000,
    form: dict | None = None,
    h2h: dict | None = None,
    competition_weight: float = 1.0,
) -> dict:
    """Probabilités W/D/L + score le plus probable, par simulation Monte Carlo."""
    form = form or {}
    h2h = h2h or {}
    elo_a, elo_b = ratings.get(team_a, config.ELO_BASE), ratings.get(team_b, config.ELO_BASE)
    form_a, form_b = elo.get_form(form, team_a), elo.get_form(form, team_b)
    h2h_edge = elo.get_h2h_edge(h2h, team_a, team_b)
    xg_a, xg_b = model.expected_goals(
        elo_a, elo_b, team_a, team_b, form_a=form_a, form_b=form_b,
        competition_weight=competition_weight, h2h_edge=h2h_edge,
    )

    win_a = draw = win_b = 0
    score_freq: dict[str, int] = {}

    for _ in range(trials):
        ga, gb = poisson_sample(xg_a), poisson_sample(xg_b)
        key = f"{ga}-{gb}"
        score_freq[key] = score_freq.get(key, 0) + 1
        if ga > gb:
            win_a += 1
        elif ga < gb:
            win_b += 1
        else:
            draw += 1

    most_likely = max(score_freq.items(), key=lambda kv: kv[1])[0]

    return {
        "p_win_a": win_a / trials,
        "p_draw": draw / trials,
        "p_win_b": win_b / trials,
        "xg_a": round(xg_a, 2),
        "xg_b": round(xg_b, 2),
        "most_likely_score": most_likely,
    }


# ─────────────────────────────────────────────────────────────────────────
# Phase de groupes / bracket
# ─────────────────────────────────────────────────────────────────────────
def assign_thirds(thirds: list[str], third_groups: list[str]) -> list[str]:
    """Place les 8 meilleures 3es équipes dans les 8 slots (T, i), en
    évitant qu'une 3e affronte le vainqueur de son propre groupe."""
    remaining = list(range(len(thirds)))
    assigned = []
    for i in range(len(thirds)):
        winner_group = config.TSLOT_WINNER_GROUP[i]
        pick = next((ti for ti in remaining if third_groups[ti] != winner_group), remaining[0])
        assigned.append(thirds[pick])
        remaining.remove(pick)
    return assigned


def simulate_group(
    model: PoissonGoalModel, ratings: dict, group_teams: list[str], form: dict | None = None, h2h: dict | None = None
) -> list[tuple]:
    """Simule un groupe de 4 en round-robin, retourne le classement trié
    (points, diff. de buts, buts marqués)."""
    standings = {t: {"points": 0, "gf": 0, "ga": 0} for t in group_teams}
    n = len(group_teams)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = group_teams[i], group_teams[j]
            ga, gb = simulate_match(model, ratings, a, b, form=form, h2h=h2h)
            standings[a]["gf"] += ga
            standings[a]["ga"] += gb
            standings[b]["gf"] += gb
            standings[b]["ga"] += ga
            if ga > gb:
                standings[a]["points"] += 3
            elif gb > ga:
                standings[b]["points"] += 3
            else:
                standings[a]["points"] += 1
                standings[b]["points"] += 1
    ranked = sorted(
        group_teams,
        key=lambda t: (standings[t]["points"], standings[t]["gf"] - standings[t]["ga"], standings[t]["gf"]),
        reverse=True,
    )
    return [
        (t, standings[t]["points"], standings[t]["gf"] - standings[t]["ga"], standings[t]["gf"]) for t in ranked
    ]


def simulate_knockout(
    model: PoissonGoalModel, ratings: dict, team_a: str, team_b: str, form: dict | None = None, h2h: dict | None = None
) -> str:
    """Match à élimination directe. Égalité -> tirs au but, légèrement
    pondérés par l'écart ELO."""
    elo_a, elo_b = ratings.get(team_a, config.ELO_BASE), ratings.get(team_b, config.ELO_BASE)
    ga, gb = simulate_match(model, ratings, team_a, team_b, form=form, h2h=h2h)
    if ga > gb:
        return team_a
    if gb > ga:
        return team_b
    pen_edge = min(0.6, 0.5 + (elo_a - elo_b) / 2000)
    return team_a if random.random() < pen_edge else team_b


def simulate_full_tournament(
    model: PoissonGoalModel, ratings: dict, form: dict | None = None, h2h: dict | None = None
) -> dict:
    """Simule un tournoi complet (48 équipes) et retourne le parcours de
    chaque équipe."""
    reached = {name: "groups" for _, name, _, _ in config.WC2026_TEAMS}

    winners_by_group, runners_by_group, thirds_list, third_groups = {}, {}, [], []
    for g in config.GROUPS:
        standing = simulate_group(model, ratings, config.TEAMS_BY_GROUP[g], form=form, h2h=h2h)
        winners_by_group[g] = standing[0][0]
        runners_by_group[g] = standing[1][0]
        thirds_list.append(standing[2])
        third_groups.append(g)
        reached[standing[0][0]] = "r32"
        reached[standing[1][0]] = "r32"

    order = sorted(
        range(len(thirds_list)), key=lambda i: (thirds_list[i][1], thirds_list[i][2], thirds_list[i][3]), reverse=True
    )
    best8_idx = order[:8]
    best_thirds = [thirds_list[i][0] for i in best8_idx]
    best_third_groups = [third_groups[i] for i in best8_idx]
    thirds = assign_thirds(best_thirds, best_third_groups)
    for t in thirds:
        reached[t] = "r32"

    def resolve(slot):
        kind, key = slot
        if kind == "W":
            return winners_by_group[key]
        if kind == "R":
            return runners_by_group[key]
        return thirds[key]

    pool = []
    for slot_a, slot_b in config.R32_BRACKET:
        pool.append(resolve(slot_a))
        pool.append(resolve(slot_b))

    def knockout_round(teams, stage_label):
        winners = []
        for i in range(0, len(teams), 2):
            w = simulate_knockout(model, ratings, teams[i], teams[i + 1], form=form, h2h=h2h)
            winners.append(w)
            reached[w] = stage_label
        return winners

    r16 = knockout_round(pool, "r16")
    qf = knockout_round(r16, "qf")
    sf = knockout_round(qf, "sf")
    finalists = knockout_round(sf, "final")
    champion = knockout_round(finalists, "champion")[0]

    return {"champion": champion, "reached": reached}


def _resolve_known_or_simulate(
    model: PoissonGoalModel, ratings: dict, team_a: str, team_b: str,
    known_winners: dict | None, form: dict | None, h2h: dict | None,
) -> str:
    """Retourne le vainqueur connu (dans `known_winners`, clé cherchée
    dans les deux ordres) s'il existe, sinon simule le match."""
    if known_winners:
        key_ab, key_ba = f"{team_a} vs {team_b}", f"{team_b} vs {team_a}"
        if key_ab in known_winners:
            return known_winners[key_ab]
        if key_ba in known_winners:
            return known_winners[key_ba]
    return simulate_knockout(model, ratings, team_a, team_b, form=form, h2h=h2h)


def simulate_from_rd32(
    model: PoissonGoalModel,
    ratings: dict,
    known_rd32_winners: dict,
    form: dict | None = None,
    h2h: dict | None = None,
    known_r16_winners: dict | None = None,
    known_qf_winners: dict | None = None,
) -> dict:
    """Simule le tournoi depuis le RD32, en figeant les résultats déjà
    connus (`known_rd32_winners`, clé `"Home vs Away" -> vainqueur`).
    `known_r16_winners`/`known_qf_winners` : même principe pour les tours
    suivants, déjà en cours une fois le RD32 terminé — sans ça, un match
    déjà joué en réalité serait quand même re-simulé au hasard."""
    reached = {name: "eliminated" for _, name, _, _ in config.WC2026_TEAMS}

    rd32_winners = []
    for home, away in config.BRACKET_RD32:
        key = f"{home} vs {away}"
        winner = known_rd32_winners.get(key) or simulate_knockout(model, ratings, home, away, form=form, h2h=h2h)
        rd32_winners.append(winner)
        reached[winner] = "r16"

    def knockout_round(teams, stage_label, known_winners=None):
        winners = []
        for i in range(0, len(teams), 2):
            w = _resolve_known_or_simulate(model, ratings, teams[i], teams[i + 1], known_winners, form, h2h)
            winners.append(w)
            reached[w] = stage_label
        return winners

    rd16_winners = knockout_round(rd32_winners, "qf", known_winners=known_r16_winners)
    qf_winners = knockout_round(rd16_winners, "sf", known_winners=known_qf_winners)
    sf_winners = knockout_round(qf_winners, "final")
    champion = _resolve_known_or_simulate(model, ratings, sf_winners[0], sf_winners[1], None, form, h2h)
    reached[champion] = "champion"

    return {"champion": champion, "reached": reached}


# ─────────────────────────────────────────────────────────────────────────
# Monte Carlo
# ─────────────────────────────────────────────────────────────────────────
def monte_carlo(
    model: PoissonGoalModel,
    ratings: dict,
    n_simulations: int = config.N_SIMULATIONS,
    known_rd32_winners: dict | None = None,
    known_r16_winners: dict | None = None,
    known_qf_winners: dict | None = None,
    form: dict | None = None,
    h2h: dict | None = None,
) -> dict:
    """Lance N simulations complètes et retourne, pour chaque équipe, le
    nombre de fois où elle a atteint AU MOINS chaque stade (cascade : une
    équipe championne a par définition aussi atteint la finale, les demies,
    etc.)

    Si `known_rd32_winners` est fourni, simule depuis le RD32 (résultats
    de groupes déjà connus) plutôt qu'un tournoi complet from-scratch.
    `known_r16_winners`/`known_qf_winners` gèlent en plus les tours
    suivants déjà joués. `form`/`h2h` : snapshots statiques utilisés tels
    quels pour toute la durée d'une simulation (comme `ratings`, pas mis
    à jour en cours de tournoi simulé).
    """
    counts: dict[str, dict[str, int]] = {}

    for _ in range(n_simulations):
        if known_rd32_winners is not None:
            result = simulate_from_rd32(
                model, ratings, known_rd32_winners, form=form, h2h=h2h,
                known_r16_winners=known_r16_winners, known_qf_winners=known_qf_winners,
            )
            stage_order = config.STAGE_ORDER_FROM_RD32
        else:
            result = simulate_full_tournament(model, ratings, form=form, h2h=h2h)
            stage_order = config.STAGE_ORDER

        reached = result["reached"]
        for team, stage in reached.items():
            counts.setdefault(team, {})
            if stage not in stage_order:
                continue  # ex. "eliminated" : rien à cascader
            idx = stage_order.index(stage)
            for s in stage_order[: idx + 1]:
                counts[team][s] = counts[team].get(s, 0) + 1

    return counts
