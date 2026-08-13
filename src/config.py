# -*- coding: utf-8 -*-
"""Configuration et constantes du projet WC2026 Predictions.

Toutes les valeurs "en dur" du notebook original vivent ici, en un seul
endroit, pour éviter d'avoir à les rechercher dans 1000 lignes de code
quand la FIFA publie un tirage au sort actualisé ou un résultat réel.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"

RESULTS_CSV = DATA_DIR / "results.csv"
FORMER_NAMES_CSV = DATA_DIR / "former_names.csv"
GOALSCORERS_CSV = DATA_DIR / "goalscorers.csv"
SHOOTOUTS_CSV = DATA_DIR / "shootouts.csv"

STATE_PATH = ROOT_DIR / "wc2026_model_state.pkl"

# ─────────────────────────────────────────────────────────────────────────
# ELO
# ─────────────────────────────────────────────────────────────────────────
ELO_BASE = 1000.0
HOME_ADVANTAGE = 75.0
ELO_DIFF_SCALE = 100.0  # normalisation des features pour la régression de Poisson

# ─────────────────────────────────────────────────────────────────────────
# Fenêtre d'entraînement & pondération temporelle
# ─────────────────────────────────────────────────────────────────────────
# Ne garder que les N dernières années de matchs. None = tout l'historique
# (depuis 1872).
#
# ⚠️ Désactivé par défaut : le backtest walk-forward (scripts/backtest.py)
# a montré que la fenêtre 10 ans + décroissance 2 ans faisait LÉGÈREMENT
# MOINS BIEN que l'historique complet (48.3% vs 47.6% de probabilité
# donnée au résultat réel sur 982 matchs de test, erreur xG 0.98 vs 1.03).
# L'ELO se remet déjà à jour à chaque match, ce qui capture une bonne
# partie de la "forme récente" sans qu'une pondération temporelle
# supplémentaire soit nécessaire — et couper l'historique perd des
# données utiles pour les équipes qui jouent peu de matchs officiels.
# Réactivable via --years / --half-life si tu veux retester d'autres
# valeurs (demi-vie plus longue, fenêtre plus large, etc.).
TRAINING_YEARS = None

# Demi-vie (en jours) de la pondération temporelle appliquée AU SEIN de la
# fenêtre d'entraînement : un match vieux de RECENCY_HALF_LIFE_DAYS pèse 2x
# moins qu'un match d'aujourd'hui. Affecte le K-factor ELO et le
# sample_weight de la régression de Poisson. None = pas de décroissance.
RECENCY_HALF_LIFE_DAYS = None

# ─────────────────────────────────────────────────────────────────────────
# Forme récente & poids de compétition (features Poisson additionnelles)
# ─────────────────────────────────────────────────────────────────────────
# Nombre de derniers matchs pris en compte pour la forme d'une équipe
# (moyenne glissante de résultats : victoire=1, nul=0.5, défaite=0).
FORM_WINDOW = 5

# Normalise k_factor() (20 à 60) en une feature de poids d'enjeu du
# tournoi entre ~0.33 (amical) et 1.0 (Coupe du Monde), pour la régression
# de Poisson — indépendant du "weight" de décroissance temporelle utilisé
# pour l'ELO/sample_weight.
COMPETITION_WEIGHT_MAX = 60.0


def competition_weight(tournament: str) -> float:
    return k_factor(tournament) / COMPETITION_WEIGHT_MAX


# ─────────────────────────────────────────────────────────────────────────
# Historique face-à-face (h2h)
# ─────────────────────────────────────────────────────────────────────────
# Nombre minimum de confrontations directes connues avant de faire
# confiance au h2h comme feature — sinon trop bruité pour les paires
# d'équipes qui se sont rarement affrontées (valeur neutre 0.0 sinon).
H2H_MIN_MATCHES = 3

# ─────────────────────────────────────────────────────────────────────────
# Dixon-Coles (forces attaque/défense séparées par équipe)
# ─────────────────────────────────────────────────────────────────────────
# Régularisation L2 de la régression de Poisson. Beaucoup plus forte que
# pour la version sans dummies d'équipe (alpha=1e-8) : avec ~332 équipes
# (664 colonnes attaque+défense), certaines n'ayant que quelques matchs,
# une régularisation faible ferait exploser leurs coefficients (overfitting
# sur un tout petit échantillon).
DIXON_COLES_ALPHA = 2.0

# Nombre minimum de matchs dans l'historique d'entraînement pour qu'une
# équipe ait son propre profil attaque/défense Dixon-Coles. En dessous,
# elle reste à 0.0 (neutre, l'ELO seul porte sa force) — sans ce seuil,
# les petites nations avec 2-3 matchs se calent sur du bruit (une victoire
# 5-0 contre un adversaire faible ressemble à une "attaque de classe
# mondiale" statistiquement, alors que ce n'est qu'un échantillon minuscule).
DIXON_COLES_MIN_MATCHES = 20

# ─────────────────────────────────────────────────────────────────────────
# Activation par défaut de chaque feature — décidé par backtest
# (scripts/backtest.py), pas à l'intuition. Sur 180 jours de test
# walk-forward (395 matchs) :
#   BASE (rien)         : 45.4% proba au résultat réel, xg_err 0.948
#   + forme seule       : 45.4% (neutre)
#   + enjeu seul        : 44.5% (moins bon)
#   + h2h seul          : 45.5% (légèrement mieux)
#   + Dixon-Coles seul  : 44.4% (nettement moins bon — l'ELO capture déjà
#                          la force globale d'une équipe ; les dummies
#                          attaque/défense n'ont plus grand-chose à
#                          expliquer et ajoutent surtout du bruit)
# D'où les défauts ci-dessous. Rejouer scripts/backtest.py régulièrement
# pour vérifier que ça reste vrai à mesure que l'historique s'enrichit.
USE_FORM = True
USE_COMPETITION_WEIGHT = False
USE_H2H = True
USE_DIXON_COLES = False


def k_factor(tournament: str) -> float:
    """Facteur d'amplification ELO selon l'importance de la compétition."""
    t = tournament.lower()
    if "fifa world cup" in t and "qualif" not in t:
        return 60
    if any(
        c in t
        for c in (
            "copa america",
            "uefa euro",
            "africa cup",
            "afc asian cup",
            "gold cup",
            "concacaf nations",
        )
    ):
        return 50
    if "qualif" in t:
        return 40
    if "nations league" in t or "confederation" in t:
        return 35
    return 20  # amical


# ─────────────────────────────────────────────────────────────────────────
# Normalisation des noms d'équipes disparues / non couvertes par
# former_names.csv
# ─────────────────────────────────────────────────────────────────────────
EXTRA_NAME_MAP = {
    "German DR": "Germany",
    "Yugoslavia": "Serbia",
    "Czechoslovakia": "Czech Republic",
    "CIS": "Russia",
    "Saarland": "Germany",
    "Bohemia": "Czech Republic",
}

# ─────────────────────────────────────────────────────────────────────────
# 48 équipes qualifiées — tirage officiel FIFA du 5 décembre 2025
# (csv_name, display_name, code_iso2/3, groupe)
# ─────────────────────────────────────────────────────────────────────────
WC2026_TEAMS = [
    ("Mexico", "Mexico", "MEX", "A"), ("South Africa", "South Africa", "RSA", "A"),
    ("South Korea", "South Korea", "KOR", "A"), ("Czech Republic", "Czech Republic", "CZE", "A"),
    ("Canada", "Canada", "CAN", "B"), ("Bosnia and Herzegovina", "Bosnia and Herzegovina", "BIH", "B"),
    ("Qatar", "Qatar", "QAT", "B"), ("Switzerland", "Switzerland", "SUI", "B"),
    ("Brazil", "Brazil", "BRA", "C"), ("Morocco", "Morocco", "MAR", "C"),
    ("Haiti", "Haiti", "HAI", "C"), ("Scotland", "Scotland", "SCO", "C"),
    ("United States", "United States", "USA", "D"), ("Paraguay", "Paraguay", "PAR", "D"),
    ("Australia", "Australia", "AUS", "D"), ("Turkey", "Turkey", "TUR", "D"),
    ("Germany", "Germany", "GER", "E"), ("Curaçao", "Curaçao", "CUW", "E"),
    ("Ivory Coast", "Ivory Coast", "CIV", "E"), ("Ecuador", "Ecuador", "ECU", "E"),
    ("Netherlands", "Netherlands", "NED", "F"), ("Japan", "Japan", "JPN", "F"),
    ("Sweden", "Sweden", "SWE", "F"), ("Tunisia", "Tunisia", "TUN", "F"),
    ("Belgium", "Belgium", "BEL", "G"), ("Egypt", "Egypt", "EGY", "G"),
    ("Iran", "Iran", "IRN", "G"), ("New Zealand", "New Zealand", "NZL", "G"),
    ("Spain", "Spain", "ESP", "H"), ("Cape Verde", "Cape Verde", "CPV", "H"),
    ("Saudi Arabia", "Saudi Arabia", "KSA", "H"), ("Uruguay", "Uruguay", "URU", "H"),
    ("France", "France", "FRA", "I"), ("Senegal", "Senegal", "SEN", "I"),
    ("Iraq", "Iraq", "IRQ", "I"), ("Norway", "Norway", "NOR", "I"),
    ("Argentina", "Argentina", "ARG", "J"), ("Algeria", "Algeria", "ALG", "J"),
    ("Austria", "Austria", "AUT", "J"), ("Jordan", "Jordan", "JOR", "J"),
    ("Portugal", "Portugal", "POR", "K"), ("DR Congo", "DR Congo", "COD", "K"),
    ("Uzbekistan", "Uzbekistan", "UZB", "K"), ("Colombia", "Colombia", "COL", "K"),
    ("England", "England", "ENG", "L"), ("Croatia", "Croatia", "CRO", "L"),
    ("Ghana", "Ghana", "GHA", "L"), ("Panama", "Panama", "PAN", "L"),
]

GROUPS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
TEAM_GROUP = {name: group for _, name, _, group in WC2026_TEAMS}
TEAMS_BY_GROUP = {g: [name for _, name, _, gr in WC2026_TEAMS if gr == g] for g in GROUPS}

TEAM_ISO = {
    "Mexico": "mx", "South Africa": "za", "South Korea": "kr", "Czech Republic": "cz",
    "Canada": "ca", "Bosnia and Herzegovina": "ba", "Qatar": "qa", "Switzerland": "ch",
    "Brazil": "br", "Morocco": "ma", "Haiti": "ht", "Scotland": "gb-sct",
    "United States": "us", "Paraguay": "py", "Australia": "au", "Turkey": "tr",
    "Germany": "de", "Ivory Coast": "ci", "Ecuador": "ec",
    "Netherlands": "nl", "Japan": "jp", "Sweden": "se", "Tunisia": "tn",
    "Belgium": "be", "Egypt": "eg", "Iran": "ir", "New Zealand": "nz",
    "Spain": "es", "Cape Verde": "cv", "Saudi Arabia": "sa", "Uruguay": "uy",
    "France": "fr", "Senegal": "sn", "Iraq": "iq", "Norway": "no",
    "Argentina": "ar", "Algeria": "dz", "Austria": "at", "Jordan": "jo",
    "Portugal": "pt", "DR Congo": "cd", "Uzbekistan": "uz", "Colombia": "co",
    "England": "gb-eng", "Croatia": "hr", "Ghana": "gh", "Panama": "pa",
}

# ─────────────────────────────────────────────────────────────────────────
# Bracket FIFA officiel — 16 matchs de seizièmes (R32)
# ("W", "A") = vainqueur groupe A, ("R", "C") = 2e groupe C,
# ("T", 0) = une des 8 meilleures équipes classées 3e
# ─────────────────────────────────────────────────────────────────────────
R32_BRACKET = [
    (("W", "A"), ("T", 0)), (("R", "C"), ("R", "D")),
    (("W", "E"), ("T", 1)), (("W", "G"), ("R", "H")),
    (("W", "B"), ("T", 2)), (("R", "F"), ("R", "L")),
    (("W", "I"), ("T", 3)), (("W", "K"), ("R", "J")),
    (("W", "C"), ("T", 4)), (("R", "A"), ("R", "B")),
    (("W", "F"), ("T", 5)), (("W", "H"), ("R", "G")),
    (("W", "D"), ("T", 6)), (("R", "E"), ("R", "I")),
    (("W", "J"), ("T", 7)), (("W", "L"), ("R", "K")),
]

# Groupe dont le vainqueur occupe l'autre côté de chaque slot ("T", i),
# pour éviter qu'une 3e place affronte le vainqueur de son propre groupe
TSLOT_WINNER_GROUP = {0: "A", 1: "E", 2: "B", 3: "I", 4: "C", 5: "F", 6: "D", 7: "L"}

# Bracket réel du RD32 (16 matchs, ordre FIFA) — à garder synchronisé avec
# le vrai tirage une fois la phase de groupes terminée.
BRACKET_RD32 = [
    ("France", "Sweden"), ("Germany", "Paraguay"),
    ("South Africa", "Canada"), ("Netherlands", "Morocco"),
    ("Portugal", "Croatia"), ("Spain", "Austria"),
    ("United States", "Bosnia and Herzegovina"), ("Belgium", "Senegal"),
    ("Brazil", "Japan"), ("Ivory Coast", "Norway"),
    ("Mexico", "Ecuador"), ("England", "DR Congo"),
    ("Argentina", "Cape Verde"), ("Australia", "Egypt"),
    ("Switzerland", "Algeria"), ("Colombia", "Ghana"),
]

# Résultats RD32 déjà connus (clé "Home vs Away" -> vainqueur), à mettre à
# jour au fur et à mesure que les matchs se jouent. Utilisé par
# simulate.simulate_from_rd32 / monte_carlo pour geler ce qui est déjà
# joué et ne laisser le hasard que sur le reste du tableau.
KNOWN_RD32_WINNERS = {
    "South Africa vs Canada": "Canada",
    "Brazil vs Japan": "Brazil",
    "Germany vs Paraguay": "Paraguay",
    "Netherlands vs Morocco": "Morocco",
    "France vs Sweden": "France",
    "Ivory Coast vs Norway": "Norway",
    "Mexico vs Ecuador": "Mexico",
    "England vs DR Congo": "England",
    "Belgium vs Senegal": "Belgium",
    "United States vs Bosnia and Herzegovina": "United States",
    "Spain vs Austria": "Spain",
    "Portugal vs Croatia": "Portugal",
    "Switzerland vs Algeria": "Switzerland",
    "Australia vs Egypt": "Egypt",  # tirs au but, 1-1 après prolongation
    "Argentina vs Cape Verde": "Argentina",
    "Colombia vs Ghana": "Colombia",
    # RD32 complet — les 16 vainqueurs sont connus.
}

# Résultats R16 (8es de finale) déjà connus (clé "Team1 vs Team2" dans
# n'importe quel ordre -> vainqueur), à mettre à jour au fur et à mesure.
# Nécessaire pour que --from-rd32 arrête de re-simuler au hasard un match
# déjà joué en réalité (une équipe éliminée ne doit plus apparaître avec
# des chances de titre).
KNOWN_R16_WINNERS = {
    "Mexico vs England": "England",
    "Canada vs Morocco": "Morocco",
    "Paraguay vs France": "France",
    "Brazil vs Norway": "Norway",
    "Portugal vs Spain": "Spain",
    "United States vs Belgium": "Belgium",
    "Argentina vs Egypt": "Argentina",
    "Switzerland vs Colombia": "Switzerland",
}

# Résultats QF (quarts de finale) déjà connus, même principe. Les 4 quarts
# réels : France vs Morocco, Spain vs Belgium, Norway vs England,
# Argentina vs Switzerland.
KNOWN_QF_WINNERS = {
    "France vs Morocco": "France",
    "Spain vs Belgium": "Spain",
    "Norway vs England": "England",
    "Argentina vs Switzerland": "Argentina",
}

# ─────────────────────────────────────────────────────────────────────────
# Monte Carlo
# ─────────────────────────────────────────────────────────────────────────
N_SIMULATIONS = 10_000
STAGE_ORDER = ["groups", "r32", "r16", "qf", "sf", "final", "champion"]

# Ordre des stades pour simulate_from_rd32 : pas de "groups"/"r32" (déjà
# joués et gelés), et "eliminated" n'est pas un stade "atteint" mais l'état
# par défaut d'une équipe déjà sortie — donc absent de la cascade.
STAGE_ORDER_FROM_RD32 = ["r16", "qf", "sf", "final", "champion"]
