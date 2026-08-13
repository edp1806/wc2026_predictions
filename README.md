# 🏆 WC2026 Predictions — ELO + Régression de Poisson + Monte Carlo

Modèle de machine learning pour prédire la Coupe du Monde 2026 : classement
ELO de 332 équipes, régression de Poisson pour les buts attendus (xG),
simulation de scores réalistes, et Monte Carlo sur le tournoi complet
(phase de groupes + bracket FIFA officiel à 32).

Réécrit en pipeline Python modulaire à partir d'un notebook Google Colab
(`wc2026_poisson_model.py`, ~1100 lignes en un seul fichier) pour être
utilisable en local, testable, et réentraînable au fil du tournoi sans
tout relancer depuis zéro à chaque fois.

## Pourquoi prédire des buts plutôt que Victoire/Nul/Défaite

Une première version utilisait un classifieur à 3 classes (Win/Draw/Loss).
Le football est un sport à faible score — la différence entre une victoire
1-0 et 3-0 compte, et un classifieur perd cette information (avec en plus
du mal à prédire les nuls). Une régression de Poisson prédit un nombre de
buts par équipe, dont on peut ensuite tirer un score réaliste (ex. 2-1) —
plus fidèle à la réalité du jeu.

```
1. ELO             → force de chaque équipe
        ↓
2. RÉGRESSION DE POISSON  → buts attendus (xG) de chaque équipe
        ↓
3. TIRAGE POISSON  → score réel simulé (ex. 2-1) à partir du xG
        ↓
4. TOURNOI COMPLET → 12 groupes de 4 + bracket FIFA officiel à 32
        ↓
5. MONTE CARLO     → 10 000 tournois simulés → probabilités de titre
```

## Structure du projet

```
wc2026-predictions/
├── src/
│   ├── config.py          # constantes : ELO, 48 équipes, groupes, bracket FIFA
│   ├── data_loader.py     # chargement + normalisation des noms d'équipes
│   ├── elo.py              # système ELO (K-factor, avantage terrain, historique)
│   ├── poisson_model.py    # régression de Poisson (le cœur du ML)
│   ├── simulate.py         # tirage Poisson, groupes, bracket, Monte Carlo
│   ├── tracking.py         # log_prediction / update_with_result / dashboard
│   ├── state.py            # persistance locale (pickle) entre deux sessions
│   └── visualize.py        # graphiques + rendu HTML du bracket
├── scripts/
│   ├── run_pipeline.py     # entraîne le modèle + lance Monte Carlo
│   ├── predict_match.py    # prédit un match précis
│   ├── update_result.py    # log une prédiction / intègre un résultat réel
│   └── backtest.py         # compare deux configurations en walk-forward
├── data/                   # results.csv, former_names.csv, goalscorers.csv, shootouts.csv
├── reports/                # graphiques générés (.png)
└── requirements.txt
```

## Fenêtre d'entraînement & forme récente

Le modèle utilise **tout l'historique disponible** par défaut (1872 →
aujourd'hui, ~49 500 matchs), sans pondération temporelle.

Une fenêtre d'entraînement réduite (`TRAINING_YEARS`) et une décroissance
temporelle du K-factor (`RECENCY_HALF_LIFE_DAYS`) sont disponibles dans
`config.py` et via `--years` / `--half-life`, mais **désactivées par
défaut** : un backtest walk-forward (`scripts/backtest.py`) a montré que
la combinaison "10 ans + décroissance 2 ans" faisait légèrement **moins**
bien que l'historique complet (48.3% vs 47.6% de probabilité donnée au
résultat réel sur 982 matchs de test, erreur xG 0.98 vs 1.03 but/équipe).
L'ELO se remet déjà à jour à chaque match, ce qui capture une bonne partie
de la forme récente sans pondération supplémentaire — et couper
l'historique retire des données utiles pour les équipes qui jouent peu de
matchs officiels.

```bash
# Défauts (tout l'historique, pas de décroissance)
python -m scripts.run_pipeline --fresh

# Retester d'autres réglages si besoin
python -m scripts.run_pipeline --fresh --years 10 --half-life 730
python -m scripts.backtest --test-days 365   # comparer objectivement avant/après
```

## Le modèle ELO

- Base ELO = 1000 (décalage arbitraire, ne change rien aux probabilités)
- K-factor par type de tournoi : Coupe du Monde = 60, continentaux
  (Copa América, Euro, CAN, Coupe d'Asie...) = 50, qualifications = 40,
  Ligue des Nations = 35, amical = 20
- Multiplicateur d'écart de buts plafonné : ≤1 but → x1.0, 2 buts → x1.5,
  3+ buts → x1.75
- Avantage du terrain : +75 ELO si le match n'est pas sur terrain neutre

Un snapshot de l'ELO de chaque équipe est conservé **avant** chaque match
(`elo.compute_elo_history`) — c'est exactement ce qu'utilise la régression
de Poisson comme feature d'entraînement, en respectant l'ordre chronologique
(pas de fuite d'information depuis le futur).

## La régression de Poisson

```
log(buts attendus) = β0 + β1 × (Δ ELO / 100) + β2 × domicile
```

Les coefficients sont appris par maximum de vraisemblance sur ~49 000
matchs historiques (1872 → aujourd'hui). Chaque match donne deux lignes
d'entraînement (une par équipe, avec l'écart ELO de son point de vue) pour
que le modèle apprenne une seule relation valable aussi bien côté favori
que côté outsider.

## Usage

```bash
pip install -r requirements.txt

# 1. Entraîne le modèle depuis les CSV et lance 10 000 simulations
python -m scripts.run_pipeline --fresh

# 2. Les runs suivants réutilisent l'état sauvegardé (rapide, pas de
#    recalcul de l'ELO sur 49k matchs à chaque fois)
python -m scripts.run_pipeline --simulations 20000

# 3. Prédire un match précis
python -m scripts.predict_match France Brazil

# 4. Avant un vrai match : figer ce que le modèle pensait
python -m scripts.update_result --log France Brazil

# 5. Après le match : intégrer le score réel, comparer à la prédiction,
#    réentraîner la régression de Poisson
python -m scripts.update_result France Brazil 2 1 --retrain
```

## Boucle d'amélioration continue

Le workflow recommandé pendant le tournoi :

1. **Avant** un match → `update_result.py --log` fige la prédiction du modèle
2. **Après** le match → `update_result.py <home> <away> <score> <score>`
   met à jour l'ELO et mesure l'écart entre prédiction et réalité
3. Tous les 4-5 matchs → `--retrain` réentraîne la régression de Poisson
   sur les données fraîches
4. Relancer `run_pipeline.py` pour des probabilités de titre à jour
5. `tracker.performance_dashboard()` (module `tracking.py`) pour suivre si
   le modèle devient plus ou moins précis au fil du tournoi — une
   probabilité moyenne donnée au résultat réel proche de 33% = pas mieux
   que le hasard, proche de 100% = modèle parfait

## Sources de données

- `results.csv` — ~49 000 matchs internationaux, 1872 → aujourd'hui
- `former_names.csv` — table des anciens noms de pays (Dahomey → Bénin,
  Haute-Volta → Burkina Faso...), complétée manuellement pour les cas non
  couverts (Yougoslavie → Serbie, Tchécoslovaquie → République Tchèque,
  RDA → Allemagne, CEI → Russie)
- `goalscorers.csv`, `shootouts.csv` — chargés mais non utilisés par le
  modèle actuel ; pistes d'amélioration ci-dessous

## Limites connues et pistes d'amélioration

- **Scores indépendants** : les deux tirages Poisson (domicile/extérieur)
  sont indépendants, alors qu'en réalité si une équipe attaque plus l'autre
  défend plus. Le modèle **Dixon-Coles** corrige ce biais spécifiquement
  pour les scores faibles (0-0, 1-0, 0-1, 1-1).
- **Pas de xG réel** : tout est dérivé de l'ELO. Des données xG réelles
  par match (`goalscorers.csv` est chargé mais pas encore exploité)
  amélioreraient la précision.
- **Tirs au but simplifiés** : pondérés uniquement par l'écart ELO
  (`min(0.6, 0.5 + Δelo/2000)`), pas de modèle dédié aux penalties.
- **332 équipes traitées à égalité** : une équipe jamais rencontrée démarre
  à 1000 ELO (moyenne), ce qui peut sur- ou sous-estimer les débutants au
  très haut niveau (ex. Curaçao, première participation).

## Dépendances

- `pandas`, `numpy` — traitement des données
- `scikit-learn` — `PoissonRegressor`
- `matplotlib` — graphiques (`visualize.py`)

## License

MIT.
