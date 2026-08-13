import sys
sys.path.insert(0, ".")
from src import state

tracker = state.load_state()
if tracker is None:
    print("Aucun état trouvé.")
    sys.exit(1)

print(f"Avant correction : Mexico={tracker.ratings.get('Mexico')}, England={tracker.ratings.get('England')}")
print(f"Matchs dans l'historique : {len(tracker.history)}")

# Restaure les ELO à leur état juste après la PREMIÈRE application
# (valeurs affichées dans ta première commande, avant le doublon)
tracker.ratings["Mexico"] = 1531.0
tracker.ratings["England"] = 1628.0

# Retire la dernière occurrence Mexico-England de l'historique (le doublon)
idx_to_remove = None
for i in range(len(tracker.history) - 1, -1, -1):
    m = tracker.history[i]
    if m.get("home_team") == "Mexico" and m.get("away_team") == "England":
        idx_to_remove = i
        break

if idx_to_remove is not None:
    del tracker.history[idx_to_remove]
    print(f"Doublon retiré de l'historique (index {idx_to_remove}).")
else:
    print("Aucun doublon trouvé dans l'historique (déjà propre ?).")

print(f"Matchs dans l'historique après correction : {len(tracker.history)}")

tracker.retrain()
state.save_state(tracker)

print(f"Après correction : Mexico={tracker.ratings.get('Mexico'):.0f}, England={tracker.ratings.get('England'):.0f}")
print("État sauvegardé.")
