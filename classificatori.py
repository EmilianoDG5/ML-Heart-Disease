import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# ==============================
# 1. Caricamento dataset
# ==============================
df = pd.read_csv("heart.csv")

X = df.drop("output", axis=1)
y = df["output"]

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 2. Modelli
# ==============================
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

# ==============================
# 3. Addestramento e valutazione
# ==============================
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "fpr": roc_curve(y_test, y_prob)[0],
        "tpr": roc_curve(y_test, y_prob)[1],
        "auc": auc(*roc_curve(y_test, y_prob)[:2])
    }

# ==============================
# 4. Stampa metriche
# ==============================
print("\n=== RISULTATI MODELLI ===\n")
for name, r in results.items():
    print(f"{name}")
    print(f"Accuracy : {r['accuracy']:.3f}")
    print(f"Precision: {r['precision']:.3f}")
    print(f"Recall   : {r['recall']:.3f}")
    print(f"F1-score : {r['f1']:.3f}\n")

# ==============================
# 5. Matrici di confusione
# ==============================
for name, r in results.items():
    plt.figure(figsize=(5,4))
    sns.heatmap(r["conf_matrix"], annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matrice di Confusione - {name}")
    plt.xlabel("Predetto")
    plt.ylabel("Reale")
    plt.savefig(f"confusion_matrix_{name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()

# ==============================
# 6. ROC Curve
# ==============================
plt.figure(figsize=(7,6))
for name, r in results.items():
    plt.plot(r["fpr"], r["tpr"], label=f"{name} (AUC = {r['auc']:.2f})")

plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Confronto Modelli")
plt.legend()
plt.savefig("roc_curve_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
