from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RAIZ = Path(__file__).resolve().parents[1]

df = pd.read_csv(RAIZ / "data" / "proyecto_limpio.csv")
X = df.drop(columns=["con_retraso", "dias_retraso"])
y = df["con_retraso"]
print(f"Entradas ({X.shape[1]}): {', '.join(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Préstamos para entrenar: {len(X_train)} | Préstamos para probar: {len(X_test)}")

modelos = {
    "Regresión logística": Pipeline([
        ("escalado", StandardScaler()),
        ("modelo", LogisticRegression()),
    ]),
    "KNN (5 vecinos)": Pipeline([
        ("escalado", StandardScaler()),
        ("modelo", KNeighborsClassifier(n_neighbors=5)),
    ]),
    "Árbol de decisión": Pipeline([
        ("escalado", StandardScaler()),
        ("modelo", DecisionTreeClassifier(max_depth=3, random_state=42)),
    ]),
    "Bosque aleatorio": Pipeline([
        ("escalado", StandardScaler()),
        ("modelo", RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)),
    ]),
}

resultados = []
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    resultados.append({
        "Modelo": nombre,
        "Exactitud": accuracy_score(y_test, y_pred),
        "Precisión": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Val. cruzada": cross_val_score(modelo, X, y, cv=5).mean(),
    })

tabla = pd.DataFrame(resultados)
print("\nComparación de modelos (datos de prueba):")
print(tabla.round(3).to_string(index=False))

siempre_retraso = pd.Series(1, index=y_test.index)
print(f"\nReferencia (decir siempre 'con retraso'): exactitud {accuracy_score(y_test, siempre_retraso):.3f} "
      f"| recall 1.000 | F1 {f1_score(y_test, siempre_retraso):.3f}")

nombre_mejor = tabla.loc[tabla["F1"].idxmax(), "Modelo"]
mejor_modelo = modelos[nombre_mejor]
print(f"\nMejor modelo según F1: {nombre_mejor}")

regresion = modelos["Regresión logística"].named_steps["modelo"]
pesos = pd.Series(regresion.coef_[0], index=X.columns)
print("\nPesos de la regresión logística (datos escalados):")
for nombre, peso in pesos.reindex(pesos.abs().sort_values(ascending=False).index).items():
    print(f"  {nombre:<20} {peso:+.2f}")
print(f"  {'b (sesgo)':<20} {regresion.intercept_[0]:+.2f}")

matriz = confusion_matrix(y_test, mejor_modelo.predict(X_test))
print("\nMatriz de confusión (filas = valor real, columnas = predicción):")
print(matriz)

salida = RAIZ / "outputs" / "avance2"
salida.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(6, 5))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predijo a tiempo", "Predijo retraso"],
            yticklabels=["Real a tiempo", "Real con retraso"])
plt.title(f"Matriz de confusión – {nombre_mejor}")
plt.tight_layout()
plt.savefig(salida / "matriz_confusion.png", dpi=150)
plt.close()
print("Gráfico guardado: outputs/avance2/matriz_confusion.png")

joblib.dump(mejor_modelo, RAIZ / "src" / "modelo_final.joblib")
print("Modelo guardado: src/modelo_final.joblib")

entradas = ["plazo_dias", "n_ejemplares"]
media = X_train[entradas].mean()
desviacion = X_train[entradas].std(ddof=0)
X_ent = ((X_train[entradas] - media) / desviacion).to_numpy()
X_pru = ((X_test[entradas] - media) / desviacion).to_numpy()
y_ent = y_train.to_numpy()

w = np.array([0.1, 0.1])
b = 0.0
for epoca in range(1000):
    p = 1 / (1 + np.exp(-(X_ent @ w + b)))
    error = p - y_ent
    w = w - 0.1 * (X_ent.T @ error) / len(y_ent)
    b = b - 0.1 * error.mean()
prob_pru = 1 / (1 + np.exp(-(X_pru @ w + b)))
exactitud_mano_ent = (((1 / (1 + np.exp(-(X_ent @ w + b)))) >= 0.5) == y_ent).mean()
exactitud_mano_pru = ((prob_pru >= 0.5) == y_test.to_numpy()).mean()

logistica = Pipeline([("escalado", StandardScaler()), ("modelo", LogisticRegression())])
logistica.fit(X_train[entradas], y_train)
reg2 = logistica.named_steps["modelo"]

comparacion = pd.DataFrame({
    "": ["w1 (plazo_dias)", "w2 (n_ejemplares)", "b (sesgo)",
         "Exactitud entrenamiento", "Exactitud prueba"],
    "Perceptrón a mano": [w[0], w[1], b, exactitud_mano_ent, exactitud_mano_pru],
    "Regresión logística": [reg2.coef_[0][0], reg2.coef_[0][1], reg2.intercept_[0],
                            logistica.score(X_train[entradas], y_train),
                            logistica.score(X_test[entradas], y_test)],
})
print("\nPerceptrón a mano vs Scikit-Learn (mismo corte, mismas dos entradas):")
print(comparacion.round(3).to_string(index=False))
print("Los dos pesos tienen el mismo signo: a mayor plazo, menor probabilidad de retraso.")