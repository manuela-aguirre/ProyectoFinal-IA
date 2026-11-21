from pathlib import Path

import joblib
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ROLES = ["Administrativo", "Bibliotecario", "Docente", "Estudiante", "Externo"]

modelo = joblib.load(RAIZ / "src" / "modelo_final.joblib")


def preparar_entradas(prestamos):
    for rol in ROLES:
        prestamos[f"rol_{rol}"] = (prestamos["rol"] == rol).astype(int)
    return prestamos[modelo.feature_names_in_]


prestamos_nuevos = pd.DataFrame([
    {"plazo_dias": 7,  "n_ejemplares": 1, "limite_prestamos": 5, "mes_prestamo": 3,  "dia_semana": 0, "rol": "Estudiante"},
    {"plazo_dias": 30, "n_ejemplares": 2, "limite_prestamos": 8, "mes_prestamo": 8,  "dia_semana": 2, "rol": "Docente"},
    {"plazo_dias": 14, "n_ejemplares": 1, "limite_prestamos": 2, "mes_prestamo": 10, "dia_semana": 4, "rol": "Externo"},
    {"plazo_dias": 21, "n_ejemplares": 3, "limite_prestamos": 8, "mes_prestamo": 5,  "dia_semana": 1, "rol": "Administrativo"},
])
X_nuevos = preparar_entradas(prestamos_nuevos)

probabilidad = modelo.predict_proba(X_nuevos)[:, 1]
prediccion = modelo.predict(X_nuevos)

prestamos_nuevos["probabilidad_retraso_%"] = (probabilidad * 100).round(1)
prestamos_nuevos["resultado"] = ["Con retraso" if p == 1 else "A tiempo" for p in prediccion]

print("Predicciones con el modelo guardado:")
print(prestamos_nuevos[["rol", "plazo_dias", "n_ejemplares", "limite_prestamos",
                        "probabilidad_retraso_%", "resultado"]].to_string(index=False))

mismo_prestamo = pd.DataFrame([
    {"plazo_dias": plazo, "n_ejemplares": 1, "limite_prestamos": 5,
     "mes_prestamo": 3, "dia_semana": 0, "rol": "Estudiante"}
    for plazo in [7, 14, 21, 30]
])
probabilidad_plazo = modelo.predict_proba(preparar_entradas(mismo_prestamo))[:, 1]
mismo_prestamo["probabilidad_retraso_%"] = (probabilidad_plazo * 100).round(1)

print("\nEl mismo préstamo de un estudiante (1 libro) con distintos plazos:")
print(mismo_prestamo[["plazo_dias", "probabilidad_retraso_%"]].to_string(index=False))