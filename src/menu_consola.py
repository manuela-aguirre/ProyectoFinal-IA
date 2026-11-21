from pathlib import Path

import joblib
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ROLES = ["Administrativo", "Bibliotecario", "Docente", "Estudiante", "Externo"]
PLAZOS = [7, 14, 21, 30]
LIMITES = [2, 3, 5, 8]
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

modelo = joblib.load(RAIZ / "src" / "modelo_final.joblib")


def pedir_opcion(pregunta, opciones, mostrar=None):
    mostrar = mostrar or [str(o) for o in opciones]
    print(pregunta)
    for numero, texto in enumerate(mostrar, start=1):
        print(f"  {numero}. {texto}")
    while True:
        respuesta = input("Elija el número: ").strip()
        if respuesta.isdigit() and 1 <= int(respuesta) <= len(opciones):
            return opciones[int(respuesta) - 1]
        print("Opción no válida, intente de nuevo.")


def pedir_prestamo():
    prestamo = {"rol": pedir_opcion("\nRol del usuario:", ROLES)}
    prestamo["plazo_dias"] = pedir_opcion("\nPlazo del préstamo (días):", PLAZOS)
    prestamo["n_ejemplares"] = pedir_opcion("\nNúmero de libros prestados:", [1, 2, 3])
    prestamo["limite_prestamos"] = pedir_opcion("\nLímite de préstamos del usuario:", LIMITES)
    prestamo["mes_prestamo"] = pedir_opcion("\nMes del préstamo:", list(range(1, 13)))
    prestamo["dia_semana"] = pedir_opcion("\nDía de la semana:", list(range(7)), DIAS)
    return prestamo


def probabilidad_retraso(prestamo):
    fila = pd.DataFrame([prestamo])
    for rol in ROLES:
        fila[f"rol_{rol}"] = (fila["rol"] == rol).astype(int)
    return modelo.predict_proba(fila[modelo.feature_names_in_])[0, 1]


def predecir(prestamo):
    probabilidad = probabilidad_retraso(prestamo)
    print(f"\nProbabilidad de retraso: {probabilidad:.1%}")
    if probabilidad >= 0.5:
        print("Resultado: CON RETRASO. Conviene enviar un recordatorio antes de la fecha límite.")
    else:
        print("Resultado: A TIEMPO. No se requiere ninguna acción.")


def comparar_plazos(prestamo):
    print("\nEl mismo préstamo con distintos plazos:")
    for plazo in PLAZOS:
        probabilidad = probabilidad_retraso({**prestamo, "plazo_dias": plazo})
        print(f"  {plazo:>2} días -> {probabilidad:.1%} de probabilidad de retraso")


def main():
    print("=" * 56)
    print("  BIBLIOTECA COTECNOVA - ¿SE DEVOLVERÁ A TIEMPO EL LIBRO?")
    print("=" * 56)
    ultimo = None
    while True:
        print("\nMENÚ")
        print("  1. Predecir el retraso de un préstamo")
        print("  2. Ver cómo cambia el riesgo según el plazo (último préstamo)")
        print("  3. Salir")
        opcion = input("Elija el número: ").strip()
        if opcion == "1":
            ultimo = pedir_prestamo()
            predecir(ultimo)
        elif opcion == "2":
            if ultimo is None:
                print("\nPrimero haga una predicción con la opción 1.")
            else:
                comparar_plazos(ultimo)
        elif opcion == "3":
            print("Hasta luego.")
            break
        else:
            print("Opción no válida, intente de nuevo.")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nHasta luego.")