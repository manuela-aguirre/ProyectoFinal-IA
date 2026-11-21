# Chatbot de IA para la gestión del inventario de la biblioteca de Cotecnova

## 1. Descripción

### Problema
La base de datos de la biblioteca solo registra los préstamos; no guarda la disponibilidad de cada ejemplar. Los usuarios no saben si un libro está disponible y el bibliotecario no puede anticipar qué préstamos se devolverán tarde ni decidir qué comprar con respaldo.

### Qué hace este proyecto
Recorre el ciclo completo **datos → preparación → perceptrón desde cero → modelos de Scikit-Learn → evaluación → demostración** para responder:

> Dado un préstamo (plazo, número de libros, rol del usuario, fecha…), ¿con qué probabilidad se devolverá **con retraso**?
S
Ese dato sirve para que un chatbot avise al bibliotecario antes de la fecha límite. El chatbot conversacional **aún no está implementado**; lo entregado es el módulo de predicción, una demostración en consola y las funciones de disponibilidad del primer corte.

### Resultados principales

| Elemento | Resultado |
|---|---|
| Dataset limpio | 711 préstamos cerrados × 12 columnas, 0 nulos (51.6 % con retraso) |
| Perceptrón desde cero (NumPy) | Pérdida 0.7185 → 0.5923; exactitud 69.5 % entrenamiento / 68.5 % prueba; pesos `w1 (plazo)` = −0.986, `w2 (n_ejemplares)` = −0.023 |
| Mejor modelo de Scikit-Learn | **Bosque aleatorio**: F1 = 0.727, recall = 0.757, exactitud = 0.706 (prueba, 143 préstamos) |
| Hallazgo principal | Cuanto menor el plazo, mayor la probabilidad de retraso: 80.1 % (7 días) → 22.1 % (30 días) |

Comparación de modelos (prueba de 143 préstamos; validación cruzada de 5 pliegues sobre todos los datos):

| Modelo | Exactitud | Precisión | Recall | F1 | Val. cruzada (exactitud) |
|---|---:|---:|---:|---:|---:|
| Regresión logística | 0.650 | 0.643 | 0.730 | 0.684 | 0.684 |
| KNN (5 vecinos) | 0.622 | 0.625 | 0.676 | 0.649 | 0.619 |
| Árbol de decisión | 0.706 | 0.767 | 0.622 | 0.687 | 0.684 |
| **Bosque aleatorio** | **0.706** | 0.700 | **0.757** | **0.727** | 0.681 |
| Referencia: siempre "retraso" | 0.517 | 0.517 | 1.000 | 0.682 | — |

**Métrica elegida: F1, acompañado del recall.** No avisar de un retraso real es más costoso que avisar de más, pero el recall solo se maximiza diciendo siempre "retraso"; el F1 obliga a cuidar también la precisión. Con 143 préstamos de prueba las diferencias entre modelos son pequeñas frente a la incertidumbre (margen de ±7.5 puntos en la exactitud); ver la discusión en el informe.

---

## 2. Estructura del repositorio

```
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── data/
│   ├── *.csv                    # 11 tablas originales (simuladas)
│   ├── proyecto_limpio.csv      # dataset limpio de préstamos (lo genera el código)
│   └── libros_demanda.csv       # dataset limpio por libro (lo genera el código)
├── docs/
│   └── Informe_Final_GrupoXX_ChatbotBibliotecaCotecnova.pdf
├── outputs/
│   ├── graficos/                # 5 gráficos de análisis (Seaborn) + gráficos del Avance 1
│   └── avance2/                 # curvas de aprendizaje, fronteras de decisión, matriz de confusión
└── src/
    ├── cargar_datos.py          # lee los CSV (Avance 1)
    ├── analisis_chatbot.py      # disponibilidad y ubicación de ejemplares (Avance 1)
    ├── eda_proyecto.py          # análisis exploratorio con NumPy (Avance 1)
    ├── main.py                  # punto de entrada del Avance 1
    ├── analisis_biblioteca.py   # PREPARACIÓN DE DATOS con Pandas + 5 gráficos Seaborn
    ├── perceptron_proyecto_retraso.py   # perceptrón desde cero (modelo principal)
    ├── perceptron_proyecto_demanda.py   # perceptrón desde cero (modelo complementario)
    ├── modelo_proyecto.py       # compara modelos de Scikit-Learn y guarda el mejor
    ├── usar_modelo.py           # carga el modelo guardado y predice (demostración)
    ├── menu_consola.py          # extra: menú en consola sobre el modelo guardado
    └── modelo_final.joblib      # mejor modelo guardado (Pipeline + bosque aleatorio)
```

Correspondencia con la estructura sugerida en los lineamientos: `preparacion_datos.py` → `analisis_biblioteca.py`; `perceptron_proyecto.py` → los dos archivos `perceptron_proyecto_*.py`; los gráficos están en `outputs/`.

---

## 3. Cómo ejecutar (paso a paso con Docker)

**Requisitos:** Docker y Docker Compose. No se necesita nada más.

```bash
# 1. Clonar el repositorio
git clone <URL-DEL-REPOSITORIO>
cd <carpeta-del-repositorio>

# 2. Construir y encender el contenedor (Python 3.12 + dependencias)
docker compose up -d --build

# 3. Preparar los datos y generar los 5 gráficos de análisis
docker compose exec python python src/analisis_biblioteca.py

# 4. Perceptrones desde cero (curva de aprendizaje y frontera de decisión)
docker compose exec python python src/perceptron_proyecto_retraso.py
docker compose exec python python src/perceptron_proyecto_demanda.py

# 5. Comparar modelos de Scikit-Learn, guardar el mejor y crear la matriz de confusión
docker compose exec python python src/modelo_proyecto.py

# 6. Demostración: cargar el modelo guardado y predecir préstamos nuevos (sin reentrenar)
docker compose exec python python src/usar_modelo.py

# 7. (Extra) Menú interactivo en consola
docker compose exec python python src/menu_consola.py

# 8. Apagar el contenedor
docker compose down
```

Los pasos 3 a 5 son independientes de la demostración: `usar_modelo.py` solo necesita `src/modelo_final.joblib`, que ya viene en el repositorio. Los directorios `src/`, `data/` y `outputs/` están montados como volúmenes, así que los archivos generados (CSV limpios, gráficos y modelo) aparecen en su carpeta local.

**Sin Docker** (alternativa): `pip install -r requirements.txt` y luego `python src/<script>.py` desde la raíz del repositorio.

### Salida esperada de la demostración (`usar_modelo.py`)

```
           rol  plazo_dias  n_ejemplares  limite_prestamos  probabilidad_retraso_%   resultado
    Estudiante           7             1                 5                    66.2 Con retraso
       Docente          30             2                 8                    33.9    A tiempo
       Externo          14             1                 2                    71.2 Con retraso
Administrativo          21             3                 8                    42.0    A tiempo
```

La probabilidad del bosque aleatorio es la fracción de árboles que votan por retraso; se clasifica "con retraso" si es 50 % o más. Para el mismo préstamo de un estudiante, la probabilidad baja de 66.2 % (7 días) a 29.4 % (30 días).

---

## 4. Datos

| | |
|---|---|
| Fuente | Datos simulados por el grupo a partir del modelo entidad-relación de la biblioteca de Cotecnova |
| Archivos | 11 CSV en `data/` (libros, ejemplares, ubicaciones, usuarios, préstamos, géneros, autores, editoriales y sus relaciones) |
| Periodo | Préstamos del 2 de enero de 2022 al 28 de diciembre de 2025 (900 préstamos) |
| Tamaño | Menos de 1 MB: los datos están incluidos en el repositorio |

### Preparación (`src/analisis_biblioteca.py`)

| Problema | Decisión |
|---|---|
| 189 préstamos sin `fecha_devolucion` (21 %) | No se inventa una fecha; se excluyen del modelo porque no se conoce el resultado |
| `observaciones` nulas (627) | No se usa como entrada (podría escribirse al cerrar el préstamo) |
| Fechas como texto | Se convierten con `pd.to_datetime` |
| Duplicados, valores imposibles, atípicos | Revisados; no hay nada que eliminar |
| 244 préstamos anteriores al registro del usuario | Se conservan y se marcan; ninguna entrada depende de ese campo |
| `rol` categórica | `pd.get_dummies` (one-hot), porque ningún rol es "mayor" que otro |
| Fuga de información | Se excluyen `dias_retraso`, `multa`, `dias_prestamo` y `tiene_observacion` de las entradas |
| Datos personales | Se descartan `correo`, `telefono`, `direccion` y `clave` |

Resultado: `data/proyecto_limpio.csv`, 711 filas × 12 columnas, 0 nulos. Variable objetivo: `con_retraso` (1 si `dias_retraso > 0`).

---

## 5. Modelos

### Perceptrón desde cero (NumPy)
`z = w1·x1 + w2·x2 + b` → sigmoide → pérdida logarítmica → descenso del gradiente (tasa 0.1, 1 000 épocas, valores iniciales `w1 = w2 = 0.1`, `b = 0`). Entradas: `plazo_dias` y `n_ejemplares`, estandarizadas con la media y la desviación **solo del entrenamiento**. El signo negativo de `w1` significa que a mayor plazo, menor probabilidad de retraso (cada desviación, ~8.5 días, multiplica las odds por 0.37); `w2` es casi cero.

Dónde mirar en el código: la actualización de pesos está en `entrenar()` de `perceptron_proyecto_retraso.py`; la división entrenamiento/prueba, en `cargar_y_dividir()` (perceptrón) y en `train_test_split` de `modelo_proyecto.py`.

El perceptrón se compara con `LogisticRegression` de Scikit-Learn en `modelo_proyecto.py` (sección 8): con el mismo corte, los pesos tienen el mismo signo y casi el mismo valor (−0.958 frente a −0.948).

### Modelos de Scikit-Learn (`src/modelo_proyecto.py`)
Cuatro modelos (regresión logística, KNN, árbol de decisión, bosque aleatorio), cada uno en un `Pipeline` con `StandardScaler`. División 80/20 estratificada (`random_state=42`: 568 entrenamiento, 143 prueba) y validación cruzada de 5 pliegues. El escalado se ajusta solo con los datos de entrenamiento (o de cada pliegue), por lo que no hay fuga de información. El mejor modelo (mayor F1) se guarda con `joblib` en `src/modelo_final.joblib`.

### Gráficos

| Tipo | Archivos |
|---|---|
| Análisis (Seaborn), 5 | `outputs/graficos/biblioteca_*.png` |
| Evaluación | `outputs/avance2/curva_aprendizaje_retraso.png`, `frontera_decision_retraso.png`, `matriz_confusion.png` y los equivalentes del modelo de demanda |

---

## 6. Retroalimentación del Avance 2

El grupo no recibió observaciones escritas del Avance 2 en el momento de redactar este README. Los cambios hechos para la entrega final fueron: el modelo de Scikit-Learn con `Pipeline`, validación cruzada y métricas (`modelo_proyecto.py`), la demostración (`usar_modelo.py`), el menú en consola (`menu_consola.py`), la comparación del perceptrón con la regresión logística y la exclusión explícita de variables que filtran la respuesta. <!-- Si se recibió retroalimentación, listar aquí cada observación y cómo se atendió. -->

## 7. Limitaciones y trabajo futuro

- Datos simulados: validar el método con datos reales o de una fuente oficial.
- Se excluyeron 189 préstamos abiertos (135 con más de un año), probablemente retrasos: la tasa real se subestima.
- Muestra de prueba pequeña (143); el modelo se eligió con la misma muestra de prueba, lo que puede favorecerlo ligeramente.
- Con pocas variables de entrada con señal (solo el plazo), un modelo complejo aporta una mejora modesta.
- Pendiente: chatbot conversacional con modelo de lenguaje, `GridSearchCV` y `MLPClassifier`. Si se agrega IA generativa, las claves de API deben ir en un `.env` (ya incluido en `.gitignore`), nunca en el código.
- Consideración ética: usar la predicción solo para enviar recordatorios, nunca para negar préstamos o cobrar multas de forma automática. La tabla `usuario` guarda contraseñas en texto plano (datos de prueba); en un sistema real deben guardarse con hash.

## 8. Herramientas

Python 3.12, NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, joblib y Docker.
