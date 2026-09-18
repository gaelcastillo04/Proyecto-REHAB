# Proyecto REHAB 

**Selección y comparación de modelos de aprendizaje máquina para reconocer 16 actividades a partir de sensores inerciales y un guante de flexión.**

---

## Resumen ejecutivo

| | |
|---|---|
| **Tipo de problema** | Clasificación supervisada **multiclase** (16 actividades, etiquetas 0–15) |
| **Datos** | 4 616 repeticiones · 480 características numéricas continuas por repetición |
| **Modelos comparados** | Regresión logística · Árbol de decisión · Naive Bayes gaussiano · K-Nearest Neighbors · Random Forest |
| **Criterio de selección** | F1-macro en el conjunto de **validación**; el conjunto de **test** se usa una sola vez |
| **Modelo seleccionado** | **Random Forest** (100 árboles, sin límite de profundidad, `class_weight="balanced"`) |
| **Desempeño en test** | Accuracy **95.67 %** · F1-macro **0.9556** |

---

## Tabla de contenido

1. [Estructura del proyecto](#1-estructura-del-proyecto)
2. [Identificación del problema](#2-identificación-del-problema)
3. [Los datos y su compatibilidad con los modelos](#3-los-datos-y-su-compatibilidad-con-los-modelos)
4. [Metodología de evaluación](#4-metodología-de-evaluación)
5. [Modelos investigados y configuraciones probadas](#5-modelos-investigados-y-configuraciones-probadas)
6. [Resultados](#6-resultados)
7. [Decisión final y justificación](#7-decisión-final-y-justificación)
8. [Cómo ejecutar](#8-cómo-ejecutar)
9. [Consideraciones éticas y normativas](#9-consideraciones-éticas-y-normativas)
10. [Limitaciones y trabajo futuro](#10-limitaciones-y-trabajo-futuro)

---

## 1. Estructura del proyecto

```text
Proyecto-REHAB/
│
├── Dataset/                      # Señales originales (.npy)
│   ├── 000_1.npy  000_2.npy      #   XXX_1 → 2 sensores inerciales
│   ├── ...                       #   XXX_2 → guante de flexión
│   └── 015_1.npy  015_2.npy
├── dataset_ml_ventanas.csv       # Dataset tabular generado (4 616 × 482)
│
├── webapp/
│   ├── backend/                  # Todo el código Python
│   │   ├── ml/                   #   Un script por modelo (búsqueda en malla + evaluación)
│   │   │   ├── regresion_logistica.py    # Modelo 1 · Regresión logística (One-vs-Rest)
│   │   │   ├── arboles_de_decision.py    # Modelo 2 · Árbol de decisión
│   │   │   ├── bayes.py                  # Modelo 3 · Naive Bayes gaussiano
│   │   │   ├── knn.py                    # Modelo 4 · K-Nearest Neighbors
│   │   │   └── random_forest.py          # Modelo 5 · Random Forest  seleccionado
│   │   ├── scripts/              #   Pipeline compartido
│   │   │   ├── config.py                 # Rutas (Dataset/, CSV) y semilla global (42)
│   │   │   ├── data_utils.py             # Carga y división estratificada 50/25/25
│   │   │   └── crear_dataset_ventanas.py # Ingeniería de características (señal → 480 features)
│   │   ├── api/                  #   API FastAPI de la interfaz web
│   │   │   ├── main.py                   # Endpoints; entrena los modelos bajo demanda
│   │   │   └── models.py                 # Registro de modelos e hiperparámetros expuestos
│   │   └── requirements.txt      #   fastapi, uvicorn
│   └── frontend/                 # GUI React (Vite) para entrenar y comparar modelos
│
├── Makefile                      # make install / make models / make dev
├── requirements.txt              # numpy, pandas, scikit-learn
├── .gitignore                    # .venv/ y __pycache__/
└── README.md
```

Todos los scripts de modelos viven en `webapp/backend/ml/` y comparten el mismo flujo, definido en `scripts/data_utils.py`, para que la comparación sea justa: mismos datos, misma división, misma semilla, mismas métricas.

---

## 2. Identificación del problema

La variable objetivo `actividad` toma **16 valores discretos (0–15)** que representan **categorías** de ejercicios de rehabilitación. No existe orden ni distancia entre ellas: la actividad 7 no es "mayor" que la 3.

Por lo tanto el problema es de **clasificación supervisada multiclase**, no de regresión.

> **Decisión:** En la etapa anterior se implementó una regresión lineal como primera aproximación. Sus predicciones continuas tenían que redondearse al entero más cercano para compararse con la etiqueta real, lo cual carece de sentido para categorías nominales. Por eso en esta etapa se descartó la regresión y se evaluaron **únicamente clasificadores**.

---

## 3. Los datos y su compatibilidad con los modelos

### 3.1 Origen y transformación

Las señales crudas son series de tiempo: cada repetición tiene **880 puntos temporales × 6 canales × 2 sensores**. Los modelos clásicos de scikit-learn esperan una tabla de longitud fija, así que `crear_dataset_ventanas.py` convierte cada repetición en un vector de características:

```
4 ventanas de 220 puntos × 6 canales × 2 sensores × 10 estadísticos = 480 características
```

Los 10 estadísticos por ventana y canal son: media, desviación estándar, mínimo, Q1, mediana, Q3, máximo, rango intercuartílico, asimetría y curtosis. Cada columna del CSV se nombra `s{sensor}_w{ventana}_c{canal}_{estadístico}`, por ejemplo `s1_w2_c4_curtosis`.

### 3.2 Características del dataset resultante

| Propiedad | Valor | Implicación para el modelado |
|---|---|---|
| Filas (repeticiones) | 4 616 | Tamaño moderado: modelos clásicos son suficientes y rápidos |
| Características | 480, todas **numéricas continuas** | Compatibles con cualquier clasificador de scikit-learn; no se requiere codificación |
| Valores faltantes | 0 | No se necesita imputación |
| Escalas | Muy heterogéneas (medias vs. curtosis, IMU vs. flexión) | Los modelos basados en distancia o gradiente **requieren estandarización** |
| Balance de clases | 212 a 385 muestras por clase (relación 1 : 1.8) | Ligero desbalance → se prueba `class_weight="balanced"` y se reporta **F1-macro** además de accuracy |

<details>
<summary>Distribución exacta de muestras por actividad</summary>

| Actividad | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Muestras | 232 | 212 | 267 | 250 | 287 | 293 | 260 | 385 | 299 | 307 | 311 | 235 | 293 | 313 | 359 | 313 |

</details>

### 3.3 ¿Qué modelos son compatibles?

Al tener características **numéricas, densas, sin faltantes y con etiqueta categórica**, cualquier clasificador supervisado es aplicable. Se eligieron cinco familias con **sesgos inductivos distintos** para cubrir el espectro de hipótesis:

| Familia | Frontera de decisión | ¿Necesita escalar? | Por qué se incluyó |
|---|---|---|---|
| Regresión logística | Lineal | Sí | Base sólida e interpretable; prueba si las clases son linealmente separables |
| Árbol de decisión | Ejes paralelos, no lineal | No | Captura umbrales e interacciones sin suposiciones sobre la distribución |
| Naive Bayes gaussiano | Cuadrática | Opcional | Muy rápido; prueba si la independencia condicional entre features es razonable |
| K-Nearest Neighbors | Arbitraria (local) | Sí | No paramétrico; aprovecha que repeticiones de una misma actividad producen vectores estadísticos muy parecidos |
| Random Forest | Ejes paralelos, no lineal (ensamble) | No | Promedia muchos árboles entrenados con muestras y features distintas; reduce la varianza del árbol individual y tolera features redundantes |

---

## 4. Metodología de evaluación

Para que la selección de modelo sea honesta y reproducible se siguió el mismo protocolo en los cinco scripts:

```
                    ┌──────────────┐
  dataset (4 616) ─►│ 50 % train   │ 2 308 ─► entrenar cada configuración
                    ├──────────────┤
                    │ 25 % val     │ 1 154 ─► elegir hiperparámetros (F1-macro)
                    ├──────────────┤
                    │ 25 % test    │ 1 154 ─► evaluar UNA sola vez el modelo final
                    └──────────────┘
```

1. **División estratificada 50 / 25 / 25** con `random_state=42`. La estratificación mantiene la misma proporción de cada actividad en los tres conjuntos.
2. **Búsqueda de hiperparámetros en malla** sobre el conjunto de validación. Cada combinación se entrena en *train* y se mide en *val*.
3. **Métrica de selección: F1-macro.** Promedia el F1 de las 16 clases dándoles el mismo peso, lo que evita que una clase mayoritaria (la 7, con 385 muestras) infle el resultado. Accuracy se reporta siempre como apoyo; la entropía cruzada se reporta en los modelos que producen probabilidades calibradas (regresión logística, Naive Bayes y KNN).
4. **Reentrenamiento final con train + validación** (3 462 muestras) usando la mejor configuración, para aprovechar más datos.
5. **Evaluación única en test**, con accuracy, precision, recall y F1 macro, reporte por clase y matriz de confusión.
6. **Estandarización dentro de un `Pipeline`** en los modelos que la necesitan. El `StandardScaler` se ajusta solo con datos de entrenamiento, evitando fuga de información de validación o test. Los modelos basados en árboles (árbol de decisión y Random Forest) no se escalan porque sus divisiones por umbral son invariantes a la escala.
7. **Semilla fija en todos los componentes aleatorios**: la división de datos, el árbol de decisión y el Random Forest usan `random_state=42`.

---

## 5. Modelos investigados y configuraciones probadas

### 5.1 Regresión logística · `regresion_logistica.py`

Estrategia **One-vs-Rest**: se entrenan 16 clasificadores binarios, uno por actividad. Solver `lbfgs`, `max_iter=4000` para garantizar convergencia con 480 features.

| Hiperparámetro | Valores probados | Qué controla |
|---|---|---|
| `C` | 0.01, 0.1, 1.0, 10.0 | Inverso de la regularización L2 |
| `class_weight` | `None`, `balanced` | Compensación del desbalance de clases |

**8 configuraciones.**

<details>
<summary>Resultados en validación</summary>

| C | class_weight | Accuracy | F1-macro | Cross-Entropy |
|---|---|---|---|---|
| 0.01 | None | 83.88 % | 0.8330 | 0.6575 |
| 0.01 | balanced | 85.18 % | 0.8471 | 0.8762 |
| 0.1 | None | 87.69 % | 0.8727 | **0.4769** |
| 0.1 | balanced | 87.61 % | 0.8709 | 0.6173 |
| 1.0 | None | 87.61 % | 0.8711 | 0.6084 |
| **1.0** | **balanced** | **87.87 %** | **0.8736** | 0.6945 |
| 10.0 | None | 85.79 % | 0.8528 | 1.0731 |
| 10.0 | balanced | 85.62 % | 0.8507 | 1.1231 |

</details>

**Mejor configuración:** `C=1.0`, `class_weight="balanced"`. Con `C=10` el modelo se sobreajusta (la entropía cruzada casi se duplica); con `C=0.01` se subajusta.

### 5.2 Árbol de decisión · `arboles_de_decision.py`

| Hiperparámetro | Valores probados | Qué controla |
|---|---|---|
| `criterion` | `gini`, `entropy` | Medida de impureza para dividir nodos |
| `max_depth` | `None`, 5, 10, 20 | Profundidad máxima (complejidad) |
| `min_samples_split` | 2, 5, 10 | Mínimo de muestras para dividir un nodo |
| `class_weight` | `None`, `balanced` | Compensación del desbalance |

**48 configuraciones.**

<details>
<summary>Resultados en validación (resumen de las mejores por profundidad)</summary>

| criterion | max_depth | min_split | class_weight | Accuracy | F1-macro |
|---|---|---|---|---|---|
| gini | 5 | 2 | balanced | 71.84 % | 0.7034 |
| entropy | 5 | 2 | None | 74.96 % | 0.7452 |
| gini | 10 | 2 | balanced | 84.84 % | 0.8460 |
| **entropy** | **10** | **2** | **balanced** | **86.05 %** | **0.8558** |
| gini | 20 / None | 2 | balanced | 85.88 % | 0.8556 |
| entropy | 20 / None | 5 | balanced | 85.53 % | 0.8511 |

Las 48 combinaciones completas se imprimen al ejecutar el script.

</details>

**Mejor configuración:** `criterion="entropy"`, `max_depth=10`, `min_samples_split=2`, `class_weight="balanced"`. Profundidad 5 es claramente insuficiente (≈ 72 %); a partir de 10 el rendimiento se estabiliza y `max_depth=20` equivale a `None` porque el árbol nunca crece más allá de esa profundidad.

### 5.3 Naive Bayes gaussiano · `bayes.py`

Asume que cada característica sigue una distribución normal por clase y que las características son independientes entre sí dada la clase.

| Hiperparámetro | Valores probados | Qué controla |
|---|---|---|
| `var_smoothing` | 1e-9, 1e-6, 1e-3, 1e-1 | Fracción de la varianza máxima que se suma a todas las varianzas |
| `priors` | frecuencia de clase, uniforme | Probabilidad a priori de cada actividad |
| Escalado | sin / con `StandardScaler` | Estandarización previa |

**16 configuraciones.**

<details>
<summary>Resultados en validación</summary>

| var_smoothing | priors | escalar | Accuracy | F1-macro | Cross-Entropy |
|---|---|---|---|---|---|
| **1e-9** | **uniforme** | **False** | **75.82 %** | **0.7551** | 7.5810 |
| 1e-9 | None | False / True | 75.82 % | 0.7550 | 7.5785 |
| 1e-6 | None / uniforme | False / True | 75.82 % | 0.7550 – 0.7551 | 7.58 – 7.60 |
| 1e-3 | None | True | 74.96 % | 0.7457 | 7.7701 |
| 1e-3 | None | False | 73.14 % | 0.7254 | 8.0278 |
| 1e-1 | None | True | 70.10 % | 0.6922 | 8.3300 |
| 1e-1 | None | False | 62.74 % | 0.6079 | 7.1405 |

</details>

**Mejor configuración:** `var_smoothing=1e-9`, `priors` uniforme, sin escalar. Ningún hiperparámetro ayuda de forma significativa: el techo está en ≈ 76 %. La entropía cruzada tan alta (≈ 7.6) indica probabilidades extremadamente sobreconfiadas, síntoma de que el **supuesto de independencia se viola**: los 480 estadísticos están fuertemente correlacionados entre sí (media, mediana y cuartiles de un mismo canal son casi redundantes).

### 5.4 K-Nearest Neighbors · `knn.py`

Clasifica cada repetición según la actividad mayoritaria de sus K vecinos más cercanos en el espacio de 480 dimensiones estandarizado, con distancia euclidiana.

| Hiperparámetro | Valores probados | Qué controla |
|---|---|---|
| `n_neighbors` (K) | 1, 3, 5, 7, 9 | Cantidad de vecinos que votan |
| `weights` | `uniform`, `distance` | Voto igualitario o ponderado por cercanía |

**10 configuraciones.**

<details>
<summary>Resultados en validación</summary>

| K | weights | Accuracy | F1-macro | Cross-Entropy |
|---|---|---|---|---|
| **1** | **uniform** | **90.81 %** | **0.9043** | 3.3108 |
| 1 | distance | 90.81 % | 0.9043 | 3.3108 |
| 3 | uniform | 82.06 % | 0.8183 | 2.0604 |
| 3 | distance | 86.83 % | 0.8648 | 2.0127 |
| 5 | uniform | 78.34 % | 0.7832 | 1.5752 |
| 5 | distance | 84.23 % | 0.8407 | 1.5040 |
| 7 | uniform | 75.91 % | 0.7583 | 1.3883 |
| 7 | distance | 81.63 % | 0.8143 | 1.2994 |
| 9 | uniform | 75.22 % | 0.7514 | 1.2185 |
| 9 | distance | 80.59 % | 0.8040 | **1.1171** |

</details>

**Mejor configuración:** `K=1`, `weights="uniform"` (con K=1 ambos tipos de peso son idénticos, porque solo vota un vecino). El rendimiento **decrece monótonamente al aumentar K**, y `distance` siempre supera a `uniform` para K > 1. Esto revela la geometría de los datos: cada repetición tiene un vecino casi idéntico de su misma actividad, pero al ampliar el vecindario entran ejemplos de actividades parecidas que contaminan el voto.

### 5.5 Random Forest · `random_forest.py`

Ensamble de árboles de decisión. Cada árbol se entrena con una muestra *bootstrap* del conjunto de entrenamiento y, en cada nodo, solo considera un subconjunto aleatorio de las 480 características (√480 ≈ 22). La predicción final es el voto mayoritario de todos los árboles. Se paraleliza con `n_jobs=-1`.

| Hiperparámetro | Valores probados | Qué controla |
|---|---|---|
| `n_estimators` | 50, 100, 200 | Número de árboles del ensamble |
| `max_depth` | `None`, 10, 20 | Profundidad máxima de cada árbol |
| `min_samples_split` | 2, 5 | Mínimo de muestras para dividir un nodo |
| `class_weight` | `None`, `balanced` | Compensación del desbalance |

**36 configuraciones.**

<details>
<summary>Resultados en validación (las 36 configuraciones)</summary>

| n_estimators | max_depth | min_split | class_weight | Accuracy | F1-macro |
|---|---|---|---|---|---|
| 50 | None / 20 | 2 | None | 94.54 % | 0.9409 |
| 50 | None / 20 | 2 | balanced | 95.93 % | 0.9567 |
| 50 | None / 20 | 5 | None | 95.58 % | 0.9528 |
| 50 | None / 20 | 5 | balanced | 95.15 % | 0.9485 |
| 50 | 10 | 2 | None | 94.71 % | 0.9440 |
| 50 | 10 | 2 | balanced | 95.67 % | 0.9533 |
| 50 | 10 | 5 | None | 94.54 % | 0.9413 |
| 50 | 10 | 5 | balanced | 94.28 % | 0.9396 |
| 100 | None / 20 | 2 | None | 96.01 % | 0.9571 |
| **100** | **None / 20** | **2** | **balanced** | **96.53 %** | **0.9631** |
| 100 | None / 20 | 5 | None | 96.01 % | 0.9575 |
| 100 | None / 20 | 5 | balanced | 95.93 % | 0.9564 |
| 100 | 10 | 2 | None | 95.41 % | 0.9511 |
| 100 | 10 | 2 | balanced | 95.93 % | 0.9567 |
| 100 | 10 | 5 | None | 94.89 % | 0.9451 |
| 100 | 10 | 5 | balanced | 94.89 % | 0.9458 |
| 200 | None / 20 | 2 | None | 96.01 % | 0.9574 |
| 200 | None / 20 | 2 | balanced | 96.19 % | 0.9591 |
| 200 | None / 20 | 5 | None | 95.93 % | 0.9566 |
| 200 | None / 20 | 5 | balanced | 96.36 % | 0.9611 |
| 200 | 10 | 2 | None | 95.58 % | 0.9525 |
| 200 | 10 | 2 | balanced | 95.75 % | 0.9544 |
| 200 | 10 | 5 | None | 95.32 % | 0.9500 |
| 200 | 10 | 5 | balanced | 95.49 % | 0.9517 |

`max_depth=20` y `max_depth=None` producen exactamente los mismos resultados en todos los casos, por lo que se agrupan en una sola fila.

</details>

**Mejor configuración:** `n_estimators=100`, `max_depth=None`, `min_samples_split=2`, `class_weight="balanced"`. Observaciones:

- **Las 36 configuraciones quedan entre 0.940 y 0.963 de F1-macro**: el ensamble es muy poco sensible a los hiperparámetros, a diferencia del árbol individual, que oscila entre 0.70 y 0.86.
- **Pasar de 50 a 100 árboles ayuda** (+0.006 de F1); de 100 a 200 ya no mejora. Con 100 árboles la varianza del ensamble está prácticamente saturada.
- **Limitar la profundidad a 10 perjudica ligeramente** (≈ −0.006). Los árboles completamente crecidos son los mejores miembros del ensamble porque el promedio ya se encarga de controlar el sobreajuste.
- **`class_weight="balanced"` gana en la mayoría de las comparaciones directas**, consistente con el ligero desbalance de clases.

---

## 6. Resultados

### 6.1 Comparación final en el conjunto de test

Cada modelo fue reentrenado con train + validación usando su mejor configuración y evaluado **una sola vez** en las 1 154 muestras de test.

| # | Modelo | Mejor configuración | Accuracy | Precision macro | Recall macro | **F1-macro** |
|---|---|---|---|---|---|---|
| 1 | **Random Forest** | 100 árboles, depth=None, balanced | **95.67 %** | **0.9576** | **0.9570** | **0.9556** |
| 2 | K-Nearest Neighbors | K=1, euclidiana, estandarizado | 92.89 % | 0.9332 | 0.9304 | 0.9291 |
| 3 | Regresión logística | C=1.0, balanced, OvR | 88.56 % | 0.8841 | 0.8846 | 0.8831 |
| 4 | Árbol de decisión | entropy, depth=10, balanced | 85.79 % | 0.8576 | 0.8577 | 0.8545 |
| 5 | Naive Bayes gaussiano | var_smoothing=1e-9, priors uniforme | 74.70 % | 0.7634 | 0.7510 | 0.7406 |

```
F1-macro en test
Random Forest        ████████████████████████████████████████████████  0.956
KNN (K=1)            ███████████████████████████████████████████████   0.929
Regresión logística  ████████████████████████████████████████████      0.883
Árbol de decisión    ███████████████████████████████████████████       0.855
Naive Bayes          █████████████████████████████████████             0.741
```

### 6.2 Consistencia validación → test

| Modelo | F1-macro val | F1-macro test | Diferencia |
|---|---|---|---|
| Random Forest | 0.9631 | 0.9556 | −0.008 |
| KNN | 0.9043 | 0.9291 | +0.025 |
| Regresión logística | 0.8736 | 0.8831 | +0.010 |
| Árbol de decisión | 0.8558 | 0.8545 | −0.001 |
| Naive Bayes | 0.7551 | 0.7406 | −0.015 |

Todos los modelos generalizan de forma estable: la diferencia entre validación y test es pequeña en los cinco casos. Random Forest pierde menos de un punto porcentual respecto a validación, lo cual es la variación esperada al cambiar de partición y confirma que la búsqueda de hiperparámetros no se sobreajustó al conjunto de validación. La mejora de KNN en test se explica porque el modelo final se entrena con 50 % más datos (train + val), y KNN es el modelo que más se beneficia de tener más ejemplos de referencia.

### 6.3 Desempeño por clase del modelo seleccionado (Random Forest, test)

<details>
<summary>Ver reporte completo por actividad</summary>

| Actividad | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 1.0000 | 58 |
| 1 | 0.9123 | 0.9811 | 0.9455 | 53 |
| 2 | 0.9844 | 0.9545 | 0.9692 | 66 |
| 3 | 1.0000 | 0.9516 | 0.9752 | 62 |
| 4 | 1.0000 | 0.9167 | 0.9565 | 72 |
| 5 | 0.9861 | 0.9726 | 0.9793 | 73 |
| 6 | 0.9545 | 0.9692 | 0.9618 | 65 |
| 7 | 1.0000 | 0.9897 | 0.9948 | 97 |
| 8 | 0.8889 | 0.9600 | 0.9231 | 75 |
| 9 | 0.9444 | 0.8831 | 0.9128 | 77 |
| 10 | 0.9500 | 0.9744 | 0.9620 | 78 |
| 11 | 0.7662 | 1.0000 | 0.8676 | 59 |
| 12 | 0.9688 | 0.8493 | 0.9051 | 73 |
| 13 | 1.0000 | 0.9359 | 0.9669 | 78 |
| 14 | 0.9783 | 1.0000 | 0.9890 | 90 |
| 15 | 0.9870 | 0.9744 | 0.9806 | 78 |

</details>

<details>
<summary>Ver matriz de confusión (filas = real, columnas = predicción)</summary>

```
      0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
 0 [ 58   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 1 [  0  52   0   0   0   0   0   0   0   0   0   1   0   0   0   0]
 2 [  0   2  63   0   0   0   0   0   0   0   0   1   0   0   0   0]
 3 [  0   2   0  59   0   0   0   0   0   0   1   0   0   0   0   0]
 4 [  0   0   0   0  66   1   0   0   0   0   0   5   0   0   0   0]
 5 [  0   0   0   0   0  71   0   0   0   0   0   2   0   0   0   0]
 6 [  0   0   0   0   0   0  63   0   0   0   0   2   0   0   0   0]
 7 [  0   0   0   0   0   0   0  96   0   0   0   1   0   0   0   0]
 8 [  0   0   0   0   0   0   0   0  72   3   0   0   0   0   0   0]
 9 [  0   0   0   0   0   0   0   0   9  68   0   0   0   0   0   0]
10 [  0   1   1   0   0   0   0   0   0   0  76   0   0   0   0   0]
11 [  0   0   0   0   0   0   0   0   0   0   0  59   0   0   0   0]
12 [  0   0   0   0   0   0   3   0   0   1   2   4  62   0   0   1]
13 [  0   0   0   0   0   0   0   0   0   0   0   1   2  73   2   0]
14 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  90   0]
15 [  0   0   0   0   0   0   0   0   0   0   1   1   0   0   0  76]
```

</details>

- **15 de 16 actividades** superan F1 = 0.90, y tres de ellas (0, 7 y 14) rozan la clasificación perfecta.
- La **actividad 11** es la única por debajo de 0.90 (F1 0.87): tiene recall perfecto (1.00) pero precisión baja (0.77). El modelo nunca falla una repetición real de la actividad 11, pero absorbe 18 muestras de otras actividades, principalmente de la 4 (5 muestras) y la 12 (4 muestras).
- La segunda confusión más notable es entre las actividades **8 y 9** (12 muestras cruzadas en total), lo que sugiere que son movimientos muy similares.
- La **actividad 1**, que era el punto débil de KNN (F1 0.79) y de todos los modelos anteriores, sube a F1 0.95 con Random Forest.

---

## 7. Decisión final y justificación

### Modelo seleccionado: **Random Forest con 100 árboles, sin límite de profundidad y `class_weight="balanced"`**

Las razones, por las que se opto por esta decisión:

1. **Es el mejor en todas las métricas de test**, con una ventaja de 2.8 puntos de accuracy y 0.027 de F1-macro sobre el segundo lugar (KNN). Con 1 154 muestras de test, esa diferencia equivale a ≈ 32 repeticiones más clasificadas correctamente. Frente a la regresión logística la ventaja es de 7.1 puntos.

2. **La ventaja es consistente**, no puntual: Random Forest ganó en validación (0.963 vs 0.904 de KNN) y en test (0.956 vs 0.929), y su desempeño por clase es el más uniforme (15 de 16 clases con F1 > 0.90, ninguna por debajo de 0.86).

3. **Es el modelo más robusto ante los hiperparámetros.** Las 36 configuraciones probadas quedan en un rango de 0.02 de F1-macro. Esto reduce el riesgo de que el resultado dependa de una elección afortunada de la malla de búsqueda, riesgo que sí existe en KNN (donde pasar de K=1 a K=3 cuesta 4 puntos).

4. **Encaja con la naturaleza de los datos.** Las 480 características son altamente redundantes (estadísticos correlacionados de un mismo canal). El submuestreo aleatorio de características en cada nodo hace que distintos árboles exploten distintas variables redundantes, y el promedio cancela el ruido. Esto explica el salto del árbol individual (85.8 %) al ensamble (95.7 %): la misma familia de hipótesis, pero con la varianza controlada.


### Por qué se descartaron los demás

| Modelo | Motivo de descarte |
|---|---|
| K-Nearest Neighbors | Muy buen desempeño (92.9 %) pero 2.8 puntos por debajo.Al igual, K=1 es sensible al ruido y su inferencia requiere comparar contra todo el conjunto de entrenamiento. Se conserva como **segunda alternativa** por su simplicidad. |
| Regresión logística | 88.6 %. Su frontera lineal confunde sistemáticamente actividades 9, 11 y 12. Sigue siendo la opción recomendada si se prioriza interpretabilidad de pesos o inferencia en un dispositivo con recursos muy limitados. |
| Árbol de decisión | 85.8 %. Un solo árbol con 480 features es inestable y tiende a sobreajustar; el ensamble de árboles corrige exactamente ese problema, por lo que queda superado por Random Forest. |
| Naive Bayes | 74.7 %. El supuesto de independencia condicional se viola gravemente; ningún hiperparámetro lo corrige. Se descarta por completo. |

### Riesgo conocido de la decisión

Random Forest es el modelo más pesado de los cinco: entrenar las 36 configuraciones tarda varios minutos y el modelo final almacena 100 árboles completamente crecidos. Se aceptó este costo porque el entrenamiento se hace una sola vez y la inferencia con 100 árboles sigue siendo rápida (milisegundos por muestra), y porque la ganancia de casi 3 puntos sobre KNN y 7 sobre la regresión logística justifica el uso de memoria adicional.

---

## 8. Cómo ejecutar

Guía **desde cero**: parte de una máquina sin nada del proyecto instalado. Todos los comandos se ejecutan
desde la raíz del repositorio (`Proyecto-REHAB/`).

### Paso 0 · Requisitos del sistema

Solo hacen falta estas herramientas; el resto se instala automáticamente en el Paso 2.

| Herramienta | Versión | Comprobar |
|---|---|---|
| Git | cualquiera | `git --version` |
| Python | 3.11 o superior | `python3 --version` |
| GNU Make | cualquiera | `make --version` |
| Node.js + npm | Node 20 o superior | `node --version && npm --version` |

Node y npm solo se necesitan para la interfaz web (Paso 4). Si únicamente vas a correr los scripts de
modelos, basta con Git, Python y Make.

### Paso 1 · Clonar el repositorio

```bash
git clone https://github.com/gaelcastillo04/Proyecto-REHAB.git
cd Proyecto-REHAB
```

### Paso 2 · Instalar dependencias

```bash
make install
```

Esto hace, en orden:

1. Crea el entorno virtual `.venv/` en la raíz del proyecto (si no existe).
2. Instala `requirements.txt` (numpy, pandas, scikit-learn, con versiones fijas) y
   `webapp/backend/requirements.txt` (fastapi, uvicorn) dentro de `.venv/`.
3. Ejecuta `npm install` en `webapp/frontend/`.

No hace falta activar el entorno: todos los targets del `Makefile` usan `.venv/bin/python` directamente.
Volver a ejecutar `make install` es seguro; si nada cambió responde `Nothing to be done`.

> **Sin Make** (por ejemplo en Windows sin WSL):
> ```bash
> python -m venv .venv
> source .venv/bin/activate          # Windows: .venv\Scripts\activate
> pip install -r requirements.txt -r webapp/backend/requirements.txt
> cd webapp/frontend && npm install && cd ../..
> ```

`requirements.txt` fija las versiones exactas usadas para obtener los resultados de este documento:
`numpy==2.5.3`, `pandas==3.0.5` y `scikit-learn==1.9.1`.

### Paso 3 · Entrenar y evaluar los modelos

```bash
make models
```

Ejecuta los cinco scripts de `webapp/backend/ml/` en este orden: `random_forest.py` (tarda unos
minutos: 36 configuraciones), `knn.py`, `regresion_logistica.py`, `arboles_de_decision.py` y `bayes.py`.
Para correr uno solo:

```bash
make model M=knn                 # logreg | tree | bayes | knn | rf
```

(equivale a `cd webapp/backend && ../../.venv/bin/python -m ml.knn`; los scripts se ejecutan como
módulos para que encuentren el paquete `scripts/`, y las rutas al CSV se resuelven en `scripts/config.py`).

Cada script imprime, en este orden:

1. Tamaño de la división train / val / test.
2. Tabla con **todas** las configuraciones probadas y sus métricas en validación.
3. Mejor configuración encontrada.
4. Métricas finales en test: accuracy, precision, recall y F1 macro.
5. Reporte por clase y matriz de confusión.
6. Las primeras 20 predicciones frente a su valor real.

Gracias a la semilla fija (`SEMILLA = 42` en `webapp/backend/scripts/config.py`) los resultados son **reproducibles** y coinciden
con los reportados en este documento.

> `random_forest.py` muestra una advertencia de scikit-learn (`X has feature names, but
> RandomForestClassifier was fitted without feature names`). Ocurre porque el modelo final se entrena con
> arreglos de NumPy (train + val concatenados) y se evalúa con un DataFrame. Es inofensiva y no afecta los
> resultados.

### Paso 4 · Interfaz web (opcional)

```bash
make dev
```

Levanta el backend FastAPI en `http://localhost:8000` y el frontend Vite en **`http://localhost:5173`**;
abre esa dirección en el navegador. `Ctrl+C` detiene ambos. Si el puerto 5173 está ocupado, Vite elige el
siguiente libre y lo imprime en la terminal.

Desde la interfaz se puede entrenar cada modelo con distintos hiperparámetros y comparar visualmente los
resultados: ranking, veredicto, matriz de confusión y F1 por actividad. Ver [webapp/README.md](webapp/README.md).

La misma interfaz incluye la **presentación del reto** (botón *Presentación* al pie de la barra lateral, o
directamente `http://localhost:5173/#slides/1`): definición del problema, datos, approach, resultados,
diagnóstico del modelo, despliegue, contribuciones personales y conclusiones. Se navega con `←` / `→`,
`F` pone pantalla completa y `Esc` regresa al laboratorio. No necesita el backend.

También se pueden levantar por separado con `make backend` y `make frontend`.

### Paso 5 (opcional) · Regenerar el dataset tabular

```bash
make dataset
```

Lee los `.npy` de `Dataset/` y produce `dataset_ml_ventanas.csv`. El archivo ya está en el repositorio,
así que este paso solo es necesario si se modifican las señales o la extracción de características.

### Resumen de targets

| Comando | Qué hace |
|---|---|
| `make install` | Crea `.venv/` e instala todas las dependencias (Python + frontend) |
| `make models` | Entrena y evalúa los cinco modelos en terminal |
| `make model M=knn` | Entrena uno solo (`logreg`, `tree`, `bayes`, `knn`, `rf`) |
| `make dev` | Interfaz web: backend + frontend juntos |
| `make backend` / `make frontend` | Cada servidor por separado |
| `make dataset` | Regenera `dataset_ml_ventanas.csv` |
| `make clean` | Borra `.venv/`, `node_modules/` y `__pycache__/` para empezar de cero |
| `make help` | Lista los targets |

---

## 9. Consideraciones éticas y normativas

- **Origen y consentimiento de los datos.** REHAB es un dataset público publicado en *Scientific Data* (2026)
  bajo licencia abierta; los autores reportan aprobación de comité de ética y consentimiento informado de los
  participantes. Este proyecto solo usa las señales cinemáticas de los 16 movimientos de entrenamiento y no
  contiene información personal identificable (nombres, fechas, historial clínico).
- **Uso previsto.** El modelo reconoce *qué* ejercicio se ejecutó; no diagnostica ni mide la calidad del
  movimiento ni sustituye la evaluación de un fisioterapeuta o médico. Cualquier uso clínico real requeriría
  validación con pacientes distintos a los del dataset y cumplimiento de la normatividad de dispositivos
  médicos y de datos de salud aplicable (en México, NOM-024-SSA3-2012 sobre expedientes clínicos electrónicos y
  la Ley General de Protección de Datos Personales en Posesión de Sujetos Obligados / LFPDPPP).
- **Transparencia y reproducibilidad.** Todo el código, la división de datos, la semilla y las versiones de
  las bibliotecas están en el repositorio; los resultados reportados se pueden regenerar con `make models`.
- **Sesgo del dataset.** Las señales provienen de un número limitado de sujetos y de un protocolo controlado;
  el desempeño puede no transferirse a otros pacientes, sensores o entornos domésticos.

---

## 10. Limitaciones y trabajo futuro

- **Validación simple, no cruzada.** Se usó una única partición 50/25/25. Una validación cruzada estratificada de 5 pliegues daría intervalos de confianza para las métricas y permitiría afirmar con más seguridad que la diferencia entre Random Forest y KNN no depende de la partición.
- **La actividad 11 es el punto débil** del modelo seleccionado: absorbe muestras de las actividades 4 y 12. Convendría analizar qué tienen en común esos movimientos y añadir características específicas (por ejemplo, del dominio de la frecuencia o de la correlación entre sensores).
- **Redundancia de características.** Media, mediana y cuartiles de un mismo canal están muy correlacionados. Random Forest expone `feature_importances_`, que puede usarse para recortar las 480 features a un subconjunto más pequeño sin perder desempeño y acelerar tanto el entrenamiento como la inferencia.
- **Otros ensambles no evaluados.** Gradient Boosting (por ejemplo `HistGradientBoostingClassifier`) suele superar a Random Forest en datos tabulares y queda como siguiente iteración.
- **Coste de entrenamiento.** La búsqueda en malla de Random Forest es la más lenta del proyecto. Una búsqueda aleatoria o bayesiana reduciría el tiempo si se amplía la malla.
