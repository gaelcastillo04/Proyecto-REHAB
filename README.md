#  Actividad: Redefinición de los datos

Esta actividad implementa una técnica de aprendizaje máquina sin utilizar frameworks. El proyecto se divide principalmente en dos etapas:
1. Creación de un nuevo dataset a partir de las señales originales.
2. Implementación manual de un modelo de regresión lineal.
---

## 1. Estructura del proyecto

La estructura del proyecto es la siguiente:

```text
preparacion_REHAB/
│
├── Dataset/
│   ├── 000_1.npy
│   ├── 000_2.npy
│   ├── ...
│   ├── 015_1.npy
│   └── 015_2.npy
│
├── crear_dataset_ventanas.py
├── regresion_lineal.py
└── README.md
```

Después de ejecutar el script de creación del dataset se generará también: `dataset_ml_ventanas.csv`

Por lo tanto, la estructura quedará:

```text
proyecto_ml/
│
├── Dataset/
├── crear_dataset_ventanas.py
├── regresion_lineal.py
├── dataset_ml_ventanas.csv
└── README.md
```

---

## 2. Creación del nuevo dataset
El dataset original esta compuesto por archivos `.npy` que son correspondientes a 16 actividades diferentes, cada actividad tiene dos archivos asociados a sensores (el primero se refiere a dos sensores inerciales y el segundo a un guante de flexion). Cada repeticion contiene 880 puntos temporales y 6 canales por sensor.

Para generar un nuevo dataset para una técnica clásica de aprendizaje de máquina, cada repetición de 880 puntos temporales fue dividida en 4 ventanas de 220 puntos temporales cada una. En cada ventana, para cada uno de los 6 canales de ambos sensores se calcularon  las siguientes características estadísticas:
* Media
* Desviación estándar
* Mínimo
* Primer cuartil (Q1)
* Segundo cuartil o mediana (Q2)
* Tercer cuartil (Q3)
* Máximo
* Rango intercuartílico (IQR)
* Asimetría
* Curtosis

De esta forma cada repeticion queda represantada por:
4 ventanas × 6 canales × 2 sensores × 10 caracteristicas= 480 caracteristicas.

Cada fila del nuevo dataset representa una repetición completa de una actividad. Además, se incluye las columnas "actividad" que corresponde a la variable objetivo y "repetición" utilizada únicamente como identificador

El nuevo dataset se almacena en el archivo: `dataset_ml_ventanas.csv`

---

## 3. Implementación de un modelo de regresión lineal
Se implementó un modelo de regresión lineal utilizando exclusivamente las librerías `NumPy` y `Pandas` para el manejo de datos y las operaciones matemáticas, sin recurrir a ningún framework de aprendizaje automático.

El modelo utiliza las 480 características generadas previamente como variables de entrada y utiliza la columna "actividad" como variable objetivo.

Antes del entrenamiento del modelo, el conjunto de datos se dividió de la siguiente manera:
* 60% para entrenamiento
* 20% para validación
* 20% para prueba

Después, las características son normalizadas utilizando Z-score. La media y la desviación estándar utilizadas para la normalización se calculan únicamente con el conjunto de entrenamiento. De esta manera evitamos utilizar información de validación o de prueba durante la etapa de entrenamiento.


La regresión lineal se implementa con la siguiente ecuación:
`ŷ = Xw + b`

donde `X` representa las características de entrada, `w` los pesos aprendidos por el modelo (480 características = 480 pesos aprendidos) y `b` el bias o intercepto.

Los pesos y el bias se inicializan en cero y se actualizan durante el entrenamiento  con el descenso de gradiente. Como función de error, se utilizó el error cuadrático medio (MSE) y el entrenamiento se realizó durante 200 épocas con un learning rate de 0.001.

Finalmente, para evaluar el modelo se utilizan métricas de error cuadrático medio (MSE), error absoluto medio (MAE) y un accuracy aproximado. Debido a que la regresión lineal genera valores continuos, las predicciones se redondean al entero más cercano entre 0 y 15 para compararlas con las actividades reales.

---

---

## 4. Ejecución del proyecto
Primero, se debe generar el nuevo dataset:

```bash
python crear_dataset_ventanas.py
```

Esto creará el archivo:

`dataset_ml_ventanas.csv`

En este caso, se subió la implementación junto con el archivo generado.

Después, se puede ejecutar el modelo de regresión lineal con:

```bash
python regresion_lineal.py
```

El programa mostrará la división del dataset, el proceso de entrenamiento, las métricas de validación y prueba, y algunas predicciones realizadas por el modelo.

---

## 5. Limitación de la regresión lineal
Por último, algo importante a tener en cuenta es que la regresión lineal se utiliza en este proyecto como una primera aproximación experimental.

Las actividades de 0 a 15 representan categorías y no valores continuos. Debido a esta razón, la regresión lineal no es el modelo más adecuado. Las predicciones numéricas del modelo se redondean al entero más cercano entre 0 y 15 únicamente para poder compararlas con las actividades reales.
