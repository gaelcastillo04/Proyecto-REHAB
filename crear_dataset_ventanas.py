import os
import csv
import math
import numpy as np

DATASET_DIR = "Dataset"
OUTPUT_CSV = "dataset_ml_ventanas.csv"

NUM_ACTIVIDADES = 16
NUM_VENTANAS = 4
NUM_CANALES = 6
PUNTOS_POR_REPETICION = 880
TAMANO_VENTANA = 220


def media(datos):
    return sum(datos) / len(datos)


def desviacion_estandar(datos, promedio):
    suma = 0.0
    for x in datos:
        diferencia = x - promedio
        suma += diferencia * diferencia
    return math.sqrt(suma / len(datos))


def percentil(datos_ordenados, p):
    n = len(datos_ordenados)

    if n == 1:
        return datos_ordenados[0]

    posicion = (n - 1) * (p / 100.0)
    inferior = int(math.floor(posicion))
    superior = int(math.ceil(posicion))

    if inferior == superior:
        return datos_ordenados[inferior]

    fraccion = posicion - inferior

    return (
        datos_ordenados[inferior]
        + fraccion
        * (datos_ordenados[superior] - datos_ordenados[inferior])
    )


def asimetria(datos, promedio, desviacion):
    if desviacion == 0:
        return 0.0

    suma = 0.0

    for x in datos:
        z = (x - promedio) / desviacion
        suma += z ** 3

    return suma / len(datos)


def curtosis(datos, promedio, desviacion):
    if desviacion == 0:
        return 0.0

    suma = 0.0

    for x in datos:
        z = (x - promedio) / desviacion
        suma += z ** 4

    return (suma / len(datos)) - 3.0


def extraer_caracteristicas(datos):
    datos = [float(x) for x in datos]
    promedio = media(datos)
    desviacion = desviacion_estandar(datos, promedio)

    ordenados = sorted(datos)

    minimo = ordenados[0]
    maximo = ordenados[-1]

    q1 = percentil(ordenados, 25)
    q2 = percentil(ordenados, 50)
    q3 = percentil(ordenados, 75)

    iqr = q3 - q1

    skew = asimetria(datos, promedio, desviacion)
    kurt = curtosis(datos, promedio, desviacion)

    return [
        promedio,
        desviacion,
        minimo,
        q1,
        q2,
        q3,
        maximo,
        iqr,
        skew,
        kurt
    ]


def extraer_sensor(repeticion):
    caracteristicas = []

    for ventana in range(NUM_VENTANAS):
        inicio = ventana * TAMANO_VENTANA
        fin = inicio + TAMANO_VENTANA

        datos_ventana = repeticion[inicio:fin]

        for canal in range(NUM_CANALES):
            datos_canal = datos_ventana[:, canal]
            caracteristicas.extend(
                extraer_caracteristicas(datos_canal)
            )

    return caracteristicas


def generar_nombres_columnas():
    nombres = [
        "media",
        "desviacion",
        "minimo",
        "q1",
        "q2_mediana",
        "q3",
        "maximo",
        "iqr",
        "asimetria",
        "curtosis"
    ]

    columnas = ["actividad", "repeticion"]

    for sensor in range(1, 3):
        for ventana in range(1, 5):
            for canal in range(1, 7):
                for nombre in nombres:
                    columnas.append(
                        f"s{sensor}_w{ventana}_c{canal}_{nombre}"
                    )

    return columnas


def crear_dataset():
    columnas = generar_nombres_columnas()
    total_filas = 0

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as archivo_csv:

        escritor = csv.writer(archivo_csv)
        escritor.writerow(columnas)

        for actividad in range(NUM_ACTIVIDADES):
            nombre = f"{actividad:03d}"

            ruta1 = os.path.join(
                DATASET_DIR,
                f"{nombre}_1.npy"
            )

            ruta2 = os.path.join(
                DATASET_DIR,
                f"{nombre}_2.npy"
            )

            if not os.path.exists(ruta1):
                raise FileNotFoundError(
                    f"No existe: {ruta1}"
                )

            if not os.path.exists(ruta2):
                raise FileNotFoundError(
                    f"No existe: {ruta2}"
                )

            sensor1 = np.load(ruta1)
            sensor2 = np.load(ruta2)

            print()
            print(f"Actividad {nombre}")
            print("Sensor 1:", sensor1.shape)
            print("Sensor 2:", sensor2.shape)

            if sensor1.shape[0] != sensor2.shape[0]:
                raise ValueError(
                    "Los sensores tienen diferente numero de repeticiones."
                )

            for sensor in [sensor1, sensor2]:
                if sensor.shape[1] != 880:
                    raise ValueError(
                        "Se esperaban 880 puntos temporales."
                    )

                if sensor.shape[2] != 6:
                    raise ValueError(
                        "Se esperaban 6 canales."
                    )

            for repeticion in range(sensor1.shape[0]):
                f1 = extraer_sensor(
                    sensor1[repeticion]
                )

                f2 = extraer_sensor(
                    sensor2[repeticion]
                )

                fila = [
                    actividad,
                    repeticion
                ] + f1 + f2

                escritor.writerow(fila)
                total_filas += 1

            print(
                "Repeticiones procesadas:",
                sensor1.shape[0]
            )

    print()
    print("=" * 60)
    print("DATASET CREADO CORRECTAMENTE")
    print("=" * 60)
    print("Archivo:", OUTPUT_CSV)
    print("Filas:", total_filas)
    print("Caracteristicas por repeticion: 480")
    print(
        "Columnas totales:",
        len(columnas)
    )


if __name__ == "__main__":
    crear_dataset()
