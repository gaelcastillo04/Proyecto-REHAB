import pandas as pd

from sklearn.model_selection import train_test_split

from config import (
    DATASET_CSV,
    SEMILLA
)


#Funcion que carga el dataset siendo X las caracteristicas y y la variable objetivo
def cargar_dataset(ruta=DATASET_CSV):
    df=pd.read_csv(ruta)
    y=df["actividad"]

    X=df.drop(columns=["actividad", "repeticion"])

    return X, y


#Funcion que divide el dataset: 50% entrenamiento, 25% de validacion y 25% de prueba
def dividir_dataset(X, y):
    #primera division: 50% entrenamiento y 50% temporal
    X_train, X_temp, y_train, y_temp=train_test_split(
        X,
        y,
        test_size=0.50,
        random_state=SEMILLA,
        stratify=y
    )
    #segunda division: el 50% temporal se divide en 25% validacion y 25% prueba
    X_val, X_test, y_val, y_test=train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=SEMILLA,
        stratify=y_temp
    )
    return(X_train, X_val, X_test, y_train, y_val, y_test)

