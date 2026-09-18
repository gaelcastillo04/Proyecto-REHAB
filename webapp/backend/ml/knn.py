import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    log_loss
)

from scripts.data_utils import (
    cargar_dataset,
    dividir_dataset
)


#Hiperparametros que se probaran en el conjunto de validacion
VALORES_K=[1, 3, 5, 7, 9]
TIPOS_PESOS=["uniform", "distance"]


#Funcion que crea un modelo KNN
def crear_modelo(k, pesos):
    modelo=Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=k,
                weights=pesos,
                metric="euclidean",
                n_jobs=-1
            ))
        ]
    )
    return modelo


#Funcion que prueba los diferentes hiperparametros y selecciona la mejor configuracion usando validacion
def seleccionar_hiperparametros(X_train, y_train, X_val, y_val):
    mejor_f1=-1.0
    mejor_k=None
    mejores_pesos=None

    print("Busqueda de hiperparametros:\n")

    for k in VALORES_K:
        for pesos in TIPOS_PESOS:
            modelo=crear_modelo(k, pesos)
            modelo.fit(X_train, y_train)

            pred_val=modelo.predict(X_val)
            prob_val=modelo.predict_proba(X_val)

            #Metricas de desempeño
            accuracy=accuracy_score(y_val, pred_val)
            f1=f1_score(y_val, pred_val, average="macro", zero_division=0)
            entropia=log_loss(y_val, prob_val)

            print(
                f"K={k:<2} | "
                f"pesos={pesos:<8} | "
                f"Accuracy={accuracy*100:.2f}% | "
                f"F1-macro={f1:.4f} | "
                f"Cross-Entropy={entropia:.4f}"
            )

            #Actualizamos los mejores hiperparametros
            if f1>mejor_f1:
                mejor_f1=f1
                mejor_k=k
                mejores_pesos=pesos

    return (mejor_k, mejores_pesos, mejor_f1)


#Funcion que calcula y muestra las metricas del modelo
def evaluar_modelo(modelo, X_test, y_test):
    pred_test=modelo.predict(X_test)

    accuracy=accuracy_score(y_test, pred_test)
    precision=precision_score(y_test, pred_test, average="macro", zero_division=0)
    recall=recall_score(y_test, pred_test, average="macro", zero_division=0)
    f1=f1_score(y_test, pred_test, average="macro", zero_division=0)

    print()
    print("Resultados finales en Test:\n")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision macro: {precision:.4f}")
    print(f"Recall macro: {recall:.4f}")
    print(f"F1-macro: {f1:.4f}")

    print()
    print("Reporte por clase:\n")
    print(classification_report(y_test, pred_test, digits=4, zero_division=0))

    print("Matriz de confusion")
    print(confusion_matrix(y_test, pred_test))

    print()
    print("Primeras 20 predicciones:\n")
    y_test_array=np.array(y_test)

    for i in range(min(20, len(y_test_array))):
        print(
            f"Muestra {i:03d} | "
            f"Real: {int(y_test_array[i]):02d} | "
            f"Prediccion: {int(pred_test[i]):02d}"
        )


#Funcion principal que ejecuta el flujo del modelo KNN
def main():
    print("K-Nearest Neighbors (KNN)")
    print("Division del conjunto de datos:50% train, 25% validacion, 25% test")
    print("="*50)

    #Se carga el dataset
    X,y=cargar_dataset()

    print("Muestras totales:", X.shape[0])
    print("Caracteristicas:", X.shape[1])
    print("Clases:", y.unique())

    #Dividir el dataset
    (X_train, X_val, X_test, y_train, y_val, y_test)=dividir_dataset(X,y)

    print()
    print("Division del dataset")
    print(f"Entrenamiento: {len(X_train)}")
    print(f"Validacion: {len(X_val)}")
    print(f"Test: {len(X_test)}")

    #Buscar la mejor combinacion de hiperparametros
    (mejor_k, mejores_pesos, mejor_f1)=seleccionar_hiperparametros(
        X_train,
        y_train,
        X_val,
        y_val
    )

    print()
    print("Mejor configuracion:\n")
    print(f"K: {mejor_k}")
    print(f"Pesos: {mejores_pesos}")
    print(f"F1-score: {mejor_f1:.4f}")

    #Se unen train y validation para entrenar el modelo final
    X_train_final=np.concatenate([
        np.array(X_train),
        np.array(X_val)
    ], axis=0)

    y_train_final=np.concatenate([
        np.array(y_train),
        np.array(y_val)
    ], axis=0)

    #Crear el modelo con los mejores hiperparametros
    modelo_final=crear_modelo(mejor_k, mejores_pesos)

    #Entrenar el modelo final con train+validation
    modelo_final.fit(X_train_final, y_train_final)

    #Evaluar una sola vez en test
    evaluar_modelo(modelo_final, X_test, y_test)


if __name__=="__main__":
    main()
