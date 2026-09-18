import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from scripts.data_utils import (
    cargar_dataset,
    dividir_dataset
)


#Hiperparametros que se probaran en el conjunto de validacion
NUM_ARBOLES=[50, 100, 200]
PROFUNDIDADES=[None, 10, 20]
MIN_SAMPLES_SPLIT=[2, 5]
PESOS_CLASE=[None, "balanced"]


#Funcion que crea un modelo Random Forest
def crear_modelo(n_estimators, profundidad, min_samples_split, class_weight):
    modelo=RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=profundidad,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    return modelo


#Funcion que prueba los hiperparametros y selecciona la mejor configuracion
def seleccionar_hiperparametros(X_train, y_train, X_val, y_val):
    mejor_f1=-1.0
    mejor_n_estimators=None
    mejor_profundidad=None
    mejor_min_samples_split=None
    mejor_class_weight=None

    print("Busqueda de hiperparametros:\n")

    for n_estimators in NUM_ARBOLES:
        for profundidad in PROFUNDIDADES:
            for min_samples_split in MIN_SAMPLES_SPLIT:
                for class_weight in PESOS_CLASE:

                    modelo=crear_modelo(
                        n_estimators,
                        profundidad,
                        min_samples_split,
                        class_weight
                    )

                    modelo.fit(X_train, y_train)

                    pred_val=modelo.predict(X_val)

                    accuracy=accuracy_score(y_val, pred_val)
                    f1=f1_score(
                        y_val,
                        pred_val,
                        average="macro",
                        zero_division=0
                    )

                    print(
                        f"n_estimators={n_estimators:<3} | "
                        f"depth={str(profundidad):<4} | "
                        f"min_split={min_samples_split:<2} | "
                        f"class_weight={str(class_weight):<8} | "
                        f"Accuracy={accuracy*100:.2f}% | "
                        f"F1-macro={f1:.4f}"
                    )

                    if f1>mejor_f1:
                        mejor_f1=f1
                        mejor_n_estimators=n_estimators
                        mejor_profundidad=profundidad
                        mejor_min_samples_split=min_samples_split
                        mejor_class_weight=class_weight

    return (
        mejor_n_estimators,
        mejor_profundidad,
        mejor_min_samples_split,
        mejor_class_weight,
        mejor_f1
    )


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


#Funcion principal que ejecuta el flujo del modelo Random Forest
def main():
    print("Random Forest")
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
    (
        mejor_n_estimators,
        mejor_profundidad,
        mejor_min_samples_split,
        mejor_class_weight,
        mejor_f1
    )=seleccionar_hiperparametros(
        X_train,
        y_train,
        X_val,
        y_val
    )

    print()
    print("Mejor configuracion:\n")
    print(f"N estimators: {mejor_n_estimators}")
    print(f"Max depth: {mejor_profundidad}")
    print(f"Min samples split: {mejor_min_samples_split}")
    print(f"Class weight: {mejor_class_weight}")
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
    modelo_final=crear_modelo(
        mejor_n_estimators,
        mejor_profundidad,
        mejor_min_samples_split,
        mejor_class_weight
    )

    #Entrenar modelo final con train+validation
    modelo_final.fit(X_train_final, y_train_final)

    #Evaluar una sola vez en test
    evaluar_modelo(modelo_final, X_test, y_test)


if __name__=="__main__":
    main()