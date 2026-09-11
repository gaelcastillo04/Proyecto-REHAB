import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
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

from data_utils import (
    cargar_dataset,
    dividir_dataset
) 


#Hiperparametros que se probaran en el conjunto de validacion
VALORES_C=[0.01, 0.1, 1.0, 10.0]
PESOS_CLASE=[None, "balanced"]

#Funcion que crea un modelo de regresion logistica multiclase se utiliza la estrategia One vs Rest

def crear_modelo(C, class_weight):
    regresion=LogisticRegression(
        C=C,
        class_weight= class_weight,
        solver="lbfgs",
        max_iter=4000
    )

    modelo= Pipeline(
        steps=[("scaler", StandardScaler()),("clasificador", OneVsRestClassifier(regresion))]
    )
    return modelo

#Funcion que prueba los diferentes hiperparameetros y selecciona la mejor configuracion usando el conjunto de validacion
def seleccionar_hiperparametros(X_train, y_train, X_val, y_val):
    #se inicializan los hiperparametros
    mejor_f1= -1.0
    mejor_C= None
    mejor_class_weight=None

    print("Busqueda de hiperparametros:\n")
    
    #se preuban las 8 combinacion de hiperparametros
    for C in VALORES_C:
        for class_weight in PESOS_CLASE:
            modelo=crear_modelo(C, class_weight)
            modelo.fit(X_train, y_train)
            pred_val=modelo.predict(X_val)
            prob_val = modelo.predict_proba(X_val)

            #metricas de desempeño
            accuracy=accuracy_score(y_val, pred_val)
            f1=f1_score(y_val, pred_val, average="macro", zero_division=0)
            entropia=log_loss(y_val, prob_val)

            print(
            f"C={C:<5} | "
            f"class_weight={str(class_weight):<8} | "
            f"Accuracy={accuracy * 100:.2f}% | "
            f"F1-macro={f1:.4f} | "
            f"Cross-Entropy={entropia:.4f}"
            )

            #actualizamos los hiperparametros
            if f1> mejor_f1:
                mejor_f1=f1
                mejor_C=C
                mejor_class_weight=class_weight

    return (mejor_C, mejor_class_weight, mejor_f1)


#Funcion que calcula y muestra las metricas de nuestro modelo
def evaluar_modelo(modelo,X_test, y_test):
    pred_test=modelo.predict(X_test)

    accuracy=accuracy_score(y_test, pred_test)
    precision=precision_score(y_test, pred_test, average="macro", zero_division=0)
    recall=recall_score(y_test, pred_test, average="macro", zero_division=0)
    f1=f1_score(y_test, pred_test, average="macro", zero_division=0)

    print()
    print("Resultados finales en Test: \n")
    print(
        f"Acurracy: "
        f"{accuracy*100:.2f}"
    )
    print(
        f"Precision macro: "
        f"{precision:.4f}"
    )
    print(
        f"Recall macro: "
        f"{recall:.4f}"
    )
    print(
        f"F1-macro: "
        f"{f1:.4f}"
    )

    print()
    print("Reporte por clase:\n ")
    print(classification_report(y_test,pred_test, digits=4, zero_division=0))

    print("Matriz de confusion")
    print(confusion_matrix(y_test, pred_test))

    print("Primeras 20 predicciones:\n ")
    y_test_array=np.array(y_test)

    for i in range(min(20, len(y_test_array))):
        print(
            f"Muestra {i:03d} | "
            f"Real: {int(y_test_array[i]):02d} | "
            f"Prediccion: {int(pred_test[i]):02d}"
        )

#Funcion principal que ejecuta el flujo de nuestro modelo de regresion logistica
def main():
    print("Regresion Logistica Multiclase")
    print("Estrategia de One vs Rest")
    print("Division del conjunto de datos:50% train, 25% validacion, 25% test")
    print("=" * 50)

    #Se carga el dataset
    X, y=cargar_dataset()

    print("Muestras totales:", X.shape[0])
    print("Caracteristicas:", X.shape[1])
    print("Clases:", y.unique())

    #Dividir el dataset en 50% entrenamiento, 25% validacion, 25% prueba
    (X_train, X_val, X_test, y_train, y_val, y_test)=dividir_dataset(X, y)
    print()
    print("Division del dataset")
    print(f"Entrenamiento: {len(X_train)}")
    print(f"Validacion: {len(X_val)}")
    print(f"Test: {len(X_test)}")

    #Buscar la mejor combinacion de hiperparametros
    (mejor_C,mejor_class_weight, mejor_f1 )=seleccionar_hiperparametros(X_train, y_train, X_val, y_val)
    print("Mejor configuracion: \n")
    print(f"C: {mejor_C}")
    print(f"Class weight: {mejor_class_weight}")
    print(f"f1_score: {mejor_f1:.4f}")

    #Una vez elegido los hiperparametros, se unen entrenamiento y validacion para entrenar el modelo testing con mas datos
    X_train_final = np.concatenate([
        np.array(X_train),
        np.array(X_val)
        ],
        axis=0
    )
    
    y_train_final = np.concatenate([
        np.array(y_train),
        np.array(y_val)
        ],
        axis=0
    )

    #Creamos el modelo con lo mejores hiperparametros
    modelo_final=crear_modelo(mejor_C, mejor_class_weight)
    #se entrena el modelo con train+validation
    modelo_final.fit(X_train_final, y_train_final)

    #evaluar una sola vez usando nuestro conjunto de pruebas
    evaluar_modelo(modelo_final, X_test, y_test)


if __name__ == "__main__":
    main()



