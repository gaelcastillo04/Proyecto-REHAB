import numpy as np
from sklearn.naive_bayes import GaussianNB
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
#var_smoothing: fraccion de la varianza maxima que se suma a todas las varianzas para estabilizar el calculo
VALORES_VAR_SMOOTHING=[1e-9, 1e-6, 1e-3, 1e-1]
#priors: None usa la frecuencia de cada clase en train, "uniforme" asigna la misma probabilidad a priori a todas
PRIORS=[None, "uniforme"]
#escalar: si se estandarizan las caracteristicas antes de entrenar
ESCALAR=[False, True]


#Funcion que crea un modelo Naive Bayes Gaussiano
#Cada clase tiene su propia media y varianza por caracteristica, por lo que la
#frontera de decision entre clases es cuadratica (no lineal como en regresion logistica)
def crear_modelo(var_smoothing, priors, escalar, n_clases):
    if priors=="uniforme":
        priors_valores=np.full(n_clases, 1.0/n_clases)
    else:
        priors_valores=None

    bayes=GaussianNB(
        var_smoothing=var_smoothing,
        priors=priors_valores
    )

    pasos=[]
    if escalar:
        pasos.append(("scaler", StandardScaler()))
    pasos.append(("clasificador", bayes))

    modelo=Pipeline(steps=pasos)
    return modelo


#Funcion que prueba los hiperparametros y selecciona la mejor configuracion usando el conjunto de validacion
def seleccionar_hiperparametros(X_train, y_train, X_val, y_val, n_clases):
    #se inicializan los hiperparametros
    mejor_f1=-1.0
    mejor_var_smoothing=None
    mejor_priors=None
    mejor_escalar=None

    print("Busqueda de hiperparametros:\n")

    #se prueban las 16 combinaciones de hiperparametros
    for var_smoothing in VALORES_VAR_SMOOTHING:
        for priors in PRIORS:
            for escalar in ESCALAR:
                modelo=crear_modelo(var_smoothing, priors, escalar, n_clases)
                modelo.fit(X_train, y_train)
                pred_val=modelo.predict(X_val)
                prob_val=modelo.predict_proba(X_val)

                #metricas de desempeño
                accuracy=accuracy_score(y_val, pred_val)
                f1=f1_score(y_val, pred_val, average="macro", zero_division=0)
                entropia=log_loss(y_val, prob_val)

                print(
                    f"var_smoothing={var_smoothing:<6.0e} | "
                    f"priors={str(priors):<8} | "
                    f"escalar={str(escalar):<5} | "
                    f"Accuracy={accuracy*100:.2f}% | "
                    f"F1-macro={f1:.4f} | "
                    f"Cross-Entropy={entropia:.4f}"
                )

                #actualizamos los hiperparametros
                if f1>mejor_f1:
                    mejor_f1=f1
                    mejor_var_smoothing=var_smoothing
                    mejor_priors=priors
                    mejor_escalar=escalar

    return (mejor_var_smoothing, mejor_priors, mejor_escalar, mejor_f1)


#Funcion que calcula y muestra las metricas de nuestro modelo
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
    print(classification_report(
        y_test,
        pred_test,
        digits=4,
        zero_division=0
    ))

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


#Funcion principal que ejecuta el flujo del modelo Naive Bayes
def main():
    print("Naive Bayes Gaussiano")
    print("Frontera de decision cuadratica (varianza propia por clase)")
    print("Division del conjunto de datos:50% train, 25% validacion, 25% test")
    print("="*50)

    #Se carga el dataset
    X,y=cargar_dataset()
    n_clases=y.nunique()

    print("Muestras totales:", X.shape[0])
    print("Caracteristicas:", X.shape[1])
    print("Clases:", y.unique())

    #Dividir el dataset en 50% entrenamiento, 25% validacion, 25% prueba
    (X_train, X_val, X_test, y_train, y_val, y_test)=dividir_dataset(X,y)

    print()
    print("Division del dataset")
    print(f"Entrenamiento: {len(X_train)}")
    print(f"Validacion: {len(X_val)}")
    print(f"Test: {len(X_test)}")

    #Buscar la mejor combinacion de hiperparametros
    (
        mejor_var_smoothing,
        mejor_priors,
        mejor_escalar,
        mejor_f1
    )=seleccionar_hiperparametros(
        X_train,
        y_train,
        X_val,
        y_val,
        n_clases
    )

    print()
    print("Mejor configuracion:\n")
    print(f"Var smoothing: {mejor_var_smoothing}")
    print(f"Priors: {mejor_priors}")
    print(f"Escalar: {mejor_escalar}")
    print(f"F1-score: {mejor_f1:.4f}")

    #Se unen train y validation para entrenar el modelo final con mas datos
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
        mejor_var_smoothing,
        mejor_priors,
        mejor_escalar,
        n_clases
    )
    #Entrenar modelo final
    modelo_final.fit(X_train_final, y_train_final)
    #Evaluar una sola vez usando el conjunto de prueba
    evaluar_modelo(modelo_final, np.array(X_test), y_test)


if __name__=="__main__":
    main()
