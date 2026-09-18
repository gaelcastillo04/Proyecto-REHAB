"""Registro de modelos: cada entrada describe sus hiperparametros (para la GUI)
y sabe construir el estimador de scikit-learn con esa configuracion.
Los valores por defecto son la mejor configuracion encontrada en el README."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEMILLA = 42
N_CLASES = 16


def _opt(name, label, values, default, help_text):
    return {
        "name": name,
        "label": label,
        "type": "select",
        "options": [{"value": v, "label": "None" if v is None else str(v)} for v in values],
        "default": default,
        "help": help_text,
    }


def _make_logreg(p):
    reg = LogisticRegression(C=p["C"], class_weight=p["class_weight"], solver="lbfgs", max_iter=4000)
    return Pipeline([("scaler", StandardScaler()), ("clf", OneVsRestClassifier(reg))])


def _make_tree(p):
    return DecisionTreeClassifier(
        criterion=p["criterion"], max_depth=p["max_depth"],
        min_samples_split=p["min_samples_split"], class_weight=p["class_weight"],
        random_state=SEMILLA,
    )


def _make_bayes(p):
    priors = np.full(N_CLASES, 1.0 / N_CLASES) if p["priors"] == "uniforme" else None
    steps = [("scaler", StandardScaler())] if p["escalar"] else []
    steps.append(("clf", GaussianNB(var_smoothing=p["var_smoothing"], priors=priors)))
    return Pipeline(steps)


def _make_knn(p):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"],
                                     metric="euclidean", n_jobs=-1)),
    ])


def _make_rf(p):
    return RandomForestClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        min_samples_split=p["min_samples_split"], class_weight=p["class_weight"],
        random_state=SEMILLA, n_jobs=-1,
    )


MODELS = {
    "logreg": {
        "id": "logreg",
        "name": "Regresión logística",
        "short": "LogReg",
        "family": "Lineal · One-vs-Rest",
        "description": "16 clasificadores binarios sobre features estandarizadas. Frontera lineal; prueba si las actividades son linealmente separables.",
        "scales": True,
        "has_proba": True,
        "script": "regresion_logistica.py",
        "params": [
            _opt("C", "C (inverso de regularización L2)", [0.01, 0.1, 1.0, 10.0], 1.0,
                 "Valores bajos regularizan más (subajuste); altos sobreajustan."),
            _opt("class_weight", "class_weight", [None, "balanced"], "balanced",
                 "'balanced' compensa el ligero desbalance de clases."),
        ],
        "build": _make_logreg,
    },
    "tree": {
        "id": "tree",
        "name": "Árbol de decisión",
        "short": "Árbol",
        "family": "Umbrales · no lineal",
        "description": "Un solo árbol con divisiones por umbral. Sin escalado. Captura interacciones pero es inestable con 480 features.",
        "scales": False,
        "has_proba": False,
        "script": "arboles_de_decision.py",
        "params": [
            _opt("criterion", "criterion", ["gini", "entropy"], "entropy", "Medida de impureza."),
            _opt("max_depth", "max_depth", [None, 5, 10, 20], 10, "Profundidad máxima del árbol."),
            _opt("min_samples_split", "min_samples_split", [2, 5, 10], 2, "Mínimo de muestras para dividir un nodo."),
            _opt("class_weight", "class_weight", [None, "balanced"], "balanced", "Compensación del desbalance."),
        ],
        "build": _make_tree,
    },
    "bayes": {
        "id": "bayes",
        "name": "Naive Bayes gaussiano",
        "short": "Bayes",
        "family": "Probabilístico · cuadrático",
        "description": "Asume features normales e independientes dada la clase. Muy rápido, pero el supuesto de independencia se viola con estadísticos correlacionados.",
        "scales": False,
        "has_proba": True,
        "script": "bayes.py",
        "params": [
            _opt("var_smoothing", "var_smoothing", [1e-9, 1e-6, 1e-3, 1e-1], 1e-9,
                 "Fracción de la varianza máxima sumada a todas las varianzas."),
            _opt("priors", "priors", [None, "uniforme"], "uniforme", "None = frecuencia de clase."),
            _opt("escalar", "StandardScaler", [False, True], False, "Estandarizar antes de entrenar."),
        ],
        "build": _make_bayes,
    },
    "knn": {
        "id": "knn",
        "name": "K-Nearest Neighbors",
        "short": "KNN",
        "family": "No paramétrico · local",
        "description": "Vota entre los K vecinos más cercanos (euclidiana) en el espacio estandarizado de 480 dimensiones.",
        "scales": True,
        "has_proba": True,
        "script": "knn.py",
        "params": [
            _opt("n_neighbors", "K (n_neighbors)", [1, 3, 5, 7, 9], 1, "Cantidad de vecinos que votan."),
            _opt("weights", "weights", ["uniform", "distance"], "uniform", "Voto igualitario o ponderado por cercanía."),
        ],
        "build": _make_knn,
    },
    "rf": {
        "id": "rf",
        "name": "Random Forest",
        "short": "RF",
        "family": "Ensamble de árboles",
        "description": "Promedia árboles entrenados con bootstrap y √480 ≈ 22 features por nodo. Reduce la varianza del árbol individual.",
        "scales": False,
        "has_proba": True,
        "script": "random-forest.py",
        "params": [
            _opt("n_estimators", "n_estimators", [50, 100, 200], 100, "Número de árboles."),
            _opt("max_depth", "max_depth", [None, 10, 20], None, "Profundidad máxima de cada árbol."),
            _opt("min_samples_split", "min_samples_split", [2, 5], 2, "Mínimo de muestras para dividir."),
            _opt("class_weight", "class_weight", [None, "balanced"], "balanced", "Compensación del desbalance."),
        ],
        "build": _make_rf,
    },
}

MODEL_ORDER = ["logreg", "tree", "bayes", "knn", "rf"]


def public_spec(model_id):
    m = MODELS[model_id]
    return {k: v for k, v in m.items() if k != "build"}


def coerce_params(model_id, raw):
    """La GUI manda JSON; aqui se normaliza cada valor al tipo del option correspondiente."""
    spec = MODELS[model_id]
    out = {}
    for p in spec["params"]:
        value = raw.get(p["name"], p["default"])
        allowed = [o["value"] for o in p["options"]]
        if value not in allowed:
            # tolera "None"/"null" y numeros llegados como string
            match = next((a for a in allowed if str(a) == str(value)), None)
            if match is None and value in ("null", "None", ""):
                match = None
            if match is None and None not in allowed:
                raise ValueError(f"{p['name']}={value!r} no es válido; opciones: {allowed}")
            value = match
        out[p["name"]] = value
    return out


def default_params(model_id):
    return {p["name"]: p["default"] for p in MODELS[model_id]["params"]}
