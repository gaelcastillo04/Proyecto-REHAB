"""API que entrena y evalua los modelos del Proyecto REHAB bajo demanda.

Reutiliza data_utils.py del proyecto (misma division 50/25/25 y misma semilla),
de modo que los resultados coinciden con los scripts y con el README.

    uvicorn main:app --reload --port 8000     (desde webapp/backend)
"""
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, log_loss,
    precision_recall_fscore_support, precision_score, recall_score,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
import config  # noqa: E402
from data_utils import cargar_dataset, dividir_dataset  # noqa: E402
from models import (  # noqa: E402
    MODEL_ORDER, MODELS, coerce_params, default_params, public_spec,
)

app = FastAPI(title="REHAB model lab")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---------------------------------------------------------------- datos
_data = {}
_data_lock = threading.Lock()


def get_data():
    with _data_lock:
        if not _data:
            X, y = cargar_dataset(str(PROJECT_DIR / config.DATASET_CSV))
            X_train, X_val, X_test, y_train, y_val, y_test = dividir_dataset(X, y)
            _data.update(
                X_train=X_train.to_numpy(), X_val=X_val.to_numpy(), X_test=X_test.to_numpy(),
                y_train=y_train.to_numpy(), y_val=y_val.to_numpy(), y_test=y_test.to_numpy(),
                n_features=X.shape[1], n_samples=X.shape[0],
                class_counts=y.value_counts().sort_index().tolist(),
            )
        return _data


# ---------------------------------------------------------------- trabajos
JOBS = {}
JOBS_LOCK = threading.Lock()
RUNS = []  # historial de corridas terminadas, en orden
JOB_QUEUE = queue.Queue()  # los jobs se ejecutan de uno en uno (un solo worker)


def _worker():
    while True:
        _run_job(*JOB_QUEUE.get())
        JOB_QUEUE.task_done()


class TrainRequest(BaseModel):
    model_id: str
    params: dict = {}


def _metrics(y_true, y_pred, proba=None):
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if proba is not None:
        m["log_loss"] = float(log_loss(y_true, proba, labels=list(range(16))))
    return m


def _run_job(job_id, model_id, params):
    d = get_data()
    spec = MODELS[model_id]
    try:
        def stage(name, progress):
            with JOBS_LOCK:
                JOBS[job_id].update(stage=name, progress=progress)

        # 1) validacion: entrenar en train, medir en val
        stage("Entrenando en train (2 308) y evaluando en validación", 0.1)
        t0 = time.perf_counter()
        model = spec["build"](params)
        model.fit(d["X_train"], d["y_train"])
        fit_val_s = time.perf_counter() - t0
        pred_val = model.predict(d["X_val"])
        proba_val = model.predict_proba(d["X_val"]) if spec["has_proba"] else None
        val = _metrics(d["y_val"], pred_val, proba_val)

        # 2) modelo final: train + val, evaluar UNA vez en test
        stage("Reentrenando con train + val (3 462) y evaluando en test", 0.55)
        X_final = np.concatenate([d["X_train"], d["X_val"]])
        y_final = np.concatenate([d["y_train"], d["y_val"]])
        t0 = time.perf_counter()
        final = spec["build"](params)
        final.fit(X_final, y_final)
        fit_final_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred_test = final.predict(d["X_test"])
        predict_ms = (time.perf_counter() - t0) * 1000
        proba_test = final.predict_proba(d["X_test"]) if spec["has_proba"] else None
        test = _metrics(d["y_test"], pred_test, proba_test)

        p, r, f, s = precision_recall_fscore_support(
            d["y_test"], pred_test, labels=list(range(16)), zero_division=0)
        per_class = [
            {"label": i, "precision": float(p[i]), "recall": float(r[i]),
             "f1": float(f[i]), "support": int(s[i])}
            for i in range(16)
        ]
        cm = confusion_matrix(d["y_test"], pred_test, labels=list(range(16))).tolist()

        importances = None
        if hasattr(final, "feature_importances_"):
            imp = final.feature_importances_
            top = np.argsort(imp)[::-1][:15]
            names = _feature_names()
            importances = [{"feature": names[i], "importance": float(imp[i])} for i in top]

        result = {
            "id": job_id,
            "model_id": model_id,
            "model_name": spec["name"],
            "short": spec["short"],
            "params": params,
            "val": val,
            "test": test,
            "per_class": per_class,
            "confusion": cm,
            "feature_importances": importances,
            "timing": {
                "fit_val_s": fit_val_s, "fit_final_s": fit_final_s,
                "predict_test_ms": predict_ms,
                "predict_per_sample_us": predict_ms * 1000 / len(d["y_test"]),
            },
            "finished_at": time.time(),
        }
        with JOBS_LOCK:
            JOBS[job_id].update(status="done", progress=1.0, stage="Listo", result=result)
            RUNS.append(result)
    except Exception as exc:  # noqa: BLE001
        with JOBS_LOCK:
            JOBS[job_id].update(status="error", error=str(exc), stage="Error")


_FEATURE_NAMES = []


def _feature_names():
    if not _FEATURE_NAMES:
        import pandas as pd
        cols = pd.read_csv(str(PROJECT_DIR / config.DATASET_CSV), nrows=0).columns
        _FEATURE_NAMES.extend(c for c in cols if c not in ("actividad", "repeticion"))
    return _FEATURE_NAMES


# ---------------------------------------------------------------- endpoints
threading.Thread(target=_worker, daemon=True).start()


@app.get("/api/dataset")
def dataset_info():
    d = get_data()
    return {
        "n_samples": d["n_samples"], "n_features": d["n_features"], "n_classes": 16,
        "split": {"train": len(d["y_train"]), "val": len(d["y_val"]), "test": len(d["y_test"])},
        "class_counts": d["class_counts"], "seed": config.SEMILLA,
    }


@app.get("/api/models")
def list_models():
    return [public_spec(m) for m in MODEL_ORDER]


@app.post("/api/train")
def train(req: TrainRequest):
    if req.model_id not in MODELS:
        raise HTTPException(404, f"Modelo desconocido: {req.model_id}")
    try:
        params = coerce_params(req.model_id, req.params or {})
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    job_id = uuid.uuid4().hex[:10]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id, "model_id": req.model_id, "params": params,
            "status": "running", "stage": "En cola", "progress": 0.0,
            "started_at": time.time(),
        }
    JOB_QUEUE.put((job_id, req.model_id, params))
    return {"job_id": job_id}


@app.post("/api/train-all")
def train_all():
    """Lanza los cinco modelos con su mejor configuracion (la del README)."""
    ids = [train(TrainRequest(model_id=m, params=default_params(m)))["job_id"] for m in MODEL_ORDER]
    with JOBS_LOCK:
        return {"job_ids": ids, "jobs": [dict(JOBS[i]) for i in ids]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "job no encontrado")
        return dict(job)


@app.get("/api/runs")
def runs():
    with JOBS_LOCK:
        return list(RUNS)


@app.delete("/api/runs")
def clear_runs():
    with JOBS_LOCK:
        RUNS.clear()
    return {"ok": True}
