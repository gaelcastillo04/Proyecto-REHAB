# Proyecto REHAB · atajos de desarrollo
#   make dev       levanta backend (FastAPI :8000) y frontend (Vite :5173); Ctrl+C detiene ambos
#   make install   crea .venv e instala todas las dependencias (proyecto + webapp)
#   make models    entrena y evalúa los cinco modelos desde la terminal
#   make model M=knn  entrena solo uno (logreg | tree | bayes | knn | rf)
#   make help      lista todos los targets

PYTHON   ?= python3
VENV     := .venv
PY       := $(VENV)/bin/python
BPY      := $(abspath $(PY))          # ruta absoluta: los targets del backend hacen cd
BACKEND  := webapp/backend
FRONTEND := webapp/frontend

.PHONY: help dev backend frontend install venv deps-py deps-web dataset models model clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ---------- entorno ----------

$(PY):
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

venv: $(PY) ## crea .venv si no existe

$(VENV)/.deps-py: $(PY) requirements.txt $(BACKEND)/requirements.txt
	$(PY) -m pip install -r requirements.txt -r $(BACKEND)/requirements.txt
	@touch $@

deps-py: $(VENV)/.deps-py ## instala dependencias de Python (proyecto + backend)

$(FRONTEND)/node_modules: $(FRONTEND)/package.json $(FRONTEND)/package-lock.json
	cd $(FRONTEND) && npm install
	@touch $@

deps-web: $(FRONTEND)/node_modules ## instala dependencias del frontend

install: deps-py deps-web ## instala todo

# ---------- webapp ----------

dev: install ## backend + frontend juntos (http://localhost:5173)
	@trap 'kill 0' EXIT INT TERM; \
	( cd $(BACKEND) && $(BPY) -m uvicorn api.main:app --reload --port 8000 ) & \
	( cd $(FRONTEND) && npm run dev ) & \
	wait

backend: deps-py ## solo FastAPI en :8000
	cd $(BACKEND) && $(BPY) -m uvicorn api.main:app --reload --port 8000

frontend: deps-web ## solo Vite en :5173 (proxy /api -> :8000)
	cd $(FRONTEND) && npm run dev

# ---------- pipeline ML ----------

dataset: deps-py ## regenera dataset_ml_ventanas.csv a partir de Dataset/*.npy
	cd $(BACKEND) && $(BPY) -m scripts.crear_dataset_ventanas

# nombre corto -> módulo en webapp/backend/ml/
SCRIPT_logreg := regresion_logistica
SCRIPT_tree   := arboles_de_decision
SCRIPT_bayes  := bayes
SCRIPT_knn    := knn
SCRIPT_rf     := random_forest

models: deps-py ## entrena y evalúa los cinco modelos (rf, knn, logreg, tree, bayes)
	@for m in rf knn logreg tree bayes; do $(MAKE) --no-print-directory model M=$$m || exit 1; done

model: deps-py ## entrena un modelo: make model M=knn  (logreg | tree | bayes | knn | rf)
	@test -n "$(SCRIPT_$(M))" || { echo "Uso: make model M=<logreg|tree|bayes|knn|rf>"; exit 1; }
	cd $(BACKEND) && $(BPY) -m ml.$(SCRIPT_$(M))

clean: ## borra .venv, node_modules y __pycache__
	rm -rf $(VENV) $(FRONTEND)/node_modules
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
