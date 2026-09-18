from pathlib import Path

# Raíz del repositorio (Proyecto-REHAB/): tres niveles arriba: webapp/backend/scripts/config.py
PROJECT_DIR = Path(__file__).resolve().parents[3]

DATASET_DIR = str(PROJECT_DIR / "Dataset")
DATASET_CSV = str(PROJECT_DIR / "dataset_ml_ventanas.csv")
SEMILLA = 42
