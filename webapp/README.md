# REHAB · Model Lab

Interfaz web para **entrenar, ajustar y comparar** los cinco clasificadores del Proyecto REHAB
(regresión logística, árbol de decisión, Naive Bayes, KNN y Random Forest) sin tocar la terminal,
y ver de forma visual cuál es el mejor para reconocer las 16 actividades.

```
webapp/
├── backend/          Todo el Python
│   ├── ml/           un script por modelo: regresion_logistica, arboles_de_decision, bayes, knn, random_forest
│   ├── scripts/      pipeline compartido: config.py (rutas, semilla), data_utils.py (carga y split),
│   │                 crear_dataset_ventanas.py (extracción de características)
│   ├── api/          FastAPI · entrena los modelos bajo demanda con scikit-learn
│   │   ├── main.py   endpoints (/api/models, /api/train, /api/jobs/{id}, /api/runs, /api/train-all)
│   │   └── models.py registro de modelos: hiperparámetros expuestos + constructor sklearn
│   └── requirements.txt
└── frontend/         Vite + React 19 + TypeScript + Recharts
    └── src/
        ├── App.tsx                 layout, navegación por hash (#rf, #knn, #slides/N), cola de trabajos
        ├── components/Presentation presentación del reto (9 slides, sin backend), botón al pie del sidebar
        ├── components/Leaderboard  ranking, veredicto automático, tabla comparativa, F1 por clase
        ├── components/ModelLab     formulario de hiperparámetros + resultados de un modelo
        └── components/charts       gráficas (F1 por actividad, ranking, matriz de confusión, importancias)
```

## Ejecutar

```bash
# desde Proyecto-REHAB/ — crea .venv e instala dependencias si hace falta
make dev
# abre http://localhost:5173
```

O por separado (`make backend` / `make frontend`), o a mano:

```bash
cd webapp/backend  && ../../.venv/bin/python -m pip install -r requirements.txt \
                   && ../../.venv/bin/uvicorn api.main:app --reload --port 8000
cd webapp/frontend && npm install && npm run dev      # proxy /api → :8000
```

## Qué hace

- **Mismo protocolo que los scripts**: reutiliza `scripts/data_utils.py` (división estratificada 50/25/25,
  semilla 42). Cada corrida entrena en *train*, mide en *validación*, reentrena con *train + val* y
  evalúa **una sola vez** en *test*. Con la configuración por defecto de cada modelo se reproducen
  exactamente los números del README principal (RF 95.67 % / F1 0.9556, KNN 92.89 %, …).
- **Por modelo**: selects con los mismos valores de la búsqueda en malla, métricas (accuracy, precision,
  recall, F1-macro, cross-entropy cuando aplica, tiempos de entrenamiento e inferencia), F1 por actividad
  superpuesto contra el mejor modelo global, matriz de confusión, top-15 `feature_importances_` (árbol y RF)
  e historial de configuraciones probadas.
- **Comparación**: botón para lanzar los cinco modelos con su mejor configuración, ranking por
  F1-macro o accuracy en test o validación, tabla comparativa con Δ val→test, peor clase y costo,
  mapa de calor de F1 por actividad para todos los modelos, y un **veredicto** generado a partir de las
  métricas (ventaja sobre el segundo, estabilidad val→test, clases por encima de 0.90, consistencia,
  costo). La mejor corrida de cada modelo se elige por F1-macro en validación, igual que en el proyecto.
- **Presentación** del reto integrada (`#slides/1` o el botón *Presentación* del sidebar): problema, datos,
  approach, resultados, diagnóstico, despliegue, contribuciones y conclusiones. Teclas `←`/`→`, `F`
  (pantalla completa), `Esc` (volver). Los números son los del README principal, no dependen del backend.
- Tema claro/oscuro (auto, o `?theme=light|dark`), paleta categórica fija por modelo validada para
  daltonismo, todo en español.

Las corridas viven en memoria del backend: al reiniciar `uvicorn` se pierden.
