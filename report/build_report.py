"""Ejecuta los cinco scripts de modelos y construye el reporte PDF completo del Proyecto REHAB.

Uso (desde la raíz del repositorio, Proyecto-REHAB/)::

    make report                              # ejecuta los scripts y construye el PDF
    .venv/bin/python report/build_report.py            # equivalente
    .venv/bin/python report/build_report.py --no-exec  # reutiliza output/logs/*.txt
    .venv/bin/python report/build_report.py --completo # versión larga con código y apéndices

Por defecto se genera la versión corta (máximo 6 páginas, sin código). Con ``--completo`` se
añaden el código fuente, las salidas íntegras de los scripts, la API y los README.

Las salidas capturadas de cada script se escriben en ``output/logs/``, los resultados
parseados en ``output/results/`` y el PDF en ``output/pdf/rehab_project_complete.pdf``.
El código fuente en ``webapp/backend/`` no se modifica.

Los datos de la portada (alumnos, profesor, fecha de entrega) viven en las constantes
justo debajo de los imports.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "webapp" / "backend"
ML_DIR = BACKEND / "ml"
SCRIPTS_DIR = BACKEND / "scripts"
API_DIR = BACKEND / "api"
OUT = ROOT / "output"
LOG_OUT = OUT / "logs"
RES_OUT = OUT / "results"
PDF_PATH = OUT / "pdf" / "rehab_project_complete.pdf"
LOGO_PATH = ROOT / "assets" / "tec_flame.png"
DATASET_DIR = ROOT / "Dataset"
CSV_PATH = ROOT / "dataset_ml_ventanas.csv"

# Orden de ejecución y de aparición en los apéndices.
MODELS = [
    # clave, módulo, nombre, nombre corto, letra de apéndice
    ("logreg", "regresion_logistica", "Regresión logística", "LogReg"),
    ("tree", "arboles_de_decision", "Árbol de decisión", "Árbol"),
    ("bayes", "bayes", "Naive Bayes gaussiano", "Bayes"),
    ("knn", "knn", "K-Nearest Neighbors", "KNN"),
    ("rf", "random_forest", "Random Forest", "RF"),
]
MODEL_COLORS = {
    "rf": "#1f4e79",
    "knn": "#4f81bd",
    "logreg": "#7fa7d4",
    "tree": "#9dbbe1",
    "bayes": "#c4d5eb",
}

# ---- datos de la portada ----------------------------------------------------
TITULO = "Reporte de Proyecto: Clasificación de actividades de rehabilitación (REHAB)"
ALUMNOS = [
    ("Diego Angulo", "A01643797"),
    ("Gael Castillo", "A01638638"),
    ("Diego Ibarra", "A01644350"),
]
PROFESOR = ""  # p. ej. "Nombre Apellido"; el bloque se omite mientras esté vacío
FECHA_ENTREGA = "21 de septiembre de 2026"
ENCABEZADO = "Proyecto REHAB - Reporte de proyecto"


# --------------------------------------------------------------------------- #
# Ejecución de los scripts
# --------------------------------------------------------------------------- #
def execute_scripts() -> None:
    LOG_OUT.mkdir(parents=True, exist_ok=True)
    for key, module, name, _ in MODELS:
        print(f"Ejecutando ml.{module} ({name}) ...", flush=True)
        proc = subprocess.run(
            [sys.executable, "-m", f"ml.{module}"],
            cwd=BACKEND,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        text = proc.stdout
        (LOG_OUT / f"{module}.txt").write_text(text, encoding="utf-8")
        if proc.returncode != 0:
            sys.exit(f"ml.{module} terminó con código {proc.returncode}; revisa {LOG_OUT / f'{module}.txt'}")
        print(f"  guardado {LOG_OUT / f'{module}.txt'}", flush=True)


# --------------------------------------------------------------------------- #
# Parseo de las salidas
# --------------------------------------------------------------------------- #
GRID_RE = re.compile(r"^(?P<params>.+?)\s*\|\s*Accuracy=(?P<acc>[\d.]+)%\s*\|\s*F1-macro=(?P<f1>[\d.]+)(?:\s*\|\s*Cross-Entropy=(?P<ce>[\d.]+))?\s*$")
CLASS_RE = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$")
PRED_RE = re.compile(r"Muestra\s+(\d+)\s*\|\s*Real:\s*(\d+)\s*\|\s*Prediccion:\s*(\d+)")


def _num(s: str):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def parse_log(text: str) -> dict:
    lines = text.splitlines()
    grid = []
    for line in lines:
        m = GRID_RE.match(line.strip())
        if not m:
            continue
        params = {}
        for chunk in m.group("params").split("|"):
            if "=" in chunk:
                k, v = chunk.split("=", 1)
                params[k.strip()] = v.strip()
        row = {"params": params, "accuracy": float(m.group("acc")) / 100, "f1_macro": float(m.group("f1"))}
        if m.group("ce"):
            row["cross_entropy"] = float(m.group("ce"))
        grid.append(row)

    best = {}
    in_best = False
    for line in lines:
        s = line.strip()
        if s.startswith("Mejor configuracion"):
            in_best = True
            continue
        if in_best:
            if s.startswith("Resultados finales"):
                break
            if ":" in s:
                k, v = s.split(":", 1)
                best[k.strip()] = v.strip()

    def grab(pattern):
        m = re.search(pattern, text, flags=re.M)
        return float(m.group(1)) if m else None

    acc = grab(r"^\s*Ac{1,2}ur{1,2}acy:\s*([\d.]+)")
    test = {
        "accuracy": acc / 100 if acc is not None else None,
        "precision_macro": grab(r"^\s*Precision macro:\s*([\d.]+)"),
        "recall_macro": grab(r"^\s*Recall macro:\s*([\d.]+)"),
        "f1_macro": grab(r"^\s*F1-macro:\s*([\d.]+)"),
    }

    per_class = {}
    for line in lines:
        m = CLASS_RE.match(line)
        if m:
            per_class[int(m.group(1))] = {
                "precision": float(m.group(2)),
                "recall": float(m.group(3)),
                "f1": float(m.group(4)),
                "support": int(m.group(5)),
            }

    cm = None
    m = re.search(r"Matriz de confusion\s*(.*?)\s*Primeras 20 predicciones", text, flags=re.S)
    if m:
        ints = [int(x) for x in re.findall(r"-?\d+", m.group(1).replace("[", " ").replace("]", " "))]
        n = int(round(len(ints) ** 0.5))
        if n * n == len(ints):
            cm = [ints[i * n:(i + 1) * n] for i in range(n)]

    preds = [(int(a), int(b), int(c)) for a, b, c in PRED_RE.findall(text)]
    split = {}
    for k, pat in (("train", r"Entrenamiento:\s*(\d+)"), ("val", r"Validacion:\s*(\d+)"), ("test", r"^Test:\s*(\d+)")):
        mm = re.search(pat, text, flags=re.M)
        if mm:
            split[k] = int(mm.group(1))

    return {
        "grid": grid,
        "best": best,
        "best_val_f1": max((r["f1_macro"] for r in grid), default=None),
        "test": test,
        "per_class": per_class,
        "confusion_matrix": cm,
        "predictions": preds,
        "split": split,
    }


def load_results() -> dict:
    RES_OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for key, module, name, short in MODELS:
        log = LOG_OUT / f"{module}.txt"
        if not log.exists():
            sys.exit(f"Falta la salida {log}; ejecuta sin --no-exec")
        r = parse_log(log.read_text(encoding="utf-8"))
        r.update({"key": key, "module": module, "name": name, "short": short})
        missing = [k for k, v in r["test"].items() if v is None]
        if missing or r["confusion_matrix"] is None or not r["grid"]:
            sys.exit(f"No se pudieron extraer métricas de {log} (faltan {missing or 'matriz/grid'})")
        (RES_OUT / f"{key}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        results[key] = r
    return results


# --------------------------------------------------------------------------- #
# Utilidades de texto
# --------------------------------------------------------------------------- #
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

BOX_MAP = str.maketrans({
    "─": "-", "━": "=", "│": "|", "┃": "|",
    "┌": "+", "┏": "+", "┐": "+", "┓": "+",
    "└": "+", "┗": "+", "┘": "+", "┛": "+",
    "├": "+", "┣": "+", "┤": "+", "┫": "+",
    "┬": "+", "┳": "+", "┴": "+", "┻": "+",
    "┼": "+", "╋": "+", "═": "=", "║": "|",
    "╔": "+", "╗": "+", "╚": "+", "╝": "+",
    "╠": "+", "╣": "+", "╦": "+", "╩": "+", "╬": "+",
    "█": "#", "░": ".", "▒": ":", "▓": "#",
    "✓": "v", "✔": "v", "✗": "x", "✘": "x",
    "─": "-", "►": ">", "▶": ">", "→": "->", "←": "<-",
})
for _cp in range(0x2500, 0x25A0):
    if _cp not in BOX_MAP:
        BOX_MAP[_cp] = "-" if 0x2504 <= _cp <= 0x250B or 0x254C <= _cp <= 0x254F else "+"

# Helvetica y Times sólo cubren Latin-1 (más unas cuantas tipográficas); el resto se sustituye.
PROSE_MAP = str.maketrans({
    "→": "->", "←": "<-", "⇒": "=>", "−": "-", "≈": "~", "√": "raiz", "Δ": "Delta",
    "≥": ">=", "≤": "<=", "≠": "!=", "∞": "inf", "·": "·", "×": "x", "✓": "si", "✗": "no",
    " ": " ", " ": " ", " ": " ", "‑": "-",
})
LATIN_EXTRA = set("‘’‚“”„†‡•…‰€™–—˜ˆŒœŠšŸŽž‹›ƒ")


def clean_text(text: str) -> str:
    text = ANSI_RE.sub("", text).translate(BOX_MAP)
    text = "".join(ch if ord(ch) < 256 or ch in "↵" else "?" for ch in text)
    lines = []
    for line in text.split("\n"):
        if "\r" in line:
            line = line.split("\r")[-1]
        lines.append(line.rstrip())
    return "\n".join(lines).strip("\n")


def prose(text: str) -> str:
    text = text.translate(PROSE_MAP)
    return "".join(ch if ord(ch) < 256 or ch in LATIN_EXTRA else "?" for ch in text)


def esc(text: str) -> str:
    return prose(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "log", "-1", "--format=%h - %s"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "desconocido"


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f} %"


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def build_pdf(pdf_path: Path, results: dict, full: bool = False) -> None:
    import numpy as np
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        Preformatted as _Preformatted,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.platypus.flowables import Flowable

    order = [results[k] for k, *_ in MODELS]
    ranking = sorted(order, key=lambda r: r["test"]["f1_macro"], reverse=True)
    best = ranking[0]
    second = ranking[1]
    rf, knn, logreg, tree, bayes = (results[k] for k in ("rf", "knn", "logreg", "tree", "bayes"))
    split = rf["split"] or {"train": 2308, "val": 1154, "test": 1154}
    n_total = sum(split.values())

    styles = getSampleStyleSheet()
    ACCENT = colors.HexColor("#1f4e79")
    TEC_BLUE = colors.HexColor("#1c3f94")
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5, leading=13, spaceAfter=6)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10.5, textColor=colors.HexColor("#555555"))
    caption = ParagraphStyle("caption", parent=small, alignment=TA_CENTER, spaceBefore=2, spaceAfter=10)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15, leading=19, textColor=ACCENT, spaceBefore=6, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11.5, leading=15, textColor=ACCENT, spaceBefore=10, spaceAfter=5)
    h3 = ParagraphStyle("h3", parent=styles["Heading3"], fontSize=9.5, leading=12, textColor=colors.HexColor("#333333"), spaceBefore=8, spaceAfter=3)
    code = ParagraphStyle("code", parent=styles["Code"], fontName="Courier", fontSize=7.2, leading=8.8, leftIndent=4, backColor=colors.HexColor("#f4f6f8"), borderPadding=(4, 4, 4, 4), spaceAfter=4)
    out_style = ParagraphStyle("out", parent=code, backColor=colors.HexColor("#fbf8ee"), textColor=colors.HexColor("#222222"))
    md_style = ParagraphStyle("md", parent=body, leftIndent=4, backColor=colors.HexColor("#ffffff"))
    quote = ParagraphStyle("quote", parent=body, leftIndent=14, textColor=colors.HexColor("#333333"), borderPadding=(2, 2, 2, 6), backColor=colors.HexColor("#f4f6f8"))
    li = ParagraphStyle("li", parent=body, leftIndent=12, spaceAfter=2)
    cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10, spaceAfter=0)
    cell_head = ParagraphStyle("cellhead", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    label_code = ParagraphStyle("labelcode", parent=small, textColor=ACCENT, spaceBefore=6, spaceAfter=2, fontName="Helvetica-Bold")
    label_out = ParagraphStyle("labelout", parent=small, textColor=colors.HexColor("#8a6d1f"), spaceBefore=2, spaceAfter=2, fontName="Helvetica-Bold")

    # Estilos de portada (serif, siguiendo la plantilla de reportes del curso)
    cover_title = ParagraphStyle("covertitle", parent=styles["Title"], fontName="Times-Bold", fontSize=24, leading=30, textColor=TEC_BLUE, alignment=TA_CENTER, spaceAfter=0)
    cover_label = ParagraphStyle("coverlabel", parent=body, fontName="Times-Bold", fontSize=13, leading=17, alignment=TA_CENTER, spaceAfter=0)
    cover_text = ParagraphStyle("covertext", parent=body, fontName="Times-Roman", fontSize=13, leading=17, alignment=TA_CENTER, spaceAfter=0)
    wordmark = ParagraphStyle("wordmark", parent=body, fontName="Times-Roman", fontSize=31, leading=33, textColor=TEC_BLUE, spaceAfter=0)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.6 * cm,
        title=TITULO,
        author=", ".join(name for name, _ in ALUMNOS),
    )
    width = A4[0] - doc.leftMargin - doc.rightMargin

    def on_page(canvas, document):
        if document.page == 1:
            return
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(doc.leftMargin, A4[1] - 1.1 * cm, ENCABEZADO)
        canvas.drawRightString(A4[0] - doc.rightMargin, A4[1] - 1.1 * cm, f"Página {document.page}")
        canvas.restoreState()

    def Preformatted(text, style, **kw):  # noqa: N802 - conserva el nombre de platypus
        kw.setdefault("maxLineLength", 108)
        kw.setdefault("newLineChars", "  ↵ ")
        return _Preformatted(text, style, **kw)

    def grid_style(extra=()):
        return TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d0da")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            *extra,
        ])

    def simple_table(rows, col_fracs, align_center_from=1, extra=()):
        data = [[Paragraph(esc(str(c)), cell_head) for c in rows[0]]]
        for r in rows[1:]:
            data.append([Paragraph(esc(str(c)), cell) for c in r])
        t = Table(data, colWidths=[width * f for f in col_fracs], repeatRows=1)
        st = [("ALIGN", (align_center_from, 1), (-1, -1), "CENTER")] if align_center_from is not None else []
        t.setStyle(grid_style(list(extra) + st))
        return t

    # ---- portada ----
    def logo_lockup():
        emblem_h = 2.6 * cm
        cells = []
        if LOGO_PATH.exists():
            from PIL import Image as PILImage

            w, h = PILImage.open(LOGO_PATH).size
            cells.append(Image(str(LOGO_PATH), width=emblem_h * w / h, height=emblem_h))
        cells.append(Paragraph("Tecnológico<br/>de Monterrey", wordmark))
        t = Table([cells], colWidths=[emblem_h + 0.4 * cm, 7.6 * cm] if len(cells) == 2 else [7.6 * cm], hAlign="CENTER")
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        return t

    def cover():
        flow = [
            Spacer(1, 0.6 * cm),
            logo_lockup(),
            Spacer(1, 4.2 * cm),
            Paragraph(esc(TITULO), cover_title),
            Spacer(1, 1.6 * cm),
            Paragraph("Alumnos", cover_label),
        ]
        for name, sid in ALUMNOS:
            flow.append(Paragraph(f"{esc(name)} - {esc(sid)}", cover_text))
        flow.append(Spacer(1, 5.0 * cm))
        if PROFESOR:
            flow += [Paragraph("Profesor", cover_label), Paragraph(esc(PROFESOR), cover_text), Spacer(1, 0.8 * cm)]
        flow += [
            Paragraph("<b>Fecha de entrega</b>:", cover_label),
            Paragraph(esc(FECHA_ENTREGA), cover_text),
            PageBreak(),
        ]
        return flow

    # ---- diagramas ----
    class PipelineDiagram(Flowable):
        def __init__(self, w, h=3.4 * cm):
            super().__init__()
            self.width, self.height = w, h

        def draw(self):
            c = self.canv
            stages = [
                ("Señales", "880 x 6 x 2 sensores"),
                ("Ventanas", "4 x 220 puntos"),
                ("Estadísticos", "10 por canal"),
                ("Tabla", f"{n_total:,} x 480"),
                ("División", "50 / 25 / 25"),
                ("5 modelos", "malla en val."),
                ("Selección", "F1-macro"),
            ]
            n = len(stages)
            gap = 0.35 * cm
            bw = (self.width - gap * (n - 1)) / n
            bh = 1.7 * cm
            y = (self.height - bh) / 2
            for i, (name, detail) in enumerate(stages):
                x = i * (bw + gap)
                c.setFillColor(colors.HexColor("#eef2f7"))
                c.setStrokeColor(ACCENT)
                c.roundRect(x, y, bw, bh, 6, fill=1, stroke=1)
                c.setFillColor(ACCENT)
                c.setFont("Helvetica-Bold", 8.5)
                c.drawCentredString(x + bw / 2, y + bh - 0.6 * cm, name)
                c.setFont("Helvetica", 7)
                c.setFillColor(colors.HexColor("#333333"))
                c.drawCentredString(x + bw / 2, y + 0.45 * cm, detail)
                if i < n - 1:
                    c.setStrokeColor(ACCENT)
                    c.setLineWidth(1.1)
                    c.line(x + bw + 1, y + bh / 2, x + bw + gap - 1, y + bh / 2)
                    c.line(x + bw + gap - 4, y + bh / 2 + 2.5, x + bw + gap - 1, y + bh / 2)
                    c.line(x + bw + gap - 4, y + bh / 2 - 2.5, x + bw + gap - 1, y + bh / 2)

    class SplitDiagram(Flowable):
        def __init__(self, w, h=3.0 * cm):
            super().__init__()
            self.width, self.height = w, h

        def draw(self):
            c = self.canv
            parts = [
                ("Entrenamiento 50 %", split["train"], "#1f4e79", "ajustar cada configuración"),
                ("Validación 25 %", split["val"], "#4f81bd", "elegir hiperparámetros (F1-macro)"),
                ("Prueba 25 %", split["test"], "#9dbbe1", "evaluación única del modelo final"),
            ]
            bar_h = 1.1 * cm
            y = self.height - bar_h - 0.2 * cm
            x = 0
            for label, n, col, purpose in parts:
                w = self.width * n / n_total
                c.setFillColor(colors.HexColor(col))
                c.setStrokeColor(colors.white)
                c.rect(x, y, w, bar_h, fill=1, stroke=1)
                c.setFillColor(colors.white if col != "#9dbbe1" else colors.HexColor("#1f2a3a"))
                c.setFont("Helvetica-Bold", 8.5)
                c.drawCentredString(x + w / 2, y + bar_h - 0.45 * cm, label)
                c.setFont("Helvetica", 8)
                c.drawCentredString(x + w / 2, y + 0.25 * cm, f"{n:,} muestras")
                c.setFillColor(colors.HexColor("#333333"))
                c.setFont("Helvetica", 7.5)
                c.drawCentredString(x + w / 2, y - 0.5 * cm, purpose)
                x += w
            c.setFillColor(colors.HexColor("#555555"))
            c.setFont("Helvetica-Oblique", 7.5)
            c.drawCentredString(self.width / 2, 0.1 * cm, f"Modelo final: se reentrena con entrenamiento + validación ({split['train'] + split['val']:,} muestras) y se evalúa una sola vez en prueba.")

    # ---- figuras ----
    def figure(make, cap: str, max_h=8.5 * cm):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
        fig = make(plt)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        from PIL import Image as PILImage

        w, h = PILImage.open(buf).size
        buf.seek(0)
        scale = min(width / w, max_h / h)
        return [Image(buf, width=w * scale, height=h * scale), Paragraph(cap, caption)]

    def fig_signal(plt):
        s1 = np.load(DATASET_DIR / "000_1.npy", mmap_mode="r")[0]
        s2 = np.load(DATASET_DIR / "000_2.npy", mmap_mode="r")[0]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
        for ax, sig, title in zip(axes, (s1, s2), ("Sensor 1: unidades inerciales (6 canales)", "Sensor 2: guante de flexión (6 canales)")):
            for ch in range(sig.shape[1]):
                ax.plot(np.asarray(sig[:, ch]), lw=0.9, label=f"canal {ch + 1}")
            for b in (220, 440, 660):
                ax.axvline(b, color="#1f4e79", ls="--", lw=0.8, alpha=0.6)
            ax.set_title(title)
            ax.set_xlabel("Punto temporal (0-879)")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Valor de la señal")
        axes[1].legend(fontsize=7, ncol=2, loc="upper right")
        fig.tight_layout()
        return fig

    def fig_classes(plt):
        import pandas as pd

        counts = pd.read_csv(CSV_PATH, usecols=["actividad"])["actividad"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 3.4))
        bars = ax.bar([str(i) for i in counts.index], counts.values, color="#4f81bd")
        ax.axhline(counts.mean(), color="#1f4e79", ls="--", lw=1, label=f"media = {counts.mean():.0f}")
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x() + b.get_width() / 2, v + 4, str(v), ha="center", fontsize=8)
        ax.set_xlabel("Actividad")
        ax.set_ylabel("Repeticiones")
        ax.set_title("Repeticiones por actividad en el dataset tabular")
        ax.set_ylim(0, counts.max() * 1.12)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend()
        fig.tight_layout()
        return fig

    def fig_summary(plt):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
        names = [r["short"] for r in ranking]
        cols = [MODEL_COLORS[r["key"]] for r in ranking]
        acc = [r["test"]["accuracy"] * 100 for r in ranking]
        f1 = [r["test"]["f1_macro"] for r in ranking]
        bars = axes[0].bar(names, acc, color=cols, edgecolor="#1f4e79", linewidth=0.5)
        axes[0].set_ylim(min(acc) - 8, 100)
        axes[0].set_ylabel("Accuracy en prueba (%)")
        axes[0].set_title("Accuracy en prueba")
        for b, v in zip(bars, acc):
            axes[0].text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.2f}%", ha="center", fontsize=8.5)
        bars = axes[1].bar(names, f1, color=cols, edgecolor="#1f4e79", linewidth=0.5)
        axes[1].set_ylim(min(f1) - 0.08, 1.0)
        axes[1].set_ylabel("F1-macro en prueba")
        axes[1].set_title("F1-macro en prueba (métrica de selección)")
        for b, v in zip(bars, f1):
            axes[1].text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.4f}", ha="center", fontsize=8.5)
        for ax in axes:
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)
        fig.tight_layout()
        return fig

    def fig_val_test(plt):
        fig, ax = plt.subplots(figsize=(10, 3.6))
        x = np.arange(len(ranking))
        w = 0.38
        val = [r["best_val_f1"] for r in ranking]
        tst = [r["test"]["f1_macro"] for r in ranking]
        b1 = ax.bar(x - w / 2, val, w, color="#9dbbe1", edgecolor="#1f4e79", linewidth=0.5, label="Validación (mejor configuración)")
        b2 = ax.bar(x + w / 2, tst, w, color="#1f4e79", label="Prueba (modelo final)")
        for b, v in list(zip(b1, val)) + list(zip(b2, tst)):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([r["short"] for r in ranking])
        ax.set_ylim(min(val + tst) - 0.08, 1.02)
        ax.set_ylabel("F1-macro")
        ax.set_title("Consistencia validación -> prueba")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right")
        fig.tight_layout()
        return fig

    def fig_sensitivity(plt):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        ax = axes.flat

        # Regresión logística: F1 vs C por class_weight
        for cw, col in (("None", "#9dbbe1"), ("balanced", "#1f4e79")):
            rows = [r for r in logreg["grid"] if r["params"].get("class_weight") == cw]
            rows.sort(key=lambda r: float(r["params"]["C"]))
            ax[0].plot([float(r["params"]["C"]) for r in rows], [r["f1_macro"] for r in rows], marker="o", color=col, label=f"class_weight={cw}")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("C (inverso de la regularización)")
        ax[0].set_title("Regresión logística")

        # Árbol: mejor F1 por profundidad y criterio
        depths = ["5", "10", "20", "None"]
        for crit, col in (("gini", "#9dbbe1"), ("entropy", "#1f4e79")):
            ys = []
            for d in depths:
                rows = [r for r in tree["grid"] if r["params"].get("criterion") == crit and r["params"].get("depth") == d]
                ys.append(max(r["f1_macro"] for r in rows) if rows else np.nan)
            ax[1].plot(depths, ys, marker="o", color=col, label=f"criterion={crit}")
        ax[1].set_xlabel("max_depth (mejor sobre min_split y class_weight)")
        ax[1].set_title("Árbol de decisión")

        # Bayes: F1 vs var_smoothing
        combos = sorted({(r["params"].get("priors"), r["params"].get("escalar")) for r in bayes["grid"]})
        palette = ["#1f4e79", "#4f81bd", "#7fa7d4", "#c4d5eb"]
        for (pr, sc), col in zip(combos, palette):
            rows = [r for r in bayes["grid"] if (r["params"].get("priors"), r["params"].get("escalar")) == (pr, sc)]
            rows.sort(key=lambda r: float(r["params"]["var_smoothing"]))
            ax[2].plot([float(r["params"]["var_smoothing"]) for r in rows], [r["f1_macro"] for r in rows], marker="o", color=col, label=f"priors={pr}, escalar={sc}")
        ax[2].set_xscale("log")
        ax[2].set_xlabel("var_smoothing")
        ax[2].set_title("Naive Bayes gaussiano")

        # KNN: F1 vs K por pesos
        for ws, col in (("uniform", "#9dbbe1"), ("distance", "#1f4e79")):
            rows = [r for r in knn["grid"] if r["params"].get("pesos") == ws]
            rows.sort(key=lambda r: int(r["params"]["K"]))
            ax[3].plot([int(r["params"]["K"]) for r in rows], [r["f1_macro"] for r in rows], marker="o", color=col, label=f"weights={ws}")
        ax[3].set_xlabel("K (número de vecinos)")
        ax[3].set_title("K-Nearest Neighbors")

        # RF: mejor F1 por n_estimators y class_weight
        ns = sorted({int(r["params"]["n_estimators"]) for r in rf["grid"]})
        for cw, col in (("None", "#9dbbe1"), ("balanced", "#1f4e79")):
            ys = []
            for n in ns:
                rows = [r for r in rf["grid"] if int(r["params"]["n_estimators"]) == n and r["params"].get("class_weight") == cw]
                ys.append(max(r["f1_macro"] for r in rows) if rows else np.nan)
            ax[4].plot(ns, ys, marker="o", color=col, label=f"class_weight={cw}")
        ax[4].set_xticks(ns)
        ax[4].set_xlabel("n_estimators (mejor sobre depth y min_split)")
        ax[4].set_title("Random Forest")

        # Dispersión de todas las configuraciones por modelo
        data = [[r["f1_macro"] for r in m["grid"]] for m in ranking]
        bp = ax[5].boxplot(data, tick_labels=[f"{m['short']}\n({len(m['grid'])} conf.)" for m in ranking], patch_artist=True, widths=0.55)
        for patch, m in zip(bp["boxes"], ranking):
            patch.set_facecolor(MODEL_COLORS[m["key"]])
            patch.set_edgecolor("#1f4e79")
        for med in bp["medians"]:
            med.set_color("#8a1f1f")
        ax[5].set_title("Dispersión del F1-macro en la malla")
        ax[5].set_xlabel("Modelo (configuraciones probadas)")

        for a in ax:
            a.set_ylabel("F1-macro (validación)")
            a.grid(alpha=0.3)
            a.set_axisbelow(True)
        for a in ax[:5]:
            a.legend(fontsize=7.5)
        fig.tight_layout()
        return fig

    def fig_confusion(plt):
        cm_arr = np.asarray(best["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(8.2, 7.2))
        im = ax.imshow(cm_arr, cmap="Blues")
        n = cm_arr.shape[0]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel("Actividad predicha")
        ax.set_ylabel("Actividad real")
        ax.set_title(f"Matriz de confusión en prueba: {best['name']}")
        thresh = cm_arr.max() / 2
        for i in range(n):
            for j in range(n):
                v = cm_arr[i, j]
                if v:
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=7.5, color="white" if v > thresh else "#1f2a3a")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        fig.tight_layout()
        return fig

    def fig_f1_heatmap(plt):
        classes = sorted(best["per_class"])
        mat = np.array([[m["per_class"][c]["f1"] for c in classes] for m in ranking])
        fig, ax = plt.subplots(figsize=(11, 3.2))
        im = ax.imshow(mat, cmap="Blues", vmin=0.4, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticks(range(len(ranking)))
        ax.set_yticklabels([m["short"] for m in ranking])
        ax.set_xlabel("Actividad")
        ax.set_title("F1 por actividad en prueba, todos los modelos")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if mat[i, j] > 0.8 else "#1f2a3a")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        fig.tight_layout()
        return fig

    def fig_per_class(plt):
        classes = sorted(best["per_class"])
        pc = best["per_class"]
        x = np.arange(len(classes))
        w = 0.27
        fig, ax = plt.subplots(figsize=(11, 3.4))
        ax.bar(x - w, [pc[c]["precision"] for c in classes], w, color="#9dbbe1", label="Precision")
        ax.bar(x, [pc[c]["recall"] for c in classes], w, color="#4f81bd", label="Recall")
        ax.bar(x + w, [pc[c]["f1"] for c in classes], w, color="#1f4e79", label="F1")
        ax.axhline(0.90, color="#8a1f1f", ls="--", lw=0.9, label="F1 = 0.90")
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylim(0.6, 1.02)
        ax.set_xlabel("Actividad")
        ax.set_title(f"Precision, recall y F1 por actividad ({best['name']}, prueba)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(ncol=4, fontsize=8, loc="lower left")
        fig.tight_layout()
        return fig

    def compute_importances():
        """Reentrena el Random Forest final (misma configuración y semilla) para obtener las importancias."""
        sys.path.insert(0, str(BACKEND))
        from ml.random_forest import crear_modelo
        from scripts.data_utils import cargar_dataset, dividir_dataset

        X, y = cargar_dataset()
        X_train, X_val, X_test, y_train, y_val, y_test = dividir_dataset(X, y)
        b = rf["best"]
        depth = None if b.get("Max depth", "None") == "None" else int(b["Max depth"])
        cw = None if b.get("Class weight", "None") == "None" else b["Class weight"]
        model = crear_modelo(int(b["N estimators"]), depth, int(b["Min samples split"]), cw)
        import pandas as pd

        model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        imp = sorted(zip(model.feature_importances_, X.columns), reverse=True)
        return imp

    importances = compute_importances()

    def fig_importances(plt):
        top = importances[:20]
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.barh([n for _, n in top][::-1], [v for v, _ in top][::-1], color="#1f4e79")
        ax.set_xlabel("Importancia (reducción media de impureza)")
        ax.set_title("Las 20 características más importantes del Random Forest final")
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=7.5)
        fig.tight_layout()
        return fig

    # ---- renderizado de scripts con su salida ----
    def render_script(res: dict, appendix: str):
        module = res["module"]
        src = (ML_DIR / f"{module}.py").read_text(encoding="utf-8")
        log = (LOG_OUT / f"{module}.txt").read_text(encoding="utf-8")
        n_lines = log.count("\n") + 1
        story = [
            PageBreak(),
            Paragraph(f"Apéndice {appendix}. {esc(res['name'])}: script y salida", h1),
            Paragraph(esc(f"webapp/backend/ml/{module}.py"), h2),
            Paragraph(
                f"Código fuente completo del script seguido de la salida íntegra capturada al ejecutarlo con "
                f"<font face='Courier'>python -m ml.{esc(module)}</font> desde webapp/backend/ ({n_lines} líneas registradas en "
                f"output/logs/{esc(module)}.txt). La salida incluye la búsqueda en malla completa, la mejor configuración, "
                "las métricas de prueba, el reporte por clase, la matriz de confusión y las primeras 20 predicciones.",
                small,
            ),
            Paragraph("Código fuente", label_code),
            Preformatted(clean_text(src), code),
            Paragraph("Salida (stdout)", label_out),
            Preformatted(clean_text(log), out_style),
        ]
        return story

    def source_block(rel: str, title: str | None = None):
        p = ROOT / rel
        return [
            Paragraph(esc(title or rel), h3),
            Preformatted(clean_text(p.read_text(encoding="utf-8")).strip("\n"), code),
        ]

    # ---- README en Markdown ----
    def inline_md(text: str) -> str:
        t = esc(text)
        t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
        t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
        t = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", t)
        return t

    def md_table(rows: list[str]):
        parsed = []
        for r in rows:
            cells = [c.strip() for c in r.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c) and cells:
                continue
            parsed.append(cells)
        if not parsed:
            return []
        ncol = max(len(r) for r in parsed)
        parsed = [r + [""] * (ncol - len(r)) for r in parsed]
        if ncol > 8:  # tablas muy anchas (p. ej. la distribución por actividad) se transponen
            parsed = [list(col) for col in zip(*parsed)]
            ncol = len(parsed[0])
        data = [[Paragraph(inline_md(c), cell_head) for c in parsed[0]]]
        for r in parsed[1:]:
            data.append([Paragraph(inline_md(c), cell) for c in r])
        first = 0.28 if ncol > 2 else 0.35
        widths = [width * first] + [width * (1 - first) / (ncol - 1)] * (ncol - 1) if ncol > 1 else [width]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(grid_style())
        return [t, Spacer(1, 6)]

    def md_to_flowables(text: str):
        """Renderizador de Markdown suficiente para los README del proyecto."""
        flow = []
        code_block: list[str] = []
        table_rows: list[str] = []
        in_code = False
        skip_toc = False

        def flush_table():
            nonlocal table_rows
            if table_rows:
                flow.extend(md_table(table_rows))
                table_rows = []

        for raw in text.splitlines():
            line = raw.rstrip()
            if line.strip().startswith("```"):
                flush_table()
                if in_code:
                    flow.append(Preformatted(clean_text("\n".join(code_block)), code))
                    code_block = []
                in_code = not in_code
                continue
            if in_code:
                code_block.append(line)
                continue
            stripped = line.strip()
            if stripped.startswith("|"):
                table_rows.append(stripped)
                continue
            flush_table()
            if not stripped or stripped == "---":
                continue
            if re.match(r"</?details>|</?summary>", stripped):
                m = re.match(r"<summary>(.*)</summary>", stripped)
                if m:
                    flow.append(Paragraph(inline_md(m.group(1)), h3))
                continue
            if stripped.startswith("## Tabla de contenido"):
                skip_toc = True
                continue
            if skip_toc:
                if stripped.startswith("#"):
                    skip_toc = False
                else:
                    continue
            if stripped.startswith("# "):
                flow.append(Paragraph(inline_md(stripped[2:]), h2))
            elif stripped.startswith("## "):
                flow.append(Paragraph(inline_md(stripped[3:]), h2))
            elif stripped.startswith("### "):
                flow.append(Paragraph(inline_md(stripped[4:]), h3))
            elif stripped.startswith(">"):
                inner = stripped.lstrip("> ").strip()
                if inner.startswith("```") or not inner:
                    continue
                flow.append(Paragraph(inline_md(inner), quote))
            elif re.match(r"\s*- \[x\]", line, re.I):
                flow.append(Paragraph("[HECHO] " + inline_md(re.sub(r"^\s*- \[x\]\s*", "", line, flags=re.I)), li))
            elif re.match(r"\s*- \[ \]", line):
                flow.append(Paragraph("[PENDIENTE] " + inline_md(re.sub(r"^\s*- \[ \]\s*", "", line)), li))
            elif re.match(r"\s*[-*] ", line):
                indent = len(line) - len(line.lstrip())
                st = ParagraphStyle("li2", parent=li, leftIndent=12 + indent * 3)
                flow.append(Paragraph("&bull; " + inline_md(stripped[2:]), st))
            elif re.match(r"\s*\d+\. ", line):
                m = re.match(r"\s*(\d+)\. (.*)", line)
                flow.append(Paragraph(f"{m.group(1)}. " + inline_md(m.group(2)), li))
            else:
                flow.append(Paragraph(inline_md(stripped), body))
        flush_table()
        if in_code and code_block:
            flow.append(Preformatted(clean_text("\n".join(code_block)), code))
        return flow

    # ---- ensamblado ----
    gap_f1 = best["test"]["f1_macro"] - second["test"]["f1_macro"]
    gap_acc = (best["test"]["accuracy"] - second["test"]["accuracy"]) * 100
    n_configs = sum(len(r["grid"]) for r in order)
    n_above_90 = sum(1 for c in best["per_class"].values() if c["f1"] >= 0.90)
    worst_class = min(best["per_class"], key=lambda c: best["per_class"][c]["f1"])
    rf_range = (min(r["f1_macro"] for r in rf["grid"]), max(r["f1_macro"] for r in rf["grid"]))
    tree_range = (min(r["f1_macro"] for r in tree["grid"]), max(r["f1_macro"] for r in tree["grid"]))
    cm_arr = np.asarray(best["confusion_matrix"])
    off = cm_arr.copy()
    np.fill_diagonal(off, 0)
    top_conf = sorted(((off[i, j], i, j) for i in range(16) for j in range(16) if off[i, j]), reverse=True)[:3]
    knn_gain = knn["test"]["f1_macro"] - knn["best_val_f1"]

    def best_line(res: dict) -> str:
        return ", ".join(f"{k}={v}" for k, v in res["best"].items() if not k.lower().startswith("f1"))

    def grid_desc(res: dict) -> str:
        cols = list(res["grid"][0]["params"].keys())
        parts = []
        for c in cols:
            vals = []
            for r in res["grid"]:
                if r["params"][c] not in vals:
                    vals.append(r["params"][c])
            parts.append(f"{c}: {', '.join(vals)}")
        return "; ".join(parts)

    if not full:
        story = cover()
        story += [
            Paragraph("1. Resumen ejecutivo", h1),
            Paragraph(
                "El Proyecto REHAB compara cinco familias de clasificadores supervisados para reconocer 16 ejercicios de "
                "rehabilitación a partir de dos sensores: una pareja de unidades inerciales y un guante de flexión. Las "
                "series de tiempo crudas se transforman en una tabla de 480 características estadísticas por repetición y "
                "los cinco modelos se entrenan, ajustan y evalúan con el mismo protocolo: división estratificada 50/25/25 "
                "con semilla 42, búsqueda en malla en validación con F1-macro como métrica de selección, reentrenamiento "
                "con entrenamiento + validación y una única evaluación en prueba.",
                body,
            ),
            PipelineDiagram(width, h=2.6 * cm),
            Paragraph(
                f"Dataset: REHAB (16 actividades) | Repeticiones: {n_total:,} | Características: 480 | "
                f"División: {split['train']:,} / {split['val']:,} / {split['test']:,} | Semilla: 42 | "
                f"Configuraciones evaluadas: {n_configs}",
                caption,
            ),
        ]
        rows = [["#", "Modelo", "Mejor configuración (validación)", "Accuracy", "Precision", "Recall", "F1-macro"]]
        for i, r in enumerate(ranking, 1):
            rows.append([str(i), r["name"], best_line(r), fmt_pct(r["test"]["accuracy"]), f"{r['test']['precision_macro']:.4f}", f"{r['test']['recall_macro']:.4f}", f"{r['test']['f1_macro']:.4f}"])
        story += [
            simple_table(rows, [0.04, 0.19, 0.37, 0.10, 0.10, 0.10, 0.10], align_center_from=3),
            Spacer(1, 4),
            *figure(fig_summary, "Figura 1. Accuracy y F1-macro en el conjunto de prueba de los cinco modelos con su mejor configuración.", 6 * cm),
            Paragraph(
                f"<b>{esc(best['name'])}</b> es el modelo seleccionado: {fmt_pct(best['test']['accuracy'])} de accuracy y "
                f"{best['test']['f1_macro']:.4f} de F1-macro en prueba, con {gap_acc:.2f} puntos de accuracy y {gap_f1:.3f} de "
                f"F1-macro de ventaja sobre {esc(second['name'])}. {n_above_90} de 16 actividades superan F1 = 0.90 y la más "
                f"débil es la actividad {worst_class} (F1 {best['per_class'][worst_class]['f1']:.2f}). Además es el modelo más "
                f"estable ante sus hiperparámetros: sus {len(rf['grid'])} configuraciones quedan entre {rf_range[0]:.3f} y "
                f"{rf_range[1]:.3f} de F1-macro en validación, frente al árbol individual que oscila entre {tree_range[0]:.3f} y "
                f"{tree_range[1]:.3f}.",
                body,
            ),
            Paragraph("2. Problema y datos", h1),
            Paragraph(
                "La variable objetivo <font face='Courier'>actividad</font> toma 16 valores nominales (0 a 15): no existe "
                "orden ni distancia entre ellos, por lo que el problema es de <b>clasificación supervisada multiclase</b>. En "
                "la etapa anterior se probó una regresión lineal cuyas predicciones había que redondear, lo que carece de "
                "sentido para categorías; por eso en esta etapa sólo se evaluaron clasificadores. La métrica objetivo es "
                "F1-macro porque pondera las 16 actividades por igual.",
                body,
            ),
            Paragraph(
                "Cada actividad viene en dos arreglos NumPy de forma (repeticiones, 880, 6): 880 puntos temporales y 6 "
                "canales por sensor. Como los clasificadores clásicos esperan una tabla de longitud fija, cada repetición se "
                "corta en 4 ventanas de 220 puntos y en cada ventana y canal se calculan 10 estadísticos (media, desviación, "
                "mínimo, cuartiles, máximo, rango intercuartílico, asimetría y curtosis): 4 x 6 x 2 x 10 = 480 columnas, "
                f"todas numéricas y sin faltantes. El dataset tabular tiene {n_total:,} repeticiones, con entre 212 y 385 por "
                "actividad (relación 1 : 1.8), un desbalance ligero que motiva probar class_weight=balanced. Las escalas son "
                "muy heterogéneas, así que los modelos basados en distancia o gradiente se estandarizan dentro de un Pipeline.",
                body,
            ),
            *figure(fig_signal, "Figura 2. Primera repetición de la actividad 0 en ambos sensores; las líneas punteadas marcan las cuatro ventanas de 220 puntos.", 5.2 * cm),
            Paragraph("3. Metodología de evaluación", h1),
            SplitDiagram(width),
            Spacer(1, 4),
            Paragraph(
                "Los cinco scripts comparten el mismo flujo, implementado una sola vez en "
                "<font face='Courier'>scripts/data_utils.py</font>: (1) división estratificada 50/25/25 con semilla 42; "
                "(2) búsqueda en malla entrenando en entrenamiento y midiendo en validación; (3) selección por F1-macro, con "
                "accuracy como apoyo y entropía cruzada en los modelos con probabilidades; (4) reentrenamiento con "
                f"entrenamiento + validación ({split['train'] + split['val']:,} muestras); (5) evaluación única en prueba con "
                "accuracy, precision, recall, F1 macro, reporte por clase y matriz de confusión; (6) estandarización ajustada "
                "sólo con entrenamiento para evitar fuga de información; (7) semilla fija en todos los componentes "
                "aleatorios, por lo que los resultados se regeneran exactamente con <font face='Courier'>make models</font>.",
                body,
            ),
            Paragraph("4. Modelos y búsqueda de hiperparámetros", h1),
        ]
        desc = {
            "logreg": "One-vs-Rest: 16 clasificadores binarios lineales sobre features estandarizadas (lbfgs, max_iter=4000). Línea base interpretable.",
            "tree": "Umbrales encadenados sobre features sin escalar; captura interacciones pero un solo árbol con 480 features es inestable.",
            "bayes": "Asume normalidad e independencia condicional por clase; frontera cuadrática, casi sin hiperparámetros.",
            "knn": "Voto de los K vecinos más cercanos (euclidiana) en el espacio estandarizado de 480 dimensiones.",
            "rf": "Ensamble de árboles con bootstrap y submuestreo de ~22 features por nodo; voto mayoritario, n_jobs=-1.",
        }
        mrows = [["Modelo", "Idea", "Malla de hiperparámetros", "Conf.", "F1 val. (rango)", "Mejor configuración"]]
        for r in ranking:
            f1s = [g["f1_macro"] for g in r["grid"]]
            mrows.append([r["name"], desc[r["key"]], grid_desc(r), str(len(r["grid"])), f"{min(f1s):.3f} - {max(f1s):.3f}", best_line(r)])
        story += [
            simple_table(mrows, [0.13, 0.27, 0.22, 0.05, 0.11, 0.22], align_center_from=3),
            Spacer(1, 4),
            *figure(fig_sensitivity, "Figura 3. Sensibilidad del F1-macro en validación a los hiperparámetros de cada modelo y dispersión de todas las configuraciones probadas.", 9.6 * cm),
            Paragraph(
                "Lecturas principales: la regresión logística se sobreajusta con C = 10 y se subajusta con C = 0.01; el árbol "
                "necesita profundidad de al menos 10 y a partir de ahí se estabiliza; a Naive Bayes ningún hiperparámetro lo "
                "saca de su techo de 76 %, porque los 480 estadísticos están fuertemente correlacionados y violan la "
                "independencia; en KNN el F1 decrece monótonamente con K, señal de que cada repetición tiene un vecino casi "
                "idéntico; y en Random Forest las 36 configuraciones quedan en un rango de "
                f"{rf_range[1] - rf_range[0]:.3f}, con 100 árboles ya saturando la mejora.",
                body,
            ),
            Paragraph("5. Resultados", h1),
        ]
        vt_rows = [["Modelo", "F1-macro validación", "F1-macro prueba", "Diferencia", "Peor actividad (F1)"]]
        for r in ranking:
            wc = min(r["per_class"], key=lambda c: r["per_class"][c]["f1"])
            vt_rows.append([r["name"], f"{r['best_val_f1']:.4f}", f"{r['test']['f1_macro']:.4f}", f"{r['test']['f1_macro'] - r['best_val_f1']:+.3f}", f"{wc} ({r['per_class'][wc]['f1']:.2f})"])
        story += [
            simple_table(vt_rows, [0.28, 0.18, 0.18, 0.16, 0.20], align_center_from=1),
            Paragraph(
                f"Todos los modelos generalizan de forma estable: la diferencia validación-prueba es pequeña en los cinco "
                f"casos. {esc(best['name'])} cambia {best['test']['f1_macro'] - best['best_val_f1']:+.3f}, lo esperado al "
                f"cambiar de partición, y confirma que la malla no se sobreajustó a validación. KNN mejora en prueba "
                f"({knn_gain:+.3f}) porque el modelo final se entrena con 50 % más datos de referencia.",
                body,
            ),
            *figure(fig_f1_heatmap, "Figura 4. F1 por actividad en prueba para los cinco modelos, ordenados por F1-macro.", 4.6 * cm),
            *figure(fig_confusion, f"Figura 5. Matriz de confusión de {esc(best['name'])} en prueba (filas: actividad real; columnas: predicción).", 8.6 * cm),
            Paragraph(
                f"Las actividades 8, 9, 11 y 12 son las más difíciles para todos los modelos. En el modelo seleccionado la "
                f"actividad {worst_class} es la única por debajo de F1 = 0.90: recall "
                f"{best['per_class'][worst_class]['recall']:.2f} pero precisión {best['per_class'][worst_class]['precision']:.2f}, "
                f"es decir, absorbe muestras de otras actividades. Confusiones más frecuentes: "
                + "; ".join(f"real {i} predicha {j} ({v})" for v, i, j in top_conf)
                + ". Las actividades 8 y 9 se confunden en ambos sentidos, lo que sugiere movimientos muy similares.",
                body,
            ),
            PageBreak(),
            Paragraph("6. Decisión final y justificación", h1),
            Paragraph(f"<b>Modelo seleccionado: {esc(best['name'])} con {esc(best_line(best))}.</b>", body),
            Paragraph(
                f"1. <b>Es el mejor en todas las métricas de prueba</b>: {gap_acc:.1f} puntos de accuracy y {gap_f1:.3f} de "
                f"F1-macro sobre {esc(second['name'])}, unas {round(gap_acc / 100 * split['test'])} repeticiones más bien "
                f"clasificadas de {split['test']:,}; frente a la regresión logística la ventaja es de "
                f"{(best['test']['accuracy'] - logreg['test']['accuracy']) * 100:.1f} puntos.",
                li,
            ),
            Paragraph(
                f"2. <b>La ventaja es consistente</b>: ganó en validación ({best['best_val_f1']:.3f} vs {second['best_val_f1']:.3f}) "
                f"y en prueba ({best['test']['f1_macro']:.3f} vs {second['test']['f1_macro']:.3f}), y su desempeño por clase es "
                f"el más uniforme ({n_above_90} de 16 clases con F1 &gt; 0.90).",
                li,
            ),
            Paragraph(
                f"3. <b>Es el más robusto ante los hiperparámetros</b> (rango de {rf_range[1] - rf_range[0]:.3f} en 36 "
                "configuraciones), mientras que en KNN pasar de K = 1 a K = 3 cuesta varios puntos.",
                li,
            ),
            Paragraph(
                "4. <b>Encaja con los datos</b>: las 480 características son redundantes y el submuestreo de features por nodo "
                f"hace que el promedio cancele el ruido, lo que explica el salto del árbol individual ({fmt_pct(tree['test']['accuracy'])}) "
                f"al ensamble ({fmt_pct(rf['test']['accuracy'])}).",
                li,
            ),
            simple_table(
                [
                    ["Descartado", "Accuracy", "Motivo"],
                    ["K-Nearest Neighbors", fmt_pct(knn["test"]["accuracy"]), "Muy buen desempeño pero por debajo del ensamble; K = 1 es sensible al ruido y la inferencia compara contra todo el entrenamiento. Se conserva como segunda alternativa."],
                    ["Regresión logística", fmt_pct(logreg["test"]["accuracy"]), "La frontera lineal confunde sistemáticamente las actividades 9, 11 y 12. Recomendable sólo si se prioriza interpretabilidad o recursos muy limitados."],
                    ["Árbol de decisión", fmt_pct(tree["test"]["accuracy"]), "Inestable y propenso a sobreajustar con 480 features; el ensamble corrige justo ese problema."],
                    ["Naive Bayes gaussiano", fmt_pct(bayes["test"]["accuracy"]), "El supuesto de independencia condicional se viola gravemente y ningún hiperparámetro lo corrige."],
                ],
                [0.2, 0.12, 0.68],
                align_center_from=None,
            ),
            Paragraph(
                "<b>Riesgo aceptado.</b> Random Forest es el modelo más pesado: la malla tarda varios minutos y el modelo final "
                "guarda 100 árboles completos. El entrenamiento se hace una sola vez y la inferencia sigue siendo de "
                "milisegundos por muestra, por lo que la ganancia justifica el costo. "
                "<b>Ética y límites.</b> REHAB es un dataset público con aprobación ética y sin datos personales; el modelo "
                "reconoce qué ejercicio se ejecutó y no diagnostica ni sustituye al fisioterapeuta. Queda pendiente validar con "
                "validación cruzada, atender la actividad 11 con nuevas características, recortar features con las "
                "importancias del bosque y probar Gradient Boosting.",
                body,
            ),
            Paragraph(
                f"Reproducibilidad: el código, la división, la semilla y las versiones fijas de las bibliotecas están en el "
                f"repositorio (commit {esc(git_head())}); <font face='Courier'>make models</font> regenera todos los números y "
                f"<font face='Courier'>make report</font> este documento. La versión completa con código y salidas se genera con "
                "<font face='Courier'>--completo</font>.",
                small,
            ),
        ]
    else:
        # ---- ensamblado ----
        story = cover()

        gap_f1 = best["test"]["f1_macro"] - second["test"]["f1_macro"]
        gap_acc = (best["test"]["accuracy"] - second["test"]["accuracy"]) * 100
        n_configs = sum(len(r["grid"]) for r in order)
        n_above_90 = sum(1 for c in best["per_class"].values() if c["f1"] >= 0.90)
        worst_class = min(best["per_class"], key=lambda c: best["per_class"][c]["f1"])
        rf_range = (min(r["f1_macro"] for r in rf["grid"]), max(r["f1_macro"] for r in rf["grid"]))
        tree_range = (min(r["f1_macro"] for r in tree["grid"]), max(r["f1_macro"] for r in tree["grid"]))

        story += [
            Paragraph("1. Resumen ejecutivo", h1),
            Paragraph(
                "El Proyecto REHAB compara cinco familias de clasificadores supervisados para reconocer 16 ejercicios de "
                "rehabilitación a partir de señales de dos sensores: una pareja de unidades inerciales y un guante de flexión. "
                "Las series de tiempo crudas se transforman en una tabla de 480 características estadísticas por repetición, "
                "y los cinco modelos se entrenan, ajustan y evalúan con exactamente el mismo protocolo: división estratificada "
                "50/25/25 con semilla 42, búsqueda en malla sobre el conjunto de validación con F1-macro como métrica de "
                "selección, reentrenamiento con entrenamiento + validación y una única evaluación en el conjunto de prueba.",
                body,
            ),
            PipelineDiagram(width),
            Paragraph(
                f"Dataset: REHAB (16 actividades) | Repeticiones: {n_total:,} | Características: 480 | "
                f"División: {split['train']:,} / {split['val']:,} / {split['test']:,} | Semilla: 42 | "
                f"Configuraciones evaluadas: {n_configs}",
                caption,
            ),
        ]
        rows = [["#", "Modelo", "Mejor configuración", "Accuracy", "Precision", "Recall", "F1-macro"]]
        for i, r in enumerate(ranking, 1):
            conf = ", ".join(f"{k}={v}" for k, v in r["best"].items() if not k.lower().startswith("f1"))
            rows.append([str(i), r["name"], conf, fmt_pct(r["test"]["accuracy"]), f"{r['test']['precision_macro']:.4f}", f"{r['test']['recall_macro']:.4f}", f"{r['test']['f1_macro']:.4f}"])
        story += [
            simple_table(rows, [0.04, 0.19, 0.37, 0.10, 0.10, 0.10, 0.10], align_center_from=3),
            Spacer(1, 6),
            *figure(fig_summary, "Figura 1. Accuracy y F1-macro en el conjunto de prueba de los cinco modelos, con su mejor configuración de validación.", 7 * cm),
            Paragraph(
                f"<b>{esc(best['name'])}</b> es el modelo seleccionado: obtuvo {fmt_pct(best['test']['accuracy'])} de accuracy y "
                f"{best['test']['f1_macro']:.4f} de F1-macro en prueba, con una ventaja de {gap_acc:.2f} puntos de accuracy y "
                f"{gap_f1:.3f} de F1-macro sobre {esc(second['name'])}, el segundo lugar. {n_above_90} de 16 actividades superan "
                f"F1 = 0.90 y la más débil es la actividad {worst_class} (F1 {best['per_class'][worst_class]['f1']:.2f}). El "
                f"ensamble además es el modelo más estable ante sus hiperparámetros: sus {len(rf['grid'])} configuraciones quedan "
                f"entre {rf_range[0]:.3f} y {rf_range[1]:.3f} de F1-macro en validación, mientras que el árbol individual oscila "
                f"entre {tree_range[0]:.3f} y {tree_range[1]:.3f}.",
                body,
            ),
            Paragraph(
                "Este documento reúne la documentación del proyecto, la metodología, los resultados con figuras generadas a "
                "partir de las salidas reales de los scripts, la justificación de la decisión, el código fuente completo del "
                "pipeline y de los cinco modelos con su salida íntegra, la API de la interfaz web y el README del repositorio. "
                "Los archivos binarios del dataset y los paquetes del entorno se resumen en lugar de incrustarse.",
                small,
            ),
            PageBreak(),
        ]

        # ---- 2. problema ----
        story += [
            Paragraph("2. Identificación del problema", h1),
            Paragraph(
                "La variable objetivo <font face='Courier'>actividad</font> toma 16 valores discretos (0 a 15) que representan "
                "categorías de ejercicios de rehabilitación motora posterior a un accidente cerebrovascular. No existe orden ni "
                "distancia entre ellas: la actividad 7 no es \"mayor\" que la 3. Se trata, por lo tanto, de un problema de "
                "<b>clasificación supervisada multiclase</b> con etiquetas nominales, no de regresión ni de clasificación "
                "multietiqueta: cada repetición pertenece a exactamente una actividad.",
                body,
            ),
            Paragraph(
                "En la etapa anterior del proyecto se implementó una regresión lineal como primera aproximación. Sus "
                "predicciones continuas tenían que redondearse al entero más cercano para compararse con la etiqueta, lo cual "
                "carece de sentido para categorías nominales. Por eso en esta etapa se descartó la regresión y se evaluaron "
                "únicamente clasificadores.",
                quote,
            ),
            Paragraph(
                "El modelo reconoce <i>qué</i> ejercicio se ejecutó; no diagnostica ni mide la calidad del movimiento ni "
                "sustituye la evaluación de un fisioterapeuta. La métrica objetivo es F1-macro porque pondera las 16 "
                "actividades por igual, de modo que una actividad frecuente no pueda ocultar el mal desempeño en una rara.",
                body,
            ),
            Paragraph("3. Datos e ingeniería de características", h1),
            Paragraph("3.1 Señales crudas", h2),
            Paragraph(
                "El dataset REHAB entrega, por actividad, dos arreglos NumPy: <font face='Courier'>XXX_1.npy</font> con las "
                "dos unidades inerciales y <font face='Courier'>XXX_2.npy</font> con el guante de flexión. Cada uno tiene forma "
                "(repeticiones, 880, 6): 880 puntos temporales por repetición y 6 canales por sensor. Una repetición cruda "
                "contiene así 880 x 6 x 2 = 10,560 valores que describen cómo evoluciona la señal durante el movimiento.",
                body,
            ),
            *figure(fig_signal, "Figura 2. Primera repetición de la actividad 0 en ambos sensores. Las líneas punteadas marcan las cuatro ventanas de 220 puntos usadas en la extracción de características.", 6.6 * cm),
            Paragraph("3.2 De señal a tabla", h2),
            Paragraph(
                "Los clasificadores clásicos de scikit-learn esperan una tabla de longitud fija, así que "
                "<font face='Courier'>crear_dataset_ventanas.py</font> resume cada repetición: la corta en 4 ventanas de 220 "
                "puntos (para conservar el orden temporal grueso: inicio, mitad y final del movimiento) y en cada ventana y "
                "canal calcula 10 estadísticos: media, desviación estándar, mínimo, primer cuartil, mediana, tercer cuartil, "
                "máximo, rango intercuartílico, asimetría y curtosis. El resultado son 4 x 6 x 2 x 10 = 480 columnas, cada "
                "una nombrada <font face='Courier'>s{sensor}_w{ventana}_c{canal}_{estadístico}</font>. Los estadísticos se "
                "implementan a mano en Python puro (sin depender de NumPy para los cálculos) como parte del ejercicio.",
                body,
            ),
            Paragraph("3.3 Características del dataset tabular", h2),
            simple_table(
                [
                    ["Propiedad", "Valor", "Implicación para el modelado"],
                    ["Filas (repeticiones)", f"{n_total:,}", "Tamaño moderado: los modelos clásicos son suficientes y rápidos"],
                    ["Características", "480, todas numéricas continuas", "Compatibles con cualquier clasificador de scikit-learn; no se requiere codificación"],
                    ["Valores faltantes", "0", "No se necesita imputación"],
                    ["Escalas", "Muy heterogéneas (medias vs. curtosis, IMU vs. flexión)", "Los modelos basados en distancia o gradiente requieren estandarización"],
                    ["Balance de clases", "212 a 385 muestras por clase (relación 1 : 1.8)", "Ligero desbalance: se prueba class_weight=balanced y se reporta F1-macro"],
                ],
                [0.22, 0.33, 0.45],
                align_center_from=None,
            ),
            Spacer(1, 6),
            *figure(fig_classes, "Figura 3. Distribución de repeticiones por actividad. La clase 7 es la más frecuente (385) y la 1 la menos (212).", 6 * cm),
            Paragraph(
                "Al tener características numéricas, densas, sin faltantes y una etiqueta categórica, cualquier clasificador "
                "supervisado es aplicable. Se eligieron cinco familias con sesgos inductivos distintos para cubrir el espectro "
                "de hipótesis: lineal (regresión logística), umbrales por eje (árbol de decisión), probabilística con frontera "
                "cuadrática (Naive Bayes gaussiano), no paramétrica local (KNN) y ensamble de árboles (Random Forest).",
                body,
            ),
            Paragraph("Inventario del conjunto de datos", h3),
        ]
        inv = [["Actividad", "Repeticiones", "Sensor 1 (forma)", "Sensor 2 (forma)", "Bytes (ambos)"]]
        total_bytes = 0
        for act in range(16):
            p1, p2 = DATASET_DIR / f"{act:03d}_1.npy", DATASET_DIR / f"{act:03d}_2.npy"
            if p1.exists() and p2.exists():
                a1 = np.load(p1, mmap_mode="r")
                a2 = np.load(p2, mmap_mode="r")
                b = p1.stat().st_size + p2.stat().st_size
                total_bytes += b
                inv.append([str(act), str(a1.shape[0]), str(tuple(a1.shape)), str(tuple(a2.shape)), f"{b:,}"])
            else:
                inv.append([str(act), "-", "no encontrado", "no encontrado", "-"])
        inv.append(["Total", str(n_total), "", "", f"{total_bytes:,}"])
        story += [
            simple_table(inv, [0.16, 0.18, 0.24, 0.24, 0.18], align_center_from=1),
            Paragraph(
                f"El CSV generado ({CSV_PATH.name}, {CSV_PATH.stat().st_size:,} bytes) tiene {n_total:,} filas y 482 columnas: "
                "actividad, índice de repetición y las 480 características.",
                small,
            ),
            PageBreak(),
        ]

        # ---- 4. metodología ----
        story += [
            Paragraph("4. Metodología de evaluación", h1),
            Paragraph(
                "Para que la selección de modelo sea honesta y reproducible, los cinco scripts comparten el mismo protocolo, "
                "implementado una sola vez en <font face='Courier'>scripts/data_utils.py</font>:",
                body,
            ),
            SplitDiagram(width),
            Spacer(1, 8),
            Paragraph("1. <b>División estratificada 50 / 25 / 25</b> con <font face='Courier'>random_state=42</font>. La estratificación mantiene la misma proporción de cada actividad en los tres conjuntos.", li),
            Paragraph("2. <b>Búsqueda de hiperparámetros en malla</b> sobre el conjunto de validación: cada combinación se entrena en entrenamiento y se mide en validación.", li),
            Paragraph("3. <b>Métrica de selección: F1-macro.</b> Promedia el F1 de las 16 clases con el mismo peso, lo que evita que la clase mayoritaria infle el resultado. Accuracy se reporta como apoyo y la entropía cruzada en los modelos que producen probabilidades (regresión logística, Naive Bayes y KNN).", li),
            Paragraph(f"4. <b>Reentrenamiento final con entrenamiento + validación</b> ({split['train'] + split['val']:,} muestras) usando la mejor configuración, para aprovechar más datos.", li),
            Paragraph("5. <b>Evaluación única en prueba</b>, con accuracy, precision, recall y F1 macro, reporte por clase y matriz de confusión.", li),
            Paragraph("6. <b>Estandarización dentro de un <font face='Courier'>Pipeline</font></b> en los modelos que la necesitan. El escalador se ajusta sólo con datos de entrenamiento, evitando fuga de información. Los modelos basados en árboles no se escalan porque sus divisiones por umbral son invariantes a la escala.", li),
            Paragraph("7. <b>Semilla fija en todos los componentes aleatorios</b>: la división, el árbol de decisión y el Random Forest usan la semilla 42, por lo que los resultados de este documento se regeneran exactamente con <font face='Courier'>make models</font>.", li),
            Spacer(1, 6),
        ]

        # ---- 5. modelos ----
        def grid_param_table(res: dict, labels: dict[str, str]):
            cols = list(res["grid"][0]["params"].keys())
            rows = [["Hiperparámetro", "Valores probados", "Qué controla"]]
            for c in cols:
                vals = []
                for r in res["grid"]:
                    v = r["params"][c]
                    if v not in vals:
                        vals.append(v)
                rows.append([c, ", ".join(vals), labels.get(c, "")])
            return simple_table(rows, [0.2, 0.35, 0.45], align_center_from=None)

        def best_line(res: dict) -> str:
            return ", ".join(f"{k} = {v}" for k, v in res["best"].items() if not k.lower().startswith("f1"))

        def val_summary(res: dict) -> str:
            f1s = [r["f1_macro"] for r in res["grid"]]
            return f"{len(res['grid'])} configuraciones; F1-macro en validación entre {min(f1s):.4f} y {max(f1s):.4f}. Mejor configuración: {best_line(res)} (F1 {max(f1s):.4f})."

        story += [
            Paragraph("5. Modelos investigados y configuraciones probadas", h1),
            Paragraph(
                "Cada modelo vive en su propio script dentro de <font face='Courier'>webapp/backend/ml/</font> y define su "
                "malla de hiperparámetros como constantes al inicio del archivo. Las tablas siguientes se construyen a partir "
                "de las salidas reales de los scripts (apéndices B a F), donde se imprime la malla completa.",
                body,
            ),
            Paragraph("5.1 Regresión logística (One-vs-Rest)", h2),
            Paragraph(
                "Se entrenan 16 clasificadores binarios, uno por actividad, sobre características estandarizadas. Solver "
                "<font face='Courier'>lbfgs</font> con <font face='Courier'>max_iter=4000</font> para garantizar convergencia "
                "con 480 características. Es la línea base interpretable y prueba si las clases son linealmente separables.",
                body,
            ),
            grid_param_table(logreg, {"C": "Inverso de la regularización L2", "class_weight": "Compensación del desbalance de clases"}),
            Paragraph(esc(val_summary(logreg)), small),
            Paragraph(
                "Con C = 10 el modelo se sobreajusta (la entropía cruzada casi se duplica respecto a C = 0.1); con C = 0.01 se "
                "subajusta. El óptimo en F1-macro está en C = 1.0 con pesos balanceados.",
                body,
            ),
            Paragraph("5.2 Árbol de decisión", h2),
            Paragraph(
                "Encadena preguntas del tipo \"¿s1_w2_c4_media &gt; 0.7?\" hasta llegar a una hoja. No requiere escalado y "
                "captura interacciones y umbrales sin suposiciones sobre la distribución.",
                body,
            ),
            grid_param_table(tree, {"criterion": "Medida de impureza para dividir nodos", "depth": "Profundidad máxima (complejidad)", "min_split": "Mínimo de muestras para dividir un nodo", "class_weight": "Compensación del desbalance"}),
            Paragraph(esc(val_summary(tree)), small),
            Paragraph(
                "Profundidad 5 es claramente insuficiente; a partir de 10 el rendimiento se estabiliza y max_depth = 20 "
                "equivale a None porque el árbol nunca crece más allá de esa profundidad. Un solo árbol con 480 características "
                "es inestable: la malla completa oscila en más de 15 puntos de F1-macro.",
                body,
            ),
            Paragraph("5.3 Naive Bayes gaussiano", h2),
            Paragraph(
                "Asume que cada característica sigue una distribución normal por clase y que las características son "
                "independientes entre sí dada la clase. Cada clase tiene su propia varianza, por lo que la frontera de "
                "decisión es cuadrática.",
                body,
            ),
            grid_param_table(bayes, {"var_smoothing": "Fracción de la varianza máxima que se suma a todas las varianzas", "priors": "Probabilidad a priori: frecuencia de clase (None) o uniforme", "escalar": "Estandarización previa (sin / con StandardScaler)"}),
            Paragraph(esc(val_summary(bayes)), small),
            Paragraph(
                "Ningún hiperparámetro ayuda de forma significativa: el techo está en torno al 76 % de accuracy. La entropía "
                "cruzada tan alta (alrededor de 7.6) indica probabilidades extremadamente sobreconfiadas, síntoma de que el "
                "supuesto de independencia se viola: media, mediana y cuartiles de un mismo canal son casi redundantes.",
                body,
            ),
            Paragraph("5.4 K-Nearest Neighbors", h2),
            Paragraph(
                "Clasifica cada repetición según la actividad mayoritaria de sus K vecinos más cercanos en el espacio de 480 "
                "dimensiones estandarizado, con distancia euclidiana.",
                body,
            ),
            grid_param_table(knn, {"K": "Cantidad de vecinos que votan", "pesos": "Voto igualitario (uniform) o ponderado por cercanía (distance)"}),
            Paragraph(esc(val_summary(knn)), small),
            Paragraph(
                "El rendimiento decrece monótonamente al aumentar K, y distance siempre supera a uniform para K &gt; 1 (con "
                "K = 1 ambos son idénticos porque sólo vota un vecino). Esto revela la geometría de los datos: cada repetición "
                "tiene un vecino casi idéntico de su misma actividad, pero al ampliar el vecindario entran ejemplos de "
                "actividades parecidas que contaminan el voto.",
                body,
            ),
            Paragraph("5.5 Random Forest", h2),
            Paragraph(
                "Ensamble de árboles de decisión. Cada árbol se entrena con una muestra bootstrap del conjunto de "
                "entrenamiento y, en cada nodo, sólo considera un subconjunto aleatorio de las 480 características (raíz de "
                "480, unas 22). La predicción final es el voto mayoritario. Se paraleliza con "
                "<font face='Courier'>n_jobs=-1</font>.",
                body,
            ),
            grid_param_table(rf, {"n_estimators": "Número de árboles del ensamble", "depth": "Profundidad máxima de cada árbol", "min_split": "Mínimo de muestras para dividir un nodo", "class_weight": "Compensación del desbalance"}),
            Paragraph(esc(val_summary(rf)), small),
            Paragraph(
                "Pasar de 50 a 100 árboles ayuda; de 100 a 200 ya no mejora. Limitar la profundidad a 10 perjudica ligeramente: "
                "los árboles completamente crecidos son los mejores miembros del ensamble porque el promedio ya controla el "
                "sobreajuste. max_depth = 20 y None producen exactamente los mismos resultados. class_weight = balanced gana "
                "en la mayoría de las comparaciones directas.",
                body,
            ),
            *figure(fig_sensitivity, "Figura 4. Sensibilidad del F1-macro en validación a los hiperparámetros de cada modelo (paneles 1 a 5) y dispersión de todas las configuraciones probadas por modelo (panel 6).", 13.5 * cm),
            PageBreak(),
        ]

        # ---- 6. resultados ----
        vt_rows = [["Modelo", "F1-macro validación", "F1-macro prueba", "Diferencia"]]
        for r in ranking:
            vt_rows.append([r["name"], f"{r['best_val_f1']:.4f}", f"{r['test']['f1_macro']:.4f}", f"{r['test']['f1_macro'] - r['best_val_f1']:+.3f}"])
        pc_rows = [["Actividad", "Precision", "Recall", "F1", "Soporte"]]
        for c in sorted(best["per_class"]):
            v = best["per_class"][c]
            pc_rows.append([str(c), f"{v['precision']:.4f}", f"{v['recall']:.4f}", f"{v['f1']:.4f}", str(v["support"])])
        cm_arr = np.asarray(best["confusion_matrix"])
        off = cm_arr.copy()
        np.fill_diagonal(off, 0)
        top_conf = sorted(((off[i, j], i, j) for i in range(16) for j in range(16) if off[i, j]), reverse=True)[:4]
        knn_gain = knn["test"]["f1_macro"] - knn["best_val_f1"]

        story += [
            Paragraph("6. Resultados", h1),
            Paragraph("6.1 Comparación final en el conjunto de prueba", h2),
            Paragraph(
                f"Cada modelo fue reentrenado con entrenamiento + validación usando su mejor configuración y evaluado una sola "
                f"vez en las {split['test']:,} muestras de prueba. La tabla de la sección 1 y la Figura 1 resumen las métricas; "
                "la figura siguiente contrasta el F1-macro de validación con el de prueba.",
                body,
            ),
            *figure(fig_val_test, "Figura 5. F1-macro de la mejor configuración en validación frente al del modelo final en prueba.", 6.5 * cm),
            simple_table(vt_rows, [0.34, 0.22, 0.22, 0.22], align_center_from=1),
            Paragraph(
                f"Todos los modelos generalizan de forma estable: la diferencia entre validación y prueba es pequeña en los "
                f"cinco casos. {esc(best['name'])} cambia {best['test']['f1_macro'] - best['best_val_f1']:+.3f}, la variación "
                f"esperada al cambiar de partición, lo que confirma que la búsqueda de hiperparámetros no se sobreajustó al "
                f"conjunto de validación. La mejora de KNN en prueba ({knn_gain:+.3f}) se explica porque el modelo final se "
                "entrena con 50 % más datos, y KNN es el modelo que más se beneficia de tener más ejemplos de referencia.",
                body,
            ),
            Paragraph("6.2 Desempeño por actividad", h2),
            *figure(fig_f1_heatmap, "Figura 6. F1 por actividad en prueba para los cinco modelos, ordenados por F1-macro.", 5.2 * cm),
            Paragraph(
                "Las actividades 8, 9, 11 y 12 son las más difíciles para todos los modelos, y la actividad 1 es el punto "
                "débil de los modelos más simples. El ensamble es el único que mantiene todas las actividades por encima de "
                f"F1 = {min(c['f1'] for c in best['per_class'].values()):.2f}.",
                body,
            ),
            PageBreak(),
            Paragraph(f"6.3 Modelo seleccionado: {esc(best['name'])} en prueba", h2),
            *figure(fig_per_class, f"Figura 7. Precision, recall y F1 por actividad del modelo seleccionado. {n_above_90} de 16 actividades superan F1 = 0.90.", 6 * cm),
            simple_table(pc_rows, [0.2, 0.2, 0.2, 0.2, 0.2], align_center_from=1),
            Spacer(1, 6),
            *figure(fig_confusion, "Figura 8. Matriz de confusión del modelo seleccionado sobre el conjunto de prueba (filas: actividad real; columnas: predicción).", 11.5 * cm),
            Paragraph(
                f"La actividad {worst_class} es la única por debajo de F1 = 0.90: tiene recall "
                f"{best['per_class'][worst_class]['recall']:.2f} pero precisión {best['per_class'][worst_class]['precision']:.2f}, "
                f"es decir, el modelo rara vez falla una repetición real de esa actividad pero absorbe muestras de otras. Las "
                f"confusiones más frecuentes fuera de la diagonal son: "
                + "; ".join(f"real {i} predicha {j} ({v})" for v, i, j in top_conf)
                + ". Las actividades 8 y 9 se confunden entre sí en ambos sentidos, lo que sugiere movimientos muy similares.",
                body,
            ),
            *figure(fig_importances, "Figura 9. Importancia de características del Random Forest final (reentrenado con la misma configuración y semilla para este reporte).", 7.5 * cm),
            Paragraph(
                f"Las importancias están muy repartidas: la característica más importante ({esc(importances[0][1])}) aporta "
                f"{importances[0][0] * 100:.2f} % de la reducción de impureza y las 20 primeras suman "
                f"{sum(v for v, _ in importances[:20]) * 100:.1f} %. Esto es coherente con la redundancia de los 480 "
                "estadísticos: distintos árboles explotan distintas variables casi equivalentes.",
                body,
            ),
            PageBreak(),
        ]

        # ---- 7. decisión ----
        story += [
            Paragraph("7. Decisión final y justificación", h1),
            Paragraph(f"<b>Modelo seleccionado: {esc(best['name'])} con {esc(best_line(best))}.</b>", body),
            Paragraph(
                f"1. <b>Es el mejor en todas las métricas de prueba</b>, con una ventaja de {gap_acc:.1f} puntos de accuracy y "
                f"{gap_f1:.3f} de F1-macro sobre el segundo lugar ({esc(second['name'])}). Con {split['test']:,} muestras de prueba, "
                f"esa diferencia equivale a unas {round(gap_acc / 100 * split['test'])} repeticiones más clasificadas correctamente. "
                f"Frente a la regresión logística la ventaja es de {(best['test']['accuracy'] - logreg['test']['accuracy']) * 100:.1f} puntos.",
                li,
            ),
            Paragraph(
                f"2. <b>La ventaja es consistente</b>, no puntual: ganó en validación ({best['best_val_f1']:.3f} vs "
                f"{second['best_val_f1']:.3f} de {esc(second['short'])}) y en prueba ({best['test']['f1_macro']:.3f} vs "
                f"{second['test']['f1_macro']:.3f}), y su desempeño por clase es el más uniforme ({n_above_90} de 16 clases con "
                f"F1 &gt; 0.90, ninguna por debajo de {min(c['f1'] for c in best['per_class'].values()):.2f}).",
                li,
            ),
            Paragraph(
                f"3. <b>Es el modelo más robusto ante los hiperparámetros.</b> Las {len(rf['grid'])} configuraciones probadas "
                f"quedan en un rango de {rf_range[1] - rf_range[0]:.3f} de F1-macro. Esto reduce el riesgo de que el resultado "
                "dependa de una elección afortunada de la malla, riesgo que sí existe en KNN, donde pasar de K = 1 a K = 3 "
                "cuesta varios puntos.",
                li,
            ),
            Paragraph(
                "4. <b>Encaja con la naturaleza de los datos.</b> Las 480 características son altamente redundantes. El "
                "submuestreo aleatorio de características en cada nodo hace que distintos árboles exploten distintas variables "
                "redundantes y el promedio cancela el ruido. Esto explica el salto del árbol individual "
                f"({fmt_pct(tree['test']['accuracy'])}) al ensamble ({fmt_pct(rf['test']['accuracy'])}): la misma familia de "
                "hipótesis, pero con la varianza controlada.",
                li,
            ),
            Paragraph("Por qué se descartaron los demás", h2),
            simple_table(
                [
                    ["Modelo", "Accuracy en prueba", "Motivo de descarte"],
                    ["K-Nearest Neighbors", fmt_pct(knn["test"]["accuracy"]), "Muy buen desempeño pero por debajo del ensamble. K = 1 es sensible al ruido y su inferencia requiere comparar contra todo el conjunto de entrenamiento. Se conserva como segunda alternativa por su simplicidad."],
                    ["Regresión logística", fmt_pct(logreg["test"]["accuracy"]), "Su frontera lineal confunde sistemáticamente las actividades 9, 11 y 12. Sigue siendo la opción recomendada si se prioriza interpretabilidad de pesos o inferencia en un dispositivo con recursos muy limitados."],
                    ["Árbol de decisión", fmt_pct(tree["test"]["accuracy"]), "Un solo árbol con 480 características es inestable y tiende a sobreajustar; el ensamble corrige exactamente ese problema, por lo que queda superado por Random Forest."],
                    ["Naive Bayes gaussiano", fmt_pct(bayes["test"]["accuracy"]), "El supuesto de independencia condicional se viola gravemente; ningún hiperparámetro lo corrige. Se descarta por completo."],
                ],
                [0.22, 0.16, 0.62],
                align_center_from=None,
            ),
            Paragraph("Riesgo conocido de la decisión", h2),
            Paragraph(
                "Random Forest es el modelo más pesado de los cinco: entrenar las 36 configuraciones tarda varios minutos y el "
                "modelo final almacena 100 árboles completamente crecidos. Se aceptó este costo porque el entrenamiento se hace "
                "una sola vez, la inferencia con 100 árboles sigue siendo rápida (milisegundos por muestra) y la ganancia de "
                "casi 3 puntos sobre KNN y 7 sobre la regresión logística justifica el uso de memoria adicional.",
                body,
            ),
            Paragraph("8. Interfaz web: REHAB Model Lab", h1),
            Paragraph(
                "El repositorio incluye una interfaz web para entrenar, ajustar y comparar los cinco clasificadores sin tocar "
                "la terminal. El backend es una API FastAPI (<font face='Courier'>webapp/backend/api/</font>) que reutiliza el "
                "mismo <font face='Courier'>scripts/data_utils.py</font> de los scripts, de modo que cada corrida sigue el "
                "protocolo del proyecto: entrena en entrenamiento, mide en validación, reentrena con ambos y evalúa una sola "
                "vez en prueba. Con la configuración por defecto de cada modelo se reproducen exactamente los números de este "
                "reporte. El frontend (Vite + React + TypeScript + Recharts) ofrece, por modelo, selectores con los mismos "
                "valores de la malla, métricas, F1 por actividad, matriz de confusión e importancias; y, en conjunto, un ranking "
                "por F1-macro o accuracy, un mapa de calor de F1 por actividad y un veredicto generado a partir de las métricas. "
                "También integra la presentación del reto en nueve diapositivas. Se levanta con "
                "<font face='Courier'>make dev</font> en http://localhost:5173. El código de la API se reproduce en el "
                "apéndice G y el README de la interfaz en el apéndice H.",
                body,
            ),
            Paragraph("9. Consideraciones éticas, limitaciones y trabajo futuro", h1),
            Paragraph("Consideraciones éticas y normativas", h2),
            Paragraph("&bull; <b>Origen y consentimiento de los datos.</b> REHAB es un dataset público publicado en <i>Scientific Data</i> (2026) bajo licencia abierta; los autores reportan aprobación de comité de ética y consentimiento informado. Este proyecto sólo usa las señales cinemáticas de los 16 movimientos y no contiene información personal identificable.", li),
            Paragraph("&bull; <b>Uso previsto.</b> El modelo reconoce qué ejercicio se ejecutó; no diagnostica ni mide la calidad del movimiento ni sustituye a un fisioterapeuta. Un uso clínico real requeriría validación con pacientes distintos y cumplimiento de la normatividad de dispositivos médicos y datos de salud (en México, NOM-024-SSA3-2012 y la legislación de protección de datos personales).", li),
            Paragraph("&bull; <b>Transparencia y reproducibilidad.</b> Todo el código, la división, la semilla y las versiones de las bibliotecas están en el repositorio; los resultados se regeneran con <font face='Courier'>make models</font>.", li),
            Paragraph("&bull; <b>Sesgo del dataset.</b> Las señales provienen de un número limitado de sujetos y de un protocolo controlado; el desempeño puede no transferirse a otros pacientes, sensores o entornos domésticos.", li),
            Paragraph("Limitaciones y trabajo futuro", h2),
            Paragraph("&bull; <b>Validación simple, no cruzada.</b> Se usó una única partición 50/25/25. Una validación cruzada estratificada de 5 pliegues daría intervalos de confianza y permitiría afirmar con más seguridad que la diferencia entre Random Forest y KNN no depende de la partición.", li),
            Paragraph(f"&bull; <b>La actividad {worst_class} es el punto débil</b> del modelo seleccionado. Convendría analizar qué tienen en común los movimientos que absorbe y añadir características específicas (dominio de la frecuencia o correlación entre sensores).", li),
            Paragraph("&bull; <b>Redundancia de características.</b> Las importancias del Random Forest (Figura 9) pueden usarse para recortar las 480 columnas a un subconjunto más pequeño sin perder desempeño y acelerar entrenamiento e inferencia.", li),
            Paragraph("&bull; <b>Otros ensambles no evaluados.</b> Gradient Boosting (por ejemplo <font face='Courier'>HistGradientBoostingClassifier</font>) suele superar a Random Forest en datos tabulares y queda como siguiente iteración.", li),
            Paragraph("&bull; <b>Coste de entrenamiento.</b> La búsqueda en malla de Random Forest es la más lenta del proyecto; una búsqueda aleatoria o bayesiana reduciría el tiempo si se amplía la malla.", li),
        ]

        # ---- apéndices ----
        story += [
            PageBreak(),
            Paragraph("Apéndice A. Pipeline compartido", h1),
            Paragraph("Código fuente completo de los módulos de webapp/backend/scripts/ usados por los cinco modelos y por la API.", small),
            *source_block("webapp/backend/scripts/config.py"),
            *source_block("webapp/backend/scripts/data_utils.py"),
            *source_block("webapp/backend/scripts/crear_dataset_ventanas.py"),
        ]
        for (key, module, name, short), letter in zip(MODELS, "BCDEF"):
            story += render_script(results[key], letter)
        story += [
            PageBreak(),
            Paragraph("Apéndice G. API de la interfaz web", h1),
            Paragraph("Código fuente de la API FastAPI que entrena los modelos bajo demanda para la interfaz web.", small),
            *source_block("webapp/backend/api/models.py"),
            *source_block("webapp/backend/api/main.py"),
            PageBreak(),
            Paragraph("Apéndice H. README del repositorio", h1),
            Paragraph("Reproducción del README.md principal (sin la tabla de contenido) y del README de la interfaz web.", small),
            *md_to_flowables((ROOT / "README.md").read_text(encoding="utf-8")),
            Paragraph("webapp/README.md", h2),
            *md_to_flowables((ROOT / "webapp" / "README.md").read_text(encoding="utf-8")),
            PageBreak(),
            Paragraph("Apéndice I. Entorno y metadatos del repositorio", h1),
            Paragraph("requirements.txt", h3),
            Preformatted((ROOT / "requirements.txt").read_text(encoding="utf-8").strip(), code),
            Paragraph("webapp/backend/requirements.txt", h3),
            Preformatted((BACKEND / "requirements.txt").read_text(encoding="utf-8").strip(), code),
            Paragraph("report/requirements.txt", h3),
            Preformatted((ROOT / "report" / "requirements.txt").read_text(encoding="utf-8").strip(), code),
            Paragraph("Makefile", h3),
            Preformatted(clean_text((ROOT / "Makefile").read_text(encoding="utf-8")).strip(), code),
            Paragraph(".gitignore", h3),
            Preformatted((ROOT / ".gitignore").read_text(encoding="utf-8").strip(), code),
            Paragraph("Archivos generados", h3),
            Paragraph(
                "output/logs/ contiene la salida íntegra de cada script tal como se reproduce en los apéndices B a F; "
                "output/results/ guarda las métricas, la malla, el reporte por clase y la matriz de confusión extraídos de "
                "esas salidas en formato JSON y usados para todas las tablas y figuras de este documento.",
                body,
            ),
            Paragraph("Nota del repositorio", h3),
            Paragraph(
                f"Commit de la implementación: {esc(git_head())}. El entorno virtual y los archivos compilados se excluyen del "
                "control de versiones. Este PDF se generó con report/build_report.py.",
                body,
            ),
        ]

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Escrito {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-exec", action="store_true", help="no ejecuta los scripts; reutiliza output/logs/")
    parser.add_argument("--completo", action="store_true", help="versión larga con código fuente, salidas íntegras y apéndices")
    args = parser.parse_args()
    if not args.no_exec:
        execute_scripts()
    results = load_results()
    build_pdf(PDF_PATH, results, full=args.completo)


if __name__ == "__main__":
    main()
