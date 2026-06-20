"""
Uso:
    python scripts/predict_today.py

Corre después de daily_features.py.
Carga las features del último día disponible y genera la predicción
de los 4 modelos entrenados para el día de hoy.

Requiere que los modelos estén entrenados (correr notebooks 03 y 05 primero).
"""

import os
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import load_model

# ── Rutas ──────────────────────────────────────────────────────────────────────
FEATURES_DAILY = os.path.join(os.path.dirname(__file__), "../data/processed/features_daily.csv")
MODELS_DIR     = os.path.join(os.path.dirname(__file__), "../models")

FEATURE_COLS_BASE = [
    "RSI_14", "MACD", "BB_position", "return_1d", "return_5d", "volume_change",
    "vix", "t10y2y", "fedfunds", "cpi", "unrate",
]
FEATURE_COLS_SENT = FEATURE_COLS_BASE + ["sentiment_mean", "sentiment_std", "news_count"]

MODELOS = [
    ("Logistic Regression",  "logistic_regression",  FEATURE_COLS_BASE),
    ("XGBoost",              "xgboost_baseline",     FEATURE_COLS_BASE),
    ("XGBoost + Sentiment",  "xgboost_sentiment",    FEATURE_COLS_SENT),
    ("MLP + Sentiment",      "mlp_sentiment",        FEATURE_COLS_SENT),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def cargar_ultima_fila() -> tuple[pd.Series, pd.Timestamp]:
    """Carga la última fila de features_daily.csv."""
    if not os.path.exists(FEATURES_DAILY):
        raise FileNotFoundError(
            f"No se encontró {FEATURES_DAILY}.\n"
            "Corré primero: python scripts/daily_features.py"
        )

    df = pd.read_csv(FEATURES_DAILY, index_col=0, parse_dates=True)
    if df.empty:
        raise ValueError("features_daily.csv está vacío.")

    ultima_fecha = df.index[-1]
    return df.iloc[-1], ultima_fecha


def predecir(fila: pd.Series, nombre_modelo: str,
             model_name: str, feature_cols: list) -> dict:
    """Carga el modelo y genera predicción para la fila dada."""
    try:
        modelo, scaler = load_model(model_name, MODELS_DIR)
    except FileNotFoundError:
        return {"nombre": nombre_modelo, "error": "modelo no entrenado aún"}

    X = fila[feature_cols].values.reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred  = int(modelo.predict(X_scaled)[0])
    proba = float(modelo.predict_proba(X_scaled)[0][1])

    return {
        "nombre":      nombre_modelo,
        "prediccion":  pred,
        "probabilidad": proba,
        "confianza":   max(proba, 1 - proba),
    }


def imprimir_resultado(fecha: pd.Timestamp, resultados: list[dict]):
    """Imprime la predicción de forma clara y legible."""
    print("\n" + "=" * 55)
    print(f"  PREDICCIÓN S&P 500 — {fecha.strftime('%d/%m/%Y')}")
    print(f"  (basada en datos del {fecha.strftime('%d/%m/%Y')})")
    print("=" * 55)

    votos_sube = 0
    votos_baja = 0

    for r in resultados:
        if "error" in r:
            print(f"\n  {r['nombre']:<25} ⚠ {r['error']}")
            continue

        emoji  = "▲ SUBE" if r["prediccion"] == 1 else "▼ BAJA"
        confia = f"({r['confianza']*100:.1f}% confianza)"

        print(f"\n  {r['nombre']:<25} {emoji}  {confia}")

        if r["prediccion"] == 1:
            votos_sube += 1
        else:
            votos_baja += 1

    # Consenso
    total = votos_sube + votos_baja
    if total > 0:
        print("\n" + "-" * 55)
        if votos_sube > votos_baja:
            print(f"  CONSENSO: ▲ SUBE  ({votos_sube}/{total} modelos de acuerdo)")
        elif votos_baja > votos_sube:
            print(f"  CONSENSO: ▼ BAJA  ({votos_baja}/{total} modelos de acuerdo)")
        else:
            print(f"  CONSENSO: ⚖ DIVIDIDO (2 vs 2)")

    print("=" * 55)
    print("\n⚠  Esto no es asesoramiento financiero.")


# ── Pipeline principal ─────────────────────────────────────────────────────────

def main():
    # Cargar última fila de features
    fila, fecha = cargar_ultima_fila()
    print(f"[predict_today] Usando features del {fecha.strftime('%d/%m/%Y')}")

    # Verificar que tenemos todas las features necesarias
    faltantes = [c for c in FEATURE_COLS_SENT if c not in fila.index]
    if faltantes:
        print(f"[predict_today] ADVERTENCIA: features faltantes: {faltantes}")

    # Generar predicciones con cada modelo
    resultados = []
    for nombre, model_name, feature_cols in MODELOS:
        r = predecir(fila, nombre, model_name, feature_cols)
        resultados.append(r)

    # Mostrar resultado
    imprimir_resultado(fecha, resultados)


if __name__ == "__main__":
    main()
