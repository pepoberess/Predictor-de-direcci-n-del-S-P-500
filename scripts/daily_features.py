"""
Uso:
    python scripts/daily_features.py

Corre cada mañana antes de abrir el mercado.
Descarga los datos del día de trading anterior, calcula las 14 features
y las appendea a data/processed/features_daily.csv.
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv

# Permitir imports desde src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import features as feat
from src.sentiment import run_finbert_batch

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

# ── Rutas ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR     = os.path.join(os.path.dirname(__file__), "../data/processed")
FEATURES_DAILY    = os.path.join(PROCESSED_DIR, "features_daily.csv")

FEATURE_COLS_BASE = [
    "RSI_14", "MACD", "BB_position", "return_1d", "return_5d", "volume_change",
    "vix", "t10y2y", "fedfunds", "cpi", "unrate",
]
FEATURE_COLS_SENT = FEATURE_COLS_BASE + ["sentiment_mean", "sentiment_std", "news_count"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def ultimo_dia_trading() -> datetime.date:
    """Devuelve la fecha del último día de trading completo (ayer o viernes si hoy es lunes)."""
    sp500 = yf.download("^GSPC", period="5d", progress=False)
    return sp500.index[-1].date()


def obtener_precios(fecha: datetime.date) -> pd.DataFrame:
    """Descarga los últimos 60 días de precios del S&P 500.

    Necesitamos la ventana para calcular RSI-14, MACD-26 y BB-20 con precisión.
    """
    inicio = (fecha - timedelta(days=90)).strftime("%Y-%m-%d")
    fin    = fecha.strftime("%Y-%m-%d")
    df = yf.download("^GSPC", start=inicio, end=fin, progress=False)

    # Normalizar columnas al formato que espera src/features.py
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns={"Adj Close": "Close"}) if "Adj Close" in df.columns else df
    return df


def obtener_macro(fecha: datetime.date) -> dict:
    """Obtiene los indicadores macro más recientes disponibles.

    VIX y T10Y2Y: del día anterior via yfinance.
    FEDFUNDS, CPI, UNRATE: último valor publicado en FRED (puede ser del mes anterior).
    """
    # VIX desde yfinance (disponible al día siguiente)
    vix_hist = yf.download("^VIX", period="5d", progress=False)
    vix = float(vix_hist["Close"].iloc[-1])

    # T10Y2Y desde FRED
    from fredapi import Fred
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))

    t10y2y  = float(fred.get_series("T10Y2Y").dropna().iloc[-1])
    fedfunds = float(fred.get_series("FEDFUNDS").dropna().iloc[-1])
    cpi      = float(fred.get_series("CPIAUCSL").dropna().iloc[-1])
    unrate   = float(fred.get_series("UNRATE").dropna().iloc[-1])

    return {
        "vix":      vix,
        "t10y2y":   t10y2y,
        "fedfunds": fedfunds,
        "cpi":      cpi,
        "unrate":   unrate,
    }


def obtener_noticias(fecha: datetime.date) -> list[str]:
    """Obtiene los títulos de noticias del S&P 500 del día anterior via yfinance.

    Filtra por fecha para quedarse solo con noticias de `fecha`.
    """
    ticker = yf.Ticker("^GSPC")
    noticias = ticker.news

    if not noticias:
        print("[daily_features] Sin noticias disponibles en yfinance.")
        return []

    titulos = []
    for n in noticias:
        # providerPublishTime es un timestamp Unix
        fecha_noticia = datetime.fromtimestamp(n.get("providerPublishTime", 0)).date()
        if fecha_noticia == fecha:
            titulo = n.get("title", "")
            if titulo:
                titulos.append(titulo)

    print(f"[daily_features] {len(titulos)} noticias encontradas para {fecha}.")
    return titulos


def calcular_sentiment(titulos: list[str]) -> dict:
    """Corre FinBERT sobre los títulos y devuelve las 3 features de sentiment.

    Si no hay noticias, devuelve ceros (neutralidad de información).
    """
    if not titulos:
        return {"sentiment_mean": 0.0, "sentiment_std": 0.0, "news_count": 0}

    resultados = run_finbert_batch(titulos, batch_size=len(titulos))

    scores = []
    for r in resultados:
        if r["label"] == "positive":
            scores.append(r["score"])
        elif r["label"] == "negative":
            scores.append(-r["score"])
        else:
            scores.append(0.0)

    return {
        "sentiment_mean": float(np.mean(scores)),
        "sentiment_std":  float(np.std(scores)) if len(scores) > 1 else 0.0,
        "news_count":     len(scores),
    }


# ── Pipeline principal ─────────────────────────────────────────────────────────

def main():
    fecha = ultimo_dia_trading()
    print(f"\n[daily_features] Procesando features para: {fecha}")

    # Verificar si ya tenemos features para este día
    if os.path.exists(FEATURES_DAILY):
        existente = pd.read_csv(FEATURES_DAILY, index_col=0, parse_dates=True)
        if pd.Timestamp(fecha) in existente.index:
            print(f"[daily_features] Ya tenemos features para {fecha}. Nada que hacer.")
            return

    # 1. Precios e indicadores técnicos
    print("[daily_features] Descargando precios...")
    precios = obtener_precios(fecha)
    precios_feat = feat.add_technical_indicators(precios)
    fila_tecnica = precios_feat.iloc[-1]  # solo el último día

    # 2. Indicadores macro
    print("[daily_features] Obteniendo macro...")
    macro = obtener_macro(fecha)

    # 3. Noticias y sentiment
    print("[daily_features] Obteniendo noticias...")
    titulos = obtener_noticias(fecha)
    print("[daily_features] Calculando sentiment con FinBERT...")
    sentiment = calcular_sentiment(titulos)

    # 4. Ensamblar vector de features
    fila = {
        # Técnicas
        "RSI_14":        fila_tecnica["RSI_14"],
        "MACD":          fila_tecnica["MACD"],
        "BB_position":   fila_tecnica["BB_position"],
        "return_1d":     fila_tecnica["return_1d"],
        "return_5d":     fila_tecnica["return_5d"],
        "volume_change": fila_tecnica["volume_change"],
        # Macro
        **macro,
        # Sentiment
        **sentiment,
    }

    nueva_fila = pd.DataFrame([fila], index=[pd.Timestamp(fecha)])
    nueva_fila.index.name = "Date"

    # 5. Appendear a features_daily.csv
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if os.path.exists(FEATURES_DAILY):
        nueva_fila.to_csv(FEATURES_DAILY, mode="a", header=False)
    else:
        nueva_fila.to_csv(FEATURES_DAILY)

    print(f"[daily_features] Features guardadas para {fecha}:")
    print(nueva_fila.to_string())


if __name__ == "__main__":
    main()
