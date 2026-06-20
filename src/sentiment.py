import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_or_run_finbert(news_df: pd.DataFrame, cache_path: str) -> pd.DataFrame:
    """Punto de entrada principal para el pipeline de sentiment.

    Si sentiment_cache.csv existe y tiene la misma cantidad de filas que news_df,
    lo carga directamente sin volver a correr FinBERT.
    Si no, corre FinBERT sobre todos los titulares y guarda el resultado en cache_path.

    Retorna DataFrame con columnas: Title, Date, CP, sentiment_label, sentiment_score.
    """
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path, parse_dates=["Date"])
        if len(cache) == len(news_df):
            print(f"[sentiment] Cache encontrado en {cache_path} ({len(cache):,} filas). Saltando FinBERT.")
            return cache
        else:
            print(f"[sentiment] Cache incompleto ({len(cache):,}/{len(news_df):,} filas). Re-corriendo FinBERT.")

    print(f"[sentiment] Corriendo FinBERT sobre {len(news_df):,} titulares...")
    resultados = run_finbert_batch(news_df["Title"].tolist())

    df_out = news_df.copy()
    df_out["sentiment_label"] = [r["label"] for r in resultados]
    df_out["sentiment_score"] = [r["score"] for r in resultados]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_out.to_csv(cache_path, index=False)
    print(f"[sentiment] Cache guardado en {cache_path}.")
    return df_out


def run_finbert_batch(texts: list, batch_size: int = 32, device: str = "mps") -> list:
    """Corre ProsusAI/finbert sobre una lista de textos en batches.

    Usa el backend MPS de PyTorch (GPU de Apple Silicon M-series).
    Fallback automático a CPU si MPS no está disponible.

    Args:
        texts: lista de strings (titulares de noticias)
        batch_size: tamaño del batch. 32 es óptimo para M5 Pro con MPS.
        device: 'mps' (Apple Silicon), 'cuda' (NVIDIA), o 'cpu'

    Retorna lista de dicts con keys:
        - 'label': 'positive', 'negative', o 'neutral'
        - 'score': confianza del modelo (float, 0-1)
    """
    import torch
    from transformers import pipeline

    # Fallback a CPU si el device pedido no está disponible
    if device == "mps" and not torch.backends.mps.is_available():
        print("[sentiment] MPS no disponible, usando CPU.")
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("[sentiment] CUDA no disponible, usando CPU.")
        device = "cpu"

    print(f"[sentiment] Cargando ProsusAI/finbert en device='{device}'...")
    pipe = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        device=device,
        truncation=True,
        max_length=512,
    )

    resultados = []
    for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT"):
        batch = texts[i : i + batch_size]
        salida = pipe(batch, batch_size=batch_size)
        resultados.extend(salida)

    return resultados


def aggregate_daily_sentiment(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega scores de FinBERT a nivel diario y shiftea 1 día para evitar leakage.

    Convierte los labels a score numérico firmado:
        positive  → +score
        negative  → -score
        neutral   → 0

    Agrega por fecha:
        sentiment_mean: promedio del score firmado (señal direccional)
        sentiment_std:  desvío estándar (dispersión/incertidumbre del día)
        news_count:     cantidad de noticias del día

    CRÍTICO: aplica .shift(1) antes de retornar para que el modelo reciba el
    sentiment de ayer, no el de hoy. Esto evita data leakage temporal.

    Retorna DataFrame con índice DatetimeIndex.
    """
    df = raw_df.copy()

    # Convertir label a score firmado
    def signed_score(row):
        if row["sentiment_label"] == "positive":
            return row["sentiment_score"]
        elif row["sentiment_label"] == "negative":
            return -row["sentiment_score"]
        return 0.0

    df["signed_score"] = df.apply(signed_score, axis=1)

    # Asegurar que Date sea datetime y sea el índice
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # Agregación diaria
    diario = df.groupby("Date").agg(
        sentiment_mean=("signed_score", "mean"),
        sentiment_std=("signed_score", "std"),
        news_count=("signed_score", "count"),
    )

    # Rellenar NaN en std (días con una sola noticia)
    diario["sentiment_std"] = diario["sentiment_std"].fillna(0)

    # SHIFT de 1 día: el modelo ve el sentiment de ayer para predecir hoy
    diario = diario.shift(1)

    return diario
