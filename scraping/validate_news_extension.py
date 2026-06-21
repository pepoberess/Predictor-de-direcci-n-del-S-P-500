"""
Valida si el dataset de noticias extendido (Wayback Machine) sigue una
distribución similar al dataset original (Kaggle), en 3 niveles:

1. Estadísticas descriptivas: longitud de título, densidad de noticias/día,
   cobertura de días.
2. Similaridad de vocabulario: TF-IDF + cosine similarity entre los
   centroides de cada corpus, + superposición de términos más frecuentes.
3. Distribución de sentiment FinBERT: % positive/negative/neutral, con test
   chi-cuadrado (y Cramér's V como tamaño de efecto, porque con miles de
   filas el chi-cuadrado solo casi siempre da "significativo").

Uso:
    python scraping/validate_news_extension.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import sentiment

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

ORIGINAL_PATH = os.path.join(RAW_DIR, "sp500_news.csv")
EXTENSION_PATH = os.path.join(RAW_DIR, "sp500_news_extension.csv")

CACHE_ORIGINAL = os.path.join(PROCESSED_DIR, "sentiment_cache.csv")
CACHE_EXTENSION = os.path.join(PROCESSED_DIR, "sentiment_cache_extension.csv")


def cargar_datasets() -> tuple:
    orig = pd.read_csv(ORIGINAL_PATH, parse_dates=["Date"])
    ext = pd.read_csv(EXTENSION_PATH, parse_dates=["Date"])
    return orig, ext


# ── 1. Estadísticas descriptivas ────────────────────────────────────────────

def estadisticas_descriptivas(orig: pd.DataFrame, ext: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("1. ESTADISTICAS DESCRIPTIVAS")
    print("=" * 70)

    for nombre, df in [("Original", orig), ("Extension", ext)]:
        dias_calendario = (df["Date"].max() - df["Date"].min()).days + 1
        dias_con_noticias = df["Date"].nunique()
        cobertura = 100 * dias_con_noticias / dias_calendario
        promedio_con_cobertura = df.groupby(df["Date"].dt.date).size().mean()
        promedio_calendario = len(df) / dias_calendario

        print(f"\n{nombre}:")
        print(f"  Filas totales:                {len(df):,}")
        print(f"  Periodo:                      {df['Date'].min().date()} -> {df['Date'].max().date()}")
        print(f"  Cobertura de dias:            {cobertura:.1f}%")
        print(f"  Noticias/dia (con cobertura): {promedio_con_cobertura:.2f}")
        print(f"  Noticias/dia (calendario):    {promedio_calendario:.2f}")
        print(f"  Longitud titulo (media):      {df['Title'].str.len().mean():.1f}")
        print(f"  Longitud titulo (mediana):    {df['Title'].str.len().median():.1f}")


# ── 2. Similaridad de vocabulario: TF-IDF + cosine ─────────────────────────

def similaridad_tfidf(orig: pd.DataFrame, ext: pd.DataFrame, top_n: int = 15) -> None:
    print("\n" + "=" * 70)
    print("2. SIMILARIDAD DE VOCABULARIO (TF-IDF + cosine similarity)")
    print("=" * 70)

    corpus = pd.concat([orig["Title"], ext["Title"]], ignore_index=True)
    es_original = np.array([True] * len(orig) + [False] * len(ext))

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    matriz = vectorizer.fit_transform(corpus)

    centroide_orig = np.asarray(matriz[es_original].mean(axis=0))
    centroide_ext = np.asarray(matriz[~es_original].mean(axis=0))

    sim = cosine_similarity(centroide_orig, centroide_ext)[0, 0]
    print(f"\nCosine similarity entre centroides (original vs extension): {sim:.4f}")
    print("(1.0 = vocabulario identico en proporcion de uso, 0.0 = sin superposicion)")

    vocab = np.array(vectorizer.get_feature_names_out())
    top_orig = vocab[centroide_orig.flatten().argsort()[::-1][:top_n]]
    top_ext = vocab[centroide_ext.flatten().argsort()[::-1][:top_n]]

    print(f"\nTop {top_n} terminos - ORIGINAL:")
    print(", ".join(top_orig))
    print(f"\nTop {top_n} terminos - EXTENSION:")
    print(", ".join(top_ext))

    overlap = len(set(top_orig) & set(top_ext))
    print(f"\nSuperposicion en el top {top_n}: {overlap}/{top_n} terminos en comun")


# ── 3. Distribucion de sentiment FinBERT ────────────────────────────────────

def distribucion_finbert(orig: pd.DataFrame, ext: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("3. DISTRIBUCION DE SENTIMENT (FinBERT)")
    print("=" * 70)

    print("\n[validate] Cargando/corriendo FinBERT sobre el original...")
    raw_orig = sentiment.load_or_run_finbert(orig, CACHE_ORIGINAL)

    print("[validate] Cargando/corriendo FinBERT sobre la extension...")
    raw_ext = sentiment.load_or_run_finbert(ext, CACHE_EXTENSION)

    dist_orig = raw_orig["sentiment_label"].value_counts(normalize=True) * 100
    dist_ext = raw_ext["sentiment_label"].value_counts(normalize=True) * 100

    tabla = pd.DataFrame({"Original (%)": dist_orig, "Extension (%)": dist_ext}).round(1)
    print(f"\n{tabla}")

    print(f"\nConfianza promedio - Original:  {raw_orig['sentiment_score'].mean():.3f}")
    print(f"Confianza promedio - Extension: {raw_ext['sentiment_score'].mean():.3f}")

    tabla_contingencia = pd.DataFrame({
        "original": raw_orig["sentiment_label"].value_counts(),
        "extension": raw_ext["sentiment_label"].value_counts(),
    }).fillna(0)

    chi2, p, dof, _ = chi2_contingency(tabla_contingencia.T)
    n = tabla_contingencia.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(tabla_contingencia.shape) - 1)))

    print(f"\nTest chi-cuadrado (H0: misma distribucion de labels):")
    print(f"  chi2 = {chi2:.2f}, p-value = {p:.6f}")
    print(f"  Cramer's V (tamano de efecto, 0=nulo, 1=maximo) = {cramers_v:.4f}")
    print(f"\n  Nota: con miles de filas el chi-cuadrado casi siempre da p < 0.05")
    print(f"  aunque la diferencia practica sea chica. Cramer's V es la medida")
    print(f"  que importa para decidir si el efecto es relevante (>0.1 = chico,")
    print(f"  >0.3 = moderado, >0.5 = grande).")


def main():
    orig, ext = cargar_datasets()
    estadisticas_descriptivas(orig, ext)
    similaridad_tfidf(orig, ext)
    distribucion_finbert(orig, ext)


if __name__ == "__main__":
    main()
