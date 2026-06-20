"""
Script de predicción para el test set externo publicado por la cátedra.

Uso:
    python predict.py --test_path ruta/al/test.csv --output_dir ./predicciones/

El test set debe tener al menos las columnas: Date, Close, High, Low, Open, Volume.
Se generan tres archivos CSV con las predicciones de cada modelo:
    Salivaras_Beresten_LogisticRegression_predictions.csv
    Salivaras_Beresten_XGBoost_predictions.csv
    Salivaras_Beresten_MLP_predictions.csv

Cada CSV tiene columnas: Date, prediction (0=baja, 1=sube), probability.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src import data_loader, features, sentiment, models

RAW_DIR       = os.path.join(os.path.dirname(__file__), "data/raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data/processed")
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "models")
CACHE_PATH    = os.path.join(PROCESSED_DIR, "sentiment_cache.csv")

FEATURE_COLS_BASE = [
    "RSI_14", "MACD", "BB_position", "return_1d", "return_5d", "volume_change",
    "vix", "t10y2y", "fedfunds", "cpi", "unrate"
]
FEATURE_COLS_SENT = FEATURE_COLS_BASE + ["sentiment_mean", "sentiment_std", "news_count"]

APELLIDOS = "Salivaras_Beresten"


def load_test_prices(test_path: str) -> pd.DataFrame:
    """Carga y valida el CSV del test set publicado por la cátedra."""
    print(f"[predict] Cargando test set desde {test_path}...")
    df = pd.read_csv(test_path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    df = df.sort_index()

    cols_requeridas = {"Close", "High", "Low", "Open", "Volume"}
    faltantes = cols_requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"El test set no tiene las columnas: {faltantes}")

    print(f"[predict] Test set: {df.shape} | {df.index[0].date()} → {df.index[-1].date()}")
    return df


def build_test_features(test_df: pd.DataFrame, include_sentiment: bool) -> pd.DataFrame:
    """Construye las features del test set aplicando el mismo pipeline que en entrenamiento.

    Para indicadores que requieren datos históricos (RSI, MACD, BB), concatenamos
    los precios de data/raw/sp500_raw.csv con el test set para garantizar que los
    primeros días del test no queden en NaN.
    """
    # Cargar precios históricos para contexto de indicadores técnicos
    hist = data_loader.load_sp500(RAW_DIR)

    # Concatenar histórico + test, eliminando solapamientos
    combined = pd.concat([hist, test_df[~test_df.index.isin(hist.index)]])
    combined = combined.sort_index()

    # Calcular indicadores técnicos sobre el dataset completo
    combined_feat = features.add_technical_indicators(combined)

    # Cargar macro y hacer merge
    macro = data_loader.load_macro(RAW_DIR)
    combined_macro = features.add_macro_features(combined_feat, macro)

    # Filtrar solo las filas del test set
    test_features = combined_macro[combined_macro.index.isin(test_df.index)].copy()

    if include_sentiment:
        # Cargar sentiment cacheado y agregarlo
        if not os.path.exists(CACHE_PATH):
            print("[predict] ADVERTENCIA: sentiment_cache.csv no encontrado. "
                  "Corriendo predicción sin sentiment para modelos con sentiment.")
            test_features["sentiment_mean"] = 0.0
            test_features["sentiment_std"]  = 0.0
            test_features["news_count"]     = 0.0
        else:
            news = data_loader.load_news(RAW_DIR)
            raw_sent = pd.read_csv(CACHE_PATH, parse_dates=["Date"])
            sent_diario = sentiment.aggregate_daily_sentiment(raw_sent)
            test_features = test_features.join(
                sent_diario[["sentiment_mean", "sentiment_std", "news_count"]],
                how="left"
            )
            test_features["sentiment_mean"] = test_features["sentiment_mean"].fillna(0)
            test_features["sentiment_std"]  = test_features["sentiment_std"].fillna(0)
            test_features["news_count"]     = test_features["news_count"].fillna(0)

    return test_features


def generate_predictions(test_features: pd.DataFrame, model_name: str,
                         feature_cols: list, output_dir: str) -> str:
    """Carga el modelo guardado y genera predicciones sobre el test set.

    Guarda el CSV en output_dir con el formato requerido por la cátedra.
    Retorna la ruta del archivo generado.
    """
    modelo, scaler = models.load_model(model_name, MODELS_DIR)

    X_test = scaler.transform(test_features[feature_cols].fillna(0).values)
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    # Formato del CSV requerido por la cátedra
    nombre_modelo_display = {
        "logistic_regression": "LogisticRegression",
        "xgboost_baseline":    "XGBoost",
        "xgboost_sentiment":   "XGBoost",
        "mlp_sentiment":       "MLP",
    }.get(model_name, model_name)

    nombre_archivo = f"{APELLIDOS}_{nombre_modelo_display}_predictions.csv"
    ruta = os.path.join(output_dir, nombre_archivo)

    resultado = pd.DataFrame({
        "Date":        test_features.index,
        "prediction":  y_pred,
        "probability": y_prob.round(4),
    })
    resultado.to_csv(ruta, index=False)
    print(f"[predict] Predicciones guardadas en {ruta} ({len(resultado)} filas)")
    return ruta


def main():
    parser = argparse.ArgumentParser(description="Genera predicciones sobre el test set del S&P 500.")
    parser.add_argument("--test_path",  required=True, help="Ruta al CSV del test set")
    parser.add_argument("--output_dir", default="./predicciones", help="Directorio para los CSVs de salida")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Cargar y preprocesar test set
    test_df = load_test_prices(args.test_path)

    # Generar predicciones con features base (LR + XGBoost baseline)
    test_base = build_test_features(test_df, include_sentiment=False)
    generate_predictions(test_base, "logistic_regression", FEATURE_COLS_BASE, args.output_dir)
    generate_predictions(test_base, "xgboost_baseline",    FEATURE_COLS_BASE, args.output_dir)

    # Generar predicciones con sentiment (XGBoost + MLP)
    test_sent = build_test_features(test_df, include_sentiment=True)
    generate_predictions(test_sent, "xgboost_sentiment", FEATURE_COLS_SENT, args.output_dir)
    generate_predictions(test_sent, "mlp_sentiment",     FEATURE_COLS_SENT, args.output_dir)

    print(f"\n[predict] Listo. Predicciones generadas en {args.output_dir}/")
    print(f"  Archivos: {os.listdir(args.output_dir)}")


if __name__ == "__main__":
    main()
