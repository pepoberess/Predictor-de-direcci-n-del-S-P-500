import os
import pandas as pd
import numpy as np


# ── Indicadores técnicos internos ─────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI con suavizado exponencial (método Wilder)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """MACD: diferencia entre EMA rápida y EMA lenta."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def _bb_position(close: pd.Series, period: int = 20) -> pd.Series:
    """Posición dentro de las Bandas de Bollinger.

    Resultado entre 0 y 1: 0 = en la banda inferior, 1 = en la banda superior.
    Valores fuera de [0,1] son posibles en breakouts extremos.
    """
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    # Evitar división por cero en períodos de volatilidad cero (muy raro en S&P 500)
    band_width = upper - lower
    return (close - lower) / band_width.replace(0, np.nan)


# ── API pública ────────────────────────────────────────────────────────────────

def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columna 'target': 1 si Close mañana > Close hoy, 0 si no.

    CRÍTICO: usa shift(-1) para mirar el cierre del día siguiente.
    La última fila se elimina porque no tiene target conocido.
    No shifteamos las features — el modelo ve features de hoy para predecir mañana.
    """
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.iloc[:-1]  # eliminar última fila sin target
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 6 indicadores técnicos calculados sobre los precios del S&P 500.

    Features agregadas:
    - RSI_14: Relative Strength Index de 14 días (0–100)
    - MACD: diferencia EMA-12 - EMA-26 (momentum tendencial)
    - BB_position: posición relativa dentro de Bandas de Bollinger de 20 días
    - return_1d: retorno aritmético diario (Close_t / Close_{t-1} - 1)
    - return_5d: retorno acumulado últimos 5 días de trading
    - volume_change: variación porcentual del volumen respecto al día anterior

    Las primeras filas tendrán NaN hasta que haya suficientes períodos (máx 26 días).
    """
    df = df.copy()
    close = df["Close"]
    volume = df["Volume"]

    df["RSI_14"]       = _rsi(close, 14)
    df["MACD"]         = _macd(close)
    df["BB_position"]  = _bb_position(close)
    df["return_1d"]    = close.pct_change(1)
    df["return_5d"]    = close.pct_change(5)
    df["volume_change"] = volume.pct_change(1)

    return df


def add_macro_features(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Hace merge de precios con indicadores macro y aplica forward fill.

    VIX y T10Y2Y son series diarias con pocos huecos (fines de semana/feriados).
    FEDFUNDS, CPI y UNRATE son mensuales — el forward fill las lleva a frecuencia diaria.
    Solo se conservan las fechas que aparecen en el índice de prices (días de trading).
    """
    # Reindexar macro al rango de fechas de prices, luego forward fill
    macro_reindexed = macro.reindex(prices.index, method=None)
    macro_ffill = macro_reindexed.ffill()

    merged = prices.join(macro_ffill, how="left")
    return merged


def build_feature_matrix(
    sp500_features: pd.DataFrame,
    sentiment: pd.DataFrame | None = None,
    include_sentiment: bool = True,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Construye la matriz de features final lista para entrenar.

    Args:
        sp500_features: output de add_technical_indicators() + add_macro_features()
                        con columna 'target' ya calculada
        sentiment: output de sentiment.aggregate_daily_sentiment() (ya shifteado 1 día)
                   Columnas esperadas: sentiment_mean, sentiment_std, news_count
        include_sentiment: si False, devuelve solo técnicas + macro (Experimentos 1 y 2)
        save_path: si se provee, guarda el CSV en esa ruta

    Retorna DataFrame con índice DatetimeIndex, sin NaN en las features definidas,
    listo para pasar a models.temporal_split().
    """
    df = sp500_features.copy()

    if include_sentiment and sentiment is not None:
        df = df.join(sentiment[["sentiment_mean", "sentiment_std", "news_count"]], how="left")
        # Días sin noticias: sentiment neutro (0) y 0 noticias
        df["sentiment_mean"] = df["sentiment_mean"].fillna(0)
        df["sentiment_std"]  = df["sentiment_std"].fillna(0)
        df["news_count"]     = df["news_count"].fillna(0)

    # Eliminar filas con NaN en features técnicas (los primeros ~26 días)
    feature_cols = [c for c in df.columns if c != "target"]
    df = df.dropna(subset=feature_cols)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print(f"[features] feature_matrix guardada en {save_path} ({len(df):,} filas).")

    return df
