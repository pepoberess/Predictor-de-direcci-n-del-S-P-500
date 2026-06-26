import os
import pandas as pd
import numpy as np


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calcula el RSI con suavizado exponencial (método Wilder).

    Parámetros
    ----------
    close : pd.Series
        Serie de precios de cierre.
    period : int
        Cantidad de períodos para el promedio exponencial.

    Retorna
    -------
    pd.Series
        RSI en el rango [0, 100].
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """
    Calcula el MACD: diferencia entre EMA rápida y EMA lenta.

    Parámetros
    ----------
    close : pd.Series
        Serie de precios de cierre.
    fast : int
        Cantidad de períodos de la EMA rápida.
    slow : int
        Cantidad de períodos de la EMA lenta.

    Retorna
    -------
    pd.Series
        MACD (momentum tendencial).
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def _bb_position(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Calcula la posición relativa dentro de las Bandas de Bollinger.

    Resultado entre 0 y 1: 0 = en la banda inferior, 1 = en la banda superior.
    Valores fuera de [0,1] son posibles en breakouts extremos.

    Parámetros
    ----------
    close : pd.Series
        Serie de precios de cierre.
    period : int
        Ventana para la media móvil y el desvío estándar.

    Retorna
    -------
    pd.Series
        Posición relativa dentro de las bandas.
    """
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    # Evitar división por cero en períodos de volatilidad cero (muy raro en S&P 500)
    band_width = upper - lower
    return (close - lower) / band_width.replace(0, np.nan)


def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega la columna 'target': 1 si Close de mañana > Close de hoy, 0 si no.

    CRÍTICO: usa shift(-1) para mirar el cierre del día siguiente.
    La última fila se elimina porque no tiene target conocido.
    No shifteamos las features — el modelo ve features de hoy para predecir mañana.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con índice DatetimeIndex y columna 'Close'.

    Retorna
    -------
    pd.DataFrame
        Mismo DataFrame con columna 'target' agregada y la última fila eliminada.
    """
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.iloc[:-1]  # eliminar última fila sin target
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega 6 indicadores técnicos calculados sobre los precios del S&P 500.

    Features agregadas:
        RSI_14        : Relative Strength Index de 14 días (0-100)
        MACD          : diferencia EMA-12 - EMA-26 (momentum tendencial)
        BB_position   : posición relativa dentro de Bandas de Bollinger de 20 días
        return_1d     : retorno aritmético diario (Close_t / Close_{t-1} - 1)
        return_5d     : retorno acumulado de los últimos 5 días de trading
        volume_change : variación porcentual del volumen respecto al día anterior

    Las primeras filas tendrán NaN hasta que haya suficientes períodos (máx 26 días).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con índice DatetimeIndex y columnas 'Close', 'Volume'.

    Retorna
    -------
    pd.DataFrame
        Mismo DataFrame con las 6 columnas de indicadores agregadas.
    """
    df = df.copy()
    close = df["Close"]
    volume = df["Volume"]

    df["RSI_14"]        = _rsi(close, 14)
    df["MACD"]          = _macd(close)
    df["BB_position"]   = _bb_position(close)
    df["return_1d"]     = close.pct_change(1)
    df["return_5d"]     = close.pct_change(5)
    df["volume_change"] = volume.pct_change(1).replace([np.inf, -np.inf], np.nan)

    return df


def add_macro_features(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Hace merge de precios con indicadores macro y aplica forward fill.

    VIX, T10Y2Y y FEDFUNDS ya vienen forward-filled desde download_macro().
    Esta función se encarga únicamente de shiftear CPI y UNRATE a su fecha real
    de publicación (look-ahead bias fix) y de hacer ffill de esas dos series
    tras el shift (quedan sparse). Solo se conservan días de trading.

    CRÍTICO: CPI y UNRATE se shiftean a su fecha real de *publicación*, no la
    fecha de *observación* que trae el archivo (ej. el dato de marzo viene fechado
    2024-03-01, pero el BLS lo publica recién a mediados de ABRIL).
    Sin este shift el modelo vería el dato antes de que existiera públicamente.
    CPI: día 13 del mes siguiente al observado.
    UNRATE: primer viernes del mes siguiente al observado.
    FEDFUNDS no se shiftea — es la tasa efectiva, conocida casi sin demora.

    Parámetros
    ----------
    prices : pd.DataFrame
        DataFrame de precios con índice DatetimeIndex (días de trading).
    macro : pd.DataFrame
        DataFrame de indicadores macro, salida de data_loader.load_macro_extended().

    Retorna
    -------
    pd.DataFrame
        prices con las columnas de macro agregadas y forward-filled.
    """
    macro = macro.copy()

    def _dia_13_mes_siguiente(fecha):
        return (fecha + pd.DateOffset(months=1)).replace(day=13)

    def _primer_viernes_mes_siguiente(fecha):
        primero = (fecha + pd.DateOffset(months=1)).replace(day=1)
        dias_hasta_viernes = (4 - primero.weekday()) % 7  # weekday(): lunes=0 ... viernes=4
        return primero + pd.Timedelta(days=dias_hasta_viernes)

    # resample("MS") extrae la observación del 1ro de cada mes sin depender de nulls:
    # funciona igual si CPI/UNRATE vienen sparse (raw FRED) o ya ffilled (download_macro).
    cpi_publicado = macro["cpi"].resample("MS").first().dropna()
    cpi_publicado.index = cpi_publicado.index.map(_dia_13_mes_siguiente)

    unrate_publicado = macro["unrate"].resample("MS").first().dropna()
    unrate_publicado.index = unrate_publicado.index.map(_primer_viernes_mes_siguiente)

    macro = macro.drop(columns=["cpi", "unrate"])
    macro = macro.join(cpi_publicado, how="outer").join(unrate_publicado, how="outer")

    # ffill cubre CPI/UNRATE tras el shift (quedan sparse) y los huecos del outer join.
    # vix, t10y2y y fedfunds ya vienen ffilled desde download_macro().
    calendario_completo = macro.index.union(prices.index)
    macro_reindexed = macro.reindex(calendario_completo).ffill()
    macro_reindexed = macro_reindexed.reindex(prices.index)

    merged = prices.join(macro_reindexed, how="left")
    return merged


def build_feature_matrix(
    sp500_features: pd.DataFrame,
    sentiment: pd.DataFrame | None = None,
    include_sentiment: bool = True,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Construye la matriz de features final lista para entrenar.

    Parámetros
    ----------
    sp500_features : pd.DataFrame
        Output de add_technical_indicators() + add_macro_features(),
        con columna 'target' ya calculada.
    sentiment : pd.DataFrame, opcional
        Output de sentiment.aggregate_daily_sentiment() (ya shifteado 1 día).
        Columnas esperadas: sentiment_mean, sentiment_std, news_count.
    include_sentiment : bool
        Si False, devuelve solo técnicas + macro (Experimentos 1 y 2).
    save_path : str, opcional
        Si se provee, guarda la matriz resultante como CSV en esa ruta.

    Retorna
    -------
    pd.DataFrame
        Índice DatetimeIndex, sin NaN en las features definidas,
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
