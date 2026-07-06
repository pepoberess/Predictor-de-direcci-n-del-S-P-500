import os
import pandas as pd
import numpy as np
from io import BytesIO
import requests


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calcula el RSI (Relative Strength Index) con el suavizado de Wilder,
    día por día, siguiendo la fórmula original:

        ganancia[t] = cambio[t] si cambio[t] > 0, si no 0
        pérdida[t]  = -cambio[t] si cambio[t] < 0, si no 0

        primer promedio (día `period`):
            avg_gain = promedio simple de las primeras `period` ganancias
            avg_loss = promedio simple de las primeras `period` pérdidas

        días siguientes (suavizado de Wilder):
            avg_gain[t] = (avg_gain[t-1] * (period - 1) + ganancia[t]) / period
            avg_loss[t] = (avg_loss[t-1] * (period - 1) + pérdida[t]) / period

        RS[t]  = avg_gain[t] / avg_loss[t]
        RSI[t] = 100 - 100 / (1 + RS[t])

    Parámetros
    ----------
    close : pd.Series
        Serie de precios de cierre.
    period : int
        Cantidad de días para el promedio (14 por defecto).

    Retorna
    -------
    pd.Series
        RSI en el rango [0, 100]. NaN en los primeros `period` días.
    """
    rsi = pd.Series(index=close.index, dtype=float)
    ganancias_iniciales = []
    perdidas_iniciales = []
    avg_gain = None
    avg_loss = None

    for i in range(len(close)):
        if i == 0:
            rsi.iloc[i] = np.nan
            continue

        cambio = close.iloc[i] - close.iloc[i - 1]
        ganancia = cambio if cambio > 0 else 0.0
        perdida = -cambio if cambio < 0 else 0.0

        if i < period:
            # Todavía no hay suficientes días para el primer promedio
            ganancias_iniciales.append(ganancia)
            perdidas_iniciales.append(perdida)
            rsi.iloc[i] = np.nan
        elif i == period:
            # Primer promedio: simple, sobre las primeras `period` ganancias/pérdidas
            ganancias_iniciales.append(ganancia)
            perdidas_iniciales.append(perdida)
            avg_gain = sum(ganancias_iniciales) / period
            avg_loss = sum(perdidas_iniciales) / period
            rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
            rsi.iloc[i] = 100 - (100 / (1 + rs))
        else:
            # Suavizado de Wilder: pondera el promedio anterior con el valor de hoy
            avg_gain = (avg_gain * (period - 1) + ganancia) / period
            avg_loss = (avg_loss * (period - 1) + perdida) / period
            rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
            rsi.iloc[i] = 100 - (100 / (1 + rs))

    return rsi


def _macd(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """
    Calcula el MACD: diferencia entre la EMA rápida y la EMA lenta del precio,
    día por día, siguiendo la fórmula recursiva de la EMA:

        alpha = 2 / (period + 1)
        EMA[0] = close[0]
        EMA[t] = alpha * close[t] + (1 - alpha) * EMA[t-1]

        MACD[t] = EMA_rápida[t] - EMA_lenta[t]

    Parámetros
    ----------
    close : pd.Series
        Serie de precios de cierre.
    fast : int
        Cantidad de días de la EMA rápida (12 por defecto).
    slow : int
        Cantidad de días de la EMA lenta (26 por defecto).

    Retorna
    -------
    pd.Series
        MACD, en las mismas unidades que el precio (puntos, no %).
    """
    alpha_fast = 2 / (fast + 1)
    alpha_slow = 2 / (slow + 1)

    macd = pd.Series(index=close.index, dtype=float)
    ema_fast = None
    ema_slow = None

    for i in range(len(close)):
        precio = close.iloc[i]

        if i == 0:
            ema_fast = precio
            ema_slow = precio
        else:
            ema_fast = alpha_fast * precio + (1 - alpha_fast) * ema_fast
            ema_slow = alpha_slow * precio + (1 - alpha_slow) * ema_slow

        macd.iloc[i] = ema_fast - ema_slow

    return macd


def _bb_position(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Calcula la posición del precio dentro de las Bandas de Bollinger,
    día por día, siguiendo la fórmula:

        promedio[t] = promedio de close en los últimos `period` días (incluido hoy)
        desvío[t]   = desvío estándar muestral (ddof=1) de esa misma ventana
        banda_superior[t] = promedio[t] + 2 * desvío[t]
        banda_inferior[t] = promedio[t] - 2 * desvío[t]
        posición[t] = (close[t] - banda_inferior[t]) / (banda_superior[t] - banda_inferior[t])

    posición = 0 -> el precio está en la banda inferior
    posición = 1 -> el precio está en la banda superior
    Valores fuera de [0, 1] son posibles en movimientos muy bruscos (breakouts).

    Parámetros
    ----------
    close : pd.Series
        Serie de precios de cierre.
    period : int
        Cantidad de días de la ventana (20 por defecto).

    Retorna
    -------
    pd.Series
        Posición relativa dentro de las bandas. NaN en los primeros `period - 1` días.
    """
    posicion = pd.Series(index=close.index, dtype=float)

    for i in range(len(close)):
        if i < period - 1:
            posicion.iloc[i] = np.nan
            continue

        ventana = close.iloc[i - period + 1 : i + 1]

        promedio = sum(ventana) / period
        varianza = sum((x - promedio) ** 2 for x in ventana) / (period - 1)
        desvio = varianza ** 0.5

        banda_superior = promedio + 2 * desvio
        banda_inferior = promedio - 2 * desvio
        ancho_banda = banda_superior - banda_inferior

        if ancho_banda == 0:
            # Volatilidad cero en la ventana: evita división por cero (muy raro en S&P 500)
            posicion.iloc[i] = np.nan
        else:
            posicion.iloc[i] = (close.iloc[i] - banda_inferior) / ancho_banda

    return posicion


def _retorno(serie: pd.Series, dias: int) -> pd.Series:
    """
    Calcula el retorno porcentual entre el valor de hoy y el de hace días,
    día por día:

        retorno[t] = (serie[t] - serie[t - dias]) / serie[t - dias]

    Se usa tanto para retornos de precio (return_1d, return_5d) como para la
    variación de volumen (volume_change).

    Parámetros
    ----------
    serie : pd.Series
        Serie de valores (precio de cierre o volumen).
    dias : int
        Cantidad de días hacia atrás con los que comparar.

    Retorna
    -------
    pd.Series
        Retorno porcentual (ej. 0.01 = +1%). NaN en los primeros `dias` días.
    """
    retorno = pd.Series(index=serie.index, dtype=float)

    for i in range(len(serie)):
        if i < dias:
            retorno.iloc[i] = np.nan
            continue

        valor_hoy = serie.iloc[i]
        valor_anterior = serie.iloc[i - dias]

        if valor_anterior == 0:
            # Evita división por cero (irrelevante para Close, posible en teoría para Volume)
            retorno.iloc[i] = np.nan
        else:
            retorno.iloc[i] = (valor_hoy - valor_anterior) / valor_anterior

    return retorno


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

    Todos los indicadores están escritos a mano, día por día, siguiendo
    directamente su fórmula matemática — sin usar atajos de pandas como
    ewm(), rolling() o pct_change() — para que se entienda exactamente
    qué cuenta se hace en cada paso (ver _rsi, _macd, _bb_position, _retorno).

    Features agregadas:
        RSI_14        : Relative Strength Index de 14 días (0-100)
        MACD          : diferencia EMA-12 - EMA-26, en puntos (momentum tendencial)
        BB_position   : posición relativa dentro de Bandas de Bollinger de 20 días
        return_1d     : retorno porcentual respecto al día anterior
        return_5d     : retorno porcentual respecto a hace 5 días
        volume_change : variación porcentual del volumen respecto al día anterior

    Las primeras filas tendrán NaN hasta que haya suficientes días de historia
    (RSI_14: 14 días, BB_position: 19 días). MACD queda definido desde el primer día.

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

    df["RSI_14"]        = _rsi(close, period=14)
    df["MACD"]          = _macd(close, fast=12, slow=26)
    df["BB_position"]   = _bb_position(close, period=20)
    df["return_1d"]     = _retorno(close, dias=1)
    df["return_5d"]     = _retorno(close, dias=5)
    df["volume_change"] = _retorno(volume, dias=1)

    return df


def descargar_gpr(GPR_URL: str, GPR_COLS: list) -> pd.DataFrame:
    """
    Descarga el índice GPR diario (Caldara & Iacoviello) desde matteoiacoviello.com.

    Parámetros
    ----------
    GPR_URL : str
        URL del .xls con la serie diaria reciente.
    GPR_COLS : list
        Columnas de interés a extraer (ej. GPRD, GPRD_ACT, GPRD_THREAT).

    Retorna
    -------
    pd.DataFrame
        Índice DatetimeIndex (fecha de observación, sin shiftear), columnas GPR_COLS.
    """
    print(f"[gpr] Descargando {GPR_URL} ...")
    resp = requests.get(GPR_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()

    df = pd.read_excel(BytesIO(resp.content))
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date"] + GPR_COLS].set_index("date").sort_index()
    return df


def mergear_archivo(path: str, gpr: pd.DataFrame, GPR_COLS: list) -> None:
    """
    Mergea las columnas GPR en un CSV de macro existente (in-place).

    Idempotente: si el archivo ya tiene columnas GPR de una corrida anterior,
    las pisa en vez de duplicarlas.

    Parámetros
    ----------
    path : str
        Ruta al CSV de macro a enriquecer.
    gpr : pd.DataFrame
        Salida de descargar_gpr().
    GPR_COLS : list
        Nombres de las columnas GPR a mergear.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    if any(c in df.columns for c in GPR_COLS):
        print(f"[gpr] {path} ya tiene columnas GPR, se sobreescriben.")
        df = df.drop(columns=[c for c in GPR_COLS if c in df.columns])

    antes = df.shape
    df = df.join(gpr, how="left")
    df.to_csv(path)

    faltantes = df[GPR_COLS].isnull().any(axis=1).sum()
    print(f"[gpr] {path}: {antes} -> {df.shape} | filas sin match GPR: {faltantes}")


def add_geopolitical_risk_index_macro() -> None:
    """
    Agrega el índice GPR (Geopolitical Risk, Caldara & Iacoviello) como columnas nuevas
    en los archivos de macro de FRED. GPR cubre 1985-hoy sin huecos.

    Se llama una sola vez desde el notebook 00, después de download_macro().

    Modifica (append de columnas, no de filas):
        data/raw/macro_fred.csv
        data/raw/macro_fred_production.csv

    IMPORTANTE — sin shift todavía: esto agrega el valor RAW de GPR (fecha de
    observación tal como viene en el .xls). La serie solo se actualiza los
    lunes (ver scraping/A eliminar/test_gpr.py), así que antes de usar estas
    columnas como features hace falta el mismo tipo de shift a fecha de
    publicación real que ya tienen CPI/UNRATE en add_macro_features() — eso
    todavía NO está implementado acá.
    """
    GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
    GPR_COLS = ["GPRD", "GPRD_ACT", "GPRD_THREAT"]

    RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    TARGETS = [
        os.path.join(RAW_DIR, "macro_fred.csv"),
    ]

    gpr = descargar_gpr(GPR_URL, GPR_COLS)
    print(f"[gpr] GPR descargado: {gpr.shape} | {gpr.index[0].date()} -> {gpr.index[-1].date()}")

    for path in TARGETS:
        mergear_archivo(path, gpr, GPR_COLS)





def add_macro_features(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Hace merge de precios con indicadores macro y aplica forward fill.

    VIX, T10Y2Y y FEDFUNDS ya vienen forward-filled desde download_macro().
    Esta función shiftea a su fecha real de publicación (look-ahead bias fix)
    CPI, UNRATE (mensuales) y GPRD/GPRD_ACT/GPRD_THREAT (semanales, si están
    presentes en el DataFrame de macro), y hace ffill de todas ellas tras el
    shift (quedan sparse). Solo se conservan días de trading.

    CRÍTICO: cada serie se shiftea a su fecha real de *publicación*, no la
    fecha de *observación* que trae el archivo (ej. el dato de marzo viene fechado
    2024-03-01, pero el BLS lo publica recién a mediados de ABRIL).
    Sin este shift el modelo vería el dato antes de que existiera públicamente.
    CPI: día 13 del mes siguiente al observado.
    UNRATE: primer viernes del mes siguiente al observado.
    GPRD/GPRD_ACT/GPRD_THREAT: próximo lunes después del día observado — la
    serie diaria de Caldara & Iacoviello se actualiza en batch los lunes, no
    en tiempo real día a día (ver scraping/A eliminar/test_gpr.py). Si el
    DataFrame de macro no tiene estas columnas, se ignoran sin error.
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

    def _proximo_lunes(fecha):
        dias_hasta_lunes = (7 - fecha.weekday()) % 7  # weekday(): lunes=0 ... domingo=6
        if dias_hasta_lunes == 0:
            dias_hasta_lunes = 7  # si "fecha" ya es lunes, el próximo lunes es en 7 días
        return fecha + pd.Timedelta(days=dias_hasta_lunes)

    # resample("MS") extrae la observación del 1ro de cada mes sin depender de nulls:
    # funciona igual si CPI/UNRATE vienen sparse (raw FRED) o ya ffilled (download_macro).
    cpi_publicado = macro["cpi"].resample("MS").first().dropna()
    cpi_publicado.index = cpi_publicado.index.map(_dia_13_mes_siguiente)

    unrate_publicado = macro["unrate"].resample("MS").first().dropna()
    unrate_publicado.index = unrate_publicado.index.map(_primer_viernes_mes_siguiente)

    macro = macro.drop(columns=["cpi", "unrate"])
    macro = macro.join(cpi_publicado, how="outer").join(unrate_publicado, how="outer")

    # GPR (si está presente): es diaria, pero se publica en batch los lunes.
    # Agrupamos cada día observado bajo su "próximo lunes" (fecha real de
    # publicación) y nos quedamos con el último valor de cada semana — mismo
    # criterio que CPI/UNRATE (un valor representativo por período), antes
    # de shiftear.
    GPR_COLS = ["GPRD", "GPRD_ACT", "GPRD_THREAT"]
    gpr_cols_presentes = [c for c in GPR_COLS if c in macro.columns]

    if gpr_cols_presentes:
        gpr_crudo = macro[gpr_cols_presentes].dropna(how="all")
        fechas_publicacion = gpr_crudo.index.map(_proximo_lunes)
        gpr_publicado = gpr_crudo.groupby(fechas_publicacion).last()

        macro = macro.drop(columns=gpr_cols_presentes)
        macro = macro.join(gpr_publicado, how="outer")

    # ffill cubre CPI/UNRATE/GPR tras el shift (quedan sparse) y los huecos del outer join.
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
