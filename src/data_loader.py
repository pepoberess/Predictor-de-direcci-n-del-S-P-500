import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()


def load_sp500(raw_dir: str) -> pd.DataFrame:
    """Carga precios S&P 500 desde data/raw/sp500_raw.csv.

    Maneja el formato especial que genera yfinance al guardar con to_csv():
    las primeras 3 filas son encabezados del multi-index, no datos.
    Retorna DataFrame con índice DatetimeIndex y columnas: Close, High, Low, Open, Volume.
    """
    path = os.path.join(raw_dir, "sp500_raw.csv")
    df = pd.read_csv(
        path,
        skiprows=3,
        header=None,
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
        index_col=0,
        parse_dates=True,
    )
    df.index.name = "Date"
    df = df.sort_index()
    return df


def load_macro(raw_dir: str) -> pd.DataFrame:
    """Carga indicadores macro de FRED desde data/raw/macro_fred.csv.

    No aplica forward fill — eso lo hace features.py al alinear con días de trading.
    Retorna DataFrame con índice DatetimeIndex y columnas: vix, t10y2y, fedfunds, cpi, unrate.
    """
    path = os.path.join(raw_dir, "macro_fred.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    df = df.sort_index()
    return df


def load_news(raw_dir: str) -> pd.DataFrame:
    """Carga el dataset de noticias desde data/raw/sp500_news.csv.

    Retorna DataFrame con columnas: Title (str), Date (datetime), CP (float).
    19,127 noticias, 3,507 fechas únicas, cobertura 2008-2024.
    """
    path = os.path.join(raw_dir, "sp500_news.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def download_sp500(start: str, end: str, save_path: str) -> pd.DataFrame:
    """Descarga precios del S&P 500 via yfinance y guarda en save_path.

    Si el archivo ya existe, lo carga sin descargar. Útil para no re-descargar
    en cada ejecución y para reproducibilidad offline.
    Args:
        start: fecha inicio en formato 'YYYY-MM-DD'
        end: fecha fin en formato 'YYYY-MM-DD'
        save_path: ruta absoluta donde guardar el CSV
    """
    if os.path.exists(save_path):
        print(f"[data_loader] sp500_raw.csv ya existe en {save_path}, se carga sin descargar.")
        return load_sp500(os.path.dirname(save_path))

    print(f"[data_loader] Descargando ^GSPC de yfinance ({start} → {end})...")
    df = yf.download("^GSPC", start=start, end=end)
    df.to_csv(save_path)
    print(f"[data_loader] Guardado en {save_path} ({len(df):,} filas).")
    return load_sp500(os.path.dirname(save_path))


def download_macro(start: str, end: str, save_path: str,
                   api_key: str = None) -> pd.DataFrame:
    """Descarga indicadores de FRED via fredapi y guarda en save_path.

    Si el archivo ya existe, lo carga sin descargar.
    Lee FRED_API_KEY desde .env si api_key no se pasa explícitamente.
    Series descargadas: VIXCLS, T10Y2Y, FEDFUNDS, CPIAUCSL, UNRATE.
    """
    if os.path.exists(save_path):
        print(f"[data_loader] macro_fred.csv ya existe en {save_path}, se carga sin descargar.")
        return load_macro(os.path.dirname(save_path))

    key = api_key or os.getenv("FRED_API_KEY")
    if not key:
        raise ValueError("FRED_API_KEY no encontrada. Definila en .env o pasala como argumento.")

    fred = Fred(api_key=key)
    print(f"[data_loader] Descargando indicadores de FRED ({start} → {end})...")

    series = {
        "vix":      fred.get_series("VIXCLS",    observation_start=start, observation_end=end),
        "t10y2y":   fred.get_series("T10Y2Y",    observation_start=start, observation_end=end),
        "fedfunds": fred.get_series("FEDFUNDS",  observation_start=start, observation_end=end),
        "cpi":      fred.get_series("CPIAUCSL",  observation_start=start, observation_end=end),
        "unrate":   fred.get_series("UNRATE",    observation_start=start, observation_end=end),
    }

    df = pd.DataFrame(series)
    df.index.name = "Date"
    df.to_csv(save_path)
    print(f"[data_loader] Guardado en {save_path} ({len(df):,} filas).")
    return df
