"""
Extiende los indicadores macro de FRED desde donde termina macro_fred.csv (2024-12-31)
hasta hoy, y separa el resultado en pool de modelado + set de producción.

Uso:
    python scraping/extend_fred.py

Genera:
    data/raw/macro_fred_extension.csv    (2025-01-01 -> CUTOFF-1, pool train/val/test)
    data/raw/macro_fred_production.csv   (CUTOFF -> hoy, fuera del split)

CUTOFF coincide con el usado para sp500_news_production.csv (2025-12-21), para que
los tres datasets (news, precios, macro) compartan el mismo corte de "producción".
"""

import os
import sys
from datetime import date

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

START = "2025-01-01"
CUTOFF = "2025-12-21"  # mismo corte que sp500_news_production.csv
END = date.today().strftime("%Y-%m-%d")

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
EXT_PATH = os.path.join(RAW_DIR, "macro_fred_extension.csv")
PROD_PATH = os.path.join(RAW_DIR, "macro_fred_production.csv")


def descargar_macro(start: str, end: str) -> pd.DataFrame:
    """Replica la lógica de data_loader.download_macro() para un rango arbitrario."""
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))

    print(f"[extend_fred] Descargando FRED ({start} -> {end})...")
    series = {
        "vix":      fred.get_series("VIXCLS",   observation_start=start, observation_end=end),
        "t10y2y":   fred.get_series("T10Y2Y",   observation_start=start, observation_end=end),
        "fedfunds": fred.get_series("FEDFUNDS", observation_start=start, observation_end=end),
        "cpi":      fred.get_series("CPIAUCSL", observation_start=start, observation_end=end),
        "unrate":   fred.get_series("UNRATE",   observation_start=start, observation_end=end),
    }
    df = pd.DataFrame(series)
    df.index.name = "Date"
    df = df.sort_index()
    return df


def main():
    df = descargar_macro(START, END)
    print(f"[extend_fred] Filas totales descargadas: {len(df)} | {df.index[0].date()} -> {df.index[-1].date()}")

    extension = df[df.index < CUTOFF]
    produccion = df[df.index >= CUTOFF]

    extension.to_csv(EXT_PATH)
    produccion.to_csv(PROD_PATH)

    print(f"[extend_fred] {EXT_PATH}: {len(extension)} filas ({extension.index[0].date()} -> {extension.index[-1].date()})")
    print(f"[extend_fred] {PROD_PATH}: {len(produccion)} filas ({produccion.index[0].date()} -> {produccion.index[-1].date()})")


if __name__ == "__main__":
    main()
