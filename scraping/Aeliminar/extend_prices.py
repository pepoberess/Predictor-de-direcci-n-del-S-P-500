"""
Extiende los precios del S&P 500 desde donde termina sp500_raw.csv (2024-12-30)
hasta hoy, y separa el resultado en pool de modelado + set de producción.

Uso:
    python scraping/extend_prices.py

Genera:
    data/raw/sp500_raw_extension.csv    (2025-01-01 -> CUTOFF-1, pool train/val/test)
    data/raw/sp500_raw_production.csv   (CUTOFF -> hoy, fuera del split)

CUTOFF coincide con el usado para sp500_news_production.csv y macro_fred_production.csv.
"""

import os
from datetime import date

import yfinance as yf
import pandas as pd

START = "2025-01-01"
CUTOFF = "2025-12-21"
END = date.today().strftime("%Y-%m-%d")

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
EXT_PATH = os.path.join(RAW_DIR, "sp500_raw_extension.csv")
PROD_PATH = os.path.join(RAW_DIR, "sp500_raw_production.csv")


def main():
    print(f"[extend_prices] Descargando ^GSPC ({START} -> {END})...")
    df = yf.download("^GSPC", start=START, end=END, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.sort_index()

    print(f"[extend_prices] Filas totales descargadas: {len(df)} | {df.index[0].date()} -> {df.index[-1].date()}")

    extension = df[df.index < CUTOFF]
    produccion = df[df.index >= CUTOFF]

    extension.to_csv(EXT_PATH)
    produccion.to_csv(PROD_PATH)

    print(f"[extend_prices] {EXT_PATH}: {len(extension)} filas ({extension.index[0].date()} -> {extension.index[-1].date()})")
    print(f"[extend_prices] {PROD_PATH}: {len(produccion)} filas ({produccion.index[0].date()} -> {produccion.index[-1].date()})")


if __name__ == "__main__":
    main()
