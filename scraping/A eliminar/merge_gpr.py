"""
Agrega el índice GPR (Geopolitical Risk, Caldara & Iacoviello) como columnas nuevas
en los 3 archivos de macro de FRED: el original (2008-2024), la extensión y el de
producción. GPR cubre 1985-hoy sin huecos, así que alcanza para los tres.

Uso:
    python scraping/merge_gpr.py

Modifica (append de columnas, no de filas):
    data/raw/macro_fred.csv
    data/raw/macro_fred_extension.csv
    data/raw/macro_fred_production.csv
"""

import os
from io import BytesIO

import requests
import pandas as pd

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
GPR_COLS = ["GPRD", "GPRD_ACT", "GPRD_THREAT"]

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
TARGETS = [
    os.path.join(RAW_DIR, "macro_fred.csv"),
    os.path.join(RAW_DIR, "macro_fred_extension.csv"),
    os.path.join(RAW_DIR, "macro_fred_production.csv"),
]


def descargar_gpr() -> pd.DataFrame:
    print(f"[merge_gpr] Descargando {GPR_URL} ...")
    resp = requests.get(GPR_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()

    df = pd.read_excel(BytesIO(resp.content))
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date"] + GPR_COLS].set_index("date").sort_index()
    return df


def mergear_archivo(path: str, gpr: pd.DataFrame) -> None:
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    if any(c in df.columns for c in GPR_COLS):
        print(f"[merge_gpr] {path} ya tiene columnas GPR, se sobreescriben.")
        df = df.drop(columns=[c for c in GPR_COLS if c in df.columns])

    antes = df.shape
    df = df.join(gpr, how="left")
    df.to_csv(path)

    faltantes = df[GPR_COLS].isnull().any(axis=1).sum()
    print(f"[merge_gpr] {path}: {antes} -> {df.shape} | filas sin match GPR: {faltantes}")


def main():
    gpr = descargar_gpr()
    print(f"[merge_gpr] GPR descargado: {gpr.shape} | {gpr.index[0].date()} -> {gpr.index[-1].date()}")

    for path in TARGETS:
        mergear_archivo(path, gpr)


if __name__ == "__main__":
    main()
