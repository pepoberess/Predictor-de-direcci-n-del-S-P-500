"""
Test de importación: índice GPR (Geopolitical Risk Index) diario.

Fuente: Caldara & Iacoviello (economistas de la Fed), matteoiacoviello.com/gpr.htm
Archivo: data_gpr_daily_recent.xls (serie diaria, se actualiza todos los lunes)

Uso:
    python scraping/test_gpr.py

Solo descarga, parsea y muestra 10 filas — es para validar que el import
funciona y para ver de un vistazo si la serie pinta útil para el modelo
(cobertura temporal, escala de valores, NaNs). No genera ningún archivo
en data/raw/ todavía.
"""

import requests
import pandas as pd

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"

# Columnas reales de la serie (var_name/var_label son un codebook embebido
# en las primeras filas de esas dos columnas, no son datos diarios)
COLS = ["date", "GPRD", "GPRD_ACT", "GPRD_THREAT"]


def descargar_gpr(url: str = GPR_URL) -> pd.DataFrame:
    """Descarga el .xls de GPR diario y lo carga en un DataFrame."""
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    from io import BytesIO
    df = pd.read_excel(BytesIO(resp.content))
    return df


def main():
    print(f"[test_gpr] Descargando {GPR_URL} ...")
    df = descargar_gpr()
    print(f"[test_gpr] Descarga OK. Shape completo: {df.shape}")

    df = df[COLS].copy()
    df["date"] = pd.to_datetime(df["date"])

    print(f"\nPeríodo cubierto: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Nulls por columna:\n{df.isnull().sum()}")

    print("\nPrimeras 10 filas:")
    print(df.head(10).to_string(index=False))

    print("\nÚltimas 10 filas:")
    print(df.tail(10).to_string(index=False))

    print("\nEstadísticas de GPRD (índice diario):")
    print(df["GPRD"].describe())


if __name__ == "__main__":
    main()
