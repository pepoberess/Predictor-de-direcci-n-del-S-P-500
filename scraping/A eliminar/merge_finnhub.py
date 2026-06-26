"""
Reemplaza por completo la porción ago-2025/mar-2026 de sp500_news_extension.csv
y sp500_news_production.csv (Wayback, floja en esa ventana) por la versión
filtrada de Finnhub (sp500_news_finnhub_filtered.csv).

El corte extension/production se mantiene igual al que ya tenían los archivos
(la fecha mínima actual de production.csv), para no introducir una inconsistencia
nueva de fechas entre Wayback y Finnhub.

Uso:
    python scraping/merge_finnhub.py
"""

import os
import csv
from datetime import datetime

from wayback_scraper import RAW_DIR

GAP_START = "2025-08-01"
GAP_END = "2026-03-31"

EXTENSION_PATH = os.path.join(RAW_DIR, "sp500_news_extension.csv")
PRODUCTION_PATH = os.path.join(RAW_DIR, "sp500_news_production.csv")
FINNHUB_FILTERED_PATH = os.path.join(RAW_DIR, "sp500_news_finnhub_filtered.csv")


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_rows(path: str, rows: list[dict]) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["Date"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Date"])
        for r in rows_sorted:
            writer.writerow([r["Title"], r["Date"]])


def main() -> None:
    extension = load_rows(EXTENSION_PATH)
    production = load_rows(PRODUCTION_PATH)
    finnhub = load_rows(FINNHUB_FILTERED_PATH)

    cutoff = min(r["Date"] for r in production)
    print(f"[merge_finnhub] Corte extension/production: {cutoff}")

    n_ext_antes, n_prod_antes = len(extension), len(production)

    extension = [r for r in extension if not (GAP_START <= r["Date"] <= GAP_END)]
    production = [r for r in production if not (GAP_START <= r["Date"] <= GAP_END)]

    n_ext_quitadas = n_ext_antes - len(extension)
    n_prod_quitadas = n_prod_antes - len(production)
    print(f"[merge_finnhub] Filas de Wayback quitadas en la ventana {GAP_START}/{GAP_END}: "
          f"{n_ext_quitadas} (extension) + {n_prod_quitadas} (production)")

    n_finnhub_ext, n_finnhub_prod = 0, 0
    for r in finnhub:
        if r["Date"] >= cutoff:
            production.append(r)
            n_finnhub_prod += 1
        else:
            extension.append(r)
            n_finnhub_ext += 1

    print(f"[merge_finnhub] Filas de Finnhub agregadas: {n_finnhub_ext} (extension) + {n_finnhub_prod} (production)")

    save_rows(EXTENSION_PATH, extension)
    save_rows(PRODUCTION_PATH, production)

    print(f"[merge_finnhub] {EXTENSION_PATH}: {len(extension):,} filas finales")
    print(f"[merge_finnhub] {PRODUCTION_PATH}: {len(production):,} filas finales")


if __name__ == "__main__":
    main()
