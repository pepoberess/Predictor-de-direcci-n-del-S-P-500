"""
Aplica el filtro de relevancia (is_relevant) + el cap diario (DAILY_CAP) sobre
el crudo de Finnhub (sp500_news_finnhub_raw.csv), agrupando por fecha.

No fusiona todavía con sp500_news_extension.csv / production.csv — eso es
un paso aparte, para poder revisar el resultado filtrado antes de reemplazar
la ventana de Wayback.

Uso:
    python scraping/filter_finnhub.py

Genera:
    data/raw/sp500_news_finnhub_filtered.csv   (Title, Date — filtrado y cappeado)
"""

import os
import csv
from collections import defaultdict

from wayback_scraper import filter_relevant, RAW_DIR

INPUT_PATH = os.path.join(RAW_DIR, "sp500_news_finnhub_raw.csv")
OUTPUT_PATH = os.path.join(RAW_DIR, "sp500_news_finnhub_filtered.csv")


def load_raw() -> dict[str, list[str]]:
    por_fecha = defaultdict(list)
    with open(INPUT_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            por_fecha[row["Date"]].append(row["Title"])
    return por_fecha


def main() -> None:
    por_fecha = load_raw()
    total_crudo = sum(len(v) for v in por_fecha.values())

    n_rows = 0
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Date"])
        for fecha in sorted(por_fecha):
            relevantes = filter_relevant(por_fecha[fecha])
            for titulo in relevantes:
                writer.writerow([titulo, fecha])
                n_rows += 1

    dias_vacios = sum(1 for fecha in por_fecha if not filter_relevant(por_fecha[fecha]))
    print(f"[filter_finnhub] Crudo: {total_crudo:,} | Filtrado+cappeado: {n_rows:,}")
    print(f"[filter_finnhub] Fechas totales: {len(por_fecha)} | Fechas con 0 relevantes: {dias_vacios}")
    print(f"[filter_finnhub] Promedio relevantes/día: {n_rows / len(por_fecha):.1f}")
    print(f"[filter_finnhub] Guardado en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
