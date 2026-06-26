"""
Trae titulares CRUDOS (sin filtrar) de Finnhub (símbolo SPY) para el rango
ago-2025 a mar-2026, donde Wayback Machine tiene muy poca cobertura.

No aplica ningún filtro de relevancia acá a propósito — eso se decide después,
con el corpus completo ya en mano, para no depender de pegarle de nuevo a la
API si se ajustan las keywords.

Uso:
    python scraping/finnhub_backfill.py

Genera:
    data/raw/sp500_news_finnhub_raw.csv   (Title, Date — sin filtrar)
"""

import os
import csv
import json
import time
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

FINNHUB_API = "https://finnhub.io/api/v1/company-news"
SYMBOL = "SPY"
GAP_START = "2025-08-01"
GAP_END = "2026-03-31"

SCRAPING_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(SCRAPING_DIR, "checkpoint", "finnhub_raw.jsonl")
RAW_DIR = os.path.join(SCRAPING_DIR, "..", "data", "raw")
OUTPUT_PATH = os.path.join(RAW_DIR, "sp500_news_finnhub_raw.csv")

REQUEST_DELAY_SECONDS = 1.1  # margen bajo el límite de 60 req/min del free tier
MAX_RETRIES = 3


def daterange(start: str, end: str):
    d0 = datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.strptime(end, "%Y-%m-%d")
    while d0 <= d1:
        yield d0.strftime("%Y-%m-%d")
        d0 += timedelta(days=1)


def fetch_day(token: str, date_str: str) -> list[str]:
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(FINNHUB_API, params={
                "symbol": SYMBOL, "from": date_str, "to": date_str, "token": token,
            }, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            return [item["headline"] for item in data]
        except requests.exceptions.RequestException as e:
            print(f"[finnhub] {date_str}: intento {attempt + 1}/{MAX_RETRIES} falló: {e}")
            time.sleep(3 * (attempt + 1))
    return []


def load_checkpoint() -> dict[str, list[str]]:
    done = {}
    if not os.path.exists(CHECKPOINT_PATH):
        return done
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done[row["date"]] = row["titles"]
    return done


def append_checkpoint(date_str: str, titles: list[str]) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"date": date_str, "titles": titles}) + "\n")


def fetch_all(token: str) -> None:
    done = load_checkpoint()
    dias = list(daterange(GAP_START, GAP_END))
    pending = [d.replace("-", "") for d in dias if d.replace("-", "") not in done]
    pending_fmt = [d for d in dias if d.replace("-", "") not in done]
    print(f"[finnhub] {len(dias)} días en el rango, {len(pending)} pendientes de bajar.")

    for date_str in pending_fmt:
        titles = fetch_day(token, date_str)
        append_checkpoint(date_str.replace("-", ""), titles)
        print(f"[finnhub] {date_str}: {len(titles)} titulares crudos.")
        time.sleep(REQUEST_DELAY_SECONDS)


def save_raw_csv() -> None:
    checkpoint = load_checkpoint()
    os.makedirs(RAW_DIR, exist_ok=True)

    n_rows = 0
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Date"])
        for date_str in sorted(checkpoint):
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            for title in checkpoint[date_str]:
                writer.writerow([title, formatted_date])
                n_rows += 1

    print(f"[finnhub] {OUTPUT_PATH}: {n_rows} filas (SIN filtrar, símbolo {SYMBOL}).")


def main() -> None:
    token = os.getenv("FINNHUB_API_KEY")
    if not token:
        raise ValueError("FINNHUB_API_KEY no encontrada. Definila en .env.")
    fetch_all(token)
    save_raw_csv()


if __name__ == "__main__":
    main()
