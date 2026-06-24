import os
import re
import json
import csv
import time
from datetime import datetime

import requests
from dateutil.relativedelta import relativedelta

CDX_API = "https://web.archive.org/cdx/search/cdx"
WAYBACK_BASE = "https://web.archive.org/web"
TARGET_URL = "https://finance.yahoo.com/news/"
FALLBACK_URL = "https://finance.yahoo.com/"

START_DATE = "2024-03-04"
PRODUCTION_MONTHS = 6

PHRASE_KEYWORDS = [
    "s&p 500", "s&p500", "dow jones", "stock market", "wall street", "wall st",
    "federal reserve", "interest rate", "jobs report", "treasury yield",
    "bond yield", "market rally", "market sell-off", "earnings season",
    "rate cut", "rate hike", "treasury secretary", "retail sales",
    "industrial production",
]
WORD_KEYWORDS = [
    "dow", "stocks?", "markets?", "fed", "nasdaq", "tariffs?", "recession",
    "unemployment", "inflation", "gdp", "powell", "breadth", "index(es)?",
    "indices", "valuations?", "correction",
]
WORD_KEYWORD_PATTERN = re.compile(r"\b(" + "|".join(WORD_KEYWORDS) + r")\b")
DAILY_CAP = 20

# Excluye boilerplate de "rate roundup" personal-finance que matchea las keywords
# de arriba por casualidad (ej. "market" en "money market account") pero no tiene
# nada que ver con la dirección del S&P 500. Validado contra 8,040 titulares ya
# scrapeados: 431 filas (5.4%), sin falsos positivos en muestra de 20 al azar.
EXCLUDE_PATTERN = re.compile(
    r"money market account rates"
    r"|savings (interest|account) rates today"
    r"|best [a-z-]* ?savings [a-z ]*rates today"
    r"|mortgage (and refinance )?rates? today"
    r"|real estate market trends in"
    r"|\bdow inc\b"
    r"|nyse: ?dow\b",
    re.IGNORECASE,
)

SCRAPING_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(SCRAPING_DIR, "checkpoint", "raw_headlines.jsonl")
RAW_DIR = os.path.join(SCRAPING_DIR, "..", "data", "raw")
EXTENSION_PATH = os.path.join(RAW_DIR, "sp500_news_extension.csv")
PRODUCTION_PATH = os.path.join(RAW_DIR, "sp500_news_production.csv")

REQUEST_DELAY_SECONDS = 1.5
MAX_RETRIES = 3
USER_AGENT = "Mozilla/5.0 (compatible; research-scraper/1.0)"

STORY_PATTERN = re.compile(
    r'\\"contentType\\":\\"STORY\\",\\"title\\":\\"((?:\\\\.|[^"\\])*)\\"'
)
H3_FALLBACK_PATTERN = re.compile(r"<h3[^>]*>([^<]+)</h3>")
TICKER_PREFIX_PATTERN = re.compile(r"(nasdaq|nyse):")


def list_snapshots(url: str, date_from: str, date_to: str) -> list[tuple[str, str]]:
    params = {
        "url": url,
        "from": date_from.replace("-", ""),
        "to": date_to.replace("-", ""),
        "output": "json",
        "collapse": "timestamp:8",
        "filter": "statuscode:200",
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(CDX_API, params=params, timeout=90)
            resp.raise_for_status()
            rows = resp.json()[1:]
            return [(row[1][:8], row[1]) for row in rows]
        except requests.exceptions.RequestException as e:
            print(f"[list_snapshots] intento {attempt + 1}/{MAX_RETRIES} falló: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("No se pudo listar snapshots tras varios reintentos.")


def snapshot_for_date(url: str, date_str: str) -> str | None:
    """Busca un único snapshot puntual de `url` para `date_str` (YYYYMMDD).

    A diferencia de list_snapshots(), no debe interrumpir todo el scraping
    si falla — es una búsqueda de respaldo, no la lista principal.
    """
    try:
        snapshots = list_snapshots(url, date_str, date_str)
    except RuntimeError:
        return None
    return snapshots[0][1] if snapshots else None


def fetch_snapshot_html(url: str, timestamp: str) -> str | None:
    full_url = f"{WAYBACK_BASE}/{timestamp}/{url}"
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(full_url, headers={"User-Agent": USER_AGENT}, timeout=60)
        except requests.exceptions.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if resp.status_code == 200:
            return resp.text
        if resp.status_code == 503:
            time.sleep(5 * (attempt + 1))
            continue
        return None
    return None


def extract_titles(html: str) -> list[str]:
    matches = STORY_PATTERN.findall(html)
    if matches:
        titles = []
        for m in matches:
            try:
                titles.append(json.loads(f'"{m}"'))
            except json.JSONDecodeError:
                continue
        return titles
    return [t.strip() for t in H3_FALLBACK_PATTERN.findall(html) if t.strip()]


def is_relevant(title: str) -> bool:
    if EXCLUDE_PATTERN.search(title):
        return False
    lowered = TICKER_PREFIX_PATTERN.sub("", title.lower())
    if any(kw in lowered for kw in PHRASE_KEYWORDS):
        return True
    return bool(WORD_KEYWORD_PATTERN.search(lowered))


def filter_relevant(titles: list[str], limit: int = DAILY_CAP) -> list[str]:
    seen = set()
    relevant = []
    for title in titles:
        if not is_relevant(title):
            continue
        if title in seen:
            continue
        seen.add(title)
        relevant.append(title)
    return relevant[:limit]


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


def scrape(date_from: str, date_to: str) -> None:
    done = load_checkpoint()
    snapshots = list_snapshots(TARGET_URL, date_from, date_to)
    # Reprocesar también los días ya en checkpoint pero vacíos (0 titulares):
    # ahí es donde el fallback puede recuperar algo que antes no se intentó.
    pending = [(d, ts) for d, ts in snapshots if d not in done or len(done[d]) == 0]
    print(f"[scrape] {len(snapshots)} días con snapshot, {len(pending)} pendientes de procesar.")

    n_fallback = 0
    for date_str, timestamp in pending:
        html = fetch_snapshot_html(TARGET_URL, timestamp)
        titles = filter_relevant(extract_titles(html)) if html else []

        if not titles:
            fallback_ts = snapshot_for_date(FALLBACK_URL, date_str)
            if fallback_ts:
                fallback_html = fetch_snapshot_html(FALLBACK_URL, fallback_ts)
                if fallback_html:
                    titles = filter_relevant(extract_titles(fallback_html))
                    if titles:
                        n_fallback += 1

        append_checkpoint(date_str, titles)
        print(f"[scrape] {date_str}: {len(titles)} titulares relevantes.")
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"[scrape] Días recuperados vía fallback ({FALLBACK_URL}): {n_fallback}")


def dedupe_across_days(checkpoint: dict[str, list[str]]) -> dict[str, str]:
    first_seen = {}
    for date_str in sorted(checkpoint):
        for title in checkpoint[date_str]:
            if title not in first_seen:
                first_seen[title] = date_str
    return first_seen


def split_and_save(first_seen: dict[str, str], date_to: str) -> None:
    cutoff_dt = datetime.strptime(date_to, "%Y-%m-%d") - relativedelta(months=PRODUCTION_MONTHS)
    cutoff = cutoff_dt.strftime("%Y%m%d")

    rows = sorted(first_seen.items(), key=lambda x: x[1])
    os.makedirs(RAW_DIR, exist_ok=True)

    n_ext, n_prod = 0, 0
    with open(EXTENSION_PATH, "w", newline="", encoding="utf-8") as f_ext, \
         open(PRODUCTION_PATH, "w", newline="", encoding="utf-8") as f_prod:
        w_ext = csv.writer(f_ext)
        w_prod = csv.writer(f_prod)
        w_ext.writerow(["Title", "Date"])
        w_prod.writerow(["Title", "Date"])
        for title, date_str in rows:
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            if date_str >= cutoff:
                w_prod.writerow([title, formatted_date])
                n_prod += 1
            else:
                w_ext.writerow([title, formatted_date])
                n_ext += 1

    print(f"[split] {EXTENSION_PATH}: {n_ext} filas (pool train/val/test)")
    print(f"[split] {PRODUCTION_PATH}: {n_prod} filas (producción, últimos {PRODUCTION_MONTHS} meses)")


def main() -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    scrape(START_DATE, today)
    checkpoint = load_checkpoint()
    first_seen = dedupe_across_days(checkpoint)
    split_and_save(first_seen, today)


if __name__ == "__main__":
    main()
