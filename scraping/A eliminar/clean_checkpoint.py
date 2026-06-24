"""
Limpia el checkpoint de Wayback (raw_headlines.jsonl) aplicando el filtro de
exclusión agregado a is_relevant() en wayback_scraper.py (boilerplate de
"rate roundup" personal-finance que se coló en el scraping original), y
regenera sp500_news_extension.csv / sp500_news_production.csv a partir del
checkpoint ya limpio.

Uso:
    python scraping/clean_checkpoint.py
"""

import json
import os

from wayback_scraper import (
    CHECKPOINT_PATH,
    is_relevant,
    load_checkpoint,
    dedupe_across_days,
    split_and_save,
)


def limpiar_checkpoint() -> dict:
    checkpoint = load_checkpoint()

    total_antes = sum(len(v) for v in checkpoint.values())
    limpio = {
        fecha: [t for t in titulos if is_relevant(t)]
        for fecha, titulos in checkpoint.items()
    }
    total_despues = sum(len(v) for v in limpio.values())

    print(f"[clean] Titulares antes: {total_antes:,}")
    print(f"[clean] Titulares despues: {total_despues:,}")
    print(f"[clean] Eliminados por el filtro de exclusion: {total_antes - total_despues:,}")

    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        for fecha in sorted(limpio):
            f.write(json.dumps({"date": fecha, "titles": limpio[fecha]}) + "\n")

    return limpio


def main():
    checkpoint_limpio = limpiar_checkpoint()
    first_seen = dedupe_across_days(checkpoint_limpio)

    today = max(checkpoint_limpio.keys())
    today_fmt = f"{today[:4]}-{today[4:6]}-{today[6:]}"
    split_and_save(first_seen, today_fmt)


if __name__ == "__main__":
    main()
