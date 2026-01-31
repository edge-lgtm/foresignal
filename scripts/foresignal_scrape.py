# scripts/foresignal_scrape.py
#
# Daily dynamic scraper for foresignal.com that works WITHOUT running JavaScript.
# It decodes the site's obfuscated values from <script>f('...')</script>.
#
# Output:
#   data/foresignal_signals_YYYY-MM-DD.csv
#   data/foresignal_signals_history.csv  (appends each run)
#
# Install:
#   pip install requests beautifulsoup4 pandas
#
# Run:
#   python scripts/foresignal_scrape.py

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://foresignal.com/en/"
TZ = ZoneInfo("Asia/Manila")

# Foresignal's decode map (from site JS):
# '670429+-. 5,813'.charAt(s.charCodeAt(i)-65-i)
MAP = "670429+-. 5,813"

# Fallback numeric regex (for pages where values appear as plain text)
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def fetch_html(url: str, timeout: int = 30) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def decode_f(encoded: str) -> str:
    """
    Decode foresignal's obfuscated string used in:
      <script>f('NJPIOK');</script>
    """
    out = []
    for i, ch in enumerate(encoded):
        idx = ord(ch) - 65 - i
        if 0 <= idx < len(MAP):
            out.append(MAP[idx])
    return "".join(out).strip()


def extract_encoded_from_script(script_text: str) -> str | None:
    """
    Extract the 'XXXX' from f('XXXX');
    """
    if not script_text:
        return None
    m = re.search(r"f\(\s*'([^']+)'\s*\)", script_text)
    return m.group(1) if m else None


def value_from_signal_value(value_el) -> str | None:
    """
    Extract the numeric value from a .signal-value element.
    Prefer decoding <script>f('...')</script>, otherwise fallback to plain numeric text.
    """
    if value_el is None:
        return None

    # 1) If there's a script inside, decode it
    script_el = value_el.find("script")
    if script_el and script_el.get_text(strip=True):
        enc = extract_encoded_from_script(script_el.get_text(strip=True))
        if enc:
            decoded = decode_f(enc)
            if decoded:
                return decoded

    # 2) Fallback: attempt to read any plain number text
    txt = value_el.get_text(" ", strip=True)
    txt = re.sub(r"\s+", " ", txt)

    # Remove any leftover "f('...')" artifacts if present in text
    txt = re.sub(r"f\(\s*'[^']+'\s*\)\s*;?", "", txt).strip()

    m = NUM_RE.search(txt)
    return m.group(0) if m else None


def parse_signals(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")

    rows: list[dict] = []

    for card in soup.select(".card.signal-card"):
        pair_el = card.select_one(".card-header a[href*='/signals/']")
        if not pair_el:
            continue

        pair = pair_el.get_text(strip=True)

        status_el = card.select_one(".signal-row.signal-status")
        status = status_el.get_text(strip=True) if status_el else None

        data = {
            "pair": pair,
            "status": status,
            "sell_at": None,
            "take_profit_at": None,
            "stop_loss_at": None,
            "buy_at": None,
            "bought_at": None,
            "sold_at": None,
        }

        for row in card.select(".signal-row"):
            title_el = row.select_one(".signal-title")
            value_el = row.select_one(".signal-value")
            if not title_el or not value_el:
                continue

            title = title_el.get_text(" ", strip=True)
            value = value_from_signal_value(value_el)

            if title == "Sell at":
                data["sell_at"] = value
            elif title.startswith("Take profit"):
                data["take_profit_at"] = value
            elif title == "Stop loss at":
                data["stop_loss_at"] = value
            elif title == "Buy at":
                data["buy_at"] = value
            elif title == "Bought at":
                data["bought_at"] = value
            elif title == "Sold at":
                data["sold_at"] = value

        rows.append(data)

    df = pd.DataFrame(rows)
    cols = [
        "pair",
        "status",
        "sell_at",
        "take_profit_at",
        "stop_loss_at",
        "buy_at",
        "bought_at",
        "sold_at",
    ]
    return df.reindex(columns=cols)


def main() -> None:
    now = datetime.now(TZ)
    date_tag = now.strftime("%Y-%m-%d")

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    html = fetch_html(URL)
    df = parse_signals(html)

    # Daily snapshot
    daily_path = out_dir / f"foresignal_signals_{date_tag}.csv"
    df.to_csv(daily_path, index=False)

    # Rolling history (append)
    history_path = out_dir / "foresignal_signals_history.csv"
    df2 = df.copy()
    df2.insert(0, "pulled_at", now.isoformat(timespec="seconds"))

    header = not history_path.exists()
    df2.to_csv(history_path, mode="a", header=header, index=False)

    print(df.to_string(index=False))
    print(f"\nWrote: {daily_path}")
    print(f"Updated: {history_path}")


if __name__ == "__main__":
    main()
