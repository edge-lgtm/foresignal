# scripts/foresignal_scrape.py
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

MAP = "670429+-. 5,813"
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
    out = []
    for i, ch in enumerate(encoded):
        idx = ord(ch) - 65 - i
        if 0 <= idx < len(MAP):
            out.append(MAP[idx])
    return "".join(out).strip()


def extract_encoded_from_script(script_text: str) -> str | None:
    if not script_text:
        return None
    m = re.search(r"f\(\s*'([^']+)'\s*\)", script_text)
    return m.group(1) if m else None


def value_from_signal_value(value_el) -> str | None:
    if value_el is None:
        return None

    # Prefer decoding <script>f('...')</script>
    script_el = value_el.find("script")
    if script_el:
        enc = extract_encoded_from_script(script_el.get_text(strip=True))
        if enc:
            decoded = decode_f(enc)
            if decoded:
                return decoded

    # Fallback: plain numeric text
    txt = value_el.get_text(" ", strip=True)
    txt = re.sub(r"\s+", " ", txt).strip()
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
        status = status_el.get_text(strip=True) if status_el else ""

        data = {
            "pair": pair,
            "status": status,
            "sell_at": "",
            "take_profit_at": "",
            "stop_loss_at": "",
            "buy_at": "",
            "bought_at": "",
            "sold_at": "",
        }

        for row in card.select(".signal-row"):
            title_el = row.select_one(".signal-title")
            value_el = row.select_one(".signal-value")
            if not title_el or not value_el:
                continue

            title = title_el.get_text(" ", strip=True)
            value = value_from_signal_value(value_el) or ""

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
    df = pd.DataFrame(rows, columns=cols)

    # Ensure all price columns stay as strings (prevents NaN)
    for c in ["sell_at", "take_profit_at", "stop_loss_at", "buy_at", "bought_at", "sold_at"]:
        df[c] = df[c].astype(str).replace({"None": "", "nan": ""})

    return df


def main() -> None:
    now = datetime.now(TZ)
    date_tag = now.strftime("%Y-%m-%d")

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    html = fetch_html(URL)
    df = parse_signals(html)

    daily_path = out_dir / f"foresignal_signals_{date_tag}.csv"
    df.to_csv(daily_path, index=False)

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
