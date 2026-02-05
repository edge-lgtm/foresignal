from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup
import smtplib

from decimal import Decimal, ROUND_HALF_UP

SCALE_EURUSD = Decimal("10000")

def mini_to_price(value):
    return (Decimal(str(value)) / SCALE_EURUSD).quantize(
        Decimal("0.00001"), rounding=ROUND_HALF_UP
    )
# =========================
# ENV + CONFIG
# =========================
def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


URL = "https://foresignal.com/en/"
TZ = ZoneInfo("Asia/Manila")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

LAST_STATE_FILE = DATA_DIR / "latest_signals.json"
TRADES_LOG_FILE = DATA_DIR / "trades_history.jsonl"
ORDERED_FILE = DATA_DIR / "ordered_keys.json"

IG_DEMO = (os.getenv("IG_DEMO", "true").lower() in ("1", "true", "yes", "y"))

PAIR_TO_EPIC = {
    "NZD/USD": "CS.D.NZDUSD.MINI.IP",
    "EUR/USD": "CS.D.EURUSD.MINI.IP",
    "GBP/USD": "CS.D.GBPUSD.MINI.IP",
    "USD/JPY": "CS.D.USDJPY.MINI.IP",   # make consistent (MINI.IP)
    "AUD/USD": "CS.D.AUDUSD.MINI.IP",
    "EUR/JPY": "CS.D.EURJPY.MINI.IP",
    "USD/CHF": "CS.D.USDCHF.MINI.IP",
    "USD/CAD": "CS.D.USDCAD.MINI.IP",
    "GBP/CHF": "CS.D.GBPCHF.MINI.IP",
}

PAIR_CCY = {
    "NZD/USD": "USD",
    "EUR/USD": "USD",
    "GBP/USD": "USD",
    "AUD/USD": "USD",
    "USD/JPY": "JPY",
    "EUR/JPY": "JPY",
    "USD/CHF": "CHF",
    "GBP/CHF": "CHF",
    "USD/CAD": "CAD",
}


# =========================
# FORESIGNAL DECODER
# =========================
MAP: str = ""
MAP_BASE: int = 68  # fallback base
F_MAP_RE = re.compile(
    r"""function\s+f\s*\(\s*s\s*\)\s*\{.*?w\(\s*'([^']+)'\.charAt\(\s*s\.charCodeAt\(\s*i\s*\)\s*-\s*(\d+)\s*-\s*i\s*\)\s*\)\s*\).*?\}""",
    re.DOTALL,
)

NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
F_ENC_RE = re.compile(r"f\(\s*'([^']+)'\s*\)")
HHMM_RE = re.compile(r"hhmm\((\d+)\)")


def init_decoder_from_html(html: str) -> None:
    global MAP, MAP_BASE
    m = F_MAP_RE.search(html)
    if not m:
        print("⚠️ Could not extract decoder MAP/BASE from HTML. Using fallback.")
        if not MAP:
            MAP = " 7032,-5.4981+6"
        return
    MAP = m.group(1)
    MAP_BASE = int(m.group(2))


def decode_f(encoded: str) -> str:
    out = []
    for i, ch in enumerate(encoded):
        idx = ord(ch) - MAP_BASE - i
        if 0 <= idx < len(MAP):
            out.append(MAP[idx])
    return "".join(out).strip()


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text


# =========================
# TELEGRAM + BLOGGER
# =========================
def send_telegram_html(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram not configured.")
        return False

    try:
        # Telegram limit ~4096 chars
        chunks = []
        while len(text) > 3900:
            cut = text.rfind("\n\n", 0, 3900)
            if cut == -1:
                cut = 3900
            chunks.append(text[:cut].strip())
            text = text[cut:].strip()
        if text:
            chunks.append(text)

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        for chunk in chunks:
            r = requests.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": chunk,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=30,
            )
            r.raise_for_status()
        return True
    except Exception as e:
        print(f"Telegram send failed: {e}")
        return False


def post_to_blogger(subject: str, html_body: str) -> bool:
    to_addr = os.getenv("BLOGGER_POST_EMAIL")
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")

    if not to_addr or not smtp_user or not smtp_pass:
        print("Blogger SMTP not configured.")
        return False

    try:
        msg = EmailMessage()
        msg["From"] = smtp_user
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.set_content("This post requires an HTML-capable email client.")
        msg.add_alternative(html_body, subtype="html")

        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as s:
            s.ehlo()
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.send_message(msg)

        return True
    except Exception as e:
        print(f"Blogger post failed: {e}")
        return False


# =========================
# STATE
# =========================
def load_previous() -> Optional[list[dict]]:
    if not LAST_STATE_FILE.exists():
        return None
    return json.loads(LAST_STATE_FILE.read_text(encoding="utf-8"))


def save_current(signals: list["Signal"]) -> None:
    LAST_STATE_FILE.write_text(
        json.dumps([s.to_dict() for s in signals], indent=2),
        encoding="utf-8",
    )


def load_ordered_keys() -> set[str]:
    if not ORDERED_FILE.exists():
        return set()
    try:
        obj = json.loads(ORDERED_FILE.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return set(str(x) for x in obj)
    except Exception:
        pass
    return set()


def save_ordered_keys(keys: set[str]) -> None:
    ORDERED_FILE.write_text(json.dumps(sorted(keys), indent=2), encoding="utf-8")


# =========================
# MODEL
# =========================
@dataclass
class Signal:
    pair: str
    status: str
    from_ts: int | None
    till_ts: int | None

    sell_at: str
    buy_at: str
    bought_at: str
    sold_at: str
    take_profit_at: str
    stop_loss_at: str

    pips: int | None

    def key(self) -> str:
        return f"{self.pair}|{self.from_ts or 0}|{self.till_ts or 0}"

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "status": self.status,
            "from_ts": self.from_ts,
            "till_ts": self.till_ts,
            "sell_at": self.sell_at,
            "buy_at": self.buy_at,
            "bought_at": self.bought_at,
            "sold_at": self.sold_at,
            "take_profit_at": self.take_profit_at,
            "stop_loss_at": self.stop_loss_at,
            "pips": self.pips,
            "key": self.key(),
        }


# =========================
# FORESIGNAL PARSER
# =========================
def extract_value(value_el) -> str:
    script = value_el.find("script")
    if script:
        m = F_ENC_RE.search(script.get_text(strip=True))
        if m:
            return decode_f(m.group(1))

    txt = value_el.get_text(" ", strip=True)
    txt = re.sub(r"\s+", " ", txt).strip()
    m = NUM_RE.search(txt)
    return m.group(0) if m else ""


def parse_time_range(card) -> tuple[int | None, int | None]:
    from_ts = None
    till_ts = None

    for r in card.select(".signal-row"):
        title_el = r.select_one(".signal-title")
        if not title_el:
            continue
        title = title_el.get_text(" ", strip=True)

        if title in ("From", "Till"):
            for s in r.find_all("script"):
                m = HHMM_RE.search(s.get_text(" ", strip=True))
                if m:
                    ts = int(m.group(1))
                    if title == "From":
                        from_ts = ts
                    else:
                        till_ts = ts
    return from_ts, till_ts


def parse_pips(card) -> int | None:
    for r in card.select(".signal-row"):
        title_el = r.select_one(".signal-title")
        value_el = r.select_one(".signal-value")
        if not title_el or not value_el:
            continue
        title = title_el.get_text(" ", strip=True)
        if title in ("Profit, pips", "Loss, pips"):
            v = extract_value(value_el)
            m = re.search(r"[-+]?\d+", v)
            return int(m.group(0)) if m else None
    return None


def parse_signals(html: str) -> list[Signal]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[Signal] = []

    for card in soup.select(".card.signal-card"):
        pair_el = card.select_one(".card-header a[href*='/signals/']")
        if not pair_el:
            continue
        pair = pair_el.get_text(strip=True)

        status_el = card.select_one(".signal-row.signal-status")
        status = status_el.get_text(strip=True) if status_el else ""

        from_ts, till_ts = parse_time_range(card)
        pips = parse_pips(card)

        fields = {
            "sell_at": "",
            "buy_at": "",
            "bought_at": "",
            "sold_at": "",
            "take_profit_at": "",
            "stop_loss_at": "",
        }

        for r in card.select(".signal-row"):
            title_el = r.select_one(".signal-title")
            value_el = r.select_one(".signal-value")
            if not title_el or not value_el:
                continue

            title = title_el.get_text(" ", strip=True)
            value = extract_value(value_el)

            if title == "Sell at":
                fields["sell_at"] = value
            elif title == "Buy at":
                fields["buy_at"] = value
            elif title == "Bought at":
                fields["bought_at"] = value
            elif title == "Sold at":
                fields["sold_at"] = value
            elif title.startswith("Take profit"):
                fields["take_profit_at"] = value
            elif title == "Stop loss at":
                fields["stop_loss_at"] = value

        out.append(
            Signal(
                pair=pair,
                status=status,
                from_ts=from_ts,
                till_ts=till_ts,
                sell_at=fields["sell_at"],
                buy_at=fields["buy_at"],
                bought_at=fields["bought_at"],
                sold_at=fields["sold_at"],
                take_profit_at=fields["take_profit_at"],
                stop_loss_at=fields["stop_loss_at"],
                pips=pips,
            )
        )

    return sorted(out, key=lambda s: (s.pair, s.from_ts or 0, s.till_ts or 0))


# =========================
# REPORT HELPERS
# =========================
def now_unix() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def fmt_time(ts: int | None) -> str:
    if not ts:
        return "-"
    dt = datetime.fromtimestamp(ts, TZ)
    return dt.strftime("%Y-%m-%d %H:%M")


def status_badge(status: str | None, pips: int | None) -> str:
    s = (status or "").strip().lower()
    if s == "filled":
        return "🔒"
    if s == "cancelled":
        return "🚫"
    if s == "expired":
        return "⏱️"

    if pips is None:
        return "🟡"
    if pips > 0:
        return "🟢"
    if pips < 0:
        return "🔴"
    return "🟡"


def build_full_snapshot(signals: list[Signal], pulled_at: str, prefix: str = "") -> str:
    lines = [
        "<b>🔔 Foresignal Update</b>",
        f"<i>🕒 {pulled_at} (UTC+8)</i>",
    ]
    if prefix:
        lines += ["", f"<b>{prefix}</b>"]
    lines.append("")

    for s in signals:
        em = status_badge(s.status, s.pips)
        lines.append(f"<b>{s.pair}</b> {em} <i>{s.status}</i>")
        lines.append(f"From: <code>{fmt_time(s.from_ts)}</code>")
        lines.append(f"Till: <code>{fmt_time(s.till_ts)}</code>")

        if s.sell_at:
            lines.append(f"Sell at: <code>{s.sell_at}</code>")
        if s.buy_at:
            lines.append(f"Buy at: <code>{s.buy_at}</code>")
        if s.bought_at:
            lines.append(f"Bought at: <code>{s.bought_at}</code>")
        if s.sold_at:
            lines.append(f"Sold at: <code>{s.sold_at}</code>")
        if s.take_profit_at:
            lines.append(f"TP: <code>{s.take_profit_at}</code>")
        if s.stop_loss_at:
            lines.append(f"SL: <code>{s.stop_loss_at}</code>")
        if s.pips is not None:
            lines.append(f"Pips: <code>{s.pips}</code>")
        lines.append("")

    return "\n".join(lines).strip()


# =========================
# OPEN / FILLED DETECTION
# =========================
def is_open_and_unfilled(signal: Signal, now_ts: int) -> bool:
    if not signal.from_ts or not signal.till_ts:
        return False
    if not (signal.from_ts <= now_ts <= signal.till_ts):
        return False

    status = (signal.status or "").lower()
    if status in ("filled", "cancelled", "expired"):
        return False

    # if the page already shows executed prices, treat as not-open
    if signal.bought_at or signal.sold_at:
        return False

    return True


def get_new_open_signals(prev: list[dict] | None, cur: list[Signal]) -> list[Signal]:
    now_ts = now_unix()

    prev_keys: set[str] = set()
    if prev:
        prev_keys = {p.get("key", "") for p in prev if isinstance(p, dict) and p.get("key")}

    new_open: list[Signal] = []
    for s in cur:
        if s.key() in prev_keys:
            continue
        if is_open_and_unfilled(s, now_ts):
            new_open.append(s)

    return new_open


def index_by_key(items: list[dict]) -> dict[str, dict]:
    return {it["key"]: it for it in items if isinstance(it, dict) and "key" in it}


def get_newly_filled(prev: list[dict] | None, cur: list[Signal]) -> list[Signal]:
    if not prev:
        return []
    prev_map = index_by_key(prev)
    newly: list[Signal] = []

    for s in cur:
        old = prev_map.get(s.key())
        if not old:
            continue
        old_status = (old.get("status") or "").lower()
        new_status = (s.status or "").lower()
        if old_status != "filled" and new_status == "filled":
            newly.append(s)

    return newly


# =========================
# IG (LOGIN + TRADING)
# =========================
def ig_login() -> dict:
    api_key = require_env("IG_API_KEY")
    username = require_env("IG_USERNAME")
    password = require_env("IG_PASSWORD")

    base = "https://demo-api.ig.com" if IG_DEMO else "https://api.ig.com"
    url = f"{base}/gateway/deal/session"

    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "Accept": "application/json; charset=UTF-8",
        "X-IG-API-KEY": api_key,
        "Version": "2",
    }
    payload = {"identifier": username, "password": password}

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if not r.ok:
        raise RuntimeError(f"IG login failed {r.status_code}: {r.text}")

    body = r.json()
    return {
        "api_key": api_key,
        "cst": r.headers.get("CST"),
        "xst": r.headers.get("X-SECURITY-TOKEN"),
        "account_id": body.get("currentAccountId") or body.get("accountId"),
        "base": f"{base}/gateway/deal",
    }


def _ig_headers(auth: dict, version: str = "2") -> dict:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Version": version,
        "X-IG-API-KEY": auth["api_key"],
        "CST": auth["cst"],
        "X-SECURITY-TOKEN": auth["xst"],
    }


def ig_open_market(auth: dict, epic: str, direction: str, size: float, pair: str) -> str:
    direction = direction.upper().strip()
    if direction not in ("BUY", "SELL"):
        raise ValueError("direction must be BUY/SELL")
    if size <= 0:
        raise ValueError("size must be > 0")

    currency_code = PAIR_CCY.get(pair, "USD")

    url = f"{auth['base'].rstrip('/')}/positions/otc"
    payload = {
        "epic": epic,
        "expiry": "-",
        "direction": direction,
        "orderType": "MARKET",
        "size": float(size),
        "forceOpen": True,
        "currencyCode": currency_code,
        "guaranteedStop": False,
    }

    r = requests.post(url, headers=_ig_headers(auth, "2"), json=payload, timeout=20)
    r.raise_for_status()
    data = r.json()
    if "dealReference" not in data:
        raise RuntimeError(f"Unexpected IG response: {data}")
    return data["dealReference"]


def ig_confirm(auth: dict, deal_ref: str, tries: int = 6) -> dict:
    url = f"{auth['base'].rstrip('/')}/confirms/{deal_ref}"
    headers = _ig_headers(auth, "1")

    delay = 0.4
    last_err: str | None = None

    for _ in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code >= 500:
                last_err = f"{r.status_code} {r.text}"
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = str(e)
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"Confirm failed after retries: {last_err}")


def ig_attach_sl_then_tp(
    auth: dict,
    deal_id: str,
    sl: float,
    tp: float,
    guaranteed_stop: bool = False,
    trailing_stop: bool = False,
    timeout: int = 20,
) -> dict:
    base = auth["base"].rstrip("/")
    url = f"{base}/positions/otc/{deal_id}"
    headers = _ig_headers(auth, "2")

    results: dict[str, Any] = {"deal_id": deal_id}

    # STOP first
    sl_payload = {
        "stopLevel": float(sl),
        "guaranteedStop": bool(guaranteed_stop),
        "trailingStop": bool(trailing_stop),
    }
    r1 = requests.put(url, headers=headers, json=sl_payload, timeout=timeout)
    r1.raise_for_status()
    results["stop_loss_response"] = r1.json()

    # TP second
    tp_payload = {
        "limitLevel": float(tp),
        "stopLevel": float(sl),
        "guaranteedStop": bool(guaranteed_stop),
        "trailingStop": bool(trailing_stop),
    }
    r2 = requests.put(url, headers=headers, json=tp_payload, timeout=timeout)
    r2.raise_for_status()
    results["take_profit_response"] = r2.json()

    return results


def ig_get_positions(auth: dict) -> list[dict]:
    url = f"{auth['base'].rstrip('/')}/positions"
    r = requests.get(url, headers=_ig_headers(auth, "2"), timeout=20)
    r.raise_for_status()
    return r.json().get("positions", [])


def ig_get_working_orders(auth: dict) -> list[dict]:
    url = f"{auth['base'].rstrip('/')}/workingorders"
    r = requests.get(url, headers=_ig_headers(auth, "2"), timeout=20)
    r.raise_for_status()
    return r.json().get("workingOrders", [])


def ig_delete_working_orders_for_epic(auth: dict, epic: str) -> None:
    orders = ig_get_working_orders(auth)
    headers = _ig_headers(auth, "2")
    base = auth["base"].rstrip("/")

    for o in orders:
        wo = o.get("workingOrder", {}) if isinstance(o, dict) else {}
        if wo.get("epic") != epic:
            continue
        deal_id = wo.get("dealId")
        if not deal_id:
            continue
        url = f"{base}/workingorders/otc/{deal_id}"
        rd = requests.delete(url, headers=headers, timeout=20)
        print(f"🗑️ Deleting working order epic={epic} dealId={deal_id} -> {rd.status_code}")
        rd.raise_for_status()


def ig_close_position(
    auth: dict,
    *,
    deal_id: str,
    epic: str,
    open_direction: str,
    size: float,
    currency_code: str,
) -> None:
    open_direction = (open_direction or "").upper()
    close_direction = "SELL" if open_direction == "BUY" else "BUY"

    payload = {
        "dealId": deal_id,
        "epic": epic,
        "direction": close_direction,
        "orderType": "MARKET",
        "size": float(size),
        "expiry": "-",
        "forceOpen": False,
        "guaranteedStop": False,
        "currencyCode": currency_code,
    }

    url = f"{auth['base'].rstrip('/')}/positions/otc"
    r = requests.post(url, headers=_ig_headers(auth, "2"), json=payload, timeout=20)
    print("CLOSE RESPONSE:", r.status_code, r.text)
    r.raise_for_status()


def ig_close_all_positions_for_epic(auth: dict, epic: str, pair: str) -> None:
    currency_code = PAIR_CCY.get(pair, "USD")
    positions = ig_get_positions(auth)

    for p in positions:
        if not isinstance(p, dict):
            continue
        mkt = p.get("market", {}) or {}
        pos = p.get("position", {}) or {}
        if mkt.get("epic") != epic:
            continue

        deal_id = pos.get("dealId")
        direction = pos.get("direction")
        size = pos.get("size")

        if not deal_id or not direction or not size:
            continue

        print(f"♻️ Closing OPEN position epic={epic} dealId={deal_id}")
        ig_close_position(
            auth,
            deal_id=str(deal_id),
            epic=epic,
            open_direction=str(direction),
            size=float(size),
            currency_code=currency_code,
        )


# =========================
# MAIN
# =========================
def main() -> None:
    html = fetch_html(URL)
    init_decoder_from_html(html)
    signals = parse_signals(html)

    prev = load_previous()
    ordered_keys = load_ordered_keys()

    # 1) Close on newly FILLED (Foresignal -> IG)
    newly_filled = get_newly_filled(prev, signals)
    if newly_filled:
        try:
            ig_auth = ig_login()
        except Exception as e:
            print(f"IG login failed (close step): {e}")
            ig_auth = None

        if ig_auth:
            for s in newly_filled:
                epic = PAIR_TO_EPIC.get(s.pair)
                if not epic:
                    continue
                ig_delete_working_orders_for_epic(ig_auth, epic)
                ig_close_all_positions_for_epic(ig_auth, epic, s.pair)

            if newly_filled:
                print(f"✅ Closed IG positions for {len(newly_filled)} FILLED signals:")
                for i, s in enumerate(newly_filled, start=1):
                    epic = PAIR_TO_EPIC.get(s.pair, "-")
                    print(
                        f"  {i}. {s.pair} | epic={epic} | "
                        f"From={fmt_time(s.from_ts)} | Till={fmt_time(s.till_ts)} | "
                        f"status={s.status} | pips={s.pips}"
                    )
            else:
                print("✅ Closed IG positions for 0 FILLED signals")
    # ✅ ALWAYS compute new_open_signals (fixes UnboundLocalError)
    new_open_signals = get_new_open_signals(prev, signals)

    if not new_open_signals:
        print("No new open/unfilled signals.")
        save_current(signals)
        return

    pulled_at = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")

    report = build_full_snapshot(
        new_open_signals,
        pulled_at,
        prefix="🆕 New OPEN Signals (Unfilled)",
    )

    # 2) Notify
    sent_tg = send_telegram_html(report)
    subject = f"Foresignal NEW OPEN - {pulled_at} (UTC+8)"
    sent_blog = post_to_blogger(subject=subject, html_body=report)

    print(f"Telegram sent? {sent_tg}")
    print(f"Blogger posted? {sent_blog}")

    # 3) Place IG orders if notified
    if sent_tg or sent_blog:
        try:
            ig_auth = ig_login()
        except Exception as e:
            print(f"IG login failed (trade step): {e}")
            ig_auth = None

        if ig_auth:
            for s in new_open_signals:
                k = s.key()
                if k in ordered_keys:
                    continue

                epic = PAIR_TO_EPIC.get(s.pair)
                if not epic:
                    print(f"⚠️ No IG EPIC mapping for {s.pair}; skipping.")
                    continue

                # clean slate
                try:
                    ig_close_all_positions_for_epic(ig_auth, epic, s.pair)
                    ig_delete_working_orders_for_epic(ig_auth, epic)
                except Exception as e:
                    print(f"⚠️ Cleanup failed for {s.pair}: {e}")

                # direction
                if s.buy_at:
                    direction = "BUY"
                elif s.sell_at:
                    direction = "SELL"
                else:
                    print(f"⚠️ Missing buy_at/sell_at for {s.pair}; skipping.")
                    continue

                # require TP/SL
                if not s.take_profit_at or not s.stop_loss_at:
                    print(f"⚠️ Missing TP/SL for {s.pair}; skipping.")
                    continue
                if(s.pair == "EUR/USD"){
                    tp = float(mini_to_price(s.take_profit_at))
                    sl = float(mini_to_price(s.stop_loss_at))
                }else{
                    tp = float(s.take_profit_at)
                    sl = float(s.stop_loss_at)
                }
                

                try:
                    deal_ref = ig_open_market(ig_auth, epic, direction, 0.5, s.pair)
                    conf = ig_confirm(ig_auth, deal_ref)

                    if conf.get("dealStatus") != "ACCEPTED":
                        raise RuntimeError(f"Deal rejected: {conf.get('reason')} | {conf}")

                    deal_id = conf.get("dealId") or (conf.get("affectedDeals") or [{}])[0].get("dealId")
                    if not deal_id:
                        raise RuntimeError(f"No dealId in confirm: {conf}")

                    edit_resp = ig_attach_sl_then_tp(ig_auth, str(deal_id), sl, tp)
                    print("TP/SL attached:", edit_resp)

                    ordered_keys.add(k)

                except Exception as e:
                    print(f"❌ IG order failed for {s.pair}: {e}")

            save_ordered_keys(ordered_keys)
    else:
        print("No notification sent; IG order NOT placed.")

    # 4) Save snapshot
    save_current(signals)


if __name__ == "__main__":
    main()
