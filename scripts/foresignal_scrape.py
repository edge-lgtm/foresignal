from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup
from email.message import EmailMessage
import smtplib


IG_API_KEY = os.getenv("IG_API_KEY")
IG_USERNAME = os.getenv("IG_USERNAME")
IG_PASSWORD = os.getenv("IG_PASSWORD")
IG_ACCOUNT_ID = os.getenv("IG_ACCOUNT_ID")
IG_DEMO = (os.getenv("IG_DEMO", "true").lower() in ("1", "true", "yes", "y"))
def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

# ---------------- CONFIG ----------------
URL = "https://foresignal.com/en/"
TZ = ZoneInfo("Asia/Manila")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

LAST_STATE_FILE = DATA_DIR / "latest_signals.json"          # last snapshot (for diff)
TRADES_LOG_FILE = DATA_DIR / "trades_history.jsonl"         # append-only trade outcomes

# Foresignal obfuscation map (from site JS)
MAP: str = ""
MAP_BASE: int = 68  # default fallback
F_MAP_RE = re.compile(
    r"""function\s+f\s*\(\s*s\s*\)\s*\{.*?w\(\s*'([^']+)'\.charAt\(\s*s\.charCodeAt\(\s*i\s*\)\s*-\s*(\d+)\s*-\s*i\s*\)\s*\)\s*\).*?\}""",
    re.DOTALL
)

def post_to_blogger(subject: str, html_body: str) -> bool:
    to_addr = os.getenv("BLOGGER_POST_EMAIL")
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")

    if not to_addr or not smtp_user or not smtp_pass:
        print("Blogger SMTP not confured.")
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

def send_telegram_html(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram not configured.")
        return False

    try:
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



ORDERED_FILE = DATA_DIR / "ordered_keys.json"

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

IG_BASE = "https://demo-api.ig.com/gateway/deal/session"
headers = {
    "Content-Type": "application/json; charset=UTF-8",
    "Accept": "application/json; charset=UTF-8",
    "X-IG-API-KEY": IG_API_KEY,
    "Version": "3"
}
PAIR_TO_EPIC = {
    "EUR/USD": "CS.D.EURUSD.MINI.IP",
    "GBP/USD": "CS.D.GBPUSD.MINI.IP",
    "USD/JPY": "CS.D.USDJPY.MINI.IP",
    "AUD/USD": "CS.D.AUDUSD.MINI.IP",
    "EUR/JPY": "CS.D.EURJPY.MINI.IP",
    "USD/CHF": "CS.D.USDCHF.MINI.IP",
    "USD/CAD": "CS.D.USDCAD.MINI.IP",
    "GBP/CHF": "CS.D.GBPCHF.MINI.IP",
}
IG_EPIC_MAP = {
    "EUR/USD": "CS.D.EURUSD.MINI.IP",
    "GBP/USD": "CS.D.GBPUSD.MINI.IP",
    "USD/JPY": "CS.D.USDJPY.MINI.IP",
    "AUD/USD": "CS.D.AUDUSD.MINI.IP",
    "EUR/JPY": "CS.D.EURJPY.MINI.IP",
    "USD/CHF": "CS.D.USDCHF.MINI.IP",
    "USD/CAD": "CS.D.USDCAD.MINI.IP",
    "GBP/CHF": "CS.D.GBPCHF.MINI.IP",
}
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
    payload = {
        "identifier": "edwardlancelorilla",
        "password": "eDwArD!@#1"
    }

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    print("=== IG LOGIN RESPONSE ===")
    print("URL:", r.url)
    print("Status code:", r.status_code)
    print("Reason:", r.reason)
    
    print("\n--- HEADERS ---")
    for k, v in r.headers.items():
        print(f"{k}: {v}")
    
    print("\n--- BODY (raw text) ---")
    print(r.text)
    
    print("\n--- BODY (json parsed, if possible) ---")
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception as e:
        print("Not JSON:", e)
    
    print("=== END RESPONSE ===")
    if not r.ok:
        raise RuntimeError(f"IG login failed {r.status_code}: {r.text}")

    return {
        "api_key": api_key,                         # ✅ ADD THIS
        "cst": r.headers.get("CST"),
        "xst": r.headers.get("X-SECURITY-TOKEN"),
        "account_id": r.get("accountId"),
        "base": f"{base}/gateway/deal", 
    }
def ig_login_demo() -> dict:
    url = "https://demo-api.ig.com/gateway/deal/session"

    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "Accept": "application/json; charset=UTF-8",
        "X-IG-API-KEY": os.getenv("IG_API_KEY"),
        "Version": "2"
    }

    payload = {
        "identifier": os.getenv("IG_USERNAME"),
        "password": os.getenv("IG_PASSWORD")
    }

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    print("IG RESPONSE:", r.status_code, r.text)
    if not r.ok:
        raise RuntimeError(
            f"IG login failed {r.status_code}: {r.text}"
        )

    return {
        "cst": r.headers.get("CST"),
        "xst": r.headers.get("X-SECURITY-TOKEN"),
        "account_id": r.json().get("currentAccountId")
    }
IG_HEADERS_BASE = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Version": "2",
    "X-IG-API-KEY": IG_API_KEY,
    "clientPlatform": "WEB",
}
def ig_place_limit(
    auth: dict,
    *,
    epic: str,
    direction: str,
    entry: float,
    tp: float,
    sl: float,
    size: float = 0.5,
) -> dict:
    headers = IG_HEADERS_BASE.copy()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Version": "2",
        "X-IG-API-KEY": auth["api_key"],   # ✅ NOW PRESENT
        "CST": auth["cst"],
        "X-SECURITY-TOKEN": auth["xst"],
    }


    payload = {
        "epic": epic,
        "expiry": "-",
        "direction": direction,          # "BUY" or "SELL"
        "orderType": "LIMIT",
        "size": size,
        "level": entry,                  # entry price
        "limitLevel": tp,                # take profit
        "stopLevel": sl,                 # stop loss
        "timeInForce": "GOOD_TILL_CANCELLED",
        "forceOpen": True,
        "guaranteedStop": False,
        "currencyCode": "USD",
    }
    url = f"{auth['base']}/positions/otc"
    print("IG ORDER URL:", url)
    print("IG ORDER HEADERS:", headers)
    print("IG ORDER PAYLOAD:", payload)

    r = requests.post(url, headers=headers, json=payload, timeout=20)
    print("IG ORDER RESPONSE:", r.status_code, r.text)
    return r.json()

def init_decoder_from_html(html: str) -> None:
    """
    Pulls the live obfuscation MAP and base from the site JS:
      w('<MAP>'.charAt(s.charCodeAt(i)-<BASE>-i))
    """
    global MAP, MAP_BASE

    m = F_MAP_RE.search(html)
    if not m:
        # Fallback: keep previous values; decoding may still work via plaintext tail "1.1861"
        print("⚠️ Could not extract decoder MAP/BASE from HTML. Using fallback.")
        if not MAP:
            MAP = " 7032,-5.4981+6"  # last known example fallback
        return

    MAP = m.group(1)
    MAP_BASE = int(m.group(2))
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
F_ENC_RE = re.compile(r"f\(\s*'([^']+)'\s*\)")
HHMM_RE = re.compile(r"hhmm\((\d+)\)")  # unix seconds inside hhmm(....)
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

        msg.set_content("HTML required")
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
# ---------------- DATA MODEL ----------------
def is_open_and_unfilled(signal: Signal, now_ts: int) -> bool:
    if not signal.from_ts or not signal.till_ts:
        return False

    if not (signal.from_ts <= now_ts <= signal.till_ts):
        return False

    status = (signal.status or "").lower()
    if status in ("filled", "cancelled", "expired"):
        return False

    if signal.bought_at or signal.sold_at:
        return False

    return True
def get_new_open_signals(
    prev: list[dict] | None,
    cur: list[Signal]
) -> list[Signal]:
    now_ts = now_unix()

    prev_keys = set()
    if prev:
        prev_keys = {p["key"] for p in prev if "key" in p}

    new_open = []
    for s in cur:
        if s.key() in prev_keys:
            continue

        if is_open_and_unfilled(s, now_ts):
            new_open.append(s)

    return new_open
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

    pips: int | None  # profit/loss pips if present

    def key(self) -> str:
        # unique-ish key for tracking across runs
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


# ---------------- HELPERS ----------------
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


def decode_f(encoded: str) -> str:
    out = []
    for i, ch in enumerate(encoded):
        idx = ord(ch) - MAP_BASE - i
        if 0 <= idx < len(MAP):
            out.append(MAP[idx])
    return "".join(out).strip()


def extract_value(value_el) -> str:
    # Prefer decoding <script>f('...')</script>
    script = value_el.find("script")
    if script:
        m = F_ENC_RE.search(script.get_text(strip=True))
        if m:
            return decode_f(m.group(1))

    # Fallback: plain numeric text
    txt = value_el.get_text(" ", strip=True)
    txt = re.sub(r"\s+", " ", txt).strip()
    m = NUM_RE.search(txt)
    return m.group(0) if m else ""


def parse_time_range(card) -> tuple[int | None, int | None]:
    """
    Extract From/Till unix seconds from:
      <script>w(hhmm(1769774700));</script>
    This is reliable in raw HTML and avoids JS execution.
    """
    # Find the two signal rows that contain "From" and "Till"
    from_ts = None
    till_ts = None

    rows = card.select(".signal-row")
    for r in rows:
        title_el = r.select_one(".signal-title")
        if not title_el:
            continue
        title = title_el.get_text(" ", strip=True)

        if title in ("From", "Till"):
            scripts = r.find_all("script")
            for s in scripts:
                m = HHMM_RE.search(s.get_text(" ", strip=True))
                if m:
                    ts = int(m.group(1))
                    if title == "From":
                        from_ts = ts
                    else:
                        till_ts = ts

    return from_ts, till_ts


def parse_pips(card) -> int | None:
    """
    Looks for:
      Profit, pips  +38
      Loss, pips    -30
    """
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
    signals: list[Signal] = []

    for card in soup.select(".card.signal-card"):
        pair_el = card.select_one(".card-header a[href*='/signals/']")
        if not pair_el:
            continue
        pair = pair_el.get_text(strip=True)

        status_el = card.select_one(".signal-row.signal-status")
        status = status_el.get_text(strip=True) if status_el else ""

        from_ts, till_ts = parse_time_range(card)
        pips = parse_pips(card)

        # defaults
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

        signals.append(
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

    # stable ordering
    return sorted(signals, key=lambda s: (s.pair, s.from_ts or 0, s.till_ts or 0))


def load_previous() -> list[dict] | None:
    if not LAST_STATE_FILE.exists():
        return None
    return json.loads(LAST_STATE_FILE.read_text(encoding="utf-8"))


def save_current(signals: list[Signal]) -> None:
    LAST_STATE_FILE.write_text(
        json.dumps([s.to_dict() for s in signals], indent=2),
        encoding="utf-8"
    )


def now_unix() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def fmt_time(ts: int | None) -> str:
    if not ts:
        return "-"
    dt = datetime.fromtimestamp(ts, TZ)
    return dt.strftime("%Y-%m-%d %H:%M")


def pl_emoji(pips: int | None) -> str:
    if pips is None:
        return "🟡"
    if pips > 0:
        return "🟢"
    if pips < 0:
        return "🔴"
    return "🟡"


# ---------------- TRADE HISTORY / WIN RATE ----------------
def append_trade_outcome(signal: Signal) -> None:
    """
    Only log trades when pips is known (Filled with Profit/Loss pips on page).
    Avoid duplicates by tracking keys we've already logged.
    """
    if signal.pips is None:
        return

    # Build a set of already logged keys (lightweight scan)
    logged_keys = set()
    if TRADES_LOG_FILE.exists():
        with TRADES_LOG_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    k = obj.get("key")
                    if k:
                        logged_keys.add(k)
                except Exception:
                    continue

    k = signal.key()
    if k in logged_keys:
        return

    entry = {
        "key": k,
        "pair": signal.pair,
        "from_ts": signal.from_ts,
        "till_ts": signal.till_ts,
        "status": signal.status,
        "pips": signal.pips,
        "logged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with TRADES_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def compute_win_rate() -> tuple[str, dict]:
    """
    Returns:
      overall summary string
      per_pair dict: {pair: {"wins":x,"losses":y,"win_rate":z}}
    """
    wins = losses = 0
    per_pair: dict[str, dict] = {}

    if not TRADES_LOG_FILE.exists():
        return "No closed trades yet.", per_pair

    def consume_trade(obj: dict) -> None:
        nonlocal wins, losses, per_pair
        pair = obj.get("pair")
        pips = obj.get("pips")
        if pair is None or pips is None:
            return

        per_pair.setdefault(pair, {"wins": 0, "losses": 0})
        if pips > 0:
            wins += 1
            per_pair[pair]["wins"] += 1
        elif pips < 0:
            losses += 1
            per_pair[pair]["losses"] += 1

    with TRADES_LOG_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # ✅ handle both formats:
            if isinstance(obj, dict):
                consume_trade(obj)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        consume_trade(item)

    total = wins + losses
    overall = (
        f"Wins: {wins} | Losses: {losses} | Win rate: {((wins/total)*100):.1f}%"
        if total else
        "No wins/losses yet."
    )

    for pair, d in per_pair.items():
        t = d["wins"] + d["losses"]
        d["win_rate"] = (d["wins"] / t) * 100 if t else 0.0

    return overall, per_pair


# ---------------- DIFF + ALERTS ----------------
TRACK_FIELDS = [
    "status",
    "sell_at",
    "buy_at",
    "bought_at",
    "sold_at",
    "take_profit_at",
    "stop_loss_at",
    "from_ts",
    "till_ts",
    "pips",
]

def index_by_key(items: list[dict]) -> dict[str, dict]:
    return {it["key"]: it for it in items if "key" in it}


def build_change_report(prev: list[dict] | None, cur: list[Signal]) -> tuple[bool, str, bool]:
    """
    Returns:
      (changed?, telegram_html_text, removed_only?)
    """
    pulled_at = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")

    cur_list = [s.to_dict() for s in cur]
    if prev is None:
        text = build_full_snapshot(cur, pulled_at, prefix="🆕 First snapshot")
        return True, text, False

    prev_map = index_by_key(prev)
    cur_map = index_by_key(cur_list)

    changed_pairs_blocks: list[str] = []
    expired_blocks: list[str] = []

    now_ts = now_unix()

    removed_count = 0
    new_count = 0
    changed_count = 0

    # 1) Changes + New
    for k, new in cur_map.items():
        old = prev_map.get(k)
        if old is None:
            new_count += 1
            changed_pairs_blocks.append(format_new_signal(new))
            continue

        diffs = []
        for f in TRACK_FIELDS:
            if old.get(f) != new.get(f):
                diffs.append((f, old.get(f), new.get(f)))

        if diffs:
            changed_count += 1
            changed_pairs_blocks.append(format_changed_signal(new, diffs))

    # 2) Removed
    for k, old in prev_map.items():
        if k not in cur_map:
            till_ts = old.get("till_ts")
            if isinstance(till_ts, int) and till_ts <= now_ts:
                expired_blocks.append(format_expired(old))
            else:
                removed_count += 1
                changed_pairs_blocks.append(format_removed(old))

    # 3) Expired (still exists but past till)
    for k, new in cur_map.items():
        till_ts = new.get("till_ts")
        status = (new.get("status") or "").lower()
        if isinstance(till_ts, int) and till_ts <= now_ts:
            if status not in ("filled", "cancelled"):
                expired_blocks.append(format_expired(new))

    if not changed_pairs_blocks and not expired_blocks:
        return False, "", False

    overall_wr, _ = compute_win_rate()

    header = [
        "<b>🔔 Foresignal Update</b>",
        f"<i>🕒 {pulled_at} (UTC+8)</i>",
        "",
        f"<b>📉 Win rate:</b> {overall_wr}",
        ""
    ]

    blocks = header
    if changed_pairs_blocks:
        blocks.append("<b>Changes</b>")
        blocks.extend(changed_pairs_blocks)
        blocks.append("")
    if expired_blocks:
        blocks.append("<b>⏱️ Expired</b>")
        blocks.extend(expired_blocks)
        blocks.append("")

    msg = "\n".join(blocks).strip()

    removed_only = (removed_count > 0) and (new_count == 0) and (changed_count == 0) and (len(expired_blocks) == 0)
    return True, msg, removed_only


def format_field_name(f: str) -> str:
    return {
        "from_ts": "From",
        "till_ts": "Till",
        "take_profit_at": "TP",
        "stop_loss_at": "SL",
        "sell_at": "Sell at",
        "buy_at": "Buy at",
        "bought_at": "Bought at",
        "sold_at": "Sold at",
        "status": "Status",
        "pips": "Pips",
    }.get(f, f)


def safe_str(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, int) and ("ts" in str(v)):
        return str(v)
    return str(v)


def format_changed_signal(new: dict, diffs: list[tuple[str, object, object]]) -> str:
    pair = new.get("pair", "?")
    status = new.get("status", "")
    pips = new.get("pips", None)
    em = pl_emoji(pips if isinstance(pips, int) else None)

    lines = [f"<b>{pair}</b> {em} <i>{status}</i>"]
    # only show changed fields
    for f, oldv, newv in diffs:
        if f in ("from_ts", "till_ts"):
            old_s = fmt_time(oldv) if isinstance(oldv, int) else "-"
            new_s = fmt_time(newv) if isinstance(newv, int) else "-"
            lines.append(f"{format_field_name(f)}: <code>{old_s}</code> → <code>{new_s}</code>")
        else:
            lines.append(f"{format_field_name(f)}: <code>{safe_str(oldv)}</code> → <code>{safe_str(newv)}</code>")

    return "\n".join(lines) + "\n"


def format_new_signal(new: dict) -> str:
    pair = new.get("pair", "?")
    status = new.get("status", "")
    lines = [f"<b>{pair}</b> 🆕 <i>{status}</i>"]
    lines.append(f"From: <code>{fmt_time(new.get('from_ts'))}</code>")
    lines.append(f"Till: <code>{fmt_time(new.get('till_ts'))}</code>")

    for label, key in [
        ("Sell at", "sell_at"),
        ("Buy at", "buy_at"),
        ("Bought at", "bought_at"),
        ("Sold at", "sold_at"),
        ("TP", "take_profit_at"),
        ("SL", "stop_loss_at"),
    ]:
        v = new.get(key) or ""
        if v:
            lines.append(f"{label}: <code>{v}</code>")

    p = new.get("pips")
    if isinstance(p, int):
        lines.append(f"Pips: <code>{p}</code>")

    return "\n".join(lines) + "\n"


def format_removed(old: dict) -> str:
    pair = old.get("pair", "?")
    return f"<b>{pair}</b> ⚠️ <i>Removed from page</i>\nFrom: <code>{fmt_time(old.get('from_ts'))}</code>\nTill: <code>{fmt_time(old.get('till_ts'))}</code>\n"


def format_expired(s: dict) -> str:
    pair = s.get("pair", "?")
    return f"<b>{pair}</b> ⏱️ <i>Expired (Till reached)</i>\nTill: <code>{fmt_time(s.get('till_ts'))}</code>\n"


def build_full_snapshot(signals: list[Signal], pulled_at: str, prefix: str = "") -> str:
    lines = [
        "<b>🔔 Foresignal Update</b>",
        f"<i>🕒 {pulled_at} (UTC+8)</i>",
    ]
    if prefix:
        lines += ["", f"<b>{prefix}</b>"]
    lines.append("")

    for s in signals:
        em = pl_emoji(s.pips)
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

    overall_wr, _ = compute_win_rate()
    lines.append(f"<b>📉 Win rate:</b> {overall_wr}")

    return "\n".join(lines).strip()


# ---------------- TELEGRAM SEND ----------------
def send_telegram_html(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return

    # Telegram limit ~4096 chars. If too long, split cleanly.
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


# ---------------- MAIN ----------------
def main() -> None:
    html = fetch_html(URL)
    init_decoder_from_html(html)
    signals = parse_signals(html)

    prev = load_previous()  # previous snapshot list[dict] or None
    new_open_signals = get_new_open_signals(prev, signals)

    # Always save snapshot at end so "new" becomes "known" next run
    # (but only trade if send succeeded)
    ordered_keys = load_ordered_keys()

    if not new_open_signals:
        print("No new open/unfilled signals.")
        save_current(signals)
        return

    pulled_at = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")

    # Build a report for ONLY the new open signals (includes TP & SL already)
    report = build_full_snapshot(
        new_open_signals,
        pulled_at,
        prefix="🆕 New OPEN Signals (Unfilled)"
    )

    # 1) Send notifications
    sent_tg = send_telegram_html(report)

    subject = f"Foresignal NEW OPEN - {pulled_at} (UTC+8)"
    sent_blog = post_to_blogger(subject=subject, html_body=report)

    print(f"Telegram sent? {sent_tg}")
    print(f"Blogger posted? {sent_blog}")

    # 2) If "send triggered" → place IG orders
    if sent_tg or sent_blog:
        try:
            ig_auth = ig_login()  # expects IG_API_KEY/IG_USERNAME/IG_PASSWORD in env
        except Exception as e:
            print(f"IG login failed: {e}")
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

                # Direction + entry
                if s.buy_at:
                    direction = "BUY"
                    entry = float(s.buy_at)
                elif s.sell_at:
                    direction = "SELL"
                    entry = float(s.sell_at)
                else:
                    print(f"⚠️ Missing buy_at/sell_at for {s.pair}; skipping.")
                    continue

                # TP/SL required
                if not s.take_profit_at or not s.stop_loss_at:
                    print(f"⚠️ Missing TP/SL for {s.pair}; skipping.")
                    continue

                tp = float(s.take_profit_at)
                sl = float(s.stop_loss_at)

                try:
                    resp = ig_place_limit(
                        ig_auth,
                        epic=epic,
                        direction=direction,
                        entry=entry,
                        tp=tp,
                        sl=sl,
                        size=float(os.getenv("IG_SIZE", "0.5")),
                    )
                    print(f"✅ IG order placed for {s.pair}: {resp}")
                    ordered_keys.add(k)
                except Exception as e:
                    print(f"❌ IG order failed for {s.pair}: {e}")

            save_ordered_keys(ordered_keys)
    else:
        print("No notification sent; IG order NOT placed.")

    # 3) Save current snapshot (so these are not 'new' next run)
    save_current(signals)


if __name__ == "__main__":
    main()
