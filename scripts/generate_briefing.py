#!/usr/bin/env python3
"""
Attitash Pre-Market Briefing Generator
Runs via GitHub Actions at 7:00 AM ET on weekdays.
Outputs data/briefing.json for the Market Command Centre dashboard.

Uses raw Yahoo Finance HTTP endpoints — no yfinance library (avoids
timezone bugs and hangs in CI environments).
"""

import json
import os
import sys
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError

import pytz

ET = pytz.timezone("America/New_York")
NOW_ET = datetime.now(ET)
TODAY = NOW_ET.strftime("%Y-%m-%d")
DATE_DISPLAY = NOW_ET.strftime("%a, %b %-d, %Y")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
TIMEOUT = 12

# -- Market holiday check --
US_HOLIDAYS_2026 = [
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
    "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-12-25",
]


def is_trading_day() -> bool:
    if NOW_ET.weekday() >= 5:
        return False
    if TODAY in US_HOLIDAYS_2026:
        return False
    return True


def _yahoo_get(url: str) -> dict:
    """Safe HTTP GET to Yahoo Finance with timeout."""
    try:
        req = Request(url, headers=HEADERS)
        resp = urlopen(req, timeout=TIMEOUT)
        return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  WARN: Request failed ({url[:80]}...): {e}")
        return {}


def _yahoo_quote(symbols: list) -> dict:
    """Fetch quotes for a list of symbols using Yahoo v7 quote endpoint."""
    syms = ",".join(symbols)
    url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={syms}&fields=regularMarketPrice,regularMarketChange,regularMarketChangePercent,regularMarketPreviousClose,shortName,marketCap"
    data = _yahoo_get(url)
    results = {}
    quotes = data.get("quoteResponse", {}).get("result", [])
    for q in quotes:
        sym = q.get("symbol", "")
        results[sym] = q
    return results


# -- Futures & Macro --
def fetch_futures_macro() -> dict:
    """Fetch futures and macro quotes via Yahoo HTTP API."""
    symbol_map = {
        "ES": "ES=F",
        "NQ": "NQ=F",
        "CL": "CL=F",
        "DXY": "DX-Y.NYB",
        "VIX": "^VIX",
    }
    results = {}
    try:
        quotes = _yahoo_quote(list(symbol_map.values()))
        for label, yf_sym in symbol_map.items():
            q = quotes.get(yf_sym, {})
            price = q.get("regularMarketPrice", 0) or 0
            prev = q.get("regularMarketPreviousClose", 0) or 0
            change = q.get("regularMarketChange", 0) or 0
            change_pct = q.get("regularMarketChangePercent", 0) or 0
            results[label] = {
                "price": round(price, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "prev_close": round(prev, 2),
            }
    except Exception as e:
        print(f"  WARN: Futures fetch failed: {e}")

    # Fill any missing keys with zeros
    for label in symbol_map:
        if label not in results:
            results[label] = {"price": 0, "change": 0, "change_pct": 0, "prev_close": 0}

    return results


# -- Market Gainers (Gap Scan) --
def fetch_gainers() -> list:
    """Fetch top US stock gainers. Filter: >10% gap AND >$500M mcap."""
    gappers = []
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=day_gainers&count=25"
        data = _yahoo_get(url)
        quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])

        for q in quotes:
            symbol = q.get("symbol", "")
            change_pct = q.get("regularMarketChangePercent", 0) or 0
            price = q.get("regularMarketPrice", 0) or 0
            market_cap = q.get("marketCap", 0) or 0

            if any(x in symbol for x in ["-", "."]) and len(symbol) > 5:
                continue
            if market_cap and market_cap < 500_000_000:
                continue
            if change_pct and change_pct < 10:
                continue

            gappers.append({
                "symbol": symbol,
                "gap_pct": round(change_pct, 1),
                "price": round(price, 2),
                "market_cap": int(market_cap) if market_cap else 0,
                "catalyst": "",
            })
    except Exception as e:
        print(f"  WARN: Gainers fetch failed: {e}")

    gappers.sort(key=lambda x: x["gap_pct"], reverse=True)
    return gappers[:10]


# -- Market Sentiment --
def assess_sentiment(futures: dict) -> str:
    es = futures.get("ES", {})
    nq = futures.get("NQ", {})
    vix = futures.get("VIX", {})

    es_pct = es.get("change_pct", 0)
    nq_pct = nq.get("change_pct", 0)
    vix_price = vix.get("price", 20)

    bullish = 0
    bearish = 0

    if es_pct > 0.3:
        bullish += 1
    elif es_pct < -0.3:
        bearish += 1

    if nq_pct > 0.3:
        bullish += 1
    elif nq_pct < -0.3:
        bearish += 1

    if vix_price < 15:
        bullish += 1
    elif vix_price > 25:
        bearish += 2
    elif vix_price > 20:
        bearish += 1

    if bullish > bearish:
        return "BULLISH"
    elif bearish > bullish:
        return "BEARISH"
    else:
        return "UNCERTAIN"


# -- Earnings Calendar --
def fetch_earnings() -> list:
    earnings = []
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/calendar/earnings?start={TODAY}&end={TODAY}"
        data = _yahoo_get(url)
        events = data.get("finance", {}).get("result", [{}])[0].get("events", [])

        for ev in events:
            ticker = ev.get("ticker", "")
            time_str = ev.get("startdatetimetype", "")
            if time_str in ("BMO", "TAS"):
                timing = "BMO"
                time_et = "Before Open"
            elif time_str == "AMC":
                timing = "AMC"
                time_et = "After Close"
            else:
                timing = "BMO"
                time_et = time_str

            earnings.append({
                "ticker": ticker,
                "time_et": time_et,
                "timing": timing,
            })
    except Exception as e:
        print(f"  WARN: Earnings fetch failed: {e}")

    return earnings[:20]


# -- Economic Events --
def fetch_econ_events() -> list:
    events = []
    HIGH_IMPACT = {"ISM", "NFP", "CPI", "FOMC", "GDP", "PCE", "Fed", "Nonfarm", "Consumer Price"}
    MEDIUM_IMPACT = {"PMI", "Housing", "Retail", "Durable", "PPI", "Jobless", "Employment"}

    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/calendar/economic?start={TODAY}&end={TODAY}"
        data = _yahoo_get(url)
        items = data.get("finance", {}).get("result", [{}])[0].get("events", [])

        for item in items:
            name = item.get("eventName", item.get("title", ""))
            time_str = item.get("startdatetime", "")
            time_et = ""
            if time_str:
                try:
                    dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                    time_et = dt.astimezone(ET).strftime("%-I:%M %p")
                except Exception:
                    time_et = time_str

            impact = "low"
            if any(kw in name for kw in HIGH_IMPACT):
                impact = "high"
            elif any(kw in name for kw in MEDIUM_IMPACT):
                impact = "medium"

            events.append({
                "event": name,
                "time_et": time_et,
                "impact": impact,
            })
    except Exception as e:
        print(f"  WARN: Econ events fetch failed: {e}")

    return events


# -- Headlines --
def fetch_headlines() -> list:
    headlines = []

    # Method 1: Yahoo news API
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/news?category=market"
        data = _yahoo_get(url)
        items = data.get("items", data.get("news", []))
        for item in items[:5]:
            headlines.append({
                "text": item.get("title", ""),
                "url": item.get("link", item.get("url", "")),
            })
    except Exception:
        pass

    # Method 2: SPY search news
    if not headlines:
        try:
            url = "https://query1.finance.yahoo.com/v1/finance/search?q=SPY&newsCount=5&quotesCount=0"
            data = _yahoo_get(url)
            for item in data.get("news", [])[:5]:
                headlines.append({
                    "text": item.get("title", ""),
                    "url": item.get("link", item.get("url", "")),
                })
        except Exception:
            pass

    return headlines[:5]


# -- Main --
def main():
    print(f"Attitash Pre-Market Briefing -- {DATE_DISPLAY}")
    print(f"Current time ET: {NOW_ET.strftime('%H:%M %Z')}")

    if not is_trading_day():
        print("Not a trading day. Exiting.")
        sys.exit(0)

    print("\n1/6 Fetching futures & macro...")
    futures = fetch_futures_macro()
    for label, data in futures.items():
        sign = "+" if data["change_pct"] >= 0 else ""
        print(f"  {label}: {data['price']} ({sign}{data['change_pct']}%)")

    print("\n2/6 Assessing sentiment...")
    sentiment = assess_sentiment(futures)
    print(f"  Sentiment: {sentiment}")

    print("\n3/6 Scanning for gappers...")
    gappers = fetch_gainers()
    print(f"  Found {len(gappers)} qualifying gappers")

    print("\n4/6 Fetching earnings calendar...")
    earnings = fetch_earnings()
    print(f"  Found {len(earnings)} earnings reports today")

    print("\n5/6 Fetching economic events...")
    econ_events = fetch_econ_events()
    print(f"  Found {len(econ_events)} events")

    print("\n6/6 Fetching headlines...")
    headlines = fetch_headlines()
    print(f"  Found {len(headlines)} headlines")

    # Build briefing.json
    briefing = {
        "date": DATE_DISPLAY,
        "generated_at": NOW_ET.isoformat(),
        "sentiment": sentiment,
        "futures": futures,
        "gappers": gappers,
        "earnings": earnings,
        "econ_events": econ_events,
        "headlines": headlines,
    }

    os.makedirs("data", exist_ok=True)
    output_path = "data/briefing.json"
    with open(output_path, "w") as f:
        json.dump(briefing, f, indent=2)

    print(f"\nBriefing written to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
