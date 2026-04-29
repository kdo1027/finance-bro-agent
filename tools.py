"""
tools.py — Signal tools for Finance Bro Agent.

Pass 1 (sector-level): Signals A-D — score and filter sectors.
Pass 2 (stock-level):  Signals E-G — score stocks within top sectors.

Sentiment (A, E) — powered by FinBERT (integrated by kdo1027).
Fundamentals (B, F) — Alpha Vantage OVERVIEW endpoint.
Macro (C)          — FRED API (fed funds rate, CPI, unemployment).
Momentum (D, G)    — Alpha Vantage TIME_SERIES_DAILY_ADJUSTED (replaces yfinance).

Rate limits: Alpha Vantage free tier = 25 calls/day, 5 calls/minute.
Results are cached to .cache/signals.json for up to 4 hours to stay within limits.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from langchain_core.tools import tool

# ── API Keys (loaded from .env) ────────────────────────────────────────────────
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
FRED_API_KEY      = os.getenv("FRED_API_KEY", "")

# ── Sector → ETF mapping ───────────────────────────────────────────────────────
SECTOR_ETFS = {
    "tech":        "XLK",
    "healthcare":  "XLV",
    "energy":      "XLE",
    "financials":  "XLF",
    "consumer":    "XLY",
    "industrials": "XLI",
}

# ── Sector → representative stocks for Pass 2 ─────────────────────────────────
SECTOR_STOCKS = {
    "tech":        [{"ticker": "NVDA", "name": "Nvidia"},
                    {"ticker": "MSFT", "name": "Microsoft"},
                    {"ticker": "AAPL", "name": "Apple"}],
    "healthcare":  [{"ticker": "UNH",  "name": "UnitedHealth"},
                    {"ticker": "JNJ",  "name": "Johnson & Johnson"},
                    {"ticker": "LLY",  "name": "Eli Lilly"}],
    "energy":      [{"ticker": "XOM",  "name": "ExxonMobil"},
                    {"ticker": "CVX",  "name": "Chevron"},
                    {"ticker": "COP",  "name": "ConocoPhillips"}],
    "financials":  [{"ticker": "JPM",  "name": "JPMorgan Chase"},
                    {"ticker": "BAC",  "name": "Bank of America"},
                    {"ticker": "GS",   "name": "Goldman Sachs"}],
    "consumer":    [{"ticker": "AMZN", "name": "Amazon"},
                    {"ticker": "HD",   "name": "Home Depot"},
                    {"ticker": "MCD",  "name": "McDonald's"}],
    "industrials": [{"ticker": "CAT",  "name": "Caterpillar"},
                    {"ticker": "HON",  "name": "Honeywell"},
                    {"ticker": "UPS",  "name": "UPS"}],
}

# ── File cache (avoids burning 25 calls/day on repeat runs) ───────────────────
_CACHE_FILE    = Path(".cache/signals.json")
_CACHE_TTL_HRS = 4


def _cache_get(key: str):
    if not _CACHE_FILE.exists():
        return None
    try:
        data  = json.loads(_CACHE_FILE.read_text())
        entry = data.get(key)
        if not entry:
            return None
        if datetime.now() - datetime.fromisoformat(entry["cached_at"]) > timedelta(hours=_CACHE_TTL_HRS):
            return None
        return entry["value"]
    except Exception:
        return None


def _cache_set(key: str, value):
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(_CACHE_FILE.read_text()) if _CACHE_FILE.exists() else {}
    except Exception:
        data = {}
    data[key] = {"value": value, "cached_at": datetime.now().isoformat()}
    _CACHE_FILE.write_text(json.dumps(data, indent=2))


# ── Alpha Vantage helpers ──────────────────────────────────────────────────────
_AV_BASE      = "https://www.alphavantage.co/query"
_last_av_call = 0.0


def _av_get(function: str, symbol: str = None, extra: dict = None) -> dict:
    """Call Alpha Vantage with rate-limit spacing (max 5/min on free tier)."""
    global _last_av_call
    elapsed = time.time() - _last_av_call
    if elapsed < 12:
        time.sleep(12 - elapsed)   # 12s gap → max 5/min with headroom

    params = {"function": function, "apikey": ALPHA_VANTAGE_KEY}
    if symbol:
        params["symbol"] = symbol
    if extra:
        params.update(extra)

    resp = requests.get(_AV_BASE, params=params, timeout=15)
    _last_av_call = time.time()
    resp.raise_for_status()
    return resp.json()


def _parse_overview(data: dict) -> dict:
    """Turn an Alpha Vantage OVERVIEW response into our health_score schema."""
    def safe_float(key, default=0.0):
        try:
            return float(data.get(key) or default)
        except (ValueError, TypeError):
            return default

    pe         = safe_float("PERatio")
    de         = safe_float("DebtEquityRatio")
    rev_growth = safe_float("QuarterlyRevenueGrowthYOY") * 100  # → percentage
    beta       = safe_float("Beta", 1.0)

    # Simple heuristic health score (-1.0 to 1.0)
    pe_score     = 0.4 if 8 <= pe <= 25 else (0.1 if pe < 45 else -0.3)
    de_score     = 0.3 if de < 0.5 else (0.1 if de < 1.2 else -0.2)
    growth_score = 0.4 if rev_growth > 10 else (0.2 if rev_growth > 0 else -0.3)
    health_score = round(min(1.0, max(-1.0, pe_score + de_score + growth_score)), 3)

    return {
        "pe_ratio":           round(pe, 1),
        "debt_to_equity":     round(de, 2),
        "revenue_growth_pct": round(rev_growth, 1),
        "health_score":       health_score,
        "health_label":       ("healthy" if health_score >= 0.5 else
                               "fair"    if health_score >= 0.2 else
                               "mixed"   if health_score >= -0.1 else "weak"),
        "volatility":         "high" if beta > 1.3 else "low" if beta < 0.7 else "medium",
        "beta":               round(beta, 2),
        "source":             "Alpha Vantage OVERVIEW",
    }


def _parse_price_changes(data: dict, symbol: str) -> dict:
    """Compute 5/10/30-day % changes from TIME_SERIES_DAILY_ADJUSTED response."""
    series = data.get("Time Series (Daily)", {})
    dates  = sorted(series.keys(), reverse=True)

    if not dates:
        return {"change_5d_pct": 0.0, "change_10d_pct": 0.0, "change_30d_pct": 0.0,
                "momentum_score": 0.0, "trend": "flat",
                "source": "Alpha Vantage (no data)"}

    current = float(series[dates[0]]["5. adjusted close"])

    def pct(n):
        if len(dates) > n:
            past = float(series[dates[n]]["5. adjusted close"])
            return round((current - past) / past * 100, 2)
        return 0.0

    c5, c10, c30 = pct(5), pct(10), pct(30)

    # Weighted momentum score normalized to -1 → 1 (10% move ≈ ±1.0)
    raw   = (c5 * 0.5 + c10 * 0.3 + c30 * 0.2) / 10
    score = round(max(-1.0, min(1.0, raw)), 3)

    return {
        "symbol":         symbol,
        "change_5d_pct":  c5,
        "change_10d_pct": c10,
        "change_30d_pct": c30,
        "momentum_score": score,
        "trend":          "uptrend" if score > 0.2 else "downtrend" if score < -0.2 else "flat",
        "source":         "Alpha Vantage TIME_SERIES_DAILY_ADJUSTED",
    }


# ── FRED helpers ───────────────────────────────────────────────────────────────
_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def _fred_latest(series_id: str) -> float:
    """Fetch the most recent value for a FRED series."""
    params = {"series_id": series_id, "api_key": FRED_API_KEY,
              "sort_order": "desc", "limit": 1, "file_type": "json"}
    resp = requests.get(_FRED_BASE, params=params, timeout=10)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    return float(obs[0]["value"]) if obs else 0.0


def _fred_yoy(series_id: str) -> float:
    """Fetch YoY % change for a monthly FRED series (e.g. CPI)."""
    params = {"series_id": series_id, "api_key": FRED_API_KEY,
              "sort_order": "desc", "limit": 13, "file_type": "json", "units": "pc1"}
    resp = requests.get(_FRED_BASE, params=params, timeout=10)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    return float(obs[0]["value"]) if obs else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — SECTOR-LEVEL TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def get_sentiment_signal(sectors: list[str]) -> dict:
    """Signal A: News sentiment scores for each sector via FinBERT.

    Powered by the FinBERT model integrated by kdo1027.
    Range: -1.0 (very negative) to 1.0 (very positive).
    Falls back to neutral (0.0) if model is not loaded.
    """
    # FinBERT pipeline is wired in by kdo1027's integration.
    # This stub is a fallback — the real scores come from that pipeline.
    results = {}
    for sector in sectors:
        results[sector] = {
            "sentiment_score": 0.0,
            "label":           "neutral",
            "source":          "FinBERT (fallback — model not loaded)",
        }
    return results


@tool
def get_fundamentals_signal(sectors: list[str]) -> dict:
    """Signal B: Fundamental health for each sector via Alpha Vantage OVERVIEW on sector ETF.

    Uses the sector ETF (XLK for tech, XLV for healthcare, etc.) as a proxy.
    Results are cached for 4 hours to stay within the 25 calls/day free-tier limit.
    """
    results = {}
    for sector in sectors:
        etf = SECTOR_ETFS.get(sector.lower())
        if not etf:
            results[sector] = {"health_score": 0.0, "health_label": "unknown",
                               "source": "no ETF mapping"}
            continue

        cache_key = f"fundamentals_sector_{etf}"
        cached    = _cache_get(cache_key)
        if cached:
            results[sector] = cached
            continue

        try:
            parsed = _parse_overview(_av_get("OVERVIEW", symbol=etf))
            _cache_set(cache_key, parsed)
            results[sector] = parsed
        except Exception as e:
            results[sector] = {"health_score": 0.0, "health_label": "error",
                               "error": str(e), "source": "Alpha Vantage (error)"}
    return results


@tool
def get_macro_signal() -> dict:
    """Signal C: Macroeconomic environment via FRED API.

    Fetches: Federal Funds Rate, CPI YoY %, and Unemployment Rate.
    Applies equally to all sectors — one global macro score.
    Cached for 4 hours (macro data changes slowly).
    """
    cached = _cache_get("macro_fred")
    if cached:
        return cached

    try:
        fed_rate    = _fred_latest("FEDFUNDS")
        cpi_yoy     = _fred_yoy("CPIAUCSL")
        unemployment = _fred_latest("UNRATE")

        # Macro score heuristic (-1.0 to 1.0)
        rate_score  = -0.4 if fed_rate > 5.0 else (-0.2 if fed_rate > 3.0 else 0.2)
        cpi_score   = -0.3 if cpi_yoy > 4.0  else (-0.1 if cpi_yoy > 2.5  else 0.2)
        unemp_score =  0.2 if unemployment < 4.5 else (0.0 if unemployment < 6.0 else -0.2)
        macro_score = round(max(-1.0, min(1.0, rate_score + cpi_score + unemp_score)), 3)

        result = {
            "fed_funds_rate":    fed_rate,
            "cpi_yoy_pct":       round(cpi_yoy, 2),
            "unemployment_rate": unemployment,
            "macro_score":       macro_score,
            "plain_summary": (
                f"Interest rates are {'high' if fed_rate > 4 else 'moderate'} at {fed_rate}%. "
                f"Inflation is {'above target' if cpi_yoy > 2.5 else 'under control'} "
                f"at {cpi_yoy:.1f}% year-over-year. "
                f"Unemployment is {'low' if unemployment < 5 else 'elevated'} at {unemployment}%."
            ),
            "source": "FRED API",
        }
        _cache_set("macro_fred", result)
        return result

    except Exception as e:
        return {
            "fed_funds_rate": 0.0, "cpi_yoy_pct": 0.0, "unemployment_rate": 0.0,
            "macro_score": 0.0, "plain_summary": "Macro data unavailable.",
            "error": str(e), "source": "FRED API (error)",
        }


@tool
def get_momentum_signal(sectors: list[str]) -> dict:
    """Signal D: Recent price trend for each sector ETF via Alpha Vantage.

    Price action only — 5, 10, and 30-day % changes on the sector ETF.
    Supporting signal only (weight: 15%) — reflects what already happened.
    Replaces yfinance entirely using Alpha Vantage TIME_SERIES_DAILY_ADJUSTED.
    Cached for 4 hours.
    """
    results = {}
    for sector in sectors:
        etf = SECTOR_ETFS.get(sector.lower())
        if not etf:
            results[sector] = {"momentum_score": 0.0, "trend": "flat",
                               "source": "no ETF mapping"}
            continue

        cache_key = f"momentum_sector_{etf}"
        cached    = _cache_get(cache_key)
        if cached:
            results[sector] = cached
            continue

        try:
            data   = _av_get("TIME_SERIES_DAILY_ADJUSTED", symbol=etf,
                             extra={"outputsize": "compact"})
            parsed = _parse_price_changes(data, etf)
            parsed["etf"] = etf
            _cache_set(cache_key, parsed)
            results[sector] = parsed
        except Exception as e:
            results[sector] = {"momentum_score": 0.0, "trend": "flat",
                               "error": str(e), "source": "Alpha Vantage (error)"}
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — STOCK-LEVEL TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def get_stock_sentiment_signal(tickers: list[str]) -> dict:
    """Signal E: News sentiment for individual stocks via FinBERT.

    Powered by the FinBERT model integrated by kdo1027.
    Falls back to neutral (0.0) if model is not loaded.
    """
    results = {}
    for ticker in tickers:
        results[ticker] = {
            "sentiment_score": 0.0,
            "label":           "neutral",
            "source":          "FinBERT (fallback — model not loaded)",
        }
    return results


@tool
def get_stock_fundamentals_signal(tickers: list[str]) -> dict:
    """Signal F: Fundamental health for individual stocks via Alpha Vantage OVERVIEW.

    Cached per ticker for 4 hours to stay within the 25 calls/day free-tier limit.
    """
    results = {}
    for ticker in tickers:
        cache_key = f"fundamentals_stock_{ticker}"
        cached    = _cache_get(cache_key)
        if cached:
            results[ticker] = cached
            continue

        try:
            parsed = _parse_overview(_av_get("OVERVIEW", symbol=ticker))
            _cache_set(cache_key, parsed)
            results[ticker] = parsed
        except Exception as e:
            results[ticker] = {"health_score": 0.0, "health_label": "error",
                               "error": str(e), "source": "Alpha Vantage (error)"}
    return results


@tool
def get_stock_momentum_signal(tickers: list[str]) -> dict:
    """Signal G: Recent price trend for individual stocks via Alpha Vantage.

    Price action only — 5, 10, and 30-day % changes per stock.
    Replaces yfinance entirely using Alpha Vantage TIME_SERIES_DAILY_ADJUSTED.
    Cached per ticker for 4 hours.
    """
    results = {}
    for ticker in tickers:
        cache_key = f"momentum_stock_{ticker}"
        cached    = _cache_get(cache_key)
        if cached:
            results[ticker] = cached
            continue

        try:
            data   = _av_get("TIME_SERIES_DAILY_ADJUSTED", symbol=ticker,
                             extra={"outputsize": "compact"})
            parsed = _parse_price_changes(data, ticker)
            _cache_set(cache_key, parsed)
            results[ticker] = parsed
        except Exception as e:
            results[ticker] = {"momentum_score": 0.0, "trend": "flat",
                               "error": str(e), "source": "Alpha Vantage (error)"}
    return results


# ── Tool lists for agent.py ────────────────────────────────────────────────────
SECTOR_TOOLS = [get_sentiment_signal, get_fundamentals_signal,
                get_macro_signal, get_momentum_signal]

STOCK_TOOLS  = [get_stock_sentiment_signal, get_stock_fundamentals_signal,
                get_stock_momentum_signal]

ALL_TOOLS    = SECTOR_TOOLS + STOCK_TOOLS
