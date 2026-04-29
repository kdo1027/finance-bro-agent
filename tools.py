"""
tools.py — Signal tools for Signals A-D (sector level) and E-G (stock level).

Sector-level tools run first (Pass 1) to grade and filter sectors.
Stock-level tools run second (Pass 2) only for the top sectors that passed.

Signal A (sector sentiment) — fine-tuned FinBERT via finbert_adapter.
Signal B (sector fundamentals) — Alpha Vantage OVERVIEW on sector ETFs.
Signal C (macro) — FRED API: fed funds rate, CPI YoY, unemployment.
Signal D (sector momentum) — Alpha Vantage TIME_SERIES_DAILY_ADJUSTED on sector ETFs.
Signal E (stock sentiment) — fine-tuned FinBERT via finbert_adapter.
Signal F (stock fundamentals) — Alpha Vantage OVERVIEW per ticker.
Signal G (stock momentum) — Alpha Vantage TIME_SERIES_DAILY_ADJUSTED per ticker.

Rate limits: Alpha Vantage free tier = 25 calls/day, 5 calls/min.
Results cached to .cache/signals.json for 4 hours to stay within limits.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from langchain_core.tools import tool

from finbert_adapter import get_sector_sentiments, get_stock_sentiments

# ── API keys (from .env) ───────────────────────────────────────────────────────
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
FRED_API_KEY      = os.getenv("FRED_API_KEY", "")

# ── Sector ETF mapping ─────────────────────────────────────────────────────────
SECTOR_ETFS = {
    "tech":        "XLK",
    "healthcare":  "XLV",
    "energy":      "XLE",
    "financials":  "XLF",
    "consumer":    "XLY",
    "industrials": "XLI",
}

# ── 4-hour file cache to avoid burning 25 calls/day ───────────────────────────
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
    """Call Alpha Vantage, spacing calls 12s apart to respect 5 calls/min limit."""
    global _last_av_call
    elapsed = time.time() - _last_av_call
    if elapsed < 12:
        time.sleep(12 - elapsed)
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
    """Convert Alpha Vantage OVERVIEW response to health_score schema."""
    def safe_float(key, default=0.0):
        try:
            return float(data.get(key) or default)
        except (ValueError, TypeError):
            return default

    pe         = safe_float("PERatio")
    de         = safe_float("DebtEquityRatio")
    rev_growth = safe_float("QuarterlyRevenueGrowthYOY") * 100
    beta       = safe_float("Beta", 1.0)

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
    """Compute 5/10/30-day % changes from TIME_SERIES_DAILY_ADJUSTED."""
    series = data.get("Time Series (Daily)", {})
    dates  = sorted(series.keys(), reverse=True)
    if not dates:
        return {"change_5d_pct": 0.0, "change_10d_pct": 0.0, "change_30d_pct": 0.0,
                "momentum_score": 0.0, "trend": "flat", "source": "Alpha Vantage (no data)"}

    current = float(series[dates[0]]["5. adjusted close"])

    def pct(n):
        if len(dates) > n:
            past = float(series[dates[n]]["5. adjusted close"])
            return round((current - past) / past * 100, 2)
        return 0.0

    c5, c10, c30 = pct(5), pct(10), pct(30)
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
    params = {"series_id": series_id, "api_key": FRED_API_KEY,
              "sort_order": "desc", "limit": 1, "file_type": "json"}
    resp = requests.get(_FRED_BASE, params=params, timeout=10)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    return float(obs[0]["value"]) if obs else 0.0


def _fred_yoy(series_id: str) -> float:
    params = {"series_id": series_id, "api_key": FRED_API_KEY,
              "sort_order": "desc", "limit": 13, "file_type": "json", "units": "pc1"}
    resp = requests.get(_FRED_BASE, params=params, timeout=10)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    return float(obs[0]["value"]) if obs else 0.0


def _ts(hours_ago: float) -> str:
    """Return an ISO-8601 UTC timestamp for N hours ago."""
    dt = datetime.now(tz=timezone.utc) - timedelta(hours=hours_ago)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# Stub headlines fed to FinBERT for Signal A (sector-level sentiment).
# Replace the list values with live NewsAPI headlines keyed by FinBERT sector name.
_SECTOR_HEADLINES: dict[str, list[dict]] = {
    "technology": [
        {"text": "NVIDIA reports record quarterly revenue driven by surging AI chip demand", "timestamp": _ts(2)},
        {"text": "Microsoft Azure cloud growth accelerates as enterprise AI adoption rises", "timestamp": _ts(5)},
        {"text": "Apple faces slowing iPhone sales in key China market amid economic headwinds", "timestamp": _ts(9)},
        {"text": "Semiconductor stocks rally after AMD posts strong earnings and raises guidance", "timestamp": _ts(14)},
        {"text": "Google DeepMind announces breakthrough in AI reasoning capabilities", "timestamp": _ts(20)},
    ],
    "healthcare": [
        {"text": "Eli Lilly obesity drug Zepbound shows strong demand driving revenue growth", "timestamp": _ts(3)},
        {"text": "FDA approves new Pfizer vaccine for respiratory virus following clinical trial success", "timestamp": _ts(7)},
        {"text": "UnitedHealth raises full-year earnings outlook on strong membership growth", "timestamp": _ts(11)},
        {"text": "Johnson and Johnson faces new litigation over talc products", "timestamp": _ts(16)},
        {"text": "Biotech sector gains as new cancer therapeutics show promising trial results", "timestamp": _ts(22)},
    ],
    "energy": [
        {"text": "OPEC maintains production cuts as crude oil prices stabilize near multi-month lows", "timestamp": _ts(1)},
        {"text": "ExxonMobil beats earnings estimates on strong refinery margins", "timestamp": _ts(6)},
        {"text": "Natural gas futures fall sharply on warmer-than-expected winter forecasts", "timestamp": _ts(10)},
        {"text": "Chevron expands renewable energy investments with new solar partnership", "timestamp": _ts(15)},
        {"text": "Oil prices drop on rising US crude inventories and demand concerns", "timestamp": _ts(21)},
    ],
    "financials": [
        {"text": "JPMorgan Chase reports record profit on strong investment banking revenue", "timestamp": _ts(2)},
        {"text": "Federal Reserve holds interest rates steady signaling cautious outlook", "timestamp": _ts(6)},
        {"text": "Goldman Sachs trading revenue surges amid market volatility", "timestamp": _ts(10)},
        {"text": "Bank of America net interest income falls short of analyst expectations", "timestamp": _ts(17)},
        {"text": "Credit card delinquencies rise as consumer financial stress increases", "timestamp": _ts(23)},
    ],
    "consumer": [
        {"text": "Amazon Prime membership growth accelerates boosting e-commerce revenue", "timestamp": _ts(3)},
        {"text": "McDonald's global same-store sales beat estimates on value menu strength", "timestamp": _ts(8)},
        {"text": "Home Depot cuts outlook citing softer housing market and cautious consumer spending", "timestamp": _ts(12)},
        {"text": "Walmart gains market share as budget-conscious consumers trade down", "timestamp": _ts(18)},
        {"text": "Consumer confidence index rises for third consecutive month", "timestamp": _ts(24)},
    ],
    "industrials": [
        {"text": "Caterpillar construction equipment demand remains robust on infrastructure spending", "timestamp": _ts(4)},
        {"text": "Boeing faces ongoing production challenges with 737 MAX delivery delays", "timestamp": _ts(8)},
        {"text": "Honeywell raises guidance on strong aerospace and defense orders", "timestamp": _ts(13)},
        {"text": "UPS warns of softer package volumes as e-commerce growth moderates", "timestamp": _ts(19)},
        {"text": "US manufacturing PMI expands for second straight month on new orders", "timestamp": _ts(25)},
    ],
}

# Stub headlines fed to FinBERT for Signal E (stock-level sentiment).
# Replace with live NewsAPI queries filtered by company name or ticker.
_STOCK_HEADLINES: dict[str, list[str]] = {
    "NVDA": [
        "NVIDIA dominates AI chip market with Blackwell GPU demand exceeding supply",
        "NVIDIA data center revenue hits record high as hyperscalers expand AI infrastructure",
        "NVIDIA stock surges after analyst upgrades on strong forward guidance",
    ],
    "MSFT": [
        "Microsoft Copilot AI integration drives Office 365 subscription growth",
        "Microsoft Azure revenue growth beats expectations on AI workload demand",
        "Microsoft raises dividend as cloud business generates strong free cash flow",
    ],
    "AAPL": [
        "Apple iPhone 16 sales disappoint in China amid local competition",
        "Apple services revenue grows steadily offsetting hardware weakness",
        "Apple launches new AI features in iOS update to spur upgrade cycle",
    ],
    "UNH": [
        "UnitedHealth raises full-year earnings guidance on strong managed care margins",
        "UnitedHealth faces federal investigation into Medicare billing practices",
        "UnitedHealth membership growth exceeds expectations across commercial plans",
    ],
    "JNJ": [
        "Johnson and Johnson MedTech segment posts solid growth on surgical procedure recovery",
        "Johnson and Johnson reaches talc settlement reducing long-running legal uncertainty",
        "JNJ pharmaceutical pipeline advances with positive late-stage trial results",
    ],
    "LLY": [
        "Eli Lilly Mounjaro and Zepbound sales surge well above analyst forecasts",
        "Eli Lilly expands manufacturing capacity to meet overwhelming GLP-1 drug demand",
        "Eli Lilly obesity drug trial shows additional cardiovascular benefits",
    ],
    "XOM": [
        "ExxonMobil earnings beat estimates on strong downstream refining margins",
        "ExxonMobil increases shareholder returns with expanded buyback program",
        "ExxonMobil faces pressure as crude oil prices slide on demand concerns",
    ],
    "CVX": [
        "Chevron free cash flow remains strong supporting dividend growth",
        "Chevron Hess acquisition faces regulatory scrutiny delaying close",
        "Chevron cuts capital spending forecast amid lower oil price environment",
    ],
    "COP": [
        "ConocoPhillips production volumes disappoint on operational disruptions",
        "ConocoPhillips completes Marathon Oil acquisition expanding US shale footprint",
        "ConocoPhillips warns lower oil prices will pressure 2025 cash flow",
    ],
    "JPM": [
        "JPMorgan Chase investment banking fees surge on renewed deal activity",
        "JPMorgan posts record annual profit driven by high interest rates",
        "JPMorgan CEO warns of elevated risks from geopolitical and fiscal uncertainty",
    ],
    "BAC": [
        "Bank of America net interest income stabilizes after months of pressure",
        "Bank of America wealth management division posts record client assets",
        "Bank of America loan loss provisions rise as consumer credit quality softens",
    ],
    "GS": [
        "Goldman Sachs trading desk delivers strongest quarter in two years",
        "Goldman Sachs asset and wealth management revenue grows on market appreciation",
        "Goldman Sachs investment banking pipeline strengthens as IPO market reopens",
    ],
    "AMZN": [
        "Amazon AWS cloud revenue accelerates on AI infrastructure spending boom",
        "Amazon advertising business grows rapidly becoming major profit contributor",
        "Amazon Prime Video gains subscribers following NFL streaming rights deal",
    ],
    "HD": [
        "Home Depot revenue misses estimates as high mortgage rates dampen renovation demand",
        "Home Depot professional contractor business remains resilient amid soft DIY sales",
        "Home Depot acquires SRS Distribution to expand pro customer reach",
    ],
    "MCD": [
        "McDonald's value menu drives traffic recovery after consumer pullback",
        "McDonald's international markets show solid comparable sales growth",
        "McDonald's faces ongoing consumer pushback over elevated menu prices",
    ],
    "CAT": [
        "Caterpillar construction segment benefits from sustained US infrastructure investment",
        "Caterpillar raises dividend reflecting confidence in long-term earnings power",
        "Caterpillar dealer inventory normalization creates near-term order headwind",
    ],
    "HON": [
        "Honeywell aerospace division backlog hits record driven by commercial aviation recovery",
        "Honeywell raises full-year guidance on strong industrial automation demand",
        "Honeywell explores separation of business units to unlock shareholder value",
    ],
    "UPS": [
        "UPS volume declines as customers shift to cheaper delivery alternatives",
        "UPS restructuring program targets cost reductions to protect margins",
        "UPS international package business shows modest recovery in European markets",
    ],
}


# Maps each sector to its representative stocks for Pass 2 analysis
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


# ── PASS 1: SECTOR-LEVEL TOOLS ─────────────────────────────────────────────────

@tool
def get_sentiment_signal(sectors: list[str]) -> dict:
    """Signal A: News sentiment scores for each sector via fine-tuned FinBERT.

    Args:
        sectors: List of sector names (e.g. ["tech", "healthcare", "energy"])

    Returns:
        Dict mapping sector to sentiment score (-1.0 to 1.0) and label.
        Score > 0.2 = positive, < -0.2 = negative, else neutral.
    """
    # Gather headlines for the requested sectors (keyed by FinBERT sector name)
    headlines = []
    for sector in sectors:
        fb_sector = "technology" if sector == "tech" else sector
        headlines.extend(_SECTOR_HEADLINES.get(fb_sector, []))

    scores = get_sector_sentiments(sectors, headlines)

    results = {}
    for sector in sectors:
        score = scores[sector]
        results[sector] = {
            "sentiment_score": score,
            "label": "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral",
            "headline_count": len(_SECTOR_HEADLINES.get("technology" if sector == "tech" else sector, [])),
            "source": "finbert_finetuned",
        }
    return results


@tool
def get_fundamentals_signal(sectors: list[str]) -> dict:
    """Signal B: Fundamental health for each sector via Alpha Vantage OVERVIEW on sector ETF.

    Uses the sector ETF (XLK, XLV, XLE, XLF, XLY, XLI) as a proxy for sector health.
    Cached 4 hours to stay within the 25 calls/day free-tier limit.
    """
    results = {}
    for sector in sectors:
        etf       = SECTOR_ETFS.get(sector.lower())
        cache_key = f"fundamentals_sector_{etf}"
        cached    = _cache_get(cache_key)
        if cached:
            results[sector] = cached
            continue
        if not etf:
            results[sector] = {"health_score": 0.0, "health_label": "unknown", "source": "no ETF mapping"}
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

    Fetches live Federal Funds Rate, CPI YoY %, and Unemployment Rate.
    Applies equally to all sectors — one global macro score.
    Cached 4 hours (macro data changes slowly).
    """
    cached = _cache_get("macro_fred")
    if cached:
        return cached
    try:
        fed_rate     = _fred_latest("FEDFUNDS")
        cpi_yoy      = _fred_yoy("CPIAUCSL")
        unemployment = _fred_latest("UNRATE")

        rate_score   = -0.4 if fed_rate > 5.0 else (-0.2 if fed_rate > 3.0 else 0.2)
        cpi_score    = -0.3 if cpi_yoy > 4.0  else (-0.1 if cpi_yoy > 2.5  else 0.2)
        unemp_score  =  0.2 if unemployment < 4.5 else (0.0 if unemployment < 6.0 else -0.2)
        macro_score  = round(max(-1.0, min(1.0, rate_score + cpi_score + unemp_score)), 3)

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
        return {"fed_funds_rate": 0.0, "cpi_yoy_pct": 0.0, "unemployment_rate": 0.0,
                "macro_score": 0.0, "plain_summary": "Macro data unavailable.",
                "error": str(e), "source": "FRED API (error)"}


@tool
def get_momentum_signal(sectors: list[str]) -> dict:
    """Signal D: Recent price trend for each sector ETF via Alpha Vantage.

    Price action only — 5, 10, 30-day % changes. Supporting signal (weight: 15%).
    Replaces yfinance with Alpha Vantage TIME_SERIES_DAILY_ADJUSTED. Cached 4 hours.
    """
    results = {}
    for sector in sectors:
        etf       = SECTOR_ETFS.get(sector.lower())
        cache_key = f"momentum_sector_{etf}"
        cached    = _cache_get(cache_key)
        if cached:
            results[sector] = cached
            continue
        if not etf:
            results[sector] = {"momentum_score": 0.0, "trend": "flat", "source": "no ETF mapping"}
            continue
        try:
            data   = _av_get("TIME_SERIES_DAILY_ADJUSTED", symbol=etf, extra={"outputsize": "compact"})
            parsed = _parse_price_changes(data, etf)
            parsed["etf"] = etf
            _cache_set(cache_key, parsed)
            results[sector] = parsed
        except Exception as e:
            results[sector] = {"momentum_score": 0.0, "trend": "flat",
                               "error": str(e), "source": "Alpha Vantage (error)"}
    return results


# ── PASS 2: STOCK-LEVEL TOOLS ──────────────────────────────────────────────────

@tool
def get_stock_sentiment_signal(tickers: list[str]) -> dict:
    """Signal E: News sentiment for individual stocks via fine-tuned FinBERT.

    Args:
        tickers: List of stock tickers (e.g. ["NVDA", "MSFT", "AAPL"])

    Returns:
        Dict mapping ticker to sentiment score and label.
    """
    ticker_headlines = {t.upper(): _STOCK_HEADLINES.get(t.upper(), []) for t in tickers}
    scores = get_stock_sentiments(ticker_headlines)

    results = {}
    for ticker in tickers:
        score = scores[ticker.upper()]
        results[ticker] = {
            "sentiment_score": score,
            "label": "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral",
            "headline_count": len(_STOCK_HEADLINES.get(ticker.upper(), [])),
            "source": "finbert_finetuned",
        }
    return results


@tool
def get_stock_fundamentals_signal(tickers: list[str]) -> dict:
    """Signal F: Fundamental health for individual stocks via Alpha Vantage OVERVIEW.

    Cached per ticker for 4 hours to stay within the 25 calls/day free-tier limit.
    """
    results = {}
    for ticker in tickers:
        cache_key = f"fundamentals_stock_{ticker.upper()}"
        cached    = _cache_get(cache_key)
        if cached:
            results[ticker] = cached
            continue
        try:
            parsed = _parse_overview(_av_get("OVERVIEW", symbol=ticker.upper()))
            _cache_set(cache_key, parsed)
            results[ticker] = parsed
        except Exception as e:
            results[ticker] = {"health_score": 0.0, "health_label": "error",
                               "error": str(e), "source": "Alpha Vantage (error)"}
    return results


@tool
def get_stock_momentum_signal(tickers: list[str]) -> dict:
    """Signal G: Recent price trend for individual stocks via Alpha Vantage.

    Price action only — 5, 10, 30-day % changes. Supporting signal (weight: 15%).
    Replaces yfinance with Alpha Vantage TIME_SERIES_DAILY_ADJUSTED. Cached per ticker.
    """
    results = {}
    for ticker in tickers:
        cache_key = f"momentum_stock_{ticker.upper()}"
        cached    = _cache_get(cache_key)
        if cached:
            results[ticker] = cached
            continue
        try:
            data   = _av_get("TIME_SERIES_DAILY_ADJUSTED", symbol=ticker.upper(),
                             extra={"outputsize": "compact"})
            parsed = _parse_price_changes(data, ticker.upper())
            _cache_set(cache_key, parsed)
            results[ticker] = parsed
        except Exception as e:
            results[ticker] = {"momentum_score": 0.0, "trend": "flat",
                               "error": str(e), "source": "Alpha Vantage (error)"}
    return results


# Sector-level tools (Pass 1)
SECTOR_TOOLS = [
    get_sentiment_signal,
    get_fundamentals_signal,
    get_macro_signal,
    get_momentum_signal,
]

# Stock-level tools (Pass 2)
STOCK_TOOLS = [
    get_stock_sentiment_signal,
    get_stock_fundamentals_signal,
    get_stock_momentum_signal,
]

# All tools (for LangChain tool binding if needed)
ALL_TOOLS = SECTOR_TOOLS + STOCK_TOOLS
