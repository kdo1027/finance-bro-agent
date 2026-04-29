"""
tools.py — Signal tools for Signals A-D (sector level) and E-G (stock level).

Sector-level tools run first (Pass 1) to grade and filter sectors.
Stock-level tools run second (Pass 2) only for the top sectors that passed.

Signal A (sector sentiment) and Signal E (stock sentiment) use the local
fine-tuned FinBERT model via finbert_adapter. Signals B-D and F-G remain
stubs — see inline comments for real API replacements.
"""

from datetime import datetime, timedelta, timezone

from langchain_core.tools import tool

from finbert_adapter import get_sector_sentiments, get_stock_sentiments


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
    """Signal B: Fundamental financial health for each sector.

    Args:
        sectors: List of sector names.

    Returns:
        Dict mapping sector to P/E ratio, debt-to-equity, revenue growth, and health score.
        health_score is -1.0 to 1.0 (positive = healthy, negative = stretched/weak).
    """
    # STUB — replace with SimFin or Alpha Vantage sector aggregates
    stub_data = {
        "tech":        {"pe_ratio": 28.5, "debt_to_equity": 0.45, "revenue_growth_pct": 12, "health_score":  0.55, "health_label": "healthy but stretched"},
        "healthcare":  {"pe_ratio": 18.2, "debt_to_equity": 0.35, "revenue_growth_pct":  8, "health_score":  0.70, "health_label": "healthy"},
        "energy":      {"pe_ratio": 12.1, "debt_to_equity": 0.60, "revenue_growth_pct": -3, "health_score": -0.10, "health_label": "mixed"},
        "financials":  {"pe_ratio": 15.0, "debt_to_equity": 0.80, "revenue_growth_pct":  5, "health_score":  0.30, "health_label": "fair"},
        "consumer":    {"pe_ratio": 22.0, "debt_to_equity": 0.50, "revenue_growth_pct":  4, "health_score":  0.25, "health_label": "fair"},
        "industrials": {"pe_ratio": 19.5, "debt_to_equity": 0.40, "revenue_growth_pct":  6, "health_score":  0.40, "health_label": "healthy"},
    }
    results = {}
    for sector in sectors:
        data = stub_data.get(sector.lower(), {
            "pe_ratio": 20.0, "debt_to_equity": 0.50, "revenue_growth_pct": 5,
            "health_score": 0.0, "health_label": "unknown"
        })
        data["source"] = "stub — replace with SimFin/Alpha Vantage"
        results[sector] = data
    return results


@tool
def get_macro_signal() -> dict:
    """Signal C: Macroeconomic environment — applies globally to all sectors.

    Returns:
        Dict with interest rate, inflation, unemployment, and a macro_score (-1.0 to 1.0).
        Positive score = economy supports investment; negative = headwinds.
    """
    # STUB — replace with FRED API
    return {
        "fed_funds_rate": 5.25,
        "cpi_yoy_pct": 3.2,
        "unemployment_rate": 3.8,
        "macro_score": 0.10,
        "plain_summary": "Rates are still high and inflation is cooling but not gone. "
                         "This favors stable, value-oriented companies over fast-growing ones.",
        "source": "stub — replace with FRED API",
    }


@tool
def get_momentum_signal(sectors: list[str]) -> dict:
    """Signal D: Recent price trend for each sector ETF.

    Momentum = price action only. Shows how much the sector ETF price
    has moved over the last 5, 10, and 30 days. This is a supporting
    signal — it reflects what already happened, not what will happen.

    Args:
        sectors: List of sector names.

    Returns:
        Dict mapping sector to price change percentages and a momentum_score (-1.0 to 1.0).
    """
    # STUB — replace with yfinance on sector ETFs (XLK, XLV, XLE, XLF, etc.)
    stub_data = {
        "tech":        {"etf": "XLK", "change_5d_pct":  2.1, "change_10d_pct":  3.8, "change_30d_pct":  5.2, "momentum_score":  0.60, "trend": "uptrend"},
        "healthcare":  {"etf": "XLV", "change_5d_pct": -0.4, "change_10d_pct": -1.2, "change_30d_pct": -2.1, "momentum_score": -0.20, "trend": "downtrend"},
        "energy":      {"etf": "XLE", "change_5d_pct": -1.1, "change_10d_pct": -2.5, "change_30d_pct": -3.8, "momentum_score": -0.35, "trend": "downtrend"},
        "financials":  {"etf": "XLF", "change_5d_pct":  0.8, "change_10d_pct":  1.5, "change_30d_pct":  1.9, "momentum_score":  0.20, "trend": "slight uptrend"},
        "consumer":    {"etf": "XLY", "change_5d_pct":  0.3, "change_10d_pct":  0.9, "change_30d_pct":  1.2, "momentum_score":  0.15, "trend": "flat"},
        "industrials": {"etf": "XLI", "change_5d_pct":  1.0, "change_10d_pct":  2.0, "change_30d_pct":  3.1, "momentum_score":  0.35, "trend": "uptrend"},
    }
    results = {}
    for sector in sectors:
        data = stub_data.get(sector.lower(), {
            "etf": "N/A", "change_5d_pct": 0.0, "change_10d_pct": 0.0,
            "change_30d_pct": 0.0, "momentum_score": 0.0, "trend": "flat"
        })
        data["source"] = "stub — replace with yfinance"
        results[sector] = data
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
    """Signal F: Fundamental financial health for individual stocks.

    Args:
        tickers: List of stock tickers.

    Returns:
        Dict mapping ticker to P/E, debt-to-equity, revenue growth, and health score.
    """
    # STUB — replace with SimFin or Alpha Vantage per-stock data
    stub_data = {
        "NVDA":  {"pe_ratio": 55.0, "debt_to_equity": 0.40, "revenue_growth_pct": 122, "health_score":  0.75, "volatility": "high"},
        "MSFT":  {"pe_ratio": 32.0, "debt_to_equity": 0.35, "revenue_growth_pct":  17, "health_score":  0.80, "volatility": "medium"},
        "AAPL":  {"pe_ratio": 28.0, "debt_to_equity": 1.80, "revenue_growth_pct":   2, "health_score":  0.60, "volatility": "low"},
        "UNH":   {"pe_ratio": 20.0, "debt_to_equity": 0.60, "revenue_growth_pct":  12, "health_score":  0.65, "volatility": "low"},
        "JNJ":   {"pe_ratio": 16.5, "debt_to_equity": 0.45, "revenue_growth_pct":   4, "health_score":  0.55, "volatility": "low"},
        "LLY":   {"pe_ratio": 60.0, "debt_to_equity": 1.20, "revenue_growth_pct":  36, "health_score":  0.70, "volatility": "medium"},
        "XOM":   {"pe_ratio": 13.0, "debt_to_equity": 0.20, "revenue_growth_pct":  -5, "health_score":  0.20, "volatility": "medium"},
        "CVX":   {"pe_ratio": 12.5, "debt_to_equity": 0.15, "revenue_growth_pct":  -3, "health_score":  0.25, "volatility": "medium"},
        "COP":   {"pe_ratio": 11.0, "debt_to_equity": 0.30, "revenue_growth_pct":  -8, "health_score":  0.10, "volatility": "high"},
        "JPM":   {"pe_ratio": 12.0, "debt_to_equity": 1.20, "revenue_growth_pct":   8, "health_score":  0.55, "volatility": "medium"},
        "BAC":   {"pe_ratio": 11.5, "debt_to_equity": 1.10, "revenue_growth_pct":   5, "health_score":  0.45, "volatility": "medium"},
        "GS":    {"pe_ratio": 13.5, "debt_to_equity": 2.50, "revenue_growth_pct":  10, "health_score":  0.50, "volatility": "high"},
    }
    results = {}
    for ticker in tickers:
        data = stub_data.get(ticker.upper(), {
            "pe_ratio": 20.0, "debt_to_equity": 0.50, "revenue_growth_pct": 5,
            "health_score": 0.0, "volatility": "medium"
        })
        data["source"] = "stub — replace with SimFin/Alpha Vantage"
        results[ticker] = data
    return results


@tool
def get_stock_momentum_signal(tickers: list[str]) -> dict:
    """Signal G: Recent price trend for individual stocks.

    Price action only — how much each stock has moved in the last 5, 10, 30 days.
    Low weight signal (15%) — reflects what already happened, not a forecast.

    Args:
        tickers: List of stock tickers.

    Returns:
        Dict mapping ticker to price change percentages and momentum score.
    """
    # STUB — replace with yfinance per-stock price history
    stub_data = {
        "NVDA":  {"change_5d_pct":  4.2, "change_10d_pct":  7.5, "change_30d_pct": 15.3, "momentum_score":  0.80, "trend": "strong uptrend"},
        "MSFT":  {"change_5d_pct":  1.5, "change_10d_pct":  2.8, "change_30d_pct":  4.1, "momentum_score":  0.45, "trend": "uptrend"},
        "AAPL":  {"change_5d_pct":  0.8, "change_10d_pct":  1.2, "change_30d_pct":  2.0, "momentum_score":  0.20, "trend": "slight uptrend"},
        "UNH":   {"change_5d_pct": -1.0, "change_10d_pct": -2.1, "change_30d_pct": -3.5, "momentum_score": -0.25, "trend": "downtrend"},
        "JNJ":   {"change_5d_pct": -0.5, "change_10d_pct": -0.8, "change_30d_pct": -1.5, "momentum_score": -0.10, "trend": "flat"},
        "LLY":   {"change_5d_pct":  3.0, "change_10d_pct":  5.5, "change_30d_pct":  9.8, "momentum_score":  0.65, "trend": "uptrend"},
        "XOM":   {"change_5d_pct": -0.8, "change_10d_pct": -1.5, "change_30d_pct": -3.0, "momentum_score": -0.20, "trend": "downtrend"},
        "CVX":   {"change_5d_pct": -0.5, "change_10d_pct": -1.2, "change_30d_pct": -2.5, "momentum_score": -0.15, "trend": "slight downtrend"},
        "COP":   {"change_5d_pct": -1.2, "change_10d_pct": -2.8, "change_30d_pct": -5.1, "momentum_score": -0.40, "trend": "downtrend"},
        "JPM":   {"change_5d_pct":  1.2, "change_10d_pct":  2.0, "change_30d_pct":  3.8, "momentum_score":  0.35, "trend": "uptrend"},
        "BAC":   {"change_5d_pct":  0.9, "change_10d_pct":  1.5, "change_30d_pct":  2.5, "momentum_score":  0.25, "trend": "uptrend"},
        "GS":    {"change_5d_pct":  1.5, "change_10d_pct":  2.5, "change_30d_pct":  4.2, "momentum_score":  0.40, "trend": "uptrend"},
    }
    results = {}
    for ticker in tickers:
        data = stub_data.get(ticker.upper(), {
            "change_5d_pct": 0.0, "change_10d_pct": 0.0, "change_30d_pct": 0.0,
            "momentum_score": 0.0, "trend": "flat"
        })
        data["source"] = "stub — replace with yfinance"
        results[ticker] = data
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
