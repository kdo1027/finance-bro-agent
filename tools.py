"""
tools.py — Stub tools for Signals A-D (sector level) and E-G (stock level).

Sector-level tools run first (Pass 1) to grade and filter sectors.
Stock-level tools run second (Pass 2) only for the top sectors that passed.

Each tool returns fake data for now — see inline comments for real API replacements.
"""

from langchain_core.tools import tool


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
    """Signal A: News sentiment scores for each sector.

    Args:
        sectors: List of sector names (e.g. ["tech", "healthcare", "energy"])

    Returns:
        Dict mapping sector to sentiment score (-1.0 to 1.0) and label.
        Score > 0.2 = positive, < -0.2 = negative, else neutral.
    """
    # STUB — replace with NewsAPI headlines + FinBERT inference
    stub_scores = {
        "tech":        0.74,
        "healthcare":  0.31,
        "energy":     -0.12,
        "financials":  0.45,
        "consumer":    0.22,
        "industrials": 0.15,
        "real_estate":-0.08,
        "utilities":   0.05,
    }
    results = {}
    for sector in sectors:
        score = stub_scores.get(sector.lower(), 0.0)
        results[sector] = {
            "sentiment_score": score,
            "label": "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral",
            "headline_count": 25,
            "source": "stub — replace with NewsAPI + FinBERT",
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
    """Signal E: News sentiment for individual stocks.

    Args:
        tickers: List of stock tickers (e.g. ["NVDA", "MSFT", "AAPL"])

    Returns:
        Dict mapping ticker to sentiment score and label.
    """
    # STUB — replace with NewsAPI stock-specific headlines + FinBERT
    stub_scores = {
        "NVDA":  0.82, "MSFT":  0.60, "AAPL":  0.45,
        "UNH":   0.38, "JNJ":   0.28, "LLY":   0.55,
        "XOM":  -0.15, "CVX":  -0.08, "COP":  -0.22,
        "JPM":   0.50, "BAC":   0.35, "GS":    0.40,
        "AMZN":  0.65, "HD":    0.30, "MCD":   0.20,
        "CAT":   0.45, "HON":   0.38, "UPS":   0.25,
    }
    results = {}
    for ticker in tickers:
        score = stub_scores.get(ticker.upper(), 0.0)
        results[ticker] = {
            "sentiment_score": score,
            "label": "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral",
            "source": "stub — replace with NewsAPI + FinBERT",
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
