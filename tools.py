"""
tools.py — Stub tools for Signals A-D.

Each tool returns fake data for now

The agent calls these tools via LangChain's @tool decorator.
The function signatures define what the agent passes in.
"""

from langchain_core.tools import tool


@tool
def get_sentiment_signal(sectors: list[str]) -> dict:
    """Signal A: Get news sentiment scores for given sectors using FinBERT.

    Args:
        sectors: List of sector names (e.g. ["tech", "healthcare", "energy"])

    Returns:
        Dict mapping sector to sentiment score (-1.0 to 1.0) and sample headlines.
    """
    # STUB — replace with real FinBERT inference on live headlines
    stub_scores = {
        "tech": 0.74,
        "healthcare": 0.31,
        "energy": -0.12,
        "financials": 0.45,
        "consumer": 0.22,
        "industrials": 0.15,
        "real_estate": -0.08,
        "utilities": 0.05,
    }

    results = {}
    for sector in sectors:
        score = stub_scores.get(sector.lower(), 0.0)
        results[sector] = {
            "sentiment_score": score,
            "label": "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral",
            "headline_count": 25,
            "sample_headlines": [
                f"[STUB] {sector} sector shows mixed signals",
                f"[STUB] Analysts divided on {sector} outlook",
            ],
            "source": "stub_data — replace with NewsAPI + FinBERT",
        }
    return results


@tool
def get_fundamentals_signal(sectors: list[str]) -> dict:
    """Signal B: Get fundamental financial metrics for given sectors.

    Args:
        sectors: List of sector names to evaluate.

    Returns:
        Dict mapping sector to P/E ratio, debt-to-equity, revenue growth, etc.
    """
    # STUB — replace with SimFin or Alpha Vantage API calls
    stub_data = {
        "tech": {"pe_ratio": 28.5, "debt_to_equity": 0.45, "revenue_growth": 0.12, "health": "stretched"},
        "healthcare": {"pe_ratio": 18.2, "debt_to_equity": 0.35, "revenue_growth": 0.08, "health": "healthy"},
        "energy": {"pe_ratio": 12.1, "debt_to_equity": 0.60, "revenue_growth": -0.03, "health": "mixed"},
        "financials": {"pe_ratio": 15.0, "debt_to_equity": 0.80, "revenue_growth": 0.05, "health": "fair"},
    }

    results = {}
    for sector in sectors:
        data = stub_data.get(sector.lower(), {
            "pe_ratio": 20.0, "debt_to_equity": 0.50, "revenue_growth": 0.05, "health": "unknown"
        })
        data["source"] = "stub_data — replace with SimFin/Alpha Vantage"
        results[sector] = data
    return results


@tool
def get_macro_signal() -> dict:
    """Signal C: Get current macroeconomic environment from FRED.

    Returns:
        Dict with interest rates, inflation, unemployment, and interpretation.
    """
    # STUB — replace with FRED API calls
    return {
        "fed_funds_rate": 5.25,
        "cpi_yoy": 3.2,
        "unemployment_rate": 3.8,
        "interpretation": "Rates are elevated, inflation cooling but above target." "Environment pressures growth stocks, favors value and bonds.",
        "source": "stub_data — replace with FRED API",
    }


@tool
def get_momentum_signal(sectors: list[str]) -> dict:
    """Signal D: Get momentum/technical signals using moving averages on sector ETFs.

    Args:
        sectors: List of sector names to check momentum for.

    Returns:
        Dict mapping sector to 30-day vs 90-day moving average comparison.
    """
    # STUB — replace with yfinance calls on sector ETFs
    # (XLK for tech, XLV for healthcare, XLE for energy, etc.)
    stub_data = {
        "tech": {"etf": "XLK", "ma_30": 215.40, "ma_90": 210.20, "trend": "uptrend", "strength": "moderate"},
        "healthcare": {"etf": "XLV", "ma_30": 142.10, "ma_90": 143.50, "trend": "downtrend", "strength": "weak"},
        "energy": {"etf": "XLE", "ma_30": 88.30, "ma_90": 90.10, "trend": "downtrend", "strength": "moderate"},
        "financials": {"etf": "XLF", "ma_30": 41.20, "ma_90": 40.80, "trend": "uptrend", "strength": "weak"},
    }

    results = {}
    for sector in sectors:
        data = stub_data.get(sector.lower(), {
            "etf": "N/A", "ma_30": 100.0, "ma_90": 100.0, "trend": "flat", "strength": "none"
        })
        data["source"] = "stub_data — replace with yfinance"
        results[sector] = data
    return results


# Collect all tools for easy import
ALL_TOOLS = [
    get_sentiment_signal,
    get_fundamentals_signal,
    get_macro_signal,
    get_momentum_signal,
]