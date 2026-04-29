"""Adapter that wires the local finbert_scorer package into finance-bro-agent.

Handles sys.path setup so finbert_scorer's internal imports resolve correctly,
then exposes two functions used by tools.py:
  - get_sector_sentiments()  → used by Signal A
  - get_stock_sentiments()   → used by Signal E
"""

import os
import sys

# Add finbert_scorer/ to sys.path so inference.py can resolve its own imports
_FINBERT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finbert_scorer")
if _FINBERT_DIR not in sys.path:
    sys.path.insert(0, _FINBERT_DIR)

from src.inference import score_headlines, score_sectors  # noqa: E402

# finance-bro calls this sector "tech"; FinBERT's sector router expects "technology"
_SECTOR_MAP = {"tech": "technology"}


def _to_finbert_sector(sector: str) -> str:
    return _SECTOR_MAP.get(sector, sector)


def _to_finance_score(positive_prob: float) -> float:
    """Convert FinBERT positive probability (0–1) to finance-bro score (–1 to 1).

    0.5 positive prob → 0.0 (neutral), 1.0 → 1.0, 0.0 → –1.0
    """
    return round((positive_prob - 0.5) * 2, 4)


def get_sector_sentiments(
    sectors: list[str],
    headlines_with_timestamps: list[dict],
) -> dict[str, float]:
    """Run FinBERT on timestamped headlines and return per-sector scores (–1 to 1).

    Args:
        sectors: Sector names in finance-bro naming (e.g. "tech", "healthcare").
        headlines_with_timestamps: List of {"text": str, "timestamp": ISO-8601 str}.

    Returns:
        {sector: score} — score is –1.0 (bearish) to 1.0 (bullish), 0.0 if no headlines routed.
    """
    finbert_sectors = [_to_finbert_sector(s) for s in sectors]
    result = score_sectors(headlines_with_timestamps, sectors_to_score=finbert_sectors)

    output = {}
    for sector, fb_sector in zip(sectors, finbert_sectors):
        prob = result["sector_sentiment"].get(fb_sector)
        output[sector] = _to_finance_score(prob) if prob is not None else 0.0
    return output


def get_stock_sentiments(ticker_headlines: dict[str, list[str]]) -> dict[str, float]:
    """Run FinBERT on per-ticker headlines and return sentiment scores (–1 to 1).

    Args:
        ticker_headlines: {ticker: [headline_str, ...]}

    Returns:
        {ticker: score} — score is –1.0 (bearish) to 1.0 (bullish), 0.0 if no headlines.
    """
    output = {}
    for ticker, headlines in ticker_headlines.items():
        if not headlines:
            output[ticker] = 0.0
            continue
        scored = score_headlines(headlines)
        avg_positive = sum(h["positive"] for h in scored) / len(scored)
        output[ticker] = _to_finance_score(avg_positive)
    return output
