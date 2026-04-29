"""Sector router: maps financial headlines to one of ten predefined sectors.

Primary method is keyword matching (case-insensitive). Ties are broken by
keyword-match count; the sector with the most matches wins. Headlines with
zero matches are flagged as 'unknown' and counted for the session.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List

# ---------------------------------------------------------------------------
# Keyword dictionary — extend as needed, values are compiled lazily
# ---------------------------------------------------------------------------
_SECTOR_KEYWORDS: Dict[str, List[str]] = {
    "technology": [
        "NVIDIA", "AMD", "Intel", "Apple", "Microsoft", "Google", "Alphabet",
        "Meta", "Amazon", "Tesla", "semiconductor", "AI", "cloud", "software",
        "chip", "cybersecurity", "data center", "artificial intelligence",
        "machine learning", "SaaS", "tech", "technology",
    ],
    "energy": [
        "oil", "crude", "OPEC", "natural gas", "ExxonMobil", "Chevron", "BP",
        "Shell", "pipeline", "refinery", "renewable", "solar", "wind",
        "lithium", "energy", "petroleum", "LNG", "coal", "nuclear",
    ],
    "healthcare": [
        "FDA", "Pfizer", "Moderna", "Johnson", "pharma", "drug",
        "clinical trial", "biotech", "hospital", "Medicare", "insurance",
        "vaccine", "healthcare", "health", "medical", "therapeutics",
        "pharmaceutical", "Merck", "AbbVie", "Eli Lilly",
    ],
    "financials": [
        "Federal Reserve", "Fed", "interest rate", "bank", "JPMorgan",
        "Goldman", "Morgan Stanley", "inflation", "CPI", "bond yield",
        "lending", "credit", "financial", "banking", "Wells Fargo",
        "Citigroup", "hedge fund", "private equity", "FDIC",
    ],
    "consumer": [
        "retail", "consumer spending", "Amazon", "Walmart", "Target",
        "e-commerce", "spending", "discretionary", "staples", "consumer",
        "Nike", "McDonald's", "Starbucks", "luxury", "apparel",
    ],
    "industrials": [
        "manufacturing", "Boeing", "Caterpillar", "defense", "aerospace",
        "supply chain", "logistics", "freight", "industrial", "Lockheed",
        "Raytheon", "3M", "Honeywell", "GE", "factory", "automation",
    ],
    "real_estate": [
        "housing", "mortgage", "REIT", "commercial real estate", "rent",
        "property", "real estate", "home sales", "construction", "landlord",
        "apartment", "office space", "vacancy",
    ],
    "bonds": [
        "treasury", "yield curve", "10-year", "bond", "fixed income",
        "duration", "sovereign debt", "T-bill", "coupon", "maturity",
        "investment grade", "junk bond", "credit spread",
    ],
    "international": [
        "China", "Europe", "Japan", "emerging markets", "trade war",
        "tariff", "currency", "forex", "yuan", "euro", "yen", "BRICS",
        "IMF", "World Bank", "geopolitical", "sanctions",
    ],
}

# Session-level counter for unrouted headlines
_unrouted_count: int = 0


def _compile_patterns() -> Dict[str, List[re.Pattern]]:
    """Compile keyword patterns once at module load."""
    compiled: Dict[str, List[re.Pattern]] = {}
    for sector, keywords in _SECTOR_KEYWORDS.items():
        # Use word-boundary matching; sort longest first to avoid partial shadowing
        patterns = []
        for kw in sorted(keywords, key=len, reverse=True):
            escaped = re.escape(kw)
            patterns.append(re.compile(r"\b" + escaped + r"\b", re.IGNORECASE))
        compiled[sector] = patterns
    return compiled


_PATTERNS: Dict[str, List[re.Pattern]] = _compile_patterns()


def get_unrouted_count() -> int:
    """Return the number of headlines that could not be routed this session."""
    return _unrouted_count


def reset_unrouted_count() -> None:
    """Reset the session-level unrouted counter to zero."""
    global _unrouted_count
    _unrouted_count = 0


def route_headline(headline: str) -> str:
    """Route a single headline string to its most likely sector.

    Args:
        headline: Raw headline text.

    Returns:
        Sector name string. Returns 'unknown' if no keywords match, and
        increments the session-level unrouted counter.
    """
    global _unrouted_count

    match_counts: Dict[str, int] = defaultdict(int)
    for sector, patterns in _PATTERNS.items():
        for pattern in patterns:
            if pattern.search(headline):
                match_counts[sector] += 1

    if not match_counts:
        _unrouted_count += 1
        return "unknown"

    # Sector with the most keyword hits wins
    return max(match_counts, key=lambda s: match_counts[s])


def route_batch(headlines: List[str]) -> List[str]:
    """Route a list of headlines to sectors.

    Args:
        headlines: List of raw headline strings.

    Returns:
        List of sector name strings, one per headline.
    """
    return [route_headline(h) for h in headlines]
