"""Standalone demo of the FinBERT sector sentiment inference interface.

Run with:
    python inference_demo.py

Loads the fine-tuned model and scores a hardcoded set of sample headlines
spanning the last 48 hours to demonstrate recency weighting. Prints the
full sector vector output.

Prerequisites:
    The fine-tuned model must exist at config.MODEL_SAVE_PATH.
    Run `python train_pipeline.py` first if it does not.
"""

import sys
import os
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(__file__))

from src.inference import score_sectors, score_headlines


def _make_timestamp(hours_ago: float) -> str:
    """Return an ISO 8601 UTC timestamp string for `hours_ago` hours in the past."""
    dt = datetime.now(tz=timezone.utc) - timedelta(hours=hours_ago)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Sample headlines with realistic recency spread (0 – 48 hours old)
# ---------------------------------------------------------------------------
SAMPLE_HEADLINES = [
    # Technology
    {
        "text": "NVIDIA beats earnings expectations on booming AI chip demand",
        "timestamp": _make_timestamp(1.0),
    },
    {
        "text": "Microsoft Azure cloud revenue grows 28% year-over-year",
        "timestamp": _make_timestamp(3.5),
    },
    {
        "text": "Apple unveils next-generation M4 chip with enhanced machine learning cores",
        "timestamp": _make_timestamp(8.0),
    },
    {
        "text": "AMD loses market share to NVIDIA in data center GPU segment",
        "timestamp": _make_timestamp(12.0),
    },
    {
        "text": "Google faces antitrust probe over cloud software bundling practices",
        "timestamp": _make_timestamp(24.0),
    },
    # Energy
    {
        "text": "OPEC+ agrees to production cuts as crude oil prices slide",
        "timestamp": _make_timestamp(2.0),
    },
    {
        "text": "ExxonMobil reports record refinery utilization amid strong demand",
        "timestamp": _make_timestamp(6.0),
    },
    {
        "text": "Solar panel installations hit all-time high in Q3, driven by IRA incentives",
        "timestamp": _make_timestamp(18.0),
    },
    {
        "text": "Natural gas futures plunge on warmer-than-expected winter forecasts",
        "timestamp": _make_timestamp(36.0),
    },
    # Financials
    {
        "text": "Federal Reserve signals two additional rate cuts before year-end",
        "timestamp": _make_timestamp(0.5),
    },
    {
        "text": "JPMorgan posts stronger-than-expected lending revenue amid rate environment",
        "timestamp": _make_timestamp(5.0),
    },
    {
        "text": "CPI inflation cools to 2.3%, below consensus estimates",
        "timestamp": _make_timestamp(10.0),
    },
    # Healthcare
    {
        "text": "Pfizer receives FDA approval for novel RSV vaccine targeting adults over 60",
        "timestamp": _make_timestamp(4.0),
    },
    {
        "text": "Moderna's mRNA cancer drug shows 44% reduction in tumor recurrence in trials",
        "timestamp": _make_timestamp(20.0),
    },
    {
        "text": "Hospital chains warn of Medicare reimbursement cuts in proposed budget",
        "timestamp": _make_timestamp(48.0),
    },
    # Consumer
    {
        "text": "Walmart raises full-year guidance as consumer spending holds up better than feared",
        "timestamp": _make_timestamp(7.0),
    },
    {
        "text": "Target inventory glut forces steep markdowns across discretionary categories",
        "timestamp": _make_timestamp(30.0),
    },
    # Industrials
    {
        "text": "Boeing 737 MAX production ramp-up delayed by supply chain bottlenecks",
        "timestamp": _make_timestamp(9.0),
    },
    {
        "text": "Caterpillar construction equipment orders surge on infrastructure spending",
        "timestamp": _make_timestamp(15.0),
    },
    # Real estate
    {
        "text": "US home sales drop 8% as mortgage rates remain elevated near 7%",
        "timestamp": _make_timestamp(11.0),
    },
    {
        "text": "Commercial real estate office vacancy rates reach record high in major cities",
        "timestamp": _make_timestamp(40.0),
    },
    # Bonds
    {
        "text": "10-year treasury yield climbs to 4.8% on strong jobs report",
        "timestamp": _make_timestamp(2.5),
    },
    {
        "text": "Yield curve inverts further as bond investors price in recession risk",
        "timestamp": _make_timestamp(22.0),
    },
    # International
    {
        "text": "US-China trade tensions escalate as tariffs extended to tech components",
        "timestamp": _make_timestamp(16.0),
    },
    {
        "text": "IMF raises eurozone growth forecast amid easing energy prices",
        "timestamp": _make_timestamp(32.0),
    },
    # Intentionally ambiguous — likely routes to 'unknown'
    {
        "text": "Markets brace for volatile session ahead of key policy announcements",
        "timestamp": _make_timestamp(1.5),
    },
]


def _print_sector_vector(result: dict) -> None:
    """Pretty-print the score_sectors output."""
    print("\n" + "=" * 60)
    print("Sector Sentiment Vector")
    print("=" * 60)
    sentiments = result["sector_sentiment"]
    counts = result["headline_counts"]

    max_sector_len = max(len(s) for s in sentiments)
    for sector, score in sorted(
        sentiments.items(),
        key=lambda kv: (kv[1] is None, -(kv[1] or 0)),
    ):
        bar = ""
        if score is not None:
            filled = int(score * 30)
            bar = " [" + "█" * filled + "░" * (30 - filled) + f"] {score:.4f}"
        else:
            bar = " [no data]"
        headline_n = counts.get(sector, 0)
        print(f"  {sector.ljust(max_sector_len)}{bar}  ({headline_n} headlines)")

    print()
    print(f"  Top sector    : {result['top_sector']}")
    print(f"  Bottom sector : {result['bottom_sector']}")
    print(f"  Unrouted      : {result['unrouted_count']} headlines")
    print(f"  Model version : {result['model_version']}")
    print("=" * 60)


def _print_per_headline_results(results: list) -> None:
    """Pretty-print per-headline score_headlines output."""
    print("\n" + "=" * 60)
    print("Per-Headline Scores (sample of first 5)")
    print("=" * 60)
    for item in results[:5]:
        print(f"\n  Text      : {item['text'][:75]}...")
        print(f"  Positive  : {item['positive']:.4f}")
        print(f"  Negative  : {item['negative']:.4f}")
        print(f"  Neutral   : {item['neutral']:.4f}")
        print(f"  Predicted : {item['predicted_label']}")
    print("=" * 60)


def main() -> None:
    """Run the inference demo."""
    print("FinBERT Sector Sentiment — Inference Demo")
    print(f"Headlines: {len(SAMPLE_HEADLINES)}")

    # Per-headline scoring
    texts = [h["text"] for h in SAMPLE_HEADLINES]
    per_headline = score_headlines(texts)
    _print_per_headline_results(per_headline)

    # Sector-level scoring with recency weighting
    result = score_sectors(SAMPLE_HEADLINES)
    _print_sector_vector(result)


if __name__ == "__main__":
    main()
