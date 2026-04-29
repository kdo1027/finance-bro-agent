"""Runtime inference module for the FinBERT sector sentiment scorer.

This module is fully self-contained and importable by an agent. The model is
cached in memory after the first load so repeated calls do not reload from disk.

Public API
----------
score_sectors(headlines, sectors_to_score) → dict
    Aggregate sector-level sentiment with recency weighting.

score_headlines(headlines) → list[dict]
    Per-headline probabilities and predicted labels.
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .sector_router import route_batch, get_unrouted_count, reset_unrouted_count


# ---------------------------------------------------------------------------
# Module-level model cache — populated on first call
# ---------------------------------------------------------------------------
_cached_model: Optional[AutoModelForSequenceClassification] = None
_cached_tokenizer: Optional[AutoTokenizer] = None
_cached_device: Optional[torch.device] = None


def _get_device() -> torch.device:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model() -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, torch.device]:
    """Load (or return cached) fine-tuned model and tokenizer.

    Returns:
        Tuple of (model, tokenizer, device).

    Raises:
        FileNotFoundError: If the model directory does not exist, with a
            clear message instructing the user to run train_pipeline.py.
    """
    global _cached_model, _cached_tokenizer, _cached_device

    if _cached_model is not None:
        return _cached_model, _cached_tokenizer, _cached_device

    save_path = config.MODEL_SAVE_PATH
    if not os.path.isdir(save_path):
        raise FileNotFoundError(
            f"Fine-tuned model not found at '{save_path}'.\n"
            "Run the training pipeline first:\n"
            "    python train_pipeline.py"
        )

    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model = AutoModelForSequenceClassification.from_pretrained(save_path)
    model.to(device)
    model.eval()

    _cached_model    = model
    _cached_tokenizer = tokenizer
    _cached_device   = device

    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Low-level inference
# ---------------------------------------------------------------------------

def _run_inference(
    texts: List[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> List[Dict[str, float]]:
    """Tokenize texts and return per-text softmax probability dicts.

    Args:
        texts: List of raw strings.
        model: Loaded model on device.
        tokenizer: Matching tokenizer.
        device: Compute device.

    Returns:
        List of dicts with keys: positive, negative, neutral (float 0-1).
    """
    results: List[Dict[str, float]] = []

    for batch_start in range(0, len(texts), config.BATCH_SIZE):
        batch_texts = texts[batch_start: batch_start + config.BATCH_SIZE]
        encoding = tokenizer(
            batch_texts,
            max_length=config.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = model(**encoding).logits

        probs = F.softmax(logits, dim=-1).cpu().tolist()
        for p in probs:
            results.append({
                config.ID2LABEL[i]: p[i] for i in range(config.NUM_LABELS)
            })

    return results


# ---------------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------------

def _recency_weight(timestamp_str: str, now: datetime) -> float:
    """Compute exponential recency weight for a headline.

    Weight = exp(-lambda * age_hours) where lambda = ln(2) / half_life.
    This gives a weight of 1.0 for a brand-new headline and 0.5 at exactly
    RECENCY_DECAY_HOURS hours old.

    Args:
        timestamp_str: ISO 8601 timestamp string (e.g. "2024-01-15T10:30:00Z").
        now: Current UTC datetime (timezone-aware).

    Returns:
        Float weight in (0, 1].
    """
    try:
        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_hours = max((now - ts).total_seconds() / 3600.0, 0.0)
    except (ValueError, TypeError):
        age_hours = 0.0

    lam = math.log(2) / config.RECENCY_DECAY_HOURS
    return math.exp(-lam * age_hours)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_headlines(headlines: List[str]) -> List[Dict]:
    """Return per-headline sentiment probabilities and predicted label.

    Args:
        headlines: List of raw headline strings.

    Returns:
        List of dicts, one per headline:
            text, positive, negative, neutral, predicted_label.
    """
    model, tokenizer, device = _load_model()
    probs = _run_inference(headlines, model, tokenizer, device)

    output = []
    for text, p in zip(headlines, probs):
        predicted_label = max(p, key=p.get)
        output.append({
            "text": text,
            "positive": round(p["positive"], 6),
            "negative": round(p["negative"], 6),
            "neutral":  round(p["neutral"],  6),
            "predicted_label": predicted_label,
        })
    return output


def score_sectors(
    headlines: List[Dict],
    sectors_to_score: Optional[List[str]] = None,
) -> Dict:
    """Score sector-level sentiment from a list of timestamped headlines.

    Args:
        headlines: List of dicts, each with:
            "text"      (str)  — raw headline string
            "timestamp" (str)  — ISO 8601 datetime string
        sectors_to_score: Optional list of sector names to include in output.
            If None, all known sectors plus 'unknown' are included.

    Returns:
        Dict with keys:
            sector_sentiment  — {sector: weighted_avg_positive | None}
            top_sector        — sector with highest positive score (or None)
            bottom_sector     — sector with lowest positive score (or None)
            headline_counts   — {sector: int}
            unrouted_count    — int
            model_version     — str
    """
    all_sectors = [
        "technology", "energy", "healthcare", "financials",
        "consumer", "industrials", "real_estate", "bonds",
        "international",
    ]
    if sectors_to_score is None:
        sectors_to_score = all_sectors

    model, tokenizer, device = _load_model()

    texts = [h["text"] for h in headlines]
    timestamps = [h.get("timestamp", "") for h in headlines]

    # Route headlines to sectors
    reset_unrouted_count()
    sectors_assigned = route_batch(texts)
    unrouted = get_unrouted_count()

    # Run model inference
    probs = _run_inference(texts, model, tokenizer, device)

    now = datetime.now(tz=timezone.utc)

    # Collect per-sector weighted positive probabilities
    sector_data: Dict[str, Tuple[float, float]] = {s: (0.0, 0.0) for s in sectors_to_score}
    # sector_data[sector] = (weighted_sum_positive, weight_sum)
    headline_counts: Dict[str, int] = {s: 0 for s in sectors_to_score}

    for sector, prob_dict, ts in zip(sectors_assigned, probs, timestamps):
        if sector not in sectors_to_score:
            continue
        w = _recency_weight(ts, now)
        wsum, wcount = sector_data[sector]
        sector_data[sector] = (wsum + prob_dict["positive"] * w, wcount + w)
        headline_counts[sector] += 1

    # Compute final scores; return None where no headlines routed
    sector_sentiment: Dict[str, Optional[float]] = {}
    for sector in sectors_to_score:
        wsum, wcount = sector_data[sector]
        if wcount == 0.0:
            sector_sentiment[sector] = None
        else:
            sector_sentiment[sector] = round(wsum / wcount, 6)

    # Top / bottom sector (among sectors with data)
    scored = {s: v for s, v in sector_sentiment.items() if v is not None}
    top_sector    = max(scored, key=scored.get) if scored else None
    bottom_sector = min(scored, key=scored.get) if scored else None

    return {
        "sector_sentiment":  sector_sentiment,
        "top_sector":        top_sector,
        "bottom_sector":     bottom_sector,
        "headline_counts":   headline_counts,
        "unrouted_count":    unrouted,
        "model_version":     "finbert_finetuned",
    }
