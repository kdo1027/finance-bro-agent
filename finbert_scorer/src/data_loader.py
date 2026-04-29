"""Data loading, remapping, combination, and tokenization for FinBERT training.

Combines two HuggingFace datasets:
  1. takala/financial_phrasebank  (sentences_75agree)
  2. zeroshot/twitter-financial-news-sentiment

Both are remapped to the canonical label scheme:
  positive → 0, negative → 1, neutral → 2
"""

from __future__ import annotations

import random
import zipfile
from typing import Dict, List, Tuple

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Resolve config relative to project root regardless of cwd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


# ---------------------------------------------------------------------------
# Label remapping helpers
# ---------------------------------------------------------------------------

def _remap_twitter_label(original_label: int) -> int:
    """Remap Twitter Financial News Sentiment label to canonical scheme.

    The zeroshot dataset uses:
        0 → Bearish, 1 → Bullish, 2 → Neutral
    We remap to: positive=0, negative=1, neutral=2.

    Args:
        original_label: Raw integer label from the dataset.

    Returns:
        Remapped integer label.
    """
    mapping = {
        0: 1,  # Bearish  → negative (1)
        1: 0,  # Bullish  → positive (0)
        2: 2,  # Neutral  → neutral  (2)
    }
    return mapping[original_label]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_phrasebank() -> List[Dict]:
    """Load Financial PhraseBank by downloading the raw ZIP from HuggingFace.

    Uses huggingface_hub.hf_hub_download to fetch
    data/FinancialPhraseBank-v1.0.zip directly from takala/financial_phrasebank,
    then parses FinancialPhraseBank-v1.0/Sentences_75Agree.txt (iso-8859-1,
    'sentence@label' format). This bypasses the legacy dataset script that is
    unsupported in datasets>=3.0.

    Returns:
        List of {"text": str, "label": int} dicts, or [] on failure.
    """
    try:
        zip_path = hf_hub_download(
            repo_id="takala/financial_phrasebank",
            filename="data/FinancialPhraseBank-v1.0.zip",
            repo_type="dataset",
        )

        target = "FinancialPhraseBank-v1.0/Sentences_75Agree.txt"
        records = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(target) as f:
                content = f.read().decode("iso-8859-1")
            for line in content.strip().splitlines():
                if "@" not in line:
                    continue
                sentence, label_str = line.rsplit("@", 1)
                label_str = label_str.strip().lower()
                if label_str in config.LABEL2ID:
                    records.append({
                        "text": sentence.strip(),
                        "label": config.LABEL2ID[label_str],
                    })

        print(f"  Financial PhraseBank: {len(records)} records loaded")
        return records

    except Exception as e:
        print(
            f"  WARNING: Financial PhraseBank could not be loaded "
            f"({type(e).__name__}: {e})\n"
            f"  Continuing with Twitter Financial News only."
        )
        return []


def _load_twitter_news() -> List[Dict]:
    """Load and remap the Twitter Financial News Sentiment dataset.

    Returns:
        List of {"text": str, "label": int} dicts.
    """
    from datasets import load_dataset
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    records = []
    for split_name in ds:
        for row in ds[split_name]:
            records.append({
                "text": row["text"],
                "label": _remap_twitter_label(row["label"]),
            })
    print(f"  Twitter Financial News: {len(records)} records loaded")
    return records


# ---------------------------------------------------------------------------
# Splitting and tokenization
# ---------------------------------------------------------------------------

def _split_records(
    records: List[Dict],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Shuffle and split records into train/val/test.

    Args:
        records: Combined list of {"text", "label"} dicts.

    Returns:
        Tuple of (train, val, test) record lists.
    """
    rng = random.Random(config.RANDOM_SEED)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * config.TRAIN_SPLIT)
    val_end = train_end + int(n * config.VAL_SPLIT)

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def _class_distribution(records: List[Dict]) -> Dict[str, int]:
    """Count instances per class in a record list."""
    counts: Dict[int, int] = {0: 0, 1: 0, 2: 0}
    for r in records:
        counts[r["label"]] += 1
    return {config.ID2LABEL[k]: v for k, v in counts.items()}


def _print_dataset_summary(
    train: List[Dict],
    val: List[Dict],
    test: List[Dict],
) -> None:
    """Print a human-readable dataset summary."""
    total = len(train) + len(val) + len(test)
    print("\n" + "=" * 55)
    print("Dataset Summary")
    print("=" * 55)
    print(f"  Total records : {total}")
    print(f"  Train         : {len(train)}")
    print(f"  Val           : {len(val)}")
    print(f"  Test          : {len(test)}")
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        dist = _class_distribution(split)
        print(f"\n  {name} class distribution:")
        for label, count in dist.items():
            pct = 100.0 * count / max(len(split), 1)
            print(f"    {label:<10}: {count:>5}  ({pct:.1f}%)")
    print("=" * 55 + "\n")


def _tokenize_records(
    records: List[Dict],
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """Tokenize a list of records and return a HuggingFace Dataset.

    Args:
        records: List of {"text", "label"} dicts.
        tokenizer: Tokenizer instance.

    Returns:
        Dataset with columns: input_ids, attention_mask, label.
    """
    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]

    encoding = tokenizer(
        texts,
        max_length=config.MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors=None,  # return plain lists; Dataset handles conversion
    )

    return Dataset.from_dict({
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "label": labels,
        "text": texts,
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_records_by_source() -> Tuple[List[Dict], List[Dict]]:
    """Load both datasets as raw records without tokenizing or splitting.

    Used by visualize.py to plot dataset composition. Each record is a
    {"text": str, "label": int} dict with canonical label scheme.

    Returns:
        Tuple of (phrasebank_records, twitter_records).
    """
    return _load_phrasebank(), _load_twitter_news()


def load_and_prepare_data() -> Tuple[DatasetDict, PreTrainedTokenizerBase]:
    """Load, combine, split, and tokenize both financial sentiment datasets.

    Returns:
        Tuple of (DatasetDict with keys 'train'/'val'/'test', tokenizer).
    """
    print("Loading datasets...")
    phrasebank = _load_phrasebank()
    twitter = _load_twitter_news()

    combined = phrasebank + twitter
    print(f"  Combined total: {len(combined)} records")

    train_raw, val_raw, test_raw = _split_records(combined)
    _print_dataset_summary(train_raw, val_raw, test_raw)

    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    dataset_dict = DatasetDict({
        "train": _tokenize_records(train_raw, tokenizer),
        "val":   _tokenize_records(val_raw, tokenizer),
        "test":  _tokenize_records(test_raw, tokenizer),
    })
    print("  Tokenization complete.\n")

    return dataset_dict, tokenizer
