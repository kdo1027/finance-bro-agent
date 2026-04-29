"""Central configuration for the FinBERT sector sentiment scorer.

All hyperparameters and paths live here. Nothing is hardcoded elsewhere.
"""

import os as _os
from dataclasses import dataclass, field
from typing import Dict

# Resolves relative to this file regardless of the caller's working directory
_HERE = _os.path.dirname(_os.path.abspath(__file__))


@dataclass(frozen=True)
class Config:
    # Model identity
    MODEL_NAME: str = "ProsusAI/finbert"
    NUM_LABELS: int = 3
    LABEL2ID: Dict[str, int] = field(
        default_factory=lambda: {"positive": 0, "negative": 1, "neutral": 2}
    )
    ID2LABEL: Dict[int, str] = field(
        default_factory=lambda: {0: "positive", 1: "negative", 2: "neutral"}
    )

    # Tokenization
    MAX_LENGTH: int = 128

    # Training
    BATCH_SIZE: int = 16
    NUM_EPOCHS: int = 4
    LEARNING_RATE: float = 2e-5
    WARMUP_RATIO: float = 0.1
    WEIGHT_DECAY: float = 0.01

    # Layer freezing — freeze encoder layers 0 through (FREEZE_LAYERS - 1)
    FREEZE_LAYERS: int = 8

    # Data splits
    TRAIN_SPLIT: float = 0.8
    VAL_SPLIT: float = 0.1
    TEST_SPLIT: float = 0.1

    # Reproducibility
    RANDOM_SEED: int = 42

    # Paths — absolute so inference works regardless of caller's working directory
    MODEL_SAVE_PATH: str = _os.path.join(_HERE, "models", "finbert_finetuned")

    # Inference
    RECENCY_DECAY_HOURS: float = 24.0  # half-life for recency weighting


# Singleton instance used project-wide
config = Config()
