"""FinBERT model construction with partial encoder-layer freezing."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config

from transformers import AutoModelForSequenceClassification


def freeze_encoder_layers(model: AutoModelForSequenceClassification, n_layers_to_freeze: int) -> None:
    """Freeze the embedding layer and the first n_layers_to_freeze encoder layers.

    Layers (n_layers_to_freeze) through 11 remain trainable. The pooler and
    classifier head are always trainable.

    Args:
        model: Loaded HuggingFace sequence-classification model.
        n_layers_to_freeze: Number of encoder layers to freeze (0-indexed).
    """
    # Freeze embeddings
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    # Freeze encoder layers 0 … (n_layers_to_freeze - 1)
    for i in range(n_layers_to_freeze):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False

    # Print summary
    print("\n" + "=" * 55)
    print("Layer Freeze Summary")
    print("=" * 55)
    print(f"  Embeddings      : FROZEN")
    total_layers = len(model.bert.encoder.layer)
    for i in range(total_layers):
        status = "FROZEN" if i < n_layers_to_freeze else "trainable"
        print(f"  Encoder layer {i:>2} : {status}")
    print(f"  Pooler          : trainable")
    print(f"  Classifier head : trainable")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Trainable params: {trainable:,} / {total:,}  "
          f"({100.0 * trainable / total:.1f}%)")
    print("=" * 55 + "\n")


def build_model() -> AutoModelForSequenceClassification:
    """Load the base FinBERT model and apply layer freezing per config.

    Returns:
        Model ready for fine-tuning.
    """
    print(f"Loading base model: {config.MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )

    freeze_encoder_layers(model, config.FREEZE_LAYERS)
    return model
