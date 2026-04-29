"""End-to-end training pipeline.

Run with:
    python train_pipeline.py

Steps:
    1. Load and tokenize both financial sentiment datasets.
    2. Build the FinBERT model with frozen layers.
    3. Fine-tune with validation and best-checkpoint saving.
    4. Evaluate on the held-out test set (3-way table + all plots).
    5. Generate presentation visualizations (t-SNE, composition, sector bar).
"""

import json
import sys
import os

# Ensure imports resolve correctly regardless of launch directory
sys.path.insert(0, os.path.dirname(__file__))

import random
import numpy as np
import torch

from config import config
from src.data_loader import load_and_prepare_data
from src.model import build_model
from src.train import train
from src.evaluate import evaluate
from src.visualize import plot_dataset_composition, plot_tsne_embeddings, plot_sector_bar


def _seed_everything(seed: int) -> None:
    """Seed all random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Run the full training and evaluation pipeline."""
    print("=" * 60)
    print("FinBERT Sector Sentiment Fine-tuning Pipeline")
    print("=" * 60 + "\n")

    _seed_everything(config.RANDOM_SEED)

    # ── Step 1: Data ────────────────────────────────────────────────────
    print("Step 1/4 — Loading and preparing data")
    print("-" * 40)
    dataset_dict, tokenizer = load_and_prepare_data()

    # ── Step 2: Model ───────────────────────────────────────────────────
    print("Step 2/4 — Building model")
    print("-" * 40)
    model = build_model()

    # ── Step 3: Training ────────────────────────────────────────────────
    print("Step 3/4 — Training")
    print("-" * 40)
    history = train(model, tokenizer, dataset_dict)

    # ── Step 4: Evaluation ──────────────────────────────────────────────
    print("Step 4/5 — Evaluation")
    print("-" * 40)

    # Persist training history so it is available for future visualizations
    history_path = os.path.join(os.path.dirname(config.MODEL_SAVE_PATH), "training_history.json")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved to: {history_path}")

    evaluate(dataset_dict["test"], history=history)

    # ── Step 5: Presentation visualizations ─────────────────────────────
    print("Step 5/5 — Generating presentation visualizations")
    print("-" * 40)
    models_dir = os.path.dirname(config.MODEL_SAVE_PATH)

    try:
        plot_dataset_composition(save_dir=models_dir)
    except Exception as e:
        print(f"  WARNING: dataset composition chart failed — {e}")

    try:
        plot_tsne_embeddings(dataset_dict["test"], save_dir=models_dir)
    except Exception as e:
        print(f"  WARNING: t-SNE plot failed — {e}")

    try:
        # Run inference demo headlines through the trained model for sector bar
        from src.inference import score_sectors
        from inference_demo import SAMPLE_HEADLINES
        sector_result = score_sectors(SAMPLE_HEADLINES)
        plot_sector_bar(sector_result, save_dir=models_dir)
    except Exception as e:
        print(f"  WARNING: sector bar chart failed — {e}")

    print("\nPipeline complete.")
    print(f"Model saved to : {config.MODEL_SAVE_PATH}")
    print("Plots saved to : ./models/")


if __name__ == "__main__":
    main()
