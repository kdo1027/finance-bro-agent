"""Standalone presentation visualizations.

Three independent functions — each saves one or more PNGs to models_dir:

  plot_dataset_composition(save_dir)
      Grouped bar chart showing class distribution across PhraseBank,
      Twitter, and the combined dataset.

  plot_tsne_embeddings(test_ds, save_dir)
      Side-by-side t-SNE of CLS-token embeddings from base FinBERT vs.
      the combined fine-tuned model. Visually shows how fine-tuning
      reshapes the representation space.

  plot_sector_bar(result, save_dir)
      Horizontal bar chart of sector sentiment scores returned by
      inference.score_sectors(). Green = bullish, red = bearish.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModelForSequenceClassification
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate dataset rows into batched tensors, ignoring non-tensor fields."""
    return {
        "input_ids":      torch.tensor([b["input_ids"]      for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"]  for b in batch], dtype=torch.long),
        "labels":         torch.tensor([b["label"]           for b in batch], dtype=torch.long),
    }


def _extract_cls_embeddings(
    model: AutoModelForSequenceClassification,
    test_ds: Dataset,
    device: torch.device,
    max_per_class: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract CLS-token embeddings from the model's encoder for the test set.

    Takes a stratified sample of up to max_per_class examples per class to
    keep t-SNE fast and balanced across classes.

    Args:
        model: Sequence-classification model (must have .bert attribute).
        test_ds: Test-split Dataset.
        device: Compute device.
        max_per_class: Maximum examples to sample per class.

    Returns:
        Tuple of (embeddings array shape (N, 768), labels array shape (N,)).
    """
    test_ds.set_format(type="python")
    all_labels = np.array([row["label"] for row in test_ds])

    # Stratified sample indices
    sampled_indices: List[int] = []
    rng = np.random.RandomState(config.RANDOM_SEED)
    for cls_id in range(config.NUM_LABELS):
        cls_idx = np.where(all_labels == cls_id)[0]
        n = min(max_per_class, len(cls_idx))
        sampled_indices.extend(rng.choice(cls_idx, n, replace=False).tolist())
    sampled_indices.sort()

    subset = test_ds.select(sampled_indices)
    subset.set_format(type="python")
    loader = DataLoader(subset, batch_size=config.BATCH_SIZE,
                        shuffle=False, collate_fn=_collate_fn)

    model.eval()
    model.to(device)

    cls_embeddings: List[np.ndarray] = []
    label_list: List[int] = []

    with torch.no_grad():
        for batch in loader:
            batch_device = {k: v.to(device) for k, v in batch.items()}
            # Access the BERT encoder directly to get hidden states
            bert_outputs = model.bert(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
            )
            # CLS token = position 0 of the last hidden state
            cls = bert_outputs.last_hidden_state[:, 0, :].cpu().numpy()
            cls_embeddings.append(cls)
            label_list.extend(batch["labels"].tolist())

    return np.vstack(cls_embeddings), np.array(label_list)


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_dataset_composition(save_dir: str = "./models") -> None:
    """Plot grouped bar chart of class distribution per data source.

    Loads raw (untokenized) records from both datasets, computes per-class
    counts for PhraseBank, Twitter, and the combined set, and saves a
    labelled grouped bar chart.

    Args:
        save_dir: Directory to save the output PNG.
    """
    from .data_loader import load_raw_records_by_source

    print("Loading raw records for dataset composition chart...")
    phrasebank, twitter = load_raw_records_by_source()

    label_names = [config.ID2LABEL[i] for i in range(config.NUM_LABELS)]
    colors = ["#2196F3", "#F44336", "#4CAF50"]

    def _counts(records: List[Dict]) -> List[int]:
        return [(np.array([r["label"] for r in records]) == i).sum()
                for i in range(config.NUM_LABELS)]

    sources = {
        "PhraseBank\n(75% agree)": _counts(phrasebank),
        "Twitter\nFinancial News": _counts(twitter),
        "Combined": _counts(phrasebank + twitter),
    }

    x = np.arange(len(sources))
    bar_width = 0.22
    offsets = np.linspace(-(config.NUM_LABELS - 1) / 2,
                          (config.NUM_LABELS - 1) / 2,
                          config.NUM_LABELS) * bar_width

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, color) in enumerate(zip(label_names, colors)):
        values = [counts[i] for counts in sources.values()]
        bars = ax.bar(x + offsets[i], values, bar_width,
                      label=name, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    str(val), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(list(sources.keys()), fontsize=10)
    ax.set_ylabel("Number of examples")
    ax.set_title("Dataset Composition by Source and Sentiment Class")
    ax.legend(title="Sentiment")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    save_path = os.path.join(save_dir, "dataset_composition.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_tsne_embeddings(
    test_ds: Dataset,
    save_dir: str = "./models",
) -> None:
    """Side-by-side t-SNE of CLS embeddings: base FinBERT vs. fine-tuned.

    Extracts the [CLS] token representation from the BERT encoder for both
    models, reduces to 2D with t-SNE, and plots them side-by-side coloured
    by true sentiment label.

    Args:
        test_ds: Test-split Dataset (must have 'label' column).
        save_dir: Directory to save the output PNG.
    """
    device = _get_device()
    label_names = [config.ID2LABEL[i] for i in range(config.NUM_LABELS)]
    colors = ["#2196F3", "#F44336", "#4CAF50"]

    models_to_plot = [
        ("Base FinBERT",      config.MODEL_NAME,      True),
        ("Combined Fine-tuned", config.MODEL_SAVE_PATH, False),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("t-SNE of CLS Token Embeddings", fontsize=14, fontweight="bold")

    for ax, (title, model_path, is_base) in zip(axes, models_to_plot):
        print(f"  Extracting embeddings: {title}...")
        if is_base:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=config.NUM_LABELS,
                id2label=config.ID2LABEL,
                label2id=config.LABEL2ID,
                ignore_mismatched_sizes=True,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)

        embeddings, labels = _extract_cls_embeddings(model, test_ds, device)
        del model

        print(f"    Running t-SNE on {len(embeddings)} examples...")
        tsne = TSNE(
            n_components=2,
            random_state=config.RANDOM_SEED,
            perplexity=min(30, len(embeddings) - 1),
            max_iter=1000,
            verbose=0,
        )
        coords = tsne.fit_transform(embeddings)

        for i, (name, color) in enumerate(zip(label_names, colors)):
            mask = labels == i
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=color, s=12, alpha=0.65, label=name,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared legend
    patches = [mpatches.Patch(color=c, label=n)
               for n, c in zip(label_names, colors)]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.0), frameon=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    save_path = os.path.join(save_dir, "tsne_embeddings.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_sector_bar(
    result: Dict,
    save_dir: str = "./models",
) -> None:
    """Horizontal bar chart of sector sentiment scores from score_sectors().

    Bars are coloured green (bullish > 0.5) or red (bearish ≤ 0.5).
    Sectors with no headlines are shown as a grey hatched bar.

    Args:
        result: Dict returned by inference.score_sectors().
        save_dir: Directory to save the output PNG.
    """
    sentiments = result["sector_sentiment"]
    counts     = result.get("headline_counts", {})

    # Sort: sectors with data first (descending score), then no-data sectors
    with_data    = [(s, v) for s, v in sentiments.items() if v is not None]
    without_data = [(s, None) for s, v in sentiments.items() if v is None]
    ordered = sorted(with_data, key=lambda x: x[1]) + without_data

    sector_names = [s for s, _ in ordered]
    scores       = [v if v is not None else 0.0 for _, v in ordered]
    has_data     = [v is not None for _, v in ordered]

    bar_colors  = []
    bar_hatches = []
    for scored, score in zip(has_data, scores):
        if not scored:
            bar_colors.append("#BDBDBD")
            bar_hatches.append("//")
        elif score > 0.5:
            intensity = 0.4 + 0.6 * (score - 0.5) / 0.5
            bar_colors.append(plt.cm.Greens(intensity))
            bar_hatches.append(None)
        else:
            intensity = 0.4 + 0.6 * (0.5 - score) / 0.5
            bar_colors.append(plt.cm.Reds(intensity))
            bar_hatches.append(None)

    y_pos = np.arange(len(sector_names))
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (score, color, hatch) in enumerate(zip(scores, bar_colors, bar_hatches)):
        bar = ax.barh(y_pos[i], score, color=color,
                      hatch=hatch, edgecolor="white", height=0.65)
        n = counts.get(sector_names[i], 0)
        label_txt = f"{score:.3f}  (n={n})" if has_data[i] else "no data"
        ax.text(score + 0.01, y_pos[i], label_txt,
                va="center", fontsize=9,
                color="grey" if not has_data[i] else "black")

    ax.axvline(0.5, color="black", lw=1.2, linestyle="--", label="Neutral (0.5)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s.replace("_", " ").title() for s in sector_names], fontsize=10)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Weighted Avg. Positive Sentiment Probability")
    ax.set_title(
        f"Sector Sentiment Vector\n"
        f"Top: {result.get('top_sector', '—')}   "
        f"Bottom: {result.get('bottom_sector', '—')}   "
        f"Unrouted: {result.get('unrouted_count', 0)}"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    save_path = os.path.join(save_dir, "sector_sentiment_bar.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")
