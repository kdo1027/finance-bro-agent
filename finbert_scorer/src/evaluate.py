"""Evaluation on the held-out test set with full comparison suite.

Generates (all saved to ./models/):
  - 3-way comparison table printed to stdout:
      Base FinBERT | Twitter-only fine-tuned | Combined fine-tuned
  - confusion_matrix.png      — labeled heatmap
  - training_history.png      — train/val loss per epoch
  - roc_curves.png            — per-class one-vs-rest ROC with AUC
  - pr_curves.png             — per-class precision-recall with average precision
  - confidence_distribution.png — correct vs. incorrect prediction confidence
  - Error analysis printed to stdout (top most-confidently-wrong examples)
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


TWITTER_ONLY_PATH = "./models/finbert_twitter_only"

# Consistent class colours used across all plots
_CLASS_COLORS = {"positive": "#2196F3", "negative": "#F44336", "neutral": "#4CAF50"}


# ---------------------------------------------------------------------------
# Device / collation
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


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def _predict_with_probs(
    model: AutoModelForSequenceClassification,
    test_ds: Dataset,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Run inference and return labels, predictions, softmax probs, and texts.

    Args:
        model: Model to evaluate (moved to device internally).
        test_ds: Test-split Dataset; uses 'text' column if present.
        device: Compute device.

    Returns:
        Tuple of:
          y_true  (N,)         — ground-truth integer labels
          y_pred  (N,)         — predicted integer labels
          y_probs (N, C)       — softmax probabilities
          texts   list[str]    — original sentences (empty list if unavailable)
    """
    texts: List[str] = list(test_ds["text"]) if "text" in test_ds.column_names else []

    test_ds.set_format(type="python")
    loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=_collate_fn,
    )
    model.eval()
    model.to(device)

    all_probs: List[List[float]] = []
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().tolist()
            all_probs.extend(probs)
            all_preds.extend(int(max(range(len(p)), key=lambda i: p[i])) for p in probs)
            all_labels.extend(batch["labels"].cpu().tolist())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs), texts


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute accuracy, macro F1, and per-class F1 / precision / recall.

    Args:
        y_true: Ground-truth label array.
        y_pred: Predicted label array.

    Returns:
        Flat dict of metric name → float value.
    """
    label_ids = list(range(config.NUM_LABELS))
    label_names = [config.ID2LABEL[i] for i in label_ids]

    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    for metric_fn, key_prefix in [
        (f1_score,        "f1"),
        (precision_score, "precision"),
        (recall_score,    "recall"),
    ]:
        per_class = metric_fn(y_true, y_pred, average=None, labels=label_ids, zero_division=0)
        for i, name in enumerate(label_names):
            metrics[f"{key_prefix}_{name}"] = per_class[i]

    return metrics


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
) -> None:
    """Save a labeled confusion-matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        save_path: Output PNG path.
    """
    labels = [config.ID2LABEL[i] for i in range(config.NUM_LABELS)]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_LABELS)))
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Combined Fine-tuned FinBERT")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _plot_training_history(
    history: Optional[Dict[str, List[float]]],
    save_path: str,
) -> None:
    """Save a training/validation loss curve.

    Args:
        history: Dict with 'train_loss' and 'val_loss' lists.
        save_path: Output PNG path.
    """
    if not history:
        print("  No training history — skipping history plot.")
        return
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    ax.plot(epochs, history["val_loss"],   marker="s", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training History — Combined Dataset Fine-tuning")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: str,
    title: str = "ROC Curves (one-vs-rest)",
) -> None:
    """Save per-class one-vs-rest ROC curves with AUC scores.

    Args:
        y_true: Ground-truth integer labels.
        y_probs: Softmax probabilities, shape (N, num_labels).
        save_path: Output PNG path.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(config.NUM_LABELS):
        name = config.ID2LABEL[i]
        color = _CLASS_COLORS[name]
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _plot_pr_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: str,
    title: str = "Precision-Recall Curves",
) -> None:
    """Save per-class precision-recall curves with average precision scores.

    Dotted horizontal lines show the random-classifier baseline (class prevalence).

    Args:
        y_true: Ground-truth integer labels.
        y_probs: Softmax probabilities, shape (N, num_labels).
        save_path: Output PNG path.
        title: Plot title.
    """
    total = len(y_true)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(config.NUM_LABELS):
        name = config.ID2LABEL[i]
        color = _CLASS_COLORS[name]
        y_bin = (y_true == i).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, y_probs[:, i])
        ap = average_precision_score(y_bin, y_probs[:, i])
        baseline = (y_true == i).sum() / total
        ax.plot(rec, prec, color=color, lw=2, label=f"{name}  (AP = {ap:.3f})")
        ax.axhline(baseline, color=color, lw=1, linestyle=":", alpha=0.55)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _plot_confidence_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    save_path: str,
) -> None:
    """Histogram of model confidence split by correct vs. incorrect predictions.

    Confidence is defined as max(softmax probabilities) for a given example.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_probs: Softmax probabilities, shape (N, num_labels).
        save_path: Output PNG path.
    """
    confidence = y_probs.max(axis=1)
    correct   = confidence[y_true == y_pred]
    incorrect = confidence[y_true != y_pred]

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 31)
    ax.hist(correct,   bins=bins, alpha=0.65, color="#4CAF50",
            label=f"Correct   (n={len(correct):,})")
    ax.hist(incorrect, bins=bins, alpha=0.65, color="#F44336",
            label=f"Incorrect (n={len(incorrect):,})")
    ax.axvline(correct.mean(),   color="#2E7D32", lw=1.5, linestyle="--",
               label=f"Correct mean = {correct.mean():.2f}")
    ax.axvline(incorrect.mean(), color="#C62828", lw=1.5, linestyle="--",
               label=f"Incorrect mean = {incorrect.mean():.2f}")
    ax.set_xlabel("Model Confidence  (max softmax probability)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution: Correct vs. Incorrect Predictions")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Text-based reports
# ---------------------------------------------------------------------------

def _print_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    texts: List[str],
    n: int = 12,
) -> None:
    """Print the most confidently wrong predictions to stdout.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        y_probs: Softmax probabilities, shape (N, num_labels).
        texts: Original sentence strings; empty list if unavailable.
        n: Number of examples to show.
    """
    wrong_mask = y_true != y_pred
    if wrong_mask.sum() == 0:
        print("  No incorrect predictions on this test set.")
        return

    wrong_idx = np.where(wrong_mask)[0]
    confidence = y_probs[wrong_idx].max(axis=1)
    top_idx = wrong_idx[np.argsort(confidence)[::-1]][:n]

    print("\n" + "=" * 72)
    print(f"Error Analysis — Top {n} Most Confident Wrong Predictions")
    print("=" * 72)
    for rank, idx in enumerate(top_idx, 1):
        snippet = ""
        if texts:
            raw = texts[idx]
            snippet = (raw[:80] + "…") if len(raw) > 80 else raw
        true_lbl = config.ID2LABEL[int(y_true[idx])]
        pred_lbl = config.ID2LABEL[int(y_pred[idx])]
        conf     = float(y_probs[idx].max())
        print(f"\n  #{rank:>2}  conf={conf:.3f}  true={true_lbl:<10}  pred={pred_lbl}")
        if snippet:
            print(f"       \"{snippet}\"")
    print("=" * 72 + "\n")


def _print_comparison_table(
    model_results: List[Tuple[str, Dict[str, float]]],
) -> None:
    """Print an N-way model comparison table to stdout.

    Args:
        model_results: List of (display_name, metrics_dict) tuples in desired
            column order (left → right).
    """
    label_names = [config.ID2LABEL[i] for i in range(config.NUM_LABELS)]
    rows = (
        [("Accuracy",  "accuracy"), ("Macro F1", "macro_f1")]
        + [(f"F1 ({n})",        f"f1_{n}")        for n in label_names]
        + [(f"Precision ({n})", f"precision_{n}") for n in label_names]
        + [(f"Recall ({n})",    f"recall_{n}")    for n in label_names]
    )

    col_w   = 20
    n_cols  = len(model_results)
    sep_len = col_w * (n_cols + 1) + n_cols + 1

    header = " Metric".ljust(col_w + 1)
    for name, _ in model_results:
        header += "│" + name.center(col_w)
    divider = "─" * sep_len

    print("\n┌" + divider + "┐")
    print(f"│{header} │")
    print("├" + divider + "┤")
    for display_name, key in rows:
        row = f" {display_name}".ljust(col_w + 1)
        for _, metrics in model_results:
            row += "│" + f"{metrics[key]:.4f}".center(col_w)
        print(f"│{row} │")
    print("└" + divider + "┘\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    test_ds: Dataset,
    history: Optional[Dict[str, List[float]]] = None,
) -> None:
    """Load saved models, run full evaluation, and save all plots.

    Compares Base FinBERT, Twitter-only fine-tuned (if available at
    TWITTER_ONLY_PATH), and the combined fine-tuned model.

    Args:
        test_ds: HuggingFace Dataset (test split, ideally with 'text' column).
        history: Optional training history dict from train.py.
    """
    device    = _get_device()
    models_dir = os.path.dirname(config.MODEL_SAVE_PATH)
    os.makedirs(models_dir, exist_ok=True)

    model_results: List[Tuple[str, Dict[str, float]]] = []

    # ── Base FinBERT ──────────────────────────────────────────────────────
    print(f"Loading base model: {config.MODEL_NAME}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    y_true, base_pred, base_probs, _ = _predict_with_probs(base_model, test_ds, device)
    model_results.append(("Base FinBERT", _compute_metrics(y_true, base_pred)))
    del base_model

    # ── Twitter-only fine-tuned (optional) ────────────────────────────────
    if os.path.isdir(TWITTER_ONLY_PATH):
        print(f"Loading Twitter-only model: {TWITTER_ONLY_PATH}")
        tw_model = AutoModelForSequenceClassification.from_pretrained(TWITTER_ONLY_PATH)
        _, tw_pred, tw_probs, _ = _predict_with_probs(tw_model, test_ds, device)
        model_results.append(("Twitter-only ft", _compute_metrics(y_true, tw_pred)))
        del tw_model
    else:
        print(f"  (Twitter-only model not found at '{TWITTER_ONLY_PATH}' — skipping)")

    # ── Combined fine-tuned model ─────────────────────────────────────────
    print(f"Loading combined fine-tuned model: {config.MODEL_SAVE_PATH}")
    ft_model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_SAVE_PATH)
    _, ft_pred, ft_probs, texts = _predict_with_probs(ft_model, test_ds, device)
    model_results.append(("Combined ft", _compute_metrics(y_true, ft_pred)))
    del ft_model

    # ── Comparison table ──────────────────────────────────────────────────
    _print_comparison_table(model_results)

    # ── Confusion matrix ──────────────────────────────────────────────────
    _plot_confusion_matrix(
        y_true, ft_pred,
        os.path.join(models_dir, "confusion_matrix.png"),
    )

    # ── Training history ──────────────────────────────────────────────────
    _plot_training_history(
        history,
        os.path.join(models_dir, "training_history.png"),
    )

    # ── ROC curves ────────────────────────────────────────────────────────
    _plot_roc_curves(
        y_true, ft_probs,
        os.path.join(models_dir, "roc_curves.png"),
        title="ROC Curves — Combined Fine-tuned FinBERT (one-vs-rest)",
    )

    # ── Precision-Recall curves ───────────────────────────────────────────
    _plot_pr_curves(
        y_true, ft_probs,
        os.path.join(models_dir, "pr_curves.png"),
        title="Precision-Recall Curves — Combined Fine-tuned FinBERT",
    )

    # ── Confidence distribution ───────────────────────────────────────────
    _plot_confidence_distribution(
        y_true, ft_pred, ft_probs,
        os.path.join(models_dir, "confidence_distribution.png"),
    )

    # ── Error analysis (stdout) ───────────────────────────────────────────
    _print_error_analysis(y_true, ft_pred, ft_probs, texts)

    print(f"Evaluation complete. Plots saved to: {models_dir}/")
