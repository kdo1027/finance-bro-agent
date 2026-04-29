"""Training loop with validation, best-model checkpointing, and history tracking."""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm


def _get_device() -> torch.device:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate a list of dataset rows into batched tensors."""
    return {
        "input_ids":      torch.tensor([b["input_ids"]      for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"]  for b in batch], dtype=torch.long),
        "labels":         torch.tensor([b["label"]           for b in batch], dtype=torch.long),
    }


def _run_epoch_train(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    epoch: int,
) -> float:
    """Run one full training epoch with optional fp16 mixed precision.

    Mixed precision is only active when scaler is not None (CUDA only).
    On MPS and CPU the standard float32 path is taken.

    Args:
        model: Model in training mode.
        loader: Training DataLoader.
        optimizer: AdamW optimizer.
        scheduler: LR scheduler.
        scaler: GradScaler for fp16 (CUDA only); None on MPS/CPU.
        device: Compute device.
        epoch: Current epoch index (1-based) for tqdm display.

    Returns:
        Mean training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader)


def _run_epoch_eval(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
    split_name: str = "val",
) -> Tuple[float, float]:
    """Run inference over a data split and return loss and accuracy.

    Args:
        model: Model in eval mode.
        loader: DataLoader for the split.
        device: Compute device.
        split_name: Label for tqdm display.

    Returns:
        Tuple of (mean loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[{split_name}]", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    return total_loss / len(loader), correct / total


def train(
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizerBase,
    dataset_dict: DatasetDict,
) -> Dict[str, List[float]]:
    """Fine-tune the model on the training split with validation.

    Saves the best checkpoint (by val loss) to config.MODEL_SAVE_PATH.

    Args:
        model: Model with frozen layers applied.
        tokenizer: Tokenizer to save alongside the best checkpoint.
        dataset_dict: DatasetDict with 'train' and 'val' splits.

    Returns:
        History dict with keys: train_loss, val_loss, val_acc.
    """
    device = _get_device()
    use_amp = device.type == "cuda"
    print(f"Training on device: {device}  (mixed precision: {'fp16' if use_amp else 'disabled'})\n")
    model.to(device)

    torch.manual_seed(config.RANDOM_SEED)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    train_ds = dataset_dict["train"]
    val_ds   = dataset_dict["val"]

    train_ds.set_format(type="python")
    val_ds.set_format(type="python")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=_collate_fn,
    )

    total_steps = len(train_loader) * config.NUM_EPOCHS
    warmup_steps = int(config.WARMUP_RATIO * total_steps)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss":   [],
        "val_acc":    [],
    }
    best_val_loss = float("inf")
    save_path = config.MODEL_SAVE_PATH
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss = _run_epoch_train(model, train_loader, optimizer, scheduler, scaler, device, epoch)
        val_loss, val_acc = _run_epoch_eval(model, val_loader, device, split_name="val")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{config.NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    return history
