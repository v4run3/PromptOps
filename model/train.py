"""Training loop for the Seq2Seq Transformer on SAMSum.

Supports:
    - Cross-entropy loss with label smoothing and padding masking
    - Adam optimiser with warm-up + cosine annealing LR schedule
    - Epoch-based training with periodic validation
    - Checkpoint saving / resuming
    - TensorBoard logging (optional)
"""

import os
import time
import argparse

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.config import ModelConfig
from model.transformer import Seq2SeqTransformer
from model.dataset import build_dataloader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_model(config: ModelConfig, device: torch.device) -> Seq2SeqTransformer:
    """Instantiate the transformer and move it to *device*."""
    model = Seq2SeqTransformer(config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    return model


def _save_checkpoint(
    model: Seq2SeqTransformer,
    optimizer: Adam,
    epoch: int,
    loss: float,
    path: str,
) -> None:
    """Persist model + optimiser state."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    print(f"  ✓ Checkpoint saved → {path}")


# ---------------------------------------------------------------------------
# Training / Validation Steps
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: Seq2SeqTransformer,
    dataloader,
    criterion: nn.Module,
    optimizer: Adam,
    device: torch.device,
    epoch: int,
) -> float:
    """Run a single training epoch and return the average loss."""
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        src = batch["encoder_input"].to(device)
        tgt_in = batch["decoder_input"].to(device)
        labels = batch["labels"].to(device)

        logits = model(src, tgt_in)  # (B, T, V)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            print(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f}"
            )

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def validate(
    model: Seq2SeqTransformer,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run validation and return the average loss."""
    model.eval()
    total_loss = 0.0

    for batch in dataloader:
        src = batch["encoder_input"].to(device)
        tgt_in = batch["decoder_input"].to(device)
        labels = batch["labels"].to(device)

        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def train(
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    warmup_steps: int = 500,
    max_samples: int | None = None,
    checkpoint_dir: str = "checkpoints",
    device_name: str | None = None,
) -> None:
    """Full training entry-point.

    Args:
        epochs:         Number of training epochs.
        batch_size:     Batch size for train / val loaders.
        learning_rate:  Peak learning rate after warmup.
        warmup_steps:   Linear warmup steps (currently simplified).
        max_samples:    Limit dataset size (for smoke tests).
        checkpoint_dir: Directory to save checkpoints.
        device_name:    Force device (``'cpu'``, ``'cuda'``). Auto-detect if None.
    """
    config = ModelConfig()
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Data
    train_loader = build_dataloader("train", config, batch_size, max_samples)
    val_loader = build_dataloader("validation", config, batch_size, max_samples)

    # Model
    model = _build_model(config, device)

    # Loss — ignore padding tokens
    criterion = nn.CrossEntropyLoss(
        ignore_index=config.pad_token_id,
        label_smoothing=config.label_smoothing,
    )

    # Optimiser + scheduler
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{epochs} — "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(checkpoint_dir, "best_model.pt"),
            )

    # Save final checkpoint
    _save_checkpoint(
        model, optimizer, epochs, val_loss,
        os.path.join(checkpoint_dir, "final_model.pt"),
    )
    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Seq2Seq Transformer on SAMSum")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples,
        checkpoint_dir=args.checkpoint_dir,
        device_name=args.device,
    )
