#!/usr/bin/env python
"""CLI entry-point to train the Seq2Seq Transformer.

Supports single-dataset and multi-dataset (Phase 2) training.

Usage:
    # Phase 1: SAMSum only (default)
    python scripts/train_model.py --epochs 50 --batch_size 32

    # Phase 2: SAMSum + DialogSum combined
    python scripts/train_model.py --epochs 50 --batch_size 32 --datasets samsum,dialogsum

    # Smoke test
    python scripts/train_model.py --epochs 2 --batch_size 4 --max_samples 100
"""

from model.train import train

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Seq2Seq Transformer on dialogue datasets")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Linear warmup steps")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size (debug)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint dir")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    parser.add_argument(
        "--datasets",
        type=str,
        default="samsum",
        help="Comma-separated dataset list. E.g. 'samsum,dialogsum' for Phase 2.",
    )
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_samples=args.max_samples,
        checkpoint_dir=args.checkpoint_dir,
        device_name=args.device,
        datasets=args.datasets.split(","),
    )
