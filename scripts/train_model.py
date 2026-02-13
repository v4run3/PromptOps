#!/usr/bin/env python
"""CLI entry-point to train the Seq2Seq Transformer on SAMSum.

Usage:
    python scripts/train_model.py --epochs 10 --batch_size 32
    python scripts/train_model.py --epochs 2 --batch_size 4 --max_samples 100   # smoke test
"""

from model.train import train

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Seq2Seq Transformer on SAMSum")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size (debug)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint dir")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples,
        checkpoint_dir=args.checkpoint_dir,
        device_name=args.device,
    )
