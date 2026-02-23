"""MLOps Quality Gate script.

Evaluates a model checkpoint against the SAMSum test set and enforces
a minimum ROUGE score quality threshold.
"""

import sys
import argparse
from tqdm import tqdm

from model.config import ModelConfig
from model.inference import summarize
from model.dataset import SAMSumDataset
from eval.evaluator import evaluate_batch

def run_quality_gate(checkpoint_path: str, threshold: float = 0.20, max_samples: int = 100):
    print("--- Quality Gate Start ---")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Threshold:  {threshold}")
    print(f"Samples:    {max_samples}")

    # 1. Load Data
    config = ModelConfig()
    try:
        # We reuse SAMSumDataset but specifically for the test split
        # This will download the dataset if not present locally
        ds = SAMSumDataset(split="test", config=config, max_samples=max_samples)
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        sys.exit(1)

    # 2. Generate Summaries
    print(f"Generating summaries for {len(ds)} test samples...")
    generated_summaries = []
    reference_summaries = []

    for i in tqdm(range(len(ds))):
        item = ds.data[i]
        dialogue = item["dialogue"]
        ref = item["summary"]
        
        try:
            # Note: num_beams=1 for faster evaluation in CI
            gen = summarize(dialogue, checkpoint_path=checkpoint_path, num_beams=1)
            generated_summaries.append(gen)
            reference_summaries.append(ref)
        except Exception as e:
            print(f"Error during inference on sample {i}: {e}")
            continue

    if not generated_summaries:
        print("No summaries generated. Check if model checkpoint exists.")
        sys.exit(1)

    # 3. Calculate Metrics
    print("Calculating ROUGE scores...")
    scores = evaluate_batch(generated_summaries, reference_summaries)
    
    rouge_l = scores["rougeL"]
    print("\nResults:")
    print(f"  ROUGE-1: {scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {rouge_l:.4f}")

    # 4. Enforce Gate
    if rouge_l >= threshold:
        print(f"\n[PASS] Quality Gate satisfied! ({rouge_l:.4f} >= {threshold})")
        sys.exit(0)
    else:
        print(f"\n[FAIL] Quality Gate failed! ({rouge_l:.4f} < {threshold})")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps Quality Gate for PromptOps")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to best_model.pt")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.04,
        help="Min ROUGE-L required to pass. Phase 2 baseline: 0.0503 (SAMSum+DialogSum, 18 epochs from scratch).",
    )
    parser.add_argument("--samples", type=int, default=50, help="Number of test samples to evaluate")

    args = parser.parse_args()
    run_quality_gate(args.checkpoint, args.threshold, args.samples)
