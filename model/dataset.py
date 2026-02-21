"""Dialogue summarization dataset loading and preprocessing.

Supports:
    - SAMSum   (knkarthick/samsum)   — ~16k chat dialogues
    - DialogSum (knkarthick/dialogsum) — ~13.5k task-oriented dialogues

Both are loaded from the HuggingFace Hub, tokenized with BertTokenizer,
and can be combined into a single unified DataLoader via CombinedDialogueDataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from model.config import ModelConfig


# ---------------------------------------------------------------------------
# HuggingFace dataset configs (field names differ between datasets)
# ---------------------------------------------------------------------------

_DATASET_CONFIG = {
    "samsum": {
        "hf_name": "knkarthick/samsum",
        "dialogue_key": "dialogue",
        "summary_key": "summary",
        "splits": {"train": "train", "validation": "validation", "test": "test"},
    },
    "dialogsum": {
        "hf_name": "knkarthick/dialogsum",
        "dialogue_key": "dialogue",
        "summary_key": "summary",
        "splits": {"train": "train", "validation": "validation", "test": "test"},
    },
}


# ---------------------------------------------------------------------------
# Single source Dataset
# ---------------------------------------------------------------------------


class DialogueDataset(Dataset):
    """Generic PyTorch Dataset for any HuggingFace dialogue-summarization corpus.

    Each sample yields:
        - encoder_input: tokenized dialogue  (src)
        - decoder_input: tokenized summary with BOS prefix  (teacher-forcing input)
        - labels:        tokenized summary with EOS suffix   (training target)
    """

    def __init__(
        self,
        dataset_name: str = "samsum",
        split: str = "train",
        config: ModelConfig | None = None,
        max_samples: int | None = None,
    ):
        """
        Args:
            dataset_name: One of ``'samsum'`` or ``'dialogsum'``.
            split:        One of ``'train'``, ``'validation'``, ``'test'``.
            config:       Model config (controls max_seq_len, token IDs, etc.).
            max_samples:  If set, truncate dataset to this many examples (for debugging).
        """
        from datasets import load_dataset
        from transformers import BertTokenizer

        if dataset_name not in _DATASET_CONFIG:
            raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {list(_DATASET_CONFIG.keys())}")

        self.config = config or ModelConfig()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        ds_cfg = _DATASET_CONFIG[dataset_name]
        hf_split = ds_cfg["splits"].get(split, split)

        raw = load_dataset(ds_cfg["hf_name"], split=hf_split)
        if max_samples is not None:
            raw = raw.select(range(min(max_samples, len(raw))))

        self.data = raw
        self.dialogue_key = ds_cfg["dialogue_key"]
        self.summary_key = ds_cfg["summary_key"]
        print(f"  Loaded {dataset_name} [{split}]: {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        dialogue = item[self.dialogue_key]
        summary = item[self.summary_key]

        # Tokenize source (dialogue)
        src_enc = self.tokenizer(
            dialogue,
            max_length=self.config.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoder_input = src_enc["input_ids"].squeeze(0)  # (max_seq_len,)

        # Tokenize target (summary)
        tgt_enc = self.tokenizer(
            summary,
            max_length=self.config.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tgt_ids = tgt_enc["input_ids"].squeeze(0)  # (max_seq_len,)

        # decoder_input:  [BOS, tok1, tok2, ..., tokN]  (shift right)
        # labels:         [tok1, tok2, ..., tokN, EOS]
        decoder_input = tgt_ids[:-1]
        labels = tgt_ids[1:]

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

class SAMSumDataset(DialogueDataset):
    """Backward-compatible alias for DialogueDataset(dataset_name='samsum')."""

    def __init__(
        self,
        split: str = "train",
        config: ModelConfig | None = None,
        max_samples: int | None = None,
    ):
        super().__init__(
            dataset_name="samsum",
            split=split,
            config=config,
            max_samples=max_samples,
        )


# ---------------------------------------------------------------------------
# Combined Dataset (Phase 2)
# ---------------------------------------------------------------------------


def build_combined_dataset(
    datasets: list[str],
    split: str = "train",
    config: ModelConfig | None = None,
    max_samples_per_dataset: int | None = None,
) -> ConcatDataset:
    """Build a combined dataset from multiple sources.

    Args:
        datasets:                 List of dataset names, e.g. ['samsum', 'dialogsum'].
        split:                    Dataset split: 'train', 'validation', or 'test'.
        config:                   Model configuration.
        max_samples_per_dataset:  Per-source sample limit (for debugging).

    Returns:
        A PyTorch ConcatDataset ready for DataLoader.
    """
    parts = [
        DialogueDataset(
            dataset_name=name,
            split=split,
            config=config,
            max_samples=max_samples_per_dataset,
        )
        for name in datasets
    ]
    combined = ConcatDataset(parts)
    total = sum(len(p) for p in parts)
    print(f"  Combined dataset [{split}]: {total} samples from {datasets}")
    return combined


# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------


def build_dataloader(
    split: str = "train",
    config: ModelConfig | None = None,
    batch_size: int = 32,
    max_samples: int | None = None,
    shuffle: bool | None = None,
    num_workers: int = 0,
    datasets: list[str] | None = None,
) -> DataLoader:
    """Build a DataLoader for one or more dialogue summarization datasets.

    Args:
        split:       Dataset split name.
        config:      Model configuration.
        batch_size:  Batch size.
        max_samples: Limit dataset size (useful for smoke tests).
        shuffle:     Shuffle the data. Defaults to True for train, False otherwise.
        num_workers: DataLoader worker count.
        datasets:    List of dataset names to use.
                     Defaults to ``['samsum']`` for backward compatibility.
                     Pass ``['samsum', 'dialogsum']`` for Phase 2 combined training.

    Returns:
        A PyTorch DataLoader ready for training or evaluation.
    """
    if datasets is None:
        datasets = ["samsum"]

    if shuffle is None:
        shuffle = split == "train"

    if len(datasets) == 1:
        ds = DialogueDataset(
            dataset_name=datasets[0],
            split=split,
            config=config,
            max_samples=max_samples,
        )
    else:
        ds = build_combined_dataset(
            datasets=datasets,
            split=split,
            config=config,
            max_samples_per_dataset=max_samples,
        )

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
