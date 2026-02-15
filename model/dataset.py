"""SAMSum dataset loading and preprocessing.

Loads the SAMSum dialogue summarization dataset from HuggingFace Hub,
tokenizes dialogues and summaries, and provides PyTorch Dataset/DataLoader
utilities.
"""

import torch
from torch.utils.data import Dataset, DataLoader

from model.config import ModelConfig


class SAMSumDataset(Dataset):
    """PyTorch Dataset wrapper for the SAMSum dialogue summarization corpus.

    Each sample yields:
        - encoder_input: tokenized dialogue  (src)
        - decoder_input: tokenized summary with BOS prefix  (teacher-forcing input)
        - labels:        tokenized summary with EOS suffix   (training target)
    """

    def __init__(
        self,
        split: str = "train",
        config: ModelConfig | None = None,
        max_samples: int | None = None,
    ):
        """
        Args:
            split:       One of ``'train'``, ``'validation'``, ``'test'``.
            config:      Model config (controls max_seq_len, token IDs, etc.).
            max_samples: If set, truncate dataset to this many examples (for debugging).
        """
        from datasets import load_dataset
        from transformers import BertTokenizer

        self.config = config or ModelConfig()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        raw = load_dataset("samsum", split=split)
        if max_samples is not None:
            raw = raw.select(range(min(max_samples, len(raw))))

        self.data = raw

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        dialogue = item["dialogue"]
        summary = item["summary"]

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


def build_dataloader(
    split: str = "train",
    config: ModelConfig | None = None,
    batch_size: int = 32,
    max_samples: int | None = None,
    shuffle: bool | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Convenience builder for a SAMSum DataLoader.

    Args:
        split:       Dataset split name.
        config:      Model configuration.
        batch_size:  Batch size.
        max_samples: Limit dataset size (useful for smoke tests).
        shuffle:     Shuffle the data. Defaults to True for train, False otherwise.
        num_workers: DataLoader worker count.

    Returns:
        A PyTorch DataLoader ready for training or evaluation.
    """
    ds = SAMSumDataset(split=split, config=config, max_samples=max_samples)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
