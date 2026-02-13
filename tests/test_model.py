"""Unit tests for the custom Transformer model.

Tests cover:
    - Forward pass output shapes
    - Causal masking correctness
    - Parameter count sanity check
"""

import torch
import pytest

from model.config import ModelConfig
from model.transformer import Seq2SeqTransformer


@pytest.fixture
def config():
    """Small config for fast tests."""
    return ModelConfig(
        d_model=64,
        n_heads=2,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=128,
        max_seq_len=32,
        vocab_size=100,
    )


@pytest.fixture
def model(config):
    return Seq2SeqTransformer(config)


class TestSeq2SeqTransformer:
    """Tests for the Seq2SeqTransformer model."""

    def test_forward_output_shape(self, model, config):
        """Forward pass should produce (batch, tgt_len, vocab_size) logits."""
        batch_size = 2
        src = torch.randint(1, config.vocab_size, (batch_size, 16))
        tgt = torch.randint(1, config.vocab_size, (batch_size, 10))

        logits = model(src, tgt)

        assert logits.shape == (batch_size, 10, config.vocab_size)

    def test_causal_mask_shape(self, model):
        """Causal mask should be upper-triangular boolean tensor."""
        mask = model.generate_causal_mask(8, device=torch.device("cpu"))
        assert mask.shape == (1, 1, 8, 8)
        assert mask.dtype == torch.bool
        # Diagonal and below should be False (attend), above should be True (ignore)
        assert mask[0, 0, 0, 0].item() is False  # can attend to self
        assert mask[0, 0, 0, 1].item() is True    # cannot attend to future

    def test_padding_mask_shape(self, model, config):
        """Padding mask should mark pad tokens as True."""
        tokens = torch.tensor([[1, 2, 3, 0, 0]])  # last two are padding
        mask = model.generate_padding_mask(tokens, config.pad_token_id)
        assert mask.shape == (1, 1, 1, 5)
        assert mask[0, 0, 0, 3].item() is True   # pad
        assert mask[0, 0, 0, 0].item() is False   # non-pad

    def test_parameter_count(self, model):
        """Model should have a non-trivial number of parameters."""
        n_params = model.count_parameters()
        assert n_params > 0
        print(f"Test model params: {n_params:,}")

    def test_forward_with_padding(self, model, config):
        """Forward pass should not crash with padded inputs."""
        batch_size = 2
        src = torch.zeros(batch_size, 16, dtype=torch.long)  # all padding
        src[:, :5] = torch.randint(1, config.vocab_size, (batch_size, 5))

        tgt = torch.zeros(batch_size, 10, dtype=torch.long)
        tgt[:, :3] = torch.randint(1, config.vocab_size, (batch_size, 3))

        logits = model(src, tgt)
        assert logits.shape == (batch_size, 10, config.vocab_size)
        assert not torch.isnan(logits).any()
