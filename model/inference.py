"""Inference utilities for the trained Seq2Seq Transformer.

Provides greedy decoding to generate summaries from raw dialogue text.
"""

import torch
from transformers import BertTokenizer

from model.config import ModelConfig
from model.transformer import Seq2SeqTransformer


def load_model(
    checkpoint_path: str,
    config: ModelConfig | None = None,
    device: torch.device | None = None,
) -> tuple[Seq2SeqTransformer, torch.device]:
    """Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        config:          Model config (must match the checkpoint architecture).
        device:          Target device. Auto-detects if None.

    Returns:
        Tuple of (model, device).
    """
    config = config or ModelConfig()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Seq2SeqTransformer(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


@torch.no_grad()
def greedy_decode(
    model: Seq2SeqTransformer,
    src: torch.Tensor,
    max_len: int = 128,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Auto-regressively generate an output sequence via greedy decoding.

    Args:
        model:   Trained Seq2SeqTransformer.
        src:     Source token IDs — (1, src_len).
        max_len: Maximum tokens to generate.
        device:  Device for tensors.

    Returns:
        Generated token IDs — (1, generated_len).
    """
    config = model.config
    device = device or next(model.parameters()).device

    # Encode source once
    src_pad_mask = model.generate_padding_mask(src, config.pad_token_id)
    import math
    src_emb = model.pos_encoding(model.src_embedding(src) * math.sqrt(config.d_model))
    memory = model.encoder(src_emb, src_pad_mask)

    # Start decoder with BOS token
    generated = torch.tensor([[config.bos_token_id]], device=device)

    for _ in range(max_len):
        tgt_mask = model.generate_causal_mask(generated.size(1), device)
        tgt_emb = model.pos_encoding(model.tgt_embedding(generated) * math.sqrt(config.d_model))
        decoder_out = model.decoder(tgt_emb, memory, tgt_mask, src_pad_mask)
        logits = model.output_projection(decoder_out[:, -1, :])  # last position
        next_token = logits.argmax(dim=-1, keepdim=True)  # greedy

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == config.eos_token_id:
            break

    return generated


def summarize(
    dialogue: str,
    checkpoint_path: str = "checkpoints/best_model.pt",
    config: ModelConfig | None = None,
) -> str:
    """Generate a summary for a raw dialogue string.

    Args:
        dialogue:        The input dialogue text.
        checkpoint_path: Path to the model checkpoint.
        config:          Optional model config override.

    Returns:
        The generated summary as a plain string.
    """
    config = config or ModelConfig()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model, device = load_model(checkpoint_path, config)

    # Tokenize
    src_enc = tokenizer(
        dialogue,
        max_length=config.max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    src = src_enc["input_ids"].to(device)

    # Decode
    output_ids = greedy_decode(model, src, device=device)
    summary = tokenizer.decode(output_ids.squeeze(0), skip_special_tokens=True)
    return summary
