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
    """Generate an output sequence using simple greedy decoding.

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

    # Start with [BOS]
    generated = torch.tensor([[config.bos_token_id]], device=device)

    for _ in range(max_len):
        # Forward pass
        logits = model(src, generated)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

        # Append to sequence
        generated = torch.cat([generated, next_token], dim=1)

        # Stop if [EOS] reached
        if next_token.item() == config.eos_token_id:
            break

    return generated

@torch.no_grad()
def beam_decode(
    model: Seq2SeqTransformer,
    src: torch.Tensor,
    beam_size: int = 5,
    max_len: int = 128,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate an output sequence using Beam Search.

    Args:
        model:     Trained Seq2SeqTransformer.
        src:       Source token IDs — (1, src_len).
        beam_size: Number of beams to maintain.
        max_len:   Maximum tokens to generate.
        device:    Device for tensors.

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

    # Beams: list of (sequence, score)
    # Start with BOS
    beams = [(torch.tensor([[config.bos_token_id]], device=device), 0.0)]

    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            # If sequence already reached EOS, keep it
            if seq[0, -1].item() == config.eos_token_id:
                new_beams.append((seq, score))
                continue

            # Decode next step
            tgt_mask = model.generate_causal_mask(seq.size(1), device)
            tgt_emb = model.pos_encoding(model.tgt_embedding(seq) * math.sqrt(config.d_model))
            decoder_out = model.decoder(tgt_emb, memory, tgt_mask, src_pad_mask)
            logits = model.output_projection(decoder_out[:, -1, :])
            log_probs = torch.log_softmax(logits, dim=-1)

            # Get top k candidates
            top_probs, top_indices = log_probs.topk(beam_size)

            for i in range(beam_size):
                next_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                new_score = score + top_probs[0, i].item()
                new_seq = torch.cat([seq, next_token], dim=1)
                new_beams.append((new_seq, new_score))

        # Sort all candidates by score and take top k
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

        # Stop early if all beams reach EOS
        if all(b[0][0, -1].item() == config.eos_token_id for b in beams):
            break

    # Return the best sequence
    return beams[0][0]


def summarize(
    dialogue: str,
    checkpoint_path: str = "checkpoints/best_model.pt",
    config: ModelConfig | None = None,
    num_beams: int = 5,
) -> str:
    """Generate a summary for a raw dialogue string.

    Args:
        dialogue:        The input dialogue text.
        checkpoint_path: Path to the model checkpoint.
        config:          Optional model config override.
        num_beams:       If > 1, use Beam Search. Otherwise use Greedy.

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
    if num_beams > 1:
        output_ids = beam_decode(model, src, beam_size=num_beams, device=device)
    else:
        output_ids = greedy_decode(model, src, device=device)

    summary = tokenizer.decode(output_ids.squeeze(0), skip_special_tokens=True)
    return summary
