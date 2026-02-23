"""From-scratch Encoder-Decoder Transformer for sequence-to-sequence tasks.

Every component (attention, FFN, positional encoding, encoder/decoder layers)
is implemented manually — no use of ``torch.nn.Transformer``.

Approximate parameter count with default config: ~10 M.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention.

    Args:
        d_model: Total model dimension.
        n_heads: Number of parallel attention heads.
        dropout: Dropout on attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_q, d_model)
            key:   (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask:  Broadcastable boolean mask where ``True`` = **ignore**.

        Returns:
            Output tensor of shape (batch, seq_q, d_model).
        """
        batch_size = query.size(0)

        # Linear projections → split into heads
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # (batch, heads, seq_q, d_k)

        # Concatenate heads and final linear
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context)


class PositionWiseFeedForward(nn.Module):
    """Two-layer feed-forward network applied independently to each position.

    FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_seq_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class EncoderLayer(nn.Module):
    """Single Transformer encoder block.

    self-attention → add & norm → FFN → add & norm
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


class TransformerEncoder(nn.Module):
    """Stack of N encoder layers."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class DecoderLayer(nn.Module):
    """Single Transformer decoder block.

    masked-self-attention → add & norm → cross-attention → add & norm → FFN → add & norm
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Masked self-attention
        attn_out = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # Cross-attention over encoder output
        cross_out = self.cross_attn(x, encoder_output, encoder_output, mask=memory_mask)
        x = self.norm2(x + self.dropout2(cross_out))

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x


class TransformerDecoder(nn.Module):
    """Stack of N decoder layers."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, memory_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Full Seq2Seq Model
# ---------------------------------------------------------------------------


class Seq2SeqTransformer(nn.Module):
    """Encoder-Decoder Transformer for sequence-to-sequence generation.

    Architecture:
        source tokens → embedding + pos-enc → encoder
        target tokens → embedding + pos-enc → decoder (with cross-attn)
        decoder output → linear → vocab logits
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.use_pretrained_encoder = config.use_pretrained_encoder

        if self.use_pretrained_encoder:
            # --- Phase 3: BERT-initialized encoder ---
            from transformers import BertModel

            self.bert_encoder = BertModel.from_pretrained("bert-base-uncased")
            # Project BERT's 768-d output to our decoder's d_model
            self.encoder_projection = nn.Linear(
                config.bert_hidden_size, config.d_model
            )
        else:
            # --- Original from-scratch encoder ---
            self.src_embedding = nn.Embedding(
                config.vocab_size, config.d_model, padding_idx=config.pad_token_id
            )
            self.encoder = TransformerEncoder(
                config.n_encoder_layers,
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.dropout,
            )

        # Decoder (always from scratch)
        self.tgt_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.pos_encoding = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        self.decoder = TransformerDecoder(
            config.n_decoder_layers,
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout,
        )

        self.output_projection = nn.Linear(config.d_model, config.vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for non-pretrained layers."""
        # Skip BERT parameters — they are already pre-trained
        pretrained_params = set()
        if self.use_pretrained_encoder:
            pretrained_params = {
                id(p) for p in self.bert_encoder.parameters()
            }

        for p in self.parameters():
            if id(p) not in pretrained_params and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ----- Encoder Freeze/Unfreeze (Phase 3) -----

    def freeze_encoder(self):
        """Freeze BERT encoder parameters (used during initial training epochs)."""
        if self.use_pretrained_encoder:
            for param in self.bert_encoder.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze BERT encoder parameters for fine-tuning."""
        if self.use_pretrained_encoder:
            for param in self.bert_encoder.parameters():
                param.requires_grad = True

    # ----- Mask Utilities -----

    @staticmethod
    def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask (True = ignore).

        Returns:
            Boolean mask of shape (1, 1, seq_len, seq_len).
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def generate_padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
        """Padding mask (True = ignore).

        Args:
            tokens: (batch, seq_len)

        Returns:
            Boolean mask of shape (batch, 1, 1, seq_len).
        """
        return (tokens == pad_id).unsqueeze(1).unsqueeze(2)

    # ----- Forward -----

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            src: Source token IDs — (batch, src_len).
            tgt: Target token IDs (teacher forcing) — (batch, tgt_len).

        Returns:
            Logits of shape (batch, tgt_len, vocab_size).
        """
        # --- Encode ---
        if self.use_pretrained_encoder:
            # BERT handles its own embeddings, positions, and attention masks
            attention_mask = (src != self.config.pad_token_id).long()
            bert_output = self.bert_encoder(
                input_ids=src, attention_mask=attention_mask
            )
            encoder_output = self.encoder_projection(
                bert_output.last_hidden_state
            )
            # Build src padding mask for cross-attention in decoder
            src_pad_mask = self.generate_padding_mask(
                src, self.config.pad_token_id
            )
        else:
            # Original from-scratch encoding
            src_pad_mask = self.generate_padding_mask(
                src, self.config.pad_token_id
            )
            src_emb = self.pos_encoding(
                self.src_embedding(src) * math.sqrt(self.config.d_model)
            )
            encoder_output = self.encoder(src_emb, src_pad_mask)

        # --- Decode (always the same) ---
        tgt_pad_mask = self.generate_padding_mask(tgt, self.config.pad_token_id)
        tgt_causal_mask = self.generate_causal_mask(tgt.size(1), tgt.device)
        tgt_mask = tgt_pad_mask | tgt_causal_mask

        tgt_emb = self.pos_encoding(
            self.tgt_embedding(tgt) * math.sqrt(self.config.d_model)
        )
        decoder_output = self.decoder(
            tgt_emb, encoder_output, tgt_mask, src_pad_mask
        )

        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        return logits

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
