"""Model hyperparameter configuration.

Single source of truth for all Transformer architecture parameters.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Hyperparameters for the Seq2Seq Transformer.

    Attributes:
        d_model:          Embedding / hidden dimension.
        n_heads:          Number of attention heads.
        n_encoder_layers: Number of encoder blocks.
        n_decoder_layers: Number of decoder blocks.
        d_ff:             Feed-forward inner dimension.
        max_seq_len:      Maximum sequence length (source & target).
        dropout:          Dropout probability.
        vocab_size:       Tokenizer vocabulary size (bert-base-uncased = 30 522).
        pad_token_id:     Token ID used for padding.
        bos_token_id:     Token ID used for beginning-of-sequence (decoder start).
        eos_token_id:     Token ID used for end-of-sequence.
        label_smoothing:  Label smoothing factor for cross-entropy loss.
    """

    d_model: int = 256
    n_heads: int = 4
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    vocab_size: int = 30_522
    pad_token_id: int = 0
    bos_token_id: int = 101   # [CLS] in bert-base-uncased
    eos_token_id: int = 102   # [SEP] in bert-base-uncased
    label_smoothing: float = 0.1
