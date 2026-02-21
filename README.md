# PromptOps

**Design and Evaluation of a CI/CD Pipeline for Prompt Template Management in Large Language Model Applications**

---

## Overview

PromptOps is a modular, Python-based system for managing, validating, and evaluating prompt templates used in LLM-powered applications. It provides a FastAPI backend, structured YAML-based prompt storage, Pydantic schema validation, a custom-trained 91M-parameter Transformer for dialogue summarization, ROUGE-based evaluation metrics, and CI/CD integration via GitHub Actions.

## Project Structure

```
promptops/
├── app/
│   ├── main.py              # FastAPI entry-point (Jinja2 + API router)
│   ├── schemas.py           # Pydantic request/response models
│   ├── loader.py            # YAML prompt loader
│   └── routes/
│       ├── api.py           # /summarize & /health endpoints
│       └── dashboard.py     # Dashboard HTML endpoint
├── model/
│   ├── config.py            # Hyperparameter configuration (ModelConfig)
│   ├── transformer.py       # From-scratch Encoder-Decoder Transformer (91M params)
│   ├── dataset.py           # SAMSum + DialogSum dataset loading & preprocessing
│   ├── train.py             # Training loop with resume & checkpointing
│   └── inference.py         # Greedy & Beam Search decoding for summarization
├── scripts/
│   ├── train_model.py       # CLI entry-point for training (Phase 1 & Phase 2)
│   └── quality_gate.py      # MLOps Quality Gate: ROUGE-based pass/fail evaluation
├── prompts/
│   └── example.yaml         # Example prompt template
├── eval/
│   └── evaluator.py         # ROUGE-1, ROUGE-2, ROUGE-L evaluation module
├── tests/
│   ├── test_api.py          # API endpoint tests
│   └── test_model.py        # Transformer model unit tests
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI (lint + pytest)
├── Dockerfile               # Container image
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### 3. API Endpoints

| Endpoint          | Method | Description                        |
| ----------------- | ------ | ---------------------------------- |
| `/health`         | GET    | Health check                       |
| `/summarize`      | POST   | Summarize a dialogue using the model |
| `/prompts/{name}` | GET    | Load a prompt template by name     |
| `/docs`           | GET    | Interactive Swagger UI             |

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

## Model Training

The project includes a custom **91M parameter Encoder-Decoder Transformer** for dialogue summarization.

### Datasets

| Dataset | Source | Samples | Domain |
| :--- | :--- | :--- | :--- |
| **SAMSum** | [knkarthick/samsum](https://huggingface.co/datasets/knkarthick/samsum) | ~16,000 | Messenger-style chat dialogues |
| **DialogSum** (Phase 2) | [knkarthick/dialogsum](https://huggingface.co/datasets/knkarthick/dialogsum) | ~13,500 | Daily task-oriented dialogues |

### Train the Model

```bash
# Phase 1: SAMSum only (default)
PYTHONPATH=. python scripts/train_model.py --epochs 50 --batch_size 32

# Phase 2: SAMSum + DialogSum combined
PYTHONPATH=. python scripts/train_model.py --epochs 50 --batch_size 32 --datasets samsum,dialogsum

# Smoke test (fast, small subset)
PYTHONPATH=. python scripts/train_model.py --epochs 2 --batch_size 4 --max_samples 100

# Resume training: automatically resumes from checkpoints/best_model.pt if it exists
# Save to Google Drive for Colab persistence:
PYTHONPATH=. python scripts/train_model.py --epochs 50 --batch_size 32 \
    --datasets samsum,dialogsum \
    --checkpoint_dir /content/drive/MyDrive/PromptOps/checkpoints
```

### Model Architecture

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| `d_model` | 512 | Hidden dimension / Embedding size |
| `n_heads` | 8 | Number of attention heads |
| `n_encoder_layers` | 6 | Number of encoder blocks |
| `n_decoder_layers` | 6 | Number of decoder blocks |
| `d_ff` | 2048 | Feed-forward network inner dimension |
| `max_seq_len` | 256 | Max tokens for dialogue & summary |
| `vocab_size` | 30522 | Tokenizer vocabulary (BERT uncased) |
| `dropout` | 0.1 | Regularization |

### Quality Gate (MLOps)

The `scripts/quality_gate.py` script runs batch inference on the test set and enforces a minimum ROUGE-L threshold. This acts as an automated "model regression test" in the CI/CD pipeline.

```bash
PYTHONPATH=. python scripts/quality_gate.py \
    --checkpoint checkpoints/best_model.pt \
    --threshold 0.20 \
    --samples 50
```

If the ROUGE-L score falls below the threshold, the script exits with code `1` (failure).

## Docker

```bash
docker build -t promptops .
docker run -p 8000:8000 promptops
```

## Roadmap

- [x] Custom Transformer model (91M params) trained on SAMSum
- [x] ROUGE-1, ROUGE-2, ROUGE-L evaluation metrics
- [x] MLOps Quality Gate script
- [x] CI/CD with GitHub Actions (lint + tests)
- [x] Phase 2: Data augmentation with DialogSum
- [ ] Prompt versioning and diff tracking
- [ ] Dashboard UI for prompt management
- [ ] LLM API integration (OpenAI, Anthropic) for comparative evaluation
- [ ] Prompt A/B testing framework

## License

This project is part of a master's-level research study.
