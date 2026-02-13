# PromptOps

**Design and Evaluation of a CI/CD Pipeline for Prompt Template Management in Large Language Model Applications**

---

## Overview

PromptOps is a modular, Python-based system for managing, validating, and evaluating prompt templates used in LLM-powered applications. It provides a FastAPI backend, structured YAML-based prompt storage, Pydantic schema validation, and CI/CD integration via GitHub Actions.

## Project Structure

```
promptops/
├── app/
│   ├── __init__.py          # App package
│   ├── main.py              # FastAPI entry-point
│   ├── schemas.py           # Pydantic prompt models
│   └── loader.py            # YAML prompt loader
├── model/
│   ├── __init__.py          # Model package
│   ├── config.py            # Hyperparameter configuration
│   ├── transformer.py       # From-scratch Encoder-Decoder Transformer (~10M params)
│   ├── dataset.py           # SAMSum dataset loading & preprocessing
│   ├── train.py             # Training loop with checkpointing
│   └── inference.py         # Greedy decoding for summarization
├── scripts/
│   └── train_model.py       # CLI entry-point for training
├── prompts/
│   └── example.yaml         # Example prompt template
├── eval/
│   ├── __init__.py          # Eval package
│   └── evaluator.py         # Evaluation placeholder
├── tests/
│   ├── __init__.py          # Tests package
│   ├── test_health.py       # Health & prompt tests
│   └── test_model.py        # Transformer model unit tests
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI
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

### 3. Test Endpoints

| Endpoint              | Method | Description                     |
| --------------------- | ------ | ------------------------------- |
| `/health`             | GET    | Health check                    |
| `/prompts/{name}`     | GET    | Load a prompt template by name  |
| `/docs`               | GET    | Interactive Swagger UI          |

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

## Model Training

The project includes a custom ~10M parameter Encoder-Decoder Transformer for dialogue summarization, trained on the [SAMSum](https://huggingface.co/datasets/Samsung/samsum) dataset.

### Train the Model

```bash
# Full training
python scripts/train_model.py --epochs 10 --batch_size 32

# Smoke test (fast, small subset)
python scripts/train_model.py --epochs 2 --batch_size 4 --max_samples 100
```

### Model Architecture

| Hyperparameter     | Value  |
|--------------------|--------|
| `d_model`          | 256    |
| `n_heads`          | 4      |
| `n_encoder_layers` | 4      |
| `n_decoder_layers` | 4      |
| `d_ff`             | 1024   |
| `max_seq_len`      | 256    |
| `dropout`          | 0.1    |
| `vocab_size`       | 30,522 |

## Docker

```bash
docker build -t promptops .
docker run -p 8000:8000 promptops
```

## Roadmap

- [ ] Prompt versioning and diff tracking
- [ ] LLM API integration (OpenAI, Anthropic)
- [ ] Evaluation metrics (BLEU, ROUGE, custom rubrics)
- [ ] Prompt A/B testing framework
- [ ] Dashboard UI for prompt management
- [ ] Advanced CI/CD with automated prompt regression tests

## License

This project is part of a master's-level research study.


Test text
