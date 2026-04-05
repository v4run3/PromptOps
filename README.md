# PromptOps

**Design and Evaluation of a CI/CD Pipeline for Prompt Template Management in Large Language Model Applications**

---

## Overview

PromptOps is a modular, Python-based MLOps platform for managing, evaluating, and serving LLM-powered dialogue summarization pipelines. It provides a FastAPI backend, a modern dark-themed web dashboard, structured YAML-based prompt storage, Pydantic schema validation, a custom-trained 91M-parameter Seq2Seq Transformer, a HuggingFace BART baseline, a QA module powered by Flan-T5, ROUGE-based quality gates, and CI/CD integration via GitHub Actions.

---

## Project Structure

```
promptops/
├── app/
│   ├── main.py                  # FastAPI entry-point (Jinja2 + API routers)
│   ├── schemas.py               # Pydantic request/response models
│   ├── loader.py                # YAML prompt loader
│   ├── routes/
│   │   ├── api.py               # All REST API endpoints
│   │   └── dashboard.py         # Page route handlers (HTML responses)
│   ├── static/
│   │   └── script.js            # Frontend JS (summarize, QA, metrics, inference panel)
│   └── templates/
│       ├── dashboard.html       # Main summarization dashboard
│       ├── prompts.html         # Prompt template management page
│       ├── evaluation.html      # ROUGE evaluation leaderboard & charts
│       ├── models.html          # Model architecture comparison page
│       ├── cicd.html            # Live CI/CD GitHub Actions log viewer
│       └── settings.html        # Detailed model config & quality gate status
├── model/
│   ├── config.py                # Hyperparameter configuration (ModelConfig)
│   ├── transformer.py           # From-scratch Encoder-Decoder Transformer (91M params)
│   ├── dataset.py               # SAMSum + DialogSum dataset loading & preprocessing
│   ├── train.py                 # Training loop with resume & checkpointing
│   └── inference.py             # Greedy & Beam Search decoding for summarization
├── scripts/
│   ├── train_model.py           # CLI entry-point for training (Phase 1, 2 & 3)
│   └── quality_gate.py          # MLOps Quality Gate: ROUGE-based pass/fail evaluation
├── prompts/
│   └── example.yaml             # Example prompt template
├── eval/
│   ├── evaluator.py             # ROUGE-1, ROUGE-2, ROUGE-L evaluation module
│   └── results.json             # Latest evaluation results (auto-generated)
├── tests/
│   ├── test_api.py              # API endpoint integration tests
│   └── test_model.py            # Transformer model unit tests
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI (lint + pytest)
├── Dockerfile                   # Container image
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
PYTHONPATH=. uvicorn app.main:app --reload --port 8000
```

The dashboard will be available at `http://127.0.0.1:8000`.

### 3. Run Tests

```bash
python -m pytest tests/ -v
```

---

## API Endpoints

| Endpoint                | Method | Description                                          |
| ----------------------- | ------ | ---------------------------------------------------- |
| `/api/health`           | GET    | Server health check                                  |
| `/api/summarize`        | POST   | Summarize dialogue (Custom or BART SAMSum)           |
| `/api/qa`               | POST   | Answer questions about a dialogue (Flan-T5 + rules)  |
| `/api/metrics`          | GET    | Live dashboard metrics (prompts, ROUGE, model)       |
| `/api/settings`         | GET    | Full model config, training params & quality gate    |
| `/api/evaluation/data`  | GET    | ROUGE leaderboard: Custom vs. BART baseline          |
| `/api/cicd/logs`        | GET    | Live GitHub Actions workflow runs (last 15)          |
| `/docs`                 | GET    | Interactive Swagger UI                               |

---

## Dashboard UI

PromptOps includes a fully functional dark-themed web dashboard with 6 pages:

### 🏠 Dashboard (`/`)
- Dialogue input with live **summarization** using the selected model
- **Model selector** — switch between `BERT-Seq2Seq Phase 3` (custom) and `BART SAMSum` (pretrained)
- **Length Profile toggle** — `Long Dialogue (Max 142 tokens)` vs `Short Dialogue (Max 60 tokens)` to dynamically control BART generation parameters and reduce hallucinations on short inputs
- **Inference Stats panel** — topbar button showing Last Latency, Total Requests, Average Latency, Model Load Status, and Server Health
- **Q&A mode** — ask natural language questions about the dialogue, powered by a rule-based speaker parser with Flan-T5 fallback
- Live history sidebar showing recent summarization queries

### 📋 Prompts (`/prompts`)
- View and manage YAML prompt templates
- Browse registered prompt versions and their input variables

### 📊 Evaluation (`/evaluation`)
- ROUGE-1, ROUGE-2, ROUGE-L score leaderboard comparing the custom model vs. BART SAMSum baseline
- Training loss & validation curves per phase
- Quality Gate pass/fail status display

### 🤖 Models (`/models`)
- Side-by-side architecture comparison of all 3 model variants
- Training phase timeline and configuration details

### 🔁 CI/CD Logs (`/cicd`)
- Live GitHub Actions workflow run viewer (fetches last 15 runs via GitHub API)
- Status indicators for each pipeline run (success / failure / in-progress)

### ⚙️ Settings (`/settings`)
- Full model hyperparameter configuration (custom + BART + BERT-Seq2Seq)
- Quality Gate threshold and last evaluation timestamp
- System info: checkpoint path, prompt count, model load status

---

## Models

### 1. 🟣 Custom Seq2Seq Transformer — `BERT-Seq2Seq Phase 3` (Default)

A from-scratch Encoder-Decoder Transformer trained in 3 phases:

| Hyperparameter       | Value  | Description                          |
| :------------------- | :----- | :----------------------------------- |
| `d_model`            | 512    | Hidden dimension / Embedding size    |
| `n_heads`            | 8      | Number of attention heads            |
| `n_encoder_layers`   | 6      | Encoder blocks                       |
| `n_decoder_layers`   | 6      | Decoder blocks                       |
| `d_ff`               | 2048   | Feed-forward network dimension       |
| `max_seq_len`        | 256    | Max tokens for dialogue & summary    |
| `vocab_size`         | 30522  | BERT uncased tokenizer vocabulary    |
| `dropout`            | 0.1    | Regularization                       |

**Phase 3 Architecture (Current Best):** BERT-base-uncased encoder (110M params) + custom Transformer decoder, with a projection layer from BERT's 768-dim hidden space to the model's 512-dim space. The BERT encoder is frozen for the first 3 epochs.

### 2. 🟠 BART SAMSum — `philschmid/bart-large-cnn-samsum`

- HuggingFace pretrained BART-Large (406M params) fine-tuned on SAMSum and CNN/DailyMail
- Used as the **comparison baseline** in evaluation
- **Dynamic length control** via the UI toggle:
  - `Short Dialogue`: `max_length=60`, `min_length=10`, `length_penalty=0.5`
  - `Long Dialogue`: `max_length=142`, `min_length=30`, `length_penalty=1.0`
- Uses `no_repeat_ngram_size=3` and `early_stopping=True` to limit hallucinations

### 3. 🟢 Flan-T5 — `google/flan-t5-base`

- Used exclusively for the **Q&A endpoint** as a fallback
- First, a rule-based speaker parser attempts to answer "who" questions from dialogue turn structure
- If the rule-based system can't find a confident answer, Flan-T5 runs generative QA
- Loaded lazily (only on first Q&A request) to save memory

---

## Model Training & Phases

### Datasets

| Dataset      | Source                                                                   | Samples  | Domain                           |
| :----------- | :----------------------------------------------------------------------- | :------- | :--------------------------------|
| **SAMSum**   | [knkarthick/samsum](https://huggingface.co/datasets/knkarthick/samsum)   | ~16,000  | Messenger-style chat dialogues   |
| **DialogSum**| [knkarthick/dialogsum](https://huggingface.co/datasets/knkarthick/dialogsum) | ~13,500 | Daily task-oriented dialogues  |

### Training Commands

```bash
# Phase 1: Custom Transformer trained from scratch on SAMSum
PYTHONPATH=. python scripts/train_model.py --epochs 50 --batch_size 32

# Phase 2: Custom Transformer trained on SAMSum + DialogSum combined
PYTHONPATH=. python scripts/train_model.py --epochs 50 --batch_size 32 --datasets samsum,dialogsum

# Phase 3: BERT-Pretrained Encoder with Custom Decoder (Current Best)
PYTHONPATH=. python scripts/train_model.py --epochs 10 --batch_size 16 \
    --datasets samsum,dialogsum \
    --pretrained_encoder \
    --freeze_epochs 3
```

---

## Quality Gate (MLOps)

The `scripts/quality_gate.py` script runs batch inference on the test set and enforces a minimum ROUGE-L threshold. This acts as an automated model regression test in the CI/CD pipeline.

```bash
PYTHONPATH=. python scripts/quality_gate.py \
    --checkpoint checkpoints/best_model.pt \
    --threshold 0.20 \
    --samples 50
```

If the ROUGE-L score falls below the threshold, the script exits with code `1` (pipeline failure). Results are saved to `eval/results.json` and surfaced live in the Evaluation and Settings pages.

---

## CI/CD Pipeline

GitHub Actions (`.github/workflows/ci.yml`) runs on every push to `main`:

1. **Lint** — `ruff` code quality checks
2. **Test** — `pytest tests/` unit & integration tests
3. **Quality Gate** — ROUGE-L threshold check on the latest checkpoint

Live run status is viewable directly in the **CI/CD Logs** page of the dashboard.

---

## Docker

Build the container image from the project root:

```bash
docker build -t promptops:latest .
```

Run the container locally:

```bash
docker run --rm -p 8000:8000 promptops:latest
```

The application will be available at `http://127.0.0.1:8000` and the container health check targets `GET /health`.

### Docker Notes

- The image runs the FastAPI app on port `8000`
- The `Dockerfile` copies only the runtime application assets needed by the server
- Large model downloads may still happen on first request if Hugging Face artifacts are not already available
- If you rely on `checkpoints/best_model.pt`, make sure that file exists before building or mount it separately in your deployment environment

---

## Kubernetes

Starter manifests are available in the `k8s/` directory:

- `k8s/namespace.yaml`
- `k8s/deployment.yaml`
- `k8s/service.yaml`
- `k8s/ingress.yaml`

### 1. Build the Image

```bash
docker build -t promptops:latest .
```

### 2. Make the Image Available to Your Cluster

If you are using a local cluster, load or reuse the image depending on your setup.

For `kind`:

```bash
kind load docker-image promptops:latest
```

For Docker Desktop Kubernetes, the local Docker image is usually available directly.

### 3. Create the Namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

### 4. Deploy the Application

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### 5. Verify the Deployment

```bash
kubectl get pods -n promptops
kubectl rollout status deployment/promptops -n promptops
kubectl get svc -n promptops
```

### 6. Access the App Locally

Use port-forwarding for the quickest local test:

```bash
kubectl port-forward svc/promptops 8000:80 -n promptops
```

Then open:

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/health`

### 7. Enable Ingress

If your cluster has an ingress controller installed:

```bash
kubectl apply -f k8s/ingress.yaml
```

The sample ingress uses the host `promptops.local`. For local testing, map that host to `127.0.0.1` in your system hosts file.

### Kubernetes Notes

- The deployment exposes container port `8000`
- Readiness and liveness probes both call `/health`
- The current manifest uses `image: promptops:latest`, which is suitable for local clusters
- For a remote cluster, push the image to a container registry and update the image reference in `k8s/deployment.yaml`
- If the app needs a trained checkpoint or persistent generated data, add a `PersistentVolumeClaim` and mount it into the container
- If outbound internet is restricted, pre-bake or mount Hugging Face model files because the app may download them at runtime

---

## License

This project is part of a master's-level research study at KJSCE.
