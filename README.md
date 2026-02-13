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
├── prompts/
│   └── example.yaml         # Example prompt template
├── eval/
│   ├── __init__.py          # Eval package
│   └── evaluator.py         # Evaluation placeholder
├── tests/
│   ├── __init__.py          # Tests package
│   └── test_health.py       # Health & prompt tests
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
