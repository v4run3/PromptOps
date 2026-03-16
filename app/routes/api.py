import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from app.schemas import SummarizationRequest, SummarizationResponse
from model.inference import summarize


router = APIRouter(prefix="/api")

# Lazy model state (optional enhancement: use a proper singleton)
CHECKPOINT_PATH = "checkpoints/best_model.pt"
PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
EVAL_RESULTS_PATH = Path(__file__).resolve().parent.parent.parent / "eval" / "results.json"

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.post("/summarize", response_model=SummarizationResponse)
async def run_summarization(request: SummarizationRequest):
    """Generate a summary for the given dialogue."""
    if not os.path.exists(CHECKPOINT_PATH):
        # Fallback for when the model hasn't finished training yet
        return SummarizationResponse(
            summary="[SYSTEM] Model checkpoint not found. Please train the model in Colab first.",
            model_version="none"
        )
    
    try:
        summary = summarize(
            request.dialogue,
            checkpoint_path=CHECKPOINT_PATH,
            num_beams=request.num_beams
        )
        return SummarizationResponse(
            summary=summary,
            model_version="transformer-base-v1"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def metrics():
    """Return real metrics from the filesystem and eval results."""
    # Count actual prompt YAML files
    prompt_count = 0
    if PROMPTS_DIR.exists():
        prompt_count = len(list(PROMPTS_DIR.glob("*.yaml")))

    # Read evaluation results
    rouge_l = 0.0
    model_name = "No model trained"
    phase = 0

    if EVAL_RESULTS_PATH.exists():
        try:
            with open(EVAL_RESULTS_PATH, "r") as f:
                results = json.load(f)
            rouge_l = results.get("rouge_l", 0.0)
            model_name = results.get("model_name", "unknown")
            phase = results.get("phase", 0)
        except (json.JSONDecodeError, KeyError):
            pass

    # Check if checkpoint exists
    has_model = os.path.exists(CHECKPOINT_PATH)

    return {
        "active_prompts": prompt_count,
        "latest_score": round(rouge_l, 4),
        "model_version": model_name if has_model else "No checkpoint found",
        "phase": phase,
    }


@router.get("/settings")
async def settings():
    """Return model configuration, training params, and system info."""
    from model.config import ModelConfig
    config = ModelConfig()

    # Read eval results for quality gate info
    eval_data = {}
    if EVAL_RESULTS_PATH.exists():
        try:
            with open(EVAL_RESULTS_PATH, "r") as f:
                eval_data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass

    # Count prompts
    prompt_count = 0
    if PROMPTS_DIR.exists():
        prompt_count = len(list(PROMPTS_DIR.glob("*.yaml")))

    return {
        "architecture": {
            "type": "Seq2Seq Transformer",
            "encoder": "BERT-base-uncased" if config.use_pretrained_encoder else "Custom Transformer",
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_encoder_layers": config.n_encoder_layers,
            "n_decoder_layers": config.n_decoder_layers,
            "d_ff": config.d_ff,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "dropout": config.dropout,
        },
        "training": {
            "use_pretrained_encoder": config.use_pretrained_encoder,
            "bert_hidden_size": config.bert_hidden_size,
            "freeze_encoder_epochs": config.freeze_encoder_epochs,
            "label_smoothing": config.label_smoothing,
        },
        "quality_gate": {
            "threshold": eval_data.get("threshold", 0.10),
            "status": eval_data.get("quality_gate", "No data"),
            "timestamp": eval_data.get("timestamp", None),
        },
        "system": {
            "checkpoint_path": CHECKPOINT_PATH,
            "checkpoint_exists": os.path.exists(CHECKPOINT_PATH),
            "prompts_dir": str(PROMPTS_DIR),
            "prompt_count": prompt_count,
        },
    }
