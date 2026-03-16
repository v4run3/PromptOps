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

