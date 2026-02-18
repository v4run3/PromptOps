import os
from fastapi import APIRouter, HTTPException
from app.schemas import SummarizationRequest, SummarizationResponse
from model.inference import summarize


router = APIRouter(prefix="/api")

# Lazy model state (optional enhancement: use a proper singleton)
CHECKPOINT_PATH = "checkpoints/best_model.pt"

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
    return {
        "active_prompts": 12,
        "latest_score": 0.87,
        "model_version": "transformer-base-v1"
    }
