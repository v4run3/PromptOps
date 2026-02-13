from fastapi import APIRouter

router = APIRouter(prefix="/api")

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.get("/metrics")
async def metrics():
    return {
        "active_prompts": 12,
        "latest_score": 0.87,
        "model_version": "tiny-transformer-v2"
    }
