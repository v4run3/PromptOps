"""FastAPI application entry-point for PromptOps."""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles


from app.loader import load_prompt
from app.routes import dashboard, api
from app.schemas import PromptTemplate

app = FastAPI(
    title="PromptOps Dashboard",
    description="CI/CD Pipeline for Prompt Template Management in LLM Applications",
    version="0.1.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routes
app.include_router(dashboard.router)
app.include_router(api.router)


@app.get("/health")
def health_check():
    """Liveness / readiness probe."""
    return {"status": "ok"}


@app.get("/prompts/{name}", response_model=PromptTemplate)
def get_prompt(name: str):
    """Return a validated prompt template by name."""
    try:
        return load_prompt(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
