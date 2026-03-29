from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "project_name": "PromptOps",
        "version": "v1.0"
    })

@router.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "project_name": "PromptOps",
        "version": "v1.0"
    })

@router.get("/prompts", response_class=HTMLResponse)
async def prompts(request: Request):
    return templates.TemplateResponse("prompts.html", {
        "request": request,
        "project_name": "PromptOps",
        "version": "v1.0"
    })

@router.get("/models", response_class=HTMLResponse)
async def models(request: Request):
    return templates.TemplateResponse("models.html", {
        "request": request,
        "project_name": "PromptOps",
        "version": "v1.0"
    })

@router.get("/cicd", response_class=HTMLResponse)
async def cicd_logs(request: Request):
    return templates.TemplateResponse("cicd.html", {
        "request": request,
        "project_name": "PromptOps",
        "version": "v1.0"
    })

@router.get("/evaluation", response_class=HTMLResponse)
async def evaluation(request: Request):
    return templates.TemplateResponse("evaluation.html", {
        "request": request,
        "project_name": "PromptOps",
        "version": "v1.0"
    })

