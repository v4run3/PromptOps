"""Tests for the PromptOps FastAPI application."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health_check():
    """GET /health should return 200 with status ok."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_load_example_prompt():
    """GET /prompts/example should return the example prompt template."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/prompts/example")

    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "example"
    assert data["version"] == "1.0"
    assert "context" in data["input_variables"]
    assert "question" in data["input_variables"]


@pytest.mark.asyncio
async def test_prompt_not_found():
    """GET /prompts/nonexistent should return 404."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/prompts/nonexistent")

    assert response.status_code == 404
