"""Pydantic models for prompt template validation."""

from typing import Optional

from pydantic import BaseModel


class PromptTemplate(BaseModel):
    """Schema for a prompt template stored as YAML."""

    name: str
    version: str
    template: str
    input_variables: list[str]
    metadata: Optional[dict] = None


class SummarizationRequest(BaseModel):
    """Request schema for summarization."""

    dialogue: str
    num_beams: Optional[int] = 5


class SummarizationResponse(BaseModel):
    """Response schema for summarization."""

    summary: str
    model_version: str

    model_config = {"protected_namespaces": ()}
