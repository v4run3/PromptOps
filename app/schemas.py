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
    num_beams: Optional[int] = 8
    model_choice: Optional[str] = "custom"
    length_profile: Optional[str] = "long"

    model_config = {"protected_namespaces": ()}


class QARequest(BaseModel):
    """Request schema for question answering."""
    dialogue: str
    question: str


class QAResponse(BaseModel):
    """Response schema for question answering."""
    answer: str
    model_version: str


class SummarizationResponse(BaseModel):
    """Response schema for summarization."""

    summary: str
    model_version: str

    model_config = {"protected_namespaces": ()}
