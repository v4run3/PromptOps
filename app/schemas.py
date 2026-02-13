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
