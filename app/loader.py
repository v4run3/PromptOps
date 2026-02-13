"""Prompt template loader — reads YAML files and validates via Pydantic."""

from pathlib import Path

import yaml

from app.schemas import PromptTemplate

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> PromptTemplate:
    """Load a prompt template by name from the prompts/ directory.

    Args:
        name: Filename (without .yaml extension) inside prompts/.

    Returns:
        A validated PromptTemplate instance.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        pydantic.ValidationError: If the YAML content fails schema validation.
    """
    filepath = PROMPTS_DIR / f"{name}.yaml"

    if not filepath.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return PromptTemplate(**data)
