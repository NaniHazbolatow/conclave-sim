\
# filepath: /Users/kbverlaan/Library/Mobile Documents/com~apple~CloudDocs/Universiteit/Computational Science/ABM/conclave-sim/conclave/config/prompt_models.py
"""Pydantic models for parsing prompts.yaml and tool definitions."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class ToolParameterProperty(BaseModel):
    """Defines a single property within a tool's parameters."""
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None

class ToolParameters(BaseModel):
    """Defines the parameters object for a tool, following OpenAI structure."""
    type: str = "object"
    properties: Dict[str, ToolParameterProperty]
    required: Optional[List[str]] = Field(default_factory=list)

class ToolDefinition(BaseModel):
    """Pydantic model for a single tool definition from prompts.yaml."""
    name: str
    description: str
    parameters: ToolParameters
    example_json: Optional[str] = None
    additional_prompt_instructions: Optional[str] = None

    # For compatibility if model_dump() is called elsewhere
    def model_dump(self, **kwargs):
        return self.dict(**kwargs)

class PromptsConfig(BaseModel):
    """Pydantic model for the entire prompts.yaml structure."""
    tool_definitions: Dict[str, ToolDefinition] = Field(default_factory=dict)
    
    # Catch-all for other top-level keys in prompts.yaml which are prompt strings
    # Pydantic will populate this with any fields not explicitly defined above.
    # We expect these to be prompt strings.
    # To make it more explicit, we can iterate in __init__ or use a root validator
    # For now, this relies on dynamic access in PromptLoader.
    # Alternatively, define known prompt keys if they are fixed, or use Extra.allow
    class Config:
        extra = "allow" # Allows other fields (prompt names)

    # If you want to ensure all extra fields are strings (prompts)
    # you might need a validator or handle it in PromptLoader.
    # For now, PromptLoader accesses them by key directly.
