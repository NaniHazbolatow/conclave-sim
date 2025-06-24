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
    properties: Dict[str, ToolParameterProperty] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)

class ToolDefinition(BaseModel):
    """Pydantic model for a single tool definition from prompts.yaml."""
    name: str
    description: str
    parameters: Optional[ToolParameters] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the tool definition to a dictionary suitable for an LLM API call."""
        tool_dict = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
            }
        }
        # Ensure parameters are included only if they exist, otherwise use a default empty schema
        if self.parameters and self.parameters.properties:
            tool_dict["function"]["parameters"] = self.parameters.model_dump()
        else:
            tool_dict["function"]["parameters"] = {"type": "object", "properties": {}, "required": []}
        return tool_dict

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
