"""
Common validation utilities for the Conclave simulation.

This module provides standardized validation functions that are used
throughout the codebase to ensure data integrity and prevent errors.
"""

from typing import Any, List, Optional, Union
from .exceptions import ValidationError, AgentError


def validate_agent(agent: Any, allow_none: bool = False, context: str = "") -> None:
    """
    Validate that an agent object is valid and has required attributes.
    
    Args:
        agent: The agent object to validate
        allow_none: Whether to allow None values
        context: Additional context for error messages
        
    Raises:
        ValidationError: If the agent is invalid
    """
    if agent is None:
        if allow_none:
            return
        raise ValidationError(
            f"Agent cannot be None{' in ' + context if context else ''}",
            field_name="agent"
        )
    
    # Check basic required attributes
    required_attrs = ['id', 'name']
    for attr in required_attrs:
        if not hasattr(agent, attr):
            raise AgentError(
                f"Agent missing required attribute '{attr}'{' in ' + context if context else ''}",
                agent_id=getattr(agent, 'id', 'unknown')
            )


def validate_list_not_empty(
    items: List[Any], 
    name: str = "list", 
    context: str = "",
    min_length: int = 1
) -> None:
    """
    Validate that a list is not empty and meets minimum length requirements.
    
    Args:
        items: The list to validate
        name: The name of the list for error messages
        context: Additional context for error messages
        min_length: Minimum required length
        
    Raises:
        ValidationError: If the list is empty or too short
    """
    if not items:
        raise ValidationError(
            f"{name.title()} cannot be empty{' in ' + context if context else ''}",
            field_name=name,
            field_value=items
        )
    
    if len(items) < min_length:
        raise ValidationError(
            f"{name.title()} must have at least {min_length} item(s), got {len(items)}"
            f"{' in ' + context if context else ''}",
            field_name=name,
            field_value=len(items)
        )


def validate_range(
    value: Union[int, float], 
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    name: str = "value",
    context: str = "",
    inclusive: bool = True
) -> None:
    """
    Validate that a numeric value is within a specified range.
    
    Args:
        value: The value to validate
        min_val: Minimum allowed value (None for no minimum)
        max_val: Maximum allowed value (None for no maximum)
        name: The name of the value for error messages
        context: Additional context for error messages
        inclusive: Whether the range bounds are inclusive
        
    Raises:
        ValidationError: If the value is outside the allowed range
    """
    if min_val is not None:
        if inclusive and value < min_val:
            raise ValidationError(
                f"{name.title()} must be >= {min_val}, got {value}"
                f"{' in ' + context if context else ''}",
                field_name=name,
                field_value=value
            )
        elif not inclusive and value <= min_val:
            raise ValidationError(
                f"{name.title()} must be > {min_val}, got {value}"
                f"{' in ' + context if context else ''}",
                field_name=name,
                field_value=value
            )
    
    if max_val is not None:
        if inclusive and value > max_val:
            raise ValidationError(
                f"{name.title()} must be <= {max_val}, got {value}"
                f"{' in ' + context if context else ''}",
                field_name=name,
                field_value=value
            )
        elif not inclusive and value >= max_val:
            raise ValidationError(
                f"{name.title()} must be < {max_val}, got {value}"
                f"{' in ' + context if context else ''}",
                field_name=name,
                field_value=value
            )


def validate_agent_id(
    agent_id: int, 
    num_agents: int, 
    context: str = "",
    allow_negative: bool = False
) -> None:
    """
    Validate that an agent ID is within valid bounds.
    
    Args:
        agent_id: The agent ID to validate
        num_agents: Total number of agents
        context: Additional context for error messages
        allow_negative: Whether to allow negative IDs
        
    Raises:
        ValidationError: If the agent ID is invalid
    """
    min_val = -1 if allow_negative else 0
    max_val = num_agents - 1
    
    validate_range(
        agent_id,
        min_val=min_val,
        max_val=max_val,
        name="agent_id",
        context=context
    )


def validate_string_not_empty(
    value: str, 
    name: str = "string",
    context: str = "",
    strip_whitespace: bool = True
) -> None:
    """
    Validate that a string is not empty or just whitespace.
    
    Args:
        value: The string to validate
        name: The name of the string for error messages
        context: Additional context for error messages
        strip_whitespace: Whether to strip whitespace before checking
        
    Raises:
        ValidationError: If the string is empty
    """
    if value is None:
        raise ValidationError(
            f"{name.title()} cannot be None{' in ' + context if context else ''}",
            field_name=name,
            field_value=value
        )
    
    check_value = value.strip() if strip_whitespace else value
    if not check_value:
        raise ValidationError(
            f"{name.title()} cannot be empty{' in ' + context if context else ''}",
            field_name=name,
            field_value=value
        )


def validate_dict_has_keys(
    data: dict, 
    required_keys: List[str], 
    name: str = "dictionary",
    context: str = ""
) -> None:
    """
    Validate that a dictionary contains all required keys.
    
    Args:
        data: The dictionary to validate
        required_keys: List of required key names
        name: The name of the dictionary for error messages
        context: Additional context for error messages
        
    Raises:
        ValidationError: If any required keys are missing
    """
    if not isinstance(data, dict):
        raise ValidationError(
            f"{name.title()} must be a dictionary, got {type(data).__name__}"
            f"{' in ' + context if context else ''}",
            field_name=name,
            field_value=type(data).__name__
        )
    
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValidationError(
            f"{name.title()} missing required keys: {missing_keys}"
            f"{' in ' + context if context else ''}",
            field_name=name,
            field_value=missing_keys
        )


def validate_choice(
    value: Any,
    valid_choices: List[Any],
    name: str = "value",
    context: str = ""
) -> None:
    """
    Validate that a value is one of the allowed choices.
    
    Args:
        value: The value to validate
        valid_choices: List of valid choices
        name: The name of the value for error messages
        context: Additional context for error messages
        
    Raises:
        ValidationError: If the value is not in valid choices
    """
    if value not in valid_choices:
        raise ValidationError(
            f"{name.title()} must be one of {valid_choices}, got {value}"
            f"{' in ' + context if context else ''}",
            field_name=name,
            field_value=value
        )
