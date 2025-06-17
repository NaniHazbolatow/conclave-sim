"""
Conclave Utilities Package

This package provides common utilities for logging, validation, error handling,
and formatting that are used throughout the Conclave simulation codebase.
"""

from .logging import get_logger, setup_logging, configure_component_loggers, log_error_with_context
from .validation import (
    validate_agent, validate_list_not_empty, validate_range, 
    validate_agent_id, validate_string_not_empty, validate_dict_has_keys, validate_choice
)
from .exceptions import (
    ConclaveError, AgentError, EnvironmentError, PromptError, 
    LLMError, ConfigurationError, ValidationError
)
from .formatting import (
    format_candidate_info, format_agent_list, format_discussion_transcript,
    format_voting_results, format_stance_history, format_summary_statistics
)

__all__ = [
    # Logging utilities
    'get_logger',
    'setup_logging',
    'configure_component_loggers',
    'log_error_with_context',
    
    # Validation utilities
    'validate_agent',
    'validate_list_not_empty', 
    'validate_range',
    'validate_agent_id',
    'validate_string_not_empty',
    'validate_dict_has_keys',
    'validate_choice',
    
    # Exception classes
    'ConclaveError',
    'AgentError',
    'EnvironmentError',
    'PromptError',
    'LLMError',
    'ConfigurationError',
    'ValidationError',
    
    # Formatting utilities
    'format_candidate_info',
    'format_agent_list', 
    'format_discussion_transcript',
    'format_voting_results',
    'format_stance_history',
    'format_summary_statistics'
]
