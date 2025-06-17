"""
Custom exception classes for the Conclave simulation.

This module defines a hierarchy of exception classes that provide
better error handling and debugging throughout the simulation.
"""

class ConclaveError(Exception):
    """Base exception class for all Conclave simulation errors."""
    
    def __init__(self, message: str, component: str = None, context: dict = None):
        """
        Initialize a Conclave error.
        
        Args:
            message: The error message
            component: The component where the error occurred
            context: Additional context information
        """
        super().__init__(message)
        self.component = component
        self.context = context or {}
    
    def __str__(self):
        base_msg = super().__str__()
        if self.component:
            base_msg = f"[{self.component}] {base_msg}"
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg = f"{base_msg} (Context: {context_str})"
        
        return base_msg


class AgentError(ConclaveError):
    """Exception raised for agent-related errors."""
    
    def __init__(self, message: str, agent_id: str = None, agent_name: str = None, **kwargs):
        """
        Initialize an agent error.
        
        Args:
            message: The error message
            agent_id: The ID of the agent that caused the error
            agent_name: The name of the agent that caused the error
            **kwargs: Additional context passed to parent
        """
        context = kwargs.get('context', {})
        if agent_id:
            context['agent_id'] = agent_id
        if agent_name:
            context['agent_name'] = agent_name
        
        super().__init__(message, component="Agent", context=context)
        self.agent_id = agent_id
        self.agent_name = agent_name


class EnvironmentError(ConclaveError):
    """Exception raised for environment-related errors."""
    
    def __init__(self, message: str, round_num: int = None, phase: str = None, **kwargs):
        """
        Initialize an environment error.
        
        Args:
            message: The error message
            round_num: The round number where the error occurred
            phase: The simulation phase where the error occurred
            **kwargs: Additional context passed to parent
        """
        context = kwargs.get('context', {})
        if round_num is not None:
            context['round_num'] = round_num
        if phase:
            context['phase'] = phase
        
        super().__init__(message, component="Environment", context=context)
        self.round_num = round_num
        self.phase = phase


class PromptError(ConclaveError):
    """Exception raised for prompt generation and template errors."""
    
    def __init__(self, message: str, prompt_name: str = None, missing_vars: list = None, **kwargs):
        """
        Initialize a prompt error.
        
        Args:
            message: The error message
            prompt_name: The name of the prompt template
            missing_vars: List of missing variables
            **kwargs: Additional context passed to parent
        """
        context = kwargs.get('context', {})
        if prompt_name:
            context['prompt_name'] = prompt_name
        if missing_vars:
            context['missing_vars'] = missing_vars
        
        super().__init__(message, component="Prompting", context=context)
        self.prompt_name = prompt_name
        self.missing_vars = missing_vars


class LLMError(ConclaveError):
    """Exception raised for LLM client and tool calling errors."""
    
    def __init__(self, message: str, model: str = None, tool_name: str = None, **kwargs):
        """
        Initialize an LLM error.
        
        Args:
            message: The error message
            model: The LLM model that caused the error
            tool_name: The tool that was being called
            **kwargs: Additional context passed to parent
        """
        context = kwargs.get('context', {})
        if model:
            context['model'] = model
        if tool_name:
            context['tool_name'] = tool_name
        
        super().__init__(message, component="LLM", context=context)
        self.model = model
        self.tool_name = tool_name


class ConfigurationError(ConclaveError):
    """Exception raised for configuration-related errors."""
    
    def __init__(self, message: str, config_file: str = None, config_key: str = None, **kwargs):
        """
        Initialize a configuration error.
        
        Args:
            message: The error message
            config_file: The configuration file with the error
            config_key: The specific configuration key
            **kwargs: Additional context passed to parent
        """
        context = kwargs.get('context', {})
        if config_file:
            context['config_file'] = config_file
        if config_key:
            context['config_key'] = config_key
        
        super().__init__(message, component="Configuration", context=context)
        self.config_file = config_file
        self.config_key = config_key


class ValidationError(ConclaveError):
    """Exception raised for validation failures."""
    
    def __init__(self, message: str, field_name: str = None, field_value=None, **kwargs):
        """
        Initialize a validation error.
        
        Args:
            message: The error message
            field_name: The name of the field that failed validation
            field_value: The value that failed validation
            **kwargs: Additional context passed to parent
        """
        context = kwargs.get('context', {})
        if field_name:
            context['field_name'] = field_name
        if field_value is not None:
            context['field_value'] = str(field_value)
        
        super().__init__(message, component="Validation", context=context)
        self.field_name = field_name
        self.field_value = field_value
