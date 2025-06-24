"""
Simplified Agent class using mixin composition.

This version of the Agent         # Prompt system
        self.prompt_loader = get_prompt_loader()
        from ..prompting.unified_generator import UnifiedPromptVariableGenerator
        self.prompt_variable_generator = UnifiedPromptVariableGenerator(env=self.env, prompt_loader=self.prompt_loader)ss focuses only on core agent identity and initialization,
with behavior provided through mixin composition for better maintainability.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Any
if TYPE_CHECKING:
    from ..environments.conclave_env import ConclaveEnv

import datetime
import logging
from conclave.config import get_config_manager  # Keep for backward compatibility if needed
from config.scripts import get_config  # New config adapter
from conclave.llm import get_llm_client, SimplifiedToolCaller
from ..prompting import get_prompt_loader
from ..prompting.unified_generator import UnifiedPromptVariableGenerator
from .tool_executor import ToolExecutor

# Import mixins
from .mixins import VotingMixin, DiscussionMixin, StanceMixin, ReflectionMixin

logger = logging.getLogger("conclave.agents")

class Agent(VotingMixin, DiscussionMixin, StanceMixin, ReflectionMixin):
    """
    Agent class representing a Cardinal in the papal conclave simulation.
    
    This class focuses on core agent identity and initialization, with behavior
    provided through mixin composition for better maintainability and testability.
    """
    
    def __init__(self, 
                 agent_id: int, 
                 conclave_env: 'ConclaveEnv', 
                 name: str, 
                 personality: str, 
                 initial_stance: str, 
                 party_loyalty: float, 
                 agent_type: str = "default"):
        """
        Initialize a new Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            conclave_env: Reference to the conclave environment
            name: Cardinal's name
            personality: Agent's personality description
            initial_stance: Initial stance on papal selection
            party_loyalty: Loyalty to party/faction (0.0 to 1.0)
            agent_type: Type of agent (default: "default")
        """
        # Core identity
        self.agent_id = agent_id
        self.conclave_env = conclave_env
        self.config_manager = get_config()  # Use new config adapter
        self.name = name
        self.env = conclave_env
        
        # Agent attributes
        self.cardinal_id = str(agent_id)
        self.internal_persona = initial_stance
        self.public_profile = "Not set"
        self.profile_blurb = "N/A"
        self.persona_tag = "N/A"
        
        # State tracking
        self.vote_history = []
        self.role_tag = "ELECTOR"
        self.internal_stance = None
        self.stance_history = []
        self.last_stance_update: Optional[datetime.datetime] = None
        self.current_reflection_digest: Optional[str] = "Reflection not yet generated."
        self.last_reflection_round: int = -1
        self.last_reflection_timestamp: Optional[datetime.datetime] = None
        
        # Configuration and tools
        self.config = self.config_manager.config  # ConfigAdapter.config is the RefactoredConfig object
        self.logger = logging.getLogger(f"conclave.agents.{self.name.replace(' ', '_')}")
        
        # Prompt system
        self.prompt_loader = get_prompt_loader()
        self.prompt_variable_generator = UnifiedPromptVariableGenerator(env=self.env, prompt_loader=self.prompt_loader)
        
        # LLM client and tool caller
        try:
            self.llm_client = get_llm_client("agent")
            self.tool_caller = SimplifiedToolCaller(self.llm_client, self.logger)
            self.tool_executor = ToolExecutor(self)
            
            self.logger.info(f"Agent {self.name} initialized with {self.config.models.llm.backend} backend, model: {self.llm_client.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client for agent {self.name}: {e}")
            raise

    # Utility methods for agent management
    
    def load_persona_from_data(self, cardinal_data: Dict):
        """Load persona details from provided data (e.g., a row from CSV)."""
        self.internal_persona = cardinal_data.get('Internal_Persona', '')
        self.public_profile = cardinal_data.get('Public_Profile', '')
        self.name = cardinal_data.get('name', self.name)
        self.cardinal_id = cardinal_data.get('Cardinal_ID', self.agent_id)
        self.logger.info(f"Loaded persona for {self.name} (Cardinal ID: {self.cardinal_id})")

    def update_agent_details(self, details: Dict):
        """Update agent details, e.g., from a central data source after init."""
        if 'internal_persona' in details: 
            self.internal_persona = details['internal_persona']
        if 'public_profile' in details: 
            self.public_profile = details['public_profile']
        if 'role_tag' in details: 
            self.role_tag = details['role_tag']
        self.logger.debug(f"Agent {self.name} details updated.")

    def __repr__(self):
        return f"Agent(id={self.agent_id}, name='{self.name}', role='{self.role_tag}')"
