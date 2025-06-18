import logging
import random
import threading
import time
from typing import Dict, List, Optional, TYPE_CHECKING, Any # Ensure Any is imported
from pathlib import Path # Add Path import
from concurrent.futures import ThreadPoolExecutor # Add ThreadPoolExecutor import
from tqdm import tqdm # Add tqdm import
import pandas as pd # Add pandas import
import numpy as np # Add numpy import

from conclave.agents.base import Agent # For type hinting, removed AgentSettings, LLMSettings, EmbeddingSettings
from conclave.prompting.prompt_loader import PromptLoader # Corrected import to use prompting
from conclave.prompting.unified_generator import UnifiedPromptVariableGenerator
from conclave.config import get_config_manager # Use centralized config management
from conclave.llm import get_llm_client, SimplifiedToolCaller # Use new simplified tool calling
from conclave.network.network_manager import NetworkManager # Add network manager import
from config.scripts.models import RefactoredConfig

if TYPE_CHECKING:
    from ..agents.base import Agent # For type hinting

logger = logging.getLogger(__name__)
# Create a specific logger for discussion outputs
discussion_logger = logging.getLogger('conclave.discussions') 

class ConclaveEnv:
    def __init__(self, viz_dir: Optional[str] = None): # Add viz_dir and allow num_agents to be set by config
        # print("ConclaveEnv.__init__ STARTED") # DEBUG PRINT
        self.agents: List[Agent] = [] # Type hint for self.agents
        self.viz_dir = viz_dir # Store viz_dir
        self.votingRound = 0 # Initialize to 0. multi_round.py will set it to 1, 2, ... for each round.
        self.votingHistory = []
        self.votingBuffer = {}
        # Track individual votes: {voter_agent_id: candidate_id} for current round
        self.individual_votes_buffer = {}
        # Track individual votes history: List of {voter_agent_id: candidate_id} for each round
        self.individual_votes_history = []
        # Track which agents participated in which discussion rounds
        self.agent_discussion_participation = {}
        # Track which agents voted in each round
        self.voting_participation = {}
        self.voting_lock = threading.Lock()
        self.winner: Optional[int] = None # Winner will be an agent_id (int)
        self.discussionHistory: List[List[Dict]] = [] # List of rounds, each round is a list of comment dicts
        self.discussionRound = 0
        self.agent_discussion_participation: Dict[int, List[int]] = {} # agent_id -> list of discussion_round_numbers

        # Initialize config manager and load configuration
        # print("ConclaveEnv.__init__: Initializing ConfigManager...") # DEBUG PRINT
        self.config_manager = get_config_manager()
        self.app_config: RefactoredConfig = self.config_manager.config # Access config directly
        # print("ConclaveEnv.__init__: ConfigManager initialized.") # DEBUG PRINT

        # Load prompts and tools
        # self.prompt_loader = PromptLoader(self.app_config) # Old incorrect line
        # print("ConclaveEnv.__init__: Initializing PromptLoader...") # DEBUG PRINT
        # Construct path to prompts.yaml within the conclave/prompting directory
        prompts_file_path = Path(__file__).parent.parent / "prompting" / "prompts.yaml"
        # print(f"ConclaveEnv.__init__: Prompts file path: {prompts_file_path.resolve()}") # DEBUG PRINT
        self.prompt_loader = PromptLoader(str(prompts_file_path.resolve())) # Pass the correct path string
        # print("ConclaveEnv.__init__: PromptLoader initialized.") # DEBUG PRINT
        
        # print("ConclaveEnv.__init__: Initializing UnifiedPromptVariableGenerator...") # DEBUG PRINT
        self.prompt_variable_generator = UnifiedPromptVariableGenerator(self, self.prompt_loader) # Pass self (env) and prompt_loader
        # print("ConclaveEnv.__init__: UnifiedPromptVariableGenerator initialized.") # DEBUG PRINT

        # LLM Client for Environment-level tool calls (e.g., discussion analyzer)
        # This uses the same shared manager as agents but gets its own client instance if needed.
        # Typically, environment tasks might use a specific configuration or even a different model.
        # For now, we assume it can use a similar setup. Consider a separate config if distinct behavior is needed.
        try:
            # Use centralized LLM client management for environment tasks
            self.env_llm_client = get_llm_client("environment")
            self.env_tool_caller = SimplifiedToolCaller(self.env_llm_client, logger) # Use environment's logger
            logger.info(f"ConclaveEnv initialized its own LLM client: {self.env_llm_client.model_name} for environment tasks.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client for ConclaveEnv: {e}")
            self.env_llm_client = None
            self.env_tool_caller = None

        # Extract relevant config sections for easier access
        # print("ConclaveEnv.__init__: Extracting config sections...") # DEBUG PRINT
        self.agent_config = self.app_config.agent
        self.simulation_config = self.app_config.simulation
        self.output_config = self.app_config.output

        # Default number of speaking turns per agent within one run_discussion_round phase.
        # This can be made configurable later if multi-turn discussions before analysis are needed.
        # self.num_discussion_turns = 1 # REMOVED - simplifying to one round of speaking
        # logger.debug(f"ConclaveEnv initialized") # Simpler log
        logger.debug(f"ConclaveEnv initialized with app_config: {self.app_config.model_dump_json(indent=2)}") # More detailed log

        self.num_agents = self.simulation_config.num_cardinals # Override num_agents from constructor
        self.discussion_group_size = self.simulation_config.discussion_group_size
        self.max_election_rounds = self.simulation_config.max_election_rounds
        self.min_discussion_words = self.simulation_config.discussion_length.min_words
        self.max_discussion_words = self.simulation_config.discussion_length.max_words

        self.supermajority_threshold = self.simulation_config.voting.supermajority_threshold
        self.enable_parallel_processing = self.simulation_config.enable_parallel_processing

        # Initialize network manager for grouping
        self.network_manager = NetworkManager(self.config_manager)

        # Testing groups configuration (if enabled)
        if self.app_config.testing_groups and self.app_config.testing_groups.enabled:
            self.testing_groups_enabled = True # Explicitly set
            self.active_testing_group_name = self.app_config.testing_groups.active_group
            # Directly access the active group's settings
            active_group_details = getattr(self.app_config.testing_groups, self.active_testing_group_name)

            # Override simulation parameters if a testing group is active
            self.num_agents = active_group_details.override_settings.num_cardinals
            self.discussion_group_size = active_group_details.override_settings.discussion_group_size
            self.max_election_rounds = active_group_details.override_settings.max_election_rounds
            self.supermajority_threshold = active_group_details.override_settings.supermajority_threshold
            self.enable_parallel_processing = active_group_details.override_settings.enable_parallel_processing
            logger.info(f"RUNNING WITH TESTING GROUP: {self.active_testing_group_name}")
            logger.info(f"  Num Cardinals: {self.num_agents}")
            logger.info(f"  Discussion Group Size: {self.discussion_group_size}")
            logger.info(f"  Max Election Rounds: {self.max_election_rounds}")
            logger.info(f"  Supermajority Threshold: {self.supermajority_threshold}")
        else:
            self.testing_groups_enabled = False
        # print("ConclaveEnv.__init__: Testing groups configuration processed.") # DEBUG PRINT

        # Initialize agents
        # print("ConclaveEnv.__init__: Calling _initialize_agents...") # DEBUG PRINT
        self.agents = self._initialize_agents()
        # print("ConclaveEnv.__init__: _initialize_agents DONE.") # DEBUG PRINT
        
        # Initialize network after agents are loaded
        self.network_manager.initialize_network(self.agents)
        
        # print("ConclaveEnv.__init__ FINISHED") # DEBUG PRINT

    def _initialize_agents(self) -> List[Agent]:
        # print("ConclaveEnv._initialize_agents STARTED") # DEBUG PRINT
        """Load agents based on testing groups configuration or all agents in normal mode."""
        # Read cardinals from the master CSV file
        # Ensure the path is correct, relative to the project root
        # print("ConclaveEnv._initialize_agents: Determining project root...") # DEBUG PRINT
        project_root = Path(__file__).parent.parent.parent 
        cardinals_data_path = project_root / "data" / "cardinals_master_data.csv"
        # print(f"ConclaveEnv._initialize_agents: Cardinals data path: {cardinals_data_path}") # DEBUG PRINT
        master_df = pd.read_csv(cardinals_data_path)
        # print(f"ConclaveEnv._initialize_agents: Successfully read CSV. Shape: {master_df.shape}") # DEBUG PRINT
        
        loaded_agents = [] # Temporary list to hold loaded agents

        if self.testing_groups_enabled:
            active_group_details = getattr(self.app_config.testing_groups, self.active_testing_group_name)
            cardinal_ids_for_group = active_group_details.cardinal_ids
            # print(f"ConclaveEnv._initialize_agents: Testing groups enabled. Group: {self.active_testing_group_name}, Cardinal IDs: {cardinal_ids_for_group}") # DEBUG PRINT
            logger.info(f"Loading testing group '{self.active_testing_group_name}' with {len(cardinal_ids_for_group)} cardinals: {cardinal_ids_for_group}")
            
            # Filter dataframe to only include cardinals in the testing group
            master_df_filtered = master_df[master_df['Cardinal_ID'].isin(cardinal_ids_for_group)]
            
            if len(master_df_filtered) != len(cardinal_ids_for_group):
                loaded_ids_from_csv = set(master_df_filtered['Cardinal_ID'].tolist())
                missing_ids = set(cardinal_ids_for_group) - loaded_ids_from_csv
                if missing_ids:
                    logger.warning(f"Some cardinal IDs for testing group '{self.active_testing_group_name}' not found in CSV: {missing_ids}")
            master_df_to_load = master_df_filtered
        else:
            # In normal mode, use num_agents from simulation_config (which might have been overridden by testing group if it was active initially but then disabled, so re-check)
            num_to_load = self.simulation_config.num_cardinals 
            if self.app_config.testing_groups and self.app_config.testing_groups.enabled: # If testing groups are somehow re-enabled, defer to that
                active_group_details = getattr(self.app_config.testing_groups, self.app_config.testing_groups.active_group)
                num_to_load = active_group_details.override_settings.num_cardinals

            if num_to_load and num_to_load < len(master_df):
                master_df_to_load = master_df.head(num_to_load)
                logger.info(f"Loading first {num_to_load} cardinals from CSV as per configuration.")
            else:
                master_df_to_load = master_df
                logger.info(f"Loading all {len(master_df_to_load)} cardinals from CSV.")
        # print(f"ConclaveEnv._initialize_agents: master_df_to_load shape: {master_df_to_load.shape}") # DEBUG PRINT

        # Create Agent instances and add them to env.agents
        # print("ConclaveEnv._initialize_agents: Starting agent creation loop...") # DEBUG PRINT
        for list_index, (idx, row) in enumerate(master_df_to_load.iterrows()):
            # if list_index == 0: # DEBUG PRINT for first agent
                # print(f"ConclaveEnv._initialize_agents: Creating first agent (list_index=0): {row['Name']}") # DEBUG PRINT
            # Get all available prompts and tools for the agent
            # available_prompts = self.prompt_loader.get_all_prompts() # Not passed to Agent constructor
            # available_tools = list(self.prompt_loader.prompts_config.tool_definitions.values()) if self.prompt_loader.prompts_config else [] # Not passed to Agent constructor

            agent = Agent(
                agent_id=list_index,  # Use list index as agent_id for consistent indexing
                name=row['Name'],
                conclave_env=self, # Pass the environment instance
                personality=row.get('Internal_Persona', ''), # Assuming Internal_Persona maps to personality
                initial_stance=row.get('Public_Profile', ''), # Assuming Public_Profile maps to initial_stance
                party_loyalty=0.5 # Placeholder, needs to be sourced or defined
            )
            # Set CSV-specific attributes, including cardinal_id which is crucial for mapping
            agent.cardinal_id = row['Cardinal_ID'] # Ensure cardinal_id is set directly for mapping
            agent.background_csv = row['Background'] # Store Background from CSV
            agent.internal_persona_csv = row.get('Internal_Persona', '')
            agent.public_profile_csv = row.get('Public_Profile', '')
            agent.profile_blurb_csv = row.get('Persona_Tag', '')
            agent.persona_tag_csv = row.get('Persona_Tag', '')

            loaded_agents.append(agent)
        # print(f"ConclaveEnv._initialize_agents: Agent creation loop finished. {len(loaded_agents)} agents loaded.") # DEBUG PRINT
        
        # Update self.num_agents to reflect the actual number of loaded agents
        self.num_agents = len(loaded_agents)
        # print(f"ConclaveEnv._initialize_agents: self.num_agents updated to {self.num_agents}") # DEBUG PRINT

        # Update candidate_ids to use list indices if testing groups are enabled
        if self.testing_groups_enabled:
            active_group_details = getattr(self.app_config.testing_groups, self.active_testing_group_name)
            original_candidate_ids_for_group = active_group_details.candidate_ids
            # print(f"ConclaveEnv._initialize_agents: Mapping candidate IDs for testing group. Original: {original_candidate_ids_for_group}") # DEBUG PRINT
            
            # Create a mapping from Cardinal_ID to the list_index of the loaded agents
            # Corrected to use agent.cardinal_id which is now set directly
            cardinal_id_to_index_map = {agent.cardinal_id: agent.agent_id for agent in loaded_agents}
            
            self.candidate_ids = [cardinal_id_to_index_map[cid] for cid in original_candidate_ids_for_group if cid in cardinal_id_to_index_map]
            
            if len(self.candidate_ids) != len(original_candidate_ids_for_group):
                logger.warning(
                    f"Mismatch in candidate mapping for testing group '{self.active_testing_group_name}'. "
                    f"Original: {original_candidate_ids_for_group}, Mapped: {self.candidate_ids}. "
                    f"This might happen if some candidate Cardinal_IDs were not found or loaded."
                )
            logger.info(f"Mapped candidate Cardinal_IDs {original_candidate_ids_for_group} to agent_ids {self.candidate_ids} for testing group '{self.active_testing_group_name}'.")
        else:
            self.candidate_ids = list(range(self.num_agents)) # All loaded agents are potential candidates
        # print(f"ConclaveEnv._initialize_agents: Candidate IDs set: {self.candidate_ids}") # DEBUG PRINT

        logger.info(f"Successfully loaded {self.num_agents} agents.")
        if self.testing_groups_enabled:
            logger.info(f"Testing group \'{self.active_testing_group_name}\' active. Candidate agent_ids: {self.candidate_ids}")
            # logger.info(f"Testing group setup:\\\\n{self.get_candidates_description()}") # get_candidates_description needs to be adapted or removed
        # print("ConclaveEnv._initialize_agents FINISHED") # DEBUG PRINT
        return loaded_agents

    def freeze_agent_count(self):
        """Freeze the agent count to match the loaded roster. Call after loading all agents."""
        self.num_agents = len(self.agents)

    # Testing groups candidate/elector helper methods
    def is_candidate(self, agent_id: int) -> bool:
        """Check if an agent is a candidate in testing groups mode."""
        if not self.testing_groups_enabled:
            return True  # In normal mode, all agents can be candidates
        return agent_id in self.candidate_ids # self.candidate_ids stores agent_id (index)

    def get_candidates_list(self) -> List[int]:
        """Get list of candidate agent IDs."""
        # This now directly returns the agent_ids (list indices) stored in self.candidate_ids
        return self.candidate_ids.copy()

    def get_candidates_description(self) -> str:
        """Get a formatted description of candidates for prompts."""
        candidates_agent_ids = self.get_candidates_list()
        if not candidates_agent_ids:
            return "No candidates designated."
        
        candidate_names = []
        for agent_id in candidates_agent_ids:
            if agent_id < len(self.agents):
                agent = self.agents[agent_id]
                cardinal_id_attr = getattr(agent, 'cardinal_id', agent_id) # Use agent_id if cardinal_id not present
                candidate_names.append(f"Cardinal {cardinal_id_attr} - {agent.name} (Agent ID: {agent_id})")
            else:
                candidate_names.append(f"Agent {agent_id} - (Not loaded)")
        
        return "Designated candidates:\n" + "\n".join(candidate_names)

    def is_valid_vote_candidate(self, candidate_id: int) -> bool:
        """Check if a candidate ID is valid for voting in current mode."""
        # Basic range check against the number of loaded agents
        if not (0 <= candidate_id < self.num_agents):
            return False
        
        # In testing groups mode, only designated candidates (by agent_id) can receive votes
        # self.candidate_ids already stores agent_ids (list indices)
        return candidate_id in self.candidate_ids

    def update_internal_stances(self) -> None:
        """
        Update internal stances for all agents.
        """
        agents_to_update = self.agents # All agents update their stance
        
        if not agents_to_update:
            logger.info("No agents to update stances for.")
            return
        
        logger.info(f"Updating internal stances for all {len(agents_to_update)} agents...")
        
        if self.enable_parallel_processing:
            # Update stances in parallel
            # Consider making max_workers configurable
            with ThreadPoolExecutor(max_workers=min(self.app_config.simulation.discussion_group_size, len(agents_to_update))) as executor: 
                futures = [executor.submit(agent.generate_internal_stance) for agent in agents_to_update]
                # Wait for all futures to complete
                # The disable=True for tqdm might be from old config, consider if it should be dynamic
                for future in tqdm(futures, desc="Generating Stances", total=len(futures), disable=not self.output_config.logging.performance_logging):
                    future.result()  # This blocks until the task completes
        else:
            # Update stances sequentially
            for agent in tqdm(agents_to_update, desc="Generating Stances Sequentially", disable=not self.output_config.logging.performance_logging):
                agent.generate_internal_stance()
        
        # Log completion and summary
        updated_count = sum(1 for agent in agents_to_update if agent.internal_stance) # Make sure agent.internal_stance is the correct attribute
        logger.info(f"Internal stance update completed: {updated_count}/{len(agents_to_update)} agents updated")
        
        # Update embeddings for the new stances
        try:
            from ..embeddings import get_default_client
            embedding_client = get_default_client()
            current_round_identifier = f"ER{self.votingRound}.D{self.discussionRound}" # Use ER
            embedding_client.update_agent_embeddings_in_history(agents_to_update, current_round_identifier)
        except Exception as e:
            logger.warning(f"Failed to update embeddings after stance generation: {e}")
        
        # Log stance summary for this round
        logger.info(f"=== STANCE UPDATE SUMMARY (ER{self.votingRound}.D{self.discussionRound}) ===") # Use ER
        for agent in agents_to_update:
            if agent.internal_stance: # Make sure agent.internal_stance is the correct attribute
                stance_preview = str(agent.internal_stance)[:100].replace('\n', ' ')
                logger.info(f"  {agent.name} (ID: {agent.agent_id}): {stance_preview}...")
            else:
                logger.info(f"  {agent.name} (ID: {agent.agent_id}): No stance generated or available.")
        logger.info(f"=== END STANCE UPDATE SUMMARY ===")

    def generate_initial_stances(self) -> None:
        """
        Generate initial internal stances for all agents before the simulation begins.
        This should be called once after all agents are added to the environment.
        """
        logger.info(f"Generating initial internal stances for {len(self.agents)} agents...")
        
        if self.enable_parallel_processing:
            # Generate initial stances in parallel
            # Consider making max_workers configurable
            with ThreadPoolExecutor(max_workers=min(self.app_config.simulation.discussion_group_size, len(self.agents))) as executor: 
                futures = [executor.submit(agent.generate_internal_stance) for agent in self.agents]
                # Wait for all futures to complete
                # The disable=True for tqdm might be from old config, consider if it should be dynamic
                for future in tqdm(futures, desc="Generating Initial Stances", total=len(futures), disable=not self.output_config.logging.performance_logging):
                    future.result()  # This blocks until the task completes
        else:
            # Generate initial stances sequentially
            for agent in tqdm(self.agents, desc="Generating Initial Stances Sequentially", disable=not self.output_config.logging.performance_logging):
                agent.generate_internal_stance()
        
        logger.info("Initial internal stances generated for all agents")
        
        # Update embeddings for initial stances
        try:
            from ..embeddings import get_default_client
            embedding_client = get_default_client()
            embedding_client.update_agent_embeddings_in_history(self.agents, "initial")
        except Exception as e:
            logger.warning(f"Failed to update embeddings for initial stances: {e}")
        
        # Log initial stance summary
        # stances = self.get_all_stances() # get_all_stances method needs to be added
        # logger.info(f"=== INITIAL STANCES SUMMARY ({len(stances)} agents) ===")
        # for name, stance in stances.items():
        #     stance_preview = stance[:100].replace('\n', ' ')
        #     logger.info(f"{name}: {stance_preview}...")
        # logger.info("=== END INITIAL STANCES SUMMARY ===")
        # Temporarily comment out stance logging as get_all_stances is not yet defined
        logger.info("Stance logging in generate_initial_stances temporarily commented out.")

    def run_discussion_round(self):
        """Manages a single discussion round, including forming groups, collecting comments, and triggering analysis and reflection."""
        logger.debug(f"Entering run_discussion_round for ER {self.votingRound}, Discussion {self.discussionRound + 1}") # Use ER, rephrase DP
        self.discussionRound += 1
        self.group_discussion_analyses = {} # Initialize for the current discussion round

        if self.num_agents == 0 or self.discussion_group_size == 0:
            self.num_discussion_groups = 0
            logger.warning("No agents or zero group size, skipping discussion processing.")
        else:
            self.num_discussion_groups = (self.num_agents + self.discussion_group_size - 1) // self.discussion_group_size

        logger.info(f"Starting Discussion Phase {self.discussionRound} for ER {self.votingRound} with {self.num_discussion_groups} discussion groups.") # Use ER

        current_discussion_groups = self._generate_discussion_group_assignments()
        
        all_comments_for_phase: List[Dict[str, Any]] = [] 

        if not current_discussion_groups:
            logger.warning("No discussion groups were formed. Skipping discussion processing.")
        else:
            if self.enable_parallel_processing:
                logger.info(f"Processing {len(current_discussion_groups)} discussion groups in parallel.")
                with ThreadPoolExecutor(max_workers=self.num_discussion_groups if self.num_discussion_groups > 0 else 1) as executor:
                    futures = []
                    for group_idx, group_agent_ids in enumerate(current_discussion_groups):
                        futures.append(executor.submit(self._process_discussion_group, group_idx, group_agent_ids, self.discussionRound))
                    
                    for future in tqdm(futures, desc=f"Processing Discussion Groups & Analyzing (ER {self.votingRound})", total=len(futures), disable=self.output_config.tqdm.disable_tqdm): # Use ER, remove DP
                        try:
                            group_comments = future.result() # _process_discussion_group now only returns comments
                            if group_comments:
                                all_comments_for_phase.extend(group_comments)
                        except Exception as e:
                            logger.error(f"Error processing a discussion group future: {e}", exc_info=True)
            else:
                logger.info(f"Processing {len(current_discussion_groups)} discussion groups sequentially.")
                for group_idx, group_agent_ids in enumerate(tqdm(current_discussion_groups, desc=f"Processing Discussion Groups & Analyzing (ER {self.votingRound})", disable=self.output_config.tqdm.disable_tqdm)):
                    try:
                        group_comments = self._process_discussion_group(group_idx, group_agent_ids, self.discussionRound)
                        if group_comments:
                            all_comments_for_phase.extend(group_comments)
                    except Exception as e:
                        logger.error(f"Error processing discussion group {group_idx}: {e}", exc_info=True)
            
            if self.discussionHistory and len(self.discussionHistory) >= self.discussionRound:
                 self.discussionHistory[self.discussionRound -1] = all_comments_for_phase
            else:
                 self.discussionHistory.append(all_comments_for_phase)

            logger.info(f"All {len(current_discussion_groups)} discussion groups processed and their discussions analyzed for Discussion Phase {self.discussionRound}.")

        # Analysis is now done per group within _process_discussion_group.
        # The main analyze_discussion_round() is not called here anymore.

        logger.debug("Group discussions and analyses completed. Proceeding to agent reflection.")
        self.agents_reflect_on_discussions() # This will use self.group_discussion_analyses and self.discussionHistory
        logger.debug(f"Exiting run_discussion_round - Round: {self.discussionRound}")

    # ...existing code...
    # REMOVED _agents_speak_in_turn as its logic is integrated into run_discussion_round

    def _process_discussion_group(self, group_idx: int, group_agent_ids: List[int], current_discussion_round: int) -> List[Dict[str, Any]]:
        """Processes a single discussion group, allowing each agent to speak, then analyzes the group's discussion. Returns list of comments from the group."""
        group_comments_collected: List[Dict[str, Any]] = []
        participant_names = [self.agents[agent_id].name for agent_id in group_agent_ids]
        logger.info(f"Election Round {self.votingRound}, Discussion Group {group_idx + 1}/{self.num_discussion_groups} now speaking. Participants: {', '.join(participant_names)}") # Use ER

        for agent_id in group_agent_ids:
            agent = self.agents[agent_id]
            try:
                comment_data = agent.discuss(current_discussion_group_ids=group_agent_ids, current_discussion_round=current_discussion_round)
                if comment_data and isinstance(comment_data, dict) and "message" in comment_data:
                    group_comments_collected.append(comment_data)
                else:
                    logger.warning(f"Agent {agent.name} (ID: {agent.agent_id}) in group {group_idx+1} did not return a valid comment structure.")
                    group_comments_collected.append({"agent_id": agent.agent_id, "message": "(Agent did not provide a valid comment structure)"})
            except Exception as e:
                logger.error(f"Critical exception during discussion processing for agent {agent.name} (ID: {agent.agent_id}): {e}", exc_info=True)
                group_comments_collected.append({"agent_id": agent.agent_id, "message": f"[System Error: A critical error occurred while processing discussion for {agent.name}. See logs.]"})
        
        # Log consolidated messages for the group (existing logic)
        if group_comments_collected:
            log_header = f"ER {self.votingRound}, DG {group_idx + 1} (Participants: {', '.join(participant_names)}):" # Use ER
            messages_str_parts = [log_header]
            for comment in group_comments_collected:
                agent_name = "UnknownAgent"
                csv_cardinal_id = "N/A"
                internal_agent_id = comment.get('agent_id') 

                if isinstance(internal_agent_id, int) and 0 <= internal_agent_id < len(self.agents):
                    agent = self.agents[internal_agent_id]
                    agent_name = agent.name
                    csv_cardinal_id = getattr(agent, 'cardinal_id', 'N/A') 
                else:
                    agent_name = f"Agent (Internal ID: {internal_agent_id if internal_agent_id is not None else 'N/A'})"
                
                message_content = comment.get('message', '[No message]')
                messages_str_parts.append(f"  🗣️ {agent_name} (Cardinal ID: {csv_cardinal_id}):")
                messages_str_parts.append(f"     {message_content}")
                messages_str_parts.append("")  
            
            if messages_str_parts and messages_str_parts[-1] == "":
                messages_str_parts.pop()
            discussion_logger.info("\n".join(messages_str_parts))


        # Analyze this group's discussion immediately
        if group_comments_collected:
            logger.info(f"Analyzing discussion for Group {group_idx + 1} (ER {self.votingRound}, Discussion {current_discussion_round}) immediately after their discussion.") # Use ER, rephrase DP
            # _analyze_single_group_discussion updates self.group_discussion_analyses[group_idx]
            self._analyze_single_group_discussion(
                group_idx=group_idx,
                group_agent_ids=group_agent_ids,
                group_transcript_comments=group_comments_collected
            )
        else:
            logger.info(f"No comments to analyze for Group {group_idx + 1} (ER {self.votingRound}, Discussion {current_discussion_round}).") # Use ER, rephrase DP
            self.group_discussion_analyses[group_idx] = {"analysis_summary": "No discussion to analyze.", "raw_response": None, "error": None}


        return group_comments_collected

    def analyze_discussion_round(self):
        """
        Analyzes discussions for all groups. 
        This method is now primarily for manual use or if a full re-analysis is needed.
        The primary analysis path is via _analyze_single_group_discussion called from _process_discussion_group.
        It populates self.group_discussion_analyses.
        """
        logger.info(f"Executing analyze_discussion_round for ER {self.votingRound}, Discussion {self.discussionRound}.") # Use ER, rephrase DP
        self.group_discussion_analyses = {} # Reset/initialize

        current_round_index = self.discussionRound - 1
        if not (0 <= current_round_index < len(self.discussionHistory)):
            logger.error(f"No discussion history found for round {self.discussionRound}. Cannot analyze.")
            return
        
        all_comments_this_phase = self.discussionHistory[current_round_index]
        if not all_comments_this_phase:
            logger.warning(f"No comments recorded in discussion history for round {self.discussionRound}. Nothing to analyze.")
            return

        if not hasattr(self, '_last_discussion_groups') or not self._last_discussion_groups:
            logger.error("No discussion group assignments found (_last_discussion_groups). Cannot perform analysis.")
            return

        logger.info(f"Analyzing discussions for {len(self._last_discussion_groups)} groups using comments from Discussion Phase {self.discussionRound}.")

        for group_idx, group_agent_ids in enumerate(self._last_discussion_groups):
            # Filter comments for the current group from all_comments_this_phase
            group_transcript_comments = [
                comment for comment in all_comments_this_phase 
                if isinstance(comment.get('agent_id'), int) and self.agents[comment['agent_id']].agent_id in group_agent_ids
            ]
            
            if not group_transcript_comments:
                logger.info(f"No transcript found for group {group_idx + 1}. Skipping analysis for this group.")
                self.group_discussion_analyses[group_idx] = {"analysis_summary": "No transcript for this group.", "raw_response": None, "error": "No transcript"}
                continue
            
            self._analyze_single_group_discussion(group_idx, group_agent_ids, group_transcript_comments)
        
        logger.info(f"Finished analyze_discussion_round for Discussion Phase {self.discussionRound}.")
        # Results are in self.group_discussion_analyses

    def _generate_discussion_group_assignments(self) -> List[List[int]]:
        logger.debug("Entering _generate_discussion_group_assignments")
        
        # Get grouping configuration
        grouping_config = self.simulation_config.grouping if hasattr(self.simulation_config, 'grouping') else {}
        
        # Override with testing group config if active
        if self.testing_groups_enabled:
            active_group_details = getattr(self.app_config.testing_groups, self.active_testing_group_name)
            if hasattr(active_group_details.override_settings, 'grouping'):
                grouping_config = active_group_details.override_settings.grouping
                logger.info(f"Using testing group '{self.active_testing_group_name}' grouping configuration")
        
        # Use network manager to generate groups
        try:
            # Pass the Pydantic model directly to the network manager
            groups = self.network_manager.generate_groups(grouping_config, self.agents)
            
            if not groups:
                logger.warning("Network manager returned no groups, falling back to simple random grouping")
                groups = self._generate_fallback_groups()
            
            # Log group assignment summary
            summary = self.network_manager.get_group_summary()
            logger.info(f"Generated {summary['total_groups']} discussion groups:")
            logger.info(f"  Total agents: {summary['total_agents']}")
            logger.info(f"  Group sizes: {summary['group_sizes']}")
            logger.info(f"  Average group size: {summary['avg_group_size']:.1f}")
            
            # Store for reference
            self._last_discussion_groups = groups
            
            return groups
            
        except Exception as e:
            logger.error(f"Error generating network-based groups: {e}")
            logger.warning("Falling back to simple random grouping")
            return self._generate_fallback_groups()

    def _generate_fallback_groups(self) -> List[List[int]]:
        """Generate simple fallback groups when network grouping fails."""
        discussion_group_size = self.discussion_group_size
        
        if discussion_group_size < 2:
            logger.warning(
                f"Configured 'discussion_group_size' ({discussion_group_size}) is less than 2. "
                f"Adjusting to 2 for discussion group formation."
            )
            discussion_group_size = 2

        agent_ids = list(range(len(self.agents)))
        if not agent_ids:
            logger.warning("No agents available to form discussion groups.")
            self._last_discussion_groups = []
            return []
            
        random.shuffle(agent_ids)

        groups = []
        i = 0
        num_total_agents = len(agent_ids)
        while i < num_total_agents:
            groups.append(agent_ids[i : i + discussion_group_size])
            i += discussion_group_size

        # Avoid singleton groups
        if len(groups) > 1 and len(groups[-1]) == 1:
            if len(groups[-2]) > 0: 
                element_to_move = groups[-2].pop()
                groups[-1].insert(0, element_to_move)
        
        groups = [g for g in groups if g]
        logger.debug(f"Generated fallback discussion groups: {groups}")
        return groups

    def _analyze_single_group_discussion(self, group_idx: int, group_agent_ids: List[int], group_transcript_comments: List[Dict[str, Any]]):
        """Analyzes a single group's discussion transcript and stores the result."""
        # ... (existing logic for formatting transcript)
        formatted_transcript = []
        for comment_data in group_transcript_comments:
            agent_id = comment_data.get('agent_id')
            message = comment_data.get('message', '[message not found]')
            agent_name = f"Agent {agent_id}"
            if isinstance(agent_id, int) and 0 <= agent_id < len(self.agents):
                agent_name = self.agents[agent_id].name
            formatted_transcript.append(f"{agent_name} (ID: {agent_id}): {message}")
        full_transcript_str = "\n".join(formatted_transcript)

        if not full_transcript_str.strip():
            logger.warning(f"Group {group_idx + 1} had an empty transcript. Storing empty analysis.")
            self.group_discussion_analyses[group_idx] = {
                "analysis_summary": "Discussion was empty or resulted in an empty transcript.",
                "raw_response": None,
                "error": "Empty transcript"
            }
            return
        # ... (existing logic for preparing prompt variables)
        prompt_vars = self.prompt_variable_generator.generate_prompt_variables_for_group_analysis(
            group_id=group_idx,
            group_agent_ids=group_agent_ids,
            discussion_transcript=full_transcript_str,
            current_round=self.discussionRound  # Use discussionRound for consistency
        )
        # Ensure round_id is available if the template specifically uses it.
        # It seems discussion_round_num is the intended variable from the generator.
        # If 'round_id' is strictly required by the template, alias it:
        if 'round_id' not in prompt_vars:
            prompt_vars['round_id'] = prompt_vars.get('discussion_round_num', self.discussionRound) 

        # The variable 'group_transcript_text' is expected by the prompt template.
        # The UnifiedPromptVariableGenerator now generates 'group_transcript_text' directly.
        # Ensure it's correctly passed or aliased if the generator uses a different key.
        if 'group_transcript_text' not in prompt_vars and 'discussion_transcript' in prompt_vars:
            prompt_vars['group_transcript_text'] = prompt_vars['discussion_transcript']
        elif 'group_transcript_text' not in prompt_vars:
            # Fallback if neither is present, though the generator should provide it.
            logger.warning("Missing 'group_transcript_text' and 'discussion_transcript' in prompt_vars for discussion_analyzer. Using empty string.")
            prompt_vars['group_transcript_text'] = ""

        analysis_prompt = self.prompt_loader.get_prompt_template("discussion_analyzer").format(**prompt_vars)
        
        # self.logger.debug(f"LLM Request: Discussion Analysis prompt for Group {group_idx + 1}:\\n{analysis_prompt}")

        analyze_tool_def = self.prompt_loader.get_tool_definition("discussion_analyzer") # Corrected tool name
        if not analyze_tool_def:
            # ... (existing error handling) ...
            logger.error(f"Tool definition for 'discussion_analyzer' not found.") # Corrected tool name
            self.group_discussion_analyses[group_idx] = {"analysis_summary": "Error: Tool definition not found.", "raw_response": None, "error": "Tool definition missing"}
            return

        try:
            # ... (existing tool calling logic) ...
            messages = [{"role": "user", "content": analysis_prompt}]
            # Ensure env_tool_caller is initialized and available
            if not self.env_tool_caller:
                raise ValueError("Environment tool caller (env_tool_caller) not initialized.")

            result = self.env_tool_caller.call_tool(messages, [analyze_tool_def], tool_choice="discussion_analyzer") # Corrected tool name

            if result.success and result.arguments:
                # The tool 'discussion_analyzer' is expected to return key_points, speakers, overall_tone.
                # We need to construct a summary string from these for analysis_summary.
                key_points = result.arguments.get("key_points", [])
                speakers = result.arguments.get("speakers", [])
                overall_tone = result.arguments.get("overall_tone", "unknown")
                
                summary_parts = [
                    f"Overall Tone: {overall_tone.capitalize()}"
                ]
                if key_points:
                    summary_parts.append("Key Points:")
                    for i, point in enumerate(key_points):
                        speaker_info = f" (Speaker: {speakers[i]})" if i < len(speakers) and speakers[i] else ""
                        summary_parts.append(f"  - {point}{speaker_info}")
                else:
                    summary_parts.append("No specific key points extracted.")
                
                summary = "\n".join(summary_parts)
                
                self.group_discussion_analyses[group_idx] = {
                    "analysis_summary": summary,
                    "raw_response": result.raw_response, # Store raw LLM response if available
                    "error": None
                }
                logger.info(f"Group {group_idx + 1} discussion analysis successful. Summary:\\n{summary}")
            else:
                # ... (existing error handling) ...
                error_msg = result.error if result.error else "Unknown tool calling failure during analysis."
                logger.error(f"Tool calling failed for group {group_idx + 1} discussion analysis: {error_msg}")
                self.group_discussion_analyses[group_idx] = {"analysis_summary": f"Failed: {error_msg}", "raw_response": result.raw_response_content, "error": error_msg}
        except Exception as e:
            # ... (existing error handling) ...
            logger.error(f"Exception during discussion analysis for group {group_idx + 1}: {e}", exc_info=True)
            self.group_discussion_analyses[group_idx] = {"analysis_summary": f"Exception: {e}", "raw_response": None, "error": str(e)}

    def agents_reflect_on_discussions(self):
        """Triggers reflection for each agent based on their group's discussion and analysis."""
        logger.info(f"Starting agent reflection phase for ER {self.votingRound}, Discussion {self.discussionRound}.") # Use ER, rephrase DP

        current_round_index = self.discussionRound - 1
        if not (0 <= current_round_index < len(self.discussionHistory)):
            logger.error(f"No discussion history found for round {self.discussionRound}. Cannot provide transcripts for reflection.")
            # Fallback: agents reflect without full transcript if history is missing
            all_comments_this_phase = []
        else:
            all_comments_this_phase = self.discussionHistory[current_round_index]

        if not hasattr(self, '_last_discussion_groups') or not self._last_discussion_groups:
            logger.error("No discussion group assignments (_last_discussion_groups) found. Cannot perform reflection accurately.")
            # Agents might still reflect but without specific group context.
            # This case should ideally not happen if run_discussion_round executed correctly

        # Create a mapping from agent_id to their group_idx for quick lookup
        agent_to_group_map: Dict[int, int] = {}
        if hasattr(self, '_last_discussion_groups'):
            for idx, group_list in enumerate(self._last_discussion_groups):
                for agent_id_in_group in group_list:
                    agent_to_group_map[agent_id_in_group] = idx
        
        agents_to_reflect = self.agents # All agents reflect

        if self.enable_parallel_processing:
            with ThreadPoolExecutor(max_workers=min(self.app_config.simulation.discussion_group_size, len(agents_to_reflect))) as executor:
                futures = []
                for agent in agents_to_reflect:
                    group_idx = agent_to_group_map.get(agent.agent_id)
                    analysis_summary_for_agent = "No analysis was available for your group." # Default
                    group_transcript_for_agent_str = "No specific group transcript available for this round." # Default

                    if group_idx is not None:
                        analysis_data = self.group_discussion_analyses.get(group_idx)
                        if analysis_data:
                            if analysis_data.get("error") is None and analysis_data.get("analysis_summary"):
                                analysis_summary_for_agent = analysis_data["analysis_summary"]
                                logger.debug(f"Agent {agent.name} (ID: {agent.agent_id}) in Group {group_idx + 1} will use analysis: {analysis_summary_for_agent[:100]}...")
                            else:
                                # Analysis failed or produced no summary, use default. Log the specific error if present.
                                error_detail = analysis_data.get("error", "unknown reason")
                                logger.warning(f"Analysis for Group {group_idx + 1} (Agent {agent.name}) was not successful or summary empty (error: {error_detail}). Agent will reflect with default message.")
                                # analysis_summary_for_agent remains the default "No analysis was available for your group."
                        else: # Should not happen if _process_discussion_group always populates it
                            logger.warning(f"No analysis data structure found for Group {group_idx + 1} (Agent {agent.name}). Agent will reflect with default message.")
                            # analysis_summary_for_agent remains the default "No analysis was available for your group."

                        if all_comments_this_phase and hasattr(self, '_last_discussion_groups') and group_idx < len(self._last_discussion_groups):
                            group_agent_ids_for_transcript = self._last_discussion_groups[group_idx]
                            comments_for_agent_group = [
                                c for c in all_comments_this_phase 
                                if isinstance(c.get('agent_id'), int) and c['agent_id'] in group_agent_ids_for_transcript
                            ]
                            
                            formatted_transcript_list = []
                            for comment_data in comments_for_agent_group:
                                comment_agent_id = comment_data.get('agent_id')
                                message = comment_data.get('message', '[message not found]')
                                comment_agent_name = f"Agent {comment_agent_id}"
                                if isinstance(comment_agent_id, int) and 0 <= comment_agent_id < len(self.agents):
                                    comment_agent_name = self.agents[comment_agent_id].name
                                formatted_transcript_list.append(f"{comment_agent_name} (ID: {comment_agent_id}): {message}")
                            group_transcript_for_agent_str = "\n".join(formatted_transcript_list)
                            if not group_transcript_for_agent_str.strip():
                                 group_transcript_for_agent_str = "(No discussion took place in your group or messages were empty.)"
                    else:
                        logger.warning(f"Agent {agent.name} (ID: {agent.agent_id}) was not found in any discussion group for round {self.discussionRound}. Reflecting generally.")
                        # analysis_summary_for_agent and group_transcript_for_agent_str remain defaults

                    # Call reflect_on_discussion with the correct arguments
                    futures.append(executor.submit(agent.reflect_on_discussion, 
                                                   analysis_summary_for_agent, 
                                                   group_transcript_for_agent_str, 
                                                   self.discussionRound))

                for future in tqdm(futures, desc=f"Agents Reflecting (ER {self.votingRound})", total=len(futures), disable=not self.output_config.logging.performance_logging): # Use ER, remove DP
                    try:
                        future.result() 
                    except Exception as e:
                        logger.error(f"Error during an agent's reflection process: {e}", exc_info=True)
        else:
            # Sequential reflection
            for agent in tqdm(agents_to_reflect, desc=f"Agents Reflecting Sequentially (ER {self.votingRound})", disable=not self.output_config.logging.performance_logging):
                try:
                    group_idx = agent_to_group_map.get(agent.agent_id)
                    analysis_summary_for_agent = "No analysis was available for your group." # Default
                    group_transcript_for_agent_str = "No specific group transcript available for this round." # Default

                    if group_idx is not None:
                        analysis_data = self.group_discussion_analyses.get(group_idx)
                        if analysis_data:
                            if analysis_data.get("error") is None and analysis_data.get("analysis_summary"):
                                analysis_summary_for_agent = analysis_data["analysis_summary"]
                                logger.debug(f"Agent {agent.name} (ID: {agent.agent_id}) in Group {group_idx + 1} will use analysis: {analysis_summary_for_agent[:100]}...")
                            else:
                                # Analysis failed or produced no summary, use default. Log the specific error if present.
                                error_detail = analysis_data.get("error", "unknown reason")
                                logger.warning(f"Analysis for Group {group_idx + 1} (Agent {agent.name}) was not successful or summary empty (error: {error_detail}). Agent will reflect with default message.")
                                # analysis_summary_for_agent remains the default "No analysis was available for your group."
                        else: # Should not happen if _process_discussion_group always populates it
                            logger.warning(f"No analysis data structure found for Group {group_idx + 1} (Agent {agent.name}). Agent will reflect with default message.")
                            # analysis_summary_for_agent remains the default "No analysis was available for your group."

                        if all_comments_this_phase and hasattr(self, '_last_discussion_groups') and group_idx < len(self._last_discussion_groups):
                            group_agent_ids_for_transcript = self._last_discussion_groups[group_idx]
                            comments_for_agent_group = [
                                c for c in all_comments_this_phase 
                                if isinstance(c.get('agent_id'), int) and c['agent_id'] in group_agent_ids_for_transcript
                            ]
                            
                            formatted_transcript_list = []
                            for comment_data in comments_for_agent_group:
                                comment_agent_id = comment_data.get('agent_id')
                                message = comment_data.get('message', '[message not found]')
                                comment_agent_name = f"Agent {comment_agent_id}"
                                if isinstance(comment_agent_id, int) and 0 <= comment_agent_id < len(self.agents):
                                    comment_agent_name = self.agents[comment_agent_id].name
                                formatted_transcript_list.append(f"{comment_agent_name} (ID: {comment_agent_id}): {message}")
                            group_transcript_for_agent_str = "\n".join(formatted_transcript_list)
                            if not group_transcript_for_agent_str.strip():
                                 group_transcript_for_agent_str = "(No discussion took place in your group or messages were empty.)"
                    else:
                        logger.warning(f"Agent {agent.name} (ID: {agent.agent_id}) was not found in any discussion group for round {self.discussionRound}. Reflecting generally.")
                        # analysis_summary_for_agent and group_transcript_for_agent_str remain defaults

                    # Call reflect_on_discussion with the correct arguments
                    agent.reflect_on_discussion(analysis_summary_for_agent, 
                                               group_transcript_for_agent_str, 
                                               self.discussionRound)
                except Exception as e:
                    logger.error(f"Error during agent {agent.name}'s reflection process: {e}", exc_info=True)

        logger.info(f"Agent reflection phase completed for Discussion Phase {self.discussionRound}.")

    def run_voting_round(self) -> tuple[Optional[int], Dict[int, int]]:
        """Runs a single voting round, collects votes, tallies them, and checks for a winner.
        Assumes self.votingRound has been set by the caller to the current election round number.
        """
        # self.votingRound += 1 # REMOVED: votingRound should be managed by the main simulation loop (e.g., in multi_round.py)
        logger.info(f"Starting Election Round {self.votingRound}")

        self.votingBuffer = {}  # Reset for current round: {candidate_id: vote_count}
        self.individual_votes_buffer = {}  # Reset for current round: {voter_agent_id: candidate_id}
        self.voting_participation[self.votingRound] = [] # Track who voted this round

        agents_to_vote = self.agents
        if not agents_to_vote:
            logger.warning("No agents available to cast votes.")
            self.votingHistory.append(self.votingBuffer.copy())
            self.individual_votes_history.append(self.individual_votes_buffer.copy())
            return None, {}

        logger.info(f"Collecting votes from {len(agents_to_vote)} agents for Election Round {self.votingRound}...") # Use ER
        if self.enable_parallel_processing:
            with ThreadPoolExecutor(max_workers=min(self.app_config.simulation.discussion_group_size, len(agents_to_vote))) as executor:
                # Future -> agent_id mapping to correctly attribute votes even if order changes
                future_to_agent_id = {executor.submit(agent.cast_vote, self.votingRound): agent.agent_id for agent in agents_to_vote}
                
                for future in tqdm(future_to_agent_id.keys(), desc=f"Collecting Votes (ER {self.votingRound})", total=len(agents_to_vote), disable=not self.output_config.logging.performance_logging): # Use ER
                    voter_agent_id = future_to_agent_id[future]
                    try:
                        voted_for_candidate_id = future.result()
                        if voted_for_candidate_id is not None and self.is_valid_vote_candidate(voted_for_candidate_id):
                            self.individual_votes_buffer[voter_agent_id] = voted_for_candidate_id
                            self.voting_participation[self.votingRound].append(voter_agent_id)
                            logger.debug(f"Agent {self.agents[voter_agent_id].name} (ID: {voter_agent_id}) voted for Agent {voted_for_candidate_id} (Candidate ID: {self.agents[voted_for_candidate_id].cardinal_id if hasattr(self.agents[voted_for_candidate_id], 'cardinal_id') else voted_for_candidate_id}).")
                        elif voted_for_candidate_id is not None:
                            logger.warning(f"Agent {self.agents[voter_agent_id].name} (ID: {voter_agent_id}) cast an invalid vote for candidate ID {voted_for_candidate_id}. Vote ignored.")
                        else:
                            logger.warning(f"Agent {self.agents[voter_agent_id].name} (ID: {voter_agent_id}) abstained or failed to vote.")
                    except Exception as e:
                        logger.error(f"Error collecting vote from Agent ID {voter_agent_id} ({self.agents[voter_agent_id].name}): {e}", exc_info=True)
        else:
            # Sequential voting
            for agent in tqdm(agents_to_vote, desc=f"Collecting Votes Sequentially (ER {self.votingRound})", disable=not self.output_config.logging.performance_logging):
                try:
                    voted_for_candidate_id = agent.cast_vote(self.votingRound)
                    if voted_for_candidate_id is not None and self.is_valid_vote_candidate(voted_for_candidate_id):
                        self.individual_votes_buffer[agent.agent_id] = voted_for_candidate_id
                        self.voting_participation[self.votingRound].append(agent.agent_id)
                        logger.debug(f"Agent {agent.name} (ID: {agent.agent_id}) voted for Agent {voted_for_candidate_id} (Candidate ID: {self.agents[voted_for_candidate_id].cardinal_id if hasattr(self.agents[voted_for_candidate_id], 'cardinal_id') else voted_for_candidate_id}).")
                    elif voted_for_candidate_id is not None:
                        logger.warning(f"Agent {agent.name} (ID: {agent.agent_id}) cast an invalid vote for candidate ID {voted_for_candidate_id}. Vote ignored.")
                    else:
                        logger.warning(f"Agent {agent.name} (ID: {agent.agent_id}) abstained or failed to vote.")
                except Exception as e:
                    logger.error(f"Error collecting vote from Agent {agent.name} (ID: {agent.agent_id}): {e}", exc_info=True)

        # Tally votes from individual_votes_buffer
        for voter_id, candidate_id in self.individual_votes_buffer.items():
            self.votingBuffer[candidate_id] = self.votingBuffer.get(candidate_id, 0) + 1

        self.votingHistory.append(self.votingBuffer.copy())
        self.individual_votes_history.append(self.individual_votes_buffer.copy())

        logger.info(f"--- Election Round {self.votingRound} Results ---")
        print(f"\n--- Election Round {self.votingRound} Results ---") # ADDED
        if not self.votingBuffer:
            logger.info("No votes were cast or tallied in this round.")
            print("No votes were cast or tallied in this round.") # ADDED
        else:
            # Sort results by vote count for logging
            sorted_vote_counts = sorted(self.votingBuffer.items(), key=lambda item: item[1], reverse=True)
            for candidate_id, votes in sorted_vote_counts:
                candidate_name = self.agents[candidate_id].name if candidate_id < len(self.agents) else f"Unknown Candidate ({candidate_id})"
                log_message = f"  Candidate {candidate_name}: {votes} votes"
                logger.info(log_message)
                print(log_message) # ADDED

        # Check for winner
        # Use number of actual voters for threshold calculation
        num_voters_this_round = len(self.voting_participation.get(self.votingRound, []))
        if num_voters_this_round == 0:
            logger.info("No voters participated in this round. Cannot determine supermajority.")
            # Handle case with no voters - perhaps no winner or specific logic
        else:
            votes_needed_for_supermajority = np.ceil(self.supermajority_threshold * num_voters_this_round)
            logger.info(f"Supermajority threshold: {self.supermajority_threshold * 100}%. Votes needed: {votes_needed_for_supermajority} out of {num_voters_this_round} voters this round.")
            
            # ADDED: Debug logging for winner determination
            logger.info(f"DEBUG: num_voters_this_round = {num_voters_this_round}")
            logger.info(f"DEBUG: supermajority_threshold = {self.supermajority_threshold}")
            logger.info(f"DEBUG: votes_needed_for_supermajority = {votes_needed_for_supermajority}")
            logger.info(f"DEBUG: votingBuffer = {self.votingBuffer}")

            for candidate_id, votes in self.votingBuffer.items():
                logger.info(f"DEBUG: Checking candidate {candidate_id} with {votes} votes against threshold {votes_needed_for_supermajority}")
                if votes >= votes_needed_for_supermajority:
                    self.winner = candidate_id
                    winner_name = self.agents[self.winner].name if self.winner < len(self.agents) else f"Unknown Candidate ({self.winner})"
                    logger.info(f"🎉 Winner found in Election Round {self.votingRound}! Candidate {winner_name} (Agent ID: {self.winner}) wins with {votes} votes. 🎉") # Use ER
                    return self.winner, self.votingBuffer.copy()
                else:
                    logger.info(f"DEBUG: Candidate {candidate_id} ({votes} votes) does not meet supermajority threshold ({votes_needed_for_supermajority} votes)")

        if self.votingRound >= self.max_election_rounds:
            logger.info(f"Maximum number of election rounds ({self.max_election_rounds}) reached. No winner determined by supermajority.") # Use ER
            # Determine winner by simple majority if no supermajority after max rounds
            if self.votingBuffer:
                # Find the candidate(s) with the most votes
                max_votes = max(self.votingBuffer.values())
                potential_winners = [cid for cid, votes in self.votingBuffer.items() if votes == max_votes]
                if len(potential_winners) == 1:
                    self.winner = potential_winners[0]
                    winner_name = self.agents[self.winner].name
                    logger.info(f"🏆 Candidate {winner_name} (Agent ID: {self.winner}) wins with a simple majority of {max_votes} votes after {self.votingRound} election rounds. 🏆") # Use ER
                else:
                    logger.info(f"Multiple candidates ({len(potential_winners)}) with max votes ({max_votes}). No clear winner.") # Use ER
            else:
                logger.info(f"No votes cast in the final round. No winner determined after {self.votingRound} election rounds.") # Use ER
        else:
            logger.info(f"No votes cast in the final round. No winner determined after {self.votingRound} election rounds.") # Use ER

        logger.info(f"No winner yet after Election Round {self.votingRound}. Proceeding to next round or discussion.") # Use ER
        return None, self.votingBuffer.copy()

    def get_agent_by_id(self, agent_id: int) -> Optional[Agent]:
        """Fetch an agent by its ID."""
        if 0 <= agent_id < len(self.agents):
            return self.agents[agent_id]
        return None