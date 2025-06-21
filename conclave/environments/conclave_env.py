import logging
import random
import threading
import time
from typing import Dict, List, Optional, TYPE_CHECKING, Any # Ensure Any is imported
from pathlib import Path # Add Path import
from concurrent.futures import ThreadPoolExecutor # Add ThreadPoolExecutor import
from tqdm import tqdm # Add tqdm import
import pandas as pd # Add pandas import
import numpy as np # Add numpy import for data export methods

from conclave.agents.base import Agent # For type hinting, removed AgentSettings, LLMSettings, EmbeddingSettings
from conclave.prompting.prompt_loader import PromptLoader # Corrected import to use prompting
from conclave.prompting.unified_generator import UnifiedPromptVariableGenerator
from conclave.config import get_config_manager # Use centralized config management
from config.scripts import get_config  # Import new config adapter
from conclave.llm import get_llm_client, SimplifiedToolCaller # Use new simplified tool calling
from conclave.network.network_manager import NetworkManager # Add network manager import
from config.scripts.models import RefactoredConfig

if TYPE_CHECKING:
    from ..agents.base import Agent # For type hinting

logger = logging.getLogger(__name__)
# Create a specific logger for discussion outputs
discussion_logger = logging.getLogger('conclave.discussions') 

class ConclaveEnv:
    def __init__(self, viz_dir: Optional[str] = None):
        self.agents: List[Agent] = []
        self.viz_dir = viz_dir
        self.votingRound = 0
        self.votingHistory = []
        self.votingBuffer = {}
        self.individual_votes_buffer = {}
        self.individual_votes_history = []
        self.agent_discussion_participation = {}
        self.voting_participation = {}
        self.voting_lock = threading.Lock()
        self.winner: Optional[int] = None
        self.discussionHistory: List[List[Dict]] = []
        self.discussionRound = 0
        self.agent_discussion_participation: Dict[int, List[int]] = {}

        self.config_manager = get_config()
        self.app_config: RefactoredConfig = self.config_manager.config

        prompts_file_path = Path(__file__).parent.parent / "prompting" / "prompts.yaml"
        self.prompt_loader = PromptLoader(str(prompts_file_path.resolve()))
        
        self.prompt_variable_generator = UnifiedPromptVariableGenerator(self, self.prompt_loader)

        try:
            self.env_llm_client = get_llm_client("environment")
            self.env_tool_caller = SimplifiedToolCaller(self.env_llm_client, logger)
            logger.info(f"ConclaveEnv initialized its own LLM client: {self.env_llm_client.model_name} for environment tasks.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client for ConclaveEnv: {e}")
            self.env_llm_client = None
            self.env_tool_caller = None

        self.agent_config = self.app_config.models
        self.simulation_config = self.app_config.simulation
        self.output_config = self.app_config.output

        logger.debug(f"ConclaveEnv initialized with app_config: {self.app_config.model_dump_json(indent=2)}")

        self.num_agents = self.config_manager.get_num_cardinals()
        self.discussion_group_size = self.simulation_config.discussion_group_size
        self.max_election_rounds = self.simulation_config.max_election_rounds
        self.min_discussion_words = self.simulation_config.discussion_length.min_words
        self.max_discussion_words = self.simulation_config.discussion_length.max_words

        self.supermajority_threshold = self.simulation_config.voting.supermajority_threshold
        self.enable_parallel_processing = self.simulation_config.enable_parallel_processing

        # Initialize network manager for grouping
        self.network_manager = NetworkManager(self.config_manager)
        
        # Initialize BreakoutScheduler (will be set up when agents are created)
        self.breakout_scheduler = None

        # Predefined groups configuration (if enabled)
        if self.app_config.groups and self.app_config.groups.active:
            self.predefined_groups_enabled = True
            self.active_group_name = self.app_config.groups.active
            
            # Load the actual group data from groups.yaml using the active group from current config
            try:
                # Load groups data directly from the original project location
                # (groups.yaml is not copied to output directory)
                project_root = Path(__file__).parent.parent.parent
                groups_file_path = project_root / "config" / "groups.yaml"
                
                if groups_file_path.exists():
                    with open(groups_file_path, 'r') as f:
                        import yaml
                        groups_data = yaml.safe_load(f)
                    
                    active_group_config = groups_data["predefined_groups"].get(self.active_group_name)
                else:
                    raise FileNotFoundError(f"Groups file not found: {groups_file_path}")
                
                if active_group_config:
                    # Override simulation parameters if a predefined group is active
                    self.num_agents = active_group_config['total_cardinals']
                    # Apply simulation settings overrides if they exist
                    if 'override_settings' in active_group_config:
                        override_settings = active_group_config['override_settings']
                        self.discussion_group_size = override_settings.get('discussion_group_size', self.discussion_group_size)
                        self.max_election_rounds = override_settings.get('max_election_rounds', self.max_election_rounds) 
                        self.supermajority_threshold = override_settings.get('supermajority_threshold', self.supermajority_threshold)
                    
                    logger.info(f"RUNNING WITH PREDEFINED GROUP: {self.active_group_name}")
                    logger.info(f"  Num Cardinals: {self.num_agents}")
                    logger.info(f"  Discussion Group Size: {self.discussion_group_size}")
                    logger.info(f"  Max Election Rounds: {self.max_election_rounds}")
                    logger.info(f"  Supermajority Threshold: {self.supermajority_threshold}")
                else:
                    logger.error(f"Active group '{self.active_group_name}' not found in predefined groups")
                    self.predefined_groups_enabled = False
            except FileNotFoundError:
                logger.error(f"Groups file not found, disabling predefined groups")
                self.predefined_groups_enabled = False
        else:
            self.predefined_groups_enabled = False

        self.agents = self._initialize_agents()
        
        self.network_manager.initialize_network(self.agents)
        
        self._initialize_breakout_scheduler()

    def _initialize_agents(self) -> List[Agent]:
        """Load agents based on testing groups configuration or all agents in normal mode."""
        project_root = Path(__file__).parent.parent.parent 
        cardinals_data_path = project_root / "data" / "cardinals_master_data.csv"
        master_df = pd.read_csv(cardinals_data_path)
        
        loaded_agents = []

        if self.predefined_groups_enabled:
            try:
                # Always load groups.yaml from the project root, not from output directory
                project_root = Path(__file__).parent.parent.parent
                groups_file_path = project_root / "config" / "groups.yaml"
                
                with open(groups_file_path, 'r') as f:
                    import yaml
                    groups_data = yaml.safe_load(f)
                    
                active_group_config = groups_data["predefined_groups"].get(self.active_group_name)
                
                if active_group_config:
                    cardinal_ids_for_group = active_group_config['cardinal_ids']
                    logger.info(f"Loading predefined group '{self.active_group_name}' with {len(cardinal_ids_for_group)} cardinals: {cardinal_ids_for_group}")
                    
                    # Filter dataframe to only include cardinals in the predefined group
                    master_df_filtered = master_df[master_df['Cardinal_ID'].isin(cardinal_ids_for_group)]
                    
                    if len(master_df_filtered) != len(cardinal_ids_for_group):
                        loaded_ids_from_csv = set(master_df_filtered['Cardinal_ID'].tolist())
                        missing_ids = set(cardinal_ids_for_group) - loaded_ids_from_csv
                        if missing_ids:
                            logger.warning(f"Some cardinal IDs for predefined group '{self.active_group_name}' not found in CSV: {missing_ids}")
                    master_df_to_load = master_df_filtered
                else:
                    logger.error(f"Active group '{self.active_group_name}' not found in groups data")
                    master_df_to_load = master_df.head(self.num_agents)  # Fallback
            except FileNotFoundError:
                logger.error(f"Groups file not found, falling back to default agent loading")
                master_df_to_load = master_df.head(self.num_agents)  # Fallback
        else:
            # In normal mode, use num_agents from config manager
            num_to_load = self.config_manager.get_num_cardinals() 

            if num_to_load and num_to_load < len(master_df):
                master_df_to_load = master_df.head(num_to_load)
                logger.info(f"Loading first {num_to_load} cardinals from CSV as per configuration.")
            else:
                master_df_to_load = master_df
                logger.info(f"Loading all {len(master_df_to_load)} cardinals from CSV.")

        for list_index, (idx, row) in enumerate(master_df_to_load.iterrows()):
            agent = Agent(
                agent_id=list_index,
                name=row['Name'],
                conclave_env=self,
                personality=row.get('Internal_Persona', ''),
                initial_stance=row.get('Public_Profile', ''),
                party_loyalty=0.5
            )
            agent.cardinal_id = row['Cardinal_ID']
            agent.background_csv = row['Background']
            agent.internal_persona_csv = row.get('Internal_Persona', '')
            agent.public_profile_csv = row.get('Public_Profile', '')
            agent.profile_blurb_csv = row.get('Persona_Tag', '')
            agent.persona_tag_csv = row.get('Persona_Tag', '')

            loaded_agents.append(agent)
        
        self.num_agents = len(loaded_agents)

        if self.predefined_groups_enabled:
            try:
                # Get active group config using the current config object (handles CLI overrides)
                # Always load groups.yaml from the project root, not from output directory
                project_root = Path(__file__).parent.parent.parent
                groups_file_path = project_root / "config" / "groups.yaml"
                
                with open(groups_file_path, 'r') as f:
                    import yaml
                    groups_data = yaml.safe_load(f)
                    
                active_group_config = groups_data["predefined_groups"].get(self.active_group_name)
                
                if active_group_config:
                    original_candidate_ids_for_group = active_group_config['candidate_ids']
                    
                    # Create a mapping from Cardinal_ID to agent_id (list_index)
                    cardinal_id_to_agent_id_map = {int(row['Cardinal_ID']): list_idx for list_idx, (_, row) in enumerate(master_df_to_load.iterrows())}
                    
                    self.candidate_ids = [cardinal_id_to_agent_id_map[cid] for cid in original_candidate_ids_for_group if cid in cardinal_id_to_agent_id_map]
                    
                    if len(self.candidate_ids) != len(original_candidate_ids_for_group):
                        missing_candidates = set(original_candidate_ids_for_group) - set(cardinal_id_to_agent_id_map.keys())
                        logger.warning(
                            f"Some candidate Cardinal_IDs for predefined group '{self.active_group_name}' were not loaded: {missing_candidates}. "
                            f"Available candidates (agent_ids): {self.candidate_ids}"
                        )
                    logger.info(f"Mapped candidate Cardinal_IDs {original_candidate_ids_for_group} to agent_ids {self.candidate_ids} for predefined group '{self.active_group_name}'.")
                else:
                    # Fallback if group config not found
                    num_candidates = min(2, len(loaded_agents))
                    self.candidate_ids = list(range(num_candidates))
                    logger.warning(f"Could not load candidate mapping for group '{self.active_group_name}', using default: {self.candidate_ids}")
            except FileNotFoundError:
                # Fallback if groups file not found
                num_candidates = min(2, len(loaded_agents))
                self.candidate_ids = list(range(num_candidates))
                logger.warning(f"Groups file not found, using default candidate mapping: {self.candidate_ids}")
        else:
            self.candidate_ids = list(range(self.num_agents))

        logger.info(f"Successfully loaded {self.num_agents} agents.")
        if self.predefined_groups_enabled:
            logger.info(f"Predefined group '{self.active_group_name}' active. Candidate agent_ids: {self.candidate_ids}")
        return loaded_agents

    def freeze_agent_count(self):
        """Freeze the agent count to match the loaded roster. Call after loading all agents."""
        self.num_agents = len(self.agents)

    # Predefined groups candidate/elector helper methods
    def is_candidate(self, agent_id: int) -> bool:
        """Check if an agent is a candidate in predefined groups mode."""
        if not self.predefined_groups_enabled:
            return True  # In normal mode, all agents can be candidates
        return agent_id in self.candidate_ids

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
        
        # Store the discussion groups for reflection phase
        self._last_discussion_groups = current_discussion_groups
        
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
        """Generate discussion groups using BreakoutScheduler."""
        logger.debug("Generating discussion groups with BreakoutScheduler")
        
        # Ensure BreakoutScheduler is initialized
        if self.breakout_scheduler is None:
            logger.warning("BreakoutScheduler not initialized, initializing now")
            self._initialize_breakout_scheduler()
        
        if self.breakout_scheduler is None:
            logger.error("Failed to initialize BreakoutScheduler, cannot generate groups")
            return []
        
        # Generate groups for this round using BreakoutScheduler
        cardinal_groups = self.breakout_scheduler.next_round()
        
        # Convert cardinal_id groups to agent_id groups
        agent_groups = []
        assigned_agent_ids = set()
        
        for i, cardinal_group in enumerate(cardinal_groups):
            agent_group = []
            missing_cardinals = []
            for cardinal_id in cardinal_group:
                if cardinal_id in self.network_manager.cardinal_id_to_agent_id_map:
                    agent_id = self.network_manager.cardinal_id_to_agent_id_map[cardinal_id]
                    agent_group.append(agent_id)
                    assigned_agent_ids.add(agent_id)
                else:
                    missing_cardinals.append(cardinal_id)
            
            if missing_cardinals:
                logger.warning(f"Group {i+1}: Cardinals {missing_cardinals} not found in cardinal_id_to_agent_id_map")
            
            if agent_group:  # Only add non-empty groups
                agent_groups.append(agent_group)
        
        # Check for unassigned agents and create fallback groups
        all_agent_ids = set(range(len(self.agents)))
        unassigned_agent_ids = all_agent_ids - assigned_agent_ids
        
        if unassigned_agent_ids:
            logger.warning(f"Agents {sorted(unassigned_agent_ids)} not assigned to any group by BreakoutScheduler")
            # Create additional groups for unassigned agents
            unassigned_list = list(unassigned_agent_ids)
            # Use discussion_group_size for consistency
            for i in range(0, len(unassigned_list), self.discussion_group_size):
                fallback_group = unassigned_list[i:i + self.discussion_group_size]
                agent_groups.append(fallback_group)
                logger.info(f"Created fallback group: {fallback_group}")
        
        logger.info(f"BreakoutScheduler generated {len(agent_groups)} groups for round {self.breakout_scheduler._round}")
        for i, group in enumerate(agent_groups):
            logger.info(f"  Group {i+1}: {len(group)} members")
        
        return agent_groups

    def _initialize_breakout_scheduler(self):
        """Initialize the BreakoutScheduler with the bocconi network."""
        import pickle
        import os
        from conclave.network.breakout_scheduler import BreakoutScheduler
        
        # Load the bocconi graph
        project_root = Path(__file__).parent.parent.parent
        graph_path = project_root / "data" / "network" / "bocconi_graph.gpickle"
        
        with open(graph_path, 'rb') as f:
            G_multiplex = pickle.load(f)
        
        logger.info(f"Loaded bocconi network with {len(G_multiplex.nodes)} nodes")
        
        # Initialize the network manager mappings
        self.network_manager.initialize_network(self.agents)
        
        # Get the cardinal_ids that are actually in the simulation
        simulation_cardinal_ids = set()
        for agent in self.agents:
            if hasattr(agent, 'cardinal_id') and agent.cardinal_id is not None:
                simulation_cardinal_ids.add(agent.cardinal_id)
        
        logger.info(f"Simulation has {len(simulation_cardinal_ids)} cardinals")
        
        # Create subgraph with only cardinals from the simulation
        available_cardinal_ids = simulation_cardinal_ids.intersection(set(G_multiplex.nodes()))
        
        if len(available_cardinal_ids) == 0:
            logger.error("No cardinal IDs from simulation found in network!")
            self.breakout_scheduler = None
            return
            
        G_simulation = G_multiplex.subgraph(available_cardinal_ids).copy()
        logger.info(f"Created simulation subgraph with {len(G_simulation.nodes)} nodes and {len(G_simulation.edges)} edges")
        
        missing_simulation_cardinals = simulation_cardinal_ids - set(G_multiplex.nodes())
        if missing_simulation_cardinals:
            logger.warning(f"Cardinals in simulation but not in network: {missing_simulation_cardinals}")
        
        # Create BreakoutScheduler with the filtered network
        grouping_config = self.simulation_config.grouping
        weights = (
            grouping_config.utility_weights.connection,
            grouping_config.utility_weights.ideology,
            grouping_config.utility_weights.influence,
            grouping_config.utility_weights.interaction
        )
        
        self.breakout_scheduler = BreakoutScheduler(
            G_simulation, 
            room_size=self.discussion_group_size,  # Use discussion_group_size consistently
            rationality=self.app_config.simulation.rationality,  # Use simulation rationality
            penalty_weight=grouping_config.penalty_weight,
            weights=weights
        )
        
        logger.info("BreakoutScheduler initialized successfully")

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
            # Use the same calculation as in prompts: int(total_electors * threshold)
            votes_needed_for_supermajority = int(self.supermajority_threshold * num_voters_this_round)
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

    def save_simulation_results(self, output_dir: Path, timestamp: str) -> None:
        """
        Save comprehensive simulation results including individual votes, embeddings, and enhanced summary.
        
        Args:
            output_dir: Base output directory for the simulation
            timestamp: Simulation timestamp for file naming
        """
        results_dir = output_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        logger.info("Saving comprehensive simulation results...")
        
        # Save enhanced simulation summary
        self._save_enhanced_simulation_summary(results_dir, timestamp)
        
        # Save individual voting data
        self._save_individual_voting_data(results_dir)
        
        # Save stance embeddings
        self._save_stance_embeddings(results_dir)
        
        # Save final round vote summary
        self._save_final_round_votes(results_dir)
        
        logger.info("Simulation results saved successfully")
    
    def _save_enhanced_simulation_summary(self, results_dir: Path, timestamp: str) -> None:
        """Save enhanced simulation summary with additional metadata."""
        import json
        
        # Calculate additional statistics
        total_agents = len(self.agents)
        total_rounds = len(self.votingHistory)
        
        # Count agents with embeddings
        agents_with_embeddings = sum(1 for agent in self.agents 
                                   if hasattr(agent, 'embedding_history') and agent.embedding_history)
        
        # Count discussion participation
        total_discussions = len(self.discussionHistory)
        discussion_participation = {}
        for round_idx, round_comments in enumerate(self.discussionHistory):
            round_participants = set()
            for comment in round_comments:
                if isinstance(comment.get('agent_id'), int):
                    round_participants.add(comment['agent_id'])
            discussion_participation[f"round_{round_idx + 1}"] = len(round_participants)
        
        # Enhanced simulation summary
        enhanced_summary = {
            "metadata": {
                "timestamp": timestamp,
                "simulation_type": "multi_round_discussion",
                "version": "1.0"
            },
            "simulation_settings": {
                "num_agents": total_agents,
                "discussion_group_size": self.discussion_group_size,
                "max_election_rounds": self.max_election_rounds,
                "supermajority_threshold": self.supermajority_threshold,
                "parallel_processing_enabled": self.enable_parallel_processing
            },
            "results": {
                "winner_cardinal_id": self.winner,
                "winner_name": self.agents[self.winner].name if self.winner is not None else "N/A",
                "total_election_rounds": total_rounds,
                "winner_found": self.winner is not None,
                "final_vote_threshold_met": self._check_final_threshold() if self.winner is not None else False
            },
            "participation_stats": {
                "total_agents": total_agents,
                "agents_with_embeddings": agents_with_embeddings,
                "discussion_rounds_held": total_discussions,
                "discussion_participation_by_round": discussion_participation,
                "voting_rounds_held": total_rounds
            },
            "predefined_groups": {
                "enabled": self.predefined_groups_enabled,
                "active_group": self.active_group_name if self.predefined_groups_enabled else None,
                "candidate_agent_ids": self.candidate_ids if self.predefined_groups_enabled else None
            }
        }
        
        summary_file = results_dir / "simulation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(enhanced_summary, f, indent=4)
        
        logger.info(f"Enhanced simulation summary saved to {summary_file}")
    
    def _save_individual_voting_data(self, results_dir: Path) -> None:
        """Save detailed individual voting data for each round."""
        import json
        
        # Convert individual votes history to a more readable format
        voting_data = {}
        
        for round_idx, round_votes in enumerate(self.individual_votes_history):
            round_key = f"round_{round_idx + 1}"
            voting_data[round_key] = {}
            
            for voter_agent_id, candidate_agent_id in round_votes.items():
                # Get names and cardinal IDs for readability
                voter_name = self.agents[voter_agent_id].name if voter_agent_id < len(self.agents) else f"Agent_{voter_agent_id}"
                voter_cardinal_id = getattr(self.agents[voter_agent_id], 'cardinal_id', voter_agent_id) if voter_agent_id < len(self.agents) else voter_agent_id
                
                candidate_name = self.agents[candidate_agent_id].name if candidate_agent_id < len(self.agents) else f"Agent_{candidate_agent_id}"
                candidate_cardinal_id = getattr(self.agents[candidate_agent_id], 'cardinal_id', candidate_agent_id) if candidate_agent_id < len(self.agents) else candidate_agent_id
                
                voting_data[round_key][str(voter_cardinal_id)] = {
                    "candidate_cardinal_id": candidate_cardinal_id,
                    "voter_name": voter_name,
                    "candidate_name": candidate_name,
                    "voter_agent_id": voter_agent_id,
                    "candidate_agent_id": candidate_agent_id
                }
        
        votes_file = results_dir / "individual_votes_by_round.json"
        with open(votes_file, 'w') as f:
            json.dump(voting_data, f, indent=4)
        
        logger.info(f"Individual voting data saved to {votes_file}")
        
        # Also save in flat CSV format for analysis
        csv_data = []
        for round_key, round_votes in voting_data.items():
            round_num = int(round_key.split('_')[1])  # Extract number from "round_1"
            
            for cardinal_id, vote_info in round_votes.items():
                csv_data.append({
                    'round': round_num,
                    'agent_id': vote_info['voter_agent_id'],
                    'agent_name': vote_info['voter_name'],
                    'cardinal_id': int(cardinal_id),
                    'candidate_voted_for': vote_info['candidate_name'],
                    'candidate_cardinal_id': vote_info['candidate_cardinal_id'],
                    'candidate_agent_id': vote_info['candidate_agent_id']
                })
        
        if csv_data:
            import pandas as pd
            df = pd.DataFrame(csv_data)
            csv_file = results_dir / "voting_data.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Flat voting data saved to {csv_file}")
        else:
            logger.warning("No voting data to save in CSV format")
    
    def _save_stance_embeddings(self, results_dir: Path) -> None:
        """Save stance embeddings for all rounds in efficient format."""
        import json
        import numpy as np
        
        # Collect all embeddings and metadata
        embedding_data = {}
        metadata = {
            "rounds": [],
            "agents": [],
            "embedding_dimension": None,
            "model_used": None
        }
        
        # Get embedding client info
        try:
            from ..embeddings import get_default_client
            embedding_client = get_default_client()
            stats = embedding_client.get_embedding_stats()
            metadata["model_used"] = stats.get("model_name", "unknown")
            metadata["embedding_dimension"] = stats.get("embedding_dimension", None)
        except Exception as e:
            logger.warning(f"Could not get embedding client stats: {e}")
            metadata["model_used"] = "unknown"
            metadata["embedding_dimension"] = None
        
        # Collect agent metadata
        for agent in self.agents:
            agent_info = {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "cardinal_id": getattr(agent, 'cardinal_id', agent.agent_id)
            }
            metadata["agents"].append(agent_info)
        
        # Collect embeddings by round
        all_rounds = set()
        for agent in self.agents:
            if hasattr(agent, 'embedding_history') and agent.embedding_history:
                all_rounds.update(agent.embedding_history.keys())
        
        all_rounds = sorted(list(all_rounds), key=lambda x: (x == "initial", x))  # Sort with "initial" first
        metadata["rounds"] = all_rounds
        
        for round_key in all_rounds:
            round_embeddings = []
            for agent in self.agents:
                if (hasattr(agent, 'embedding_history') and 
                    agent.embedding_history and 
                    round_key in agent.embedding_history):
                    round_embeddings.append(agent.embedding_history[round_key])
                else:
                    # Fill with zeros if agent has no embedding for this round
                    if metadata["embedding_dimension"]:
                        round_embeddings.append(np.zeros(metadata["embedding_dimension"]))
                    else:
                        round_embeddings.append(None)
            
            if round_embeddings and round_embeddings[0] is not None:
                embedding_data[round_key] = np.array(round_embeddings)
        
        # Save embeddings as compressed numpy file
        if embedding_data:
            embeddings_file = results_dir / "stance_embeddings_by_round.npz"
            np.savez_compressed(embeddings_file, **embedding_data)
            logger.info(f"Stance embeddings saved to {embeddings_file}")
            
            # Also save in CSV format for analysis
            csv_data = []
            for round_key, round_embeddings in embedding_data.items():
                for agent_idx, embedding in enumerate(round_embeddings):
                    agent_info = metadata["agents"][agent_idx]
                    row = {
                        'round': round_key,
                        'agent_id': agent_info['agent_id'],
                        'agent_name': agent_info['name'],
                        'cardinal_id': agent_info['cardinal_id']
                    }
                    # Add embedding dimensions as separate columns
                    for dim_idx, value in enumerate(embedding):
                        row[f'embedding_{dim_idx}'] = value
                    csv_data.append(row)
            
            if csv_data:
                import pandas as pd
                df = pd.DataFrame(csv_data)
                csv_file = results_dir / "stance_embeddings.csv"
                df.to_csv(csv_file, index=False)
                logger.info(f"Stance embeddings CSV saved to {csv_file}")
        else:
            logger.warning("No embedding data found to save")
        
        # Save metadata as JSON
        metadata_file = results_dir / "stance_embeddings_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Embedding metadata saved to {metadata_file}")
    
    def _save_final_round_votes(self, results_dir: Path) -> None:
        """Save final round voting results in CSV format."""
        import pandas as pd
        
        if not self.individual_votes_history:
            logger.warning("No voting history available for final round votes")
            return
        
        # Get the final round votes
        final_round_votes = self.individual_votes_history[-1]
        final_round_number = len(self.individual_votes_history)
        
        # Create DataFrame
        vote_records = []
        for voter_agent_id, candidate_agent_id in final_round_votes.items():
            voter_name = self.agents[voter_agent_id].name if voter_agent_id < len(self.agents) else f"Agent_{voter_agent_id}"
            voter_cardinal_id = getattr(self.agents[voter_agent_id], 'cardinal_id', voter_agent_id) if voter_agent_id < len(self.agents) else voter_agent_id
            
            candidate_name = self.agents[candidate_agent_id].name if candidate_agent_id < len(self.agents) else f"Agent_{candidate_agent_id}"
            candidate_cardinal_id = getattr(self.agents[candidate_agent_id], 'cardinal_id', candidate_agent_id) if candidate_agent_id < len(self.agents) else candidate_agent_id
            
            vote_records.append({
                'voter_name': voter_name,
                'voter_cardinal_id': voter_cardinal_id,
                'candidate_name': candidate_name,
                'candidate_cardinal_id': candidate_cardinal_id,
                'round_number': final_round_number
            })
        
        if vote_records:
            df = pd.DataFrame(vote_records)
            csv_file = results_dir / "final_round_votes.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Final round votes saved to {csv_file}")
        else:
            logger.warning("No final round votes to save")
    
    def _check_final_threshold(self) -> bool:
        """Check if the winner met the required threshold in the final round."""
        if not self.votingHistory or self.winner is None:
            return False
        
        final_votes = self.votingHistory[-1]
        winner_votes = final_votes.get(self.winner, 0)
        required_votes = self._calculate_required_majority()
        
        return winner_votes >= required_votes
    
    def _calculate_required_majority(self) -> int:
        """Calculate the required majority based on current settings."""
        total_voting_agents = len(self.agents)
        return int(np.ceil(total_voting_agents * self.supermajority_threshold))