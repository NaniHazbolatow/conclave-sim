import logging
import random
import threading
import time
from typing import Dict, List, Optional, TYPE_CHECKING, Any # Ensure Any is imported
from pathlib import Path # Add Path import
from concurrent.futures import ThreadPoolExecutor # Add ThreadPoolExecutor import
from tqdm import tqdm # Add tqdm import
import pandas as pd # Add pandas import

from conclave.agents.base import Agent # For type hinting, removed AgentSettings, LLMSettings, EmbeddingSettings
from conclave.prompting.prompt_loader import PromptLoader # Corrected import to use prompting
from conclave.prompting.prompt_variable_generator import PromptVariableGenerator
from config.scripts.models import RefactoredConfig 
from config.scripts.adapter import ConfigAdapter

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
        self.votingRound = 0
        self.votingHistory = []
        self.votingBuffer = {}
        # Track which agents participated in which discussion rounds
        self.agent_discussion_participation = {}
        # Track which agents voted in each round
        self.voting_participation = {}
        self.voting_lock = threading.Lock()
        self.winner: Optional[int] = None # Winner will be an agent_id (int)
        self.discussionHistory: List[List[Dict]] = [] # List of rounds, each round is a list of comment dicts
        self.discussionRound = 0
        self.agent_discussion_participation: Dict[int, List[int]] = {} # agent_id -> list of discussion_round_numbers

        # Initialize config adapter and load configuration
        # print("ConclaveEnv.__init__: Initializing ConfigAdapter...") # DEBUG PRINT
        self.config_adapter = ConfigAdapter()
        self.app_config: RefactoredConfig = self.config_adapter.config # Access config directly
        # print("ConclaveEnv.__init__: ConfigAdapter initialized.") # DEBUG PRINT

        # Load prompts and tools
        # self.prompt_loader = PromptLoader(self.app_config) # Old incorrect line
        # print("ConclaveEnv.__init__: Initializing PromptLoader...") # DEBUG PRINT
        # Construct path to prompts.yaml within the conclave/prompting directory
        prompts_file_path = Path(__file__).parent.parent / "prompting" / "prompts.yaml"
        # print(f"ConclaveEnv.__init__: Prompts file path: {prompts_file_path.resolve()}") # DEBUG PRINT
        self.prompt_loader = PromptLoader(str(prompts_file_path.resolve())) # Pass the correct path string
        # print("ConclaveEnv.__init__: PromptLoader initialized.") # DEBUG PRINT
        
        # print("ConclaveEnv.__init__: Initializing PromptVariableGenerator...") # DEBUG PRINT
        self.prompt_variable_generator = PromptVariableGenerator(self, self.prompt_loader) # Pass self (env) and prompt_loader
        # print("ConclaveEnv.__init__: PromptVariableGenerator initialized.") # DEBUG PRINT

        # Extract relevant config sections for easier access
        # print("ConclaveEnv.__init__: Extracting config sections...") # DEBUG PRINT
        self.agent_config = self.app_config.agent
        self.simulation_config = self.app_config.simulation
        self.output_config = self.app_config.output

        self.num_agents = self.simulation_config.num_cardinals # Override num_agents from constructor
        self.discussion_group_size = self.simulation_config.discussion_group_size
        self.max_election_rounds = self.simulation_config.max_election_rounds
        self.min_discussion_words = self.simulation_config.discussion_length.min_words
        self.max_discussion_words = self.simulation_config.discussion_length.max_words

        self.supermajority_threshold = self.simulation_config.voting.supermajority_threshold

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
        
        # Update stances in parallel
        # Consider making max_workers configurable
        with ThreadPoolExecutor(max_workers=min(self.app_config.simulation.discussion_group_size, len(agents_to_update))) as executor: 
            futures = [executor.submit(agent.generate_internal_stance) for agent in agents_to_update]
            # Wait for all futures to complete
            # The disable=True for tqdm might be from old config, consider if it should be dynamic
            for future in tqdm(futures, desc="Generating Stances", total=len(futures), disable=not self.output_config.logging.performance_logging):
                future.result()  # This blocks until the task completes
        
        # Log completion and summary
        updated_count = sum(1 for agent in agents_to_update if agent.internal_stance) # Make sure agent.internal_stance is the correct attribute
        logger.info(f"Internal stance update completed: {updated_count}/{len(agents_to_update)} agents updated")
        
        # Log stance summary for this round
        logger.info(f"=== STANCE UPDATE SUMMARY (V{self.votingRound}.D{self.discussionRound}) ===")
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
        
        # Generate initial stances in parallel
        # Consider making max_workers configurable
        with ThreadPoolExecutor(max_workers=min(self.app_config.simulation.discussion_group_size, len(self.agents))) as executor: 
            futures = [executor.submit(agent.generate_internal_stance) for agent in self.agents]
            # Wait for all futures to complete
            # The disable=True for tqdm might be from old config, consider if it should be dynamic
            for future in tqdm(futures, desc="Generating Initial Stances", total=len(futures), disable=not self.output_config.logging.performance_logging):
                future.result()  # This blocks until the task completes
        
        logger.info("Initial internal stances generated for all agents")
        
        # Log initial stance summary
        # stances = self.get_all_stances() # get_all_stances method needs to be added
        # logger.info(f"=== INITIAL STANCES SUMMARY ({len(stances)} agents) ===")
        # for name, stance in stances.items():
        #     stance_preview = stance[:100].replace('\n', ' ')
        #     logger.info(f"{name}: {stance_preview}...")
        # logger.info("=== END INITIAL STANCES SUMMARY ===")
        # Temporarily comment out stance logging as get_all_stances is not yet defined
        logger.info("Stance logging in generate_initial_stances temporarily commented out.")

    def _generate_discussion_group_assignments(self) -> list[list[int]]:
        """
        Creates randomized discussion groups for a full discussion cycle.
        Ensures groups have at least 2 agents if possible, by adjusting the last two groups
        if the last group would have only 1 agent.
        """
        # Use discussion_group_size from the new config
        discussion_group_size = self.simulation_config.discussion_group_size
        
        if discussion_group_size < 2:
            logger.warning(
                f"Configured 'discussion_group_size' ({discussion_group_size}) is less than 2. "
                f"Adjusting to 2 for discussion group formation."
            )
            discussion_group_size = 2

        agent_ids = list(range(len(self.agents)))
        if not agent_ids:
            return []
            
        random.shuffle(agent_ids)

        groups = []
        i = 0
        num_total_agents = len(agent_ids)
        while i < num_total_agents:
            groups.append(agent_ids[i : i + discussion_group_size])
            i += discussion_group_size

        # Adjustment: if the last group has 1 agent and there's more than one group,
        # move one agent from the second-to-last group to the last group.
        if len(groups) > 1 and len(groups[-1]) == 1:
            if len(groups[-2]) > 0: # True if group_size >= 2 for group[-2] formation
                element_to_move = groups[-2].pop()
                groups[-1].insert(0, element_to_move)
        
        return [g for g in groups if g] # Filter out any empty groups

    def _process_discussion_group(self, group_idx: int, group_agent_ids: List[int], current_discussion_round: int): # Added current_discussion_round
        """Processes a single discussion group, allowing each agent to speak."""
        group_comments = []
        # Create a list of agent names for logging
        participant_names = [self.agents[agent_id].name for agent_id in group_agent_ids]
        logger.info(f"Voting Round {self.votingRound}, Discussion Group {group_idx + 1}/{self.num_discussion_groups} now speaking. Participants: {', '.join(participant_names)}")

        for agent_id in group_agent_ids:
            agent = self.agents[agent_id]
            try:
                # Pass current_discussion_round to agent.discuss
                comment_data = agent.discuss(current_discussion_group_ids=group_agent_ids, current_discussion_round=current_discussion_round)
                if comment_data and isinstance(comment_data, dict) and "message" in comment_data:
                    group_comments.append(comment_data)
                    # REMOVED: discussion_logger.info(f"VR {self.votingRound}, DG {group_idx + 1}, Agent {agent.name} (ID: {agent_id}): {comment_data['message']}")
                    # self.log_discussion_message(agent_id, comment_data["message"]) # Handled by agent's logger
                else:
                    logger.warning(f"Agent {agent.name} (ID: {agent_id}) in group {group_idx+1} did not return a valid comment structure.")
                    group_comments.append({"agent_id": agent_id, "message": "(Agent did not provide a valid comment structure)"})
            except Exception as e:
                logger.error(f"Critical exception during discussion processing for agent {agent.name} (ID: {agent_id}): {e}", exc_info=True)
                # Add a placeholder comment indicating the error
                group_comments.append({"agent_id": agent_id, "message": f"[System Error: A critical error occurred while processing discussion for {agent.name}. See logs.]"})
        
        # Consolidate and log all comments for the group
        if group_comments:
            log_header = f"VR {self.votingRound}, DG {group_idx + 1} (Participants: {', '.join(participant_names)}):"
            messages_str_parts = [log_header]
            for comment in group_comments:
                agent_name = "UnknownAgent"
                csv_cardinal_id = "N/A"
                internal_agent_id = comment.get('agent_id') # Get internal agent_id from comment

                if isinstance(internal_agent_id, int) and 0 <= internal_agent_id < len(self.agents):
                    agent = self.agents[internal_agent_id]
                    agent_name = agent.name
                    csv_cardinal_id = getattr(agent, 'cardinal_id', 'N/A') # Get Cardinal_ID from agent object
                else:
                    agent_name = f"Agent (Internal ID: {internal_agent_id if internal_agent_id is not None else 'N/A'})"
                
                messages_str_parts.append(f"  {agent_name} (Cardinal ID: {csv_cardinal_id}): {comment.get('message', '[No message]')}")
            
            consolidated_message = "\n".join(messages_str_parts)
            discussion_logger.info(consolidated_message)

        return group_comments

    def run_discussion_round(self):
        """Manages a single discussion round, including forming groups and collecting comments."""
        self.discussionRound += 1 # Increment discussion round for the current voting round
        
        # Correctly determine num_discussion_groups based on the number of agents and group size
        # This should be calculated based on the actual number of agents and the configured group size.
        if self.num_agents == 0 or self.discussion_group_size == 0: # Prevent division by zero
            self.num_discussion_groups = 0
        else:
            self.num_discussion_groups = (self.num_agents + self.discussion_group_size - 1) // self.discussion_group_size

        logger.info(f"Starting Discussion Phase {self.discussionRound} (part of Voting Round {self.votingRound}) with {self.num_discussion_groups} discussion groups processed in parallel.")

        discussion_groups = self._generate_discussion_group_assignments() # Corrected method name
        all_comments_for_phase = []

        with ThreadPoolExecutor(max_workers=self.num_discussion_groups) as executor:
            futures = []
            for group_idx, group_agent_ids in enumerate(discussion_groups):
                # Pass the current discussionRound to _process_discussion_group
                futures.append(executor.submit(self._process_discussion_group, group_idx, group_agent_ids, self.discussionRound))
            
            for future in tqdm(futures, desc=f"Processing Discussion Groups (VR {self.votingRound}, DP {self.discussionRound})", total=len(futures), disable=self.output_config.tqdm.disable_tqdm):
                try:
                    group_comments = future.result()
                    if group_comments:
                        all_comments_for_phase.extend(group_comments)
                except Exception as e:
                    logger.error(f"Error processing a discussion group: {e}", exc_info=True)
        
        if all_comments_for_phase:
            self.discussionHistory.append(all_comments_for_phase)
            participating_agent_ids_this_round = list(set([comment['agent_id'] for comment in all_comments_for_phase]))
            for agent_id in participating_agent_ids_this_round:
                if agent_id not in self.agent_discussion_participation:
                    self.agent_discussion_participation[agent_id] = []
                self.agent_discussion_participation[agent_id].append(self.discussionRound) 

        logger.info(f"Discussion Phase {self.discussionRound} (Voting Round {self.votingRound}) completed with {len(all_comments_for_phase)} comments.")

        # Update internal stances after discussion round
        logger.info(f"Updating internal stances for all agents after discussion phase {self.discussionRound}.")
        self.update_internal_stances()

    def run_voting_round(self) -> bool:
        """
        Manages a single voting round, including collecting votes and checking for a winner.
        Returns True if a winner is found, False otherwise.
        """
        self.votingBuffer.clear()  # Clear buffer for the new round
        logger.info(f"\\n--- Voting Phase (Round {self.votingRound}) ---")

        # Use ThreadPoolExecutor to parallelize vote casting
        # Each agent's cast_vote method will handle LLM calls and retries,
        # then call self.cast_vote(chosen_candidate_id, agent_id) to record the vote.
        futures = []
        with ThreadPoolExecutor(max_workers=self.num_agents) as executor:
            for agent in self.agents:
                # agent.cast_vote is expected to return True on success, False on failure
                # and to call self.cast_vote(chosen_candidate_id, agent.agent_id) internally
                futures.append(executor.submit(agent.cast_vote, self.votingRound))
            
            successful_votes = 0
            failed_votes = 0
            for future in tqdm(futures, desc=f"Collecting Votes (Round {self.votingRound})", total=len(futures), disable=self.output_config.tqdm.disable_tqdm):
                try:
                    if future.result(): # result() will re-raise exceptions from agent.cast_vote
                        successful_votes += 1
                    else:
                        failed_votes += 1 # Agent explicitly returned False (e.g. max retries for LLM)
                except Exception as e:
                    logger.error(f"Exception during agent vote processing: {e}", exc_info=True)
                    failed_votes +=1

            logger.info(f"Vote collection summary for Round {self.votingRound}: {successful_votes} successful, {failed_votes} failed attempts.")

        self.votingHistory.append(self.votingBuffer.copy())
        
        # SANITY CHECK: Ensure vote count matches agent count if all agents succeeded
        # Note: votingBuffer is populated by agent.cast_vote -> self.env.cast_vote
        # The number of entries in votingBuffer might not directly map to successful_votes
        # if an agent fails before calling self.env.cast_vote.
        # A better check is on the number of votes recorded in the buffer.
        
        # Let's count how many agents actually recorded a vote in the buffer for this round.
        # This requires knowing which votes in votingBuffer belong to the current round.
        # Since votingBuffer is cleared and then populated, its current state IS the current round's votes.
        
        actual_votes_recorded_in_buffer = 0
        # The votingBuffer stores {candidate_id: [voter_agent_id_1, voter_agent_id_2,...]}
        # Or, if it's {candidate_id: num_votes}, then sum(self.votingBuffer.values())
        # Based on `self.env.cast_vote`, it's:
        # self.votingBuffer[candidate_id] = self.votingBuffer.get(candidate_id, 0) + 1
        
        total_votes_in_buffer_for_round = sum(self.votingBuffer.values())

        logger.info(f"=== VOTING SANITY CHECK (Round {self.votingRound}) ===")
        logger.info(f"Total votes recorded in buffer for this round: {total_votes_in_buffer_for_round}")
        logger.info(f"Expected votes (num_agents): {self.num_agents}")
        
        if total_votes_in_buffer_for_round != self.num_agents:
            logger.warning(f"VOTE COUNT MISMATCH in buffer: Expected {self.num_agents}, got {total_votes_in_buffer_for_round}. This may be due to voting failures.")
            # Potentially log which agents failed to register a vote if that info is available

        # Check for winner
        if not self.votingBuffer:
            logger.warning(f"No votes were cast in round {self.votingRound}. Cannot determine a winner.")
            # Decide how to handle this - e.g., continue to next round or end simulation
            return False # No winner

        # Sort candidates by votes
        # self.votingBuffer is {candidate_id: num_votes}
        sorted_votes = sorted(self.votingBuffer.items(), key=lambda item: item[1], reverse=True)

        if not sorted_votes:
            logger.warning(f"Vote buffer was non-empty but produced no sorted results in round {self.votingRound}.")
            return False # No winner

        # Log vote counts for the round
        logger.info(f"--- Vote Counts (Round {self.votingRound}) ---")
        for candidate_id, num_votes in sorted_votes:
            candidate_name = self.agents[candidate_id].name if candidate_id < len(self.agents) and candidate_id != -1 else f"InvalidCandidateID ({candidate_id})"
            if candidate_id == -1: candidate_name = "Abstention"
            logger.info(f"  {candidate_name}: {num_votes} votes")
            print(f"  {candidate_name}: {num_votes} votes") # Print to terminal

        # Check for supermajority
        top_candidate_id, top_votes = sorted_votes[0]
        
        # Ensure total_votes_in_buffer_for_round is not zero to avoid DivisionByZeroError
        if total_votes_in_buffer_for_round == 0:
            logger.warning(f"Total votes in buffer is zero for round {self.votingRound}. Cannot calculate supermajority.")
            return False # No winner, or handle as per simulation rules

        if (top_votes / total_votes_in_buffer_for_round) >= self.supermajority_threshold:
            winner_name = self.agents[top_candidate_id].name if top_candidate_id < len(self.agents) else f"InvalidCandidateID ({top_candidate_id})"
            logger.info(f"\\n🎉 Winner found in Round {self.votingRound}! {winner_name} wins with {top_votes} votes ({top_votes/total_votes_in_buffer_for_round*100:.2f}% of votes).")
            self.winner = top_candidate_id
            return True # Winner found

        logger.info(f"No winner yet in Round {self.votingRound}. Top candidate has {top_votes} votes ({top_votes/total_votes_in_buffer_for_round*100:.2f}%). Threshold: {self.supermajority_threshold*100:.2f}%")
        
        # Update candidate list if dynamic candidate elimination is implemented
        # self.update_candidate_list_after_voting() # Example placeholder

        return False # No winner yet

    def cast_vote(self, candidate_id: int, voter_agent_id: int):
        """
        Records a vote from an agent. Called by the agent itself after deciding.
        Args:
            candidate_id: The agent_id of the candidate being voted for.
                          -1 can represent abstention or an invalid vote.
            voter_agent_id: The agent_id of the agent casting the vote.
        """
        if candidate_id == -1: # Abstention or invalid vote from LLM
            logger.warning(f"Agent {self.agents[voter_agent_id].name} (ID: {voter_agent_id}) abstained or cast an invalid vote.")
            # Decide how to record abstentions, e.g., in a separate counter or ignore for majority calc
            # For now, we'll count it towards the total votes if it's explicitly cast as -1
            # but it won't win. If it's not added to votingBuffer, total_votes_in_buffer_for_round will be lower.
            # Let's assume for now that an abstention means they don't add to any candidate's tally.
            # If an agent fails to produce a vote, they also don't add to the tally.
            # The current logic for total_votes_in_buffer_for_round correctly sums actual candidate votes.
            return

        if candidate_id not in self.candidate_ids and candidate_id not in self.current_candidates:
             logger.warning(f"Agent {self.agents[voter_agent_id].name} (ID: {voter_agent_id}) voted for {self.agents[candidate_id].name} (ID: {candidate_id}) who is not a current candidate. Vote ignored.")
             return


        # Ensure candidate_id is a valid agent ID (and a candidate)
        if candidate_id < 0 or candidate_id >= len(self.agents):
            logger.error(f"Agent {self.agents[voter_agent_id].name} (ID: {voter_agent_id}) tried to vote for invalid candidate_id: {candidate_id}. Vote ignored.")
            return

        # Record the vote in the buffer for the current round
        # self.votingBuffer stores: {candidate_agent_id: number_of_votes}
        self.votingBuffer[candidate_id] = self.votingBuffer.get(candidate_id, 0) + 1
        logger.debug(f"Agent {self.agents[voter_agent_id].name} (ID: {voter_agent_id}) cast vote for {self.agents[candidate_id].name} (ID: {candidate_id}). Current buffer: {self.votingBuffer}")