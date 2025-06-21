"""
Network Manager for Conclave Simulation

This module manages the cardinal network and provides grouping functionality
for the conclave simulation environment.
"""

import logging
import random
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
import networkx as nx
import pandas as pd

from conclave.network.breakout_scheduler import BreakoutScheduler

logger = logging.getLogger(__name__)


class NetworkManager:
    """Manages the cardinal network and provides grouping functionality."""
    
    def __init__(self, config_manager=None):
        """Initialize the NetworkManager with configuration."""
        self.config_manager = config_manager
        self.network_graph: Optional[nx.Graph] = None
        self.cardinal_id_to_agent_id_map: Dict[int, int] = {}
        self.agent_id_to_cardinal_id_map: Dict[int, int] = {}
        self.current_groups: List[List[int]] = []  # List of groups, each group is list of agent_ids
        self.breakout_scheduler: Optional[BreakoutScheduler] = None  # Advanced multi-round scheduler
        
    def initialize_network(self, agents: List) -> None:
        """
        Initialize the network mappings with the given agents.
        
        Args:
            agents: List of Agent objects from the conclave environment
        """
        logger.info("Initializing cardinal network mappings...")
        
        # Note: The actual network loading is now handled by BreakoutScheduler in conclave_env.py
        # This class only manages the mappings between agent_ids and cardinal_ids
        
        # Create mapping between agent_ids (list indices) and cardinal_ids
        self.cardinal_id_to_agent_id_map = {}
        self.agent_id_to_cardinal_id_map = {}
        
        for agent in agents:
            if hasattr(agent, 'cardinal_id') and agent.cardinal_id is not None:
                self.cardinal_id_to_agent_id_map[agent.cardinal_id] = agent.agent_id
                self.agent_id_to_cardinal_id_map[agent.agent_id] = agent.cardinal_id
        
        logger.info(f"Created mappings for {len(self.cardinal_id_to_agent_id_map)} cardinals")
    
    def get_cardinal_id_to_agent_id_map(self) -> Dict[int, int]:
        """Get the mapping from cardinal_id to agent_id."""
        return self.cardinal_id_to_agent_id_map.copy()
    
    def get_agent_id_to_cardinal_id_map(self) -> Dict[int, int]:
        """Get the mapping from agent_id to cardinal_id."""
        return self.agent_id_to_cardinal_id_map.copy()
    
