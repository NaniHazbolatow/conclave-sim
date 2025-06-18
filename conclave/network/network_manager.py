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

# Import the grouping functions from the existing module
from conclave.network.grouping import (
    load_network,
    generate_non_overlapping_utility_groups,
    ensure_ideology_scores
)

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
        
    def initialize_network(self, agents: List) -> None:
        """
        Initialize the network with the given agents.
        
        Args:
            agents: List of Agent objects from the conclave environment
        """
        logger.info("Initializing cardinal network...")
        
        # Load the full network
        try:
            self.network_graph = load_network()
            logger.info(f"Loaded network with {len(self.network_graph.nodes)} nodes and {len(self.network_graph.edges)} edges")
        except Exception as e:
            logger.error(f"Failed to load network: {e}")
            # Create empty network as fallback
            self.network_graph = nx.Graph()
            return
        
        # Create mapping between agent_ids (list indices) and cardinal_ids
        self.cardinal_id_to_agent_id_map = {}
        self.agent_id_to_cardinal_id_map = {}
        
        for agent in agents:
            if hasattr(agent, 'cardinal_id') and agent.cardinal_id is not None:
                self.cardinal_id_to_agent_id_map[agent.cardinal_id] = agent.agent_id
                self.agent_id_to_cardinal_id_map[agent.agent_id] = agent.cardinal_id
        
        logger.info(f"Created mappings for {len(self.cardinal_id_to_agent_id_map)} cardinals")
        
        # Filter network to only include cardinals that are actually in the simulation
        available_cardinal_ids = set(self.cardinal_id_to_agent_id_map.keys())
        network_cardinal_ids = set(self.network_graph.nodes())
        
        # Find intersection
        active_cardinal_ids = available_cardinal_ids.intersection(network_cardinal_ids)
        logger.info(f"Found {len(active_cardinal_ids)} cardinals present in both simulation and network")
        
        if len(active_cardinal_ids) < len(available_cardinal_ids):
            missing_from_network = available_cardinal_ids - network_cardinal_ids
            logger.warning(f"Cardinals in simulation but not in network: {missing_from_network}")
        
        # Create subgraph with only the active cardinals
        if active_cardinal_ids:
            self.network_graph = self.network_graph.subgraph(active_cardinal_ids).copy()
            logger.info(f"Created active network subgraph with {len(self.network_graph.nodes)} nodes and {len(self.network_graph.edges)} edges")
        else:
            logger.warning("No cardinals found in both simulation and network - grouping will use fallback methods")
            self.network_graph = nx.Graph()
    
    def generate_groups(self, grouping_config, agents: List) -> List[List[int]]:
        """
        Generate discussion groups based on the configuration.
        
        Args:
            grouping_config: Pydantic GroupingConfig object 
            agents: List of Agent objects
            
        Returns:
            List of groups, where each group is a list of agent_ids
        """
        if not grouping_config.enabled:
            logger.info("Network grouping disabled, using random grouping")
            groups = self._generate_random_groups(agents, grouping_config.group_size)
        else:
            method = grouping_config.method
            
            if method == 'network_utility':
                groups = self._generate_network_utility_groups(grouping_config, agents)
            elif method == 'random':
                groups = self._generate_random_groups(agents, grouping_config.group_size)
            elif method == 'simple':
                groups = self._generate_simple_groups(agents, grouping_config.group_size)
            else:
                logger.warning(f"Unknown grouping method '{method}', falling back to random")
                groups = self._generate_random_groups(agents, grouping_config.group_size)
        
        # Store the generated groups
        self.current_groups = groups
        return groups
    
    def _generate_network_utility_groups(self, grouping_config, agents: List) -> List[List[int]]:
        """Generate groups using the network utility-based algorithm."""
        if not self.network_graph or len(self.network_graph.nodes) == 0:
            logger.warning("No network available, falling back to random grouping")
            return self._generate_random_groups(agents, grouping_config.group_size)
        
        # Extract configuration from Pydantic model
        group_size = grouping_config.group_size
        utility_weights = grouping_config.utility_weights
        beta = (
            utility_weights.connection,
            utility_weights.ideology,
            utility_weights.influence,
            utility_weights.interaction
        )
        stochasticity = grouping_config.stochasticity
        leader_stochasticity = grouping_config.leader_stochasticity
        seed = grouping_config.seed
        
        logger.info(f"Generating network utility groups with:")
        logger.info(f"  Group size: {group_size}")
        logger.info(f"  Utility weights: {beta}")
        logger.info(f"  Stochasticity: {stochasticity}")
        logger.info(f"  Leader stochasticity: {leader_stochasticity}")
        logger.info(f"  Seed: {seed}")
        
        # Get list of cardinal_ids that are in both the simulation and the network
        cardinal_list = [self.agent_id_to_cardinal_id_map[agent.agent_id] 
                        for agent in agents 
                        if agent.agent_id in self.agent_id_to_cardinal_id_map 
                        and self.agent_id_to_cardinal_id_map[agent.agent_id] in self.network_graph.nodes]
        
        if len(cardinal_list) == 0:
            logger.warning("No agents found in network, falling back to random grouping")
            return self._generate_random_groups(agents, group_size)
        
        logger.info(f"Using {len(cardinal_list)} cardinals for network-based grouping")
        
        try:
            # Generate groups using cardinal_ids
            cardinal_groups = generate_non_overlapping_utility_groups(
                network_graph=self.network_graph,
                cardinal_list=cardinal_list,
                group_size=group_size,
                beta=beta,
                seed=seed,
                stochasticity=stochasticity,
                leader_stochasticity=leader_stochasticity
            )
            
            # Convert cardinal_id groups to agent_id groups
            agent_groups = []
            for cardinal_group in cardinal_groups:
                agent_group = []
                for cardinal_id in cardinal_group:
                    if cardinal_id in self.cardinal_id_to_agent_id_map:
                        agent_group.append(self.cardinal_id_to_agent_id_map[cardinal_id])
                if agent_group:  # Only add non-empty groups
                    agent_groups.append(agent_group)
            
            # Handle any agents not included in the network
            all_agent_ids = {agent.agent_id for agent in agents}
            grouped_agent_ids = {aid for group in agent_groups for aid in group}
            ungrouped_agent_ids = list(all_agent_ids - grouped_agent_ids)
            
            if ungrouped_agent_ids:
                logger.info(f"Adding {len(ungrouped_agent_ids)} agents not in network to groups")
                # Distribute ungrouped agents to existing groups
                for agent_id in ungrouped_agent_ids:
                    # Find the smallest group that has room (prefer groups under target size)
                    target_group = None
                    for group in agent_groups:
                        if len(group) < group_size:
                            if target_group is None or len(group) < len(target_group):
                                target_group = group
                    
                    if target_group is not None:
                        target_group.append(agent_id)
                    else:
                        # All groups are full, create a new group
                        agent_groups.append([agent_id])
            
            # Post-process to ensure minimum group size (at least 3 members)
            min_group_size = 3
            agent_groups = self._ensure_minimum_group_sizes(agent_groups, min_group_size, group_size)
            
            self.current_groups = agent_groups
            logger.info(f"Generated {len(agent_groups)} groups using network utility method")
            
            # Log group information
            for i, group in enumerate(agent_groups):
                leader_id = group[0] if group else None
                leader_cardinal_id = self.agent_id_to_cardinal_id_map.get(leader_id, 'Unknown')
                logger.info(f"  Group {i+1}: {len(group)} members, leader: agent_{leader_id} (Cardinal_ID {leader_cardinal_id})")
            
            return agent_groups
            
        except Exception as e:
            logger.error(f"Error in network utility grouping: {e}")
            logger.warning("Falling back to random grouping")
            return self._generate_random_groups(agents, group_size)
    
    def _ensure_minimum_group_sizes(self, groups: List[List[int]], min_size: int, target_size: int) -> List[List[int]]:
        """
        Ensure all groups meet minimum size requirements by redistributing members.
        
        Args:
            groups: List of groups (each group is a list of agent_ids)
            min_size: Minimum acceptable group size
            target_size: Target group size for redistribution
            
        Returns:
            List of groups with all groups meeting minimum size requirements
        """
        if not groups:
            return groups
            
        # Find groups that are too small
        small_groups = [i for i, group in enumerate(groups) if len(group) < min_size]
        
        if not small_groups:
            return groups  # All groups are already acceptable size
            
        logger.info(f"Redistributing {len(small_groups)} groups smaller than {min_size} members")
        
        # Collect all members from small groups
        members_to_redistribute = []
        for group_idx in sorted(small_groups, reverse=True):  # Remove from end to avoid index issues
            members_to_redistribute.extend(groups[group_idx])
            del groups[group_idx]
        
        # Redistribute members to existing groups
        for member in members_to_redistribute:
            # Find the smallest group that still has room
            best_group = None
            for group in groups:
                if len(group) < target_size:
                    if best_group is None or len(group) < len(best_group):
                        best_group = group
            
            if best_group is not None:
                best_group.append(member)
            else:
                # All existing groups are full, need to create new groups
                # Collect remaining members and form new groups
                remaining_members = [member] + members_to_redistribute[members_to_redistribute.index(member) + 1:]
                
                # Form new groups from remaining members
                while remaining_members:
                    new_group_size = min(target_size, len(remaining_members))
                    if len(remaining_members) == target_size + 1:
                        # Split more evenly: don't leave a single member
                        new_group_size = target_size // 2 + target_size % 2
                        if new_group_size < min_size:
                            new_group_size = target_size
                    
                    new_group = remaining_members[:new_group_size]
                    groups.append(new_group)
                    remaining_members = remaining_members[new_group_size:]
                break
        
        # Final check: if we still have groups that are too small, merge them
        small_groups = [i for i, group in enumerate(groups) if len(group) < min_size]
        if small_groups:
            logger.warning(f"Still have {len(small_groups)} small groups after redistribution, merging them")
            
            # Merge all small groups together
            merged_members = []
            for group_idx in sorted(small_groups, reverse=True):
                merged_members.extend(groups[group_idx])
                del groups[group_idx]
            
            # Create new groups from merged members
            while merged_members:
                group_size = min(target_size, len(merged_members))
                groups.append(merged_members[:group_size])
                merged_members = merged_members[group_size:]
        
        logger.info(f"Final group distribution: {[len(group) for group in groups]}")
        return groups

    def _generate_random_groups(self, agents: List, group_size: int) -> List[List[int]]:
        """Generate random groups."""
        logger.info(f"Generating random groups of size {group_size}")
        
        agent_ids = [agent.agent_id for agent in agents]
        random.shuffle(agent_ids)
        
        groups = []
        for i in range(0, len(agent_ids), group_size):
            group = agent_ids[i:i+group_size]
            groups.append(group)
        
        logger.info(f"Generated {len(groups)} random groups")
        return groups
    
    def _generate_simple_groups(self, agents: List, group_size: int) -> List[List[int]]:
        """Generate simple sequential groups."""
        logger.info(f"Generating simple sequential groups of size {group_size}")
        
        agent_ids = [agent.agent_id for agent in agents]
        groups = []
        for i in range(0, len(agent_ids), group_size):
            group = agent_ids[i:i+group_size]
            groups.append(group)
        
        logger.info(f"Generated {len(groups)} simple groups")
        return groups
    
    def get_group_for_agent(self, agent_id: int) -> Optional[List[int]]:
        """Get the group that contains the specified agent."""
        for group in self.current_groups:
            if agent_id in group:
                return group
        return None
    
    def get_group_summary(self) -> Dict:
        """Get a summary of the current grouping."""
        if not self.current_groups:
            return {"total_groups": 0, "total_agents": 0}
        
        total_agents = sum(len(group) for group in self.current_groups)
        group_sizes = [len(group) for group in self.current_groups]
        
        return {
            "total_groups": len(self.current_groups),
            "total_agents": total_agents,
            "group_sizes": group_sizes,
            "avg_group_size": sum(group_sizes) / len(group_sizes) if group_sizes else 0,
            "min_group_size": min(group_sizes) if group_sizes else 0,
            "max_group_size": max(group_sizes) if group_sizes else 0
        }
