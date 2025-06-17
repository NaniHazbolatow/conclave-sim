#!/usr/bin/env python3
"""
Cardinal visualization module for the conclave simulation.

This module provides visualization tools for displaying cardinal stance progression
across discussion rounds using embedding-based position tracking.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Optional
import logging
import os
import json
import pandas as pd
from pathlib import Path
from conclave.config import get_config_manager

logger = logging.getLogger("conclave.visualization")

class CardinalVisualizer:
    """
    Visualizes cardinal positions and ideological clusters in 2D space.
    """
    
    def __init__(self, viz_dir: str):
        """
        Initialize the visualizer with configuration settings.
        
        Args:
            viz_dir: Directory to save visualizations
        """
        self.config_manager = get_config_manager()
        self.viz_config = self.config_manager.config.output.visualization # Changed from visualization_settings
        self.viz_dir = Path(viz_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Access pydantic model attributes directly
        self.reduction_method = self.viz_config.reduction_method
        # TSNE-specific parameters
        self.perplexity = self.viz_config.tsne.perplexity 
        self.n_iter = self.viz_config.tsne.n_iter
        # UMAP-specific parameters
        self.n_neighbors = self.viz_config.umap.n_neighbors
        self.min_dist = self.viz_config.umap.min_dist
        self.metric = self.viz_config.umap.metric
        
        # General plot settings from self.viz_config.plot
        self.target_type = self.viz_config.plot.target_type
        self.show_names = self.viz_config.plot.show_names
        self.show_initial_stances = self.viz_config.plot.show_initial_stances
        self.show_final_stances = self.viz_config.plot.show_final_stances
        self.show_paths = self.viz_config.plot.show_paths
        self.show_group_colors = self.viz_config.plot.show_group_colors
        self.font_size = self.viz_config.plot.font_size
        self.figure_size_inches = tuple(self.viz_config.plot.figure_size)
        self.color_map = self.viz_config.plot.colormap
        self.path_alpha = self.viz_config.plot.path_alpha
        self.path_linewidth = self.viz_config.plot.path_linewidth
        self.marker_size = self.viz_config.plot.point_size
        self.initial_stance_marker = self.viz_config.plot.initial_stance_marker
        self.final_stance_marker = self.viz_config.plot.final_stance_marker
        self.intermediate_stance_marker = self.viz_config.plot.intermediate_stance_marker
        self.candidate_marker = self.viz_config.plot.candidate_marker
        self.non_candidate_marker = self.viz_config.plot.non_candidate_marker
        self.candidate_color = self.viz_config.plot.candidate_color
        self.non_candidate_color = self.viz_config.plot.non_candidate_color
        self.group_palette = self.viz_config.plot.group_palette
        self.stance_history_limit = self.viz_config.plot.stance_history_limit

        # Output settings from self.viz_config
        self.output_format = self.viz_config.output_format
        self.dpi = self.viz_config.dpi

        # Note: self.random_state was removed. It should be accessed within _reduce_dimensionality
        # like: self.viz_config.pca.random_state, self.viz_config.tsne.random_state, etc.
        # based on self.reduction_method.

        logger.info(
            f"CardinalVisualizer initialized. Output directory: {self.viz_dir}"
        )
    
    def generate_stance_visualization_from_env(self, env):
        """
        Generate and save stance progression visualization directly from environment.
        Uses the viz_dir provided during initialization.
        
        Args:
            env: The simulation environment with agents
        """
        logger.info(f"Generating stance progression visualization from environment into {self.viz_dir}...")
        
        try:
            # Import here to avoid circular imports
            from conclave.embeddings import get_default_client
            
            # Collect all agent stances and their history
            stance_evolution = {}
            
            for agent in env.agents:
                if hasattr(agent, 'stance_history') and agent.stance_history:
                    stance_evolution[agent.name] = []
                    for i, stance_entry in enumerate(agent.stance_history):
                        # Handle both string stances and dictionary stance entries
                        if isinstance(stance_entry, dict):
                            stance_text = stance_entry.get("stance", "")
                        else:
                            stance_text = stance_entry
                        
                        stance_evolution[agent.name].append({
                            "round": i,
                            "stance": stance_text,
                            "word_count": len(stance_text.split()) if stance_text else 0
                        })
            
            if not stance_evolution:
                logger.warning("No stance evolution data found. Skipping visualization.")
                return
            
            # Generate embeddings for stance evolution
            embedding_client = get_default_client()
            cardinal_names = list(stance_evolution.keys())
            round_embeddings = {}
            
            # Check if agents already have embedding history stored
            agents_with_embeddings = [agent for agent in env.agents if hasattr(agent, 'embedding_history') and agent.embedding_history]
            
            if agents_with_embeddings:
                logger.info("Using pre-computed embeddings from agent embedding history")
                
                # Use stored embeddings if available
                all_rounds = set()
                for agent in agents_with_embeddings:
                    all_rounds.update(agent.embedding_history.keys())
                
                for round_key in sorted(all_rounds):
                    embeddings_for_round = []
                    names_for_round = []
                    
                    for agent in env.agents:
                        if hasattr(agent, 'embedding_history') and round_key in agent.embedding_history:
                            embeddings_for_round.append(agent.embedding_history[round_key])
                            names_for_round.append(agent.name)
                    
                    if embeddings_for_round:
                        round_embeddings[round_key] = np.array(embeddings_for_round)
                        if cardinal_names != names_for_round:
                            cardinal_names = names_for_round  # Update to match available agents
                
            else:
                logger.info("No pre-computed embeddings found, generating new ones")
                
                # Fallback to generating embeddings from stance text
                # Determine which rounds we have data for
                rounds = set()
                for cardinal_data in stance_evolution.values():
                    for entry in cardinal_data:
                        rounds.add(entry["round"])
                
                rounds = sorted(rounds)
                logger.info(f"Found stance data for rounds: {rounds}")
                
                # Generate embeddings for each round
                for round_num in rounds:
                    stance_texts = []
                    
                    for cardinal_name in cardinal_names:
                        # Find stance for this round
                        for entry in stance_evolution[cardinal_name]:
                            if entry["round"] == round_num:
                                stance_texts.append(entry["stance"])
                                break
                    
                    if stance_texts:
                        embeddings = embedding_client.get_embeddings(stance_texts)
                        round_embeddings[round_num] = embeddings
                        logger.info(f"Round {round_num}: Generated embeddings for {len(stance_texts)} stances")
            
            if not round_embeddings:
                logger.warning("No embeddings generated. Skipping visualization.")
                return
            
            # Generate only the main stance progression visualization
            self.visualize_stance_progression(round_embeddings, cardinal_names, stance_evolution)
            
            logger.info(f"Stance visualization complete! Generated progression visualization for rounds: {list(round_embeddings.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to generate stance visualization: {e}")
            print(f"⚠️ Stance visualization failed: {e}")
    
    def visualize_stance_progression(self, round_embeddings: Dict[int, np.ndarray], cardinal_names: List[str], 
                                   stance_evolution: Dict[str, List[Dict]]):
        """
        Create visualizations showing stance progression across rounds with connected dots.
        Saves to the viz_dir provided during initialization.
        """
        logger.info(f"Creating stance progression visualizations in {self.viz_dir}...")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Set up the plot style
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        # Prepare data for visualization
        all_embeddings = []
        all_round_labels = []
        all_cardinal_labels = []
        
        # Collect all embeddings across rounds
        for round_num in sorted(round_embeddings.keys()):
            embeddings = round_embeddings[round_num]
            for i, embedding in enumerate(embeddings):
                all_embeddings.append(embedding)
                all_round_labels.append(round_num)
                all_cardinal_labels.append(cardinal_names[i] if i < len(cardinal_names) else f"Cardinal_{i}")
        
        all_embeddings = np.array(all_embeddings)
        
        # Reduce dimensionality for visualization
        logger.info("Reducing dimensionality for visualization...")
        
        # Use PCA first to reduce dimensions if needed, then t-SNE for final visualization
        max_pca_components = min(50, all_embeddings.shape[0] - 1, all_embeddings.shape[1] - 1)
        if all_embeddings.shape[1] > max_pca_components and max_pca_components > 2:
            pca = PCA(n_components=max_pca_components)
            embeddings_pca = pca.fit_transform(all_embeddings)
            logger.info(f"PCA reduced dimensions from {all_embeddings.shape[1]} to {max_pca_components}")
        else:
            embeddings_pca = all_embeddings
            logger.info("Skipping PCA - using original embeddings")
        
        # Apply t-SNE for 2D visualization with appropriate perplexity
        perplexity = min(5, max(1, len(all_embeddings) - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings_pca)
        
        # Create the progression visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot points for each round and cardinal
        unique_cardinals = list(set(all_cardinal_labels))
        
        for i, cardinal in enumerate(unique_cardinals):
            cardinal_color = colors[i % len(colors)]
            
            # Get indices for this cardinal across all rounds
            cardinal_indices = [j for j, name in enumerate(all_cardinal_labels) if name == cardinal]
            cardinal_rounds = [all_round_labels[j] for j in cardinal_indices]
            cardinal_positions = embeddings_2d[cardinal_indices]
            
            # Plot the progression line
            if len(cardinal_positions) > 1:
                ax.plot(cardinal_positions[:, 0], cardinal_positions[:, 1], 
                       color=cardinal_color, linewidth=2, alpha=0.7, 
                       label=f"{cardinal} progression")
            
            # Plot points for each round
            for j, (pos, round_num) in enumerate(zip(cardinal_positions, cardinal_rounds)):
                # Size increases with round number
                size = 100 + (round_num * 50)
                ax.scatter(pos[0], pos[1], 
                          color=cardinal_color, 
                          s=size,
                          alpha=0.8,
                          edgecolors='black',
                          linewidth=1,
                          zorder=5)
                
                # Add round number as text
                ax.annotate(f"R{round_num}", 
                           (pos[0], pos[1]), 
                           xytext=(5, 5), 
                           textcoords='offset points',
                           fontsize=8,
                           fontweight='bold')
        
        # Customize the plot
        ax.set_title("Cardinal Stance Progression Across Rounds", fontsize=16, fontweight='bold')
        ax.set_xlabel("Stance Dimension 1", fontsize=12)
        ax.set_ylabel("Stance Dimension 2", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add explanation text
        explanation = ("Points show cardinal positions in stance space.\n"
                      "Lines connect the same cardinal across rounds.\n"
                      "Larger points = later rounds.")
        ax.text(0.02, 0.98, explanation, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the main progression plot
        main_viz_path = os.path.join(self.viz_dir, "stance_progression.png")
        fig.savefig(main_viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved stance progression visualization to: {main_viz_path}")
        
        plt.close('all')
        
        return round_embeddings
