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
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')

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
        self.max_iter = self.viz_config.tsne.n_iter # Renamed from n_iter
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
    
    def _get_final_stance_embeddings(self, env):
        """
        Helper to get final stance embeddings for all agents.
        Returns embeddings, agent names, and agent IDs.
        """
        # Import here to avoid circular imports
        from conclave.embeddings import get_default_client
        
        embedding_client = get_default_client()

        final_stances = []
        agent_names = []
        agent_ids = []
        for agent in env.agents:
            if hasattr(agent, 'stance_history') and agent.stance_history:
                # Get the last stance
                last_stance_entry = agent.stance_history[-1]
                if isinstance(last_stance_entry, dict):
                    stance_text = last_stance_entry.get("stance", "")
                else:
                    stance_text = last_stance_entry
                
                if stance_text: # Ensure we don't send empty strings for embedding
                    final_stances.append(stance_text)
                    agent_names.append(agent.name)
                    agent_ids.append(agent.agent_id)

        if not final_stances:
            logger.warning("No final stances with content found for embeddings.")
            return None, None, None

        embeddings = embedding_client.get_embeddings(final_stances)
        return np.array(embeddings), agent_names, agent_ids

    def plot_final_votes(self, env, vote_counts: Dict[int, int]):
        """
        Plots the final votes for each candidate.

        Args:
            env: The simulation environment.
            vote_counts: A dictionary mapping candidate ID to vote count.
        """
        logger.info("Generating final vote visualization...")
        if not vote_counts:
            logger.warning("No vote counts provided. Skipping final vote visualization.")
            return

        agent_names = [agent.name for agent in env.agents]
        
        try:
            candidate_ids = [int(id) for id in vote_counts.keys()]
            candidate_names = [agent_names[id] for id in candidate_ids]
            votes = list(vote_counts.values())
        except (ValueError, IndexError) as e:
            logger.error(f"Could not resolve candidate names from vote_counts: {e}")
            candidate_names = [f"Agent {id}" for id in vote_counts.keys()]
            votes = list(vote_counts.values())

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(x=candidate_names, y=votes, ax=ax, hue=candidate_names, palette="viridis", legend=False)
        ax.set_title("Final Votes per Candidate", fontsize=16, fontweight='bold')
        ax.set_xlabel("Candidate", fontsize=12)
        ax.set_ylabel("Number of Votes", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        save_path = self.viz_dir / "final_votes.png"
        fig.savefig(save_path, dpi=self.dpi)
        logger.info(f"Saved final vote visualization to: {save_path}")
        plt.close(fig)

    def plot_agent_positions(self, embeddings: np.ndarray, agent_names: List[str], method: str, colors: List, legend_elements: List):
        """
        Plots agent positions using a given dimensionality reduction method,
        with points colored by their final vote.

        Args:
            embeddings: The embeddings of the agent stances.
            agent_names: The names of the agents.
            method: The dimensionality reduction method ('pca', 'tsne', or 'umap').
            colors: A list of colors for each agent point.
            legend_elements: A list of elements for creating the plot legend.
        """
        if embeddings is None or len(embeddings) < 2:
            logger.warning(f"Not enough embeddings for {method.upper()} plot (found {len(embeddings) if embeddings is not None else 0}). Skipping.")
            return

        logger.info(f"Generating agent position visualization using {method.upper()} colored by vote...")

        random_state = 42 # Default random state
        if method == 'pca':
            random_state = self.viz_config.pca.random_state
            reducer = PCA(n_components=2, random_state=random_state)
        elif method == 'tsne':
            perplexity = min(self.viz_config.tsne.perplexity, len(embeddings) - 1)
            if perplexity <= 0:
                logger.warning(f"Cannot run t-SNE with perplexity {perplexity} (needs to be > 0). Skipping.")
                return
            random_state = self.viz_config.tsne.random_state
            reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, max_iter=self.viz_config.tsne.n_iter)
        elif method == 'umap':
            try:
                from umap import UMAP
                random_state = self.viz_config.umap.random_state
                reducer = UMAP(n_components=2, 
                               n_neighbors=min(self.n_neighbors, len(embeddings) - 1), 
                               min_dist=self.min_dist, 
                               metric=self.metric,
                               random_state=random_state)
            except ImportError:
                logger.warning("UMAP is not installed. Skipping UMAP visualization. Run `pip install umap-learn`.")
                return
            except ValueError as e:
                logger.error(f"Error initializing UMAP: {e}. Skipping visualization.")
                return
        else:
            logger.error(f"Unknown reduction method: {method}")
            return

        embeddings_2d = reducer.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=self.figure_size_inches)
        
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker='o', s=self.marker_size, alpha=0.8, c=colors)

        if self.show_names:
            for i, name in enumerate(agent_names):
                ax.text(embeddings_2d[i, 0], embeddings_2d[i, 1], f' {name}', fontsize=self.font_size)

        ax.legend(handles=legend_elements, title="Voted For", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title(f"Agent Stance Positions ({method.upper()}) - Colored by Final Vote", fontsize=16, fontweight='bold')
        ax.set_xlabel("Dimension 1", fontsize=12)
        ax.set_ylabel("Dimension 2", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

        save_path = self.viz_dir / f"agent_positions_{method}_by_vote.png"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved {method.upper()} visualization to: {save_path}")
        plt.close(fig)

    def generate_all_visualizations(self, env, final_vote_counts, final_individual_votes: Optional[Dict[int, int]] = None):
        """
        Generates and saves all requested visualizations.
        """
        logger.info("Starting generation of all visualizations...")
        
        # 1. Final votes plot
        self.plot_final_votes(env, final_vote_counts)

        # Get final stance embeddings for position plots
        final_embeddings, agent_names, agent_ids = self._get_final_stance_embeddings(env)

        if final_embeddings is not None and agent_names is not None and final_individual_votes:
            logger.info("Generating agent position plots colored by final vote.")
            # Create color mapping based on votes
            voted_for_candidates = sorted(list(set(final_individual_votes.values())))
            
            palette = sns.color_palette(self.color_map, len(voted_for_candidates) if voted_for_candidates else 1)
            color_map = {candidate_id: palette[i] for i, candidate_id in enumerate(voted_for_candidates)}
            
            agent_id_to_name = {agent.agent_id: agent.name for agent in env.agents}
            default_color = (0.5, 0.5, 0.5, 0.5) # Grey for non-voters

            plot_colors = [color_map.get(final_individual_votes.get(agent_id), default_color) for agent_id in agent_ids]

            # Create legend elements
            from matplotlib.lines import Line2D
            legend_elements = []
            for candidate_id, color in color_map.items():
                candidate_name = agent_id_to_name.get(candidate_id, f"Agent {candidate_id}")
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label=candidate_name,
                                              markerfacecolor=color, markersize=10))
            
            if default_color in plot_colors:
                 legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Did not vote',
                                              markerfacecolor=default_color, markersize=10))

            # Generate plots
            self.plot_agent_positions(final_embeddings, agent_names, 'pca', plot_colors, legend_elements)
            self.plot_agent_positions(final_embeddings, agent_names, 'tsne', plot_colors, legend_elements)
            self.plot_agent_positions(final_embeddings, agent_names, 'umap', plot_colors, legend_elements)
        else:
            logger.warning("Skipping agent position plots colored by vote due to lack of embeddings or vote data.")

        # 5. Stance progression plot
        logger.info("Generating stance progression visualization...")
        self.generate_stance_visualization_from_env(env)
            
        logger.info("All visualizations generated.")

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
            round_data = {}
            
            # Check if agents already have embedding history stored
            agents_with_embeddings = [agent for agent in env.agents if hasattr(agent, 'embedding_history') and agent.embedding_history]
            
            if agents_with_embeddings:
                logger.info("Using pre-computed embeddings from agent embedding history")
                
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
                        round_data[round_key] = {
                            "embeddings": np.array(embeddings_for_round),
                            "names": names_for_round,
                        }
                
            else:
                logger.info("No pre-computed embeddings found, generating new ones")
                
                cardinal_names = list(stance_evolution.keys())
                rounds = set()
                for cardinal_data in stance_evolution.values():
                    for entry in cardinal_data:
                        rounds.add(entry["round"])
                
                rounds = sorted(rounds)
                logger.info(f"Found stance data for rounds: {rounds}")
                
                for round_num in rounds:
                    stance_texts = []
                    names_for_round = []
                    
                    for cardinal_name in cardinal_names:
                        for entry in stance_evolution[cardinal_name]:
                            if entry["round"] == round_num and entry["stance"]:
                                stance_texts.append(entry["stance"])
                                names_for_round.append(cardinal_name)
                                break
                    
                    if stance_texts:
                        embeddings = embedding_client.get_embeddings(stance_texts)
                        round_data[round_num] = {
                            "embeddings": embeddings,
                            "names": names_for_round,
                        }
                        logger.info(f"Generated embeddings for {len(stance_texts)} stances in round {round_num}")
            
            if not round_data:
                logger.warning("No embeddings generated. Skipping visualization.")
                return
            
            self.visualize_stance_progression(round_data, stance_evolution)
            
            logger.info(f"Stance visualization complete! Generated progression visualizations for rounds: {list(round_data.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to generate stance visualization: {e}", exc_info=True)
            print(f"⚠️ Stance visualization failed: {e}")
    
    def visualize_stance_progression(self, round_data: Dict[int, Dict], stance_evolution: Dict[str, List[Dict]]):
        """
        Create visualizations showing stance progression for each reduction method.
        Saves to the viz_dir provided during initialization.
        """
        logger.info(f"Creating stance progression visualizations in {self.viz_dir}...")
        
        os.makedirs(self.viz_dir, exist_ok=True)
        
        colors = sns.color_palette(self.group_palette, n_colors=50)
        
        all_embeddings = []
        all_round_labels = []
        all_cardinal_labels = []
        
        for round_key in sorted(round_data.keys()):
            data = round_data[round_key]
            embeddings = data['embeddings']
            names = data['names']
            for i, embedding in enumerate(embeddings):
                all_embeddings.append(embedding)
                all_round_labels.append(round_key)
                all_cardinal_labels.append(names[i])
        
        if not all_embeddings:
            logger.warning("No embeddings available for stance progression visualization.")
            return

        all_embeddings = np.array(all_embeddings)
        
        if len(all_embeddings) < 2:
            logger.warning(f"Not enough data points for dimensionality reduction (found {len(all_embeddings)}). Skipping progression plots.")
            return

        reduction_methods = ['pca', 'tsne', 'umap']
        for method in reduction_methods:
            logger.info(f"Generating stance progression plot using {method.upper()}...")
            
            embeddings_2d = None
            
            try:
                if method == 'pca':
                    reducer = PCA(n_components=2, random_state=self.viz_config.pca.random_state)
                    embeddings_2d = reducer.fit_transform(all_embeddings)
                
                elif method == 'tsne':
                    embeddings_for_tsne = all_embeddings
                    if all_embeddings.shape[1] > 50:
                        # n_components for PCA must be <= min(n_samples, n_features)
                        max_pca_components = min(all_embeddings.shape)
                        n_components = min(50, max_pca_components)
                        
                        if n_components > 1:
                            pca = PCA(n_components=n_components, random_state=self.viz_config.pca.random_state)
                            embeddings_for_tsne = pca.fit_transform(all_embeddings)
                            logger.info(f"PCA pre-reduction for t-SNE to {n_components} dimensions.")
                        else:
                            logger.info("Skipping PCA pre-reduction for t-SNE due to insufficient dimensions.")

                    perplexity = min(self.viz_config.tsne.perplexity, len(embeddings_for_tsne) - 1)
                    if perplexity <= 0:
                        logger.warning(f"Cannot run t-SNE with perplexity {perplexity}. Skipping t-SNE plot.")
                        continue
                    
                    tsne = TSNE(n_components=2, 
                                random_state=self.viz_config.tsne.random_state, 
                                perplexity=perplexity, 
                                n_iter=self.max_iter)
                    embeddings_2d = tsne.fit_transform(embeddings_for_tsne)

                elif method == 'umap':
                    try:
                        from umap import UMAP
                        n_neighbors = min(self.n_neighbors, len(all_embeddings) - 1)
                        if n_neighbors < 2:
                            logger.warning(f"Not enough neighbors for UMAP ({n_neighbors}). Skipping UMAP plot.")
                            continue

                        reducer = UMAP(n_components=2, 
                                       n_neighbors=n_neighbors, 
                                       min_dist=self.min_dist, 
                                       metric=self.metric,
                                       random_state=self.viz_config.umap.random_state)
                        embeddings_2d = reducer.fit_transform(all_embeddings)
                    except ImportError:
                        logger.warning("UMAP not installed. Skipping UMAP visualization.")
                        continue
            except Exception as e:
                logger.error(f"Error during {method.upper()} dimensionality reduction: {e}", exc_info=True)
                continue

            if embeddings_2d is None:
                logger.warning(f"Dimensionality reduction failed for {method.upper()}. Skipping plot.")
                continue

            fig, ax = plt.subplots(figsize=self.figure_size_inches)
            
            unique_cardinals = sorted(list(set(all_cardinal_labels)))
            unique_round_labels = sorted(list(set(all_round_labels)))
            round_to_numeric_map = {label: i for i, label in enumerate(unique_round_labels)}
            
            for i, cardinal in enumerate(unique_cardinals):
                cardinal_color = colors[i % len(colors)]
                
                cardinal_indices = [j for j, name in enumerate(all_cardinal_labels) if name == cardinal]
                cardinal_rounds = [all_round_labels[j] for j in cardinal_indices]
                cardinal_positions = embeddings_2d[cardinal_indices]
                
                if len(cardinal_positions) > 1:
                    sorted_indices = np.argsort([round_to_numeric_map[r] for r in cardinal_rounds])
                    sorted_positions = cardinal_positions[sorted_indices]
                    ax.plot(sorted_positions[:, 0], sorted_positions[:, 1], 
                           color=cardinal_color, linewidth=self.path_linewidth, alpha=self.path_alpha, 
                           label=f"{cardinal}")
                
                for pos, round_label in zip(cardinal_positions, cardinal_rounds):
                    numeric_round = round_to_numeric_map[round_label]
                    size = self.marker_size + (numeric_round * 20)
                    ax.scatter(pos[0], pos[1], 
                              color=cardinal_color, 
                              s=size,
                              alpha=0.8,
                              edgecolors='black',
                              linewidth=1,
                              zorder=5)
                    
                    if self.show_names:
                        ax.annotate(f"{round_label}", 
                                   (pos[0], pos[1]), 
                                   xytext=(5, 5), 
                                   textcoords='offset points',
                                   fontsize=self.font_size,
                                   fontweight='bold')
            
            ax.set_title(f"Cardinal Stance Progression ({method.upper()})", fontsize=16, fontweight='bold')
            ax.set_xlabel("Dimension 1", fontsize=12)
            ax.set_ylabel("Dimension 2", fontsize=12)
            ax.legend(title="Cardinals", bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.6)
            
            explanation = ("Lines connect stances from the same cardinal across rounds.\n"
                          "Larger points indicate later rounds.")
            ax.text(0.02, 0.98, explanation, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            viz_path = self.viz_dir / f"stance_progression_{method}.png"
            fig.savefig(viz_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved stance progression visualization to: {viz_path}")
            plt.close(fig)
