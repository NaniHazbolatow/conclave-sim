#!/usr/bin/env python3
"""
Enhanced voting analysis script for Conclave simulations.
Creates visualizations showing the relationship between agent stances and voting behavior.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
from typing import Dict, List, Optional, Tuple

# Try to import UMAP, fall back gracefully if not available
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

def load_simulation_data(results_dir: Path) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """Load all simulation data from a results directory."""
    print(f"Loading simulation data from: {results_dir}")
    
    # Load summary
    summary_file = results_dir / "simulation_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Simulation summary not found: {summary_file}")
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Load voting data
    voting_file = results_dir / "voting_data.csv"
    if not voting_file.exists():
        raise FileNotFoundError(f"Voting data not found: {voting_file}")
    
    voting_df = pd.read_csv(voting_file)
    
    # Load embeddings data
    embeddings_file = results_dir / "stance_embeddings.csv"
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings data not found: {embeddings_file}")
    
    embeddings_df = pd.read_csv(embeddings_file)
    
    print(f"✅ Loaded simulation data:")
    # Handle both old and new summary formats for display
    if 'results' in summary:
        results = summary['results']
        winner_name = results.get('winner_name', 'Unknown')
        total_rounds = results.get('total_election_rounds', 'Unknown')
    else:
        winner_name = summary.get('winner_name', summary.get('winner', 'Unknown'))
        total_rounds = summary.get('total_election_rounds', summary.get('total_rounds', 'Unknown'))
    
    print(f"  - Summary: Winner={winner_name}, Rounds={total_rounds}")
    print(f"  - Voting data: {len(voting_df)} votes")
    print(f"  - Embeddings: {len(embeddings_df)} stance embeddings")
    
    return summary, voting_df, embeddings_df

def prepare_embeddings_matrix(embeddings_df: pd.DataFrame) -> np.ndarray:
    """Convert embeddings from string format to numpy array."""
    print("Preparing embeddings matrix...")
    
    # Extract embedding columns (should be embedding_0, embedding_1, etc.)
    embedding_cols = [col for col in embeddings_df.columns if col.startswith('embedding_')]
    
    if not embedding_cols:
        # If embeddings are stored as strings, parse them
        if 'embedding' in embeddings_df.columns:
            embeddings = []
            for embedding_str in embeddings_df['embedding']:
                # Parse the embedding string (assuming it's a list representation)
                embedding = eval(embedding_str) if isinstance(embedding_str, str) else embedding_str
                embeddings.append(embedding)
            return np.array(embeddings)
        else:
            raise ValueError("No embedding columns found in embeddings data")
    else:
        return embeddings_df[embedding_cols].values

def reduce_dimensions(embeddings: np.ndarray, method: str = 'pca', n_components: int = 2) -> np.ndarray:
    """Reduce embedding dimensions for visualization."""
    print(f"Reducing dimensions using {method.upper()}...")
    
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    elif method.lower() == 'tsne':
        # Adjust perplexity for small datasets (must be less than n_samples)
        perplexity = min(5, max(1, len(embeddings) - 1))
        if perplexity >= len(embeddings):
            perplexity = max(1, len(embeddings) // 2)
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
    elif method.lower() == 'umap':
        if not UMAP_AVAILABLE:
            print("UMAP not available, falling back to PCA")
            reducer = PCA(n_components=n_components, random_state=42)
        else:
            # Adjust n_neighbors for small datasets
            n_neighbors = min(5, max(2, len(embeddings) - 1))
            reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=n_neighbors)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    reduced = reducer.fit_transform(embeddings)
    
    if hasattr(reducer, 'explained_variance_ratio_'):
        print(f"Explained variance ratio: {reducer.explained_variance_ratio_}")
    
    return reduced

def create_voting_correlation_plot(summary: Dict, voting_df: pd.DataFrame, embeddings_df: pd.DataFrame, 
                                 output_dir: Path, reduction_method: str = 'pca'):
    """Create a plot showing agent positions colored by their final vote."""
    print("Creating voting correlation plot...")
    
    # Get final round voting data
    final_round = voting_df['round'].max()
    final_votes = voting_df[voting_df['round'] == final_round].copy()
    
    # Map voting round to embedding round
    # Round 1 voting corresponds to "ER1.D1" embeddings
    embedding_round_map = {
        1: "ER1.D1",
        2: "ER2.D1",
        3: "ER3.D1"
        # Add more mappings as needed
    }
    
    embedding_round = embedding_round_map.get(final_round, f"ER{final_round}.D1")
    
    # Get corresponding embeddings
    final_embeddings = embeddings_df[embeddings_df['round'] == embedding_round].copy()
    
    if len(final_embeddings) == 0:
        # Try alternative round formats
        alternative_rounds = [f"round_{final_round}", f"R{final_round}", str(final_round)]
        for alt_round in alternative_rounds:
            final_embeddings = embeddings_df[embeddings_df['round'] == alt_round].copy()
            if len(final_embeddings) > 0:
                break
    
    print(f"Final voting round: {final_round}, embedding round: {embedding_round}")
    print(f"Votes: {len(final_votes)}, embeddings: {len(final_embeddings)}")
    
    # Merge voting and embedding data
    merged_df = pd.merge(final_votes, final_embeddings, on=['agent_id'], how='inner')
    
    if len(merged_df) == 0:
        print("⚠️  No matching data between votes and embeddings for final round")
        print(f"Vote agent IDs: {final_votes['agent_id'].tolist()}")
        print(f"Embedding agent IDs: {final_embeddings['agent_id'].tolist()}")
        return
    
    # Prepare embeddings matrix
    embeddings_matrix = prepare_embeddings_matrix(merged_df)
    
    # Reduce dimensions
    reduced_embeddings = reduce_dimensions(embeddings_matrix, method=reduction_method)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Get unique candidates and create a color map
    candidates = merged_df['candidate_voted_for'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(candidates)))
    color_map = dict(zip(candidates, colors))
    
    # Plot each agent colored by their vote
    for candidate in candidates:
        mask = merged_df['candidate_voted_for'] == candidate
        if mask.sum() > 0:
            plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], 
                       c=[color_map[candidate]], label=f'Voted for {candidate}', 
                       alpha=0.7, s=60)
    
    plt.xlabel(f'{reduction_method.upper()} Component 1')
    plt.ylabel(f'{reduction_method.upper()} Component 2')
    # Handle both old and new summary formats
    if 'results' in summary:
        # New nested format
        results = summary['results']
        winner_name = results.get('winner_name', 'Unknown')
    else:
        # Old flat format
        winner_name = summary.get('winner_name', summary.get('winner', 'Unknown'))
    
    plt.title(f'Agent Stance Positions Colored by Final Vote\n'
             f'Round {final_round} - Winner: {winner_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plot_file = output_dir / f'voting_correlation_{reduction_method}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved voting correlation plot: {plot_file}")
    plt.close()

def create_evolution_plot(embeddings_df: pd.DataFrame, voting_df: pd.DataFrame, 
                         output_dir: Path, reduction_method: str = 'pca'):
    """Create a plot showing how agent positions evolve over rounds with trajectory lines."""
    print("Creating stance evolution plot...")
    
    # Create a single larger plot
    plt.figure(figsize=(14, 10))
    
    rounds = sorted(embeddings_df['round'].unique())
    
    # Get all agents that appear in multiple rounds
    agent_round_counts = embeddings_df.groupby('agent_id')['round'].nunique()
    multi_round_agents = agent_round_counts[agent_round_counts > 1].index
    
    # Prepare all embeddings for dimensionality reduction (to ensure consistent space)
    all_embeddings_matrix = prepare_embeddings_matrix(embeddings_df)
    all_reduced = reduce_dimensions(all_embeddings_matrix, method=reduction_method)
    
    # Create a mapping from (agent_id, round) to reduced coordinates
    coord_map = {}
    for i, (_, row) in enumerate(embeddings_df.iterrows()):
        coord_map[(row['agent_id'], row['round'])] = all_reduced[i]
    
    # Plot trajectories for agents that appear in multiple rounds
    if len(multi_round_agents) > 0:
        colors = plt.cm.viridis(np.linspace(0, 1, len(multi_round_agents)))
        agent_color_map = dict(zip(multi_round_agents, colors))
        
        for agent_id in multi_round_agents:
            agent_data = embeddings_df[embeddings_df['agent_id'] == agent_id].sort_values('round')
            agent_rounds = agent_data['round'].tolist()
            
            # Get coordinates for this agent across rounds
            x_coords = []
            y_coords = []
            for round_id in agent_rounds:
                if (agent_id, round_id) in coord_map:
                    x, y = coord_map[(agent_id, round_id)]
                    x_coords.append(x)
                    y_coords.append(y)
            
            if len(x_coords) > 1:  # Only draw trajectory if agent appears in multiple rounds
                # Draw trajectory line
                plt.plot(x_coords, y_coords, '-', color=agent_color_map[agent_id], 
                        alpha=0.6, linewidth=2, label=f'Agent {agent_id}')
                # Draw points
                plt.scatter(x_coords, y_coords, c=[agent_color_map[agent_id]], 
                           s=60, alpha=0.8, edgecolors='black', linewidth=1)
                # Mark start and end
                plt.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='^', 
                           alpha=0.8, edgecolors='black', linewidth=1, zorder=5)  # Start
                plt.scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='s', 
                           alpha=0.8, edgecolors='black', linewidth=1, zorder=5)  # End
    
    # Plot single-round agents
    single_round_agents = agent_round_counts[agent_round_counts == 1].index
    single_round_coords = []
    for agent_id in single_round_agents:
        agent_data = embeddings_df[embeddings_df['agent_id'] == agent_id].iloc[0]
        round_id = agent_data['round']
        if (agent_id, round_id) in coord_map:
            x, y = coord_map[(agent_id, round_id)]
            single_round_coords.append((x, y))
    
    if single_round_coords:
        single_coords_array = np.array(single_round_coords)
        plt.scatter(single_coords_array[:, 0], single_coords_array[:, 1], 
                   c='gray', s=40, alpha=0.5, marker='o', label='Single-round agents')
    
    plt.xlabel(f'{reduction_method.upper()} Component 1')
    plt.ylabel(f'{reduction_method.upper()} Component 2')
    plt.title(f'Agent Stance Evolution Trajectories ({reduction_method.upper()})\n'
             f'Green=Start, Red=End, Lines=Agent Trajectories')
    plt.grid(True, alpha=0.3)
    
    # Add legend for trajectory agents (limit to avoid clutter)
    if len(multi_round_agents) <= 10 and len(multi_round_agents) > 0:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    elif len(single_round_coords) > 0:
        plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = output_dir / f'stance_evolution_{reduction_method}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved evolution plot: {plot_file}")
    plt.close()

def create_voting_summary_plot(summary: Dict, voting_df: pd.DataFrame, output_dir: Path):
    """Create summary plots showing voting patterns."""
    print("Creating voting summary plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Votes per round
    ax1 = axes[0, 0]
    round_counts = voting_df.groupby('round').size()
    ax1.bar(round_counts.index, round_counts.values)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Number of Votes')
    ax1.set_title('Votes per Round')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Vote distribution in final round
    ax2 = axes[0, 1]
    final_round = voting_df['round'].max()
    final_votes = voting_df[voting_df['round'] == final_round]
    vote_counts = final_votes['candidate_voted_for'].value_counts()
    
    wedges, texts, autotexts = ax2.pie(vote_counts.values, labels=vote_counts.index, autopct='%1.1f%%')
    ax2.set_title(f'Final Round Vote Distribution (Round {final_round})')
    
    # Plot 3: Candidate popularity over time
    ax3 = axes[1, 0]
    for candidate in voting_df['candidate_voted_for'].unique():
        candidate_votes = voting_df[voting_df['candidate_voted_for'] == candidate]
        round_counts = candidate_votes.groupby('round').size()
        ax3.plot(round_counts.index, round_counts.values, marker='o', label=candidate)
    
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Number of Votes')
    ax3.set_title('Candidate Support Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary text
    # Fix the key names to match actual summary structure
    # Handle both old and new summary formats
    if 'results' in summary:
        # New nested format
        results = summary['results']
        total_rounds = results.get('total_election_rounds', 'Unknown')
        winner = results.get('winner_name', 'Unknown')
    else:
        # Old flat format
        total_rounds = summary.get('total_election_rounds', summary.get('total_rounds', 'Unknown'))
        winner = summary.get('winner_name', summary.get('winner', 'Unknown'))
    
    total_votes = len(voting_df)
    unique_voters = voting_df['agent_id'].nunique()
    
    summary_text = f"""
    Simulation Summary
    
    Winner: {winner}
    Total Rounds: {total_rounds}
    Total Votes Cast: {total_votes}
    Unique Voters: {unique_voters}
    
    Final Vote Counts:
    """
    
    if len(vote_counts) > 0:
        for candidate, count in vote_counts.items():
            summary_text += f"\n    {candidate}: {count} votes"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = output_dir / 'voting_summary.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved voting summary plot: {plot_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze voting patterns and stance correlations')
    parser.add_argument('results_dir', type=str, help='Path to simulation results directory')
    parser.add_argument('--reduction-methods', type=str, nargs='+', default=['pca'], 
                       choices=['pca', 'tsne', 'umap'], help='Dimensionality reduction methods to use')
    parser.add_argument('--output-dir', type=str, help='Output directory for plots (default: visualizations folder)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Default to visualizations folder in the parent directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Go up one level from results to find visualizations folder
        parent_dir = results_dir.parent
        output_dir = parent_dir / "visualizations"
    
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load data
        summary, voting_df, embeddings_df = load_simulation_data(results_dir)
        
        # Create plots for each reduction method
        for method in args.reduction_methods:
            print(f"\n📊 Generating plots with {method.upper()}...")
            
            if method == 'umap' and not UMAP_AVAILABLE:
                print(f"⚠️ Skipping {method.upper()} - not available")
                continue
                
            create_voting_correlation_plot(summary, voting_df, embeddings_df, output_dir, method)
            create_evolution_plot(embeddings_df, voting_df, output_dir, method)
        
        # Create voting summary (only need one of these)
        create_voting_summary_plot(summary, voting_df, output_dir)
        
        print(f"\n✅ Analysis complete! Plots saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
