import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import numpy as np
import re
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

def parse_round_id(round_id):
    """Converts a round ID string like 'ER1.D5' to an integer by extracting the last number."""
    if isinstance(round_id, str):
        numbers = re.findall(r'\d+', round_id)
        if numbers:
            return int(numbers[-1])
    try:
        return int(round_id)
    except (ValueError, TypeError):
        return -1 # Return a default/error value

# --- Configuration ---
# Hardcoded path to the results folder. Change this to target a different run.
RESULTS_DIR = Path("results/snellius-collected/temp_1_00_rat_1_00/finished/13_run_38_mbelkhatir")
OUTPUT_VIS_DIR = Path("results/visualization")

# Parameters for filtering the specific simulation run
TEMPERATURE = 1.00
RATIONALITY = 1.00

# t-SNE settings
PERPLEXITY = 30
RANDOM_STATE = 42

# --- Static Color Mapping for Candidates ---
CANDIDATE_COLOR_MAP = {
    "Lazzaro You Heung-sik": "#E63946",      # Bright Red
    "Luis Antonio Gokim Tagle": "#B0926A",   # Brownish Gold
    "Robert Francis Prevost": "#52B788",     # Muted Green
    "Víctor Manuel Fernández": "#45B7D1",    # Teal/Cyan
    "Claudio Gugerotti": "#A8DADC",          # Light Blue
    "Pietro Parolin": "#C377E0",             # Purple
    # Add more candidates and colors as needed
}

def get_color(candidate_name):
    """Returns a specific color for a candidate, or a default gray if not in the map."""
    return CANDIDATE_COLOR_MAP.get(candidate_name, "#808080") # Default to gray

def visualize_stance_evolution():
    """
    Loads stance embeddings and voting data, merges them, and creates a 3-panel
    visualization showing the evolution of agent stances at the first, middle,
    and final rounds of a simulation.
    """
    # --- 1. Load Data ---
    # Create the output directory if it doesn't exist
    OUTPUT_VIS_DIR.mkdir(parents=True, exist_ok=True)
    
    stance_embeddings_path = RESULTS_DIR / "stance_embeddings.csv"
    voting_data_path = RESULTS_DIR / "voting_data.csv"

    if not stance_embeddings_path.exists() or not voting_data_path.exists():
        print(f"Error: Make sure '{stance_embeddings_path}' and '{voting_data_path}' exist.")
        return

    print("Loading data...")
    df_embeddings = pd.read_csv(stance_embeddings_path)
    df_voting = pd.read_csv(voting_data_path)

    # --- 2. Filter and Merge ---
    # Convert string round IDs (e.g., 'ER1.D1') to integers
    df_embeddings['round'] = df_embeddings['round'].apply(parse_round_id)
    df_voting['round'] = df_voting['round'].apply(parse_round_id)

    # Ensure other merge keys are of the same numeric type
    df_embeddings['agent_id'] = df_embeddings['agent_id'].astype(int)
    df_voting['agent_id'] = df_voting['agent_id'].astype(int)

    # Merge voting data to align votes with embeddings for each round
    # The agent_name and cardinal_id columns are also used to ensure correctness.
    df_merged = pd.merge(df_embeddings, df_voting, on=['round', 'agent_id', 'agent_name', 'cardinal_id'], how='left')

    # Since we are using a specific run folder, we don't need to filter by temp/rat
    run_df = df_merged.copy()
    
    if run_df.empty:
        print(f"No data found for temperature={TEMPERATURE} and rationality={RATIONALITY}. Please check the configuration.")
        return

    # --- 3. Identify Rounds to Plot ---
    # Filter out any invalid rounds (e.g., -1 from parsing)
    rounds = sorted([r for r in run_df['round'].unique() if r >= 0])
    
    if not rounds:
        print("Error: No valid rounds found to plot.")
        return

    if len(rounds) < 3:
        print("Warning: Fewer than 3 valid rounds available. Plotting all available rounds.")
        rounds_to_plot = rounds
    else:
        mid_round_index = len(rounds) // 2
        mid_round = rounds[mid_round_index]
        rounds_to_plot = [rounds[0], mid_round, rounds[-1]]
    
    print(f"Rounds to plot: {rounds_to_plot}")

    # --- 4. Create Visualization ---
    fig, axes = plt.subplots(1, len(rounds_to_plot), figsize=(18, 10), sharey=True)
    if len(rounds_to_plot) == 1: # Ensure axes is always a list
        axes = [axes]
        
    # Use the static color map for consistency
    unique_voted_for = run_df['candidate_voted_for'].dropna().unique()
    color_map = {name: get_color(name) for name in unique_voted_for}

    for i, round_num in enumerate(rounds_to_plot):
        ax = axes[i]
        round_data = run_df[run_df['round'] == round_num].copy()

        # Extract embeddings
        embedding_cols = [col for col in round_data.columns if col.startswith('embedding_')]
        embeddings = round_data[embedding_cols].values

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=min(PERPLEXITY, len(embeddings) - 1), random_state=RANDOM_STATE)
        embeddings_2d = tsne.fit_transform(embeddings)
        round_data['tsne_1'] = embeddings_2d[:, 0]
        round_data['tsne_2'] = embeddings_2d[:, 1]
        
        # Identify candidates for this round (agents who received votes)
        candidates_in_round = round_data['candidate_voted_for'].dropna().unique()
        round_data['is_candidate'] = round_data['agent_name'].isin(candidates_in_round)
        
        # --- Plotting & Bloc Analysis ---
        # Plot non-candidates (circles)
        non_candidates = round_data[~round_data['is_candidate']]
        hue_arg = 'candidate_voted_for' if not non_candidates['candidate_voted_for'].isnull().all() else None
        sns.scatterplot(data=non_candidates, x='tsne_1', y='tsne_2', hue=hue_arg, hue_order=unique_voted_for, palette=color_map, style=None, ax=ax, s=100, legend=False)

        # Plot candidates (squares)
        candidates = round_data[round_data['is_candidate']]
        hue_arg_cand = 'candidate_voted_for' if not candidates['candidate_voted_for'].isnull().all() else None
        sns.scatterplot(data=candidates, x='tsne_1', y='tsne_2', hue=hue_arg_cand, hue_order=unique_voted_for, palette=color_map, marker='s', ax=ax, s=150, legend=False, ec='black')
        
        # --- Add Bloc Information and Encirclements ---
        # Calculate bloc percentages
        bloc_counts = round_data['candidate_voted_for'].value_counts()
        bloc_percentages = (bloc_counts / len(round_data) * 100).sort_values(ascending=False)
        
        # Encircle the blocs
        for name, _ in bloc_percentages.items():
            group = round_data[round_data['candidate_voted_for'] == name]
            points = group[['tsne_1', 'tsne_2']].values
            if len(points) > 2:
                try:
                    hull = ConvexHull(points)
                    poly = Polygon(points[hull.vertices, :], facecolor=color_map.get(name, 'gray'), alpha=0.15, ec=None, zorder=-1)
                    ax.add_patch(poly)
                except Exception as e:
                    print(f"Could not draw hull for {name} in round {round_num}: {e}")

        # --- Create and place the annotation box for top 3 blocs ---
        top_3_blocs_text = "Top 3 Blocs:\n" + "\n".join([f"{name}: {perc:.1f}%" for name, perc in bloc_percentages.head(3).items()])
        
        # Place text box below the plot
        ax.text(0.5, -0.25, top_3_blocs_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        ax.set_title(f"Round {round_num}")
        ax.set_xlabel("t-SNE Component 1")
    
    axes[0].set_ylabel("t-SNE Component 2")

    # --- 5. Final Touches ---
    plt.suptitle(f"Evolution of Stance Embeddings (Temp={TEMPERATURE}, Rat={RATIONALITY})", fontsize=16)
    
    # Create a single legend for the entire figure
    # Create a comprehensive legend from all unique voted-for names across the run
    all_voted_for = run_df['candidate_voted_for'].dropna().unique()
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=name,
                                 markerfacecolor=color_map.get(name, 'gray'), markersize=10)
                      for name in all_voted_for]
    
    fig.legend(handles=legend_handles, title="Voted For", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent overlap and make space for annotations
    plt.tight_layout(rect=[0, 0.1, 0.9, 0.9])

    # Save the figure
    output_filename = f"{RESULTS_DIR.name}_stance_evolution.png"
    output_path = OUTPUT_VIS_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    visualize_stance_evolution()