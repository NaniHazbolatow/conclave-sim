import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import numpy as np
import re
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

# --- Import the custom plotting theme ---
try:
    import plotting_theme
except ImportError:
    print("Warning: plotting_theme.py not found. Using default styles.")

def parse_round_id(round_id):
    """Converts a round ID string like 'ER1.D5' to an integer by extracting the last number."""
    if isinstance(round_id, str):
        numbers = re.findall(r'\d+', round_id)
        if numbers:
            return int(numbers[-1])
    try:
        return int(round_id)
    except (ValueError, TypeError):
        return -1

# --- Configuration ---
RESULTS_DIR = Path("results/snellius-collected/temp_2_00_rat_1_00/finished/13_run_21_ehazbolatow")
OUTPUT_VIS_DIR = Path("results/visualization")
TEMPERATURE = 2.00
RATIONALITY = 1.00
PERPLEXITY = 30
RANDOM_STATE = 42

# --- Thematic Color Mapping ---
CANDIDATE_NAMES_ORDERED = [
    "Luis Antonio Gokim Tagle", "Robert Francis Prevost", "Lazzaro You Heung-sik",
    "Claudio Gugerotti", "Víctor Manuel Fernández", "Pietro Parolin", 'José Tolentino de Mendonça'
]
CANDIDATE_COLOR_MAP = {
    name: plotting_theme.papal_colors[i % len(plotting_theme.papal_colors)]
    for i, name in enumerate(CANDIDATE_NAMES_ORDERED)
}

def get_color(candidate_name):
    return CANDIDATE_COLOR_MAP.get(candidate_name, "#808080")

def visualize_stance_evolution():
    # --- 1. Load Data ---
    OUTPUT_VIS_DIR.mkdir(parents=True, exist_ok=True)
    stance_embeddings_path = RESULTS_DIR / "stance_embeddings.csv"
    voting_data_path = RESULTS_DIR / "voting_data.csv"

    if not stance_embeddings_path.exists() or not voting_data_path.exists():
        print(f"Error: Required files not found in {RESULTS_DIR}")
        return

    print("Loading data...")
    df_embeddings = pd.read_csv(stance_embeddings_path)
    df_voting = pd.read_csv(voting_data_path)

    # --- 2. Filter and Merge ---
    df_embeddings['round'] = df_embeddings['round'].apply(parse_round_id)
    df_voting['round'] = df_voting['round'].apply(parse_round_id)
    df_embeddings['agent_id'] = df_embeddings['agent_id'].astype(int)
    df_voting['agent_id'] = df_voting['agent_id'].astype(int)
    df_merged = pd.merge(df_embeddings, df_voting, on=['round', 'agent_id', 'agent_name', 'cardinal_id'], how='left')
    run_df = df_merged.copy()

    # --- 3. Identify Rounds to Plot ---
    rounds = sorted([r for r in run_df['round'].unique() if r >= 0])
    if not rounds:
        print("Error: No valid rounds found.")
        return
    rounds_to_plot = [rounds[0], rounds[len(rounds) // 2], rounds[-1]] if len(rounds) >= 3 else rounds
    print(f"Rounds to plot: {rounds_to_plot}")

    # --- 4. Create Visualization ---
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(1, len(rounds_to_plot), figsize=(24, 9), sharey=True, dpi=100)
    if len(rounds_to_plot) == 1:
        axes = [axes]
        
    for i, round_num in enumerate(rounds_to_plot):
        ax = axes[i]
        round_data = run_df[run_df['round'] == round_num].copy()
        
        embedding_cols = [col for col in round_data.columns if col.startswith('embedding_')]
        embeddings = round_data[embedding_cols].values

        tsne = TSNE(n_components=2, perplexity=min(PERPLEXITY, len(embeddings) - 1), random_state=RANDOM_STATE)
        embeddings_2d = tsne.fit_transform(embeddings)
        round_data['tsne_1'] = embeddings_2d[:, 0]
        round_data['tsne_2'] = embeddings_2d[:, 1]
        
        candidates_in_round = round_data['candidate_voted_for'].dropna().unique()
        round_data['is_candidate'] = round_data['agent_name'].isin(candidates_in_round)
        
        sns.scatterplot(data=round_data, x='tsne_1', y='tsne_2', hue='candidate_voted_for', 
                        style='is_candidate', markers={True: 's', False: 'o'},
                        palette=CANDIDATE_COLOR_MAP, ax=ax, s=150, legend=False, ec='black', zorder=10)
        
        for name in candidates_in_round:
            group = round_data[round_data['candidate_voted_for'] == name]
            points = group[['tsne_1', 'tsne_2']].values
            if len(points) > 2:
                try:
                    hull = ConvexHull(points)
                    poly = Polygon(points[hull.vertices, :], facecolor=get_color(name), alpha=0.15, ec=None)
                    ax.add_patch(poly)
                except Exception as e:
                    print(f"Could not draw hull for {name}: {e}")

        # --- Percentage Annotation ---
        bloc_percentages = (round_data['candidate_voted_for'].value_counts(normalize=True) * 100).sort_values(ascending=False)
        bloc_text = "Top Blocs:\n" + "\n".join([f"{name}: {perc:.1f}%" for name, perc in bloc_percentages.head(3).items()])
        ax.text(0.03, 0.97, bloc_text, transform=ax.transAxes, fontsize=12, family='Georgia',
                va='top', ha='left', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

        ax.set_title(f"Round {round_num}", fontfamily='Georgia', fontweight='bold', fontsize=20)
        ax.set_xlabel("t-SNE Component 1", fontfamily='Georgia', fontsize=16)
    
    axes[0].set_ylabel("t-SNE Component 2", fontfamily='Georgia', fontsize=16)

    # --- 5. Final Touches with Larger Fonts ---
    fig.suptitle(f"Evolution of Stance Embeddings (Temp={TEMPERATURE}, Rat={RATIONALITY})", fontsize=24, fontweight='bold', fontfamily='Georgia')
    
    all_voted_for = sorted(run_df['candidate_voted_for'].dropna().unique())
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=name,
                                 markerfacecolor=get_color(name), markersize=14)
                      for name in all_voted_for]
    
    fig.legend(handles=legend_handles, title="Voted For", loc='upper center', 
               bbox_to_anchor=(0.5, 0.94), ncol=len(all_voted_for), frameon=False,
               prop={'family': 'Georgia', 'size': 14}, 
               title_fontproperties={'family': 'Georgia', 'size': 16, 'weight': 'bold'})
    
    plt.tight_layout(rect=[0, 0, 1, 0.88]) # Adjust rect to make space for legend and title

    output_filename = f"{RESULTS_DIR.name}_stance_evolution_styled_large.png"
    output_path = OUTPUT_VIS_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Styled visualization saved to: {output_path}")

if __name__ == "__main__":
    visualize_stance_evolution()
