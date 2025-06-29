import os
import json
import pandas as pd
import numpy as np
import re
from enum import Enum
import argparse
from tqdm import tqdm
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

class RunOutcome(Enum):
    CONSENSUS = "CONSENSUS"
    POLARISED = "POLARISED"
    GRIDLOCK = "GRIDLOCK"

def extract_params_from_path(path):
    temp_match = re.search(r'temp_(\d+_\d+)', path)
    rat_match = re.search(r'rat_(\d+_\d+)', path)
    run_match = re.search(r'run_(\d+)', path)
    temperature = float(temp_match.group(1).replace('_', '.')) if temp_match else None
    rationality = float(rat_match.group(1).replace('_', '.')) if rat_match else None
    run_id_from_path = int(run_match.group(1)) if run_match else None
    return temperature, rationality, run_id_from_path

def calculate_polarisation(stance_embeddings):
    if stance_embeddings.empty:
        return 0.0
    embedding_columns = [col for col in stance_embeddings.columns if col.startswith('embedding_')]
    if not embedding_columns:
        return 0.0
    embeddings = stance_embeddings[embedding_columns].values
    mean_vector = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - mean_vector, axis=1)
    return np.var(distances)

def get_run_outcome(consensus_reached, max_gridlock_span, total_rounds, max_rounds=50):
    """
    Determine the outcome of a simulation run.
    
    Args:
        consensus_reached: Whether consensus was reached (winner found before max rounds)
        max_gridlock_span: Maximum consecutive rounds with no vote changes
        total_rounds: Total rounds the simulation ran
        max_rounds: Maximum allowed rounds (default 50)
    
    Returns:
        RunOutcome enum value
    """
    if consensus_reached:
        return RunOutcome.CONSENSUS.value
    
    # If no consensus and hit max rounds, determine if gridlock or polarised
    if total_rounds is not None and total_rounds >= max_rounds:
        # Check if there was significant gridlock (no changes for 1/3 of total rounds)
        if max_gridlock_span >= total_rounds / 3:
            return RunOutcome.GRIDLOCK.value
        else:
            return RunOutcome.POLARISED.value
    
    # If simulation didn't reach max rounds and no consensus, likely an error or early termination
    # But still classify based on gridlock
    if total_rounds is not None and total_rounds > 0 and max_gridlock_span >= total_rounds / 3:
        return RunOutcome.GRIDLOCK.value
    
    return RunOutcome.POLARISED.value

def process_simulation_output(run_directory, param_directory):
    summary_path = os.path.join(run_directory, 'simulation_summary.json')
    votes_path = os.path.join(run_directory, 'voting_data.csv')
    json_votes_path = os.path.join(run_directory, 'individual_votes_by_round.json')
    embeddings_path = os.path.join(run_directory, 'stance_embeddings.csv')
    try:
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
        
        # Prefer JSON votes file if it exists, otherwise fall back to CSV
        votes_df = None
        if os.path.exists(json_votes_path):
            with open(json_votes_path, 'r') as f:
                json_votes_data = json.load(f)
            
            records = []
            # Handle cases where round numbers might be strings like "round_1"
            for round_num_str, round_data in json_votes_data.items():
                match = re.search(r'\d+', str(round_num_str))
                if match:
                    round_num = int(match.group())
                    for agent_id, candidate in round_data.items():
                        # Load the raw data, cleaning will be done later
                        records.append({'round': round_num, 'agent_id': agent_id, 'candidate_voted_for': candidate})
            if records:
                votes_df = pd.DataFrame(records)
        
        if votes_df is None and os.path.exists(votes_path):
            votes_df = pd.read_csv(votes_path)


        if votes_df is None:
            # If no voting data is found at all, create an empty dataframe to avoid errors
            votes_df = pd.DataFrame(columns=['round', 'agent_id', 'candidate_voted_for'])

        # Authoritative cleaning step: handle dicts and ensure string type
        # This is the single point of truth for cleaning the candidate name data
        if not votes_df.empty:
            votes_df['candidate_voted_for'] = votes_df['candidate_voted_for'].apply(
                lambda x: x.get('candidate_name') if isinstance(x, dict) else x
            )

        embeddings_df = pd.read_csv(embeddings_path)
        if embeddings_df['round'].dtype == 'object':
            embeddings_df['round'] = embeddings_df['round'].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)
        else:
            embeddings_df['round'] = embeddings_df['round'].fillna(0).astype(int)
        results = summary_data.get('results', {})
        config = summary_data.get('config', {})
        temperature, rationality, _ = extract_params_from_path(param_directory)
        run_id_base = os.path.basename(run_directory)
        param_dir_base = os.path.basename(param_directory)
        run_id = f"{param_dir_base}_{run_id_base}"
        if temperature is None:
            temperature = config.get('llm_agent', {}).get('temperature')
        if rationality is None:
            rationality = config.get('llm_agent', {}).get('rationality')
        # Determine consensus and rounds_to_consensus
        winner_name = results.get('winner_name', None)
        total_rounds = results.get('total_election_rounds')
        max_rounds = 50
        
        # Proper consensus logic:
        # - Consensus is reached if there's a winner AND we didn't hit the max round limit
        # - If total_rounds == max_rounds AND no winner, it's a failed simulation (gridlock/polarised)
        # - If total_rounds < max_rounds AND there's a winner, consensus was reached
        if winner_name and total_rounds is not None:
            if total_rounds < max_rounds:
                # Winner found before hitting round limit = consensus reached
                consensus_reached = True
                rounds_to_consensus = total_rounds
            else:
                # Hit round limit but somehow still has winner (edge case) = treat as consensus
                consensus_reached = True
                rounds_to_consensus = total_rounds
        else:
            # No winner = no consensus
            consensus_reached = False
            rounds_to_consensus = None
        # Final leading margin
        final_leading_margin = 0
        if not votes_df.empty:
            last_round_num = votes_df['round'].max()
            last_round_df = votes_df[votes_df['round'] == last_round_num]
            if not last_round_df.empty:
                vote_counts = last_round_df['candidate_voted_for'].value_counts()
                if len(vote_counts) > 1:
                    final_leading_margin = vote_counts.iloc[0] - vote_counts.iloc[1]
                elif len(vote_counts) == 1:
                    final_leading_margin = vote_counts.iloc[0]
        # Polarisation
        polarisation_start = calculate_polarisation(embeddings_df[embeddings_df['round'] == 0])
        polarisation_final = calculate_polarisation(embeddings_df[embeddings_df['round'] == total_rounds if total_rounds is not None else 0])
        peak_polarisation = 0
        if total_rounds is not None:
            for i in range(total_rounds + 1):
                round_embeddings = embeddings_df[embeddings_df['round'] == i]
                peak_polarisation = max(peak_polarisation, calculate_polarisation(round_embeddings))
        # Grid-lock
        max_gridlock_span = 0
        gridlock_end_round = None
        if total_rounds is not None and total_rounds > 0 and not votes_df.empty:
            vote_tallies = votes_df.groupby('round')['candidate_voted_for'].value_counts().unstack(fill_value=0)
            vote_tally_diff = vote_tallies.diff()
            no_change_streaks = (vote_tally_diff == 0).all(axis=1)
            current_streak = 0
            for i, value in enumerate(no_change_streaks):
                round_number = no_change_streaks.index[i]
                if value:
                    current_streak += 1
                else:
                    if current_streak >= 3:
                        if current_streak > max_gridlock_span:
                            max_gridlock_span = current_streak
                            gridlock_end_round = round_number - 1
                    current_streak = 0
            if current_streak >= 3 and current_streak > max_gridlock_span:
                max_gridlock_span = current_streak
                gridlock_end_round = None
        # Churn / Flexibility
        mean_switches_per_agent = 0
        switch_probability_per_round = 0
        mean_switches_per_round = 0
        if not votes_df.empty and total_rounds is not None and total_rounds > 0:
            # Data is already cleaned, so just perform the groupby and calculations
            votes_df_final = votes_df.groupby(['agent_id', 'round'])['candidate_voted_for'].last().reset_index()
            votes_df_final = votes_df_final.sort_values(by=['agent_id', 'round'])
            
            # Now count switches between rounds
            votes_df_final['prev_vote'] = votes_df_final.groupby('agent_id')['candidate_voted_for'].shift()
            switches = ((votes_df_final['candidate_voted_for'] != votes_df_final['prev_vote']) & 
                       votes_df_final['prev_vote'].notna()).sum()
            
            # Robustly get the number of agents
            num_agents = summary_data.get('config', {}).get('num_agents')
            if num_agents is None or num_agents == 0:
                num_agents = votes_df['agent_id'].nunique()

            if num_agents > 0:
                mean_switches_per_agent = switches / num_agents
                # The probability of any given agent switching in any given round
                switch_probability_per_round = switches / (num_agents * total_rounds) if total_rounds > 0 and num_agents > 0 else 0
                # The average number of agents switching per round
                mean_switches_per_round = switches / total_rounds if total_rounds > 0 else 0

        # Outcome Label
        run_outcome = get_run_outcome(consensus_reached, max_gridlock_span, total_rounds, max_rounds)
        
        # Debug info for validation
        if consensus_reached and (total_rounds is None or total_rounds >= max_rounds):
            print(f"Warning: {run_id} marked as consensus but total_rounds={total_rounds}, max_rounds={max_rounds}")
        if not consensus_reached and winner_name:
            print(f"Warning: {run_id} has winner '{winner_name}' but consensus_reached=False")
        
        return {
            'run_id': run_id,
            'temperature': temperature,
            'rationality': rationality,
            'consensus_reached': consensus_reached,
            'rounds_to_consensus': rounds_to_consensus,
            'total_rounds': total_rounds,  # Add this for debugging
            'final_leading_margin': final_leading_margin,
            'polarisation_start': polarisation_start,
            'polarisation_peak': peak_polarisation,
            'polarisation_final': polarisation_final,
            'max_gridlock_span': max_gridlock_span,
            'gridlock_end_round': gridlock_end_round,
            'mean_switches_per_agent': mean_switches_per_agent,
            'switch_probability_per_round': switch_probability_per_round,
            'mean_switches_per_round': mean_switches_per_round,
            'run_outcome': run_outcome,
            'winner_name': winner_name,
        }
    except (FileNotFoundError, json.JSONDecodeError, AttributeError, pd.errors.EmptyDataError) as e:
        print(f"Error processing directory {run_directory}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Combine simulation results.")
    parser.add_argument('base_dir', nargs='?', default='results/snellius-collected', help='The base directory containing the Snellius parameter folders.')
    args = parser.parse_args()
    all_results = []
    param_dirs = [d for d in os.listdir(args.base_dir) if not d.startswith('.') and os.path.isdir(os.path.join(args.base_dir, d)) and d.startswith('temp_')]
    for param_dir_name in tqdm(param_dirs, desc='Parameter folders'):
        param_dir_path = os.path.join(args.base_dir, param_dir_name)
        finished_dir_path = os.path.join(param_dir_path, 'finished')
        if os.path.isdir(finished_dir_path):
            run_dirs = [d for d in os.listdir(finished_dir_path) if not d.startswith('.') and os.path.isdir(os.path.join(finished_dir_path, d))]
            for run_dir_name in tqdm(run_dirs, desc=f'Runs in {param_dir_name}', leave=False):
                run_dir_path = os.path.join(finished_dir_path, run_dir_name)
                processed_data = process_simulation_output(run_dir_path, param_dir_path)
                if processed_data:
                    all_results.append(processed_data)
    if not all_results:
        print("No 'finished' simulation summary files found.")
        return
    df = pd.DataFrame(all_results)
    column_order = [
        'run_id', 'temperature', 'rationality', 'consensus_reached', 
        'rounds_to_consensus', 'total_rounds', 'final_leading_margin', 'polarisation_start', 
        'polarisation_peak', 'polarisation_final', 'max_gridlock_span', 
        'gridlock_end_round', 'mean_switches_per_agent', 'switch_probability_per_round', 'mean_switches_per_round', 'run_outcome', 'winner_name'
    ]
    for col in column_order:
        if col not in df.columns:
            df[col] = None
    df = df[column_order]
    output_dir = '../../results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'combined_results_snellius.csv')
    df.to_csv(output_path, index=False)
    print(f"Combined Snellius results saved to {output_path}")

if __name__ == '__main__':
    main()
