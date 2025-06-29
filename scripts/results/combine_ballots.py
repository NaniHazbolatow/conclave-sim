import os
import re
import pandas as pd
import argparse
from tqdm import tqdm

def extract_params_from_path(path):
    """
    Extracts temperature and rationality from a given directory path using regex.
    Example path: results/snellius-collected/temp_0_10_rat_0_75
    """
    temp_match = re.search(r'temp_(\d+_\d+)', path)
    rat_match = re.search(r'rat_(\d+_\d+)', path)
    
    temperature = float(temp_match.group(1).replace('_', '.')) if temp_match else None
    rationality = float(rat_match.group(1).replace('_', '.')) if rat_match else None
    
    return temperature, rationality

def main():
    parser = argparse.ArgumentParser(description="Combine individual run ballot data into a single CSV.")
    parser.add_argument('base_dir', nargs='?', default='results/snellius-collected', 
                        help='The base directory containing parameter folders (e.g., temp_0_10_rat_0_75).')
    parser.add_argument('--output_file', default='results/combined_ballots.csv',
                        help='The path to save the combined CSV file.')
    args = parser.parse_args()

    all_ballots = []
    
    param_dirs = [d for d in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir, d)) and d.startswith('temp_')]

    for param_dir_name in tqdm(param_dirs, desc="Processing parameter directories"):
        param_dir_path = os.path.join(args.base_dir, param_dir_name)
        
        temperature, rationality = extract_params_from_path(param_dir_name)
        
        finished_dir = os.path.join(param_dir_path, 'finished')
        if not os.path.isdir(finished_dir):
            continue

        run_dirs = [d for d in os.listdir(finished_dir) if os.path.isdir(os.path.join(finished_dir, d))]
        
        for run_dir_name in tqdm(run_dirs, desc=f"Runs in {param_dir_name}", leave=False):
            run_id = f"{param_dir_name}_{run_dir_name}"
            ballots_path = os.path.join(finished_dir, run_dir_name, 'voting_data.csv')
            
            if os.path.exists(ballots_path):
                try:
                    ballots_df = pd.read_csv(ballots_path)
                    ballots_df['temperature'] = temperature
                    ballots_df['rationality'] = rationality
                    ballots_df['run_id'] = run_id
                    all_ballots.append(ballots_df)
                except pd.errors.EmptyDataError:
                    print(f"Warning: Empty voting_data.csv found in {run_dir_name}")
                    continue

    if not all_ballots:
        print("No ballot data found to combine.")
        return

    combined_df = pd.concat(all_ballots, ignore_index=True)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Save the combined dataframe
    combined_df.to_csv(args.output_file, index=False)
    print(f"\nSuccessfully combined ballot data from {len(all_ballots)} runs.")
    print(f"Output saved to: {args.output_file}")
    print(f"Dataframe shape: {combined_df.shape}")

if __name__ == '__main__':
    main()