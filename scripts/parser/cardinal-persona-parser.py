#!/usr/bin/env python3
"""
Cardinal Persona Parser

This script generates internal and external personas for cardinals based on their background information.
It loads existing cardinal data and uses GPT-4o-mini with specialized prompts to generate:
- Internal Persona: Detailed 4-bullet analysis for simulation purposes
- External Persona: Brief 2-sentence public profile for other cardinals

The output CSV contains:
- Cardinal_ID: Sequential numeric ID for each cardinal
- Name: Cardinal's name
- Internal_Persona: 4-bullet internal analysis for AI agent
- External_Persona: 2-sentence public profile

Usage:
- Run with --personas-only flag to only generate personas from existing data
- Run without flag to scrape fresh data and generate personas
"""

import pandas as pd
import sys
import os
from pathlib import Path
import argparse
import yaml
import csv

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from conclave.llm.client import RemoteLLMClient


def load_prompts():
    """Load prompt templates from the YAML file."""
    prompts_path = Path(__file__).parent / "parse_prompts.yaml"
    try:
        print(f"Loading prompts from: {prompts_path}")
        with open(prompts_path, 'r', encoding='utf-8') as file:
            content = file.read()
            print(f"YAML content length: {len(content)} characters")
            prompts = yaml.safe_load(content)
            print(f"Loaded prompts: {prompts}")
            if prompts is None:
                raise ValueError("YAML file is empty or invalid")
            return prompts
    except Exception as e:
        print(f"Error loading prompts from {prompts_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_internal_persona(name: str, background: str, llm_client: RemoteLLMClient, prompt_template: str) -> str:
    """
    Generate an internal persona using the provided prompt template.
    
    Args:
        name: Cardinal's name
        background: Cardinal's background information
        llm_client: LLM client for generation
        prompt_template: Template for the prompt
        
    Returns:
        Generated internal persona string
    """
    try:
        # Format the prompt with cardinal name and biography
        prompt = prompt_template.format(
            agent_name=name,
            biography=background
        )
        
        # Use the correct method: prompt with messages list
        messages = [{"role": "user", "content": prompt}]
        response = llm_client.prompt(messages)
        return response.strip()
        
    except Exception as e:
        print(f"Error generating internal persona for {name}: {e}")
        return "Error generating internal persona"


def generate_external_persona(name: str, internal_persona: str, llm_client: RemoteLLMClient, prompt_template: str) -> str:
    """
    Generate an external persona using the provided prompt template.
    
    Args:
        name: Cardinal's name
        internal_persona: Generated internal persona
        llm_client: LLM client for generation
        prompt_template: Template for the prompt
        
    Returns:
        Generated external persona string
    """
    try:
        # Format the prompt with cardinal name and internal persona
        prompt = prompt_template.format(
            agent_name=name,
            persona_internal=internal_persona
        )
        
        # Use the correct method: prompt with messages list
        messages = [{"role": "user", "content": prompt}]
        response = llm_client.prompt(messages)
        return response.strip()
        
    except Exception as e:
        print(f"Error generating external persona for {name}: {e}")
        return "Error generating external persona"


def generate_personas(cardinal_info, llm_client, prompts, personas_only=False, test_mode=False):
    """Generate internal and external personas for all cardinals."""
    print("Generating personas using GPT-4o-mini...")
    
    # Get prompt templates
    internal_prompt = prompts['INTERNAL_PERSONA_EXTRACTOR']
    external_prompt = prompts['EXTERNAL_PROFILE_GENERATOR']
    
    # In test mode, only process first 5 cardinals
    cardinals_to_process = list(cardinal_info.items())
    if test_mode:
        cardinals_to_process = cardinals_to_process[:5]
        print(f"TEST MODE: Processing only {len(cardinals_to_process)} cardinals")
    
    processed_count = 0
    total_count = len([name for name, data in cardinals_to_process if "Background" in data and data["Background"].strip()])
    
    for name, data in cardinals_to_process:
        if "Background" not in data or not data["Background"].strip():
            print(f"Skipping {name} (no background available)")
            continue
            
        # Skip if personas already exist and we're in personas-only mode
        if personas_only and "Internal_Persona" in data and "External_Persona" in data:
            print(f"Skipping {name} (personas already exist)")
            continue
            
        print(f"Processing {processed_count + 1}/{total_count}: {name}")
        
        # Generate internal persona
        if "Internal_Persona" not in data or not data["Internal_Persona"].strip():
            print(f"  Generating internal persona...")
            internal_persona = generate_internal_persona(name, data["Background"], llm_client, internal_prompt)
            data["Internal_Persona"] = internal_persona
            print(f"  Internal: {internal_persona[:100]}...")
        else:
            internal_persona = data["Internal_Persona"]
            print(f"  Using existing internal persona")
        
        # Generate external persona
        if "External_Persona" not in data or not data["External_Persona"].strip():
            print(f"  Generating external persona...")
            external_persona = generate_external_persona(name, internal_persona, llm_client, external_prompt)
            data["External_Persona"] = external_persona
            print(f"  External: {external_persona[:100]}...")
        else:
            print(f"  Using existing external persona")
        
        processed_count += 1
        print(f"  Completed {name}")
        print()
    
    print(f"Persona generation complete! Processed {processed_count} cardinals.")
    return cardinal_info


def generate_cardinal_id(index: int) -> str:
    """
    Generate a sequential ID for a cardinal based on their index.
    
    Args:
        index: The cardinal's index in the list (0-based)
        
    Returns:
        A sequential numeric ID as string
    """
    return str(index)


def save_to_csv(cardinal_info, output_path):
    """Save cardinal information to CSV file with persona columns only (background omitted)."""
    print(f"Saving data to {output_path}...")
    
    # Generate Cardinal IDs if they don't exist
    cardinal_ids = set()
    for i, (name, data) in enumerate(cardinal_info.items()):
        if "Cardinal_ID" not in data or not data["Cardinal_ID"]:
            data["Cardinal_ID"] = generate_cardinal_id(i)
        cardinal_ids.add(data["Cardinal_ID"])
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Only include persona columns, omit background
        writer.writerow(['Cardinal_ID', 'Name', 'Internal_Persona', 'External_Persona'])
        
        for name, data in cardinal_info.items():
            cardinal_id = data.get("Cardinal_ID", "")
            internal_persona = data.get("Internal_Persona", "").replace('\n', ' ')
            external_persona = data.get("External_Persona", "").replace('\n', ' ')
            writer.writerow([cardinal_id, name, internal_persona, external_persona])
    
    print(f"Generated {len(cardinal_ids)} unique cardinal IDs")


def load_existing_data(csv_path):
    """Load existing cardinal data from CSV file."""
    print(f"Loading existing data from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    
    # Convert to the format expected by the rest of the script
    cardinal_info = {}
    for _, row in df.iterrows():
        name = row['Name']
        cardinal_info[name] = {
            "Background": row.get('Background', ''),
        }
        
        # Add existing Cardinal_ID if present
        if 'Cardinal_ID' in row and pd.notna(row['Cardinal_ID']):
            cardinal_info[name]["Cardinal_ID"] = str(row['Cardinal_ID'])
            
        # Add existing personas if present
        if 'Internal_Persona' in row and pd.notna(row['Internal_Persona']):
            cardinal_info[name]["Internal_Persona"] = row['Internal_Persona']
        if 'External_Persona' in row and pd.notna(row['External_Persona']):
            cardinal_info[name]["External_Persona"] = row['External_Persona']
            
        # Legacy support - check for old ideological stance column
        if 'Ideological_Stance' in row and pd.notna(row['Ideological_Stance']):
            cardinal_info[name]["Ideological_Stance"] = row['Ideological_Stance']
    
    print(f"Loaded {len(cardinal_info)} cardinals from CSV")
    return cardinal_info


def main():
    """Main function to run the cardinal persona parser."""
    parser = argparse.ArgumentParser(description='Generate internal and external personas for cardinals')
    parser.add_argument('--personas-only', action='store_true', 
                       help='Only generate personas from existing data, skip scraping')
    parser.add_argument('--test', action='store_true',
                       help='Test mode - only process first 5 cardinals')
    args = parser.parse_args()
    
    print("Starting Cardinal Persona Parser...")
    if args.test:
        print("*** TEST MODE - Processing only first 5 cardinals ***")
    print("=" * 50)
    
    # Load prompt templates
    print("Loading prompt templates...")
    prompts = load_prompts()
    
    # Initialize the LLM client with GPT-4o-mini
    print("Initializing LLM client with GPT-4o-mini...")
    llm_client = RemoteLLMClient(model_name="openai/gpt-4o-mini")
    
    if not llm_client.is_available():
        print("Error: LLM client not available. Please check your OPENROUTER_API_KEY.")
        return
    
    # Load existing cardinal data
    existing_csv = project_root / "data" / "cardinal_electors_2025.csv"
    if not existing_csv.exists():
        print(f"Error: No existing CSV found at {existing_csv}")
        print("Please run the unified parser first to generate cardinal data.")
        return
    
    cardinal_info = load_existing_data(existing_csv)
    
    # Report on cardinals with/without backgrounds
    cardinals_with_background = [name for name, data in cardinal_info.items() if "Background" in data and data["Background"].strip()]
    cardinals_without_background = [name for name, data in cardinal_info.items() if "Background" not in data or not data["Background"].strip()]
    
    print(f"\nData summary:")
    print(f"Cardinals with background: {len(cardinals_with_background)}")
    if cardinals_without_background:
        print(f"Cardinals without background: {len(cardinals_without_background)}")
        for name in cardinals_without_background:
            print(f"  - {name}")
    
    # Generate personas
    cardinal_info = generate_personas(cardinal_info, llm_client, prompts, personas_only=args.personas_only, test_mode=args.test)
    
    # Save to CSV in data folder
    output_path = project_root / "data" / "cardinal_electors_personas.csv"
    save_to_csv(cardinal_info, output_path)
    
    # Summary
    total_cardinals = len(cardinal_info)
    cardinals_with_internal = len([name for name, data in cardinal_info.items() if "Internal_Persona" in data and data["Internal_Persona"].strip()])
    cardinals_with_external = len([name for name, data in cardinal_info.items() if "External_Persona" in data and data["External_Persona"].strip()])
    
    print(f"\nProcessing complete!")
    print(f"Total cardinals: {total_cardinals}")
    print(f"Cardinals with internal personas: {cardinals_with_internal}")
    print(f"Cardinals with external personas: {cardinals_with_external}")
    print(f"Output saved to: {output_path}")
    
    # Optional: Also save a backup in the scripts directory
    backup_path = Path(__file__).parent / "cardinal_electors_personas_backup.csv"
    save_to_csv(cardinal_info, backup_path)
    print(f"Backup saved to: {backup_path}")


if __name__ == "__main__":
    main()
