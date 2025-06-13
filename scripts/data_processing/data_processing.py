#!/usr/bin/env python3
"""
Data Processing Pipeline for Cardinal Information

This script consolidates cardinal data, including background information,
and generates various persona-related fields using an LLM.

It can optionally scrape cardinal names and backgrounds from
https://collegeofcardinalsreport.com/cardinals/ before processing.

Generated fields include:
- Internal_Persona: Detailed 4-bullet analysis.
- Public_Profile: Brief 2-sentence public profile.
- Profile_Blurb: Ultra-concise (<=12 words) public profile blurb.
- Persona_Tag: Hyphenated keyword tag for ideology/style.

The script outputs a single CSV file containing all these fields.
It supports selective regeneration of fields and a test mode.
"""

import pandas as pd
import sys
import os
from pathlib import Path
import argparse
import yaml
import csv
import time
import unicodedata # For to_ascii

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.common.exceptions import NoSuchElementException, TimeoutException
# from selenium.webdriver.chrome.service import Service # Consider adding
# from webdriver_manager.chrome import ChromeDriverManager # For easier chromedriver management

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from conclave.llm.client import RemoteLLMClient

# --- Configuration ---
DEFAULT_INPUT_FILE = project_root / "data" / "cardinal_electors_2025.csv"
DEFAULT_OUTPUT_FILE = project_root / "data" / "cardinals_master_data.csv"
PROMPTS_FILE = Path(__file__).parent / "parse_prompts.yaml" # Adjusted path
LLM_MODEL = "openai/gpt-4o-mini"
TEST_MODE_LIMIT = 3 # Number of records to process in test mode
SCRAPE_URL = "https://collegeofcardinalsreport.com/cardinals/"

def load_prompts():
    """Load prompt templates from the YAML file."""
    try:
        print(f"Loading prompts from: {PROMPTS_FILE}")
        with open(PROMPTS_FILE, 'r', encoding='utf-8') as file:
            prompts = yaml.safe_load(file)
        if prompts is None:
            raise ValueError("YAML file is empty or invalid")
        print(f"Successfully loaded prompts: {list(prompts.keys())}")
        return prompts
    except Exception as e:
        print(f"Error loading prompts from {PROMPTS_FILE}: {e}")
        sys.exit(1)

def _generate_with_llm(llm_client: RemoteLLMClient, prompt: str, cardinal_name_for_error: str, field_name: str) -> str:
    """Helper function to call LLM and handle errors."""
    try:
        messages = [{"role": "user", "content": prompt}]
        # Add a small delay to avoid hitting rate limits if any
        #time.sleep(0.5)
        response = llm_client.prompt(messages)
        return response.strip()
    except Exception as e:
        print(f"Error generating {field_name} for {cardinal_name_for_error}: {e}")
        return f"Error generating {field_name}"

def generate_internal_persona(name: str, background: str, llm_client: RemoteLLMClient, template: str) -> str:
    prompt = template.format(agent_name=name, biography=background)
    return _generate_with_llm(llm_client, prompt, name, "Internal Persona")

def generate_public_profile(name: str, internal_persona: str, llm_client: RemoteLLMClient, template: str) -> str:
    prompt = template.format(agent_name=name, persona_internal=internal_persona)
    return _generate_with_llm(llm_client, prompt, name, "Public Profile")

def generate_profile_blurb(name: str, internal_persona: str, llm_client: RemoteLLMClient, template: str) -> str:
    prompt = template.format(agent_name=name, persona_internal=internal_persona)
    return _generate_with_llm(llm_client, prompt, name, "Profile Blurb")

def generate_persona_tag(internal_persona: str, llm_client: RemoteLLMClient, template: str, name: str) -> str:
    # Added name parameter for consistency in error reporting, though template might not use it
    prompt = template.format(persona_internal=internal_persona)
    return _generate_with_llm(llm_client, prompt, name, "Persona Tag")

# --- Web Scraping Functions ---
def to_ascii(text: str) -> str:
    """Convert unicode text to ASCII, removing accents etc."""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

def init_driver() -> webdriver.Chrome:
    """Initializes and returns a Selenium WebDriver."""
    options = ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--lang=en-US")
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36') # Common user agent
    # For robust chromedriver management, consider using webdriver_manager:
    # driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    # For now, assumes chromedriver is in PATH or specified via environment variables.
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(30) # Set a page load timeout
    return driver

def get_cardinal_names_from_web(driver: webdriver.Chrome) -> list[dict]:
    """Fetches the list of cardinal names and their URL slugs from the website."""
    print(f"Fetching cardinal list from {SCRAPE_URL}...")
    try:
        driver.get(SCRAPE_URL)
    except TimeoutException:
        print(f"Timeout loading initial cardinal list page: {SCRAPE_URL}")
        return []
    time.sleep(3) # Allow page to load and any JS to execute

    try:
        # Try to click "Show full list" if it exists
        show_button_xpath = "//button[contains(., 'Show full list') or contains(., 'Show Full List') or contains(., 'SHOW FULL LIST')]"
        show_button = driver.find_element(By.XPATH, show_button_xpath)
        driver.execute_script("arguments[0].scrollIntoView(true);", show_button) # Scroll to button
        time.sleep(0.5)
        driver.execute_script("arguments[0].click();", show_button) # More robust click
        print("Clicked 'Show full list' button.")
        time.sleep(3) # Allow content to load
    except NoSuchElementException:
        print("'Show full list' button not found or list already fully displayed.")
    except Exception as e:
        print(f"Error interacting with 'Show full list' button: {e}")

    # Find the anchor elements which contain the name and link
    # This XPATH targets <a> tags that have an href containing '/cardinals/' and are likely within the main content area for cardinals
    anchor_elements_xpath = "//div[contains(@class, 'cardinals-loop-item')]//a[contains(@href, '/cardinals/') and .//h5]"
    anchor_elements = driver.find_elements(By.XPATH, anchor_elements_xpath)
    
    cardinals = []
    if not anchor_elements:
        print(f"No cardinal anchor elements found with XPATH: {anchor_elements_xpath}. Trying a broader search for links with h5 names.")
        # Fallback or alternative XPATH if the primary one fails
        anchor_elements = driver.find_elements(By.XPATH, "//a[contains(@href, '/cardinals/') and .//h5]")


    for el in anchor_elements:
        try:
            name_tag = el.find_element(By.TAG_NAME, 'h5')
            raw_name = name_tag.text.replace("Cardinal ", "").strip()
            
            href = el.get_attribute('href')
            if not href:
                print(f"Warning: Found cardinal '{raw_name}' but no href attribute.")
                continue

            # Extract the slug from the href. Example: https://.../cardinals/john-doe/ -> john-doe
            # Or https://.../cardinals/john-doe -> john-doe
            slug = href.strip('/').split('/')[-1] 

            if raw_name and slug:
                # Formatted_Name is now the slug, used for URL construction
                cardinals.append({"Name": raw_name, "Formatted_Name_Slug": slug, "Background": ""})
            else:
                print(f"Warning: Could not extract name or slug from element. Href: {href}, Name found: {raw_name}")
        except NoSuchElementException:
            print(f"Warning: Found a cardinal link structure (href: {el.get_attribute('href')}) but could not extract name (h5 tag).")
        except Exception as e:
            print(f"Warning: Error processing a cardinal element: {e}")
            
    print(f"Found {len(cardinals)} cardinal names and slugs from the web.")
    return cardinals

def scrape_background_for_cardinal(driver: webdriver.Chrome, name: str, formatted_name_slug: str) -> str:
    """Scrapes the background summary for a single cardinal using its slug."""
    print(f"  Scraping background for {name} (slug: {formatted_name_slug})...")
    
    # Ensure SCRAPE_URL ends with a slash if it's the base for slugs
    base_scrape_url = SCRAPE_URL if SCRAPE_URL.endswith('/') else SCRAPE_URL + '/'
    target_url = f"{base_scrape_url}{formatted_name_slug.strip('/')}" # Slug itself might have leading/trailing slashes

    summary_text = ""
    try:
        print(f"    Trying URL: {target_url}")
        driver.get(target_url)
        # Wait for a known element in the summary block to appear, or a short general wait
        time.sleep(1) # Increased sleep slightly for page rendering

        # Attempt to find the summary block and then specific p tags within its container
        # This XPATH looks for <p> tags inside a div with class "elementor-widget-container",
        # which itself is inside a div with class "cardinals-summary-block".
        summary_paragraph_xpath = "//div[contains(@class, 'cardinals-summary-block')]//div[contains(@class, 'elementor-widget-container')]//p"
        summary_elements = driver.find_elements(By.XPATH, summary_paragraph_xpath)
        
        if summary_elements:
            summary_text_parts = [p.text.strip() for p in summary_elements if p.text.strip()]
            if summary_text_parts:
                summary_text = "\\n\\n".join(summary_text_parts).strip() # Join paragraphs with double newline
                print(f"    Successfully scraped background for {name} from {target_url} using specific p-tags.")
        
        if not summary_text: # Fallback if specific p tags not found or empty
            print(f"    Specific p-tags did not yield content for {name}. Trying fallback: general summary block text.")
            try:
                # Fallback: Get all text from the "cardinals-summary-block"
                summary_div_xpath = "//div[contains(@class, 'cardinals-summary-block')]"
                summary_div = driver.find_element(By.XPATH, summary_div_xpath) 
                summary_text = summary_div.text.strip()
                if summary_text:
                    print(f"    Successfully scraped background (fallback method) for {name} from {target_url}")
            except NoSuchElementException:
                print(f"    Fallback summary block (class 'cardinals-summary-block') not found for {name} at {target_url}")
                pass # Error will be returned below if summary_text is still empty

        if not summary_text:
             print(f"    Summary content not found or empty for {name} at {target_url} after all attempts.")
             return "Error: Could not scrape background (content not found)"
        return summary_text

    except TimeoutException:
        print(f"    Timeout loading page for {name} at {target_url}")
        return "Error: Could not scrape background (timeout)"
    except Exception as e:
        print(f"    Error scraping {name} from {target_url}: {e}")
        return "Error: Could not scrape background (exception)"

def scrape_all_cardinal_data() -> list[dict]:
    """Orchestrates the scraping of all cardinal names and their backgrounds."""
    driver = None
    try:
        driver = init_driver()
        cardinals_with_names_slugs = get_cardinal_names_from_web(driver)
        if not cardinals_with_names_slugs:
            print("No cardinal names found from web. Aborting scrape.")
            return []

        print(f"\\nStarting background scraping for {len(cardinals_with_names_slugs)} cardinals...")
        scraped_data_list = []
        for i, cardinal_info in enumerate(cardinals_with_names_slugs):
            name = cardinal_info['Name']
            slug = cardinal_info['Formatted_Name_Slug']
            print(f"Scraping {i+1}/{len(cardinals_with_names_slugs)}: {name}")
            background = scrape_background_for_cardinal(driver, name, slug)
            scraped_data_list.append({
                "Name": name,
                "Background": background 
                # We don't need to carry the slug forward beyond this point
            })
            time.sleep(0.3) # Be respectful: small delay between requests
        return scraped_data_list
    except Exception as e:
        print(f"An error occurred during the scraping process: {e}")
        # Optionally, re-raise or handle more gracefully
        # raise # Re-raise if this should halt execution
        return [] # Return empty list or handle error as appropriate
    finally:
        if driver:
            driver.quit()
            print("WebDriver closed.")
# --- End Web Scraping Functions ---

def load_cardinal_data(input_file: Path) -> list[dict]:
    """Load cardinal data from the input CSV file."""
    print(f"Loading cardinal data from: {input_file}")
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found.")
        # If the default input file is missing, it's a critical error.
        # If a custom input file is missing, it's also an error.
        sys.exit(1)
        
    try:
        df = pd.read_csv(input_file, dtype=str).fillna('') # Read all as string, fill NaN with empty string
        # Expected columns for the new master file. Background is essential from input.
        # Others will be generated or preserved.
        # Ensure essential columns like 'Name' and 'Background' are present.
        if 'Name' not in df.columns or 'Background' not in df.columns:
            print(f"Error: Input CSV {input_file} must contain 'Name' and 'Background' columns.")
            sys.exit(1)
        
        # If Cardinal_ID is missing, we might need to generate it, or ensure it's there.
        # For now, assume it's present or handle its absence if necessary.
        if 'Cardinal_ID' not in df.columns:
            print(f"Warning: 'Cardinal_ID' not found in {input_file}. Records will be processed by index if saving.")
            # Add a placeholder if it's missing, though it's better if it exists.
            df['Cardinal_ID'] = [str(i) for i in df.index]


        records = df.to_dict('records')
        print(f"Loaded {len(records)} records from {input_file}.")
        return records
    except Exception as e:
        print(f"Error reading or processing CSV {input_file}: {e}")
        sys.exit(1)

def save_cardinal_data(cardinal_data: list[dict], output_file: Path):
    """Save the processed cardinal data to the output CSV file."""
    if not cardinal_data:
        print("No data to save.")
        return

    print(f"Saving processed data to: {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Define the order of columns for the output CSV
    fieldnames = [
        'Cardinal_ID', 'Name', 'Background', 
        'Internal_Persona', 'Public_Profile', 
        'Profile_Blurb', 'Persona_Tag'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in cardinal_data:
            # Ensure all fields in fieldnames are present in the row, defaulting to empty string
            for field in fieldnames:
                if field not in row:
                    row[field] = ''
            # Replace newlines in text fields to keep CSV structure clean
            for key in ['Internal_Persona', 'Public_Profile', 'Profile_Blurb', 'Background']: # Updated External_Persona to Public_Profile
                 if key in row and isinstance(row[key], str):
                    row[key] = row[key].replace('\\n', ' ').replace('\\r', ' ')
            writer.writerow(row)
    print(f"Successfully saved {len(cardinal_data)} records to {output_file}.")

def main():
    parser = argparse.ArgumentParser(description="Cardinal Data Processing Pipeline")
    parser.add_argument('--input-file', type=Path, default=DEFAULT_INPUT_FILE,
                        help=f"Input CSV file (default: {DEFAULT_INPUT_FILE})")
    parser.add_argument('--output-file', type=Path, default=DEFAULT_OUTPUT_FILE,
                        help=f"Output CSV file (default: {DEFAULT_OUTPUT_FILE})")
    parser.add_argument('--regenerate-internal-persona', action='store_true', help="Force regeneration of Internal Persona")
    parser.add_argument('--regenerate-public-profile', action='store_true', help="Force regeneration of Public Profile") # Renamed from --regenerate-external-persona
    parser.add_argument('--regenerate-profile-blurb', action='store_true', help="Force regeneration of Profile Blurb")
    parser.add_argument('--regenerate-persona-tag', action='store_true', help="Force regeneration of Persona Tag")
    parser.add_argument('--scrape-backgrounds', action='store_true', help="Scrape cardinal names and backgrounds from the web before processing")
    parser.add_argument('--test', action='store_true', help=f"Process only the first {TEST_MODE_LIMIT} records for testing")
    args = parser.parse_args()

    print("Starting Cardinal Data Processing Pipeline...")
    if args.test:
        print(f"*** TEST MODE: Processing up to {TEST_MODE_LIMIT} records ***")

    prompts = load_prompts()
    llm_client = RemoteLLMClient(model_name=LLM_MODEL)
    if not llm_client.is_available():
        print(f"Error: LLM client for model {LLM_MODEL} not available. Check API key and model name.")
        sys.exit(1)

    cardinal_data_final_list: list[dict] = []
    input_file_path = args.input_file
    
    master_df = pd.DataFrame()

    if args.scrape_backgrounds:
        print("\\n--- Scraping Phase ---")
        scraped_records = scrape_all_cardinal_data() # Returns list of dicts [{'Name', 'Background'}]

        if scraped_records:
            print(f"Successfully scraped {len(scraped_records)} records from the web.")
            scraped_df = pd.DataFrame(scraped_records)
            scraped_df.rename(columns={'Background': 'Background_Scraped'}, inplace=True)
            
            if input_file_path.exists():
                print(f"Loading existing data from {input_file_path} to merge with scraped data...")
                try:
                    existing_df = pd.read_csv(input_file_path, dtype=str)
                    if 'Name' not in existing_df.columns:
                        print(f"Error: Input CSV {input_file_path} must contain 'Name' column for merging. Skipping merge of existing data.")
                        master_df = scraped_df.rename(columns={'Background_Scraped': 'Background'})
                    else:
                        # Merge scraped data with existing data
                        # 'outer' merge to keep all cardinals from both sources
                        merged_df = pd.merge(existing_df, scraped_df, on='Name', how='outer')
                        
                        # Prioritize scraped background: if Background_Scraped exists, use it. Otherwise, keep existing Background.
                        merged_df['Background'] = merged_df['Background_Scraped'].fillna(merged_df.get('Background', pd.Series(dtype=str)))
                        merged_df.drop(columns=['Background_Scraped'], inplace=True, errors='ignore')
                        master_df = merged_df
                except Exception as e:
                    print(f"Error merging scraped data with {input_file_path}: {e}. Using only scraped data.")
                    master_df = scraped_df.rename(columns={'Background_Scraped': 'Background'})
            else: # No input file, use scraped data as is
                print("No input file found or specified. Using only scraped data.")
                master_df = scraped_df.rename(columns={'Background_Scraped': 'Background'})
        
        else: # Scraping failed or returned no data
            print("Scraping did not return any data.")
            if input_file_path.exists():
                print(f"Falling back to input file: {input_file_path}")
                master_df = pd.read_csv(input_file_path, dtype=str)
            else:
                print(f"Error: Scraping failed and input file {input_file_path} not found. No data to process.")
                sys.exit(1)
    
    else: # Not scraping, load from input file
        if not input_file_path.exists():
            print(f"Error: Input file {input_file_path} not found and scraping not requested. No data to process.")
            sys.exit(1)
        print(f"Loading data from input file: {input_file_path}")
        master_df = pd.read_csv(input_file_path, dtype=str)

    # Ensure essential columns exist and fill NaNs
    for col in ['Name', 'Background', 'Cardinal_ID', 'Internal_Persona', 'Public_Profile', 'Profile_Blurb', 'Persona_Tag']: # Updated External_Persona to Public_Profile
        if col not in master_df.columns:
            master_df[col] = '' # Initialize if missing
    master_df.fillna('', inplace=True)


    # Generate Cardinal_ID if missing or empty string
    missing_id_mask = master_df['Cardinal_ID'].eq('')
    num_missing_ids = missing_id_mask.sum()
    if num_missing_ids > 0:
        print(f"Generating {num_missing_ids} missing Cardinal_IDs...")
        # Create a series of new IDs for rows where Cardinal_ID is ''
        new_ids = [f"C_AutoGen_{i + master_df.index.max() + 1}" for i in range(num_missing_ids)] # Make them unique if some IDs exist
        master_df.loc[missing_id_mask, 'Cardinal_ID'] = new_ids


    cardinal_data_final_list = master_df.to_dict('records')

    if not cardinal_data_final_list:
        print("No cardinal data loaded after attempting all sources. Exiting.")
        sys.exit(1)

    if args.test:
        print(f"Applying test mode limit: first {TEST_MODE_LIMIT} records.")
        cardinal_data_final_list = cardinal_data_final_list[:TEST_MODE_LIMIT]
        if not cardinal_data_final_list:
            print("No data available after applying test mode limit. Check input or scraping results.")
            sys.exit(1)
    
    print(f"\\n--- LLM Generation Phase ---")
    print(f"Processing {len(cardinal_data_final_list)} records.")
    
    processed_cardinals_list = []
    for idx, record in enumerate(cardinal_data_final_list):
        # Ensure record is a mutable copy if necessary, though to_dict('records') usually gives new dicts
        current_record = record.copy()
        
        print(f"\nProcessing record {idx + 1}/{len(cardinal_data_final_list)}: ID {current_record.get('Cardinal_ID', 'N/A')}, Name {current_record.get('Name', 'N/A')}")
        
        current_name = current_record.get('Name', '')
        current_background = current_record.get('Background', '')

        # Initialize fields if they don't exist to avoid KeyErrors later
        for f in ['Internal_Persona', 'Public_Profile', 'Profile_Blurb', 'Persona_Tag']: # Updated External_Persona to Public_Profile
            current_record.setdefault(f, '')

        # Check for valid Name and Background for generation
        if not current_name or not current_background or "Error: Could not scrape background" in current_background or current_background.strip() == '':
            error_msg_detail = "Missing Name or Background"
            if not current_name: error_msg_detail = "Missing Name"
            elif not current_background or current_background.strip() == '': error_msg_detail = "Missing Background"
            elif "Error: Could not scrape background" in current_background: error_msg_detail = "Background scraping failed"
            
            full_error_msg = f"Error: {error_msg_detail}"
            print(f"  Warning: Skipping LLM generation for {current_record.get('Cardinal_ID', 'Unknown ID')} ({current_name}) due to: {error_msg_detail}.")
            
            current_record['Internal_Persona'] = current_record.get('Internal_Persona') or full_error_msg
            current_record['Public_Profile'] = current_record.get('Public_Profile') or full_error_msg # Updated External_Persona to Public_Profile
            current_record['Profile_Blurb'] = current_record.get('Profile_Blurb') or full_error_msg
            current_record['Persona_Tag'] = current_record.get('Persona_Tag') or full_error_msg
            processed_cardinals_list.append(current_record)
            continue

        # Internal Persona
        # Regenerate if flag is set, or if field is empty, or if it contains an error message
        should_gen_internal = args.regenerate_internal_persona or \
                              not current_record.get('Internal_Persona') or \
                              "Error:" in current_record.get('Internal_Persona', '')
        if should_gen_internal:
            print(f"  Generating Internal Persona for {current_name}...")
            current_record['Internal_Persona'] = generate_internal_persona(current_name, current_background, llm_client, prompts['INTERNAL_PERSONA_EXTRACTOR'])
        else:
            print(f"  Skipping Internal Persona for {current_name} (exists, not error, no regen flag).")

        current_internal_persona = current_record.get('Internal_Persona', '')
        if "Error:" in current_internal_persona or not current_internal_persona:
             print(f"  Skipping dependent LLM generations for {current_name} due to missing/failed Internal Persona.")
             error_msg_dep = "Error: Internal Persona missing or failed"
             if args.regenerate_public_profile or not current_record.get('Public_Profile') or "Error:" in current_record.get('Public_Profile',''): current_record['Public_Profile'] = error_msg_dep
             if args.regenerate_profile_blurb or not current_record.get('Profile_Blurb') or "Error:" in current_record.get('Profile_Blurb',''): current_record['Profile_Blurb'] = error_msg_dep
             if args.regenerate_persona_tag or not current_record.get('Persona_Tag') or "Error:" in current_record.get('Persona_Tag',''): current_record['Persona_Tag'] = error_msg_dep
        else:
            # Public Profile (formerly External Persona)
            should_gen_public_profile = args.regenerate_public_profile or \
                                  not current_record.get('Public_Profile') or \
                                  "Error:" in current_record.get('Public_Profile', '')
            if should_gen_public_profile:
                print(f"  Generating Public Profile for {current_name}...")
                current_record['Public_Profile'] = generate_public_profile(current_name, current_internal_persona, llm_client, prompts['EXTERNAL_PROFILE_GENERATOR'])
            else:
                print(f"  Skipping Public Profile for {current_name} (exists, not error, no regen flag).")

            # Profile Blurb
            should_gen_blurb = args.regenerate_profile_blurb or \
                               not current_record.get('Profile_Blurb') or \
                               "Error:" in current_record.get('Profile_Blurb', '')
            if should_gen_blurb:
                print(f"  Generating Profile Blurb for {current_name}...")
                current_record['Profile_Blurb'] = generate_profile_blurb(current_name, current_internal_persona, llm_client, prompts['EXTERNAL_PROFILE_MINI'])
            else:
                print(f"  Skipping Profile Blurb for {current_name} (exists, not error, no regen flag).")

            # Persona Tag
            should_gen_tag = args.regenerate_persona_tag or \
                             not current_record.get('Persona_Tag') or \
                             "Error:" in current_record.get('Persona_Tag', '')
            if should_gen_tag:
                print(f"  Generating Persona Tag for {current_name}...")
                current_record['Persona_Tag'] = generate_persona_tag(current_internal_persona, llm_client, prompts['IDEOLOGY_TAG_EXTRACTOR'], current_name)
            else:
                print(f"  Skipping Persona Tag for {current_name} (exists, not error, no regen flag).")
        
        processed_cardinals_list.append(current_record)

    save_cardinal_data(processed_cardinals_list, args.output_file)
    print("\\nCardinal Data Processing Pipeline finished successfully.")

if __name__ == "__main__":
    main()
