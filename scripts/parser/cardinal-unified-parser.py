#!/usr/bin/env python3
"""
Unified Cardinal Parser

This script combines the functionality of both the cardinal-info-parser and cardinal-brief-parser.
It scrapes detailed background information from collegeofcardinalsreport.com and then uses
GPT-4o-mini to generate brief ideological descriptions for each cardinal.

The output CSV contains:
- Cardinal_ID: Sequential numeric ID for each cardinal
- Name: Cardinal's name
- Background: Detailed background information
- Ideological_Stance: Brief ideological description

Output is saved to the data folder as cardinal_electors_unified.csv
"""

import pandas as pd
import sys
import os
from pathlib import Path
import time
import csv
import unicodedata
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from conclave.llm.client import RemoteLLMClient


def generate_cardinal_id(index: int) -> str:
    """
    Generate a sequential ID for a cardinal based on their index.
    
    Args:
        index: The cardinal's index in the list (0-based)
        
    Returns:
        A sequential numeric ID as string
    """
    return str(index)


def to_ascii(text):
    """Convert text to ASCII format for URL formatting."""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')


def scrape_cardinal_names():
    """Scrape cardinal names from collegeofcardinalsreport.com."""
    print("Scraping cardinal names...")
    
    # Setup
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    try:
        # Open the site
        driver.get("https://collegeofcardinalsreport.com/cardinals/")
        time.sleep(2)

        # Optional: click 'Show full list'
        try:
            show_button = driver.find_element(By.XPATH, '//*[contains(text(), "Show full list")]')
            show_button.click()
            time.sleep(2)  # wait for content to load
        except:
            print("Show full list button not found or already loaded.")

        # Grab all name elements
        name_elements = driver.find_elements(By.XPATH, '//*[contains(text(), "Cardinal ")]')

        # Extract names by removing the "Cardinal " prefix
        names = [el.text.replace("Cardinal ", "").strip() for el in name_elements]
        names.pop(-1)  # Remove last element (usually a duplicate or unwanted element)

        print(f"Found {len(names)} cardinals")
        return names
        
    finally:
        driver.quit()


def format_name_for_url(name):
    """Format cardinal name for URL usage."""
    return to_ascii(name.replace("ł", 'l')).strip().lower().replace(' ', '-').replace("'", '')


def scrape_cardinal_backgrounds(names):
    """Scrape detailed background information for each cardinal."""
    print("Scraping cardinal backgrounds...")
    
    info = {}
    for name in names:
        info[name] = {"formatted_name": format_name_for_url(name)}

    # Setup webdriver for background scraping
    options = Options()
    options.add_argument("--lang=en-US")  
    options.add_argument("--headless")  
    driver = webdriver.Chrome(options=options)

    try:
        for i, name in enumerate(names):
            try:
                driver.get(f"https://collegeofcardinalsreport.com/cardinals/{info[name]['formatted_name']}")

                try:
                    summary_div = driver.find_element(By.CLASS_NAME, "cardinals-summary-block")
                    summary_text = summary_div.text
                except:
                    # Try alternative URL format
                    driver.get(f"https://collegeofcardinalsreport.com/cardinals/cardinal-{info[name]['formatted_name']}")
                    summary_div = driver.find_element(By.CLASS_NAME, "cardinals-summary-block")
                    summary_text = summary_div.text

                info[name]["Background"] = summary_text
                print(f"Processed {i + 1}/{len(names)}: {name}")
                
            except Exception as e:
                print(f"Error processing {name}: {e}")
                print(f"Searched for: {info[name]['formatted_name']}")
                continue
                
    finally:
        driver.quit()

    # Add manual entries for problematic cases
    if "Christophe Pierre" in info and "Background" not in info["Christophe Pierre"]:
        info["Christophe Pierre"]["Background"] = "Cardinal Christophe Louis Yves Georges Pierre, the apostolic nuncio to the United States, is an accomplished veteran Vatican diplomat who has had to deal — not always successfully — with tense relations between the American episcopate and the Francis pontificate. Born on January 30, 1946 in Rennes, France, Pierre completed his primary education in Madagascar and secondary schooling in France and Morocco before entering the seminary. After military service, he was ordained a priest for the diocese of Rennes, in the cathedral of Saint-Malo, on 5 April 1970. Pierre then pursued higher education, obtaining a Master's in Sacred Theology from the Catholic Institute of Paris, and a Doctorate in Canon Law in Rome. After completing further studies at the Pontifical Ecclesiastical Academy, the Holy See's training school for diplomats in Rome, Pierre's diplomatic career with the Holy See began in 1977, taking him to various postings around the world including New Zealand, Mozambique, Zimbabwe, Cuba, and Brazil. From 1991 to 1995, he was the Holy See's Permanent Observer to the United Nations in Geneva. In 1995 Pope John Paul II named him apostolic nuncio to Haiti. He went on to serve as nuncio to Uganda from 1999 to 2007 and to Mexico from 2007 to 2016. In 2016, Pope Francis appointed Pierre Apostolic Nuncio to the United States, succeeding Archbishop Carlo Maria Viganò who had retired on age grounds. In September 2023, Pope Francis elevated Pierre to the rank of cardinal – an unusual step as most apostolic nuncios to the U.S. are given the red hat after they leave the office. It is also rare, at least until Francis' pontificate, for an active papal diplomat to be made a cardinal. Cardinal Pierre has been described as a diplomat who aims to quell conflicts and promote harmony within the Church and he has had occasional successes in bridging divides and promote unity among Catholics. This was particularly apparent during his posting in Mexico where he was credited with overcoming political divisions. On the Eucharist, Cardinal Pierre strongly supported the recent National Eucharistic Revival in the United States. He said he saw it as a way to 'renew the Church' and believes the revival should lead to conversion of heart, commitment to evangelization, service, and community. Pierre emphasized the importance of believing in Christ's Real Presence in the Eucharist, stressing it is a source of unity for the Church, but also saying it means recognizing Christ 'in the assembly of His believing people' and even in those struggling to connect with Him. Pierre connects the Eucharistic revival with the concept of synodality promoted by Pope Francis, and he has encouraged U.S. bishops to embrace synodality as 'the path forward for the Church.' He sees both the Eucharist and synodality as interrelated paths for the Church's renewal and evangelization efforts. The French Vatican diplomat has affirmed that the Church 'must be unapologetically pro-life' and that the she cannot abandon its defence of innocent human life. He advocates a 'synodal approach' to abortion, stressing the need to listen and understand rather than simply condemn. Pierre's tenure in the U.S. has not been without problems. He faced several challenges, including mediating tensions between American bishops and the Vatican on issues such as the McCarrick scandal, the COVID-19 pandemic response, and disagreements over liturgical and doctrinal matters. And although he found some common ground with bishops on immigration (Pierre has been a strong advocate for immigrants and participated in demonstrations with border bishops against building walls on the border with Mexico), he has also faced several criticisms. These include his handling of episcopal misconduct cases related to clerical sex abuse in the U.S. and his reported reluctance to engage with the press on such matters. Other critics have said he has shown some misunderstanding of the U.S. Church, and have asserted that he has isolated himself from U.S. bishops, leading to diminishing support for him in the episcopate. Traditional Catholics have criticized Pierre for his strident views against the traditional liturgy and for reportedly pressuring diocesan bishops to cancel thriving Latin Masses in the United States. He has spoken negatively of young priests who 'dream about wearing cassocks and celebrating Mass in the pre-Vatican II way.' He sees this as potentially problematic, and as a response to feeling lost in modern society. Meanwhile, critics on the progressive wing of the Church have noted his struggle to help U.S. bishops connect with Pope Francis's vision, particularly regarding synodality. They also contend that he is regularly at odds with progressive U.S. cardinals such as Blase Cupich and Robert McElroy who have direct lines to the Pope. For his part, Pierre has been critical of the conservative Catholic press, but reportedly unwilling to consider reasons for their criticisms of Pope Francis. Regarding his role in helping to appoint bishops, Cardinal Pierre has been credited for helping a number of conservative-leaning priests to be elevated to the U.S. episcopate. Cardinal Pierre is known for his linguistic abilities, and speaks French, English, Italian, Spanish, and Portuguese fluently."

    if "Ernest Simoni Troshani" in info and "Background" not in info["Ernest Simoni Troshani"]:
        info["Ernest Simoni Troshani"]["Background"] = "Cardinal Ernest Simoni Troshani's life is a remarkable testimony to faith, forgiveness and perseverance having spent eighteen years in jail at the hands of Albanian communists, during which time he endured torture and harsh conditions that continued even after his release. Born on October 18, 1928, in Troshani, Albania, at the age of ten he entered the Franciscan College in Troshani to begin his formation for the priesthood. However, his path was disrupted in 1948 when the communist regime of Enver Hoxha began its persecution of religious institutions in the country. Despite the challenges, Simoni continued his theological studies clandestinely and was ordained priest on April 7, 1956, in Shkodrë. His ministry was marked by dedication and courage, even in the face of growing oppression. On Christmas Eve 1963, after celebrating Mass, Simoni was arrested and imprisoned. He was initially sentenced to death, but the sentence was commuted to twenty-five years of hard labor. During his eighteen years of imprisonment, Simoni endured torture and harsh conditions, including work in mines and sewage canals. Despite these hardships, he remained steadfast in his faith, secretly celebrating Mass from memory and hearing confessions of fellow prisoners. After his release in 1981, he was still considered an 'enemy of the people' and was forced to work in the Shkodrë sewers, but he continued to exercise his priestly ministry clandestinely until the fall of the communist regime in 1990. Simoni's extraordinary witness to faith caught the attention of Pope Francis during his visit to Albania in 2014 and on November 19, 2016, he elevated Simoni to the rank of cardinal, assigning him the titular church of Santa Maria della Scala. The appointment was also symbolic, honoring the suffering of Albanian Catholics under communism and promoting their courageous witness to the wider Catholic world. Throughout his life, Cardinal Simoni has been a testament to forgiveness and perseverance. He never used words of hate or resentment towards his jailers, believing that 'only love conquers.'' Now in his 90s, Cardinal Simoni continues to share his powerful testimony with communities around the world, reminding people of the strength of faith in the face of adversity."

    return info


def create_ideological_summary(name: str, background: str, llm_client: RemoteLLMClient) -> str:
    """
    Create a brief ideological description for a cardinal using their background.
    
    Args:
        name: Cardinal's name
        background: Cardinal's detailed background description
        llm_client: LLM client for generating the summary
        
    Returns:
        Brief ideological description
    """
    prompt = f"""Based on the following detailed background of Cardinal {name}, create a brief, direct ideological description that other cardinals can use to understand their stance and ideology. The description should be 2-3 sentences maximum and focus on their theological positions, political leanings, and approach to Church governance.

Background:
{background}

Create a concise ideological profile that covers:
- Their theological stance (conservative/progressive/moderate)
- Key issues they prioritize
- Their approach to Church governance and reform

Brief ideological description:"""

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = llm_client.prompt(messages, max_tokens=200, temperature=0.3)
        return response.strip()
    except Exception as e:
        print(f"Error generating summary for {name}: {e}")
        return "Error generating ideological summary"


def generate_ideological_descriptions(cardinal_info, llm_client):
    """Generate brief ideological descriptions for all cardinals."""
    print("Generating ideological descriptions using GPT-4o-mini...")
    
    processed_count = 0
    total_count = len([name for name, data in cardinal_info.items() if "Background" in data and data["Background"].strip()])
    
    for name, data in cardinal_info.items():
        if "Background" not in data or not data["Background"].strip():
            print(f"Skipping {name} (no background available)")
            continue
            
        print(f"Processing {processed_count + 1}/{total_count}: {name}")
        
        # Generate ideological summary
        ideological_stance = create_ideological_summary(name, data["Background"], llm_client)
        data["Ideological_Stance"] = ideological_stance
        
        print(f"  Generated: {ideological_stance[:100]}...")
        processed_count += 1

    return cardinal_info


def save_to_csv(cardinal_info, output_path):
    """Save cardinal information to CSV file with Cardinal_ID as first column."""
    print(f"Saving data to {output_path}...")
    
    # Generate Cardinal IDs and check for duplicates
    cardinal_ids = []
    names_list = list(cardinal_info.keys())
    
    for index, name in enumerate(names_list):
        cardinal_id = generate_cardinal_id(index)
        cardinal_ids.append(cardinal_id)
        cardinal_info[name]["Cardinal_ID"] = cardinal_id
        print(f"  {name} -> {cardinal_id}")
    
    # Check for duplicate IDs (shouldn't happen with sequential numbers)
    unique_ids = set(cardinal_ids)
    if len(unique_ids) != len(cardinal_ids):
        print("\nError: Found duplicate IDs!")
        for i, cardinal_id in enumerate(cardinal_ids):
            if cardinal_ids.count(cardinal_id) > 1:
                print(f"  Duplicate ID {cardinal_id} for {names_list[i]}")
        print("This should not happen with sequential numbering.")
        sys.exit(1)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Put Cardinal_ID first in the column order
        writer.writerow(['Cardinal_ID', 'Name', 'Background', 'Ideological_Stance'])
        
        for name, data in cardinal_info.items():
            cardinal_id = data.get("Cardinal_ID", "")
            background = data.get("Background", "").replace('\n', ' ')
            ideological_stance = data.get("Ideological_Stance", "")
            writer.writerow([cardinal_id, name, background, ideological_stance])
    
    print(f"Generated {len(cardinal_ids)} unique cardinal IDs")


def main():
    """Main function to run the unified cardinal parser."""
    print("Starting Unified Cardinal Parser...")
    print("=" * 50)
    
    # Initialize the LLM client with GPT-4o-mini
    print("Initializing LLM client with GPT-4o-mini...")
    llm_client = RemoteLLMClient(model_name="openai/gpt-4o-mini")
    
    if not llm_client.is_available():
        print("Error: LLM client not available. Please check your OPENROUTER_API_KEY.")
        return
    
    # Check if existing CSV exists and ask user preference
    existing_csv = project_root / "data" / "cardinal_electors_2025.csv"
    if existing_csv.exists():
        print(f"\nFound existing CSV at {existing_csv}")
        use_existing = input("Use existing data (y) or scrape fresh data (n)? [y/N]: ").strip().lower()
        
        if use_existing == 'y':
            print("Loading existing data...")
            df = pd.read_csv(existing_csv)
            
            # Convert to the format expected by the rest of the script
            cardinal_info = {}
            for _, row in df.iterrows():
                name = row['Name']
                cardinal_info[name] = {
                    "Background": row['Background'],
                    "formatted_name": format_name_for_url(name)
                }
                # Add existing Cardinal_ID if present
                if 'Cardinal_ID' in row and pd.notna(row['Cardinal_ID']):
                    cardinal_info[name]["Cardinal_ID"] = str(row['Cardinal_ID'])
                # Add existing ideological stance if present
                if 'Ideological_Stance' in row and pd.notna(row['Ideological_Stance']):
                    cardinal_info[name]["Ideological_Stance"] = row['Ideological_Stance']
            
            names = list(cardinal_info.keys())
            print(f"Loaded {len(names)} cardinals from existing CSV")
        else:
            # Scrape fresh data
            names = scrape_cardinal_names()
            cardinal_info = scrape_cardinal_backgrounds(names)
    else:
        # No existing CSV, must scrape
        print("No existing CSV found, scraping fresh data...")
        names = scrape_cardinal_names()
        cardinal_info = scrape_cardinal_backgrounds(names)
    
    # Report on missing backgrounds
    no_summary = [(name, data.get("formatted_name", "")) for name, data in cardinal_info.items() if "Background" not in data or not data["Background"].strip()]
    if no_summary:
        print(f"\nWarning: Could not find backgrounds for {len(no_summary)} cardinals:")
        for name, formatted in no_summary:
            print(f"  - {name} ({formatted})")
        print("You may need to add these manually from: https://collegeofcardinalsreport.com/cardinals/")
    
    # Step 3: Generate ideological descriptions (only for those that don't have them)
    cardinal_info = generate_ideological_descriptions(cardinal_info, llm_client)
    
    # Step 4: Save to CSV in data folder
    output_path = project_root / "data" / "cardinal_electors_unified.csv"
    save_to_csv(cardinal_info, output_path)
    
    # Summary
    total_cardinals = len(names)
    processed_cardinals = len([name for name, data in cardinal_info.items() if "Ideological_Stance" in data])
    
    print(f"\nProcessing complete!")
    print(f"Total cardinals found: {total_cardinals}")
    print(f"Cardinals with complete data: {processed_cardinals}")
    print(f"Unique Cardinal IDs generated: {len(set(data.get('Cardinal_ID', '') for data in cardinal_info.values() if data.get('Cardinal_ID')))}")
    print(f"Output saved to: {output_path}")
    
    # Optional: Also save a backup in the scripts directory
    backup_path = Path(__file__).parent / "cardinal_electors_unified_backup.csv"
    save_to_csv(cardinal_info, backup_path)
    print(f"Backup saved to: {backup_path}")


if __name__ == "__main__":
    main()
