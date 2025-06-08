
#!/usr/bin/env python3
"""generate_cardinals_csv.py
Download the up‑to‑date list of the 2025 conclave’s voting cardinals
from the Wikipedia page and save it as `cardinal_electors_2025.csv`.

Columns:
  Name, Country/See, Role_Office, Date_of_Birth, Age, Background
"""

import pandas as pd, re, requests, datetime, sys
from bs4 import BeautifulSoup
from openai import OpenAI

client = OpenAI()

URL = "https://en.wikipedia.org/wiki/Cardinal_electors_in_the_2025_papal_conclave"

def fetch_table():
    res = requests.get(URL, timeout=30)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    # grab the rows from the first big table under the h2 'Cardinal electors'
    table = soup.find("table", class_="wikitable")
    rows = []
    for tr in table.find_all("tr")[1:]:  # skip header
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if len(cells) < 7:      # skip any malformed rows
            continue
        rank, name, country, born, order, consistory, office = cells[:7]

        # born is like "17 January 1955 (age 70)"
        m = re.match(r"([0-9]+\s+\w+\s+[0-9]{4})", born)
        dob_str = m.group(1) if m else ""
        dob = pd.to_datetime(dob_str, errors="coerce")
        age = int((datetime.date(2025, 4, 21) - dob.date()).days // 365.25) if not pd.isna(dob) else ""


        response = client.responses.create(
            model="gpt-4.1",
            tools=[{
                "type": "web_search_preview",
            }],
            input=f"Describe the background and views of cardinal {name} in a single concise paragraph."
        )

        background = response.output_text

        rows.append(
            {
                "Name": name,
                "Country/See": country,
                "Role_Office": office,
                "Date_of_Birth": dob_str,
                "Age": age,
                "Background": background
            }
        )
    return pd.DataFrame(rows)

def main():
    df = fetch_table()
    out_name = "../data/cardinal_electors_2025.csv"
    df.to_csv(out_name, index=False)
    print(f"Wrote {len(df)} rows to {out_name}")

if __name__ == "__main__":
    main()
