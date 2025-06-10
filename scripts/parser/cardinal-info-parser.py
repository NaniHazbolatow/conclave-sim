from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import unicodedata
from IPython.display import clear_output
import time
import csv

# Function for name formatting

def to_ascii(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')


    # Get the list of names

# Setup
options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

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
names.pop(-1)

print(names)
print(len(names))

names_formatted = [to_ascii(name.replace("ł", 'l')).strip().lower().replace(' ', '-').replace("'", '') for name in names]

driver.quit()

# Setup informatin dictionary

assert len(names) == len(names_formatted), "List length does not match"

info = {}
for i, name in enumerate(names):
    info[name] = {"formatted_name" : names_formatted[i]}

    # Obtain the summaries

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
                # time.sleep(2)
                driver.get(f"https://collegeofcardinalsreport.com/cardinals/cardinal-{info[name]['formatted_name']}")
                summary_div = driver.find_element(By.CLASS_NAME, "cardinals-summary-block")
                summary_text = summary_div.text

            i = i + 1
            clear_output(wait=True)
            print(i)

            info[name]["Background"] = summary_text
            
        except Exception as e:
            print(f"Error processing {name}: {e}")
            # Continue with next cardinal instead of quitting
            print(f"searched for: {info[name]['formatted_name']}")
            continue
            
finally:
    driver.quit()

# Go through the names that have not worked

print("Cant find the following names, please add manually via: https://collegeofcardinalsreport.com/cardinals/")

no_summary = [(name, data["formatted_name"]) for name, data in info.items() if "Background" not in data or not data["Background"].strip()]
for name, formatted in no_summary:
    print(f"{name} ({formatted})")

info["Christophe Pierre"]["Background"] = "Cardinal Christophe Louis Yves Georges Pierre, the apostolic nuncio to the United States, is an accomplished veteran Vatican diplomat who has had to deal — not always successfully — with tense relations between the American episcopate and the Francis pontificate. Born on January 30, 1946 in Rennes, France, Pierre completed his primary education in Madagascar and secondary schooling in France and Morocco before entering the seminary. After military service, he was ordained a priest for the diocese of Rennes, in the cathedral of Saint-Malo, on 5 April 1970. Pierre then pursued higher education, obtaining a Master's in Sacred Theology from the Catholic Institute of Paris, and a Doctorate in Canon Law in Rome. After completing further studies at the Pontifical Ecclesiastical Academy, the Holy See's training school for diplomats in Rome, Pierre's diplomatic career with the Holy See began in 1977, taking him to various postings around the world including New Zealand, Mozambique, Zimbabwe, Cuba, and Brazil. From 1991 to 1995, he was the Holy See's Permanent Observer to the United Nations in Geneva. In 1995 Pope John Paul II named him apostolic nuncio to Haiti. He went on to serve as nuncio to Uganda from 1999 to 2007 and to Mexico from 2007 to 2016. In 2016, Pope Francis appointed Pierre Apostolic Nuncio to the United States, succeeding Archbishop Carlo Maria Viganò who had retired on age grounds. In September 2023, Pope Francis elevated Pierre to the rank of cardinal – an unusual step as most apostolic nuncios to the U.S. are given the red hat after they leave the office. It is also rare, at least until Francis' pontificate, for an active papal diplomat to be made a cardinal. Cardinal Pierre has been described as a diplomat who aims to quell conflicts and promote harmony within the Church and he has had occasional successes in bridging divides and promote unity among Catholics. This was particularly apparent during his posting in Mexico where he was credited with overcoming political divisions. On the Eucharist, Cardinal Pierre strongly supported the recent National Eucharistic Revival in the United States. He said he saw it as a way to 'renew the Church' and believes the revival should lead to conversion of heart, commitment to evangelization, service, and community. Pierre emphasized the importance of believing in Christ's Real Presence in the Eucharist, stressing it is a source of unity for the Church, but also saying it means recognizing Christ 'in the assembly of His believing people' and even in those struggling to connect with Him. Pierre connects the Eucharistic revival with the concept of synodality promoted by Pope Francis, and he has encouraged U.S. bishops to embrace synodality as 'the path forward for the Church.' He sees both the Eucharist and synodality as interrelated paths for the Church's renewal and evangelization efforts. The French Vatican diplomat has affirmed that the Church 'must be unapologetically pro-life' and that the she cannot abandon its defence of innocent human life. He advocates a 'synodal approach' to abortion, stressing the need to listen and understand rather than simply condemn. Pierre's tenure in the U.S. has not been without problems. He faced several challenges, including mediating tensions between American bishops and the Vatican on issues such as the McCarrick scandal, the COVID-19 pandemic response, and disagreements over liturgical and doctrinal matters. And although he found some common ground with bishops on immigration (Pierre has been a strong advocate for immigrants and participated in demonstrations with border bishops against building walls on the border with Mexico), he has also faced several criticisms. These include his handling of episcopal misconduct cases related to clerical sex abuse in the U.S. and his reported reluctance to engage with the press on such matters. Other critics have said he has shown some misunderstanding of the U.S. Church, and have asserted that he has isolated himself from U.S. bishops, leading to diminishing support for him in the episcopate. Traditional Catholics have criticized Pierre for his strident views against the traditional liturgy and for reportedly pressuring diocesan bishops to cancel thriving Latin Masses in the United States. He has spoken negatively of young priests who 'dream about wearing cassocks and celebrating Mass in the pre-Vatican II way.' He sees this as potentially problematic, and as a response to feeling lost in modern society. Meanwhile, critics on the progressive wing of the Church have noted his struggle to help U.S. bishops connect with Pope Francis's vision, particularly regarding synodality. They also contend that he is regularly at odds with progressive U.S. cardinals such as Blase Cupich and Robert McElroy who have direct lines to the Pope. For his part, Pierre has been critical of the conservative Catholic press, but reportedly unwilling to consider reasons for their criticisms of Pope Francis. Regarding his role in helping to appoint bishops, Cardinal Pierre has been credited for helping a number of conservative-leaning priests to be elevated to the U.S. episcopate. Cardinal Pierre is known for his linguistic abilities, and speaks French, English, Italian, Spanish, and Portuguese fluently."
info["Ernest Simoni Troshani"]["Background"] = "Cardinal Ernest Simoni Troshani's life is a remarkable testimony to faith, forgiveness and perseverance having spent eighteen years in jail at the hands of Albanian communists, during which time he endured torture and harsh conditions that continued even after his release. Born on October 18, 1928, in Troshani, Albania, at the age of ten he entered the Franciscan College in Troshani to begin his formation for the priesthood. However, his path was disrupted in 1948 when the communist regime of Enver Hoxha began its persecution of religious institutions in the country. Despite the challenges, Simoni continued his theological studies clandestinely and was ordained priest on April 7, 1956, in Shkodrë. His ministry was marked by dedication and courage, even in the face of growing oppression. On Christmas Eve 1963, after celebrating Mass, Simoni was arrested and imprisoned. He was initially sentenced to death, but the sentence was commuted to twenty-five years of hard labor. During his eighteen years of imprisonment, Simoni endured torture and harsh conditions, including work in mines and sewage canals. Despite these hardships, he remained steadfast in his faith, secretly celebrating Mass from memory and hearing confessions of fellow prisoners. After his release in 1981, he was still considered an 'enemy of the people' and was forced to work in the Shkodrë sewers, but he continued to exercise his priestly ministry clandestinely until the fall of the communist regime in 1990. Simoni's extraordinary witness to faith caught the attention of Pope Francis during his visit to Albania in 2014 and on November 19, 2016, he elevated Simoni to the rank of cardinal, assigning him the titular church of Santa Maria della Scala. The appointment was also symbolic, honoring the suffering of Albanian Catholics under communism and promoting their courageous witness to the wider Catholic world. Throughout his life, Cardinal Simoni has been a testament to forgiveness and perseverance. He never used words of hate or resentment towards his jailers, believing that 'only love conquers.'' Now in his 90s, Cardinal Simoni continues to share his powerful testimony with communities around the world, reminding people of the strength of faith in the face of adversity."

# Save to csv

with open('cardinals_info.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Name', 'Background'])
    for name, data in info.items():
        background = data.get("Background", "").replace('\n', ' ')
        writer.writerow([name, background])