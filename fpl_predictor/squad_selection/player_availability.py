import requests
from bs4 import BeautifulSoup

# URL = "https://fantasy.premierleague.com/the-scout/player-news"

# response = requests.get(URL)
# soup = BeautifulSoup(response.content, "html.parser")
r = requests.get("https://www.fantasyfootballscout.co.uk/fantasy-football-injuries/")
soup = BeautifulSoup(r.text)
table = soup.find("table")
for table_row in table.find_all("tr"):
    for c in table_row.find_all("td"):
        print(c.name, c.text, c.attrs, c)
