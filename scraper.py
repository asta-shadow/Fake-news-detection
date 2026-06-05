import requests
from bs4 import BeautifulSoup

def get_latest_news():
    url = "https://news.google.com/topstories?hl=en-IN&gl=IN&ceid=IN:en"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        headlines = []
        # Target specific Google News CSS class
        for item in soup.find_all("a", class_="g350p")[:8]:
            headlines.append({
                "title": item.text,
                "link": "https://news.google.com" + item['href'][1:]
            })
        return headlines
    except:
        return []