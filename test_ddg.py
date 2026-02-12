import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

def test_duckduckgo():
    query = "python programming"
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    print(f"Testing DuckDuckGo search for: {query}")
    print(f"URL: {url}")
    
    response = session.get(url, timeout=10)
    print(f"Response status: {response.status_code}")
    print(f"Response length: {len(response.content)} bytes")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    result_divs = soup.find_all('div', class_='result')
    print(f"Found {len(result_divs)} result divs")
    
    # Try different selectors
    links = soup.find_all('a', class_='result__a')
    print(f"Found {len(links)} result__a links")
    
    # Print first few div classes to see structure
    all_divs = soup.find_all('div', limit=20)
    print("\nFirst 20 div classes:")
    for i, div in enumerate(all_divs):
        classes = div.get('class', [])
        if classes:
            print(f"{i}: {classes}")

if __name__ == "__main__":
    test_duckduckgo()
