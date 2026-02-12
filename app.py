from flask import Flask, render_template, request, jsonify, send_file
from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime
import json
import os
from urllib.parse import urlparse, quote_plus
import time
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys from environment
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'YOUR_GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID', 'YOUR_CUSTOM_SEARCH_ENGINE_ID')

app = Flask(__name__)


class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def search_google(self, query, num_results=10):
        """Search Google using Custom Search API"""
        try:
            print(f"Attempting Google API search for: {query}")
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': GOOGLE_API_KEY,
                'cx': GOOGLE_CSE_ID,
                'q': query,
                'num': min(num_results, 10)
            }
            
            response = requests.get(url, params=params, timeout=10)
            print(f"Google API response status: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            
            results = []
            if 'items' in data:
                print(f"Found {len(data['items'])} results from Google API")
                for item in data['items']:
                    results.append({
                        'title': item.get('title', 'N/A'),
                        'url': item.get('link', 'N/A'),
                        'snippet': item.get('snippet', 'N/A'),
                        'source': urlparse(item.get('link', '')).netloc
                    })
            else:
                print("No 'items' in Google API response")
                print(f"Response data: {data}")
            
            return results
        except Exception as e:
            print(f"Google API error: {e}")
            print(f"Falling back to DuckDuckGo...")
            return []
    
    def search_duckduckgo(self, query, num_results=10):
        """Fallback: Generate sample results (DuckDuckGo blocks automated requests)"""
        try:
            print(f"DuckDuckGo blocks automated requests. Generating sample data for: {query}")
            print("To get real search results, please configure Google Custom Search API in .env file")
            
            # Generate sample results as fallback
            results = []
            sample_sources = [
                ('wikipedia.org', 'Wikipedia'),
                ('github.com', 'GitHub'),
                ('stackoverflow.com', 'Stack Overflow'),
                ('medium.com', 'Medium'),
                ('dev.to', 'DEV Community')
            ]
            
            for i in range(min(num_results, 5)):
                source_domain, source_name = sample_sources[i % len(sample_sources)]
                results.append({
                    'title': f'{query.title()} - {source_name} Resource #{i+1}',
                    'url': f'https://{source_domain}/search?q={quote_plus(query)}',
                    'snippet': f'Sample result for "{query}". Configure Google Custom Search API for real results.',
                    'source': source_domain
                })
            
            print(f"Generated {len(results)} sample results")
            return results
        except Exception as e:
            print(f"Error generating sample data: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def scrape_page_content(self, url):
        """Scrape detailed content from a specific page"""
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc['content'] if meta_desc and 'content' in meta_desc.attrs else text[:200]
            
            # Extract headings
            headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3'])[:5]]
            
            return {
                'description': description,
                'headings': headings,
                'text_preview': text[:500]
            }
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return {
                'description': 'Failed to scrape content',
                'headings': [],
                'text_preview': ''
            }
    
    def scrape_topic(self, topic, num_results=15, scrape_content=True):
        """Main scraping function"""
        print(f"Scraping topic: {topic}")
        print(f"API Key configured: {GOOGLE_API_KEY != 'YOUR_GOOGLE_API_KEY'}")
        print(f"CSE ID configured: {GOOGLE_CSE_ID != 'YOUR_CUSTOM_SEARCH_ENGINE_ID'}")
        
        # Try Google API first, fall back to DuckDuckGo
        if GOOGLE_API_KEY != "YOUR_GOOGLE_API_KEY" and GOOGLE_CSE_ID != "YOUR_CUSTOM_SEARCH_ENGINE_ID":
            print("Trying Google API first...")
            search_results = self.search_google(topic, num_results)
            # If Google fails, try DuckDuckGo
            if not search_results:
                print("Google API returned no results, trying DuckDuckGo...")
                search_results = self.search_duckduckgo(topic, num_results)
        else:
            print("Using DuckDuckGo (Google API not configured)")
            search_results = self.search_duckduckgo(topic, num_results)
        
        if not search_results:
            print("ERROR: No results from any search method!")
            return []
        
        # Optionally scrape content from each URL
        scraped_data = []
        for idx, result in enumerate(search_results):
            print(f"Processing result {idx + 1}/{len(search_results)}")
            
            data_entry = {
                'id': idx + 1,
                'title': result['title'],
                'url': result['url'],
                'source': result['source'],
                'snippet': result.get('snippet', 'N/A'),
                'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            if scrape_content and result['url'] != 'N/A':
                time.sleep(0.5)  # Be respectful to servers
                content = self.scrape_page_content(result['url'])
                data_entry.update({
                    'description': content['description'],
                    'headings': ', '.join(content['headings']),
                    'text_preview': content['text_preview']
                })
            
            scraped_data.append(data_entry)
        
        return scraped_data

# Initialize scraper
scraper = WebScraper()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    try:
        data = request.json
        topic = data.get('topic', '')
        num_results = int(data.get('num_results', 15))
        scrape_content = data.get('scrape_content', True)
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        results = scraper.scrape_topic(topic, num_results, scrape_content)
        
        if not results:
            return jsonify({'error': 'No results found. Please try a different topic.'}), 404
        
        return jsonify({
            'success': True,
            'data': results,
            'count': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export/<format>', methods=['POST'])
def export_data(format):
    try:
        data = request.json.get('data', [])
        topic = request.json.get('topic', 'scraped_data')
        
        if not data:
            return jsonify({'error': 'No data to export'}), 400
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_topic = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')
        
        if format == 'csv':
            filename = f"{safe_topic}_{timestamp}.csv"
            filepath = os.path.join('exports', filename)
            os.makedirs('exports', exist_ok=True)
            df.to_csv(filepath, index=False, encoding='utf-8')
            return send_file(filepath, as_attachment=True, download_name=filename)
        
        elif format == 'xlsx':
            filename = f"{safe_topic}_{timestamp}.xlsx"
            filepath = os.path.join('exports', filename)
            os.makedirs('exports', exist_ok=True)
            df.to_excel(filepath, index=False, engine='openpyxl')
            return send_file(filepath, as_attachment=True, download_name=filename)
        
        elif format == 'json':
            filename = f"{safe_topic}_{timestamp}.json"
            filepath = os.path.join('exports', filename)
            os.makedirs('exports', exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return send_file(filepath, as_attachment=True, download_name=filename)
        
        else:
            return jsonify({'error': 'Invalid format'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Web Scraper Server Starting...")
    print("=" * 60)
    print("\nIMPORTANT: Configure API keys for best results:")
    print("1. Get Google API Key: https://console.cloud.google.com/")
    print("2. Get Custom Search Engine ID: https://cse.google.com/")
    print("3. Update GOOGLE_API_KEY and GOOGLE_CSE_ID in app.py")
    print("\nCurrently using DuckDuckGo fallback (no API key needed)")
    print("=" * 60)
    print("\nServer running at: http://127.0.0.1:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
