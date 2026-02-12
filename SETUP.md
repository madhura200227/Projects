# Quick Setup Guide

## For New Users Cloning from GitHub

Follow these steps to get the web scraper running on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/web-scraper.git
cd web-scraper
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

**Copy the example file:**
```bash
cp .env.example .env
```

**Edit `.env` with your favorite editor:**
```bash
nano .env
# or
code .env
# or
vim .env
```

**Add your API keys:**
```
GOOGLE_API_KEY=your_actual_google_api_key_here
GOOGLE_CSE_ID=your_actual_search_engine_id_here
```

### 4. Get Your API Credentials

#### Google API Key:
1. Go to https://console.cloud.google.com/
2. Create a new project (or select existing)
3. Enable "Custom Search API"
4. Go to "Credentials" → "Create Credentials" → "API Key"
5. Copy the API key

#### Custom Search Engine ID:
1. Go to https://cse.google.com/
2. Click "Add" to create a new search engine
3. Add sites to search (see tips below)
4. Copy the "Search engine ID"

**Sites to Add for General Scraping:**
```
*.wikipedia.org
*.reddit.com
*.medium.com
*.stackoverflow.com
*.github.com
*.bbc.com
*.cnn.com
*.techcrunch.com
```
(Add up to 50 domains for broad coverage)

### 5. Run the Application
```bash
python app.py
```

### 6. Open Your Browser
Navigate to: http://127.0.0.1:5000

---

## Troubleshooting

**Q: I don't want to use Google API**  
A: No problem! The scraper will automatically use DuckDuckGo. Just skip the .env setup.

**Q: My API key doesn't work**  
A: Make sure you've enabled the "Custom Search API" in Google Cloud Console.

**Q: I get "quota exceeded" error**  
A: Google's free tier allows 100 searches/day. The scraper will fall back to DuckDuckGo.

**Q: Can I use this commercially?**  
A: Check Google's Custom Search API terms and the websites you're scraping.

---

## Security Note

🔒 **NEVER commit your `.env` file to Git!**

The `.gitignore` file is already configured to exclude it, but always double-check:
```bash
git status
```

You should NOT see `.env` in the list of files to be committed.

---

Happy Scraping! 🚀
