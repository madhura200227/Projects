<<<<<<< HEAD
# Web Data Scraper - Python Edition

A powerful web scraper built with Python, Flask, and BeautifulSoup that allows you to search any topic and extract structured data from the web.

## Features

✅ **Search any topic** - Enter any subject and scrape relevant data  
✅ **Multiple data sources** - Uses Google Custom Search API or DuckDuckGo fallback  
✅ **Deep content extraction** - Scrapes full page content using BeautifulSoup  
✅ **Multiple export formats** - Download as CSV, XLSX, or JSON  
✅ **Beautiful UI** - Modern, responsive web interface  
✅ **Customizable** - Adjust number of results and scraping depth  

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/web-scraper.git
cd web-scraper
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

**IMPORTANT:** Set up your API keys before running the application.

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your actual API keys:**
   ```bash
   GOOGLE_API_KEY=your_actual_google_api_key
   GOOGLE_CSE_ID=your_actual_search_engine_id
   ```

3. **Get your API credentials:**

   - **Google API Key**: Visit [Google Cloud Console](https://console.cloud.google.com/)
     - Create a new project
     - Enable "Custom Search API"
     - Create credentials (API Key)

   - **Custom Search Engine ID**: Visit [Google CSE](https://cse.google.com/)
     - Create a new search engine
     - Add sites to search (or use broad sites for general scraping)
     - Copy the Search Engine ID

**Note:** The `.env` file is in `.gitignore` and will never be committed to Git. Keep your API keys secret!

**Alternative:** Without API keys, the scraper will automatically use DuckDuckGo (no setup required, but fewer features).

## Usage

### 1. Start the Server

```bash
python app.py
```

The server will start at `http://127.0.0.1:5000`

### 2. Open Your Browser

Navigate to: `http://127.0.0.1:5000`

### 3. Scrape Data

1. Enter a topic (e.g., "artificial intelligence", "climate change")
2. Set number of results (5-50)
3. Choose whether to scrape full content
4. Click "Scrape Data"
5. Preview results in the table
6. Export as CSV, XLSX, or JSON

## Project Structure

```
web_scraper/
├── app.py                 # Flask backend with scraping logic
├── templates/
│   └── index.html        # Web interface
├── exports/              # Export files saved here
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## How It Works

1. **Search**: Uses Google Custom Search API or DuckDuckGo HTML scraping
2. **Extract**: BeautifulSoup parses HTML and extracts:
   - Page titles
   - URLs
   - Snippets/descriptions
   - Meta descriptions
   - Headings (H1, H2, H3)
   - Text content preview
3. **Process**: Data is cleaned and structured into a pandas DataFrame
4. **Export**: Convert to CSV, XLSX, or JSON format

## API Endpoints

- `GET /` - Main web interface
- `POST /scrape` - Scrape data for a topic
  ```json
  {
    "topic": "string",
    "num_results": 15,
    "scrape_content": true
  }
  ```
- `POST /export/<format>` - Export data (csv, xlsx, json)

## Examples

### Example 1: Basic Search
```
Topic: "Python web scraping tutorials"
Results: 15
Scrape Content: Yes
```

### Example 2: News Articles
```
Topic: "latest AI developments 2024"
Results: 20
Scrape Content: Yes
```

### Example 3: Product Research
```
Topic: "best laptops 2024"
Results: 30
Scrape Content: No (faster)
```

## Troubleshooting

### Issue: No results found
- Try a different, more specific topic
- Check your internet connection
- If using Google API, verify your API keys are correct

### Issue: Scraping too slow
- Reduce number of results
- Disable "Scrape full content" option
- Some websites may block scraping attempts

### Issue: Export fails
- Ensure `exports/` directory exists
- Check write permissions
- Verify pandas and openpyxl are installed

## Limitations

- Respects robots.txt and server rate limits (0.5s delay between requests)
- Some websites may block scraping attempts
- Google Custom Search API has daily quota limits (100 queries/day free tier)
- DuckDuckGo scraping may be less reliable than API

## Best Practices

1. **Be respectful**: Don't overload servers with too many requests
2. **Use API keys**: Better results and more reliable
3. **Start small**: Test with 5-10 results first
4. **Check legality**: Ensure you have the right to scrape target websites
5. **Rate limiting**: Built-in delays prevent server overload

## Technologies Used

- **Python 3.x**
- **Flask** - Web framework
- **BeautifulSoup4** - HTML parsing
- **Requests** - HTTP library
- **Pandas** - Data manipulation
- **OpenPyXL** - Excel file creation

## License

MIT License - Feel free to use and modify

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Check console logs for error messages
4. Try with a different topic

---

**Happy Scraping! 🔍**
=======
# Projects
projects I have contributed
>>>>>>> fd18f164622d9792dfb6aab0e0a4d54109e6cd02
