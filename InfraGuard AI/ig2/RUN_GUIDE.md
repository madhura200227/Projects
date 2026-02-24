# 🏗️ InfraGuard AI v2 — Run Guide

> **Multimodal Infrastructure Failure Prediction with Real-Time Data**

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start (Local — No Docker)](#quick-start-local--no-docker)
- [Docker Compose (Full Stack)](#docker-compose-full-stack)
- [Environment Variables](#environment-variables)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement       | Version   | Notes                                              |
|-------------------|-----------|----------------------------------------------------|
| **Python**        | 3.11+     | Required for local development                     |
| **pip**           | latest    | Comes with Python                                  |
| **Docker**        | 20+       | Only if using Docker Compose                       |
| **Docker Compose**| v2+       | Only if using Docker Compose                       |
| **Redis**         | 7.x       | Optional — app falls back to in-memory cache       |
| **PostgreSQL**    | 16+       | Optional for local dev — needed for data persistence|

---

## Quick Start (Local — No Docker)

This is the fastest way to run the application for development. Redis and PostgreSQL are **optional** — the app gracefully falls back to in-memory caching if they are unavailable.

### 1. Navigate to the Project Directory

```powershell
cd "c:\Users\anish\OneDrive\Desktop\InfraGuard AI\ig2"
```

### 2. Create a Virtual Environment (first time only)

```powershell
python -m venv venv
```

### 3. Activate the Virtual Environment

```powershell
.\venv\Scripts\activate
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

> **Note:** `torch` and `torchvision` are commented out in `requirements.txt`. Uncomment them only if you have trained CNN model weights (`ml/weights/crack_cnn.pt`).

### 5. Configure Environment Variables

Copy the template and edit as needed:

```powershell
copy .env.template .env
```

Open `.env` and fill in your API keys. The following APIs work **without any key**:
- OSM Overpass
- ReliefWeb
- USGS Seismic
- World Bank Open Data

For full functionality, add these **free-tier** API keys:
- **TomTom** — traffic data ([Sign up](https://developer.tomtom.com/user/register))
- **OpenWeatherMap** — weather & AQI ([Sign up](https://home.openweathermap.org/users/sign_up))
- **WAQI** — air quality ([Sign up](https://aqicn.org/api/))
- **NOAA** — flood data ([Get token](https://www.ncdc.noaa.gov/cdo-web/token))

### 6. Create Required Directories

```powershell
mkdir logs -Force
mkdir ml\weights -Force
```

### 7. Run the Application

```powershell
.\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or with the standard activate:

```powershell
# If venv is activated:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or using Python directly:

```powershell
.\venv\Scripts\python.exe main.py
```

### 8. Verify the App is Running

Open your browser and navigate to:

| URL                                      | Description            |
|------------------------------------------|------------------------|
| http://localhost:8000                     | Root / welcome message |
| http://localhost:8000/health              | Health check endpoint  |
| http://localhost:8000/docs                | Swagger UI (interactive API docs) |
| http://localhost:8000/redoc               | ReDoc (alternative API docs) |

---

## Docker Compose (Full Stack)

This starts the entire stack: **FastAPI + Redis + PostgreSQL + Celery Workers + Celery Beat**.

### 1. Build and Start All Services

```powershell
docker compose up --build
```

### 2. Start in Detached Mode (Background)

```powershell
docker compose up --build -d
```

### 3. View Logs

```powershell
docker compose logs -f api
```

### 4. Stop All Services

```powershell
docker compose down
```

### 5. Stop and Remove All Data (Clean Restart)

```powershell
docker compose down -v
```

### 6. Start with Monitoring (Flower for Celery)

```powershell
docker compose --profile monitoring up --build
```

Then open **http://localhost:5555** for the Celery Flower dashboard.

### Service Ports

| Service           | Port  | Description                       |
|-------------------|-------|-----------------------------------|
| **API (FastAPI)** | 8000  | Main application                  |
| **Redis**         | 6379  | Cache + message broker            |
| **PostgreSQL**    | 5432  | Persistent database               |
| **Flower**        | 5555  | Celery task monitor (optional)    |

---

## Environment Variables

All configuration is managed via the `.env` file. Key variables:

| Variable                | Default                        | Description                       |
|-------------------------|--------------------------------|-----------------------------------|
| `HOST`                  | `0.0.0.0`                      | Server bind address               |
| `PORT`                  | `8000`                         | Server port                       |
| `DEBUG`                 | `false`                        | Enable debug mode & auto-reload   |
| `REDIS_URL`             | `redis://localhost:6379`       | Redis connection URL              |
| `DATABASE_URL`          | `postgresql+asyncpg://...`     | PostgreSQL connection URL         |
| `TOMTOM_API_KEY`        | *(empty)*                      | TomTom traffic API key            |
| `OPENWEATHER_API_KEY`   | *(empty)*                      | OpenWeatherMap API key            |
| `WAQI_API_KEY`          | *(empty)*                      | World Air Quality Index token     |
| `NOAA_API_KEY`          | *(empty)*                      | NOAA climate data token           |
| `LOG_LEVEL`             | `INFO`                         | Logging level                     |
| `RATE_LIMIT_PER_MIN`    | `120`                          | Max API requests per minute per IP|

---

## API Endpoints

### Core Endpoints

| Method | Endpoint                              | Description                                |
|--------|---------------------------------------|--------------------------------------------|
| GET    | `/`                                   | Welcome message with useful links          |
| GET    | `/health`                             | Health check for load balancers            |
| GET    | `/docs`                               | Swagger interactive API documentation      |
| GET    | `/redoc`                              | ReDoc API documentation                    |

### Prediction

| Method | Endpoint                              | Description                                |
|--------|---------------------------------------|--------------------------------------------|
| GET    | `/api/v1/predict/`                    | Predict infrastructure failure probability |
|        |                                       | Params: `lat`, `lon`, `infra_type`         |

### Data Sources

| Method | Endpoint                              | Description                                |
|--------|---------------------------------------|--------------------------------------------|
| GET    | `/api/v1/data/...`                    | Fetch real-time data from external APIs    |

### Alerts

| Method | Endpoint                              | Description                                |
|--------|---------------------------------------|--------------------------------------------|
| GET    | `/api/v1/alerts/...`                  | Get infrastructure alerts                  |
| WS     | `/api/v1/alerts/ws?city=Mumbai`       | WebSocket real-time alert streaming        |

### Reports

| Method | Endpoint                              | Description                                |
|--------|---------------------------------------|--------------------------------------------|
| GET    | `/api/v1/reports/...`                 | City-level infrastructure reports          |

### Scraper

| Method | Endpoint                              | Description                                |
|--------|---------------------------------------|--------------------------------------------|
| GET    | `/api/v1/scraper/...`                 | On-demand OSM and government data scraping |

---

## Troubleshooting

### Unicode/Emoji Errors on Windows Console

You may see errors like:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f680'
```

**Fix:** Set the console encoding to UTF-8 before running:
```powershell
$env:PYTHONIOENCODING = "utf-8"
chcp 65001
.\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Redis Not Available

The app starts successfully without Redis. You will see a warning:
```
⚠️  Redis unavailable. Using in-memory fallback.
```

This is expected for local development. Install Redis locally or use Docker Compose for full functionality.

### Port Already in Use

```
ERROR: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000)
```

**Fix:** Kill the process using port 8000 or use a different port:
```powershell
# Find the process
netstat -ano | findstr :8000

# Kill it (replace <PID> with actual PID)
taskkill /PID <PID> /F

# Or use a different port
uvicorn main:app --host 0.0.0.0 --port 8001
```

### Missing Dependencies

If you get `ModuleNotFoundError`:
```powershell
.\venv\Scripts\pip.exe install -r requirements.txt
```

### Database Connection Errors

For local development without Docker, PostgreSQL is optional. The app may log database connection errors but will still serve prediction endpoints using cached/computed data.

---

## Running Tests

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v
```

---

## Project Structure

```
ig2/
├── main.py              # FastAPI app entry point
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container image definition
├── docker-compose.yml   # Full stack orchestration
├── .env                 # Environment variables (API keys, config)
├── .env.template        # Template for .env
├── routers/             # API route handlers
│   ├── predict.py       #   Failure prediction endpoints
│   ├── data_sources.py  #   External data fetch endpoints
│   ├── alerts.py        #   Alert system + WebSocket
│   ├── reports.py       #   City reports
│   └── scraper.py       #   OSM/gov data scraping
├── ml/                  # Machine learning
│   ├── model.py         #   XGBoost + CNN ensemble model
│   ├── data_fetcher.py  #   Multi-API data aggregator
│   └── weights/         #   Model weight files
├── utils/               # Shared utilities
│   ├── config.py        #   Pydantic settings
│   ├── cache.py         #   Redis cache + in-memory fallback
│   └── tasks.py         #   Celery background tasks
├── scrapers/            # Data scrapers
│   ├── osm_scraper.py   #   OpenStreetMap data
│   └── gov_scraper.py   #   Government data portals
├── data/                # Data storage
│   └── synthetic/       #   Synthetic training data
├── logs/                # Application logs
└── tests/               # Test suite
```
