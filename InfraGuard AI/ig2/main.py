"""
InfraGuard AI – FastAPI Backend v2
====================================
Multimodal infrastructure failure prediction with real-time data.

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Docker:
  docker compose up
"""

import asyncio
import logging
import logging.handlers
import time
from collections import defaultdict
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routers import predict, data_sources, alerts, reports, scraper
from utils.cache import init_redis
from utils.config import settings
from ml.model import InfraGuardModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            settings.LOG_FILE, maxBytes=10_485_760, backupCount=5
        ) if settings.LOG_FILE else logging.NullHandler(),
    ],
)
logger = logging.getLogger("infraguard")

# ── Startup / Shutdown ────────────────────────────────────────────────────────
_background_tasks = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 InfraGuard AI v2 starting up…")
    logger.info(f"   Configured APIs: {settings.configured_apis or ['OSM (free)', 'ReliefWeb (free)', 'USGS (free)', 'WorldBank (free)']}")

    # Load ML model
    app.state.model = InfraGuardModel()
    app.state.model.load()
    logger.info("✅ Ensemble model loaded (XGBoost + CNN, 15 features)")

    # Connect Redis
    app.state.redis = await init_redis()
    logger.info("✅ Redis cache initialised")

    # Start background alert polling
    alert_task = asyncio.create_task(alerts.poll_alerts_background())
    _background_tasks.append(alert_task)
    logger.info("✅ Background alert polling started (10min interval)")

    # Initial alert fetch for all cities
    asyncio.create_task(_initial_alert_fetch())

    yield

    # Shutdown
    logger.info("🛑 InfraGuard AI shutting down…")
    for task in _background_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    await app.state.redis.close()


async def _initial_alert_fetch():
    """Fetch alerts for all cities on startup."""
    await asyncio.sleep(5)  # Give Redis a moment to settle
    for city in ["Mumbai", "Pune", "New York", "Tokyo"]:
        try:
            city_alerts = await alerts.alert_engine.generate_city_alerts(city)
            await alerts.alert_cache.set(f"recent:{city}", city_alerts, ttl=3600)
            logger.info(f"✅ Initial alerts fetched for {city}: {len(city_alerts)} alerts")
        except Exception as e:
            logger.warning(f"Initial alert fetch failed for {city}: {e}")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="InfraGuard AI",
    description="""
## 🏗️ InfraGuard AI – Infrastructure Failure Prediction API v2

Predicts infrastructure failure probability using multimodal real-time data:

### Data Sources (8 free APIs, no key required)
- **OSM Overpass** – Road network, bridges, waterways
- **USGS Seismic** – Real-time seismic hazard maps
- **ReliefWeb** – Flood disaster events database
- **World Bank** – Transport & environment indicators

### Data Sources (4 free APIs, free registration)
- **TomTom Traffic v4** – Real-time flow & congestion
- **HERE Traffic v7** – Incidents & flow data
- **WAQI** – World Air Quality Index (AQI)
- **OpenWeatherMap** – Weather + air pollution

### Model
- XGBoost tabular (15 features) + CNN image analysis
- Monte Carlo uncertainty quantification (95% CI)
- SHAP explainability values
- Infra-type specific weight adjustments (roads/bridges/pipelines)

### Real-time
- WebSocket alert streaming at `/api/v1/alerts/ws`
- Alerts updated every 10 minutes
    """,
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Prediction",   "description": "Failure probability prediction endpoints"},
        {"name": "Data Sources", "description": "Real-time data fetch endpoints"},
        {"name": "Alerts",       "description": "Real-time alert streaming (REST + WebSocket)"},
        {"name": "Reports",      "description": "City-level infrastructure reports"},
        {"name": "Scraper",      "description": "On-demand OSM and government data scraping"},
        {"name": "Health",       "description": "Server health and version info"},
    ],
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    ms = (time.time() - start) * 1000
    response.headers["X-Process-Time"] = f"{ms:.1f}ms"
    response.headers["X-InfraGuard-Version"] = "2.1.0"
    return response


# ── Rate limiting (per-IP, sliding window) ────────────────────────────────────
_rate_buckets: dict = defaultdict(lambda: {"count": 0, "reset": time.time() + 60})


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    ip = getattr(request.client, "host", "unknown")
    bucket = _rate_buckets[ip]
    if time.time() > bucket["reset"]:
        bucket["count"] = 0
        bucket["reset"] = time.time() + 60
    bucket["count"] += 1
    if bucket["count"] > settings.RATE_LIMIT_PER_MIN:
        return JSONResponse(
            {"error": "Rate limit exceeded", "retry_after_seconds": int(bucket["reset"] - time.time())},
            status_code=429,
            headers={"Retry-After": str(int(bucket["reset"] - time.time()))},
        )
    return await call_next(request)


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(predict.router,      prefix="/api/v1/predict",  tags=["Prediction"])
app.include_router(data_sources.router, prefix="/api/v1/data",     tags=["Data Sources"])
app.include_router(alerts.router,       prefix="/api/v1/alerts",   tags=["Alerts"])
app.include_router(reports.router,      prefix="/api/v1/reports",  tags=["Reports"])
app.include_router(scraper.router,      prefix="/api/v1/scraper",  tags=["Scraper"])


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    """Quick health check for load balancers."""
    return {
        "status": "healthy",
        "version": "2.1.0",
        "model": "XGBoost+CNN Ensemble (15 features)",
        "cities_monitored": ["Mumbai", "Pune", "New York", "Tokyo"],
        "configured_apis": settings.configured_apis,
        "free_apis_active": ["OSM Overpass", "ReliefWeb", "USGS Seismic", "World Bank"],
        "timestamp": time.time(),
    }


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "InfraGuard AI v2 API",
        "docs": "/docs",
        "health": "/health",
        "prediction": "/api/v1/predict/?lat=19.076&lon=72.877&infra_type=roads",
        "websocket_alerts": "ws://localhost:8000/api/v1/alerts/ws?city=Mumbai",
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
    )
