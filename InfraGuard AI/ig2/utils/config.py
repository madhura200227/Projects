"""
utils/config.py – Centralised settings with validation
All values loaded from .env; sensible defaults for local dev.
"""

from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Optional
import os


class Settings(BaseSettings):
    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    RATE_LIMIT_PER_MIN: int = 120
    ENVIRONMENT: str = "development"  # development | staging | production

    # ── CORS ──────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://infraguard.vercel.app",
        "*",  # remove in production
    ]

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://infraguard:infraguard_secret@localhost:5432/infraguard"
    POSTGRES_PASSWORD: str = "infraguard_secret"
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_TTL_TRAFFIC: int = 300       # 5 minutes
    REDIS_TTL_AQI: int = 600           # 10 minutes
    REDIS_TTL_WEATHER: int = 900       # 15 minutes
    REDIS_TTL_FLOOD: int = 3600        # 1 hour
    REDIS_TTL_PREDICTION: int = 1800   # 30 minutes
    REDIS_TTL_SCRAPE: int = 86400      # 24 hours

    # ── Auth ──────────────────────────────────────────────────────────────────
    JWT_SECRET: str = "change-me-in-production-use-256bit-key"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60

    # ── Traffic APIs ──────────────────────────────────────────────────────────
    TOMTOM_API_KEY: str = ""
    HERE_API_KEY: str = ""
    GOOGLE_MAPS_API_KEY: str = ""
    # TomTom free tier: 2,500 req/day → used for real-time flow
    # HERE free tier: 250,000 req/month → used for incidents
    # Google Maps: used for geocoding & Places API

    # ── AQI / Weather APIs ────────────────────────────────────────────────────
    OPENWEATHER_API_KEY: str = ""
    # Free tier: 60 calls/min, 1M calls/month
    # Endpoints: /air_pollution, /weather, /forecast
    WAQI_API_KEY: str = ""
    # World Air Quality Index – free for non-commercial
    # Endpoint: /feed/geo:{lat};{lon}/

    # ── Flood / Disaster APIs ─────────────────────────────────────────────────
    NOAA_API_KEY: str = ""
    # NOAA Climate Data Online API – free with registration
    # Endpoint: https://www.ncdc.noaa.gov/cdo-web/api/v2/

    NASA_EARTHDATA_TOKEN: str = ""
    # NASA EarthData – free; needed for GFMS flood monitoring
    # https://urs.earthdata.nasa.gov/

    RELIEFWEB_API_KEY: str = ""
    # ReliefWeb Disasters API – free, no key needed but for higher limits

    # ── Geospatial / Satellite APIs ───────────────────────────────────────────
    GOOGLE_EARTH_ENGINE_KEY: str = ""
    # Google Earth Engine – free for research
    SENTINEL_HUB_CLIENT_ID: str = ""
    SENTINEL_HUB_CLIENT_SECRET: str = ""
    # Sentinel Hub – free tier: 30,000 processing units/month

    # ── OSM / OpenData ────────────────────────────────────────────────────────
    # OpenStreetMap Overpass API – completely free, no key
    OVERPASS_API_URL: str = "https://overpass-api.de/api/interpreter"
    # Fallback: "https://overpass.kumi.systems/api/interpreter"

    # ── World Bank / IMF Open Data ────────────────────────────────────────────
    # World Bank Open Data API – completely free, no key required
    WORLD_BANK_API_URL: str = "https://api.worldbank.org/v2"

    # ── India Open Data (data.gov.in) ─────────────────────────────────────────
    DATA_GOV_IN_API_KEY: str = ""
    # Free registration at data.gov.in; needed for Indian infrastructure datasets

    # ── ML paths ──────────────────────────────────────────────────────────────
    MODEL_PATH: str = "ml/weights/xgboost_model.json"
    CNN_MODEL_PATH: str = "ml/weights/crack_cnn.pt"
    SYNTHETIC_DATA_PATH: str = "data/synthetic/training_data.csv"

    # ── Celery ────────────────────────────────────────────────────────────────
    CELERY_BROKER: str = "redis://localhost:6379/0"
    CELERY_BACKEND: str = "redis://localhost:6379/1"

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/infraguard.log"

    # ── Scraper ───────────────────────────────────────────────────────────────
    SCRAPER_TIMEOUT_SECONDS: int = 15
    SCRAPER_MAX_RETRIES: int = 3
    SCRAPER_USER_AGENT: str = (
        "InfraGuardAI/2.0 (+https://infraguard.ai/bot; research@infraguard.ai)"
    )

    class Config:
        env_file = ".env"
        case_sensitive = True

    @property
    def has_traffic_api(self) -> bool:
        return bool(self.TOMTOM_API_KEY or self.HERE_API_KEY or self.GOOGLE_MAPS_API_KEY)

    @property
    def has_aqi_api(self) -> bool:
        return bool(self.OPENWEATHER_API_KEY or self.WAQI_API_KEY)

    @property
    def has_weather_api(self) -> bool:
        return bool(self.OPENWEATHER_API_KEY)

    @property
    def configured_apis(self) -> List[str]:
        apis = []
        if self.TOMTOM_API_KEY: apis.append("TomTom Traffic")
        if self.HERE_API_KEY: apis.append("HERE Incidents")
        if self.GOOGLE_MAPS_API_KEY: apis.append("Google Maps")
        if self.OPENWEATHER_API_KEY: apis.append("OpenWeather")
        if self.WAQI_API_KEY: apis.append("WAQI AQI")
        if self.NOAA_API_KEY: apis.append("NOAA Flood")
        if self.NASA_EARTHDATA_TOKEN: apis.append("NASA EarthData")
        if self.DATA_GOV_IN_API_KEY: apis.append("data.gov.in")
        return apis


settings = Settings()
