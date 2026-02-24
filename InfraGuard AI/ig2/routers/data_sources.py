"""
routers/data_sources.py – /api/v1/data endpoints
Live data fetch status, API health, and city data snapshots.
"""

import logging
from datetime import datetime
from fastapi import APIRouter, Query, HTTPException

from ml.data_fetcher import DataAggregator
from utils.config import settings

logger = logging.getLogger("infraguard.data")
router = APIRouter()

aggregator = DataAggregator()
VALID_CITIES = {"Mumbai", "Pune", "New York", "Tokyo"}

CITY_COORDS = {
    "Mumbai":   (19.076, 72.877),
    "Pune":     (18.520, 73.856),
    "New York": (40.712, -74.006),
    "Tokyo":    (35.689, 139.692),
}


@router.get("/live/{city}", summary="Fetch live data snapshot for a city")
async def get_live_data(city: str):
    """
    Fetches real-time data from all configured APIs for a city:
    traffic, AQI, weather, flood risk, infrastructure, seismic.
    Returns raw feature values ready for model input.
    """
    if city not in VALID_CITIES:
        raise HTTPException(400, f"Unknown city. Valid: {sorted(VALID_CITIES)}")

    lat, lon = CITY_COORDS[city]
    agg = await aggregator.fetch_all(lat, lon)

    return {
        "city": city,
        "lat": lat,
        "lon": lon,
        "features": agg["features"],
        "sources_used": agg["sources_used"],
        "fetched_at": agg["fetched_at"],
    }


@router.get("/apis/status", summary="Check which APIs are configured and healthy")
async def api_status():
    """
    Returns status of all external API integrations.
    Shows which are configured (have API keys) and which are free/no-key.
    """
    statuses = await aggregator.get_api_status()

    configured_count = sum(1 for s in statuses if s["configured"])
    return {
        "apis": statuses,
        "total_apis": len(statuses),
        "configured_count": configured_count,
        "configured_apis": settings.configured_apis,
        "timestamp": datetime.utcnow().isoformat(),
        "setup_guide": {
            "free_no_key": ["OSM Overpass", "ReliefWeb Disasters", "USGS Seismic", "World Bank Open Data"],
            "free_with_registration": [
                {"name": "OpenWeatherMap", "url": "https://openweathermap.org/api", "limit": "60 calls/min"},
                {"name": "WAQI AQI", "url": "https://aqicn.org/api/", "limit": "Unlimited for non-commercial"},
                {"name": "TomTom Traffic", "url": "https://developer.tomtom.com", "limit": "2,500 req/day"},
                {"name": "HERE Traffic", "url": "https://developer.here.com", "limit": "250,000 req/month"},
                {"name": "NOAA CDO", "url": "https://www.ncdc.noaa.gov/cdo-web/webservices/v2", "limit": "1,000 req/day"},
                {"name": "data.gov.in", "url": "https://data.gov.in/api-access", "limit": "Varies"},
            ],
        },
    }


@router.get("/traffic/{city}", summary="Fetch live traffic data for a city")
async def get_traffic(city: str):
    if city not in VALID_CITIES:
        raise HTTPException(400, f"Unknown city")
    lat, lon = CITY_COORDS[city]
    result = await aggregator.traffic.fetch(lat, lon)
    return {"city": city, **result}


@router.get("/aqi/{city}", summary="Fetch live AQI data for a city")
async def get_aqi(city: str):
    if city not in VALID_CITIES:
        raise HTTPException(400, f"Unknown city")
    lat, lon = CITY_COORDS[city]
    result = await aggregator.aqi.fetch(lat, lon)
    return {"city": city, **result}


@router.get("/weather/{city}", summary="Fetch live weather data for a city")
async def get_weather(city: str):
    if city not in VALID_CITIES:
        raise HTTPException(400, f"Unknown city")
    lat, lon = CITY_COORDS[city]
    result = await aggregator.weather.fetch(lat, lon)
    return {"city": city, **result}


@router.get("/flood/{city}", summary="Fetch flood risk data for a city")
async def get_flood(city: str):
    if city not in VALID_CITIES:
        raise HTTPException(400, f"Unknown city")
    lat, lon = CITY_COORDS[city]
    result = await aggregator.flood.fetch(lat, lon)
    return {"city": city, **result}


@router.get("/seismic/{city}", summary="Fetch seismic risk data from USGS")
async def get_seismic(city: str):
    if city not in VALID_CITIES:
        raise HTTPException(400, f"Unknown city")
    lat, lon = CITY_COORDS[city]
    result = await aggregator.seismic.fetch(lat, lon)
    return {"city": city, **result}
