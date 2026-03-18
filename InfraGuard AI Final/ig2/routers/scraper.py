"""
routers/scraper.py – /api/v1/scraper endpoints
On-demand scraping of OSM, government portals, and open datasets.
"""

import logging
from datetime import datetime
from fastapi import APIRouter, Query, HTTPException

from scrapers.osm_scraper import scrape_city_infrastructure
from scrapers.gov_scraper import scrape_all_sources, WorldBankScraper, ReliefWebScraper
from utils.cache import scrape_cache

logger = logging.getLogger("infraguard.scraper")
router = APIRouter()

VALID_CITIES = {"Mumbai", "Pune", "New York", "Tokyo"}


@router.get("/city/{city}", summary="Scrape all infrastructure data for a city")
async def scrape_city(
    city: str,
    force: bool = Query(False, description="Force refresh even if cached"),
):
    """
    Triggers parallel scraping of:
    - OpenStreetMap (roads, bridges, waterways)
    - World Bank open data (vehicles, road network, PM2.5)
    - ReliefWeb disasters API (flood events)
    - NOAA CDO (rainfall records, if API key set)
    - FHWA/USDOT (USA only, if city=New York)
    - data.gov.in (India only, if city=Mumbai/Pune and API key set)

    Results are cached for 24 hours. Use `?force=true` to bypass cache.
    """
    if city not in VALID_CITIES:
        raise HTTPException(400, f"Unknown city. Valid: {sorted(VALID_CITIES)}")

    cache_key = f"full_scrape:{city.lower().replace(' ','_')}"
    if not force:
        cached = await scrape_cache.get(cache_key)
        if cached:
            return {**cached, "source": "redis_cache"}

    # Run OSM + government scrapers in parallel
    import asyncio
    osm_data, gov_data = await asyncio.gather(
        scrape_city_infrastructure(city),
        scrape_all_sources(city),
        return_exceptions=True,
    )

    result = {
        "city": city,
        "osm": osm_data if isinstance(osm_data, dict) else {"error": str(osm_data)},
        "government": gov_data if isinstance(gov_data, dict) else {"error": str(gov_data)},
        "scraped_at": datetime.utcnow().isoformat(),
        "source": "fresh_scrape",
    }

    await scrape_cache.set(cache_key, result)
    return result


@router.get("/osm/{city}", summary="Scrape OpenStreetMap infrastructure data")
async def scrape_osm(city: str):
    """
    Queries OSM Overpass API for roads, bridges, and waterways.
    Free — no API key required. Results cached 24 hours.
    """
    if city not in VALID_CITIES:
        raise HTTPException(400, f"Unknown city. Valid: {sorted(VALID_CITIES)}")

    return await scrape_city_infrastructure(city)


@router.get("/worldbank/{city}", summary="Fetch World Bank transport indicators")
async def scrape_worldbank(city: str):
    """
    Fetches transport and environment indicators from World Bank Open Data API.
    Free — no API key required. Covers motor vehicles, road network, PM2.5, road deaths.
    """
    if city not in VALID_CITIES:
        raise HTTPException(400, f"Unknown city. Valid: {sorted(VALID_CITIES)}")

    wb = WorldBankScraper()
    return await wb.scrape(city)


@router.get("/floods/{city}", summary="Get flood event history from ReliefWeb")
async def scrape_floods(
    city: str,
    years: int = Query(5, ge=1, le=10, description="Years of history to fetch"),
):
    """
    Queries ReliefWeb Disasters API for flood events.
    Free — no API key required.
    """
    if city not in VALID_CITIES:
        raise HTTPException(400, f"Unknown city. Valid: {sorted(VALID_CITIES)}")

    rw = ReliefWebScraper()
    return await rw.scrape(city, years=years)


@router.delete("/cache", summary="Clear scrape cache (admin)")
async def clear_scrape_cache(
    city: str = Query(None, description="Clear cache for specific city, or all if not provided"),
):
    """Clear cached scrape results. Forces fresh data on next request."""
    import asyncio
    from utils.cache import get_redis

    r = await get_redis()
    pattern = f"infraguard:scrape:*{city.lower().replace(' ','_') if city else ''}*"

    try:
        keys = await r.keys(pattern)
        if keys:
            await r.delete(*keys)
            deleted = len(keys)
        else:
            deleted = 0
    except Exception:
        deleted = 0

    return {
        "deleted_keys": deleted,
        "pattern": pattern,
        "timestamp": datetime.utcnow().isoformat(),
    }
