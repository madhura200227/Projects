"""
scrapers/gov_scraper.py – Government Open Data portal scrapers
==============================================================
Scrapes / queries official open data APIs:

India:
  - data.gov.in API: road accident statistics, infrastructure spend
  - National Highways Authority of India (public datasets)
  - IMD (India Met Department) for flood/rainfall records

USA:
  - data.transportation.gov: FHWA highway statistics
  - Bureau of Transportation Statistics
  - NOAA GHCND: precipitation events

Japan:
  - e-Stat (Japan's official statistics) API
  - MLIT (Ministry of Land) road statistics

Global:
  - World Bank transport indicators
  - OpenData Soft public datasets
"""

import asyncio
import httpx
import logging
import csv
import io
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from utils.config import settings
from utils.cache import scrape_cache

logger = logging.getLogger("infraguard.scraper.gov")


def _client(timeout: float = 15.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=timeout,
        headers={
            "User-Agent": settings.SCRAPER_USER_AGENT,
            "Accept": "application/json, text/csv, */*",
        },
        follow_redirects=True,
    )


class WorldBankScraper:
    """
    World Bank Open Data API (free, no key):
    Fetches transport and infrastructure indicators.

    Indicators used:
      IS.VEH.NVEH.P3  – Motor vehicles per 1000 people
      IS.ROD.TOTL.KM  – Road network total length (km)
      IS.ROD.PAVE.ZS  – Paved roads (% of total)
      EN.ATM.PM25.MC.M3 – PM2.5 air pollution mean exposure
      SH.STA.TRAF.P5  – Road traffic deaths per 100,000
    """

    BASE = "https://api.worldbank.org/v2"

    COUNTRY_ISO3 = {"Mumbai": "IND", "Pune": "IND", "New York": "USA", "Tokyo": "JPN"}

    INDICATORS = {
        "IS.VEH.NVEH.P3":    "motor_vehicles_per_1000",
        "IS.ROD.TOTL.KM":    "road_network_km",
        "IS.ROD.PAVE.ZS":    "paved_roads_pct",
        "EN.ATM.PM25.MC.M3": "pm25_annual_mean",
        "SH.STA.TRAF.P5":    "road_deaths_per_100k",
        "EN.CO2.TRAN.ZS":    "transport_co2_share",
    }

    async def scrape(self, city: str) -> Dict[str, Any]:
        cache_key = f"worldbank:{city.lower().replace(' ','_')}"
        cached = await scrape_cache.get(cache_key)
        if cached:
            return {**cached, "source": "redis_cache"}

        iso3 = self.COUNTRY_ISO3.get(city)
        if not iso3:
            return {"error": f"No ISO3 code for {city}"}

        results: Dict[str, Any] = {"city": city, "iso3": iso3}

        tasks = [self._fetch_indicator(iso3, ind) for ind in self.INDICATORS]
        values = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (indicator, field) in enumerate(self.INDICATORS.items()):
            v = values[i]
            if isinstance(v, (int, float)):
                results[field] = round(v, 2)

        results["scraped_at"] = datetime.utcnow().isoformat()
        results["source"] = "worldbank_api"

        await scrape_cache.set(cache_key, results)
        logger.info(f"World Bank data scraped for {city}/{iso3}: {len(results)} fields")
        return results

    async def _fetch_indicator(self, iso3: str, indicator: str) -> Optional[float]:
        url = f"{self.BASE}/country/{iso3}/indicator/{indicator}"
        try:
            async with _client() as c:
                r = await c.get(url, params={"format": "json", "mrv": 5, "per_page": 5})
                r.raise_for_status()
                data = r.json()
                if len(data) < 2 or not data[1]:
                    return None
                for item in data[1]:
                    if item.get("value") is not None:
                        return float(item["value"])
        except Exception as e:
            logger.debug(f"WB indicator {indicator} for {iso3}: {e}")
        return None


class NOAAScraper:
    """
    NOAA Climate Data Online API.
    Requires free API key from ncdc.noaa.gov

    Gets precipitation events (daily GHCND) to count
    flood-likely days (rain > threshold) over past 5 years.
    """

    BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"

    CITY_STATION_IDS = {
        "Mumbai":   "GHCND:IND01326",  # Santacruz airport
        "Pune":     "GHCND:IND01225",  # Pune airport
        "New York": "GHCND:USW00094728", # Central Park
        "Tokyo":    "GHCND:JA000047662", # Tokyo
    }

    HEAVY_RAIN_MM = 50  # >50mm/day = likely flood conditions

    async def scrape(self, city: str) -> Dict[str, Any]:
        cache_key = f"noaa:{city.lower().replace(' ','_')}"
        cached = await scrape_cache.get(cache_key)
        if cached:
            return {**cached, "source": "redis_cache"}

        if not settings.NOAA_API_KEY:
            logger.info(f"NOAA API key not configured, skipping NOAA scrape for {city}")
            return {"city": city, "source": "not_configured", "note": "Set NOAA_API_KEY in .env"}

        station = self.CITY_STATION_IDS.get(city)
        if not station:
            return {"error": f"No NOAA station for {city}"}

        end_date   = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%d")

        try:
            async with _client(timeout=20) as c:
                # Fetch daily precipitation records
                r = await c.get(f"{self.BASE}/data", params={
                    "datasetid": "GHCND",
                    "datatypeid": "PRCP",
                    "stationid": station,
                    "startdate": start_date,
                    "enddate": end_date,
                    "limit": 1000,
                    "units": "metric",
                }, headers={"token": settings.NOAA_API_KEY})
                r.raise_for_status()
                d = r.json()

            records = d.get("results", [])
            if not records:
                return {"city": city, "source": "noaa", "note": "No records found"}

            # PRCP in tenths of mm → convert
            daily_mm = [float(rec["value"]) / 10 for rec in records]
            heavy_days = sum(1 for mm in daily_mm if mm >= self.HEAVY_RAIN_MM)
            annual_mm  = sum(daily_mm) / 5  # average over 5 years
            max_1d     = max(daily_mm) if daily_mm else 0

            result = {
                "city": city,
                "station_id": station,
                "period_years": 5,
                "heavy_rain_days_5yr": heavy_days,
                "annual_rainfall_mm_avg": round(annual_mm, 0),
                "max_1day_rainfall_mm": round(max_1d, 1),
                "flood_events_estimate": round(heavy_days / 15),  # ~15 heavy days per event
                "scraped_at": datetime.utcnow().isoformat(),
                "source": "noaa_cdo",
            }
            await scrape_cache.set(cache_key, result)
            logger.info(f"NOAA scraped for {city}: {heavy_days} heavy rain days, annual={annual_mm:.0f}mm")
            return result

        except httpx.HTTPStatusError as e:
            logger.warning(f"NOAA HTTP {e.response.status_code} for {city}: {e}")
        except Exception as e:
            logger.warning(f"NOAA scrape error for {city}: {e}")

        return {"city": city, "source": "noaa_error", "error": "Data fetch failed"}


class USTransportationScraper:
    """
    data.transportation.gov (US DOT open data):
    Fetches highway condition, bridge inspection, and safety data.
    Uses Socrata API (free, no key for public datasets).
    """

    BASE = "https://data.transportation.gov/resource"

    # Highway Statistics summary (lane-miles, pavement condition)
    HIGHWAY_STATS_ID = "rzwg-fyv9"
    # National Bridge Inventory (FHWA)
    NBI_ID = "vfxm-3b3p"

    async def scrape_highway_stats(self, state: str = "NY") -> Dict[str, Any]:
        cache_key = f"usdot_hwy:{state.lower()}"
        cached = await scrape_cache.get(cache_key)
        if cached:
            return {**cached, "source": "redis_cache"}

        try:
            async with _client() as c:
                r = await c.get(
                    f"{self.BASE}/{self.HIGHWAY_STATS_ID}.json",
                    params={"$where": f"state_code='{state}'", "$limit": 5, "$order": "year DESC"},
                )
                r.raise_for_status()
                records = r.json()

                if not records:
                    return {"state": state, "source": "usdot", "note": "No data"}

                latest = records[0]
                result = {
                    "state": state,
                    "year": latest.get("year"),
                    "lane_miles": latest.get("lane_miles"),
                    "vmts_millions": latest.get("vehicle_miles_traveled_millions"),
                    "source": "usdot_open_data",
                    "scraped_at": datetime.utcnow().isoformat(),
                }
                await scrape_cache.set(cache_key, result)
                return result
        except Exception as e:
            logger.warning(f"US DOT scrape error: {e}")
        return {"state": state, "source": "usdot_error"}

    async def scrape_bridge_data(self, state: str = "NY", limit: int = 20) -> Dict[str, Any]:
        """Scrape National Bridge Inventory for a state."""
        cache_key = f"nbi:{state.lower()}"
        cached = await scrape_cache.get(cache_key)
        if cached:
            return {**cached, "source": "redis_cache"}

        try:
            async with _client(timeout=20) as c:
                r = await c.get(
                    f"{self.BASE}/{self.NBI_ID}.json",
                    params={
                        "$where": f"state_code='{state}' AND str_rating < 5",
                        "$limit": limit,
                        "$order": "str_rating ASC",
                    },
                )
                r.raise_for_status()
                bridges = r.json()

                if not bridges:
                    return {"state": state, "source": "nbi", "bridges": []}

                result = {
                    "state": state,
                    "deficient_bridges_count": len(bridges),
                    "bridges": [
                        {
                            "name": b.get("bridge_name", "Unknown"),
                            "year_built": b.get("year_built"),
                            "structural_rating": b.get("str_rating"),
                            "avg_daily_traffic": b.get("aadt"),
                        }
                        for b in bridges[:10]
                    ],
                    "scraped_at": datetime.utcnow().isoformat(),
                    "source": "nbi_fhwa",
                }
                await scrape_cache.set(cache_key, result)
                logger.info(f"NBI scraped for {state}: {len(bridges)} deficient bridges")
                return result
        except Exception as e:
            logger.warning(f"NBI scrape error: {e}")
        return {"state": state, "source": "nbi_error"}


class IndiaGovScraper:
    """
    data.gov.in API (Indian government open data):
    Requires free API key from data.gov.in
    """

    BASE = "https://api.data.gov.in/resource"

    # Road accident data
    ACCIDENT_RESOURCE = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
    # National Highways construction
    NH_RESOURCE = "0a3ab6b3-2d27-4e7c-a4c4-0d62e13ce4e4"

    async def scrape(self, state: str = "Maharashtra") -> Dict[str, Any]:
        cache_key = f"datagov_in:{state.lower()}"
        cached = await scrape_cache.get(cache_key)
        if cached:
            return {**cached, "source": "redis_cache"}

        if not settings.DATA_GOV_IN_API_KEY:
            return {
                "state": state,
                "source": "not_configured",
                "note": "Set DATA_GOV_IN_API_KEY in .env (free at data.gov.in)"
            }

        try:
            async with _client() as c:
                r = await c.get(
                    f"{self.BASE}/{self.ACCIDENT_RESOURCE}",
                    params={
                        "api-key": settings.DATA_GOV_IN_API_KEY,
                        "format": "json",
                        "filters[State/UT]": state,
                        "limit": 10,
                    },
                )
                r.raise_for_status()
                d = r.json()
                records = d.get("records", [])

                result = {
                    "state": state,
                    "accident_records": records[:5],
                    "total_records": d.get("total"),
                    "scraped_at": datetime.utcnow().isoformat(),
                    "source": "data_gov_in",
                }
                await scrape_cache.set(cache_key, result)
                return result
        except Exception as e:
            logger.warning(f"data.gov.in error: {e}")
        return {"state": state, "source": "data_gov_in_error"}


class ReliefWebScraper:
    """
    ReliefWeb Disasters API (free, no auth needed):
    Gets flood events for a country over a time period.
    Useful for getting authoritative flood event counts.
    """

    BASE = "https://api.reliefweb.int/v1/disasters"

    COUNTRY_MAP = {
        "Mumbai": "India", "Pune": "India",
        "New York": "United States", "Tokyo": "Japan",
    }

    async def scrape(self, city: str, years: int = 5) -> Dict[str, Any]:
        cache_key = f"reliefweb:{city.lower().replace(' ','_')}:{years}yr"
        cached = await scrape_cache.get(cache_key)
        if cached:
            return {**cached, "source": "redis_cache"}

        country = self.COUNTRY_MAP.get(city, "")
        if not country:
            return {"city": city, "source": "reliefweb", "note": "Country not mapped"}

        cutoff = (datetime.now() - timedelta(days=365*years)).strftime("%Y-%m-%dT00:00:00+00:00")

        payload = {
            "filter": {
                "operator": "AND",
                "conditions": [
                    {"field": "type.name", "value": "Flood"},
                    {"field": "date.created", "value": {"from": cutoff}},
                    {"field": "country.name", "value": country},
                ]
            },
            "fields": {"include": ["id", "name", "date", "status", "glide"]},
            "limit": 100,
            "sort": ["date.created:desc"],
        }

        try:
            async with _client(timeout=15) as c:
                r = await c.post(self.BASE, json=payload)
                r.raise_for_status()
                d = r.json()

            total = d.get("totalCount", 0)
            events = d.get("data", [])

            result = {
                "city": city,
                "country": country,
                "period_years": years,
                "flood_events_count": total,
                "recent_events": [
                    {
                        "name": e["fields"].get("name", ""),
                        "date": e["fields"].get("date", {}).get("created", ""),
                        "status": e["fields"].get("status", ""),
                    }
                    for e in events[:5]
                ],
                "scraped_at": datetime.utcnow().isoformat(),
                "source": "reliefweb_api",
            }
            await scrape_cache.set(cache_key, result)
            logger.info(f"ReliefWeb: {total} flood events in {country} ({years}yr)")
            return result

        except Exception as e:
            logger.warning(f"ReliefWeb error for {city}: {e}")
        return {"city": city, "source": "reliefweb_error"}


async def scrape_all_sources(city: str) -> Dict[str, Any]:
    """Run all government scrapers for a city concurrently."""
    wb = WorldBankScraper()
    noaa = NOAAScraper()
    rw = ReliefWebScraper()

    # City → country/state mappings
    city_state = {"Mumbai": "Maharashtra", "Pune": "Maharashtra", "New York": "NY"}

    tasks = [
        wb.scrape(city),
        noaa.scrape(city),
        rw.scrape(city),
    ]

    # Add city-specific scrapers
    if city == "New York":
        us = USTransportationScraper()
        tasks.extend([
            us.scrape_bridge_data("NY"),
            us.scrape_highway_stats("NY"),
        ])
    elif city in ("Mumbai", "Pune"):
        india = IndiaGovScraper()
        tasks.append(india.scrape("Maharashtra"))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        "city": city,
        "world_bank": results[0] if isinstance(results[0], dict) else {"error": str(results[0])},
        "noaa": results[1] if isinstance(results[1], dict) else {"error": str(results[1])},
        "reliefweb": results[2] if isinstance(results[2], dict) else {"error": str(results[2])},
        "city_specific": [r for r in results[3:] if isinstance(r, dict)],
        "scraped_at": datetime.utcnow().isoformat(),
    }
