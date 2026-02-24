"""
ml/data_fetcher.py – Real-time multimodal data ingestion
==========================================================
Priority order for each data type:
  1. Live API call (with retries)
  2. Redis cache (short TTL)
  3. OSM / Open Data scrape fallback
  4. Synthetic historical baseline

APIs used (all with free tiers):
  Traffic  : TomTom Flow v4, HERE Traffic v7
  AQI      : WAQI (World Air Quality Index), OpenWeatherMap Air Pollution
  Weather  : OpenWeatherMap Current + Forecast
  Flood    : NOAA Climate Data Online, ReliefWeb Disasters API
  OSM      : Overpass API (free, no key)
  WorldBank: Vehicles per 1000 people indicator
"""

import asyncio
import httpx
import logging
import json
import math
import random
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime, timedelta

from utils.config import settings
from utils.cache import (
    traffic_cache, aqi_cache, weather_cache,
    flood_cache, scrape_cache
)

logger = logging.getLogger("infraguard.fetcher")

# ── HTTP client factory ───────────────────────────────────────────────────────
def _client(timeout: float = 10.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        headers={"User-Agent": settings.SCRAPER_USER_AGENT},
        follow_redirects=True,
    )


# ── City profiles (fallback when no API key) ──────────────────────────────────
CITY_PROFILES = {
    "Mumbai": {
        "lat": 19.076, "lon": 72.877,
        "avg_daily_traffic": 94_000, "heavy_vehicle_pct": 38,
        "aqi": 178, "flood_events_5yr": 14,
        "road_age_years": 32, "rainfall_mm_annual": 2400,
        "soil_moisture": 72, "drainage_quality_score": 0.28,
        "surface_material_score": 0.65, "population_density": 20_600,
        "temperature_extremes": 14, "seismic_zone": 3,
    },
    "Pune": {
        "lat": 18.520, "lon": 73.856,
        "avg_daily_traffic": 68_000, "heavy_vehicle_pct": 31,
        "aqi": 155, "flood_events_5yr": 9,
        "road_age_years": 27, "rainfall_mm_annual": 1800,
        "soil_moisture": 65, "drainage_quality_score": 0.32,
        "surface_material_score": 0.62, "population_density": 11_000,
        "temperature_extremes": 16, "seismic_zone": 3,
    },
    "New York": {
        "lat": 40.712, "lon": -74.006,
        "avg_daily_traffic": 142_000, "heavy_vehicle_pct": 22,
        "aqi": 68, "flood_events_5yr": 6,
        "road_age_years": 58, "rainfall_mm_annual": 1200,
        "soil_moisture": 45, "drainage_quality_score": 0.62,
        "surface_material_score": 0.80, "population_density": 10_700,
        "temperature_extremes": 30, "seismic_zone": 1,
    },
    "Tokyo": {
        "lat": 35.689, "lon": 139.692,
        "avg_daily_traffic": 198_000, "heavy_vehicle_pct": 18,
        "aqi": 42, "flood_events_5yr": 3,
        "road_age_years": 18, "rainfall_mm_annual": 1500,
        "soil_moisture": 50, "drainage_quality_score": 0.88,
        "surface_material_score": 0.95, "population_density": 6_400,
        "temperature_extremes": 28, "seismic_zone": 5,
    },
}

CITY_COORDS = {k: (v["lat"], v["lon"]) for k, v in CITY_PROFILES.items()}


def nearest_city(lat: float, lon: float) -> str:
    return min(CITY_COORDS, key=lambda c: (CITY_COORDS[c][0]-lat)**2 + (CITY_COORDS[c][1]-lon)**2)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAFFIC FETCHER – TomTom Flow v4 + HERE Traffic v7 + OSM fallback
# ═══════════════════════════════════════════════════════════════════════════════
class TrafficFetcher:
    """
    TomTom Flow API v4:
      GET https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json
      Params: key, point (lat,lon)
      Returns: currentSpeed, freeFlowSpeed → used to estimate congestion ratio

    HERE Traffic v7:
      GET https://data.traffic.hereapi.com/v7/flow
      Params: apiKey, locationReferencing=shape, in=circle:{lat},{lon};r=1000
      Returns: flow items with currentFlow.speed, freeFlow.speed

    OSM Overpass fallback:
      Counts road segments of different classes within 2km radius
    """

    TOMTOM_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    HERE_URL   = "https://data.traffic.hereapi.com/v7/flow"

    async def fetch(self, lat: float, lon: float) -> Dict[str, Any]:
        cache_key = f"{lat:.3f}:{lon:.3f}"
        cached = await traffic_cache.get(cache_key)
        if cached:
            logger.debug(f"Traffic cache hit for {cache_key}")
            return {**cached, "source": "redis_cache"}

        result = None

        # 1. Try TomTom
        if settings.TOMTOM_API_KEY:
            result = await self._fetch_tomtom(lat, lon)

        # 2. Try HERE
        if result is None and settings.HERE_API_KEY:
            result = await self._fetch_here(lat, lon)

        # 3. OSM Overpass fallback
        if result is None:
            result = await self._fetch_osm(lat, lon)

        # 4. Synthetic
        if result is None:
            city = nearest_city(lat, lon)
            p = CITY_PROFILES[city]
            result = {
                "avg_daily_traffic": p["avg_daily_traffic"],
                "heavy_vehicle_pct": p["heavy_vehicle_pct"],
                "congestion_pct": 45.0,
                "source": "synthetic_profile",
            }

        await traffic_cache.set(cache_key, result)
        return result

    async def _fetch_tomtom(self, lat: float, lon: float) -> Optional[Dict]:
        try:
            async with _client() as c:
                r = await c.get(self.TOMTOM_URL, params={
                    "key": settings.TOMTOM_API_KEY,
                    "point": f"{lat},{lon}",
                    "unit": "KMPH",
                })
                r.raise_for_status()
                fd = r.json().get("flowSegmentData", {})
                current_speed = fd.get("currentSpeed", 0)
                free_speed = fd.get("freeFlowSpeed", 1) or 1
                congestion_pct = max(0, (1 - current_speed / free_speed) * 100)

                # Estimate daily traffic from functional road class
                frc = fd.get("frc", "FRC3")  # FRC0=motorway, FRC6=local
                frc_map = {"FRC0": 150000, "FRC1": 100000, "FRC2": 70000,
                           "FRC3": 45000,  "FRC4": 25000,  "FRC5": 12000, "FRC6": 5000}
                estimated_traffic = frc_map.get(frc, 45000)
                # Scale by congestion
                estimated_traffic = int(estimated_traffic * (1 + congestion_pct / 200))

                logger.info(f"TomTom traffic: speed={current_speed} kmph, frc={frc}, congestion={congestion_pct:.1f}%")
                return {
                    "avg_daily_traffic": estimated_traffic,
                    "current_speed_kmph": current_speed,
                    "free_flow_speed_kmph": free_speed,
                    "congestion_pct": round(congestion_pct, 1),
                    "road_class": frc,
                    "heavy_vehicle_pct": 25.0,  # TomTom doesn't give HV% on free tier
                    "source": "tomtom_live",
                }
        except httpx.HTTPStatusError as e:
            logger.warning(f"TomTom HTTP error {e.response.status_code}: {e}")
        except Exception as e:
            logger.warning(f"TomTom error: {e}")
        return None

    async def _fetch_here(self, lat: float, lon: float) -> Optional[Dict]:
        try:
            async with _client() as c:
                r = await c.get(self.HERE_URL, params={
                    "apiKey": settings.HERE_API_KEY,
                    "locationReferencing": "shape",
                    "in": f"circle:{lat},{lon};r=500",
                })
                r.raise_for_status()
                items = r.json().get("results", [])
                if not items:
                    return None

                speeds, free_speeds = [], []
                for item in items[:10]:
                    cf = item.get("currentFlow", {})
                    ff = item.get("freeFlow", {})
                    if cf.get("speed"): speeds.append(cf["speed"])
                    if ff.get("speed"): free_speeds.append(ff["speed"])

                if not speeds:
                    return None

                avg_speed = sum(speeds) / len(speeds)
                avg_free  = sum(free_speeds) / len(free_speeds) if free_speeds else avg_speed
                congestion = max(0, (1 - avg_speed / avg_free) * 100) if avg_free else 0

                logger.info(f"HERE traffic: avg_speed={avg_speed:.1f}, segments={len(items)}")
                return {
                    "avg_daily_traffic": int(30000 + congestion * 800),
                    "current_speed_kmph": round(avg_speed, 1),
                    "free_flow_speed_kmph": round(avg_free, 1),
                    "congestion_pct": round(congestion, 1),
                    "segments_sampled": len(items),
                    "heavy_vehicle_pct": 20.0,
                    "source": "here_live",
                }
        except Exception as e:
            logger.warning(f"HERE error: {e}")
        return None

    async def _fetch_osm(self, lat: float, lon: float) -> Optional[Dict]:
        """Count road segments from OSM Overpass to estimate traffic class."""
        query = f"""
        [out:json][timeout:10];
        (
          way["highway"~"motorway|trunk|primary|secondary"](around:2000,{lat},{lon});
        );
        out count;
        """
        try:
            async with _client(timeout=12) as c:
                r = await c.post(settings.OVERPASS_API_URL, data={"data": query})
                r.raise_for_status()
                total = r.json().get("elements", [{}])[0].get("tags", {}).get("total", 0)
                total = int(total) if total else 0
                # Rough heuristic: more roads → higher traffic area
                estimated = min(150000, total * 3000 + 20000)
                logger.info(f"OSM road count: {total}, estimated traffic: {estimated}")
                return {
                    "avg_daily_traffic": estimated,
                    "heavy_vehicle_pct": 20.0,
                    "road_segments": total,
                    "source": "osm_overpass",
                }
        except Exception as e:
            logger.warning(f"OSM Overpass error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# AQI FETCHER – WAQI + OpenWeather Air Pollution API
# ═══════════════════════════════════════════════════════════════════════════════
class AQIFetcher:
    """
    WAQI API (World Air Quality Index):
      GET https://api.waqi.info/feed/geo:{lat};{lon}/?token=KEY
      Returns: aqi (US AQI standard 0-500), dominant pollutant,
               individual pollutant concentrations

    OpenWeather Air Pollution API (free):
      GET https://api.openweathermap.org/data/2.5/air_pollution
      Returns: AQI on EU scale 1-5, CO, NO2, O3, PM2.5, PM10

    Both APIs are free with registration.
    """

    WAQI_URL = "https://api.waqi.info/feed/geo:{lat};{lon}/"
    OW_URL   = "https://api.openweathermap.org/data/2.5/air_pollution"
    # PM2.5 to US AQI breakpoints
    PM25_BREAKPOINTS = [
        (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    def _pm25_to_aqi(self, pm25: float) -> float:
        for lo_c, hi_c, lo_a, hi_a in self.PM25_BREAKPOINTS:
            if lo_c <= pm25 <= hi_c:
                return lo_a + (pm25 - lo_c) / (hi_c - lo_c) * (hi_a - lo_a)
        return 500.0

    async def fetch(self, lat: float, lon: float) -> Dict[str, Any]:
        cache_key = f"{lat:.3f}:{lon:.3f}"
        cached = await aqi_cache.get(cache_key)
        if cached:
            return {**cached, "source": "redis_cache"}

        result = None

        if settings.WAQI_API_KEY:
            result = await self._fetch_waqi(lat, lon)

        if result is None and settings.OPENWEATHER_API_KEY:
            result = await self._fetch_openweather(lat, lon)

        if result is None:
            city = nearest_city(lat, lon)
            result = {
                "aqi": CITY_PROFILES[city]["aqi"],
                "category": self._aqi_category(CITY_PROFILES[city]["aqi"]),
                "source": "synthetic_profile",
            }

        await aqi_cache.set(cache_key, result)
        return result

    async def _fetch_waqi(self, lat: float, lon: float) -> Optional[Dict]:
        try:
            async with _client() as c:
                url = self.WAQI_URL.format(lat=lat, lon=lon)
                r = await c.get(url, params={"token": settings.WAQI_API_KEY})
                r.raise_for_status()
                d = r.json()
                if d.get("status") != "ok":
                    return None

                data = d["data"]
                aqi_val = float(data["aqi"])
                iaqi = data.get("iaqi", {})
                pm25 = iaqi.get("pm25", {}).get("v")
                pm10 = iaqi.get("pm10", {}).get("v")
                no2  = iaqi.get("no2",  {}).get("v")
                o3   = iaqi.get("o3",   {}).get("v")
                dominant = data.get("dominentpol", "pm25")
                station  = data.get("city", {}).get("name", "Unknown")

                logger.info(f"WAQI AQI={aqi_val}, station={station}, dominant={dominant}")
                return {
                    "aqi": aqi_val,
                    "category": self._aqi_category(aqi_val),
                    "dominant_pollutant": dominant,
                    "pm25_ugm3": pm25,
                    "pm10_ugm3": pm10,
                    "no2_ugm3": no2,
                    "o3_ugm3": o3,
                    "station": station,
                    "source": "waqi_live",
                }
        except Exception as e:
            logger.warning(f"WAQI error: {e}")
        return None

    async def _fetch_openweather(self, lat: float, lon: float) -> Optional[Dict]:
        try:
            async with _client() as c:
                r = await c.get(self.OW_URL, params={
                    "lat": lat, "lon": lon, "appid": settings.OPENWEATHER_API_KEY
                })
                r.raise_for_status()
                d = r.json()
                item = d["list"][0]
                comp = item["components"]

                pm25 = comp.get("pm2_5", 0)
                aqi_us = self._pm25_to_aqi(pm25)
                eu_aqi = item["main"]["aqi"]  # 1-5

                logger.info(f"OpenWeather AQI: EU={eu_aqi}, PM2.5={pm25}µg/m³, US_AQI≈{aqi_us:.0f}")
                return {
                    "aqi": round(aqi_us, 1),
                    "eu_aqi_index": eu_aqi,
                    "category": self._aqi_category(aqi_us),
                    "pm25_ugm3": pm25,
                    "pm10_ugm3": comp.get("pm10"),
                    "no2_ugm3": comp.get("no2"),
                    "o3_ugm3": comp.get("o3"),
                    "co_ugm3": comp.get("co"),
                    "source": "openweather_live",
                }
        except Exception as e:
            logger.warning(f"OpenWeather AQI error: {e}")
        return None

    def _aqi_category(self, aqi: float) -> str:
        if aqi <= 50:  return "Good"
        if aqi <= 100: return "Moderate"
        if aqi <= 150: return "Unhealthy for Sensitive Groups"
        if aqi <= 200: return "Unhealthy"
        if aqi <= 300: return "Very Unhealthy"
        return "Hazardous"


# ═══════════════════════════════════════════════════════════════════════════════
# WEATHER FETCHER – OpenWeatherMap Current + 5-day Forecast
# ═══════════════════════════════════════════════════════════════════════════════
class WeatherFetcher:
    """
    OpenWeatherMap (free tier: 60 calls/min):
      Current: GET /data/2.5/weather
      Forecast: GET /data/2.5/forecast (5 days / 3hr intervals)

    Derived features:
      - Annual rainfall estimate from 5-day total × seasonality factor
      - Temperature extremes from min/max
      - Soil moisture approximation from humidity + rainfall
    """

    CURRENT_URL  = "https://api.openweathermap.org/data/2.5/weather"
    FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

    async def fetch(self, lat: float, lon: float) -> Dict[str, Any]:
        cache_key = f"wx:{lat:.3f}:{lon:.3f}"
        cached = await weather_cache.get(cache_key)
        if cached:
            return {**cached, "source": "redis_cache"}

        if not settings.OPENWEATHER_API_KEY:
            city = nearest_city(lat, lon)
            p = CITY_PROFILES[city]
            return {
                "temperature_c": 28.0,
                "humidity_pct": 70.0,
                "rainfall_mm_1h": 0.0,
                "rainfall_mm_3h": 0.0,
                "rainfall_mm_annual_estimate": p["rainfall_mm_annual"],
                "soil_moisture_estimate": p["soil_moisture"],
                "temperature_extremes_delta": p["temperature_extremes"],
                "wind_speed_ms": 3.5,
                "source": "synthetic_profile",
            }

        current, forecast = await asyncio.gather(
            self._fetch_current(lat, lon),
            self._fetch_forecast(lat, lon),
            return_exceptions=True,
        )

        result: Dict[str, Any] = {"source": "openweather_live"}

        if isinstance(current, dict):
            result.update({
                "temperature_c": current.get("temp"),
                "feels_like_c": current.get("feels_like"),
                "humidity_pct": current.get("humidity"),
                "rainfall_mm_1h": current.get("rain_1h", 0),
                "wind_speed_ms": current.get("wind_speed"),
                "weather_desc": current.get("description"),
                "visibility_m": current.get("visibility"),
            })

        if isinstance(forecast, dict):
            result.update({
                "rainfall_mm_3h": forecast.get("rain_3h", 0),
                "rainfall_mm_5day": forecast.get("total_rain_5day", 0),
                "rainfall_mm_annual_estimate": forecast.get("annual_estimate", 1000),
                "temp_min_c": forecast.get("temp_min"),
                "temp_max_c": forecast.get("temp_max"),
                "temperature_extremes_delta": forecast.get("temp_range", 20),
                "soil_moisture_estimate": self._estimate_soil_moisture(
                    result.get("humidity_pct", 60),
                    forecast.get("total_rain_5day", 0),
                ),
            })

        await weather_cache.set(cache_key, result)
        return result

    async def _fetch_current(self, lat: float, lon: float) -> Optional[Dict]:
        try:
            async with _client() as c:
                r = await c.get(self.CURRENT_URL, params={
                    "lat": lat, "lon": lon,
                    "appid": settings.OPENWEATHER_API_KEY,
                    "units": "metric",
                })
                r.raise_for_status()
                d = r.json()
                return {
                    "temp": d["main"]["temp"],
                    "feels_like": d["main"]["feels_like"],
                    "humidity": d["main"]["humidity"],
                    "wind_speed": d.get("wind", {}).get("speed"),
                    "rain_1h": d.get("rain", {}).get("1h", 0),
                    "description": d.get("weather", [{}])[0].get("description"),
                    "visibility": d.get("visibility"),
                }
        except Exception as e:
            logger.warning(f"OW current weather error: {e}")
        return None

    async def _fetch_forecast(self, lat: float, lon: float) -> Optional[Dict]:
        try:
            async with _client() as c:
                r = await c.get(self.FORECAST_URL, params={
                    "lat": lat, "lon": lon,
                    "appid": settings.OPENWEATHER_API_KEY,
                    "units": "metric", "cnt": 40,
                })
                r.raise_for_status()
                items = r.json().get("list", [])
                total_rain = sum(i.get("rain", {}).get("3h", 0) for i in items)
                temps = [i["main"]["temp"] for i in items]
                t_min, t_max = min(temps), max(temps)
                # Extrapolate 5-day rain to annual (rough seasonal factor)
                season_factor = self._season_factor(lat)
                annual_est = total_rain * 73 * season_factor  # 40 periods × 73 = ~1yr
                return {
                    "rain_3h": items[0].get("rain", {}).get("3h", 0) if items else 0,
                    "total_rain_5day": round(total_rain, 1),
                    "annual_estimate": round(annual_est),
                    "temp_min": round(t_min, 1),
                    "temp_max": round(t_max, 1),
                    "temp_range": round(t_max - t_min, 1),
                }
        except Exception as e:
            logger.warning(f"OW forecast error: {e}")
        return None

    def _season_factor(self, lat: float) -> float:
        """Estimate seasonal multiplier based on latitude."""
        if abs(lat) < 15: return 1.2  # tropical – high annual rain
        if abs(lat) < 30: return 1.0
        if abs(lat) < 50: return 0.9
        return 0.8

    def _estimate_soil_moisture(self, humidity: float, rain_5day: float) -> float:
        base = humidity * 0.6 + min(40, rain_5day * 2)
        return round(min(100, max(0, base)), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# FLOOD FETCHER – NOAA CDO + ReliefWeb + OSM waterway analysis
# ═══════════════════════════════════════════════════════════════════════════════
class FloodFetcher:
    """
    NOAA Climate Data Online (CDO) API:
      GET https://www.ncdc.noaa.gov/cdo-web/api/v2/data
      Params: datasetid=GHCND, datatypeid=PRCP, stationid=..., date range
      Returns: precipitation records → used to count high-rainfall events

    ReliefWeb Disasters API (free, no auth):
      POST https://api.reliefweb.int/v1/disasters
      Returns: flood disasters for a given country / region

    OSM waterway proximity:
      Queries Overpass for rivers / canals near the location
    """

    NOAA_URL      = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    NOAA_STATION_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/stations"
    RELIEFWEB_URL = "https://api.reliefweb.int/v1/disasters"

    # Country code mapping from lat/lon (simplified)
    COUNTRY_MAP = {
        "Mumbai": "IND", "Pune": "IND",
        "New York": "USA", "Tokyo": "JPN",
    }

    async def fetch(self, lat: float, lon: float) -> Dict[str, Any]:
        cache_key = f"flood:{lat:.3f}:{lon:.3f}"
        cached = await flood_cache.get(cache_key)
        if cached:
            return {**cached, "source": "redis_cache"}

        results = await asyncio.gather(
            self._fetch_reliefweb(lat, lon),
            self._fetch_osm_waterways(lat, lon),
            return_exceptions=True,
        )

        relief = results[0] if isinstance(results[0], dict) else None
        osm    = results[1] if isinstance(results[1], dict) else None

        city = nearest_city(lat, lon)
        base = CITY_PROFILES[city]

        flood_events = base["flood_events_5yr"]
        nearby_waterways = 0

        if relief:
            flood_events = relief.get("flood_events_5yr", flood_events)

        if osm:
            nearby_waterways = osm.get("waterway_count", 0)
            # More waterways → higher flood risk modifier
            flood_events = int(flood_events * (1 + min(0.5, nearby_waterways * 0.1)))

        result = {
            "flood_events_5yr": flood_events,
            "nearby_waterways": nearby_waterways,
            "reliefweb_events": relief.get("raw_count") if relief else None,
            "flood_risk_score": min(10, flood_events * 0.7 + nearby_waterways * 0.3),
            "source": f"combined_{('reliefweb' if relief else 'synthetic')}+osm",
        }
        await flood_cache.set(cache_key, result)
        return result

    async def _fetch_reliefweb(self, lat: float, lon: float) -> Optional[Dict]:
        city = nearest_city(lat, lon)
        country = self.COUNTRY_MAP.get(city, "")
        if not country:
            return None

        # Country name for text search
        country_names = {"IND": "India", "USA": "United States", "JPN": "Japan"}
        country_name = country_names.get(country, country)
        cutoff = (datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%dT%H:%M:%S+00:00")

        payload = {
            "filter": {
                "operator": "AND",
                "conditions": [
                    {"field": "type.name", "value": "Flood"},
                    {"field": "date.created", "value": {"from": cutoff}},
                    {"field": "country.name", "value": country_name},
                ]
            },
            "fields": {"include": ["id", "name", "date", "status"]},
            "limit": 50,
        }
        try:
            async with _client(timeout=12) as c:
                r = await c.post(
                    self.RELIEFWEB_URL,
                    json=payload,
                    headers={"Accept": "application/json"},
                )
                r.raise_for_status()
                d = r.json()
                count = d.get("totalCount", 0)
                logger.info(f"ReliefWeb floods in {country_name} (5yr): {count}")
                return {
                    "flood_events_5yr": min(20, count),
                    "raw_count": count,
                    "country": country_name,
                    "source": "reliefweb",
                }
        except Exception as e:
            logger.warning(f"ReliefWeb error: {e}")
        return None

    async def _fetch_osm_waterways(self, lat: float, lon: float) -> Optional[Dict]:
        query = f"""
        [out:json][timeout:10];
        (
          way["waterway"~"river|canal|stream|drain"](around:3000,{lat},{lon});
        );
        out count;
        """
        try:
            async with _client(timeout=12) as c:
                r = await c.post(settings.OVERPASS_API_URL, data={"data": query})
                r.raise_for_status()
                count = int(r.json().get("elements", [{}])[0].get("tags", {}).get("total", 0))
                logger.info(f"OSM waterways within 3km: {count}")
                return {"waterway_count": count, "source": "osm_overpass"}
        except Exception as e:
            logger.warning(f"OSM waterway error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE FETCHER – OSM roads + World Bank vehicle data
# ═══════════════════════════════════════════════════════════════════════════════
class InfrastructureFetcher:
    """
    OSM Overpass: road network density, road types, bridge count
    World Bank API (free): vehicles per 1000 people (IS.VEH.NVEH.P3)
    """

    WORLDBANK_URL = "https://api.worldbank.org/v2/country/{iso}/indicator/{indicator}"

    COUNTRY_ISO = {"Mumbai": "IN", "Pune": "IN", "New York": "US", "Tokyo": "JP"}
    COUNTRY_VEHICLE_INDICATOR = "IS.VEH.NVEH.P3"

    async def fetch(self, lat: float, lon: float) -> Dict[str, Any]:
        cache_key = f"infra:{lat:.3f}:{lon:.3f}"
        cached = await scrape_cache.get(cache_key)
        if cached:
            return {**cached, "source": "redis_cache"}

        city = nearest_city(lat, lon)
        iso  = self.COUNTRY_ISO.get(city, "US")

        osm_data, wb_data = await asyncio.gather(
            self._fetch_osm_infra(lat, lon),
            self._fetch_worldbank_vehicles(iso),
            return_exceptions=True,
        )

        result: Dict[str, Any] = {"source": "combined"}
        if isinstance(osm_data, dict):
            result.update(osm_data)
        if isinstance(wb_data, dict):
            result.update(wb_data)

        await scrape_cache.set(cache_key, result)
        return result

    async def _fetch_osm_infra(self, lat: float, lon: float) -> Optional[Dict]:
        # Get roads + bridges + age proxies
        query = f"""
        [out:json][timeout:15];
        (
          way["highway"~"motorway|trunk|primary|secondary|tertiary"](around:5000,{lat},{lon});
          way["bridge"="yes"](around:5000,{lat},{lon});
        );
        out tags;
        """
        try:
            async with _client(timeout=18) as c:
                r = await c.post(settings.OVERPASS_API_URL, data={"data": query})
                r.raise_for_status()
                elements = r.json().get("elements", [])

                road_count = sum(1 for e in elements if "highway" in e.get("tags", {}))
                bridge_count = sum(1 for e in elements if e.get("tags", {}).get("bridge") == "yes")

                # Surface type distribution
                surfaces = [e["tags"].get("surface", "") for e in elements]
                asphalt_pct = surfaces.count("asphalt") / max(1, len(surfaces)) * 100
                concrete_pct = surfaces.count("concrete") / max(1, len(surfaces)) * 100

                # Start year / age proxy from "start_date" tag
                years = []
                for e in elements:
                    sd = e.get("tags", {}).get("start_date", "")
                    if sd and len(sd) >= 4:
                        try:
                            years.append(datetime.now().year - int(sd[:4]))
                        except ValueError:
                            pass
                avg_age = sum(years) / len(years) if years else None

                logger.info(f"OSM infra: roads={road_count}, bridges={bridge_count}")
                return {
                    "osm_road_count": road_count,
                    "osm_bridge_count": bridge_count,
                    "osm_asphalt_pct": round(asphalt_pct, 1),
                    "osm_concrete_pct": round(concrete_pct, 1),
                    "osm_avg_road_age_years": round(avg_age, 1) if avg_age else None,
                }
        except Exception as e:
            logger.warning(f"OSM infra error: {e}")
        return None

    async def _fetch_worldbank_vehicles(self, iso: str) -> Optional[Dict]:
        url = self.WORLDBANK_URL.format(iso=iso, indicator=self.COUNTRY_VEHICLE_INDICATOR)
        try:
            async with _client() as c:
                r = await c.get(url, params={"format": "json", "mrv": 3, "per_page": 3})
                r.raise_for_status()
                data = r.json()
                if len(data) < 2 or not data[1]:
                    return None
                # Take most recent non-null value
                for item in data[1]:
                    if item.get("value") is not None:
                        veh_per_1000 = float(item["value"])
                        logger.info(f"World Bank vehicles/{iso}: {veh_per_1000} per 1000 people")
                        return {
                            "vehicles_per_1000": veh_per_1000,
                            "vehicle_data_year": item.get("date"),
                        }
        except Exception as e:
            logger.warning(f"World Bank error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# POLLUTION / SEISMIC RISK FETCHER
# ═══════════════════════════════════════════════════════════════════════════════
class SeismicRiskFetcher:
    """
    USGS Earthquake Hazard API (free, no key):
      GET https://earthquake.usgs.gov/ws/designmaps/asce7-22.json
      Params: latitude, longitude, riskCategory=III
      Returns: Ss (short-period spectral accel.) → seismic zone proxy
    """

    USGS_URL = "https://earthquake.usgs.gov/ws/designmaps/asce7-22.json"

    async def fetch(self, lat: float, lon: float) -> Dict[str, Any]:
        cache_key = f"seismic:{lat:.3f}:{lon:.3f}"
        cached = await scrape_cache.get(cache_key)
        if cached:
            return cached

        try:
            async with _client(timeout=12) as c:
                r = await c.get(self.USGS_URL, params={
                    "latitude": lat, "longitude": lon,
                    "riskCategory": "III", "siteClass": "D", "title": "InfraGuard"
                })
                r.raise_for_status()
                d = r.json().get("output", {}).get("designSpectrum", {})
                ss = float(d.get("ss", 0))
                # Convert Ss (0-3+) to seismic zone 1-5
                zone = 1 + min(4, int(ss / 0.5))
                result = {
                    "seismic_ss": round(ss, 3),
                    "seismic_zone": zone,
                    "source": "usgs_live",
                }
                logger.info(f"USGS seismic Ss={ss:.3f}, zone={zone}")
                await scrape_cache.set(cache_key, result)
                return result
        except Exception as e:
            logger.warning(f"USGS seismic error: {e}")
            city = nearest_city(lat, lon)
            return {
                "seismic_zone": CITY_PROFILES[city].get("seismic_zone", 2),
                "source": "synthetic_profile",
            }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════════
class DataAggregator:
    """
    Concurrently fetches from all sources and merges into a unified feature vector.
    Manual overrides from the API request take highest priority.
    """

    def __init__(self):
        self.traffic   = TrafficFetcher()
        self.aqi       = AQIFetcher()
        self.weather   = WeatherFetcher()
        self.flood     = FloodFetcher()
        self.infra     = InfrastructureFetcher()
        self.seismic   = SeismicRiskFetcher()

    async def fetch_all(
        self,
        lat: float,
        lon: float,
        manual_overrides: Optional[Dict] = None,
    ) -> Dict[str, Any]:

        city = nearest_city(lat, lon)
        baseline = dict(CITY_PROFILES[city])
        baseline.pop("lat", None)
        baseline.pop("lon", None)

        # Concurrent fetch from all sources
        traffic_r, aqi_r, weather_r, flood_r, infra_r, seismic_r = await asyncio.gather(
            self.traffic.fetch(lat, lon),
            self.aqi.fetch(lat, lon),
            self.weather.fetch(lat, lon),
            self.flood.fetch(lat, lon),
            self.infra.fetch(lat, lon),
            self.seismic.fetch(lat, lon),
            return_exceptions=True,
        )

        features = dict(baseline)
        sources_used = {"synthetic_baseline": city}

        def _safe(r, key, fallback=None):
            if isinstance(r, dict) and key in r and r[key] is not None:
                return r[key]
            return fallback

        # Layer in traffic
        if isinstance(traffic_r, dict):
            features["avg_daily_traffic"] = _safe(traffic_r, "avg_daily_traffic", features["avg_daily_traffic"])
            features["heavy_vehicle_pct"] = _safe(traffic_r, "heavy_vehicle_pct", features["heavy_vehicle_pct"])
            features["congestion_pct"] = _safe(traffic_r, "congestion_pct", 40)
            sources_used["traffic"] = traffic_r.get("source", "unknown")

        # Layer in AQI
        if isinstance(aqi_r, dict):
            features["aqi"] = _safe(aqi_r, "aqi", features["aqi"])
            features["pm25_ugm3"] = _safe(aqi_r, "pm25_ugm3")
            features["dominant_pollutant"] = _safe(aqi_r, "dominant_pollutant")
            sources_used["aqi"] = aqi_r.get("source", "unknown")

        # Layer in weather
        if isinstance(weather_r, dict):
            features["rainfall_mm_annual"] = _safe(weather_r, "rainfall_mm_annual_estimate", features["rainfall_mm_annual"])
            features["soil_moisture"] = _safe(weather_r, "soil_moisture_estimate", features["soil_moisture"])
            features["temperature_extremes"] = _safe(weather_r, "temperature_extremes_delta", features["temperature_extremes"])
            features["current_temp_c"] = _safe(weather_r, "temperature_c")
            features["current_humidity_pct"] = _safe(weather_r, "humidity_pct")
            features["rainfall_mm_1h"] = _safe(weather_r, "rainfall_mm_1h", 0)
            sources_used["weather"] = weather_r.get("source", "unknown")

        # Layer in flood
        if isinstance(flood_r, dict):
            features["flood_events_5yr"] = _safe(flood_r, "flood_events_5yr", features["flood_events_5yr"])
            features["nearby_waterways"] = _safe(flood_r, "nearby_waterways", 0)
            sources_used["flood"] = flood_r.get("source", "unknown")

        # Layer in infra
        if isinstance(infra_r, dict):
            osm_age = _safe(infra_r, "osm_avg_road_age_years")
            if osm_age:
                features["road_age_years"] = osm_age
            features["bridge_count_5km"] = _safe(infra_r, "osm_bridge_count", 0)
            sources_used["infrastructure"] = "osm+worldbank"

        # Layer in seismic
        if isinstance(seismic_r, dict):
            features["seismic_zone"] = _safe(seismic_r, "seismic_zone", 2)
            features["seismic_ss"] = _safe(seismic_r, "seismic_ss", 0)
            sources_used["seismic"] = seismic_r.get("source", "unknown")

        # Apply manual overrides (highest priority)
        if manual_overrides:
            for k, v in manual_overrides.items():
                if v is not None:
                    features[k] = float(v)
            sources_used["manual_override"] = list(manual_overrides.keys())

        # Normalise types
        for k in list(features.keys()):
            if isinstance(features[k], float) and math.isnan(features[k]):
                features[k] = 0.0

        return {
            "city": city,
            "lat": lat,
            "lon": lon,
            "features": features,
            "sources_used": sources_used,
            "fetched_at": datetime.utcnow().isoformat(),
        }

    async def get_api_status(self) -> List[Dict]:
        """Check which APIs are configured and responding."""
        statuses = []

        statuses.append({
            "name": "TomTom Traffic",
            "configured": bool(settings.TOMTOM_API_KEY),
            "endpoint": "api.tomtom.com/traffic",
            "type": "traffic",
        })
        statuses.append({
            "name": "HERE Traffic",
            "configured": bool(settings.HERE_API_KEY),
            "endpoint": "data.traffic.hereapi.com",
            "type": "traffic",
        })
        statuses.append({
            "name": "WAQI AQI",
            "configured": bool(settings.WAQI_API_KEY),
            "endpoint": "api.waqi.info",
            "type": "aqi",
        })
        statuses.append({
            "name": "OpenWeatherMap",
            "configured": bool(settings.OPENWEATHER_API_KEY),
            "endpoint": "api.openweathermap.org",
            "type": "weather+aqi",
        })
        statuses.append({
            "name": "OSM Overpass",
            "configured": True,  # always free
            "endpoint": "overpass-api.de",
            "type": "infrastructure",
        })
        statuses.append({
            "name": "ReliefWeb Disasters",
            "configured": True,  # no key needed
            "endpoint": "api.reliefweb.int",
            "type": "flood",
        })
        statuses.append({
            "name": "USGS Seismic",
            "configured": True,  # always free
            "endpoint": "earthquake.usgs.gov",
            "type": "seismic",
        })
        statuses.append({
            "name": "World Bank Open Data",
            "configured": True,  # always free
            "endpoint": "api.worldbank.org",
            "type": "infrastructure",
        })
        statuses.append({
            "name": "NOAA CDO",
            "configured": bool(settings.NOAA_API_KEY),
            "endpoint": "www.ncdc.noaa.gov/cdo-web/api",
            "type": "flood+climate",
        })
        return statuses
