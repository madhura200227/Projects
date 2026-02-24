"""
scrapers/osm_scraper.py – Real-time OpenStreetMap data scraping
=================================================================
Uses Overpass API (free, no auth required) to fetch:
  - Road network stats (density, surface types, speed limits)
  - Bridge/flyover inventory
  - Infrastructure age from OSM tags
  - Waterway proximity
  - Industrial/port zones (pollution risk proxy)

Rate limit: ~10k requests/day on public instance.
For production, self-host Overpass or use overpass.kumi.systems as fallback.
"""

import asyncio
import httpx
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from utils.config import settings
from utils.cache import scrape_cache

logger = logging.getLogger("infraguard.scraper.osm")

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]


async def _overpass_query(query: str, timeout: int = 15) -> Optional[Dict]:
    """Try multiple Overpass endpoints with fallback."""
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as c:
                r = await c.post(endpoint, data={"data": query})
                r.raise_for_status()
                return r.json()
        except Exception as e:
            logger.debug(f"Overpass endpoint {endpoint} failed: {e}")
            continue
    logger.warning("All Overpass endpoints failed")
    return None


class OSMRoadScraper:
    """Scrape road network data for a given city bounding box."""

    # City bounding boxes [south, west, north, east]
    CITY_BBOX = {
        "Mumbai":   [18.85, 72.75, 19.30, 73.00],
        "Pune":     [18.40, 73.75, 18.65, 74.00],
        "New York": [40.50, -74.30, 40.92, -73.70],
        "Tokyo":    [35.50, 139.50, 35.90, 139.90],
    }

    async def scrape(self, city: str) -> Dict[str, Any]:
        cache_key = f"osm_roads:{city.lower().replace(' ', '_')}"
        cached = await scrape_cache.get(cache_key)
        if cached:
            return {**cached, "source": "redis_cache"}

        bbox = self.CITY_BBOX.get(city)
        if not bbox:
            return {"error": f"No bounding box for {city}"}

        s, w, n, e = bbox
        query = f"""
        [out:json][timeout:30];
        (
          way["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]({s},{w},{n},{e});
        );
        out tags;
        >;
        out skel qt;
        """

        data = await _overpass_query(query, timeout=35)
        if not data:
            return {"error": "Overpass query failed", "city": city}

        elements = data.get("elements", [])
        ways = [e for e in elements if e["type"] == "way"]

        if not ways:
            return {"city": city, "road_count": 0, "source": "osm_overpass"}

        # Analyse tags
        highway_types = {}
        surfaces = {}
        max_speeds = []
        has_lanes  = []
        start_years = []
        lit_count  = 0

        for w in ways:
            tags = w.get("tags", {})

            ht = tags.get("highway", "unknown")
            highway_types[ht] = highway_types.get(ht, 0) + 1

            surf = tags.get("surface", "unknown")
            surfaces[surf] = surfaces.get(surf, 0) + 1

            ms = tags.get("maxspeed", "").replace(" mph", "").replace(" kmh", "").replace("mph", "")
            try:
                max_speeds.append(int(ms))
            except ValueError:
                pass

            lanes = tags.get("lanes", "")
            try:
                has_lanes.append(int(lanes))
            except ValueError:
                pass

            if tags.get("lit") == "yes":
                lit_count += 1

            sd = tags.get("start_date", "")
            if sd and len(sd) >= 4:
                try:
                    start_years.append(int(sd[:4]))
                except ValueError:
                    pass

        total_roads = len(ways)
        asphalt_pct = round((surfaces.get("asphalt", 0) + surfaces.get("paved", 0)) / max(1, total_roads) * 100, 1)
        avg_speed   = round(sum(max_speeds) / len(max_speeds), 1) if max_speeds else None
        avg_lanes   = round(sum(has_lanes) / len(has_lanes), 1) if has_lanes else None
        avg_age_yrs = None
        if start_years:
            avg_age_yrs = round(datetime.now().year - (sum(start_years) / len(start_years)), 1)
        lit_pct     = round(lit_count / max(1, total_roads) * 100, 1)

        # Surface material score (0-1): asphalt > concrete > paved > other
        surface_score = (
            (surfaces.get("asphalt",   0) * 0.85
           + surfaces.get("concrete",  0) * 0.75
           + surfaces.get("paved",     0) * 0.65
           + surfaces.get("sett",      0) * 0.55
           + surfaces.get("cobblestone",0)* 0.40) / max(1, total_roads)
        )

        result = {
            "city": city,
            "road_count": total_roads,
            "highway_types": highway_types,
            "surface_types": surfaces,
            "asphalt_pct": asphalt_pct,
            "surface_material_score": round(min(1, surface_score), 3),
            "avg_max_speed_kmph": avg_speed,
            "avg_lanes": avg_lanes,
            "avg_road_age_years": avg_age_yrs,
            "lit_roads_pct": lit_pct,
            "scraped_at": datetime.utcnow().isoformat(),
            "source": "osm_overpass",
        }
        await scrape_cache.set(cache_key, result)
        logger.info(f"OSM roads scraped for {city}: {total_roads} segments")
        return result


class OSMBridgeScraper:
    """Scrape bridge/flyover inventory from OSM."""

    CITY_BBOX = OSMRoadScraper.CITY_BBOX

    async def scrape(self, city: str) -> Dict[str, Any]:
        cache_key = f"osm_bridges:{city.lower().replace(' ', '_')}"
        cached = await scrape_cache.get(cache_key)
        if cached:
            return {**cached, "source": "redis_cache"}

        bbox = self.CITY_BBOX.get(city)
        if not bbox:
            return {"error": f"No bounding box for {city}"}

        s, w, n, e = bbox
        query = f"""
        [out:json][timeout:25];
        (
          way["bridge"="yes"]({s},{w},{n},{e});
          way["bridge"="viaduct"]({s},{w},{n},{e});
          node["man_made"="bridge"]({s},{w},{n},{e});
        );
        out tags;
        """

        data = await _overpass_query(query, timeout=30)
        if not data:
            return {"error": "Overpass query failed", "city": city}

        elements = data.get("elements", [])
        bridges  = [e for e in elements if e["type"] == "way"]

        materials, ages, bridge_types = {}, [], {}
        for b in bridges:
            tags = b.get("tags", {})
            mat = tags.get("material", tags.get("bridge:structure", "unknown"))
            materials[mat] = materials.get(mat, 0) + 1

            sd = tags.get("start_date", "")
            if sd and len(sd) >= 4:
                try:
                    ages.append(datetime.now().year - int(sd[:4]))
                except ValueError:
                    pass

            bt = tags.get("bridge", "yes")
            bridge_types[bt] = bridge_types.get(bt, 0) + 1

        avg_age = round(sum(ages) / len(ages), 1) if ages else None

        result = {
            "city": city,
            "bridge_count": len(bridges),
            "bridge_types": bridge_types,
            "materials": materials,
            "avg_bridge_age_years": avg_age,
            "bridges_over_40yr": sum(1 for a in ages if a >= 40),
            "scraped_at": datetime.utcnow().isoformat(),
            "source": "osm_overpass",
        }
        await scrape_cache.set(cache_key, result)
        logger.info(f"OSM bridges scraped for {city}: {len(bridges)} structures")
        return result


class OSMWaterwayScraper:
    """Scrape waterway proximity data from OSM."""

    CITY_BBOX = OSMRoadScraper.CITY_BBOX

    async def scrape(self, city: str) -> Dict[str, Any]:
        cache_key = f"osm_waterways:{city.lower().replace(' ', '_')}"
        cached = await scrape_cache.get(cache_key)
        if cached:
            return {**cached, "source": "redis_cache"}

        bbox = self.CITY_BBOX.get(city)
        if not bbox:
            return {"error": f"No bounding box for {city}"}

        s, w, n, e = bbox
        query = f"""
        [out:json][timeout:20];
        (
          way["waterway"~"river|canal|stream|drain|ditch"]({s},{w},{n},{e});
          relation["waterway"="river"]({s},{w},{n},{e});
        );
        out tags;
        """

        data = await _overpass_query(query, timeout=25)
        if not data:
            return {"error": "Overpass query failed", "city": city}

        elements = data.get("elements", [])
        waterways = [e for e in elements if e["type"] in ("way", "relation")]

        wtype_count = {}
        named_rivers = []
        for w in waterways:
            tags = w.get("tags", {})
            wt = tags.get("waterway", "unknown")
            wtype_count[wt] = wtype_count.get(wt, 0) + 1
            name = tags.get("name", "")
            if name and wt == "river":
                named_rivers.append(name)

        result = {
            "city": city,
            "total_waterways": len(waterways),
            "waterway_types": wtype_count,
            "major_rivers": list(set(named_rivers))[:5],
            "flood_risk_modifier": min(2.0, 1 + len(waterways) * 0.05),
            "scraped_at": datetime.utcnow().isoformat(),
            "source": "osm_overpass",
        }
        await scrape_cache.set(cache_key, result)
        logger.info(f"OSM waterways scraped for {city}: {len(waterways)} features")
        return result


async def scrape_city_infrastructure(city: str) -> Dict[str, Any]:
    """Run all OSM scrapers for a city concurrently."""
    road_scraper     = OSMRoadScraper()
    bridge_scraper   = OSMBridgeScraper()
    waterway_scraper = OSMWaterwayScraper()

    roads, bridges, waterways = await asyncio.gather(
        road_scraper.scrape(city),
        bridge_scraper.scrape(city),
        waterway_scraper.scrape(city),
        return_exceptions=True,
    )

    return {
        "city": city,
        "roads": roads if isinstance(roads, dict) else {"error": str(roads)},
        "bridges": bridges if isinstance(bridges, dict) else {"error": str(bridges)},
        "waterways": waterways if isinstance(waterways, dict) else {"error": str(waterways)},
        "scraped_at": datetime.utcnow().isoformat(),
    }
