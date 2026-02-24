"""
utils/tasks.py – Celery background task definitions
Periodic tasks for data refresh, model retraining triggers, and alert delivery.
"""

import asyncio
import logging
from datetime import datetime

logger = logging.getLogger("infraguard.tasks")

# ── Celery app ────────────────────────────────────────────────────────────────
try:
    from celery import Celery
    from celery.schedules import crontab
    from utils.config import settings

    celery_app = Celery(
        "infraguard",
        broker=settings.CELERY_BROKER,
        backend=settings.CELERY_BACKEND,
    )

    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        beat_schedule={
            # Refresh city data every 15 minutes
            "refresh-city-data": {
                "task": "utils.tasks.refresh_all_cities",
                "schedule": crontab(minute="*/15"),
            },
            # Clear old prediction cache hourly
            "clear-prediction-cache": {
                "task": "utils.tasks.clear_stale_cache",
                "schedule": crontab(minute=0),
            },
            # Full OSM scrape daily at 3am UTC
            "daily-osm-scrape": {
                "task": "utils.tasks.daily_osm_scrape",
                "schedule": crontab(hour=3, minute=0),
            },
        },
    )

    @celery_app.task(name="utils.tasks.refresh_all_cities", bind=True, max_retries=3)
    def refresh_all_cities(self):
        """Refresh live data cache for all monitored cities."""
        from ml.data_fetcher import DataAggregator, CITY_PROFILES

        async def _run():
            agg = DataAggregator()
            for city, profile in CITY_PROFILES.items():
                try:
                    lat, lon = profile["lat"], profile["lon"]
                    result = await agg.fetch_all(lat, lon)
                    logger.info(f"✅ Refreshed data for {city} at {datetime.utcnow().isoformat()}")
                except Exception as e:
                    logger.error(f"Failed to refresh {city}: {e}")

        asyncio.run(_run())
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

    @celery_app.task(name="utils.tasks.clear_stale_cache")
    def clear_stale_cache():
        """Redis TTLs handle expiry automatically, this just logs cache stats."""
        logger.info(f"Cache maintenance run at {datetime.utcnow().isoformat()}")
        return {"status": "ok"}

    @celery_app.task(name="utils.tasks.daily_osm_scrape")
    def daily_osm_scrape():
        """Full OSM infrastructure scrape for all cities, runs daily."""
        from scrapers.osm_scraper import scrape_city_infrastructure

        async def _run():
            for city in ["Mumbai", "Pune", "New York", "Tokyo"]:
                try:
                    result = await scrape_city_infrastructure(city)
                    road_count = result.get("roads", {}).get("road_count", "?")
                    logger.info(f"✅ OSM scrape complete for {city}: {road_count} roads")
                except Exception as e:
                    logger.error(f"OSM scrape failed for {city}: {e}")

        asyncio.run(_run())
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

    @celery_app.task(name="utils.tasks.predict_city", bind=True)
    def predict_city(self, city: str, infra_type: str = "roads"):
        """Run a prediction for a city asynchronously (called from API)."""
        from ml.data_fetcher import DataAggregator, CITY_PROFILES
        from ml.model import InfraGuardModel

        async def _run():
            profile = CITY_PROFILES.get(city, {})
            lat = profile.get("lat", 0)
            lon = profile.get("lon", 0)
            agg = DataAggregator()
            data = await agg.fetch_all(lat, lon)
            model = InfraGuardModel()
            model.load()
            result = model.predict(data["features"], infra_type=infra_type)
            return {
                "city": city,
                "infra_type": infra_type,
                "failure_probability_pct": result["failure_probability_pct"],
                "risk_level": result["risk_level"],
                "timestamp": datetime.utcnow().isoformat(),
            }

        return asyncio.run(_run())

except ImportError:
    logger.info("Celery not installed — background tasks disabled. Install: pip install celery[redis]")
    celery_app = None
