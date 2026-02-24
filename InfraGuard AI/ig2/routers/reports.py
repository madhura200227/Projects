"""
routers/reports.py – /api/v1/reports endpoints
Generate structured city/infrastructure reports with historical trends.
"""

import logging
import asyncio
from datetime import datetime
from fastapi import APIRouter, Query, HTTPException

from ml.data_fetcher import DataAggregator
from ml.model import InfraGuardModel
from utils.cache import prediction_cache

logger = logging.getLogger("infraguard.reports")
router = APIRouter()

aggregator = DataAggregator()

VALID_CITIES = {"Mumbai", "Pune", "New York", "Tokyo"}
CITY_COORDS = {
    "Mumbai":   (19.076, 72.877),
    "Pune":     (18.520, 73.856),
    "New York": (40.712, -74.006),
    "Tokyo":    (35.689, 139.692),
}


@router.get("/city/{city}", summary="Full infrastructure risk report for a city")
async def city_report(
    request,
    city: str,
):
    """
    Generates a comprehensive infrastructure risk report for a city.
    Runs predictions for all 3 infra types (roads, bridges, pipelines).
    """
    if city not in VALID_CITIES:
        raise HTTPException(400, f"Unknown city. Valid: {sorted(VALID_CITIES)}")

    model: InfraGuardModel = request.app.state.model
    lat, lon = CITY_COORDS[city]

    agg = await aggregator.fetch_all(lat, lon)
    features = agg["features"]

    # Predict all 3 infra types concurrently
    infra_types = ["roads", "bridges", "pipelines"]
    predictions = {}
    for it in infra_types:
        res = model.predict(dict(features), infra_type=it)
        predictions[it] = {
            "failure_probability_pct": res["failure_probability_pct"],
            "probability_ci_low":      round(res["probability_low"] * 100),
            "probability_ci_high":     round(res["probability_high"] * 100),
            "risk_level":              res["risk_level"],
            "predicted_failure_years": res["predicted_failure_years"],
            "dominant_factors":        res["dominant_risk_factors"][:2],
        }

    # Overall city risk = weighted average (roads 50%, bridges 30%, pipelines 20%)
    overall_risk = (
        predictions["roads"]["failure_probability_pct"] * 0.50
        + predictions["bridges"]["failure_probability_pct"] * 0.30
        + predictions["pipelines"]["failure_probability_pct"] * 0.20
    )

    return {
        "city": city,
        "report_generated_at": datetime.utcnow().isoformat(),
        "overall_risk_pct": round(overall_risk, 1),
        "infrastructure": predictions,
        "key_metrics": {
            "avg_daily_traffic":  features.get("avg_daily_traffic"),
            "aqi":                features.get("aqi"),
            "flood_events_5yr":   features.get("flood_events_5yr"),
            "road_age_years":     features.get("road_age_years"),
            "seismic_zone":       features.get("seismic_zone"),
            "rainfall_mm_annual": features.get("rainfall_mm_annual"),
        },
        "data_sources": agg["sources_used"],
    }


@router.get("/compare", summary="Compare risk across all 4 monitored cities")
async def compare_cities(request):
    """
    Runs risk prediction for all 4 monitored cities and returns a comparison.
    Useful for the comparative risk dashboard widget.
    """
    model: InfraGuardModel = request.app.state.model
    cities = list(CITY_COORDS.keys())

    async def _predict_city(city):
        lat, lon = CITY_COORDS[city]
        agg = await aggregator.fetch_all(lat, lon)
        res = model.predict(agg["features"], infra_type="roads")
        return {
            "city": city,
            "failure_probability_pct": res["failure_probability_pct"],
            "risk_level": res["risk_level"],
            "predicted_failure_years": res["predicted_failure_years"],
            "confidence_score": res["confidence_score"],
            "top_factor": res["dominant_risk_factors"][0] if res["dominant_risk_factors"] else "N/A",
        }

    results = await asyncio.gather(*[_predict_city(c) for c in cities], return_exceptions=True)
    valid = [r for r in results if isinstance(r, dict)]
    valid.sort(key=lambda x: x["failure_probability_pct"], reverse=True)

    return {
        "cities": valid,
        "highest_risk": valid[0]["city"] if valid else None,
        "lowest_risk": valid[-1]["city"] if valid else None,
        "generated_at": datetime.utcnow().isoformat(),
    }
