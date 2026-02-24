"""
routers/predict.py – /api/v1/predict endpoints (v2)
"""

import logging
import io
import numpy as np
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Query

from ml.data_fetcher import DataAggregator
from ml.model import InfraGuardModel, FEATURE_NAMES
from utils.cache import prediction_cache

logger = logging.getLogger("infraguard.predict")
router = APIRouter()
aggregator = DataAggregator()


def _currency_format(amount_usd: float, city: str) -> str:
    """Format monetary values in local currency."""
    if "Mumbai" in city or "Pune" in city:
        cr = amount_usd * 83 / 1e7
        return f"₹{cr:.1f} Cr"
    if "Tokyo" in city:
        b_jpy = amount_usd * 149 / 1e8
        return f"¥{b_jpy:.2f}B"
    m_usd = amount_usd / 1e6
    return f"${m_usd:.1f}M"


@router.post("/", summary="Predict infrastructure failure probability")
async def predict(
    request: Request,
    lat: float = Query(..., description="Latitude", ge=-90, le=90),
    lon: float = Query(..., description="Longitude", ge=-180, le=180),
    infra_type: str = Query("roads", description="roads | bridges | pipelines"),
    city: Optional[str] = Query(None, description="City name override"),
    # Manual feature overrides
    avg_daily_traffic: Optional[float] = Query(None),
    heavy_vehicle_pct: Optional[float] = Query(None),
    aqi_override: Optional[float] = Query(None, alias="aqi"),
    flood_events_5yr: Optional[float] = Query(None),
    road_age_years: Optional[float] = Query(None),
    image_damage_score: Optional[float] = Query(None),
):
    """
    Main prediction endpoint.

    Fetches live data from all configured APIs (TomTom/HERE, WAQI/OpenWeather,
    ReliefWeb, USGS), runs XGBoost + ensemble model, returns:
    - Failure probability with 95% confidence interval
    - Risk level (CRITICAL/HIGH/MEDIUM/LOW)
    - Time-to-failure estimate
    - SHAP explainability values
    - Tailored government recommendations
    - Cost-benefit estimates in local currency
    """
    if infra_type not in ("roads", "bridges", "pipelines"):
        raise HTTPException(400, "infra_type must be 'roads', 'bridges', or 'pipelines'")

    model: InfraGuardModel = request.app.state.model

    cache_key = f"pred:{lat:.3f}:{lon:.3f}:{infra_type}"
    cached = await prediction_cache.get(cache_key)
    if cached:
        logger.info(f"Prediction cache hit: {cache_key}")
        return {**cached, "cache_hit": True}

    manual_overrides = {
        "avg_daily_traffic": avg_daily_traffic,
        "heavy_vehicle_pct": heavy_vehicle_pct,
        "aqi": aqi_override,
        "flood_events_5yr": flood_events_5yr,
        "road_age_years": road_age_years,
        "image_damage_score": image_damage_score,
    }
    manual_overrides = {k: v for k, v in manual_overrides.items() if v is not None}

    try:
        agg = await aggregator.fetch_all(lat, lon, manual_overrides=manual_overrides)
    except Exception as e:
        logger.error(f"Data aggregation failed: {e}")
        raise HTTPException(500, f"Data aggregation failed: {e}")

    resolved_city = city or agg["city"]
    features = agg["features"]
    sources  = agg["sources_used"]

    # Run model
    result = model.predict(features, infra_type=infra_type)

    # Recommendations
    recs = model.compute_recommendations(features, result["risk_level"], infra_type)

    # Cost estimates with ROI
    prob_pct = result["failure_probability_pct"]
    base_cost_usd = prob_pct * 1_200  # $1200 per risk point as base unit

    cost_actions = [
        {"action": "Emergency Structural Repair",     "mult": 38, "save_mult": 220, "months": 3,  "pri": "CRITICAL"},
        {"action": "Surface Resurfacing Program",      "mult": 110,"save_mult": 650, "months": 6,  "pri": "CRITICAL"},
        {"action": "Drainage Infrastructure Upgrade",  "mult": 62, "save_mult": 400, "months": 8,  "pri": "HIGH"},
        {"action": "Sensor & SHM Network",             "mult": 20, "save_mult": 85,  "months": 4,  "pri": "HIGH"},
        {"action": "Preventive Maintenance Program",   "mult": 16, "save_mult": 70,  "months": 5,  "pri": "MEDIUM"},
        {"action": "Material Reinforcement Overlay",   "mult": 28, "save_mult": 140, "months": 7,  "pri": "MEDIUM"},
    ]

    cost_estimates = []
    for i, ca in enumerate(cost_actions):
        cost_usd  = ca["mult"] * base_cost_usd
        save_usd  = ca["save_mult"] * base_cost_usd
        roi       = round(save_usd / max(1, cost_usd), 1)
        cost_estimates.append({
            "rank": i + 1,
            "action":              ca["action"],
            "priority":            ca["pri"],
            "cost_estimate":       _currency_format(cost_usd, resolved_city),
            "potential_savings":   _currency_format(save_usd, resolved_city),
            "timeline":            f"{ca['months']} months",
            "roi_multiplier":      roi,
        })

    # Economic / carbon impact
    carbon_cost  = _currency_format(prob_pct * 22_000, resolved_city)
    economic_cost= _currency_format(prob_pct * 160_000, resolved_city)

    response = {
        "city":                      resolved_city,
        "lat":                       lat,
        "lon":                       lon,
        "infra_type":                infra_type,
        "failure_probability":       result["failure_probability"],
        "failure_probability_pct":   result["failure_probability_pct"],
        "probability_ci_low":        result["probability_low"],
        "probability_ci_high":       result["probability_high"],
        "uncertainty_std":           result["uncertainty_std"],
        "risk_level":                result["risk_level"],
        "confidence_score":          result["confidence_score"],
        "predicted_failure_years":   result["predicted_failure_years"],
        "predicted_failure_date":    result["predicted_failure_date"],
        "warning_threshold_date":    result["warning_threshold_date"],
        "features_used":             features,
        "shap_values":               result["shap_values"],
        "dominant_risk_factors":     result["dominant_risk_factors"],
        "recommendations":           recs,
        "cost_estimates":            cost_estimates,
        "carbon_cost_annual":        carbon_cost,
        "economic_delay_cost_annual":economic_cost,
        "model_version":             result["model_version"],
        "data_sources_used":         sources,
        "fetched_at":                agg["fetched_at"],
        "cache_hit":                 False,
    }

    await prediction_cache.set(cache_key, response)
    return response


@router.post("/image", summary="Analyse infrastructure image for damage")
async def predict_image(
    request: Request,
    file: UploadFile = File(..., description="Road/bridge/pipeline image (JPEG/PNG/WebP)"),
    infra_type: Optional[str] = Query(None, description="Optional type hint: roads|bridges|pipelines|building|manhole"),
):
    """
    Upload an infrastructure image.
    - Auto-detects infrastructure type via CNN classifier
    - Rejects non-infrastructure images with error
    - Returns per-category damage scores + overall damage
    - Heatmap overlay generated via OpenCV (if available)
    """
    model: InfraGuardModel = request.app.state.model

    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/jpg"):
        raise HTTPException(400, "Only JPEG / PNG / WebP images are supported")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 10 MB)")

    try:
        from PIL import Image
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_arr = np.array(pil)
    except Exception as e:
        raise HTTPException(422, f"Invalid image file: {e}")

    # Classify infrastructure type
    classification = model.cnn.classify_infra_type(img_arr)
    detected_type  = infra_type or classification["detected_type"]
    confidence     = classification["confidence"]

    # Reject non-infrastructure images (low confidence = wrong type)
    VALID_TYPES = {"road", "bridge", "pipeline", "building", "manhole"}
    if detected_type not in VALID_TYPES or confidence < 0.55:
        raise HTTPException(422, {
            "error": "Non-infrastructure image detected",
            "detected_type": detected_type,
            "confidence": confidence,
            "message": "Please upload an image of a road, bridge, pipeline, building, or manhole.",
        })

    # Damage detection
    scores = model.cnn.predict_from_array(img_arr)

    # Type-specific scoring weights
    WEIGHTS_BY_TYPE = {
        "road":     {"crack": 0.30, "pothole": 0.25, "wear": 0.22, "water": 0.13, "deformation": 0.10},
        "bridge":   {"crack": 0.25, "pothole": 0.08, "wear": 0.18, "water": 0.24, "deformation": 0.25},
        "pipeline": {"crack": 0.20, "pothole": 0.05, "wear": 0.18, "water": 0.32, "deformation": 0.25},
        "building": {"crack": 0.35, "pothole": 0.03, "wear": 0.20, "water": 0.22, "deformation": 0.20},
        "manhole":  {"crack": 0.28, "pothole": 0.20, "wear": 0.25, "water": 0.17, "deformation": 0.10},
    }
    weights = WEIGHTS_BY_TYPE.get(detected_type, WEIGHTS_BY_TYPE["road"])
    overall = sum(scores.get(k, 0) * w for k, w in weights.items())

    if overall >= 75:   level = "critical";  rec_prefix = "🚨 CRITICAL"
    elif overall >= 50: level = "severe";    rec_prefix = "⛔ SEVERE"
    elif overall >= 25: level = "moderate";  rec_prefix = "⚠️ MODERATE"
    else:               level = "mild";      rec_prefix = "✅ MILD"

    # Type-specific recommendation
    TYPE_ACTIONS = {
        "road":     "pavement condition index survey, crack sealing, and pothole patching",
        "bridge":   "non-destructive evaluation (NDE), load testing, and corrosion inspection",
        "pipeline": "CCTV pipe inspection, pressure testing, and cathodic protection check",
        "building": "structural engineer assessment, crack monitoring gauges, and foundation survey",
        "manhole":  "confined-space inspection, load capacity testing, and joint sealing",
    }

    recommendation = (
        f"{rec_prefix} damage detected on {detected_type}. "
        f"Recommended action: {TYPE_ACTIONS.get(detected_type, 'structural assessment')}. "
        f"Overall damage: {overall:.1f}%."
    )

    # Generate heatmap via OpenCV if available
    heatmap_b64 = None
    try:
        import cv2, base64
        gray     = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges    = cv2.Canny(blurred, 30, 100)
        # Colour overlay (green-yellow-red based on overall severity)
        heatmap  = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
        overlay  = cv2.addWeighted(img_arr[:,:,::-1], 0.7, heatmap, 0.3, 0)
        _, buf   = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        heatmap_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Heatmap generation failed: {e}")

    return {
        "detected_infra_type":       detected_type,
        "classification_confidence": round(confidence, 3),
        "classification_probs":      classification.get("probabilities", {}),
        "crack_severity_pct":        round(scores.get("crack", 0), 1),
        "pothole_detection_pct":     round(scores.get("pothole", 0), 1),
        "surface_wear_pct":          round(scores.get("wear", 0), 1),
        "water_damage_pct":          round(scores.get("water", 0), 1),
        "structural_deformation_pct":round(scores.get("deformation", 0), 1),
        "overall_damage_score":      round(overall, 1),
        "damage_level":              level,
        "recommendation":            recommendation,
        "heatmap_base64":            heatmap_b64,
        "analysed_at":               datetime.utcnow().isoformat(),
    }


@router.post("/batch", summary="Batch prediction for multiple locations")
async def predict_batch(
    request: Request,
    locations: list,
):
    """
    Batch prediction for up to 10 lat/lon locations.
    Useful for city-wide infrastructure dashboard updates.
    """
    if len(locations) > 10:
        raise HTTPException(400, "Maximum 10 locations per batch request")

    model: InfraGuardModel = request.app.state.model
    results = []

    for loc in locations:
        try:
            lat = float(loc.get("lat", 0))
            lon = float(loc.get("lon", 0))
            infra_type = loc.get("infra_type", "roads")
            agg = await aggregator.fetch_all(lat, lon)
            res = model.predict(agg["features"], infra_type=infra_type)
            results.append({
                "lat": lat, "lon": lon,
                "infra_type": infra_type,
                "city": agg["city"],
                "failure_probability_pct": res["failure_probability_pct"],
                "risk_level": res["risk_level"],
                "predicted_failure_years": res["predicted_failure_years"],
            })
        except Exception as e:
            results.append({"lat": loc.get("lat"), "lon": loc.get("lon"), "error": str(e)})

    return {"results": results, "count": len(results)}
