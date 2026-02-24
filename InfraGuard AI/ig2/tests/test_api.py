"""
tests/test_api.py – Integration tests for InfraGuard AI v2 API
Run: pytest tests/ -v --asyncio-mode=auto
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from main import app


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture(scope="session")
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── Health ────────────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert "2.1.0" in data["version"]


@pytest.mark.anyio
async def test_root(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert "InfraGuard" in r.json()["message"]


# ── Prediction ────────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_predict_roads(client):
    r = await client.post(
        "/api/v1/predict/",
        params={"lat": 19.076, "lon": 72.877, "infra_type": "roads"},
    )
    assert r.status_code == 200
    data = r.json()
    assert 0 <= data["failure_probability_pct"] <= 100
    assert data["risk_level"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    assert data["city"] == "Mumbai"
    assert "shap_values" in data
    assert len(data["shap_values"]) > 0
    assert "probability_ci_low" in data
    assert "probability_ci_high" in data


@pytest.mark.anyio
async def test_predict_bridges(client):
    r = await client.post(
        "/api/v1/predict/",
        params={"lat": 40.712, "lon": -74.006, "infra_type": "bridges"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["city"] == "New York"
    assert data["infra_type"] == "bridges"
    assert data["failure_probability_pct"] >= 0


@pytest.mark.anyio
async def test_predict_invalid_infra_type(client):
    r = await client.post(
        "/api/v1/predict/",
        params={"lat": 19.076, "lon": 72.877, "infra_type": "trains"},
    )
    assert r.status_code == 400


@pytest.mark.anyio
async def test_predict_with_manual_overrides(client):
    r = await client.post(
        "/api/v1/predict/",
        params={
            "lat": 19.076, "lon": 72.877,
            "infra_type": "roads",
            "aqi": 250,
            "avg_daily_traffic": 120000,
        },
    )
    assert r.status_code == 200
    data = r.json()
    # High AQI + high traffic → should push risk higher
    assert data["failure_probability_pct"] > 50


# ── Data sources ─────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_api_status(client):
    r = await client.get("/api/v1/data/apis/status")
    assert r.status_code == 200
    data = r.json()
    assert "apis" in data
    assert data["total_apis"] >= 8
    # Free APIs should always be configured
    free_names = {a["name"] for a in data["apis"] if a["configured"]}
    assert "OSM Overpass" in free_names


@pytest.mark.anyio
async def test_live_data_mumbai(client):
    r = await client.get("/api/v1/data/live/Mumbai")
    assert r.status_code == 200
    data = r.json()
    assert data["city"] == "Mumbai"
    assert "features" in data
    assert "aqi" in data["features"]


@pytest.mark.anyio
async def test_live_data_invalid_city(client):
    r = await client.get("/api/v1/data/live/Atlantis")
    assert r.status_code == 400


# ── Alerts ────────────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_get_alerts(client):
    r = await client.get("/api/v1/alerts/")
    assert r.status_code == 200
    data = r.json()
    assert "alerts" in data
    assert isinstance(data["alerts"], list)


@pytest.mark.anyio
async def test_get_alerts_city_filter(client):
    r = await client.get("/api/v1/alerts/", params={"city": "Tokyo"})
    assert r.status_code == 200
    data = r.json()
    # All returned alerts should be for Tokyo
    for alert in data["alerts"]:
        assert alert.get("city") == "Tokyo"


@pytest.mark.anyio
async def test_alert_stats(client):
    r = await client.get("/api/v1/alerts/stats")
    assert r.status_code == 200
    data = r.json()
    assert "stats" in data
    for city in ["Mumbai", "Pune", "New York", "Tokyo"]:
        assert city in data["stats"]


# ── Scraper ───────────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_worldbank_scrape(client):
    r = await client.get("/api/v1/scraper/worldbank/Mumbai")
    assert r.status_code == 200
    data = r.json()
    assert "city" in data


@pytest.mark.anyio
async def test_reliefweb_floods(client):
    r = await client.get("/api/v1/scraper/floods/New York", params={"years": 3})
    assert r.status_code == 200
    data = r.json()
    assert "city" in data
    assert "flood_events_count" in data or "source" in data


# ── Model features ────────────────────────────────────────────────────────────
def test_model_feature_count():
    from ml.model import FEATURE_NAMES
    assert len(FEATURE_NAMES) == 15, f"Expected 15 features, got {len(FEATURE_NAMES)}"


def test_model_weights_sum():
    """Weights for positive and negative features should be balanced."""
    from ml.model import WEIGHTS
    pos = sum(v for v in WEIGHTS.values() if v > 0)
    neg = sum(abs(v) for v in WEIGHTS.values() if v < 0)
    # Roughly balanced (positive features drive risk, negative reduce it)
    assert 0.5 <= pos / max(neg, 0.01) <= 3.0, "Weight imbalance detected"


def test_normalise():
    from ml.model import _normalise
    assert _normalise(0, "aqi") == 0.0
    assert _normalise(500, "aqi") == 1.0
    assert _normalise(250, "aqi") == 0.5
    assert _normalise(-100, "aqi") == 0.0   # clamps at 0


def test_sigmoid_bounds():
    from ml.model import _sigmoid
    for x in [-10, -1, 0, 1, 10]:
        s = _sigmoid(x)
        assert 0 < s < 1


def test_heuristic_prediction():
    from ml.model import XGBoostTabularModel
    model = XGBoostTabularModel("nonexistent.json")
    model.load()

    # High-risk Mumbai-like features → should give HIGH risk
    features = {
        "avg_daily_traffic": 100000, "heavy_vehicle_pct": 40,
        "aqi": 200, "flood_events_5yr": 12, "road_age_years": 35,
        "rainfall_mm_annual": 2400, "soil_moisture": 75,
        "surface_material_score": 0.4, "drainage_quality_score": 0.25,
        "image_damage_score": 0, "population_density": 20000,
        "temperature_extremes": 15, "seismic_zone": 3,
        "congestion_pct": 65, "nearby_waterways": 8,
    }
    score_high = model.predict(features, "roads")

    # Low-risk Tokyo-like features
    features_low = {
        "avg_daily_traffic": 50000, "heavy_vehicle_pct": 15,
        "aqi": 40, "flood_events_5yr": 2, "road_age_years": 10,
        "rainfall_mm_annual": 1000, "soil_moisture": 40,
        "surface_material_score": 0.95, "drainage_quality_score": 0.90,
        "image_damage_score": 0, "population_density": 5000,
        "temperature_extremes": 25, "seismic_zone": 5,
        "congestion_pct": 20, "nearby_waterways": 1,
    }
    score_low = model.predict(features_low, "roads")

    # Mumbai should score higher than Tokyo
    from ml.model import _sigmoid
    assert _sigmoid(score_high) > _sigmoid(score_low), (
        f"Expected Mumbai risk > Tokyo risk. Got {_sigmoid(score_high):.2f} vs {_sigmoid(score_low):.2f}"
    )
