import httpx
import asyncio
import json

async def test_custom_location():
    """Test prediction with a custom location not in the hardcoded list."""
    
    # Test location: Sydney, Australia (far from hardcoded cities)
    custom_locations = [
        {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
        {"name": "Singapore", "lat": 1.3521, "lon": 103.8198},
        {"name": "Bangkok", "lat": 13.7563, "lon": 100.5018},
    ]
    
    async with httpx.AsyncClient() as client:
        for loc in custom_locations:
            print(f"\nTesting {loc['name']} ({loc['lat']}, {loc['lon']})...")
            try:
                resp = await client.post(
                    "http://localhost:8000/api/v1/predict/",
                    params={
                        "lat": loc["lat"],
                        "lon": loc["lon"],
                        "infra_type": "roads"
                    },
                    timeout=15
                )
                data = resp.json()
                print(f"  Status: {resp.status_code}")
                print(f"  City detected: {data.get('city', 'N/A')}")
                print(f"  Failure probability: {data.get('failure_probability_pct', 'N/A')}%")
                print(f"  Data source: {data.get('sources_used', {}).get('traffic', 'unknown')}")
            except Exception as e:
                print(f"  Error: {e}")

asyncio.run(test_custom_location())
