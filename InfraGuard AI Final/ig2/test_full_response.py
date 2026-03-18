import httpx
import asyncio
import json

async def test_api_response():
    """Test the API and show full response."""
    
    test_cases = [
        {"name": "New York", "lat": 40.7128, "lon": -74.0060},
        {"name": "London", "lat": 51.5074, "lon": -0.1278},
        {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
    ]
    
    async with httpx.AsyncClient() as client:
        for loc in test_cases:
            print(f"\n{'='*70}")
            print(f"Testing {loc['name']} ({loc['lat']}, {loc['lon']})")
            print('='*70)
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
                print(f"Status: {resp.status_code}")
                print(f"City: {data.get('city', 'N/A')}")
                print(f"Failure Probability: {data.get('failure_probability_pct', 'N/A')}%")
                print(f"Risk Level: {data.get('risk_level', 'N/A')}")
                print(f"Risk Score: {data.get('risk_score', 'N/A')}")
                print(f"Cache Hit: {data.get('cache_hit', False)}")
                print(f"\nTop 3 Contributing Factors:")
                shap_vals = data.get('shap_values', [])
                for i, item in enumerate(shap_vals[:3], 1):
                    print(f"  {i}. {item.get('feature', 'Unknown')}: {item.get('shap_value', 0):.3f}")
                print(f"\nData Sources Used:")
                sources = data.get('sources_used', {})
                for key, val in sources.items():
                    print(f"  {key}: {val}")
            except Exception as e:
                print(f"Error: {e}")

asyncio.run(test_api_response())
