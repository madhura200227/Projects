import httpx
import asyncio
import json

async def test_apis():
    results = {}
    
    # TomTom Traffic API
    print("Testing TomTom Traffic API...")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json",
                params={
                    "key": "iWrYOmhuyEPShwv3gQeeulqVJ3FDP4Nl",
                    "point": "19.076,72.877"
                }
            )
            results["TomTom Traffic"] = {
                "status_code": resp.status_code,
                "working": resp.status_code < 400,
                "message": resp.json().get("error", "OK") if resp.status_code >= 400 else "Connected"
            }
    except Exception as e:
        results["TomTom Traffic"] = {"working": False, "error": str(e)}
    
    # OpenWeatherMap Air Pollution API
    print("Testing OpenWeatherMap API...")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                "https://api.openweathermap.org/data/2.5/air_pollution",
                params={
                    "lat": "19.076",
                    "lon": "72.877",
                    "appid": "a51d9cd0fa72dcb767656b2d32c30b63"
                }
            )
            results["OpenWeatherMap"] = {
                "status_code": resp.status_code,
                "working": resp.status_code < 400,
                "message": resp.json().get("message", "OK") if resp.status_code >= 400 else "Connected"
            }
    except Exception as e:
        results["OpenWeatherMap"] = {"working": False, "error": str(e)}
    
    # NOAA API
    print("Testing NOAA Flood API...")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                "https://www.ncdc.noaa.gov/cdo-web/api/v2/data",
                params={
                    "datasetid": "GHCND",
                    "stationid": "GHCND:USW00012920",
                    "startdate": "2024-01-01",
                    "enddate": "2024-01-08",
                    "limit": 1000
                },
                headers={"token": "LCsIQYuYfoUFNkOlebGcvkLCxlzhSzjJ"}
            )
            results["NOAA Flood"] = {
                "status_code": resp.status_code,
                "working": resp.status_code < 400,
                "message": resp.json().get("message", "OK") if resp.status_code >= 400 else "Connected"
            }
    except Exception as e:
        results["NOAA Flood"] = {"working": False, "error": str(e)}
    
    # Free APIs (no key needed)
    print("Testing OSM Overpass API...")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(
                "https://overpass-api.de/api/interpreter",
                content='[bbox:72.7,18.8,73.0,19.3];(way["highway"];);out geom;',
                timeout=10
            )
            results["OSM Overpass"] = {
                "status_code": resp.status_code,
                "working": resp.status_code < 400,
                "message": "Connected" if resp.status_code < 400 else "Error"
            }
    except Exception as e:
        results["OSM Overpass"] = {"working": False, "error": str(e)}
    
    print("Testing ReliefWeb API...")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                "https://api.reliefweb.int/v1/disasters",
                params={"limit": 1}
            )
            results["ReliefWeb"] = {
                "status_code": resp.status_code,
                "working": resp.status_code < 400,
                "message": "Connected" if resp.status_code < 400 else "Error"
            }
    except Exception as e:
        results["ReliefWeb"] = {"working": False, "error": str(e)}
    
    print("Testing World Bank API...")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                "https://api.worldbank.org/v2/country/IND/indicator/IS.ROD.PAVE.ZS",
                params={"format": "json", "per_page": 1}
            )
            results["World Bank"] = {
                "status_code": resp.status_code,
                "working": resp.status_code < 400,
                "message": "Connected" if resp.status_code < 400 else "Error"
            }
    except Exception as e:
        results["World Bank"] = {"working": False, "error": str(e)}
    
    # Print results
    print("\n" + "="*70)
    print("API KEY TEST RESULTS")
    print("="*70)
    for api, result in results.items():
        status = "✅ WORKING" if result.get("working") else "❌ FAILED"
        print(f"\n{api:<25} {status}")
        if "status_code" in result:
            print(f"  Status: {result['status_code']}")
        if "message" in result and result["message"] != "OK":
            print(f"  Message: {result['message']}")
        if "error" in result:
            print(f"  Error: {result['error']}")

asyncio.run(test_apis())
