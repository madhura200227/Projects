import asyncio
import httpx
from utils.config import settings

async def main():
    lat, lon = 19.076, 72.877  # Mumbai
    token = settings.WAQI_API_KEY
    print(f"Using WAQI token from settings: '{token}'")
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params={"token": token})
            print("Status:", r.status_code)
            try:
                print("Response:", r.json())
            except Exception:
                print("Raw response:", r.text[:1000])
    except Exception as e:
        print("Request failed:", e)

if __name__ == '__main__':
    asyncio.run(main())
