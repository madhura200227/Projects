"""
routers/alerts.py – Real-time alert streaming via WebSocket + REST
===================================================================
Alert sources:
  - Model-triggered: when predictions exceed thresholds
  - API-triggered: TomTom/HERE incident feeds
  - Scraper-triggered: ReliefWeb flood events
  - Periodic: Background task scans all monitored cities every 10 min
"""

import asyncio
import logging
import json
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Query, HTTPException

from utils.cache import alert_cache
from utils.config import settings

logger = logging.getLogger("infraguard.alerts")
router = APIRouter()

# ── WebSocket connection manager ─────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, WebSocket] = {}  # client_id → ws
        self.city_subscriptions: Dict[str, Set[str]] = {}  # city → set of client_ids

    async def connect(self, ws: WebSocket, client_id: str, city: Optional[str] = None):
        await ws.accept()
        self.active[client_id] = ws
        if city:
            if city not in self.city_subscriptions:
                self.city_subscriptions[city] = set()
            self.city_subscriptions[city].add(client_id)
        logger.info(f"WebSocket connected: {client_id}, city={city}")

    def disconnect(self, client_id: str):
        self.active.pop(client_id, None)
        for city_set in self.city_subscriptions.values():
            city_set.discard(client_id)
        logger.info(f"WebSocket disconnected: {client_id}")

    async def send_to_client(self, client_id: str, message: Dict):
        ws = self.active.get(client_id)
        if ws:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.debug(f"Send failed to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast_to_city(self, city: str, message: Dict):
        clients = list(self.city_subscriptions.get(city, set()))
        tasks = [self.send_to_client(cid, message) for cid in clients]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def broadcast_all(self, message: Dict):
        clients = list(self.active.keys())
        tasks = [self.send_to_client(cid, message) for cid in clients]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


manager = ConnectionManager()

# ── Alert template generator ─────────────────────────────────────────────────
class AlertEngine:
    """
    Generates alerts from live API data and model predictions.
    """

    CITY_COORDS = {
        "Mumbai":    (19.076, 72.877),
        "Pune":      (18.520, 73.856),
        "Delhi":     (28.613, 77.209),
        "Bangalore": (12.971, 77.594),
        "New York":  (40.712, -74.006),
        "Tokyo":     (35.689, 139.692),
        "London":    (51.507, -0.127),
        "Singapore": (1.352, 103.819),
        "Dubai":     (25.204, 55.270),
    }

    def _generate_stable_id(self, data: Dict) -> str:
        """Create a stable hash-based ID from alert content."""
        # Significant fields that define a unique event
        key_content = {
            "city": data.get("city"),
            "type": data.get("type"),
            "title": data.get("title"),
            "description": data.get("description"),
            "source": data.get("source"),
        }
        # For ReliefWeb, use the original event date if available
        if data.get("source") == "reliefweb":
             key_content["original_date"] = data.get("timestamp")
        
        serialised = json.dumps(key_content, sort_keys=True)
        return hashlib.md5(serialised.encode()).hexdigest()

    async def fetch_traffic_incidents(self, city: str) -> List[Dict]:
        """Fetch real traffic incidents from HERE Incidents API. Fallback to mock for demo."""
        alerts = []
        lat, lon = self.CITY_COORDS.get(city, (0, 0))

        # Real API check
        if settings.HERE_API_KEY and "your_here" not in settings.HERE_API_KEY.lower():
            try:
                import httpx
                async with httpx.AsyncClient(timeout=10) as c:
                    r = await c.get(
                        "https://data.traffic.hereapi.com/v7/incidents",
                        params={
                            "apiKey": settings.HERE_API_KEY,
                            "in": f"circle:{lat},{lon};r=20000",
                            "locationReferencing": "none",
                        },
                    )
                    r.raise_for_status()
                    items = r.json().get("results", [])

                    for item in items[:5]:
                        desc   = item.get("incidentDetails", {}).get("description", {}).get("value", "Traffic incident")
                        impact = item.get("incidentDetails", {}).get("impact", "unknown")
                        sev    = "CRITICAL" if impact == "blocking" else "HIGH" if impact == "heavy" else "INFO"
                        alert_data = {
                            "city": city,
                            "severity": sev,
                            "type": "traffic_incident",
                            "title": f"🚧 Traffic Incident – {city}",
                            "description": desc,
                            "source": "here_incidents",
                        }
                        alerts.append({
                            "id": self._generate_stable_id(alert_data),
                            **alert_data,
                            "timestamp": datetime.utcnow().isoformat(),
                        })
            except Exception as e:
                logger.debug(f"HERE incidents error for {city}: {e}")

        # Fallback: Mock alerts for demo purposes if no real alerts found
        if not alerts:
            import random
            if random.random() < 0.3:  # 30% chance each poll to generate a mock live incident
                mock_incidents = [
                    "Slow traffic on major corridor due to surface wear",
                    "Emergency maintenance on bridge expansion joints",
                    "Utility line repair causing lane restrictions",
                    "Heavy vehicle load causing temporary slowdowns",
                    "Minor road surface failure reported by commuters",
                ]
                alert_data = {
                    "city": city,
                    "severity": random.choice(["HIGH", "INFO"]),
                    "type": "traffic_incident",
                    "title": f"🚧 Demo Alert: Traffic – {city}",
                    "description": random.choice(mock_incidents),
                    "source": "demo_engine",
                }
                alerts.append({
                    "id": self._generate_stable_id(alert_data),
                    **alert_data,
                    "timestamp": datetime.utcnow().isoformat(),
                })

        return alerts

    async def fetch_flood_alerts(self, city: str) -> List[Dict]:
        """Check ReliefWeb for recent flood events."""
        alerts = []
        country_map = {
            "Mumbai": "India", "Pune": "India", "Delhi": "India", "Bangalore": "India",
            "New York": "United States", "Tokyo": "Japan", "London": "United Kingdom",
            "Singapore": "Singapore", "Dubai": "United Arab Emirates",
        }
        country = country_map.get(city, "")
        if not country:
            return alerts

        cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00+00:00")
        try:
            import httpx
            payload = {
                "filter": {
                    "operator": "AND",
                    "conditions": [
                        {"field": "type.name", "value": "Flood"},
                        {"field": "date.created", "value": {"from": cutoff}},
                        {"field": "country.name", "value": country},
                    ]
                },
                "limit": 3,
                "sort": ["date.created:desc"],
            }
            async with httpx.AsyncClient(timeout=10) as c:
                # appname is REQUIRED by ReliefWeb API
                r = await c.post("https://api.reliefweb.int/v1/disasters?appname=infraguard", json=payload)
                r.raise_for_status()
                events = r.json().get("data", [])

                for event in events:
                    f = event.get("fields", {})
                    alert_data = {
                        "city": city,
                        "severity": "HIGH",
                        "type": "flood_event",
                        "title": f"🌊 Flood Alert – {country}",
                        "description": f["name"] if "name" in f else "Flood event reported",
                        "source": "reliefweb",
                        "timestamp": f.get("date", {}).get("created", datetime.utcnow().isoformat()),
                    }
                    alerts.append({
                        "id": self._generate_stable_id(alert_data),
                        **alert_data,
                    })
        except Exception as e:
            logger.debug(f"ReliefWeb alert error for {city}: {e}")

        return alerts

    async def check_aqi_threshold(self, city: str) -> Optional[Dict]:
        """Return alert if AQI exceeds critical threshold."""
        from utils.cache import aqi_cache
        lat, lon = self.CITY_COORDS.get(city, (0, 0))
        cached = await aqi_cache.get(f"{lat:.3f}:{lon:.3f}")
        if not cached:
            return None

        aqi_val = cached.get("aqi", 0)
        if aqi_val >= 200:
            alert_data = {
                "city": city,
                "severity": "CRITICAL",
                "type": "aqi_hazardous",
                "title": f"🏭 Hazardous AQI – {city}",
                "description": f"AQI reached {aqi_val:.0f} ({cached.get('category','Hazardous')}). Infrastructure degradation risk +35% above baseline.",
                "source": "aqi_monitor",
                "aqi_value": aqi_val,
            }
            return {
                "id": self._generate_stable_id(alert_data),
                **alert_data,
                "timestamp": datetime.utcnow().isoformat(),
            }
            return {
                "id": self._generate_stable_id(alert_data),
                **alert_data,
                "timestamp": datetime.utcnow().isoformat(),
            }
        elif aqi_val >= 100:
            alert_data = {
                "city": city,
                "severity": "INFO",
                "type": "aqi_moderate",
                "title": f"🌫️ Moderate AQI – {city}",
                "description": f"AQI at {aqi_val:.0f}. Increased atmospheric particulate matter. Baseline monitoring continued.",
                "source": "aqi_monitor",
                "aqi_value": aqi_val,
            }
            return {
                "id": self._generate_stable_id(alert_data),
                **alert_data,
                "timestamp": datetime.utcnow().isoformat(),
            }
        return None

    async def generate_city_alerts(self, city: str) -> List[Dict]:
        """Gather all alert types for a city concurrently."""
        traffic_task = self.fetch_traffic_incidents(city)
        flood_task   = self.fetch_flood_alerts(city)
        aqi_task     = self.check_aqi_threshold(city)

        traffic_alerts, flood_alerts, aqi_alert = await asyncio.gather(
            traffic_task, flood_task, aqi_task, return_exceptions=True
        )

        alerts = []
        if isinstance(traffic_alerts, list): alerts.extend(traffic_alerts)
        if isinstance(flood_alerts, list):   alerts.extend(flood_alerts)
        if isinstance(aqi_alert, dict):      alerts.append(aqi_alert)

        # Sort: CRITICAL → HIGH → INFO
        sev_order = {"CRITICAL": 0, "HIGH": 1, "INFO": 2}
        alerts.sort(key=lambda a: sev_order.get(a.get("severity", "INFO"), 2))

        return alerts


alert_engine = AlertEngine()


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@router.websocket("/ws")
async def websocket_alerts(
    ws: WebSocket,
    city: Optional[str] = Query(None, description="Subscribe to a specific city"),
):
    """
    WebSocket endpoint for real-time alert streaming.

    Connect: ws://localhost:8000/api/v1/alerts/ws?city=Mumbai

    Message format sent to client:
    {
        "type": "alert" | "ping" | "connection",
        "alert": { ...alert object... }  // when type=="alert"
        "timestamp": "ISO-8601"
    }

    Client can send:
    { "action": "subscribe", "city": "Tokyo" }
    { "action": "unsubscribe", "city": "Tokyo" }
    { "action": "ping" }
    """
    client_id = str(uuid.uuid4())
    await manager.connect(ws, client_id, city)

    # Send welcome message
    await ws.send_json({
        "type": "connection",
        "client_id": client_id,
        "subscribed_city": city,
        "message": f"Connected to InfraGuard AI alert stream. City filter: {city or 'ALL'}",
        "timestamp": datetime.utcnow().isoformat(),
    })

    try:
        # Send buffered recent alerts on connect
        cached_alerts = await alert_cache.get(f"recent:{city or 'all'}")
        if cached_alerts:
            for alert in cached_alerts[:5]:
                await ws.send_json({"type": "alert", "alert": alert, "buffered": True,
                                    "timestamp": datetime.utcnow().isoformat()})

        # Ping loop + message handler
        while True:
            try:
                # Wait for client message (with timeout for server-side ping)
                msg = await asyncio.wait_for(ws.receive_json(), timeout=30.0)
                action = msg.get("action", "")

                if action == "ping":
                    await ws.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
                elif action == "subscribe" and msg.get("city"):
                    new_city = msg["city"]
                    if new_city not in manager.city_subscriptions:
                        manager.city_subscriptions[new_city] = set()
                    manager.city_subscriptions[new_city].add(client_id)
                    await ws.send_json({
                        "type": "subscribed", "city": new_city,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                elif action == "unsubscribe" and msg.get("city"):
                    old_city = msg["city"]
                    if old_city in manager.city_subscriptions:
                        manager.city_subscriptions[old_city].discard(client_id)
                    await ws.send_json({
                        "type": "unsubscribed", "city": old_city,
                        "timestamp": datetime.utcnow().isoformat()
                    })

            except asyncio.TimeoutError:
                # Server-side keepalive ping
                await ws.send_json({"type": "ping", "timestamp": datetime.utcnow().isoformat()})

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)


# ── Background alert polling task ────────────────────────────────────────────
async def poll_alerts_background():
    """
    Background task: poll all alert sources every 10 minutes.
    Push new alerts to subscribed WebSocket clients.
    Should be started during app lifespan.
    """
    CITIES = list(AlertEngine.CITY_COORDS.keys())
    INTERVAL_SECONDS = 600  # 10 minutes

    logger.info("🔔 Alert polling task started")
    while True:
        try:
            # Keep track of alerts sent in this process to avoid duplicate broadcasts
            # We use a set for speed, but ideally this would be a shared cache (Redis)
            # if multiple workers were running.
            sent_alerts_cache = await alert_cache.get("sent_ids") or []
            sent_ids = set(sent_alerts_cache)

            for city in CITIES:
                alerts = await alert_engine.generate_city_alerts(city)

                # Cache recent alerts (replaces the old set for REST GET /)
                await alert_cache.set(f"recent:{city}", alerts, ttl=3600)

                # Push only NEW alerts to subscribers
                new_alerts_count = 0
                for alert in alerts:
                    aid = alert.get("id")
                    if aid not in sent_ids:
                        msg = {"type": "alert", "alert": alert, "timestamp": datetime.utcnow().isoformat()}
                        await manager.broadcast_to_city(city, msg)
                        sent_ids.add(aid)
                        new_alerts_count += 1

                if new_alerts_count > 0:
                    logger.info(f"🔔 Pushed {new_alerts_count} NEW alerts for {city}")

            # Update the sent IDs cache (keep last 500 to prevent unbounded growth)
            await alert_cache.set("sent_ids", list(sent_ids)[-500:], ttl=86400)

            await asyncio.sleep(INTERVAL_SECONDS)

        except asyncio.CancelledError:
            logger.info("Alert polling task cancelled")
            break
        except Exception as e:
            logger.error(f"Alert polling error: {e}")
            await asyncio.sleep(60)  # retry after 1 minute on error


# ── REST endpoints ────────────────────────────────────────────────────────────
@router.get("/", summary="Get current alerts for all cities")
async def get_alerts(
    city: Optional[str] = Query(None, description="Filter by city"),
    severity: Optional[str] = Query(None, description="Filter: CRITICAL|HIGH|INFO"),
    limit: int = Query(20, ge=1, le=100),
):
    """Get current active alerts. Alerts are refreshed every 10 minutes."""
    cities = [city] if city else list(AlertEngine.CITY_COORDS.keys())
    all_alerts = []

    for c in cities:
        cached = await alert_cache.get(f"recent:{c}")
        if cached:
            all_alerts.extend(cached)

    # Filter by severity
    if severity:
        all_alerts = [a for a in all_alerts if a.get("severity") == severity.upper()]

    # Sort by severity then timestamp
    sev_order = {"CRITICAL": 0, "HIGH": 1, "INFO": 2}
    all_alerts.sort(key=lambda a: (sev_order.get(a.get("severity", "INFO"), 2), a.get("timestamp", "")))

    return {
        "alerts": all_alerts[:limit],
        "total": len(all_alerts),
        "city_filter": city,
        "severity_filter": severity,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/refresh", summary="Manually trigger alert refresh for a city")
async def refresh_alerts(city: str = Query(...)):
    """Force-refresh alerts for a city and push to WebSocket subscribers."""
    if city not in AlertEngine.CITY_COORDS:
        raise HTTPException(400, f"Unknown city: {city}")

    alerts = await alert_engine.generate_city_alerts(city)
    await alert_cache.set(f"recent:{city}", alerts, ttl=3600)

    # Push only new alerts to subscribers
    sent_alerts_cache = await alert_cache.get("sent_ids") or []
    sent_ids = set(sent_alerts_cache)
    pushed_count = 0

    for alert in alerts:
        aid = alert.get("id")
        if aid not in sent_ids:
            await manager.broadcast_to_city(city, {
                "type": "alert", "alert": alert,
                "timestamp": datetime.utcnow().isoformat()
            })
            sent_ids.add(aid)
            pushed_count += 1

    if pushed_count > 0:
        await alert_cache.set("sent_ids", list(sent_ids)[-500:], ttl=86400)

    return {
        "city": city,
        "alerts_generated": len(alerts),
        "new_alerts_pushed": pushed_count,
        "active_alerts": alerts,
        "pushed_to_subscribers": len(manager.city_subscriptions.get(city, set())),
    }


@router.get("/stats", summary="Alert statistics summary")
async def alert_stats():
    """Get alert counts by city and severity."""
    stats = {}
    for city in AlertEngine.CITY_COORDS.keys():
        cached = await alert_cache.get(f"recent:{city}") or []
        stats[city] = {
            "total":    len(cached),
            "critical": sum(1 for a in cached if a.get("severity") == "CRITICAL"),
            "high":     sum(1 for a in cached if a.get("severity") == "HIGH"),
            "info":     sum(1 for a in cached if a.get("severity") == "INFO"),
        }

    return {
        "stats": stats,
        "active_websocket_clients": len(manager.active),
        "timestamp": datetime.utcnow().isoformat(),
    }
