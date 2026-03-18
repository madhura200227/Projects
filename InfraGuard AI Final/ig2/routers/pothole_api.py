"""
routers/pothole_api.py – Pothole Detection API Endpoints
========================================================
FastAPI endpoints for pothole detection and analysis.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import shutil
import os
from pathlib import Path
import tempfile

from ml.pothole_detector import PotholeDetector, analyze_pothole_image
from utils.config import settings
from utils.data_collection import persist_image_sample

logger = logging.getLogger("infraguard.pothole_api")

router = APIRouter(prefix="/api/pothole", tags=["Pothole Detection"])

# Initialize detector (singleton)
_detector = None

def get_detector() -> PotholeDetector:
    """Get or create pothole detector instance."""
    global _detector
    if _detector is None:
        _detector = PotholeDetector()
        _detector.load_models()
        logger.info("Pothole detector initialized")
    return _detector


@router.post("/analyze")
async def analyze_pothole(
    file: UploadFile = File(..., description="Road image file (JPG/PNG)"),
    avg_daily_traffic: Optional[int] = Form(None),
    road_age_years: Optional[int] = Form(None),
    flood_events_5yr: Optional[int] = Form(None),
    aqi: Optional[int] = Form(None),
    heavy_vehicle_pct: Optional[float] = Form(None),
    collect: bool = Query(False, description="If true, persist upload + metadata for training (requires DATA_COLLECTION_DIR)"),
):
    """
    Analyze a road image for potholes.
    
    Returns:
        - Detected potholes with severity scores
        - Traffic impact prediction
        - Repair instructions
        - Cost estimates
        - Failure risk assessment
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image (JPG/PNG)")
    
    # Save uploaded file temporarily
    temp_dir = Path(tempfile.gettempdir()) / "infraguard_uploads"
    temp_dir.mkdir(exist_ok=True)
    
    temp_file = temp_dir / f"upload_{file.filename}"
    
    try:
        # Save file
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        collected = None
        if collect and settings.DATA_COLLECTION_DIR:
            try:
                collected = persist_image_sample(
                    base_dir=settings.DATA_COLLECTION_DIR,
                    image_bytes=temp_file.read_bytes(),
                    filename_hint=file.filename or "upload",
                    metadata={"endpoint": "/api/pothole/analyze"},
                    subdir="pothole_analyze",
                    image_ext=temp_file.suffix if temp_file.suffix else ".jpg",
                )
            except Exception as e:
                logger.warning(f"Data collection failed (pothole/analyze): {e}")
        
        logger.info(f"Analyzing uploaded image: {file.filename}")
        
        # Prepare location data if provided
        location_data = None
        if any([avg_daily_traffic, road_age_years, flood_events_5yr, aqi, heavy_vehicle_pct]):
            location_data = {}
            if avg_daily_traffic:
                location_data['avg_daily_traffic'] = avg_daily_traffic
            if road_age_years:
                location_data['road_age_years'] = road_age_years
            if flood_events_5yr:
                location_data['flood_events_5yr'] = flood_events_5yr
            if aqi:
                location_data['aqi'] = aqi
            if heavy_vehicle_pct:
                location_data['heavy_vehicle_pct'] = heavy_vehicle_pct
        
        # Analyze image
        detector = get_detector()
        result = detector.analyze_image(str(temp_file), location_data)
        
        logger.info(f"Analysis complete: {result['potholes_detected']} potholes detected")
        
        if collected:
            result["collected"] = collected
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Cleanup
        if temp_file.exists():
            os.remove(temp_file)


@router.post("/quick-analyze")
async def quick_analyze(file: UploadFile = File(...)):
    """
    Quick pothole analysis without location data.
    Faster endpoint for basic detection.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_dir = Path(tempfile.gettempdir()) / "infraguard_uploads"
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / f"quick_{file.filename}"
    
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = analyze_pothole_image(str(temp_file))
        
        # Return simplified response
        return {
            "potholes_detected": result['potholes_detected'],
            "overall_condition": result['overall_condition'],
            "failure_risk_score": result['failure_risk_score'],
            "max_severity": result.get('max_severity', 0),
            "traffic_impact": result['traffic_impact'],
            "recommended_action": result['recommended_action'],
        }
    
    except Exception as e:
        logger.error(f"Quick analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_file.exists():
            os.remove(temp_file)


@router.get("/health")
async def health_check():
    """Check if pothole detection service is ready."""
    try:
        detector = get_detector()
        return {
            "status": "healthy",
            "yolo_loaded": detector.yolo_model is not None,
            "severity_loaded": detector.severity_model is not None,
            "device": str(detector.device),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@router.get("/info")
async def get_info():
    """Get information about the pothole detection system."""
    return {
        "name": "InfraGuard Pothole Detection System",
        "version": "1.0.0",
        "capabilities": [
            "Pothole detection using YOLOv8",
            "Severity classification (0-100 score)",
            "Traffic jam prediction",
            "Repair cost estimation",
            "Step-by-step repair instructions",
            "Infrastructure failure risk assessment",
        ],
        "supported_formats": ["JPG", "PNG"],
        "max_file_size_mb": 10,
        "severity_levels": ["MINIMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"],
    }
