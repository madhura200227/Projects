# Pothole Detection System - Complete Guide

## Overview

This system provides complete pothole detection and infrastructure analysis:

1. **Detection**: Identify potholes in road images using YOLOv8
2. **Severity Analysis**: Classify pothole severity (0-100 score)
3. **Traffic Impact**: Predict traffic jam probability
4. **Repair Instructions**: Generate step-by-step repair guides
5. **Cost Estimation**: Calculate repair costs
6. **Fine-tuning**: Train on custom datasets

## Quick Start

### 1. Install Dependencies

```bash
# Basic dependencies (already in requirements.txt)
pip install -r requirements.txt

# For training/fine-tuning (optional)
pip install ultralytics torch torchvision
```

### 2. Analyze a Pothole Image

```python
from ml.pothole_detector import analyze_pothole_image

# Simple analysis
result = analyze_pothole_image("path/to/road_image.jpg")

print(f"Potholes detected: {result['potholes_detected']}")
print(f"Overall condition: {result['overall_condition']}")
print(f"Failure risk: {result['failure_risk_score']}/100")

# With location data for better risk assessment
location_data = {
    "avg_daily_traffic": 85000,
    "road_age_years": 28,
    "flood_events_5yr": 12,
}

result = analyze_pothole_image("image.jpg", location_data)
```

### 3. Get Repair Instructions

```python
# Repair instructions are included in the result
for instruction in result['repair_instructions']:
    print(f"Step {instruction['step']}: {instruction['title']}")
    print(f"  {instruction['description']}")
    print(f"  Duration: {instruction['duration']}")
    print(f"  Equipment: {', '.join(instruction['equipment'])}")
    print()
```

## Dataset Preparation for Fine-tuning

### For YOLO Detection Model

Create dataset in this structure:

```
ig2/data/pothole_yolo/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt
│       ├── img002.txt
│       └── ...
└── val/
    ├── images/
    └── labels/
```

**Label format** (YOLO format - one line per object):
```
<class_id> <x_center> <y_center> <width> <height>
```

Example `img001.txt`:
```
0 0.5 0.6 0.15 0.12
0 0.3 0.4 0.08 0.10
```

Where:
- `class_id`: 0=pothole, 1=crack, 2=patch
- All coordinates normalized to 0-1

### For Severity Classifier

Create dataset in this structure:

```
ig2/data/severity/
├── train/
│   ├── minimal/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── low/
│   ├── medium/
│   ├── high/
│   └── critical/
└── val/
    ├── minimal/
    ├── low/
    ├── medium/
    ├── high/
    └── critical/
```

**Severity Guidelines:**
- **Minimal**: Surface wear, <2cm depth, <500cm² area
- **Low**: Small potholes, 2-4cm depth, 500-1500cm² area
- **Medium**: Moderate potholes, 4-7cm depth, 1500-3000cm² area
- **High**: Large potholes, 7-12cm depth, 3000-6000cm² area
- **Critical**: Severe potholes, >12cm depth, >6000cm² area

## Training Models

### Train YOLO Detection Model

```bash
python ig2/ml/train_pothole_model.py \
    --mode yolo \
    --yolo-data ig2/data/pothole_dataset.yaml \
    --epochs 50 \
    --batch-size 16
```

### Train Severity Classifier

```bash
python ig2/ml/train_pothole_model.py \
    --mode severity \
    --train-dir ig2/data/severity/train \
    --val-dir ig2/data/severity/val \
    --epochs 30 \
    --batch-size 32
```

### Train Both Models

```bash
python ig2/ml/train_pothole_model.py \
    --mode both \
    --epochs 50
```

## Finding Pothole Datasets

### Public Datasets

1. **RDD2022** (Road Damage Detection)
   - 47,000+ images from multiple countries
   - https://github.com/sekilab/RoadDamageDetector

2. **Pothole Dataset** (Kaggle)
   - 665 annotated images
   - https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset

3. **CrackForest Dataset**
   - 118 images with pixel-level annotations
   - https://github.com/cuilimeng/CrackForest-dataset

4. **GAPs Dataset** (German Asphalt Pavement)
   - 1,969 images
   - https://www.tu-ilmenau.de/neurob/data-sets-code/german-asphalt-pavement-distress-dataset-gaps

### Creating Your Own Dataset

1. **Collect Images**:
   - Use dashcam footage
   - Municipal road inspection cameras
   - Smartphone photos (ensure consistent angle)
   - Aim for 500+ images minimum

2. **Annotation Tools**:
   - **LabelImg**: For YOLO format bounding boxes
     ```bash
     pip install labelImg
     labelImg
     ```
   - **CVAT**: Web-based annotation tool
   - **Roboflow**: Online platform with auto-labeling

3. **Data Augmentation** (built into training):
   - Horizontal flips
   - Rotation (±10°)
   - Brightness/contrast adjustment
   - Mosaic augmentation

## API Integration

### FastAPI Endpoint

```python
from fastapi import FastAPI, UploadFile, File
from ml.pothole_detector import PotholeDetector
import shutil

app = FastAPI()
detector = PotholeDetector()
detector.load_models()

@app.post("/api/analyze-pothole")
async def analyze_pothole(file: UploadFile = File(...)):
    # Save uploaded file
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Analyze
    result = detector.analyze_image(temp_path)
    
    # Cleanup
    os.remove(temp_path)
    
    return result
```

## Output Format

```json
{
  "status": "success",
  "timestamp": "2026-02-26T10:30:00",
  "potholes_detected": 3,
  "overall_condition": "POOR",
  "average_severity": 65.3,
  "max_severity": 82.1,
  "failure_risk_score": 74.5,
  "potholes": [
    {
      "id": 1,
      "detection": {
        "bbox": [120, 340, 280, 450],
        "confidence": 0.92,
        "area_pixels": 17600
      },
      "severity_score": 82.1,
      "severity_level": "CRITICAL",
      "estimated_area_cm2": 1760.0,
      "estimated_depth_cm": 12.3,
      "repair_urgency": "Immediate (0-24 hours)",
      "danger_level": "Extreme - Vehicle damage likely"
    }
  ],
  "traffic_impact": {
    "jam_probability": 0.68,
    "jam_risk": "HIGH",
    "speed_reduction_pct": 41,
    "lane_blockage": true,
    "estimated_delay_minutes": 10
  },
  "repair_instructions": [
    {
      "step": 1,
      "title": "Safety Setup & Traffic Control",
      "description": "Set up traffic cones 50m before pothole...",
      "duration": "15-30 minutes",
      "equipment": ["Traffic cones", "Warning signs"],
      "priority": "CRITICAL"
    }
  ],
  "estimated_repair_cost": {
    "total_usd": 485.50,
    "material_cost": 264.00,
    "labor_cost": 171.50,
    "equipment_cost": 50.00
  },
  "recommended_action": "🚨 IMMEDIATE ACTION REQUIRED: Close lane and begin emergency repairs within 24 hours."
}
```

## Performance Tips

1. **Image Quality**: Use 1920x1080 or higher resolution
2. **Lighting**: Avoid extreme shadows or overexposure
3. **Angle**: Overhead or 45° angle works best
4. **Distance**: Camera 2-3 meters from road surface
5. **Batch Processing**: Process multiple images in parallel

## Troubleshooting

### Models Not Loading
- Check if weight files exist in `ig2/ml/weights/`
- System will fall back to heuristic detection if models missing
- Download pre-trained weights or train your own

### Low Detection Accuracy
- Fine-tune on your specific road conditions
- Collect more training data
- Adjust confidence threshold in detection

### Memory Issues
- Reduce batch size during training
- Use smaller image size (416x416 instead of 640x640)
- Process images one at a time instead of batches

## Advanced Usage

### Custom Severity Scoring

```python
from ml.pothole_detector import PotholeDetector, SCORING_WEIGHTS

# Adjust weights for your use case
SCORING_WEIGHTS['depth_cm'] = 0.40  # Prioritize depth
SCORING_WEIGHTS['area_cm2'] = 0.20

detector = PotholeDetector()
result = detector.analyze_image("image.jpg")
```

### Integration with Main Model

```python
from ml.pothole_detector import PotholeDetector
from ml.model import InfraGuardModel

# Analyze image
pothole_detector = PotholeDetector()
pothole_detector.load_models()
pothole_result = pothole_detector.analyze_image("road.jpg")

# Use in infrastructure risk model
infra_model = InfraGuardModel()
infra_model.load()

features = {
    "avg_daily_traffic": 85000,
    "heavy_vehicle_pct": 35,
    "aqi": 165,
    "flood_events_5yr": 12,
    "road_age_years": 28,
    "rainfall_mm_annual": 2200,
    "soil_moisture": 68,
    "surface_material_score": 0.60,
    "drainage_quality_score": 0.30,
    "image_damage_score": pothole_result['max_severity'],  # Use pothole severity
    # ... other features
}

prediction = infra_model.predict(features, infra_type="roads")
```

## License & Attribution

When using public datasets, ensure proper attribution:
- RDD2022: Cite the RoadDamageDetector paper
- Kaggle datasets: Follow dataset-specific licenses
- Always check license terms before commercial use

## Support

For issues or questions:
1. Check logs in `ig2/logs/infraguard.log`
2. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
3. Review dataset format and structure
4. Ensure all dependencies are installed
