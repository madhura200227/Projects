# Pothole Detection - Quick Start Guide

## 🚀 Test Right Now (No Setup Required)

```bash
cd ig2
python test_pothole_detection.py
```

This creates a synthetic image and analyzes it. You'll see:
- Potholes detected
- Severity scores
- Repair instructions
- Cost estimates

## 📸 Analyze Your Own Image

```python
from ml.pothole_detector import analyze_pothole_image

result = analyze_pothole_image("path/to/your/road_image.jpg")

print(f"Potholes: {result['potholes_detected']}")
print(f"Condition: {result['overall_condition']}")
print(f"Risk: {result['failure_risk_score']}/100")
```

## 🌐 Use the API

```bash
# Start server
uvicorn main:app --reload

# Analyze image
curl -X POST "http://localhost:8000/api/pothole/quick-analyze" \
  -F "file=@road_image.jpg"
```

## 🎓 Train on Your Data

### Step 1: Get Training Data

**Option A: Download Public Dataset**
```bash
# RDD2022 (47,000+ images)
git clone https://github.com/sekilab/RoadDamageDetector
```

**Option B: Use Your Own Images**
- Collect 500+ road images
- Annotate with LabelImg: `pip install labelImg && labelImg`

### Step 2: Prepare Dataset

```bash
# Create structure
python ig2/ml/prepare_dataset.py --action create

# Add your images to:
#   ig2/data/pothole_yolo/train/images/
#   ig2/data/pothole_yolo/train/labels/
#   ig2/data/severity/train/{minimal,low,medium,high,critical}/

# Validate
python ig2/ml/prepare_dataset.py --action validate-yolo
```

### Step 3: Train

```bash
# Install dependencies
pip install ultralytics torch torchvision

# Train both models
python ig2/ml/train_pothole_model.py --mode both --epochs 50
```

Training takes 2-4 hours on GPU, 8-12 hours on CPU.

## 📊 What You Get

```json
{
  "potholes_detected": 3,
  "overall_condition": "POOR",
  "failure_risk_score": 74.5,
  "traffic_impact": {
    "jam_probability": 0.68,
    "jam_risk": "HIGH",
    "speed_reduction_pct": 41
  },
  "repair_instructions": [
    {
      "step": 1,
      "title": "Safety Setup & Traffic Control",
      "duration": "15-30 minutes",
      "equipment": ["Traffic cones", "Warning signs"]
    }
  ],
  "estimated_repair_cost": {
    "total_usd": 485.50
  }
}
```

## 🔗 Public Datasets

### Kaggle Datasets (Download Directly - No Zip File!)

I've created a script to download Kaggle datasets directly:

```bash
# Install Kaggle API
pip install kaggle

# Set your credentials (get from https://www.kaggle.com/settings/account)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key

# Download dataset directly (no zip file saved!)
python ig2/ml/download_kaggle_dataset.py --dataset potholes
```

**Available Datasets:**
1. **Potholes** (665 images) - `--dataset potholes`
2. **Road Damage (RDD2022)** (47K images) - `--dataset road-damage`
3. **Crack Detection** - `--dataset crack-detection`

### Manual Download (if API not available)

1. Go to dataset URL
2. Click "Download" button
3. Save zip to `ig2/data/kaggle/`
4. Extract the zip file

### Public Datasets (Free)

1. **RDD2022**: https://github.com/sekilab/RoadDamageDetector (47K images)
2. **Kaggle**: https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset (665 images)
3. **GAPs**: https://www.tu-ilmenau.de/neurob/data-sets-code/german-asphalt-pavement-distress-dataset-gaps (1,969 images)

## 📚 Full Documentation

- Complete guide: `ig2/ml/POTHOLE_DETECTION_GUIDE.md`
- System summary: `ig2/POTHOLE_SYSTEM_SUMMARY.md`
- API docs: http://localhost:8000/docs (after starting server)

## ⚡ Key Features

✅ Works without training (heuristic fallback)
✅ Detects potholes and scores severity (0-100)
✅ Predicts traffic jam probability
✅ Generates step-by-step repair instructions
✅ Estimates repair costs
✅ Fine-tunable on custom datasets
✅ FastAPI endpoints ready
✅ Integrates with main InfraGuard risk model

## 🎯 Next Steps

1. Run test: `python ig2/test_pothole_detection.py`
2. Try with your images
3. Download a public dataset
4. Fine-tune the models
5. Integrate with your frontend

**Need help?** Check `ig2/ml/POTHOLE_DETECTION_GUIDE.md` for detailed instructions.
