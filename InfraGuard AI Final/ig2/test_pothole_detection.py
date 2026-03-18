"""
test_pothole_detection.py – Quick Test Script
==============================================
Test the pothole detection system with sample images.
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ml.pothole_detector import PotholeDetector, analyze_pothole_image


def test_basic_detection():
    """Test basic pothole detection."""
    print("="*60)
    print("TEST 1: Basic Pothole Detection")
    print("="*60)
    
    # Create detector
    detector = PotholeDetector()
    detector.load_models()
    
    print("\n✅ Models loaded successfully")
    print(f"   Device: {detector.device}")
    print(f"   YOLO model: {'Loaded' if detector.yolo_model else 'Using heuristic'}")
    print(f"   Severity model: {'Loaded' if detector.severity_model else 'Using heuristic'}")


def test_with_sample_image():
    """Test with a sample image (if available)."""
    print("\n" + "="*60)
    print("TEST 2: Sample Image Analysis")
    print("="*60)
    
    # Check for sample images
    sample_dir = Path("ig2/data/samples")
    if not sample_dir.exists():
        print("\n⚠️  No sample images found")
        print(f"   Create directory: {sample_dir}")
        print("   Add sample road images to test")
        return
    
    sample_images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
    
    if not sample_images:
        print("\n⚠️  No images in samples directory")
        return
    
    # Analyze first image
    image_path = str(sample_images[0])
    print(f"\nAnalyzing: {image_path}")
    
    # Sample location data
    location_data = {
        "avg_daily_traffic": 85000,
        "road_age_years": 28,
        "flood_events_5yr": 12,
        "aqi": 165,
        "heavy_vehicle_pct": 35,
    }
    
    result = analyze_pothole_image(image_path, location_data)
    
    print("\n📊 ANALYSIS RESULTS:")
    print(f"   Potholes detected: {result['potholes_detected']}")
    print(f"   Overall condition: {result['overall_condition']}")
    print(f"   Failure risk score: {result['failure_risk_score']}/100")
    
    if result['potholes_detected'] > 0:
        print(f"   Average severity: {result['average_severity']}/100")
        print(f"   Max severity: {result['max_severity']}/100")
        
        print("\n🚗 TRAFFIC IMPACT:")
        traffic = result['traffic_impact']
        print(f"   Jam probability: {traffic['jam_probability']*100:.1f}%")
        print(f"   Jam risk: {traffic['jam_risk']}")
        print(f"   Speed reduction: {traffic['speed_reduction_pct']}%")
        print(f"   Estimated delay: {traffic['estimated_delay_minutes']} minutes")
        
        print("\n💰 REPAIR COST ESTIMATE:")
        cost = result['estimated_repair_cost']
        print(f"   Total: ${cost['total_usd']:.2f}")
        print(f"   Material: ${cost['material_cost']:.2f}")
        print(f"   Labor: ${cost['labor_cost']:.2f}")
        print(f"   Equipment: ${cost['equipment_cost']:.2f}")
        
        print("\n🔧 REPAIR INSTRUCTIONS:")
        for instruction in result['repair_instructions'][:3]:  # Show first 3 steps
            print(f"\n   Step {instruction['step']}: {instruction['title']}")
            print(f"   Duration: {instruction['duration']}")
            print(f"   Priority: {instruction['priority']}")
        
        print(f"\n   ... {len(result['repair_instructions']) - 3} more steps")
        
        print(f"\n⚠️  RECOMMENDED ACTION:")
        print(f"   {result['recommended_action']}")
    
    # Save full result to JSON
    output_file = Path("ig2/data/test_result.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Full results saved to: {output_file}")


def test_synthetic_image():
    """Test with synthetic/generated test image."""
    print("\n" + "="*60)
    print("TEST 3: Synthetic Image Test")
    print("="*60)
    
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Create synthetic road image with dark spots (simulated potholes)
        img = Image.new('RGB', (640, 480), color=(100, 100, 100))  # Gray road
        draw = ImageDraw.Draw(img)
        
        # Draw some dark ellipses (simulated potholes)
        draw.ellipse([150, 200, 250, 280], fill=(30, 30, 30))  # Large pothole
        draw.ellipse([400, 150, 450, 190], fill=(40, 40, 40))  # Medium pothole
        draw.ellipse([300, 350, 330, 375], fill=(50, 50, 50))  # Small pothole
        
        # Save
        test_img_path = Path("ig2/data/test_synthetic.jpg")
        test_img_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(test_img_path)
        
        print(f"\n✅ Created synthetic test image: {test_img_path}")
        
        # Analyze
        result = analyze_pothole_image(str(test_img_path))
        
        print(f"\n📊 RESULTS:")
        print(f"   Potholes detected: {result['potholes_detected']}")
        print(f"   Overall condition: {result['overall_condition']}")
        
        if result['potholes_detected'] > 0:
            print(f"\n   Detected potholes:")
            for pothole in result['potholes']:
                print(f"   - Pothole #{pothole['id']}: Severity {pothole['severity_score']:.1f}/100 ({pothole['severity_level']})")
        
    except Exception as e:
        print(f"\n❌ Error creating synthetic image: {e}")


def print_usage_guide():
    """Print usage guide."""
    print("\n" + "="*60)
    print("USAGE GUIDE")
    print("="*60)
    
    print("\n1. ANALYZE AN IMAGE:")
    print("   from ml.pothole_detector import analyze_pothole_image")
    print("   result = analyze_pothole_image('path/to/image.jpg')")
    
    print("\n2. PREPARE DATASET:")
    print("   python ig2/ml/prepare_dataset.py --action create")
    
    print("\n3. TRAIN MODELS:")
    print("   # Install training dependencies first:")
    print("   pip install ultralytics torch torchvision")
    print("   ")
    print("   # Train YOLO detector:")
    print("   python ig2/ml/train_pothole_model.py --mode yolo --epochs 50")
    print("   ")
    print("   # Train severity classifier:")
    print("   python ig2/ml/train_pothole_model.py --mode severity --epochs 30")
    
    print("\n4. VALIDATE DATASET:")
    print("   python ig2/ml/prepare_dataset.py --action validate-yolo")
    print("   python ig2/ml/prepare_dataset.py --action validate-severity")
    
    print("\n5. FIND DATASETS:")
    print("   - RDD2022: https://github.com/sekilab/RoadDamageDetector")
    print("   - Kaggle Potholes: https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset")
    print("   - See ig2/ml/POTHOLE_DETECTION_GUIDE.md for more")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("\n🛣️  INFRAGUARD POTHOLE DETECTION SYSTEM TEST")
    print("="*60)
    
    try:
        # Run tests
        test_basic_detection()
        test_synthetic_image()
        test_with_sample_image()
        print_usage_guide()
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
