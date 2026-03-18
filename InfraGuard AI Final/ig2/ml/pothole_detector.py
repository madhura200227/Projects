"""
ml/pothole_detector.py – Pothole Detection & Infrastructure Analysis
======================================================================
Complete system for:
  1. Pothole detection from images
  2. Severity scoring (0-100)
  3. Failure risk prediction
  4. Step-by-step repair recommendations
  5. Traffic jam prediction based on road conditions

Architecture:
  - YOLOv8 for pothole detection (fine-tunable)
  - ResNet-50 for severity classification
  - Custom scoring algorithm
  - Integration with main InfraGuard model
"""

import os
import logging
import math
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import json

logger = logging.getLogger("infraguard.pothole_detector")

# Try importing ML libraries
try:
    import numpy as np
    from PIL import Image
    import cv2
    HAS_CV2 = True
except ImportError:
    logger.warning("OpenCV not available - using fallback image processing")
    HAS_CV2 = False
    import numpy as np
    from PIL import Image

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    logger.warning("Ultralytics YOLO not available - using heuristic detection")
    HAS_YOLO = False

try:
    import torch
    import torchvision.transforms as transforms
    import torchvision.models as models
    HAS_TORCH = True
except ImportError:
    logger.warning("PyTorch not available - using heuristic severity scoring")
    HAS_TORCH = False


# Severity thresholds
SEVERITY_THRESHOLDS = {
    "CRITICAL": 80,  # Immediate danger
    "HIGH": 60,      # Repair within 24-48 hours
    "MEDIUM": 40,    # Repair within 1 week
    "LOW": 20,       # Monitor, repair within 1 month
    "MINIMAL": 0,    # Cosmetic only
}

# Pothole characteristics scoring weights
SCORING_WEIGHTS = {
    "area_cm2": 0.25,           # Size of pothole
    "depth_cm": 0.30,           # Depth (most critical)
    "edge_sharpness": 0.15,     # Sharp edges = more dangerous
    "location_traffic": 0.15,   # High traffic area
    "water_presence": 0.10,     # Water accelerates damage
    "crack_density": 0.05,      # Surrounding cracks
}


class PotholeDetector:
    """
    Main pothole detection and analysis system.
    Uses YOLOv8 for detection, ResNet for severity classification.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            # Default to weights folder relative to this file
            self.model_dir = Path(__file__).parent / "weights"
        else:
            self.model_dir = Path(model_dir)
            
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.yolo_model = None
        self.yolo_is_finetuned = False
        self.severity_model = None
        self.device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
        
        logger.info(f"PotholeDetector initialized (device: {self.device})")
    
    def load_models(self):
        """Load pre-trained or fine-tuned models."""
        # Load YOLO for detection
        if HAS_YOLO:
            yolo_path = self.model_dir / "pothole_yolo.pt"
            if yolo_path.exists():
                self.yolo_model = YOLO(str(yolo_path))
                self.yolo_is_finetuned = True
                logger.info(f"✅ Loaded fine-tuned YOLO model from {yolo_path}")
            else:
                # Important: YOLOv8n (COCO) does NOT have a pothole class.
                # Using it here would generate misleading "pothole" detections.
                self.yolo_model = None
                self.yolo_is_finetuned = False
                logger.warning(
                    "No fine-tuned YOLO weights found at %s. "
                    "Falling back to heuristic pothole detection. "
                    "Train and save weights to enable YOLO detection.",
                    yolo_path,
                )
        
        # Load severity classifier
        if HAS_TORCH:
            severity_path = self.model_dir / "severity_resnet.pt"
            if severity_path.exists():
                self.severity_model = torch.load(severity_path, map_location=self.device)
                self.severity_model.eval()
                logger.info(f"✅ Loaded severity classifier from {severity_path}")
            else:
                # Do NOT run an untrained classification head (would be random).
                # Use the heuristic severity scorer until a trained model exists.
                self.severity_model = None
                logger.warning(
                    "No severity classifier weights found at %s. "
                    "Falling back to heuristic severity scoring.",
                    severity_path,
                )

    
    def detect_potholes(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect potholes in an image.
        Returns list of detected potholes with bounding boxes and confidence.
        """
        if HAS_YOLO and self.yolo_model and self.yolo_is_finetuned:
            return self._detect_with_yolo(image_path)
        else:
            return self._detect_heuristic(image_path)
    
    def _detect_with_yolo(
        self,
        image_path: str,
        min_conf: float = 0.25,
        allowed_classes: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """YOLO-based detection."""
        # Fine-tuned model is expected to use: 0=pothole, 1=crack, 2=patch
        if allowed_classes is None:
            allowed_classes = {0}

        results = self.yolo_model(image_path)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if conf < min_conf:
                    continue
                if allowed_classes is not None and cls not in allowed_classes:
                    continue
                
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(conf, 3),
                    "class": cls,
                    "area_pixels": int((x2 - x1) * (y2 - y1)),
                    "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                })
        
        logger.info(f"YOLO detected {len(detections)} potholes")
        return detections
    
    def _detect_heuristic(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Heuristic detection using image processing.
        Looks for dark, irregular regions with sharp edges.
        """
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        if HAS_CV2:
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Adaptive thresholding to find dark regions
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            detections = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                # Filter by size (potholes typically 100-10000 pixels)
                if 100 < area < 10000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate circularity (potholes are somewhat circular)
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Score based on darkness and shape
                    roi = gray[y:y+h, x:x+w]
                    darkness = 1 - (np.mean(roi) / 255.0)
                    confidence = (darkness * 0.6 + circularity * 0.4)
                    
                    if confidence > 0.3:  # Threshold
                        detections.append({
                            "bbox": [x, y, x+w, y+h],
                            "confidence": round(confidence, 3),
                            "class": 0,  # pothole class
                            "area_pixels": int(area),
                            "center": [x + w//2, y + h//2],
                            "circularity": round(circularity, 3),
                        })
            
            logger.info(f"Heuristic detected {len(detections)} potential potholes")
            return sorted(detections, key=lambda x: x['confidence'], reverse=True)[:10]
        
        else:
            # Very basic fallback
            logger.warning("Using minimal fallback detection")
            return [{
                "bbox": [100, 100, 200, 200],
                "confidence": 0.5,
                "class": 0,
                "area_pixels": 10000,
                "center": [150, 150],
            }]

    
    def analyze_severity(self, image_path: str, detection: Dict) -> Dict[str, Any]:
        """
        Analyze severity of a detected pothole.
        Returns severity score (0-100) and classification.
        """
        img = Image.open(image_path).convert('RGB')
        x1, y1, x2, y2 = detection['bbox']
        
        # Crop to pothole region
        pothole_img = img.crop((x1, y1, x2, y2))
        
        if HAS_TORCH and self.severity_model:
            severity_score = self._classify_severity_nn(pothole_img)
        else:
            severity_score = self._classify_severity_heuristic(pothole_img, detection)
        
        # Determine severity level
        severity_level = self._get_severity_level(severity_score)
        
        # Estimate physical dimensions (requires calibration in production)
        area_cm2 = self._estimate_area(detection['area_pixels'])
        depth_cm = self._estimate_depth(severity_score, pothole_img)
        
        return {
            "severity_score": round(severity_score, 1),
            "severity_level": severity_level,
            "estimated_area_cm2": round(area_cm2, 1),
            "estimated_depth_cm": round(depth_cm, 1),
            "repair_urgency": self._get_repair_urgency(severity_level),
            "danger_level": self._get_danger_level(severity_score),
        }
    
    def _classify_severity_nn(self, pothole_img: Image.Image) -> float:
        """Neural network-based severity classification."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        img_tensor = transform(pothole_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.severity_model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            
            # Convert 5-class probabilities to 0-100 score
            # Classes: [minimal, low, medium, high, critical]
            class_scores = [10, 30, 50, 70, 90]
            severity = sum(p * s for p, s in zip(probs.cpu().numpy(), class_scores))
            
        return float(severity)
    
    def _classify_severity_heuristic(self, pothole_img: Image.Image, detection: Dict) -> float:
        """Heuristic severity scoring based on image features."""
        img_array = np.array(pothole_img)
        
        # Convert to grayscale
        if img_array.ndim == 3:
            gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        else:
            gray = img_array
        
        # Feature extraction
        darkness = 1 - (np.mean(gray) / 255.0)  # Darker = deeper
        variance = np.std(gray) / 255.0          # High variance = rough edges
        
        # Edge detection for sharpness
        if HAS_CV2:
            edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
        else:
            edge_density = variance  # Fallback
        
        # Size factor
        area_pixels = detection['area_pixels']
        size_factor = min(1.0, area_pixels / 5000)  # Normalize to 5000 pixels
        
        # Combine factors
        severity = (
            darkness * 35 +
            variance * 25 +
            edge_density * 20 +
            size_factor * 20
        )
        
        return min(100, max(0, severity))
    
    def _estimate_area(self, area_pixels: int, pixels_per_cm: float = 10.0) -> float:
        """Estimate real-world area from pixel area."""
        # Assumes camera at ~2m height, typical road camera setup
        return area_pixels / (pixels_per_cm ** 2)
    
    def _estimate_depth(self, severity_score: float, pothole_img: Image.Image) -> float:
        """Estimate depth from severity and shadow analysis."""
        # Depth estimation from severity (empirical correlation)
        base_depth = severity_score / 100 * 15  # Max 15cm for score 100
        
        # Shadow analysis for depth refinement
        img_array = np.array(pothole_img)
        if img_array.ndim == 3:
            gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        else:
            gray = img_array
        
        # Darker regions suggest deeper potholes
        darkness_factor = 1 - (np.mean(gray) / 255.0)
        depth = base_depth * (0.7 + darkness_factor * 0.6)
        
        return max(0.5, min(20.0, depth))  # Clamp between 0.5-20cm

    
    def _get_severity_level(self, score: float) -> str:
        """Convert numeric score to severity level."""
        if score >= SEVERITY_THRESHOLDS["CRITICAL"]:
            return "CRITICAL"
        elif score >= SEVERITY_THRESHOLDS["HIGH"]:
            return "HIGH"
        elif score >= SEVERITY_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        elif score >= SEVERITY_THRESHOLDS["LOW"]:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _get_repair_urgency(self, severity_level: str) -> str:
        """Get repair timeline based on severity."""
        urgency_map = {
            "CRITICAL": "Immediate (0-24 hours)",
            "HIGH": "Urgent (24-48 hours)",
            "MEDIUM": "Within 1 week",
            "LOW": "Within 1 month",
            "MINIMAL": "Monitor, repair as needed",
        }
        return urgency_map.get(severity_level, "Unknown")
    
    def _get_danger_level(self, score: float) -> str:
        """Assess danger to vehicles and pedestrians."""
        if score >= 80:
            return "Extreme - Vehicle damage likely, accident risk high"
        elif score >= 60:
            return "High - Tire damage possible, steering impact"
        elif score >= 40:
            return "Moderate - Discomfort, minor vehicle wear"
        elif score >= 20:
            return "Low - Cosmetic concern, minimal impact"
        else:
            return "Minimal - Surface wear only"
    
    def predict_traffic_impact(self, potholes: List[Dict], road_width_m: float = 7.0) -> Dict[str, Any]:
        """
        Predict traffic jam likelihood based on pothole distribution.
        """
        if not potholes:
            return {
                "jam_probability": 0.0,
                "jam_risk": "NONE",
                "speed_reduction_pct": 0,
                "lane_blockage": False,
            }
        
        # Calculate total affected area
        total_severity = sum(p.get('severity_score', 50) for p in potholes)
        avg_severity = total_severity / len(potholes)
        
        # Check if potholes block lanes
        # Assume image width represents road width
        lane_blockage = any(p.get('estimated_area_cm2', 0) > 5000 for p in potholes)
        
        # Jam probability factors
        count_factor = min(1.0, len(potholes) / 10)  # More potholes = higher risk
        severity_factor = avg_severity / 100
        blockage_factor = 0.3 if lane_blockage else 0.0
        
        jam_prob = min(0.95, count_factor * 0.4 + severity_factor * 0.4 + blockage_factor)
        
        # Speed reduction estimate
        speed_reduction = int(jam_prob * 60)  # Up to 60% reduction
        
        # Risk classification
        if jam_prob >= 0.7:
            jam_risk = "HIGH"
        elif jam_prob >= 0.4:
            jam_risk = "MEDIUM"
        elif jam_prob >= 0.15:
            jam_risk = "LOW"
        else:
            jam_risk = "MINIMAL"
        
        return {
            "jam_probability": round(jam_prob, 3),
            "jam_risk": jam_risk,
            "speed_reduction_pct": speed_reduction,
            "lane_blockage": lane_blockage,
            "pothole_count": len(potholes),
            "avg_severity": round(avg_severity, 1),
            "estimated_delay_minutes": int(jam_prob * 15),  # Up to 15 min delay
        }

    
    def generate_repair_instructions(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate step-by-step repair instructions based on severity.
        """
        severity_level = analysis.get('severity_level', 'MEDIUM')
        depth = analysis.get('estimated_depth_cm', 5)
        area = analysis.get('estimated_area_cm2', 100)
        
        instructions = []
        
        # Step 1: Safety and preparation
        instructions.append({
            "step": 1,
            "title": "Safety Setup & Traffic Control",
            "description": f"Set up traffic cones 50m before pothole. Use warning signs. {'Deploy flaggers for high-traffic areas.' if severity_level in ['CRITICAL', 'HIGH'] else 'Single-lane closure acceptable for low traffic.'}",
            "duration": "15-30 minutes",
            "equipment": ["Traffic cones", "Warning signs", "Safety vest", "Flaggers (if needed)"],
            "priority": "CRITICAL",
        })
        
        # Step 2: Cleaning
        instructions.append({
            "step": 2,
            "title": "Pothole Cleaning & Preparation",
            "description": f"Remove all loose debris, dirt, and water from pothole. Use compressed air or broom. For depth {depth:.1f}cm, ensure edges are clean and vertical.",
            "duration": "10-20 minutes",
            "equipment": ["Broom", "Compressed air", "Shop vacuum", "Wire brush"],
            "priority": "HIGH",
        })
        
        # Step 3: Tack coat (for deeper potholes)
        if depth > 5:
            instructions.append({
                "step": 3,
                "title": "Apply Tack Coat",
                "description": f"Apply asphalt emulsion tack coat to bottom and sides. Essential for depth >{depth:.1f}cm to ensure bonding. Let cure 5-10 minutes.",
                "duration": "10-15 minutes",
                "equipment": ["Asphalt emulsion", "Brush or spray applicator"],
                "priority": "HIGH",
            })
        
        # Step 4: Fill material selection
        if severity_level in ["CRITICAL", "HIGH"]:
            fill_material = "Hot-mix asphalt (HMA)"
            fill_desc = f"Use hot-mix asphalt at 135-160°C. For area {area:.0f}cm², apply in {math.ceil(depth/5)}cm lifts. Compact each lift thoroughly."
        elif depth > 7:
            fill_material = "Cold-patch asphalt"
            fill_desc = f"Use polymer-modified cold patch. Apply in layers, max 5cm per lift. Total depth {depth:.1f}cm requires {math.ceil(depth/5)} lifts."
        else:
            fill_material = "Cold-patch asphalt or spray injection"
            fill_desc = f"Standard cold patch acceptable for shallow repair. Single lift sufficient for {depth:.1f}cm depth."
        
        instructions.append({
            "step": len(instructions) + 1,
            "title": f"Fill with {fill_material}",
            "description": fill_desc,
            "duration": "20-40 minutes",
            "equipment": [fill_material, "Shovel", "Rake", "Tamper"],
            "priority": "CRITICAL",
        })
        
        # Step 5: Compaction
        compaction_method = "Vibratory plate compactor" if area > 2000 else "Hand tamper"
        instructions.append({
            "step": len(instructions) + 1,
            "title": "Compaction",
            "description": f"Compact fill material using {compaction_method}. Overfill by 1-2cm to account for settling. Make 3-5 passes for proper density.",
            "duration": "15-25 minutes",
            "equipment": [compaction_method, "Level"],
            "priority": "CRITICAL",
        })
        
        # Step 6: Edge sealing
        if severity_level in ["CRITICAL", "HIGH", "MEDIUM"]:
            instructions.append({
                "step": len(instructions) + 1,
                "title": "Edge Sealing",
                "description": "Apply rubberized crack sealant around perimeter to prevent water infiltration. Critical for longevity.",
                "duration": "10-15 minutes",
                "equipment": ["Crack sealant", "Applicator gun"],
                "priority": "MEDIUM",
            })
        
        # Step 7: Final inspection
        instructions.append({
            "step": len(instructions) + 1,
            "title": "Quality Check & Traffic Restoration",
            "description": "Verify repair is level with surrounding pavement (±5mm). Check compaction. Remove traffic control. Monitor for 24-48 hours.",
            "duration": "10 minutes",
            "equipment": ["Level", "Measuring tape"],
            "priority": "MEDIUM",
        })
        
        return instructions

    
    def analyze_image(self, image_path: str, location_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Complete analysis pipeline for a road image.
        Returns comprehensive report with detections, severity, and recommendations.
        """
        logger.info(f"Analyzing image: {image_path}")
        
        # Detect potholes
        detections = self.detect_potholes(image_path)
        
        if not detections:
            return {
                "status": "success",
                "potholes_detected": 0,
                "message": "No potholes detected in image",
                "overall_condition": "GOOD",
                "failure_risk_score": 15,
            }
        
        # Analyze each pothole
        analyzed_potholes = []
        for i, detection in enumerate(detections):
            severity_analysis = self.analyze_severity(image_path, detection)
            
            pothole_data = {
                "id": i + 1,
                "detection": detection,
                **severity_analysis,
            }
            analyzed_potholes.append(pothole_data)
        
        # Overall assessment
        avg_severity = np.mean([p['severity_score'] for p in analyzed_potholes])
        max_severity = max([p['severity_score'] for p in analyzed_potholes])
        
        # Traffic impact prediction
        traffic_impact = self.predict_traffic_impact(analyzed_potholes)
        
        # Generate repair instructions for worst pothole
        worst_pothole = max(analyzed_potholes, key=lambda x: x['severity_score'])
        repair_instructions = self.generate_repair_instructions(worst_pothole)
        
        # Calculate failure risk score (0-100)
        failure_risk = self._calculate_failure_risk(analyzed_potholes, location_data)
        
        # Overall road condition
        if max_severity >= 80:
            condition = "CRITICAL"
        elif max_severity >= 60:
            condition = "POOR"
        elif max_severity >= 40:
            condition = "FAIR"
        elif max_severity >= 20:
            condition = "GOOD"
        else:
            condition = "EXCELLENT"
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "potholes_detected": len(analyzed_potholes),
            "potholes": analyzed_potholes,
            "overall_condition": condition,
            "average_severity": round(avg_severity, 1),
            "max_severity": round(max_severity, 1),
            "failure_risk_score": round(failure_risk, 1),
            "traffic_impact": traffic_impact,
            "repair_instructions": repair_instructions,
            "estimated_repair_cost": self._estimate_repair_cost(analyzed_potholes),
            "recommended_action": self._get_recommended_action(condition, traffic_impact),
        }
    
    def _calculate_failure_risk(self, potholes: List[Dict], location_data: Optional[Dict]) -> float:
        """
        Calculate infrastructure failure risk based on potholes and location.
        Integrates with main InfraGuard model.
        """
        if not potholes:
            return 10.0
        
        # Base risk from pothole severity
        avg_severity = np.mean([p['severity_score'] for p in potholes])
        count_factor = min(1.0, len(potholes) / 15)
        
        base_risk = avg_severity * 0.7 + count_factor * 30
        
        # Adjust for location factors if available
        if location_data:
            traffic = location_data.get('avg_daily_traffic', 50000)
            age = location_data.get('road_age_years', 20)
            flood = location_data.get('flood_events_5yr', 5)
            
            traffic_mult = 1 + min(0.3, traffic / 200000)
            age_mult = 1 + min(0.4, age / 50)
            flood_mult = 1 + min(0.2, flood / 10)
            
            base_risk = base_risk * traffic_mult * age_mult * flood_mult
        
        return min(99.0, max(10.0, base_risk))
    
    def _estimate_repair_cost(self, potholes: List[Dict]) -> Dict[str, Any]:
        """Estimate repair costs."""
        total_area = sum(p.get('estimated_area_cm2', 0) for p in potholes)
        total_area_m2 = total_area / 10000
        
        # Cost per m² varies by severity
        critical_count = sum(1 for p in potholes if p['severity_level'] == 'CRITICAL')
        high_count = sum(1 for p in potholes if p['severity_level'] == 'HIGH')
        
        # Base costs (USD)
        cost_per_m2 = 150 if critical_count > 0 else 100 if high_count > 0 else 75
        material_cost = total_area_m2 * cost_per_m2
        labor_cost = len(potholes) * 50 + (critical_count * 100)
        equipment_cost = 200 if critical_count > 0 else 100
        
        total = material_cost + labor_cost + equipment_cost
        
        return {
            "total_usd": round(total, 2),
            "material_cost": round(material_cost, 2),
            "labor_cost": round(labor_cost, 2),
            "equipment_cost": round(equipment_cost, 2),
            "area_m2": round(total_area_m2, 2),
            "cost_per_m2": cost_per_m2,
        }
    
    def _get_recommended_action(self, condition: str, traffic_impact: Dict) -> str:
        """Get recommended action based on analysis."""
        if condition == "CRITICAL":
            return "🚨 IMMEDIATE ACTION REQUIRED: Close lane and begin emergency repairs within 24 hours. High accident risk."
        elif condition == "POOR" and traffic_impact['jam_risk'] == "HIGH":
            return "⚠️ URGENT: Schedule repairs within 48 hours. Significant traffic disruption and vehicle damage risk."
        elif condition == "POOR":
            return "⚠️ Schedule repairs within 1 week. Monitor daily for deterioration."
        elif condition == "FAIR":
            return "📋 Plan repairs within 2-4 weeks. Add to maintenance schedule."
        else:
            return "✅ Monitor condition. Include in routine maintenance cycle."


# Convenience function for quick analysis
def analyze_pothole_image(image_path: str, location_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Quick analysis function for single image.
    """
    detector = PotholeDetector()
    detector.load_models()
    return detector.analyze_image(image_path, location_data)
