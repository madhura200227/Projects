from ultralytics import YOLO

# Load model once when server starts
model = YOLO("C:/Users/anish/runs/detect/train3/weights/best.pt")

def analyze_potholes(image_path):
    results = model(image_path)

    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            detections.append({
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "confidence": round(conf, 3)
            })

    return detections