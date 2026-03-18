## Goal
Make both image endpoints accurate by training real models and evaluating them:
- **`/api/pothole/analyze`**: train `pothole_yolo.pt` + (optional) `severity_resnet.pt`
- **`/api/v1/predict/image`**: train
  - `crack_cnn.pt` (multi-label damage scores: crack/pothole/wear/water/deformation)
  - `infra_type_resnet.pt` (infra type: road/bridge/pipeline/building/manhole)

## 1) Build a dataset (recommended workflow)

### A) Collect uploads automatically (optional)
Set in `ig2/.env`:

```env
DATA_COLLECTION_DIR=data/collected
```

Then call:
- `POST /api/v1/predict/image?collect=true`
- `POST /api/pothole/analyze?collect=true`

This will save images + metadata under `data/collected/...`.

### B) Label the collected data
- **Potholes (boxes)**: use LabelImg / CVAT → export YOLO format
- **Infra type**: sort images into folders by class
- **Damage scores**: create a CSV with per-image scores

## 2) Train pothole models

### YOLO detector
Prepare:
- `ig2/data/pothole_yolo/train/images`, `ig2/data/pothole_yolo/train/labels`
- `ig2/data/pothole_yolo/val/images`, `ig2/data/pothole_yolo/val/labels`

Train:

```bash
python ig2/ml/train_pothole_model.py --mode yolo --epochs 50
```

Outputs:
- `ig2/ml/weights/pothole_yolo.pt`

### Severity classifier (optional)
Prepare folder dataset:
- `ig2/data/severity/train/{minimal,low,medium,high,critical}`
- `ig2/data/severity/val/{minimal,low,medium,high,critical}`

Train:

```bash
python ig2/ml/train_pothole_model.py --mode severity --epochs 30
```

Outputs:
- `ig2/ml/weights/severity_resnet.pt`

## 3) Train `/predict/image` models

### A) Infra type classifier
Dataset:

```
ig2/data/infra_type/
  train/road ... val/road ...
```

Train:

```bash
python ig2/ml/train_infra_type_classifier.py --data ig2/data/infra_type
```

Outputs:
- `ig2/ml/weights/infra_type_resnet.pt`

### B) Damage multi-label CNN
Create a CSV like `ig2/data/damage_labels.csv`:

```csv
image_path,crack,pothole,wear,water,deformation
ig2/data/samples/img1.jpg,10,80,20,0,15
```

Train:

```bash
python ig2/ml/train_damage_cnn.py --csv ig2/data/damage_labels.csv --epochs 10
```

Outputs:
- `ig2/ml/weights/crack_cnn.pt`

## 4) Evaluate (before/after)
Edit `ig2/data/eval/manifest.jsonl` and run:

```bash
python ig2/tests/eval_image_accuracy.py --mode both
```

It writes `ig2/data/eval/report.json` with summary metrics.

