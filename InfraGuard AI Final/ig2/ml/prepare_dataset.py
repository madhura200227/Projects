"""
ml/prepare_dataset.py – Dataset Preparation Utilities
=====================================================
Helper functions to prepare and validate pothole datasets.
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import json
import random

def create_dataset_structure(base_dir: str = "ig2/data"):
    """Create directory structure for pothole datasets."""
    base = Path(base_dir)
    
    # YOLO detection dataset
    yolo_dirs = [
        "pothole_yolo/train/images",
        "pothole_yolo/train/labels",
        "pothole_yolo/val/images",
        "pothole_yolo/val/labels",
    ]
    
    # Severity classification dataset
    severity_classes = ["minimal", "low", "medium", "high", "critical"]
    severity_dirs = []
    for split in ["train", "val"]:
        for cls in severity_classes:
            severity_dirs.append(f"severity/{split}/{cls}")
    
    all_dirs = yolo_dirs + severity_dirs
    
    for dir_path in all_dirs:
        full_path = base / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {full_path}")
    
    print(f"\n✅ Dataset structure created in {base}")
    print("\nNext steps:")
    print("1. Add images to train/images and val/images")
    print("2. Add YOLO labels to train/labels and val/labels")
    print("3. Organize severity images into class folders")
    print("4. Run: python ig2/ml/train_pothole_model.py")


def split_dataset(
    source_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Split dataset into train and validation sets.
    
    Args:
        source_dir: Directory containing images
        train_ratio: Ratio of training data (0.0-1.0)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_files, val_files)
    """
    source = Path(source_dir)
    image_files = list(source.glob("*.jpg")) + list(source.glob("*.png"))
    
    random.seed(seed)
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Total images: {len(image_files)}")
    print(f"Training: {len(train_files)} ({train_ratio*100:.0f}%)")
    print(f"Validation: {len(val_files)} ({(1-train_ratio)*100:.0f}%)")
    
    return [str(f) for f in train_files], [str(f) for f in val_files]


def validate_yolo_dataset(dataset_dir: str) -> Dict[str, any]:
    """
    Validate YOLO dataset format and report statistics.
    
    Args:
        dataset_dir: Path to dataset root (contains train/ and val/)
    
    Returns:
        Validation report dictionary
    """
    base = Path(dataset_dir)
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }
    
    for split in ["train", "val"]:
        img_dir = base / split / "images"
        lbl_dir = base / split / "labels"
        
        if not img_dir.exists():
            report["errors"].append(f"Missing directory: {img_dir}")
            report["valid"] = False
            continue
        
        if not lbl_dir.exists():
            report["errors"].append(f"Missing directory: {lbl_dir}")
            report["valid"] = False
            continue
        
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        labels = list(lbl_dir.glob("*.txt"))
        
        # Check for matching labels
        missing_labels = []
        for img in images:
            label_path = lbl_dir / f"{img.stem}.txt"
            if not label_path.exists():
                missing_labels.append(img.name)
        
        if missing_labels:
            report["warnings"].append(
                f"{split}: {len(missing_labels)} images without labels"
            )
        
        # Validate label format
        invalid_labels = []
        total_objects = 0
        for label_file in labels:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            invalid_labels.append(label_file.name)
                            break
                        # Check if values are valid
                        cls, x, y, w, h = map(float, parts)
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            invalid_labels.append(label_file.name)
                            break
                        total_objects += 1
            except Exception as e:
                invalid_labels.append(f"{label_file.name}: {str(e)}")
        
        if invalid_labels:
            report["errors"].extend([f"{split}: Invalid label format in {lbl}" for lbl in invalid_labels[:5]])
            report["valid"] = False
        
        report["stats"][split] = {
            "images": len(images),
            "labels": len(labels),
            "missing_labels": len(missing_labels),
            "invalid_labels": len(invalid_labels),
            "total_objects": total_objects,
            "avg_objects_per_image": round(total_objects / len(images), 2) if images else 0,
        }
    
    return report


def validate_severity_dataset(dataset_dir: str) -> Dict[str, any]:
    """
    Validate severity classification dataset.
    
    Args:
        dataset_dir: Path to dataset root (contains train/ and val/)
    
    Returns:
        Validation report dictionary
    """
    base = Path(dataset_dir)
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }
    
    classes = ["minimal", "low", "medium", "high", "critical"]
    
    for split in ["train", "val"]:
        split_dir = base / split
        if not split_dir.exists():
            report["errors"].append(f"Missing directory: {split_dir}")
            report["valid"] = False
            continue
        
        class_counts = {}
        for cls in classes:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                report["warnings"].append(f"Missing class directory: {cls_dir}")
                class_counts[cls] = 0
            else:
                images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
                class_counts[cls] = len(images)
        
        total = sum(class_counts.values())
        if total == 0:
            report["errors"].append(f"{split}: No images found")
            report["valid"] = False
        
        # Check for class imbalance
        if total > 0:
            min_count = min(class_counts.values())
            max_count = max(class_counts.values())
            if max_count > min_count * 5:
                report["warnings"].append(
                    f"{split}: Significant class imbalance detected (ratio {max_count/min_count:.1f}:1)"
                )
        
        report["stats"][split] = {
            "total_images": total,
            "class_distribution": class_counts,
            "class_balance": {
                cls: f"{count/total*100:.1f}%" if total > 0 else "0%"
                for cls, count in class_counts.items()
            }
        }
    
    return report


def print_validation_report(report: Dict):
    """Pretty print validation report."""
    print("\n" + "="*60)
    print("DATASET VALIDATION REPORT")
    print("="*60)
    
    if report["valid"]:
        print("✅ Dataset is valid")
    else:
        print("❌ Dataset has errors")
    
    if report["errors"]:
        print("\n🔴 ERRORS:")
        for error in report["errors"]:
            print(f"  - {error}")
    
    if report["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
    
    print("\n📊 STATISTICS:")
    for split, stats in report["stats"].items():
        print(f"\n  {split.upper()}:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"    {key}:")
                for k, v in value.items():
                    print(f"      {k}: {v}")
            else:
                print(f"    {key}: {value}")
    
    print("\n" + "="*60)


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare and validate pothole datasets")
    parser.add_argument("--action", choices=["create", "validate-yolo", "validate-severity"],
                       required=True, help="Action to perform")
    parser.add_argument("--dataset-dir", type=str, default="ig2/data",
                       help="Dataset directory")
    
    args = parser.parse_args()
    
    if args.action == "create":
        create_dataset_structure(args.dataset_dir)
    
    elif args.action == "validate-yolo":
        yolo_dir = Path(args.dataset_dir) / "pothole_yolo"
        report = validate_yolo_dataset(str(yolo_dir))
        print_validation_report(report)
    
    elif args.action == "validate-severity":
        severity_dir = Path(args.dataset_dir) / "severity"
        report = validate_severity_dataset(str(severity_dir))
        print_validation_report(report)
