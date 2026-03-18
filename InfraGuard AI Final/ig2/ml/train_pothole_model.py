"""
ml/train_pothole_model.py – Fine-tuning Script for Pothole Detection
=====================================================================
Train and fine-tune models on custom pothole datasets.

Supports:
  1. YOLOv8 fine-tuning for detection
  2. ResNet-50 fine-tuning for severity classification
  3. Data augmentation
  4. Transfer learning from pre-trained weights

Dataset format:
  - YOLO format for detection (images + labels in txt files)
  - Classification format for severity (images in folders by class)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("infraguard.training")

try:
    from ultralytics import YOLO
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms, models
    from PIL import Image
    import numpy as np
    HAS_DEPS = True
except ImportError:
    logger.error("Training dependencies not installed. Run: pip install ultralytics torch torchvision")
    HAS_DEPS = False


class PotholeDataset(Dataset):
    """Dataset for severity classification."""
    
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ['minimal', 'low', 'medium', 'high', 'critical']
        self.samples = []
        
        for idx, class_name in enumerate(self.classes):
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg') + class_dir.glob('*.png'):
                    self.samples.append((str(img_path), idx))
        
        logger.info(f"Loaded {len(self.samples)} images from {root_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label



class PotholeTrainer:
    """Main training class for pothole detection models."""
    
    def __init__(self, output_dir: str = "ig2/ml/weights"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on device: {self.device}")
    
    def train_yolo_detector(
        self,
        dataset_yaml: str,
        epochs: int = 50,
        batch_size: int = 16,
        img_size: int = 640,
        pretrained: str = "yolov8n.pt",
    ) -> Dict[str, Any]:
        """
        Fine-tune YOLOv8 for pothole detection.
        
        Args:
            dataset_yaml: Path to dataset YAML config
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Input image size
            pretrained: Pre-trained weights to start from
        
        Returns:
            Training results dictionary
        """
        if not HAS_DEPS:
            raise RuntimeError("Training dependencies not installed")
        
        logger.info(f"Starting YOLO training with {epochs} epochs")
        
        # Load pre-trained model
        model = YOLO(pretrained)
        
        # Train
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=self.device,
            project=str(self.output_dir),
            name="pothole_yolo",
            exist_ok=True,
            patience=10,  # Early stopping
            save=True,
            plots=True,
            # Data augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
        )
        
        # Save best model
        best_model_path = self.output_dir / "pothole_yolo.pt"
        model.save(str(best_model_path))
        
        logger.info(f"✅ YOLO training complete. Model saved to {best_model_path}")
        
        return {
            "model_path": str(best_model_path),
            "metrics": results.results_dict if hasattr(results, 'results_dict') else {},
        }

    
    def train_severity_classifier(
        self,
        train_dir: str,
        val_dir: str,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Fine-tune ResNet-50 for severity classification.
        
        Dataset structure:
            train_dir/
                minimal/
                    img1.jpg
                    img2.jpg
                low/
                medium/
                high/
                critical/
            val_dir/
                (same structure)
        
        Args:
            train_dir: Training data directory
            val_dir: Validation data directory
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Initial learning rate
        
        Returns:
            Training results dictionary
        """
        if not HAS_DEPS:
            raise RuntimeError("Training dependencies not installed")
        
        logger.info(f"Starting severity classifier training with {epochs} epochs")
        
        # Data transforms
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Load datasets
        train_dataset = PotholeDataset(train_dir, transform=train_transform)
        val_dataset = PotholeDataset(val_dir, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Load pre-trained ResNet-50
        model = models.resnet50(pretrained=True)
        
        # Freeze early layers
        for param in list(model.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace final layer for 5 classes
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 5)
        model = model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        
        # Training loop
        best_val_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100.0 * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100.0 * val_correct / val_total
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = self.output_dir / "severity_resnet.pt"
                torch.save(model, str(model_path))
                logger.info(f"✅ Saved best model with val_acc: {val_acc:.2f}%")
        
        logger.info(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
        
        return {
            "model_path": str(self.output_dir / "severity_resnet.pt"),
            "best_val_acc": best_val_acc,
            "history": history,
        }



def create_dataset_yaml(
    train_images: str,
    val_images: str,
    output_path: str = "ig2/data/pothole_dataset.yaml"
) -> str:
    """
    Create YOLO dataset configuration file.
    
    Args:
        train_images: Path to training images directory
        val_images: Path to validation images directory
        output_path: Where to save the YAML file
    
    Returns:
        Path to created YAML file
    """
    config = {
        'path': str(Path(train_images).parent.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'pothole',
            1: 'crack',
            2: 'patch',
        },
        'nc': 3,  # number of classes
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created dataset config at {output_path}")
    return str(output_path)


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train pothole detection models")
    parser.add_argument("--mode", choices=["yolo", "severity", "both"], default="both",
                       help="Which model to train")
    parser.add_argument("--yolo-data", type=str, default="ig2/data/pothole_dataset.yaml",
                       help="Path to YOLO dataset YAML")
    parser.add_argument("--train-dir", type=str, default="ig2/data/severity/train",
                       help="Training data directory for severity classifier")
    parser.add_argument("--val-dir", type=str, default="ig2/data/severity/val",
                       help="Validation data directory for severity classifier")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--output-dir", type=str, default="ig2/ml/weights",
                       help="Output directory for trained models")
    
    args = parser.parse_args()
    
    if not HAS_DEPS:
        logger.error("Please install training dependencies: pip install ultralytics torch torchvision")
        exit(1)
    
    trainer = PotholeTrainer(output_dir=args.output_dir)
    
    if args.mode in ["yolo", "both"]:
        logger.info("=" * 60)
        logger.info("Training YOLO Detection Model")
        logger.info("=" * 60)
        yolo_results = trainer.train_yolo_detector(
            dataset_yaml=args.yolo_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        logger.info(f"YOLO Results: {yolo_results}")
    
    if args.mode in ["severity", "both"]:
        logger.info("=" * 60)
        logger.info("Training Severity Classifier")
        logger.info("=" * 60)
        severity_results = trainer.train_severity_classifier(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        logger.info(f"Severity Results: {severity_results}")
    
    logger.info("=" * 60)
    logger.info("✅ Training Complete!")
    logger.info("=" * 60)
