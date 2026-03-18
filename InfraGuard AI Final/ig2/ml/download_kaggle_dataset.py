"""
ml/download_kaggle_dataset.py – Download Kaggle Datasets Directly
==================================================================
Download Kaggle datasets without saving zip files locally.
Extracts directly to your project directory.
"""

import os
import sys
import io
import zipfile
import requests
from pathlib import Path
from typing import Optional
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("infraguard.kaggle")

# Kaggle dataset IDs
KAGGLE_DATASETS = {
    "potholes": {
        "id": "chitholian/annotated-potholes-dataset",
        "name": "Annotated Potholes Dataset",
        "description": "665 annotated pothole images with bounding boxes",
        "url": "https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset",
    },
    "road-damage": {
        "id": "sekilab/road-damage-dataset",
        "name": "Road Damage Dataset (RDD2022)",
        "description": "47,000+ images from 6 countries",
        "url": "https://www.kaggle.com/datasets/sekilab/road-damage-dataset",
    },
    "crack-detection": {
        "id": "cuicui0908/crack-detection-dataset",
        "name": "Crack Detection Dataset",
        "description": "Crack and pothole images for detection",
        "url": "https://www.kaggle.com/datasets/cuicui0908/crack-detection-dataset",
    },
}


def download_kaggle_dataset(
    dataset_id: str,
    target_dir: str = "ig2/data/kaggle",
    kaggle_username: Optional[str] = None,
    kaggle_key: Optional[str] = None,
) -> Path:
    """
    Download and extract a Kaggle dataset directly.
    
    Args:
        dataset_id: Kaggle dataset ID (e.g., 'chitholian/annotated-potholes-dataset')
        target_dir: Where to extract the dataset
        kaggle_username: Your Kaggle username (optional, uses env var if not provided)
        kaggle_key: Your Kaggle API key (optional, uses env var if not provided)
    
    Returns:
        Path to extracted dataset directory
    """
    # Get credentials from environment or parameters
    username = kaggle_username or os.environ.get("KAGGLE_USERNAME")
    key = kaggle_key or os.environ.get("KAGGLE_KEY")
    
    if not username or not key:
        logger.warning("Kaggle credentials not provided")
        logger.info("To use Kaggle API, set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        logger.info("Or download manually from: https://www.kaggle.com/datasets")
        return None
    
    # Create target directory
    target = Path(target_dir) / dataset_id.split("/")[-1]
    target.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading dataset: {dataset_id}")
    logger.info(f"Target: {target}")
    
    try:
        # Download using Kaggle API
        import kaggle
        kaggle.api.dataset_download_files(dataset_id, path=str(target), unzip=True)
        
        logger.info(f"✅ Dataset downloaded and extracted to {target}")
        return target
    
    except ImportError:
        logger.warning("kaggle package not installed")
        logger.info("Install with: pip install kaggle")
        logger.info("Or use manual download method below")
        return None
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


def download_kaggle_via_api(
    dataset_id: str,
    target_dir: str = "ig2/data/kaggle",
    kaggle_username: Optional[str] = None,
    kaggle_key: Optional[str] = None,
) -> Path:
    """
    Download Kaggle dataset using direct API calls.
    Downloads zip to memory, extracts to disk.
    """
    # Get credentials from environment variables (loaded via .env)
    # Also accept explicit parameters for override
    username = kaggle_username or os.environ.get("KAGGLE_USERNAME")
    key = kaggle_key or os.environ.get("KAGGLE_KEY")
    
    if not username or not key:
        logger.error("Kaggle credentials not found")
        logger.info("1. Go to https://www.kaggle.com/settings/account")
        logger.info("2. Click 'Create New API Token' (downloads kaggle.json)")
        logger.info("3. Copy the values from kaggle.json:")
        logger.info("   - username: your_kaggle_username")
        logger.info("   - key: your_kaggle_api_key_here")
        logger.info("4. Add to ig2/.env file:")
        logger.info("   KAGGLE_USERNAME=your_kaggle_username")
        logger.info("   KAGGLE_KEY=your_kaggle_api_key_here")
        logger.info("5. Restart your terminal/session")
        return None
    
    # Construct download URL
    # Kaggle API endpoint for dataset download
    download_url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_id}"
    
    target = Path(target_dir) / dataset_id.split("/")[-1]
    target.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {dataset_id}...")
    
    try:
        # Download to memory
        response = requests.get(
            download_url,
            auth=(username, key),
            stream=True,
            headers={"User-Agent": "InfraGuardAI/1.0"}
        )
        response.raise_for_status()
        
        # Read zip file into memory
        zip_data = io.BytesIO(response.content)
        
        # Extract to target directory
        with zipfile.ZipFile(zip_data, 'r') as zip_ref:
            zip_ref.extractall(str(target))
        
        logger.info(f"✅ Extracted to {target}")
        return target
    
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Trying alternative method...")
        return None


def manual_download_instructions(dataset_name: str, dataset_url: str):
    """Print instructions for manual download."""
    logger.info(f"\n{'='*60}")
    logger.info(f"MANUAL DOWNLOAD INSTRUCTIONS: {dataset_name}")
    logger.info(f"{'='*60}")
    logger.info(f"1. Open: {dataset_url}")
    logger.info("2. Click 'Download' button")
    logger.info("3. Save the zip file to: ig2/data/kaggle/")
    logger.info("4. Extract the zip file")
    logger.info(f"5. Dataset will be at: ig2/data/kaggle/")
    logger.info(f"{'='*60}\n")


def list_available_datasets():
    """List all available Kaggle datasets."""
    logger.info("\nAvailable Kaggle Datasets:")
    logger.info("="*60)
    
    for key, info in KAGGLE_DATASETS.items():
        logger.info(f"\n📦 {info['name']}")
        logger.info(f"   ID: {info['id']}")
        logger.info(f"   Description: {info['description']}")
        logger.info(f"   URL: {info['url']}")
    
    logger.info("\n" + "="*60)


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Kaggle datasets for pothole detection")
    parser.add_argument("--dataset", choices=list(KAGGLE_DATASETS.keys()), default="potholes",
                       help="Which dataset to download")
    parser.add_argument("--manual", action="store_true",
                       help="Show manual download instructions only")
    parser.add_argument("--list", action="store_true",
                       help="List available datasets")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_datasets()
        exit(0)
    
    if args.manual:
        dataset = KAGGLE_DATASETS[args.dataset]
        manual_download_instructions(dataset["name"], dataset["url"])
        exit(0)
    
    # Try automatic download
    dataset_id = KAGGLE_DATASETS[args.dataset]["id"]
    target = download_kaggle_via_api(dataset_id)
    
    if target is None:
        dataset = KAGGLE_DATASETS[args.dataset]
        manual_download_instructions(dataset["name"], dataset["url"])
