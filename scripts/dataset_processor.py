"""
Dataset Processing Script for Rico Icon Dataset
Converts Rico annotations to YOLO format for icon detection
"""

import os
import json
import cv2
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np


class RicoDatasetProcessor:
    """Process Rico dataset and convert to YOLO format"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_config = self.config['dataset']
        self.class_names = self.dataset_config['class_names']
        self.num_classes = self.dataset_config['num_classes']
        
        # Create output directories
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directory structure"""
        paths = self.dataset_config['paths']
        
        for split in ['train', 'val', 'test']:
            images_dir = Path(paths[split]) / 'images'
            labels_dir = Path(paths[split]) / 'labels'
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
        
        print("✓ Directory structure created")
    
    def parse_rico_annotation(self, annotation_file: str) -> List[Dict]:
        """
        Parse Rico dataset annotation JSON file
        
        Args:
            annotation_file: Path to Rico annotation JSON
            
        Returns:
            List of UI element annotations
        """
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        elements = []
        
        # Rico format varies, handle common structures
        if 'children' in data:
            elements = self._parse_hierarchy(data)
        elif isinstance(data, list):
            elements = data
        
        return elements
    
    def _parse_hierarchy(self, node: Dict, elements: List = None) -> List:
        """Recursively parse UI hierarchy"""
        if elements is None:
            elements = []
        
        if 'bounds' in node and 'componentLabel' in node:
            elements.append(node)
        
        if 'children' in node:
            for child in node['children']:
                self._parse_hierarchy(child, elements)
        
        return elements
    
    def convert_to_yolo_format(
        self, 
        elements: List[Dict], 
        img_width: int, 
        img_height: int
    ) -> List[str]:
        """
        Convert Rico annotations to YOLO format
        
        YOLO format: <class_id> <x_center> <y_center> <width> <height> (normalized)
        
        Args:
            elements: List of UI elements with bounds
            img_width: Image width
            img_height: Image height
            
        Returns:
            List of YOLO formatted annotation lines
        """
        yolo_annotations = []
        
        for element in elements:
            # Check if element is an icon class
            component_label = element.get('componentLabel', '')
            icon_class = self._map_to_icon_class(component_label)
            
            if icon_class == -1:
                continue  # Skip non-icon elements
            
            # Parse bounds [x1, y1, x2, y2]
            bounds = element.get('bounds', [])
            if len(bounds) != 4:
                continue
            
            x1, y1, x2, y2 = bounds
            
            # Convert to YOLO format (normalized center x, y, width, height)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # Validate bounds
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                   0 < width <= 1 and 0 < height <= 1):
                continue
            
            # Format: class_id x_center y_center width height
            annotation = f"{icon_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_annotations.append(annotation)
        
        return yolo_annotations
    
    def _map_to_icon_class(self, component_label: str) -> int:
        """
        Map Rico component label to icon class index
        
        Args:
            component_label: Original Rico component label
            
        Returns:
            Class index (-1 if not an icon)
        """
        # Mapping dictionary (customize based on your dataset)
        label_mapping = {
            'Icon': -1,  # Generic icon, needs further classification
            'back': 0,
            'search': 1,
            'menu': 2,
            'home': 3,
            'settings': 4,
            'share': 5,
            'delete': 6,
            'edit': 7,
            'add': 8,
            'close': 9,
            'favorite': 10,
            'profile': 11,
            'notification': 12,
            'camera': 13,
            'gallery': 14,
            'download': 15,
            'upload': 16,
            'play': 17,
            'pause': 18,
            'refresh': 19,
            'filter': 20,
            'sort': 21,
            'calendar': 22,
            'location': 23,
            'phone': 24,
            'email': 25
        }
        
        # Try to find match in label
        label_lower = component_label.lower()
        for key, value in label_mapping.items():
            if key in label_lower:
                return value
        
        return -1  # Not an icon class
    
    def process_dataset(
        self, 
        rico_images_dir: str, 
        rico_annotations_dir: str,
        max_samples: int = None
    ):
        """
        Process entire Rico dataset
        
        Args:
            rico_images_dir: Directory containing Rico screenshots
            rico_annotations_dir: Directory containing Rico JSON annotations
            max_samples: Maximum number of samples to process (None for all)
        """
        # Get all image files
        image_files = list(Path(rico_images_dir).glob("*.jpg")) + \
                     list(Path(rico_images_dir).glob("*.png"))
        
        if max_samples:
            image_files = image_files[:max_samples]
        
        print(f"Processing {len(image_files)} images...")
        
        # Split dataset
        np.random.shuffle(image_files)
        train_split = int(len(image_files) * self.dataset_config['train_split'])
        val_split = int(len(image_files) * (self.dataset_config['train_split'] + 
                                            self.dataset_config['val_split']))
        
        splits = {
            'train': image_files[:train_split],
            'val': image_files[train_split:val_split],
            'test': image_files[val_split:]
        }
        
        # Process each split
        for split_name, split_files in splits.items():
            print(f"\nProcessing {split_name} split ({len(split_files)} images)...")
            self._process_split(split_files, rico_annotations_dir, split_name)
        
        # Create data.yaml for YOLO
        self.create_data_yaml()
        
        print("\n✓ Dataset processing complete!")
    
    def _process_split(
        self, 
        image_files: List[Path], 
        annotations_dir: str, 
        split_name: str
    ):
        """Process a single data split"""
        paths = self.dataset_config['paths']
        split_path = Path(paths[split_name])
        
        processed_count = 0
        skipped_count = 0
        
        for img_path in tqdm(image_files, desc=f"Processing {split_name}"):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                skipped_count += 1
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Find corresponding annotation file
            annotation_file = Path(annotations_dir) / f"{img_path.stem}.json"
            if not annotation_file.exists():
                skipped_count += 1
                continue
            
            # Parse annotations
            try:
                elements = self.parse_rico_annotation(str(annotation_file))
                yolo_annotations = self.convert_to_yolo_format(
                    elements, img_width, img_height
                )
                
                # Skip if no valid annotations
                if not yolo_annotations:
                    skipped_count += 1
                    continue
                
                # Save image
                dest_img_path = split_path / 'images' / img_path.name
                shutil.copy(img_path, dest_img_path)
                
                # Save annotations
                dest_label_path = split_path / 'labels' / f"{img_path.stem}.txt"
                with open(dest_label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                skipped_count += 1
                continue
        
        print(f"  Processed: {processed_count}, Skipped: {skipped_count}")
    
    def create_data_yaml(self):
        """Create data.yaml file for YOLO training"""
        paths = self.dataset_config['paths']
        
        data_yaml = {
            'path': str(Path('data/processed').absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': self.num_classes,
            'names': self.class_names
        }
        
        with open('data/data.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print("✓ data.yaml created")


def main():
    """Main execution function"""
    # Initialize processor
    processor = RicoDatasetProcessor()
    
    # Example usage - update paths to your Rico dataset
    rico_images_dir = "data/raw/rico_screenshots"
    rico_annotations_dir = "data/raw/rico_annotations"
    
    # Check if directories exist
    if not os.path.exists(rico_images_dir):
        print(f"Creating placeholder directory: {rico_images_dir}")
        os.makedirs(rico_images_dir, exist_ok=True)
        print("Please place Rico dataset screenshots in this directory")
    
    if not os.path.exists(rico_annotations_dir):
        print(f"Creating placeholder directory: {rico_annotations_dir}")
        os.makedirs(rico_annotations_dir, exist_ok=True)
        print("Please place Rico dataset annotations in this directory")
    
    # Process dataset (use max_samples for testing)
    # processor.process_dataset(
    #     rico_images_dir=rico_images_dir,
    #     rico_annotations_dir=rico_annotations_dir,
    #     max_samples=1000  # Remove or set to None for full dataset
    # )


if __name__ == "__main__":
    main()
