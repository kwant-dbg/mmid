"""
YOLOv8 Model Training Script for Icon Detection
Trains YOLOv8 Nano model on Rico icon dataset
"""

import os
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import logging


class IconDetectionTrainer:
    """Trainer class for YOLOv8 icon detection model"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.dataset_config = self.config['dataset']
        
        # Setup logging
        self.setup_logging()
        
        # Initialize model
        self.model = None
        
    def setup_logging(self):
        """Configure logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Training logger initialized")
    
    def initialize_model(self, pretrained: bool = True):
        """
        Initialize YOLOv11 model (Latest 2025)
        
        Args:
            pretrained: Whether to use pretrained weights
        """
        model_name = self.model_config['name']
        
        if pretrained:
            # Load pretrained YOLOv11 model
            self.model = YOLO(f"{model_name}.pt")
            self.logger.info(f"Loaded pretrained {model_name} model")
        else:
            # Load model architecture only
            self.model = YOLO(f"{model_name}.yaml")
            self.logger.info(f"Initialized {model_name} architecture from scratch")
        
        # Enable latest optimizations for YOLOv11
        if torch.cuda.is_available():
            # Enable TF32 for 3x faster training on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            self.logger.info("Enabled TF32 and cuDNN optimizations")
        
        # Log model parameters
        total_params = sum(p.numel() for p in self.model.model.parameters())
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Model: {model_name} (YOLOv11 - Latest 2024/2025)")
        
        return self.model
    
    def train(
        self, 
        data_yaml: str = "data/data.yaml",
        resume: bool = False,
        weights_path: str = None
    ):
        """
        Train the YOLOv8 model
        
        Args:
            data_yaml: Path to data configuration file
            resume: Whether to resume from last checkpoint
            weights_path: Path to custom weights for transfer learning
        """
        if self.model is None:
            self.initialize_model(pretrained=True)
        
        # If custom weights provided, load them
        if weights_path and os.path.exists(weights_path):
            self.model = YOLO(weights_path)
            self.logger.info(f"Loaded custom weights from {weights_path}")
        
        # Training arguments
        train_args = {
            'data': data_yaml,
            'epochs': self.training_config['epochs'],
            'batch': self.training_config['batch_size'],
            'imgsz': self.model_config['input_size'],
            'lr0': self.training_config['learning_rate'],
            'weight_decay': self.training_config['weight_decay'],
            'momentum': self.training_config['momentum'],
            'optimizer': self.training_config['optimizer'],
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'workers': 8,  # Increased for faster data loading
            'project': 'runs/train',
            'name': f'icon_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'pretrained': True,
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'resume': resume,
            
            # YOLOv11 Advanced Features
            'amp': True,  # Automatic Mixed Precision (2x faster)
            'fraction': 1.0,  # Use 100% of dataset
            'cache': 'ram',  # Cache images in RAM for faster training
            'multi_scale': True,  # Multi-scale training
            'close_mosaic': 10,  # Disable mosaic last 10 epochs
            
            # Augmentation parameters
            'hsv_h': self.training_config['augmentation']['hsv_h'],
            'hsv_s': self.training_config['augmentation']['hsv_s'],
            'hsv_v': self.training_config['augmentation']['hsv_v'],
            'degrees': self.training_config['augmentation']['degrees'],
            'translate': self.training_config['augmentation']['translate'],
            'scale': self.training_config['augmentation']['scale'],
            'shear': self.training_config['augmentation']['shear'],
            'perspective': self.training_config['augmentation']['perspective'],
            'flipud': self.training_config['augmentation']['flipud'],
            'fliplr': self.training_config['augmentation']['fliplr'],
            'mosaic': self.training_config['augmentation']['mosaic'],
            'mixup': self.training_config['augmentation']['mixup'],
            'copy_paste': 0.0,  # Copy-paste augmentation
            
            # Early stopping
            'patience': self.training_config['early_stopping']['patience'],
            
            # Save best model
            'save': True,
            'save_period': 10,
            'plots': True,
        }
        
        self.logger.info("Starting training with configuration:")
        for key, value in train_args.items():
            self.logger.info(f"  {key}: {value}")
        
        # Train model
        try:
            results = self.model.train(**train_args)
            self.logger.info("Training completed successfully!")
            return results
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def validate(self, data_yaml: str = "data/data.yaml"):
        """
        Validate the trained model
        
        Args:
            data_yaml: Path to data configuration file
        """
        if self.model is None:
            self.logger.error("Model not initialized. Please train or load a model first.")
            return
        
        self.logger.info("Starting validation...")
        
        val_args = {
            'data': data_yaml,
            'imgsz': self.model_config['input_size'],
            'batch': self.training_config['batch_size'],
            'conf': self.model_config['confidence_threshold'],
            'iou': self.model_config['iou_threshold'],
            'device': 0 if torch.cuda.is_available() else 'cpu',
        }
        
        results = self.model.val(**val_args)
        
        # Log metrics
        self.logger.info(f"Validation Results:")
        self.logger.info(f"  mAP50: {results.box.map50:.4f}")
        self.logger.info(f"  mAP50-95: {results.box.map:.4f}")
        self.logger.info(f"  Precision: {results.box.mp:.4f}")
        self.logger.info(f"  Recall: {results.box.mr:.4f}")
        
        return results
    
    def export_model(
        self, 
        format: str = "onnx",
        output_dir: str = "models/exported"
    ):
        """
        Export trained model to different formats
        
        Args:
            format: Export format (onnx, torchscript, tflite, etc.)
            output_dir: Directory to save exported model
        """
        if self.model is None:
            self.logger.error("Model not initialized")
            return
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Exporting model to {format} format...")
        
        try:
            export_path = self.model.export(
                format=format,
                imgsz=self.model_config['input_size'],
                optimize=True
            )
            self.logger.info(f"Model exported to: {export_path}")
            return export_path
        except Exception as e:
            self.logger.error(f"Export failed: {str(e)}")
            raise
    
    def save_model(self, output_path: str = "models/best_icon_detector.pt"):
        """
        Save the trained model
        
        Args:
            output_path: Path to save the model
        """
        if self.model is None:
            self.logger.error("Model not initialized")
            return
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # The best model is automatically saved during training
        # Copy it to a standard location
        import shutil
        
        # Find the best.pt file in the latest training run
        runs_dir = Path("runs/train")
        if runs_dir.exists():
            # Get the most recent training directory
            train_dirs = sorted(runs_dir.glob("icon_detection_*"), 
                              key=os.path.getmtime, reverse=True)
            if train_dirs:
                best_pt = train_dirs[0] / "weights" / "best.pt"
                if best_pt.exists():
                    shutil.copy(best_pt, output_path)
                    self.logger.info(f"Model saved to: {output_path}")
                    return output_path
        
        self.logger.warning("Could not find best.pt file")
        return None


def main():
    """Main training execution"""
    # Initialize trainer
    trainer = IconDetectionTrainer()
    
    # Check if data.yaml exists
    data_yaml = "data/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"Error: {data_yaml} not found!")
        print("Please run dataset_processor.py first to prepare the dataset.")
        return
    
    # Initialize model
    trainer.initialize_model(pretrained=True)
    
    # Train model
    print("\n" + "="*60)
    print("Starting YOLOv8 Icon Detection Training")
    print("="*60 + "\n")
    
    results = trainer.train(data_yaml=data_yaml)
    
    # Validate model
    print("\n" + "="*60)
    print("Validating Model")
    print("="*60 + "\n")
    
    val_results = trainer.validate(data_yaml=data_yaml)
    
    # Save model
    model_path = trainer.save_model("models/best_icon_detector.pt")
    
    # Export to ONNX for deployment
    print("\n" + "="*60)
    print("Exporting Model to ONNX")
    print("="*60 + "\n")
    
    try:
        trainer.export_model(format="onnx", output_dir="models/exported")
    except Exception as e:
        print(f"Warning: ONNX export failed: {e}")
    
    print("\n" + "="*60)
    print("Training Pipeline Complete!")
    print("="*60)
    print(f"\nBest model saved to: {model_path}")
    print("Check runs/train/ for detailed training logs and visualizations")


if __name__ == "__main__":
    main()
