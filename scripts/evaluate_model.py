"""
Model Evaluation and Benchmarking Script
Evaluates YOLOv8 icon detection model performance
"""

import os
import yaml
import torch
import time
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Evaluate icon detection model performance"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.dataset_config = self.config['dataset']
        self.model = None
        
    def load_model(self, model_path: str):
        """Load trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = YOLO(model_path)
        print(f"✓ Model loaded from {model_path}")
    
    def evaluate_on_test_set(self, data_yaml: str = "data/data.yaml"):
        """
        Evaluate model on test dataset
        
        Args:
            data_yaml: Path to data configuration
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        print("\nEvaluating model on test set...")
        
        results = self.model.val(
            data=data_yaml,
            split='test',
            imgsz=self.model_config['input_size'],
            conf=self.model_config['confidence_threshold'],
            iou=self.model_config['iou_threshold'],
            device=0 if torch.cuda.is_available() else 'cpu',
            plots=True,
            save_json=True
        )
        
        # Extract metrics
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1_score': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr) 
                       if (results.box.mp + results.box.mr) > 0 else 0.0
        }
        
        # Per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(self.dataset_config['class_names']):
            if i < len(results.box.maps):
                class_metrics[class_name] = {
                    'ap50': float(results.box.maps[i]),
                    'ap': float(results.box.map[i]) if hasattr(results.box, 'map') else 0.0
                }
        
        metrics['per_class'] = class_metrics
        
        # Print results
        self._print_metrics(metrics)
        
        return metrics
    
    def benchmark_inference_speed(
        self, 
        test_images_dir: str = "data/processed/test/images",
        num_samples: int = 100
    ):
        """
        Benchmark inference speed
        
        Args:
            test_images_dir: Directory with test images
            num_samples: Number of images to test
            
        Returns:
            Dictionary with speed metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        print(f"\nBenchmarking inference speed on {num_samples} images...")
        
        # Get test images
        image_paths = list(Path(test_images_dir).glob("*.jpg")) + \
                     list(Path(test_images_dir).glob("*.png"))
        image_paths = image_paths[:num_samples]
        
        if len(image_paths) == 0:
            print(f"No images found in {test_images_dir}")
            return {}
        
        # Warm-up
        print("Warming up...")
        for _ in range(5):
            self.model.predict(
                str(image_paths[0]),
                conf=self.model_config['confidence_threshold'],
                verbose=False
            )
        
        # Benchmark
        inference_times = []
        
        print(f"Running inference on {len(image_paths)} images...")
        for img_path in image_paths:
            start_time = time.time()
            
            results = self.model.predict(
                str(img_path),
                conf=self.model_config['confidence_threshold'],
                iou=self.model_config['iou_threshold'],
                imgsz=self.model_config['input_size'],
                verbose=False
            )
            
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time)
        
        # Calculate statistics
        speed_metrics = {
            'mean_inference_time_ms': float(np.mean(inference_times)),
            'median_inference_time_ms': float(np.median(inference_times)),
            'min_inference_time_ms': float(np.min(inference_times)),
            'max_inference_time_ms': float(np.max(inference_times)),
            'std_inference_time_ms': float(np.std(inference_times)),
            'fps': float(1000 / np.mean(inference_times)),
            'num_samples': len(image_paths)
        }
        
        # Print results
        print("\nInference Speed Metrics:")
        print(f"  Mean: {speed_metrics['mean_inference_time_ms']:.2f} ms")
        print(f"  Median: {speed_metrics['median_inference_time_ms']:.2f} ms")
        print(f"  Min: {speed_metrics['min_inference_time_ms']:.2f} ms")
        print(f"  Max: {speed_metrics['max_inference_time_ms']:.2f} ms")
        print(f"  FPS: {speed_metrics['fps']:.2f}")
        
        return speed_metrics
    
    def analyze_model_size(self, model_path: str):
        """
        Analyze model file size and parameters
        
        Args:
            model_path: Path to model file
            
        Returns:
            Dictionary with model size metrics
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # File size
        file_size_bytes = os.path.getsize(model_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Count parameters
        if self.model is None:
            self.load_model(model_path)
        
        total_params = sum(p.numel() for p in self.model.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        
        size_metrics = {
            'file_size_mb': float(file_size_mb),
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'model_architecture': self.model_config['name']
        }
        
        print("\nModel Size Metrics:")
        print(f"  File size: {size_metrics['file_size_mb']:.2f} MB")
        print(f"  Total parameters: {size_metrics['total_parameters']:,}")
        print(f"  Trainable parameters: {size_metrics['trainable_parameters']:,}")
        
        return size_metrics
    
    def generate_visualizations(self, metrics: dict, output_dir: str = "evaluation/plots"):
        """
        Generate visualization plots
        
        Args:
            metrics: Evaluation metrics dictionary
            output_dir: Directory to save plots
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Per-class mAP plot
        if 'per_class' in metrics:
            self._plot_per_class_metrics(metrics['per_class'], output_dir)
        
        print(f"\n✓ Visualizations saved to {output_dir}")
    
    def _plot_per_class_metrics(self, class_metrics: dict, output_dir: str):
        """Plot per-class mAP"""
        class_names = list(class_metrics.keys())
        ap50_values = [class_metrics[c]['ap50'] for c in class_names]
        
        plt.figure(figsize=(14, 8))
        bars = plt.barh(class_names, ap50_values, color='steelblue')
        
        # Color bars based on performance
        for i, bar in enumerate(bars):
            if ap50_values[i] >= 0.8:
                bar.set_color('green')
            elif ap50_values[i] >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.xlabel('Average Precision @ IoU=0.50', fontsize=12)
        plt.ylabel('Icon Class', fontsize=12)
        plt.title('Per-Class Detection Performance (AP50)', fontsize=14, fontweight='bold')
        plt.xlim(0, 1.0)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'per_class_ap50.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_metrics(self, metrics: dict):
        """Print evaluation metrics in formatted way"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"mAP@0.5:      {metrics['mAP50']:.4f}")
        print(f"mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print(f"F1-Score:     {metrics['f1_score']:.4f}")
        print("="*60)
    
    def save_results(self, results: dict, output_path: str = "evaluation/metrics.json"):
        """Save evaluation results to JSON"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")


def main():
    """Main evaluation execution"""
    print("\n" + "="*60)
    print("Model Evaluation & Benchmarking")
    print("="*60 + "\n")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Model path
    model_path = "models/best_icon_detector.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using scripts/train_model.py")
        return
    
    # Load model
    evaluator.load_model(model_path)
    
    # Check if test data exists
    data_yaml = "data/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"Error: Data configuration not found at {data_yaml}")
        print("Please prepare the dataset first using scripts/dataset_processor.py")
        return
    
    all_results = {}
    
    # 1. Model size analysis
    print("\n" + "="*60)
    print("1. MODEL SIZE ANALYSIS")
    print("="*60)
    size_metrics = evaluator.analyze_model_size(model_path)
    all_results['model_size'] = size_metrics
    
    # 2. Evaluate on test set
    print("\n" + "="*60)
    print("2. TEST SET EVALUATION")
    print("="*60)
    try:
        eval_metrics = evaluator.evaluate_on_test_set(data_yaml)
        all_results['evaluation'] = eval_metrics
        
        # Generate visualizations
        evaluator.generate_visualizations(eval_metrics)
    except Exception as e:
        print(f"Test set evaluation failed: {e}")
    
    # 3. Benchmark inference speed
    print("\n" + "="*60)
    print("3. INFERENCE SPEED BENCHMARK")
    print("="*60)
    try:
        speed_metrics = evaluator.benchmark_inference_speed()
        all_results['speed'] = speed_metrics
    except Exception as e:
        print(f"Speed benchmark failed: {e}")
    
    # Save all results
    evaluator.save_results(all_results, "evaluation/complete_metrics.json")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print("\nCheck the 'evaluation' directory for detailed results and plots.")


if __name__ == "__main__":
    main()
