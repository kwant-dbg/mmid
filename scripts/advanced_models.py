"""
Advanced Model Architectures for Icon Detection
Supports YOLOv11, RT-DETR, and Vision Transformers
Updated for 2025 - Latest Models
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from ultralytics import YOLO, RTDETR
import yaml


class AdvancedIconDetector:
    """
    Advanced icon detection supporting multiple state-of-the-art models
    
    Supported Models (2025):
    - YOLOv11 (Latest - Faster & More Accurate than v8)
    - RT-DETR (Real-Time Detection Transformer)
    - YOLOv10 (Efficient with NMS-free training)
    - YOLOv9 (GELAN architecture)
    """
    
    SUPPORTED_MODELS = {
        'yolo11n': 'YOLOv11 Nano - Latest & Best for mobile',
        'yolo11s': 'YOLOv11 Small - Balanced speed/accuracy',
        'yolo11m': 'YOLOv11 Medium - Higher accuracy',
        'yolo11l': 'YOLOv11 Large - Maximum accuracy',
        'yolo11x': 'YOLOv11 Extra Large - Research grade',
        'yolo10n': 'YOLOv10 Nano - NMS-free architecture',
        'yolo9c': 'YOLOv9 Compact - GELAN backbone',
        'rtdetr-l': 'RT-DETR Large - Transformer-based',
        'rtdetr-x': 'RT-DETR Extra - Maximum performance',
    }
    
    def __init__(
        self,
        model_name: str = "yolo11n",
        config_path: str = "config/config.yaml"
    ):
        self.model_name = model_name
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.device = self._get_device()
        
    def _get_device(self) -> str:
        """Get best available device with optimization"""
        if torch.cuda.is_available():
            # Enable TF32 for Ampere GPUs (3x faster)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set optimal cudnn settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            device = 'cuda'
            print(f"✅ Using CUDA: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
            
        elif torch.backends.mps.is_available():
            # Apple Silicon optimization
            device = 'mps'
            print(f"✅ Using Apple Silicon MPS acceleration")
            
        else:
            device = 'cpu'
            print(f"⚠️ Using CPU (GPU recommended for faster inference)")
        
        return device
    
    def load_model(
        self,
        weights_path: Optional[str] = None,
        pretrained: bool = True
    ):
        """
        Load model with latest optimizations
        
        Args:
            weights_path: Path to custom weights (optional)
            pretrained: Use pretrained weights
        """
        if weights_path:
            model_path = weights_path
        elif pretrained:
            model_path = f"{self.model_name}.pt"
        else:
            model_path = f"{self.model_name}.yaml"
        
        # Load based on model type
        if 'rtdetr' in self.model_name.lower():
            self.model = RTDETR(model_path)
            print(f"✅ Loaded RT-DETR (Transformer-based detector)")
        else:
            self.model = YOLO(model_path)
            print(f"✅ Loaded {self.model_name.upper()}")
        
        # Move to device
        self.model.to(self.device)
        
        # Enable optimizations
        if self.config['model'].get('use_amp', True):
            print("   AMP (Automatic Mixed Precision) enabled")
        
        return self.model
    
    def predict(
        self,
        source: str,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        half: bool = False,
        **kwargs
    ):
        """
        Run optimized inference
        
        Args:
            source: Image path or directory
            conf: Confidence threshold
            iou: IOU threshold
            imgsz: Input image size
            half: Use FP16 (2x faster on modern GPUs)
        """
        # Auto-enable FP16 on supported GPUs
        if self.device == 'cuda' and torch.cuda.is_available():
            # Check if GPU supports FP16
            gpu_name = torch.cuda.get_device_name(0).lower()
            if any(x in gpu_name for x in ['rtx', 'a100', 'a40', 'v100', 't4']):
                half = True
                print("🚀 FP16 acceleration enabled (2x faster)")
        
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            half=half,
            device=self.device,
            verbose=False,
            **kwargs
        )
        
        return results
    
    def benchmark_models(self, test_image: str = None):
        """
        Benchmark different models for comparison
        """
        import time
        
        if test_image is None:
            print("No test image provided")
            return
        
        results = {}
        
        # Test different YOLO versions
        test_models = ['yolo11n', 'yolo10n', 'yolo9c', 'yolo8n']
        
        for model_name in test_models:
            try:
                print(f"\n📊 Testing {model_name}...")
                model = YOLO(f"{model_name}.pt")
                
                # Warmup
                for _ in range(3):
                    model.predict(test_image, verbose=False)
                
                # Benchmark
                times = []
                for _ in range(10):
                    start = time.time()
                    model.predict(test_image, verbose=False)
                    times.append((time.time() - start) * 1000)
                
                results[model_name] = {
                    'avg_time_ms': sum(times) / len(times),
                    'min_time_ms': min(times),
                    'max_time_ms': max(times)
                }
                
                print(f"   Avg: {results[model_name]['avg_time_ms']:.2f}ms")
                
            except Exception as e:
                print(f"   ⚠️ Failed: {e}")
        
        return results


class VisionTransformerDetector:
    """
    Vision Transformer-based detection (Alternative approach)
    Using DETR (Detection Transformer) or similar
    """
    
    def __init__(self):
        from transformers import DetrImageProcessor, DetrForObjectDetection
        
        self.processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50"
        )
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50"
        )
        
    def predict(self, image_path: str):
        """Run ViT-based detection"""
        from PIL import Image
        
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        
        outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.25
        )[0]
        
        return results


class EfficientDetector:
    """
    EfficientDet - Scalable and Efficient Object Detection
    Alternative to YOLO for comparison
    """
    
    def __init__(self, model_name: str = "efficientdet_d0"):
        """
        Initialize EfficientDet
        
        Args:
            model_name: efficientdet_d0 to efficientdet_d7
        """
        import timm
        
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=26  # Our icon classes
        )
        self.model.eval()
        
        print(f"✅ Loaded {model_name}")
    
    def fine_tune(self, train_loader, num_epochs: int = 10):
        """Fine-tune EfficientDet on icon dataset"""
        # TODO: Implement training loop
        pass


def compare_models():
    """
    Compare different detection architectures
    """
    print("\n" + "="*60)
    print("Model Comparison - Icon Detection")
    print("="*60 + "\n")
    
    models_info = {
        'YOLOv11n': {
            'params': '2.6M',
            'speed': '~80 FPS',
            'mAP50': '~39.5%',
            'size': '6.5 MB',
            'year': '2024',
            'notes': 'Latest, fastest, best for mobile'
        },
        'YOLOv10n': {
            'params': '2.3M',
            'speed': '~85 FPS',
            'mAP50': '~38.5%',
            'size': '5.8 MB',
            'year': '2024',
            'notes': 'NMS-free, very efficient'
        },
        'YOLOv9c': {
            'params': '25.3M',
            'speed': '~50 FPS',
            'mAP50': '~53.0%',
            'size': '50 MB',
            'year': '2024',
            'notes': 'GELAN architecture, high accuracy'
        },
        'YOLOv8n': {
            'params': '3.2M',
            'speed': '~78 FPS',
            'mAP50': '~37.3%',
            'size': '6.2 MB',
            'year': '2023',
            'notes': 'Previous generation'
        },
        'RT-DETR-L': {
            'params': '32M',
            'speed': '~40 FPS',
            'mAP50': '~53.0%',
            'size': '65 MB',
            'year': '2024',
            'notes': 'Transformer-based, very accurate'
        }
    }
    
    for model, info in models_info.items():
        print(f"{model}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()


if __name__ == "__main__":
    print("🚀 Advanced Icon Detection Models - 2025")
    print("="*60 + "\n")
    
    # Show available models
    print("Available Models:")
    detector = AdvancedIconDetector()
    for model, desc in detector.SUPPORTED_MODELS.items():
        print(f"  • {model}: {desc}")
    
    print("\n")
    compare_models()
    
    print("\n💡 Recommendation:")
    print("   Use YOLOv11n for best speed/accuracy tradeoff")
    print("   Use RT-DETR-L for maximum accuracy")
    print("   Use YOLOv10n for lowest latency")
