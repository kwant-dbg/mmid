"""
Production Deployment Script
Complete end-to-end inference pipeline with multi-modal analysis
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional
import yaml
import sys

# Add scripts to path
sys.path.append(str(Path(__file__).parent))

from ocr_integration import MultiModalUIAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionInferenceEngine:
    """Production-ready inference engine with multi-modal analysis"""
    
    def __init__(
        self,
        model_path: str = "models/best.pt",
        config_path: str = "config/config.yaml",
        use_ocr: bool = True,
        device: str = 'auto'
    ):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration
            use_ocr: Enable OCR for multi-modal analysis
            device: Device for inference ('cpu', 'cuda', 'auto')
        """
        self.config_path = Path(config_path)
        self.use_ocr = use_ocr
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize YOLO model
        logger.info(f"Loading model from {model_path}")
        self.model = YOLO(model_path)
        
        # Set device
        if device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize multi-modal analyzer
        if use_ocr:
            logger.info("Initializing multi-modal analyzer...")
            self.analyzer = MultiModalUIAnalyzer(ocr_engine='easyocr')
        else:
            self.analyzer = None
        
        # Get class names
        self.class_names = self.config['dataset']['classes']
        
    def predict(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        save_visualization: bool = True,
        output_dir: str = "output"
    ) -> Dict:
        """
        Run inference on image
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold
            iou_threshold: IOU threshold for NMS
            save_visualization: Save visualized results
            output_dir: Output directory
            
        Returns:
            Results dictionary
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Load image
        logger.info(f"Processing {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Run YOLO detection
        logger.info("Running icon detection...")
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        # Parse detections
        detections = self._parse_detections(results, image.shape)
        
        # Multi-modal analysis
        multimodal_results = None
        if self.use_ocr and self.analyzer and len(detections) > 0:
            logger.info("Performing multi-modal analysis...")
            multimodal_results = self.analyzer.analyze_ui(image, detections)
        
        # Create visualization
        if save_visualization:
            vis_path = output_dir / f"{image_path.stem}_detection.jpg"
            self._visualize_results(
                image, detections, multimodal_results, vis_path
            )
        
        # Compile results
        output = {
            'image': str(image_path),
            'image_size': {
                'width': image.shape[1],
                'height': image.shape[0]
            },
            'detections': detections,
            'detection_count': len(detections),
            'class_distribution': self._get_class_distribution(detections),
            'multimodal_analysis': multimodal_results
        }
        
        # Save results JSON
        json_path = output_dir / f"{image_path.stem}_results.json"
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {json_path}")
        
        return output
    
    def predict_batch(
        self,
        image_dir: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        output_dir: str = "output/batch"
    ) -> List[Dict]:
        """
        Run batch inference on directory of images
        
        Args:
            image_dir: Directory containing images
            conf_threshold: Confidence threshold
            iou_threshold: IOU threshold
            output_dir: Output directory
            
        Returns:
            List of result dictionaries
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            f for f in image_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        logger.info(f"Processing {len(image_files)} images...")
        
        results = []
        for image_file in image_files:
            try:
                result = self.predict(
                    image_file,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    save_visualization=True,
                    output_dir=output_dir
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_file}: {e}")
        
        # Save batch summary
        summary = self._create_batch_summary(results)
        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Batch processing complete. Summary: {summary_path}")
        return results
    
    def _parse_detections(self, results, image_shape) -> List[Dict]:
        """Parse YOLO results into detection dictionaries"""
        h, w = image_shape[:2]
        detections = []
        
        boxes = results.boxes
        for i in range(len(boxes)):
            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            
            # Normalize coordinates
            x1_norm, y1_norm = x1 / w, y1 / h
            x2_norm, y2_norm = x2 / w, y2 / h
            
            # Get class and confidence
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            
            detection = {
                'class': self.class_names[cls_id],
                'class_id': cls_id,
                'confidence': conf,
                'bbox': [float(x1_norm), float(y1_norm), float(x2_norm), float(y2_norm)],
                'bbox_pixels': [int(x1), int(y1), int(x2), int(y2)]
            }
            detections.append(detection)
        
        return detections
    
    def _visualize_results(
        self,
        image: np.ndarray,
        detections: List[Dict],
        multimodal_results: Optional[Dict],
        output_path: Path
    ):
        """Create visualization of results"""
        vis_image = image.copy()
        h, w = image.shape[:2]
        
        # Draw icon detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox_pixels']
            label = f"{det['class']} {det['confidence']:.2f}"
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                vis_image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
        
        # Draw text detections if available
        if multimodal_results and 'text_detections' in multimodal_results:
            for text_det in multimodal_results['text_detections']:
                x1, y1, x2, y2 = text_det['bbox']
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        # Save visualization
        cv2.imwrite(str(output_path), vis_image)
        logger.info(f"Visualization saved to {output_path}")
    
    def _get_class_distribution(self, detections: List[Dict]) -> Dict:
        """Get distribution of detected classes"""
        distribution = {}
        for det in detections:
            cls = det['class']
            distribution[cls] = distribution.get(cls, 0) + 1
        return distribution
    
    def _create_batch_summary(self, results: List[Dict]) -> Dict:
        """Create summary of batch processing results"""
        total_detections = sum(r['detection_count'] for r in results)
        
        # Aggregate class distribution
        class_dist = {}
        for result in results:
            for cls, count in result['class_distribution'].items():
                class_dist[cls] = class_dist.get(cls, 0) + count
        
        summary = {
            'total_images': len(results),
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / len(results) if results else 0,
            'class_distribution': class_dist,
            'images_processed': [r['image'] for r in results]
        }
        
        return summary


def main():
    """Command-line interface for production inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production inference for icon detection")
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--image-dir', type=str, help='Directory of images for batch processing')
    parser.add_argument('--model', type=str, default='models/best.pt',
                       help='Path to trained model')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IOU threshold')
    parser.add_argument('--no-ocr', action='store_true',
                       help='Disable OCR multi-modal analysis')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Inference device')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = ProductionInferenceEngine(
        model_path=args.model,
        use_ocr=not args.no_ocr,
        device=args.device
    )
    
    # Run inference
    if args.image:
        # Single image
        results = engine.predict(
            args.image,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            output_dir=args.output
        )
        
        print("\n" + "="*60)
        print(f"DETECTION RESULTS: {args.image}")
        print("="*60)
        print(f"Total detections: {results['detection_count']}")
        print(f"\nClass distribution:")
        for cls, count in results['class_distribution'].items():
            print(f"  {cls}: {count}")
        
    elif args.image_dir:
        # Batch processing
        results = engine.predict_batch(
            args.image_dir,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            output_dir=args.output
        )
        
        print(f"\n✅ Processed {len(results)} images")
        
    else:
        parser.print_help()
        print("\nError: Please provide either --image or --image-dir")
        sys.exit(1)


if __name__ == "__main__":
    main()
