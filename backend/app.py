"""
Flask Backend API for Icon Detection
Provides REST API endpoint for real-time icon detection
"""

import os
import cv2
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import torch


class IconDetectionAPI:
    """Flask API for icon detection service"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.webapp_config = self.config['webapp']
        self.dataset_config = self.config['dataset']
        
        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for frontend access
        
        # Configure upload settings
        self.app.config['MAX_CONTENT_LENGTH'] = \
            self.webapp_config['max_upload_size_mb'] * 1024 * 1024
        self.app.config['UPLOAD_FOLDER'] = self.webapp_config['upload_folder']
        self.app.config['RESULTS_FOLDER'] = self.webapp_config['results_folder']
        
        # Create necessary directories
        Path(self.app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
        Path(self.app.config['RESULTS_FOLDER']).mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = None
        self.class_names = self.dataset_config['class_names']
        
        # Setup routes
        self.setup_routes()
    
    def load_model(self, model_path: str = "models/best_icon_detector.pt"):
        """
        Load the trained YOLOv8 model
        
        Args:
            model_path: Path to the trained model weights
        """
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            print("Using pretrained YOLOv8n as fallback")
            model_path = "yolov8n.pt"
        
        try:
            self.model = YOLO(model_path)
            
            # Set device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(device)
            
            print(f"✓ Model loaded successfully from {model_path}")
            print(f"  Device: {device}")
            print(f"  Number of classes: {len(self.class_names)}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in \
               self.webapp_config['allowed_extensions']
    
    def predict_image(
        self, 
        image_path: str,
        conf_threshold: float = None,
        iou_threshold: float = None
    ) -> Dict[str, Any]:
        """
        Run icon detection on an image
        
        Args:
            image_path: Path to the image file
            conf_threshold: Confidence threshold (optional)
            iou_threshold: IOU threshold for NMS (optional)
            
        Returns:
            Dictionary containing detection results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Use default thresholds if not provided
        if conf_threshold is None:
            conf_threshold = self.model_config['confidence_threshold']
        if iou_threshold is None:
            iou_threshold = self.model_config['iou_threshold']
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        img_height, img_width = img.shape[:2]
        
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=self.model_config['input_size'],
            max_det=self.model_config['max_detections'],
            verbose=False
        )
        
        # Parse results
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get box coordinates (xyxy format)
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = box
                
                # Get confidence and class
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                # Get class name
                class_name = self.class_names[class_id] if class_id < len(self.class_names) \
                            else f"class_{class_id}"
                
                detection = {
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                        'width': float(x2 - x1),
                        'height': float(y2 - y1)
                    },
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                }
                
                detections.append(detection)
        
        # Prepare response
        response = {
            'image': {
                'width': img_width,
                'height': img_height,
                'filename': os.path.basename(image_path)
            },
            'detections': detections,
            'num_detections': len(detections),
            'model_config': {
                'confidence_threshold': conf_threshold,
                'iou_threshold': iou_threshold
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return response
    
    def draw_detections(
        self, 
        image_path: str, 
        detections: List[Dict],
        output_path: str = None
    ) -> str:
        """
        Draw bounding boxes on image
        
        Args:
            image_path: Path to the original image
            detections: List of detection dictionaries
            output_path: Path to save annotated image (optional)
            
        Returns:
            Path to the annotated image
        """
        img = cv2.imread(image_path)
        
        # Define colors for different classes (BGR format)
        np.random.seed(42)
        colors = {}
        for class_name in self.class_names:
            colors[class_name] = tuple(map(int, np.random.randint(0, 255, 3)))
        
        # Draw each detection
        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Get coordinates
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])
            
            # Get color for this class
            color = colors.get(class_name, (0, 255, 0))
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(
                img,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        # Save annotated image
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"annotated_{timestamp}.jpg"
            output_path = os.path.join(self.app.config['RESULTS_FOLDER'], filename)
        
        cv2.imwrite(output_path, img)
        return output_path
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/', methods=['GET'])
        def index():
            """Health check endpoint"""
            return jsonify({
                'status': 'ok',
                'message': 'Multi-Modal Icon Vision API',
                'version': '1.0.0',
                'model_loaded': self.model is not None
            })
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Main prediction endpoint"""
            # Check if file is present
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({'error': 'Empty filename'}), 400
            
            if not self.allowed_file(file.filename):
                return jsonify({
                    'error': f'Invalid file type. Allowed: {self.webapp_config["allowed_extensions"]}'
                }), 400
            
            try:
                # Save uploaded file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_filename = f"{timestamp}_{filename}"
                filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                
                # Get optional parameters
                conf_threshold = request.form.get('confidence', type=float)
                iou_threshold = request.form.get('iou', type=float)
                
                # Run prediction
                results = self.predict_image(
                    filepath,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
                
                # Draw detections and save annotated image
                annotated_path = self.draw_detections(filepath, results['detections'])
                results['annotated_image'] = os.path.basename(annotated_path)
                
                # Clean up uploaded file (optional)
                # os.remove(filepath)
                
                return jsonify(results)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/results/<filename>', methods=['GET'])
        def get_result(filename):
            """Serve result images"""
            return send_from_directory(self.app.config['RESULTS_FOLDER'], filename)
        
        @self.app.route('/classes', methods=['GET'])
        def get_classes():
            """Get list of supported icon classes"""
            return jsonify({
                'num_classes': len(self.class_names),
                'classes': self.class_names
            })
    
    def run(self, host: str = None, port: int = None, debug: bool = None):
        """
        Run the Flask application
        
        Args:
            host: Host address (optional, uses config if not provided)
            port: Port number (optional, uses config if not provided)
            debug: Debug mode (optional, uses config if not provided)
        """
        host = host or self.webapp_config['host']
        port = port or self.webapp_config['port']
        debug = debug if debug is not None else self.webapp_config['debug']
        
        print("\n" + "="*60)
        print("Multi-Modal Icon Vision API Server")
        print("="*60)
        print(f"Host: {host}")
        print(f"Port: {port}")
        print(f"Debug: {debug}")
        print(f"Model loaded: {self.model is not None}")
        print("="*60 + "\n")
        
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Main execution function"""
    # Initialize API
    api = IconDetectionAPI()
    
    # Load model
    model_path = "models/best_icon_detector.pt"
    try:
        api.load_model(model_path)
    except Exception as e:
        print(f"Warning: Could not load trained model: {e}")
        print("Starting with pretrained YOLOv8n")
        api.load_model("yolov8n.pt")
    
    # Run server
    api.run()


if __name__ == "__main__":
    main()
