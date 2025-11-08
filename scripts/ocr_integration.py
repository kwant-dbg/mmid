"""
OCR Integration Module - Phase 2 Implementation
Extracts text from mobile UI screenshots using EasyOCR and Tesseract
Implements text-icon correlation and semantic mapping
"""

import cv2
import numpy as np
import easyocr
import pytesseract
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextDetection:
    """Represents detected text in UI"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    language: str = 'en'
    
    def to_dict(self):
        return asdict(self)


@dataclass
class IconTextPair:
    """Represents icon-text correlation"""
    icon_class: str
    icon_bbox: Tuple[float, float, float, float]
    icon_confidence: float
    nearby_text: List[str]
    text_positions: List[Tuple[int, int, int, int]]
    spatial_relationship: str  # 'left', 'right', 'above', 'below', 'overlap'
    semantic_score: float


class OCREngine:
    """Unified OCR interface supporting multiple engines"""
    
    def __init__(self, engine: str = 'easyocr', languages: List[str] = ['en']):
        """
        Initialize OCR engine
        
        Args:
            engine: 'easyocr' or 'tesseract'
            languages: List of language codes
        """
        self.engine = engine
        self.languages = languages
        
        if engine == 'easyocr':
            logger.info(f"Initializing EasyOCR with languages: {languages}")
            self.reader = easyocr.Reader(languages, gpu=True)
        elif engine == 'tesseract':
            logger.info("Using Tesseract OCR")
            # Verify Tesseract installation
            try:
                pytesseract.get_tesseract_version()
            except Exception as e:
                logger.warning(f"Tesseract not found: {e}")
                logger.warning("Falling back to EasyOCR")
                self.engine = 'easyocr'
                self.reader = easyocr.Reader(languages, gpu=True)
        else:
            raise ValueError(f"Unsupported OCR engine: {engine}")
    
    def extract_text(self, image: np.ndarray, preprocess: bool = True) -> List[TextDetection]:
        """
        Extract text from image
        
        Args:
            image: Input image (BGR format)
            preprocess: Apply preprocessing for better OCR
            
        Returns:
            List of TextDetection objects
        """
        if preprocess:
            image = self._preprocess_image(image)
        
        if self.engine == 'easyocr':
            return self._extract_easyocr(image)
        else:
            return self._extract_tesseract(image)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        # Convert back to BGR for EasyOCR
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    def _extract_easyocr(self, image: np.ndarray) -> List[TextDetection]:
        """Extract text using EasyOCR"""
        results = self.reader.readtext(image)
        
        detections = []
        for bbox, text, conf in results:
            # Convert bbox to x1, y1, x2, y2 format
            points = np.array(bbox, dtype=np.int32)
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            
            detection = TextDetection(
                text=text.strip(),
                confidence=float(conf),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                language=self.languages[0]
            )
            detections.append(detection)
        
        return detections
    
    def _extract_tesseract(self, image: np.ndarray) -> List[TextDetection]:
        """Extract text using Tesseract OCR"""
        # Get detailed data
        data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT,
            lang='+'.join(self.languages)
        )
        
        detections = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            
            # Skip low confidence and empty text
            if conf < 0 or not text:
                continue
            
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            detection = TextDetection(
                text=text,
                confidence=conf / 100.0,  # Normalize to 0-1
                bbox=(x, y, x + w, y + h),
                language=self.languages[0]
            )
            detections.append(detection)
        
        return detections


class IconTextCorrelator:
    """Correlates icon detections with nearby text"""
    
    def __init__(self, proximity_threshold: int = 100):
        """
        Initialize correlator
        
        Args:
            proximity_threshold: Max pixel distance for icon-text correlation
        """
        self.proximity_threshold = proximity_threshold
        
    def correlate(
        self, 
        icon_detections: List[Dict],
        text_detections: List[TextDetection]
    ) -> List[IconTextPair]:
        """
        Correlate icons with nearby text
        
        Args:
            icon_detections: List of icon detection dicts with bbox, class, conf
            text_detections: List of TextDetection objects
            
        Returns:
            List of IconTextPair objects
        """
        pairs = []
        
        for icon in icon_detections:
            icon_bbox = icon['bbox']  # x1, y1, x2, y2 (normalized 0-1)
            icon_class = icon['class']
            icon_conf = icon['confidence']
            
            # Find nearby text
            nearby_texts = []
            text_positions = []
            relationships = []
            
            for text_det in text_detections:
                relationship = self._compute_spatial_relationship(
                    icon_bbox, text_det.bbox
                )
                
                if relationship:  # Text is nearby
                    nearby_texts.append(text_det.text)
                    text_positions.append(text_det.bbox)
                    relationships.append(relationship)
            
            # Compute semantic score
            semantic_score = self._compute_semantic_score(
                icon_class, nearby_texts
            )
            
            # Determine dominant spatial relationship
            dominant_relationship = max(
                set(relationships), key=relationships.count
            ) if relationships else 'none'
            
            pair = IconTextPair(
                icon_class=icon_class,
                icon_bbox=icon_bbox,
                icon_confidence=icon_conf,
                nearby_text=nearby_texts,
                text_positions=text_positions,
                spatial_relationship=dominant_relationship,
                semantic_score=semantic_score
            )
            pairs.append(pair)
        
        return pairs
    
    def _compute_spatial_relationship(
        self,
        icon_bbox: Tuple[float, float, float, float],
        text_bbox: Tuple[int, int, int, int],
        image_width: int = 1440,
        image_height: int = 2560
    ) -> Optional[str]:
        """
        Compute spatial relationship between icon and text
        
        Args:
            icon_bbox: Icon bbox (normalized)
            text_bbox: Text bbox (pixel coordinates)
            image_width: Image width for denormalization
            image_height: Image height for denormalization
            
        Returns:
            Relationship string or None if too far
        """
        # Denormalize icon bbox
        ix1 = icon_bbox[0] * image_width
        iy1 = icon_bbox[1] * image_height
        ix2 = icon_bbox[2] * image_width
        iy2 = icon_bbox[3] * image_height
        
        tx1, ty1, tx2, ty2 = text_bbox
        
        # Compute centers
        icon_center = ((ix1 + ix2) / 2, (iy1 + iy2) / 2)
        text_center = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)
        
        # Compute distance
        distance = np.sqrt(
            (icon_center[0] - text_center[0]) ** 2 +
            (icon_center[1] - text_center[1]) ** 2
        )
        
        if distance > self.proximity_threshold:
            return None
        
        # Determine relationship
        dx = text_center[0] - icon_center[0]
        dy = text_center[1] - icon_center[1]
        
        # Check for overlap
        if (ix1 <= tx2 and ix2 >= tx1 and iy1 <= ty2 and iy2 >= ty1):
            return 'overlap'
        
        # Determine primary direction
        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'below' if dy > 0 else 'above'
    
    def _compute_semantic_score(
        self,
        icon_class: str,
        nearby_texts: List[str]
    ) -> float:
        """
        Compute semantic relevance score between icon and text
        
        Args:
            icon_class: Icon class name
            nearby_texts: List of nearby text strings
            
        Returns:
            Semantic score (0-1)
        """
        if not nearby_texts:
            return 0.0
        
        # Semantic mapping: icon class -> expected keywords
        semantic_map = {
            'search_icon': ['search', 'find', 'lookup', 'query'],
            'back_button': ['back', 'return', 'previous'],
            'menu_icon': ['menu', 'options', 'more'],
            'home_icon': ['home', 'main', 'dashboard'],
            'settings_icon': ['settings', 'preferences', 'config'],
            'profile_icon': ['profile', 'account', 'user'],
            'cart_icon': ['cart', 'basket', 'shopping'],
            'heart_icon': ['favorite', 'like', 'love', 'wishlist'],
            'share_icon': ['share', 'send', 'forward'],
            'delete_icon': ['delete', 'remove', 'trash'],
            'edit_icon': ['edit', 'modify', 'change'],
            'add_icon': ['add', 'new', 'create', 'plus'],
            'close_icon': ['close', 'exit', 'dismiss'],
            'filter_icon': ['filter', 'sort', 'refine'],
            'notification_icon': ['notification', 'alert', 'message'],
        }
        
        # Get expected keywords for icon
        expected_keywords = semantic_map.get(icon_class, [])
        if not expected_keywords:
            return 0.5  # Neutral score for unknown icons
        
        # Check for keyword matches
        combined_text = ' '.join(nearby_texts).lower()
        matches = sum(1 for kw in expected_keywords if kw in combined_text)
        
        # Normalize score
        score = min(matches / len(expected_keywords), 1.0)
        
        return score


class MultiModalUIAnalyzer:
    """Complete multi-modal UI analysis combining vision + text"""
    
    def __init__(
        self,
        ocr_engine: str = 'easyocr',
        proximity_threshold: int = 100
    ):
        """
        Initialize multi-modal analyzer
        
        Args:
            ocr_engine: OCR engine to use
            proximity_threshold: Proximity threshold for correlation
        """
        self.ocr = OCREngine(engine=ocr_engine)
        self.correlator = IconTextCorrelator(proximity_threshold)
        
    def analyze_ui(
        self,
        image: np.ndarray,
        icon_detections: List[Dict]
    ) -> Dict:
        """
        Perform complete multi-modal UI analysis
        
        Args:
            image: Input UI screenshot
            icon_detections: List of icon detections from YOLO
            
        Returns:
            Analysis results dictionary
        """
        logger.info("Extracting text from UI...")
        text_detections = self.ocr.extract_text(image)
        
        logger.info(f"Found {len(text_detections)} text regions")
        
        logger.info("Correlating icons with text...")
        icon_text_pairs = self.correlator.correlate(
            icon_detections, text_detections
        )
        
        # Generate UI structure
        ui_structure = self._generate_ui_structure(
            icon_text_pairs, text_detections
        )
        
        # Compute analysis metrics
        metrics = self._compute_metrics(
            icon_detections, text_detections, icon_text_pairs
        )
        
        return {
            'text_detections': [det.to_dict() for det in text_detections],
            'icon_text_pairs': [asdict(pair) for pair in icon_text_pairs],
            'ui_structure': ui_structure,
            'metrics': metrics
        }
    
    def _generate_ui_structure(
        self,
        icon_text_pairs: List[IconTextPair],
        text_detections: List[TextDetection]
    ) -> Dict:
        """Generate hierarchical UI structure"""
        structure = {
            'navigation': [],
            'content': [],
            'actions': [],
            'information': []
        }
        
        # Categorize by icon type
        navigation_icons = {'back_button', 'menu_icon', 'home_icon', 'tab_icon'}
        action_icons = {'search_icon', 'add_icon', 'delete_icon', 'edit_icon', 'share_icon'}
        
        for pair in icon_text_pairs:
            element = {
                'type': pair.icon_class,
                'text': ' '.join(pair.nearby_text),
                'confidence': pair.icon_confidence,
                'semantic_score': pair.semantic_score
            }
            
            if pair.icon_class in navigation_icons:
                structure['navigation'].append(element)
            elif pair.icon_class in action_icons:
                structure['actions'].append(element)
            else:
                structure['content'].append(element)
        
        return structure
    
    def _compute_metrics(
        self,
        icon_detections: List[Dict],
        text_detections: List[TextDetection],
        icon_text_pairs: List[IconTextPair]
    ) -> Dict:
        """Compute analysis metrics"""
        # Count icons with correlated text
        icons_with_text = sum(1 for pair in icon_text_pairs if pair.nearby_text)
        
        # Average semantic score
        avg_semantic = np.mean([
            pair.semantic_score for pair in icon_text_pairs
        ]) if icon_text_pairs else 0.0
        
        # Spatial distribution
        relationships = [pair.spatial_relationship for pair in icon_text_pairs]
        relationship_dist = {
            rel: relationships.count(rel) / len(relationships) if relationships else 0
            for rel in ['left', 'right', 'above', 'below', 'overlap', 'none']
        }
        
        return {
            'total_icons': len(icon_detections),
            'total_text_regions': len(text_detections),
            'icons_with_text': icons_with_text,
            'icon_text_ratio': icons_with_text / len(icon_detections) if icon_detections else 0,
            'avg_semantic_score': float(avg_semantic),
            'spatial_distribution': relationship_dist
        }


def visualize_multimodal_results(
    image: np.ndarray,
    text_detections: List[TextDetection],
    icon_text_pairs: List[IconTextPair],
    output_path: str
):
    """
    Visualize multi-modal analysis results
    
    Args:
        image: Original image
        text_detections: Text detections
        icon_text_pairs: Icon-text correlations
        output_path: Path to save visualization
    """
    vis_image = image.copy()
    
    # Draw text bounding boxes (green)
    for text_det in text_detections:
        x1, y1, x2, y2 = text_det.bbox
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis_image, text_det.text[:20], (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
    
    # Draw icon-text correlations (blue lines)
    h, w = image.shape[:2]
    for pair in icon_text_pairs:
        if not pair.nearby_text:
            continue
        
        # Icon center
        ix1, iy1, ix2, iy2 = pair.icon_bbox
        icon_center = (int((ix1 + ix2) * w / 2), int((iy1 + iy2) * h / 2))
        
        # Draw lines to correlated text
        for text_bbox in pair.text_positions:
            tx1, ty1, tx2, ty2 = text_bbox
            text_center = ((tx1 + tx2) // 2, (ty1 + ty2) // 2)
            cv2.line(vis_image, icon_center, text_center, (255, 0, 0), 2)
    
    # Save visualization
    cv2.imwrite(output_path, vis_image)
    logger.info(f"Visualization saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Test with sample image
    analyzer = MultiModalUIAnalyzer(ocr_engine='easyocr')
    
    # Load test image
    test_image_path = "data/test/sample_ui.png"
    if Path(test_image_path).exists():
        image = cv2.imread(test_image_path)
        
        # Mock icon detections for testing
        icon_detections = [
            {
                'class': 'search_icon',
                'confidence': 0.95,
                'bbox': (0.1, 0.05, 0.15, 0.08)
            },
            {
                'class': 'back_button',
                'confidence': 0.92,
                'bbox': (0.02, 0.05, 0.06, 0.08)
            }
        ]
        
        # Analyze
        results = analyzer.analyze_ui(image, icon_detections)
        
        # Print results
        print(json.dumps(results, indent=2))
        
        # Visualize
        visualize_multimodal_results(
            image,
            [TextDetection(**det) for det in results['text_detections']],
            [IconTextPair(**pair) for pair in results['icon_text_pairs']],
            "output/multimodal_visualization.png"
        )
    else:
        logger.warning("Test image not found. Skipping example.")
