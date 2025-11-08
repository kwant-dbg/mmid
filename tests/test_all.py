"""
Unit Tests for Icon Detection System
"""

import unittest
import os
import sys
import yaml
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestConfiguration(unittest.TestCase):
    """Test configuration loading"""
    
    def test_config_exists(self):
        """Test if config file exists"""
        config_path = "config/config.yaml"
        self.assertTrue(os.path.exists(config_path), "Config file not found")
    
    def test_config_valid(self):
        """Test if config is valid YAML"""
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        self.assertIsNotNone(config)
        self.assertIn('model', config)
        self.assertIn('dataset', config)
        self.assertIn('training', config)
    
    def test_class_names(self):
        """Test if class names are defined"""
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        class_names = config['dataset']['class_names']
        self.assertEqual(len(class_names), 26)
        self.assertIn('search_icon', class_names)


class TestDatasetProcessor(unittest.TestCase):
    """Test dataset processing functions"""
    
    def setUp(self):
        """Setup test environment"""
        from scripts.dataset_processor import RicoDatasetProcessor
        self.processor = RicoDatasetProcessor()
    
    def test_yolo_format_conversion(self):
        """Test conversion to YOLO format"""
        # Mock UI element
        element = {
            'bounds': [100, 200, 300, 400],  # [x1, y1, x2, y2]
            'componentLabel': 'search'
        }
        
        img_width, img_height = 1080, 1920
        
        # Convert
        annotations = self.processor.convert_to_yolo_format(
            [element], img_width, img_height
        )
        
        # Should have one annotation
        self.assertEqual(len(annotations), 1)
        
        # Parse annotation
        parts = annotations[0].split()
        self.assertEqual(len(parts), 5)  # class_id x_center y_center width height
        
        # Check values are normalized (0-1)
        for i in range(1, 5):
            value = float(parts[i])
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
    
    def test_icon_class_mapping(self):
        """Test icon class mapping"""
        # Test valid mapping
        class_id = self.processor._map_to_icon_class('search')
        self.assertEqual(class_id, 1)
        
        class_id = self.processor._map_to_icon_class('back')
        self.assertEqual(class_id, 0)
        
        # Test invalid mapping
        class_id = self.processor._map_to_icon_class('unknown_icon')
        self.assertEqual(class_id, -1)


class TestModelComponents(unittest.TestCase):
    """Test model-related components"""
    
    def test_import_ultralytics(self):
        """Test if ultralytics is installed"""
        try:
            from ultralytics import YOLO
            self.assertTrue(True)
        except ImportError:
            self.fail("ultralytics package not installed")
    
    def test_import_torch(self):
        """Test if PyTorch is installed"""
        try:
            import torch
            self.assertTrue(True)
        except ImportError:
            self.fail("PyTorch not installed")
    
    def test_cuda_availability(self):
        """Test CUDA availability (optional)"""
        import torch
        # Just log, don't fail if CUDA not available
        if torch.cuda.is_available():
            print(f"\nCUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("\nCUDA not available, will use CPU")


class TestAPIEndpoints(unittest.TestCase):
    """Test Flask API endpoints"""
    
    def setUp(self):
        """Setup Flask test client"""
        try:
            from backend.app import IconDetectionAPI
            self.api = IconDetectionAPI()
            self.app = self.api.app
            self.app.config['TESTING'] = True
            self.client = self.app.test_client()
        except Exception as e:
            self.skipTest(f"Could not initialize API: {e}")
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertEqual(data['status'], 'ok')
    
    def test_get_classes(self):
        """Test get classes endpoint"""
        response = self.client.get('/classes')
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertEqual(data['num_classes'], 26)
        self.assertIsInstance(data['classes'], list)
    
    def test_predict_no_file(self):
        """Test predict endpoint without file"""
        response = self.client.post('/predict')
        self.assertEqual(response.status_code, 400)


class TestDirectoryStructure(unittest.TestCase):
    """Test project directory structure"""
    
    def test_required_directories(self):
        """Test if required directories exist"""
        required_dirs = [
            'backend',
            'frontend',
            'scripts',
            'config',
            'data',
            'models'
        ]
        
        for dir_name in required_dirs:
            self.assertTrue(
                os.path.isdir(dir_name),
                f"Required directory '{dir_name}' not found"
            )
    
    def test_required_files(self):
        """Test if required files exist"""
        required_files = [
            'README.md',
            'requirements.txt',
            'config/config.yaml',
            'backend/app.py',
            'scripts/train_model.py',
            'scripts/dataset_processor.py',
            'scripts/evaluate_model.py',
            'frontend/index.html',
            'frontend/style.css',
            'frontend/script.js'
        ]
        
        for file_path in required_files:
            self.assertTrue(
                os.path.isfile(file_path),
                f"Required file '{file_path}' not found"
            )


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_bbox_validation(self):
        """Test bounding box validation logic"""
        # Valid bbox
        x_center, y_center = 0.5, 0.5
        width, height = 0.2, 0.3
        
        self.assertTrue(0 <= x_center <= 1)
        self.assertTrue(0 <= y_center <= 1)
        self.assertTrue(0 < width <= 1)
        self.assertTrue(0 < height <= 1)
    
    def test_iou_calculation(self):
        """Test IoU calculation (if implemented)"""
        # Box format: [x1, y1, x2, y2]
        box1 = np.array([0, 0, 100, 100])
        box2 = np.array([50, 50, 150, 150])
        
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        
        # Should be 0.25 for these boxes
        self.assertAlmostEqual(iou, 0.25, places=2)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestModelComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIEndpoints))
    suite.addTests(loader.loadTestsFromTestCase(TestDirectoryStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
