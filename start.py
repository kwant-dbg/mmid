"""
Simple starter script for the Multi-Modal Icon Vision System
Handles missing models and provides helpful guidance
"""

import os
import sys
from pathlib import Path


def check_requirements():
    """Check if all requirements are installed"""
    required_packages = [
        'flask',
        'torch',
        'ultralytics',
        'cv2',
        'yaml'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'yaml':
                __import__('yaml')
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nRun: pip install -r requirements.txt")
        print("Or:  python setup.bat  (Windows)")
        print("Or:  bash setup.sh     (Linux/Mac)")
        return False
    
    return True


def check_model():
    """Check if trained model exists"""
    model_path = Path("models/best_icon_detector.pt")
    
    if model_path.exists():
        print(f"✅ Found trained model: {model_path}")
        return str(model_path)
    else:
        print("⚠️  Trained model not found at models/best_icon_detector.pt")
        print("\nOptions:")
        print("  1. Train a model: python scripts/train_model.py")
        print("  2. Use pretrained YOLOv8n (will download automatically)")
        print("\nContinuing with pretrained YOLOv8n for demo...")
        return None


def check_dataset():
    """Check if dataset is available"""
    data_yaml = Path("data/data.yaml")
    
    if data_yaml.exists():
        print(f"✅ Found dataset configuration: {data_yaml}")
        return True
    else:
        print("⚠️  Dataset not found at data/data.yaml")
        print("\nTo prepare dataset:")
        print("  1. Download Rico dataset")
        print("  2. Place in data/raw/")
        print("  3. Run: python scripts/dataset_processor.py")
        print("\nYou can still use the demo with sample images!")
        return False


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*60)
    print("🔍 Multi-Modal Icon Vision System")
    print("="*60)
    print("AI-Powered Mobile UI Icon Detection")
    print("\nTeam: Harshit Sharma, Sushant Thakur, Kamal")
    print("TIET Patiala • 2025")
    print("="*60 + "\n")


def print_usage():
    """Print usage instructions"""
    print("\n" + "="*60)
    print("📝 Usage Instructions")
    print("="*60)
    print("\n1. Open your browser and visit:")
    print("   http://localhost:8000")
    print("\n2. Upload a mobile screenshot")
    print("\n3. Adjust detection settings (optional)")
    print("\n4. Click 'Detect Icons'")
    print("\n5. View and download results")
    print("\n" + "="*60)
    print("\n💡 Tip: To serve the frontend, run in a new terminal:")
    print("   cd frontend")
    print("   python -m http.server 8000")
    print("="*60 + "\n")


def main():
    """Main startup function"""
    print_banner()
    
    print("🔍 Checking system requirements...\n")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    print("✅ All required packages installed\n")
    
    # Check for model
    print("🤖 Checking for trained model...\n")
    model_path = check_model()
    print()
    
    # Check for dataset
    print("📊 Checking for dataset...\n")
    check_dataset()
    print()
    
    # Start backend
    print("🚀 Starting backend server...")
    print("="*60 + "\n")
    
    try:
        # Import and run the API
        sys.path.insert(0, os.path.dirname(__file__))
        from backend.app import IconDetectionAPI
        
        api = IconDetectionAPI()
        
        # Load model
        if model_path:
            api.load_model(model_path)
        else:
            print("Loading pretrained YOLOv8n model (first run may take a moment)...")
            api.load_model("yolov8n.pt")
        
        print_usage()
        
        # Run server
        api.run()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        print("Goodbye! 👋\n")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("\nPlease check:")
        print("  1. All dependencies are installed")
        print("  2. Port 5000 is available")
        print("  3. Configuration file is correct")
        sys.exit(1)


if __name__ == "__main__":
    main()
