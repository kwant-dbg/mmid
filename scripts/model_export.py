"""
Model Export Utilities
Exports trained models to various formats for deployment
Supports: ONNX, TensorRT, OpenVINO, CoreML, TFLite
"""

import torch
import onnx
import onnxruntime
from ultralytics import YOLO
from pathlib import Path
import logging
import json
import numpy as np
from typing import Dict, Optional, List
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExporter:
    """Export trained models to deployment formats"""
    
    def __init__(self, model_path: str, config_path: str = "config/config.yaml"):
        """
        Initialize model exporter
        
        Args:
            model_path: Path to trained model weights
            config_path: Path to configuration file
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = YOLO(str(model_path))
        
        # Create exports directory
        self.exports_dir = Path("exports")
        self.exports_dir.mkdir(exist_ok=True)
        
    def export_onnx(
        self,
        dynamic: bool = True,
        simplify: bool = True,
        opset: int = 12
    ) -> Path:
        """
        Export model to ONNX format
        
        Args:
            dynamic: Use dynamic input shapes
            simplify: Simplify ONNX graph
            opset: ONNX opset version
            
        Returns:
            Path to exported ONNX model
        """
        logger.info("Exporting to ONNX format...")
        
        output_path = self.exports_dir / f"{self.model_path.stem}.onnx"
        
        # Export using Ultralytics
        self.model.export(
            format='onnx',
            dynamic=dynamic,
            simplify=simplify,
            opset=opset
        )
        
        # Verify ONNX model
        onnx_path = self.model_path.with_suffix('.onnx')
        self._verify_onnx(onnx_path)
        
        # Move to exports directory
        onnx_path.rename(output_path)
        
        logger.info(f"✅ ONNX model exported to {output_path}")
        return output_path
    
    def export_tensorrt(
        self,
        workspace: int = 4,
        half: bool = True,
        int8: bool = False
    ) -> Path:
        """
        Export model to TensorRT format (NVIDIA GPUs)
        
        Args:
            workspace: Maximum workspace size in GB
            half: Use FP16 precision
            int8: Use INT8 quantization
            
        Returns:
            Path to exported TensorRT engine
        """
        logger.info("Exporting to TensorRT format...")
        
        output_path = self.exports_dir / f"{self.model_path.stem}.engine"
        
        try:
            # Export using Ultralytics
            self.model.export(
                format='engine',
                workspace=workspace,
                half=half,
                int8=int8
            )
            
            # Move to exports directory
            engine_path = self.model_path.with_suffix('.engine')
            engine_path.rename(output_path)
            
            logger.info(f"✅ TensorRT engine exported to {output_path}")
            logger.info(f"   Precision: {'FP16' if half else 'FP32'}")
            logger.info(f"   INT8: {int8}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ TensorRT export failed: {e}")
            logger.info("TensorRT requires NVIDIA GPU and TensorRT installation")
            return None
    
    def export_openvino(self, half: bool = False) -> Path:
        """
        Export model to OpenVINO format (Intel CPUs/GPUs)
        
        Args:
            half: Use FP16 precision
            
        Returns:
            Path to exported OpenVINO model directory
        """
        logger.info("Exporting to OpenVINO format...")
        
        output_dir = self.exports_dir / f"{self.model_path.stem}_openvino_model"
        
        try:
            # Export using Ultralytics
            self.model.export(
                format='openvino',
                half=half
            )
            
            # Move to exports directory
            openvino_path = self.model_path.with_suffix('_openvino_model')
            if openvino_path.exists():
                openvino_path.rename(output_dir)
            
            logger.info(f"✅ OpenVINO model exported to {output_dir}")
            logger.info(f"   Precision: {'FP16' if half else 'FP32'}")
            
            return output_dir
            
        except Exception as e:
            logger.error(f"❌ OpenVINO export failed: {e}")
            return None
    
    def export_coreml(self, nms: bool = True) -> Path:
        """
        Export model to CoreML format (iOS/macOS)
        
        Args:
            nms: Include NMS layer
            
        Returns:
            Path to exported CoreML model
        """
        logger.info("Exporting to CoreML format...")
        
        output_path = self.exports_dir / f"{self.model_path.stem}.mlmodel"
        
        try:
            # Export using Ultralytics
            self.model.export(
                format='coreml',
                nms=nms
            )
            
            # Move to exports directory
            coreml_path = self.model_path.with_suffix('.mlmodel')
            coreml_path.rename(output_path)
            
            logger.info(f"✅ CoreML model exported to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ CoreML export failed: {e}")
            logger.info("CoreML export requires macOS")
            return None
    
    def export_tflite(
        self,
        int8: bool = False,
        data: Optional[str] = None
    ) -> Path:
        """
        Export model to TensorFlow Lite format (Mobile/Edge)
        
        Args:
            int8: Use INT8 quantization
            data: Path to calibration data for INT8
            
        Returns:
            Path to exported TFLite model
        """
        logger.info("Exporting to TensorFlow Lite format...")
        
        output_path = self.exports_dir / f"{self.model_path.stem}.tflite"
        
        try:
            # Export using Ultralytics
            self.model.export(
                format='tflite',
                int8=int8,
                data=data
            )
            
            # Move to exports directory
            tflite_path = self.model_path.with_suffix('.tflite')
            tflite_path.rename(output_path)
            
            logger.info(f"✅ TFLite model exported to {output_path}")
            logger.info(f"   Quantization: {'INT8' if int8 else 'FP32'}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ TFLite export failed: {e}")
            return None
    
    def export_all(self) -> Dict[str, Path]:
        """
        Export model to all supported formats
        
        Returns:
            Dictionary mapping format name to export path
        """
        logger.info("Exporting to all formats...")
        
        exports = {}
        
        # ONNX (universal)
        try:
            exports['onnx'] = self.export_onnx()
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
        
        # TensorRT (NVIDIA)
        try:
            trt_path = self.export_tensorrt()
            if trt_path:
                exports['tensorrt'] = trt_path
        except Exception as e:
            logger.error(f"TensorRT export failed: {e}")
        
        # OpenVINO (Intel)
        try:
            ov_path = self.export_openvino()
            if ov_path:
                exports['openvino'] = ov_path
        except Exception as e:
            logger.error(f"OpenVINO export failed: {e}")
        
        # CoreML (Apple)
        try:
            coreml_path = self.export_coreml()
            if coreml_path:
                exports['coreml'] = coreml_path
        except Exception as e:
            logger.error(f"CoreML export failed: {e}")
        
        # TFLite (Mobile/Edge)
        try:
            tflite_path = self.export_tflite()
            if tflite_path:
                exports['tflite'] = tflite_path
        except Exception as e:
            logger.error(f"TFLite export failed: {e}")
        
        # Save export manifest
        self._save_export_manifest(exports)
        
        logger.info(f"\n✅ Exported {len(exports)} formats successfully")
        return exports
    
    def _verify_onnx(self, onnx_path: Path):
        """Verify ONNX model validity"""
        try:
            # Load and check ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # Test inference with ONNX Runtime
            session = onnxruntime.InferenceSession(
                str(onnx_path),
                providers=['CPUExecutionProvider']
            )
            
            # Get input shape
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            
            # Create dummy input
            dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
            
            # Run inference
            outputs = session.run(None, {input_name: dummy_input})
            
            logger.info("✅ ONNX model verification passed")
            logger.info(f"   Input shape: {input_shape}")
            logger.info(f"   Output shapes: {[out.shape for out in outputs]}")
            
        except Exception as e:
            logger.warning(f"⚠️ ONNX verification failed: {e}")
    
    def _save_export_manifest(self, exports: Dict[str, Path]):
        """Save export manifest with metadata"""
        manifest = {
            'model_name': self.model_path.stem,
            'original_model': str(self.model_path),
            'input_size': self.config['model']['input_size'],
            'num_classes': len(self.config['dataset']['classes']),
            'classes': self.config['dataset']['classes'],
            'exports': {
                fmt: str(path) for fmt, path in exports.items()
            },
            'export_date': str(Path.ctime(self.model_path))
        }
        
        manifest_path = self.exports_dir / "export_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Export manifest saved to {manifest_path}")


class ModelBenchmarker:
    """Benchmark exported models for deployment"""
    
    def __init__(self, exports_dir: Path = Path("exports")):
        """
        Initialize benchmarker
        
        Args:
            exports_dir: Directory containing exported models
        """
        self.exports_dir = Path(exports_dir)
        
    def benchmark_onnx(self, onnx_path: Path, num_runs: int = 100) -> Dict:
        """
        Benchmark ONNX model performance
        
        Args:
            onnx_path: Path to ONNX model
            num_runs: Number of inference runs
            
        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Benchmarking ONNX model: {onnx_path}")
        
        # Load ONNX model
        session = onnxruntime.InferenceSession(
            str(onnx_path),
            providers=['CPUExecutionProvider']
        )
        
        # Prepare dummy input
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            session.run(None, {input_name: dummy_input})
        
        # Benchmark
        import time
        times = []
        for _ in range(num_runs):
            start = time.time()
            session.run(None, {input_name: dummy_input})
            times.append(time.time() - start)
        
        results = {
            'format': 'ONNX',
            'avg_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'fps': 1.0 / np.mean(times)
        }
        
        logger.info(f"   Avg inference time: {results['avg_time_ms']:.2f} ms")
        logger.info(f"   FPS: {results['fps']:.2f}")
        
        return results
    
    def compare_exports(self) -> Dict:
        """
        Compare performance across all exported formats
        
        Returns:
            Comparison results dictionary
        """
        results = {}
        
        # Benchmark ONNX if available
        onnx_files = list(self.exports_dir.glob("*.onnx"))
        if onnx_files:
            results['onnx'] = self.benchmark_onnx(onnx_files[0])
        
        # TODO: Add benchmarks for other formats
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export trained model to deployment formats")
    parser.add_argument('--model', type=str, default="models/best.pt",
                       help='Path to trained model weights')
    parser.add_argument('--format', type=str, default='all',
                       choices=['onnx', 'tensorrt', 'openvino', 'coreml', 'tflite', 'all'],
                       help='Export format')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark exported models')
    
    args = parser.parse_args()
    
    # Initialize exporter
    exporter = ModelExporter(args.model)
    
    # Export based on format
    if args.format == 'all':
        exports = exporter.export_all()
    elif args.format == 'onnx':
        exports = {'onnx': exporter.export_onnx()}
    elif args.format == 'tensorrt':
        exports = {'tensorrt': exporter.export_tensorrt()}
    elif args.format == 'openvino':
        exports = {'openvino': exporter.export_openvino()}
    elif args.format == 'coreml':
        exports = {'coreml': exporter.export_coreml()}
    elif args.format == 'tflite':
        exports = {'tflite': exporter.export_tflite()}
    
    # Benchmark if requested
    if args.benchmark:
        benchmarker = ModelBenchmarker()
        results = benchmarker.compare_exports()
        
        print("\n" + "="*50)
        print("BENCHMARK RESULTS")
        print("="*50)
        for fmt, res in results.items():
            print(f"\n{fmt.upper()}:")
            for key, value in res.items():
                print(f"  {key}: {value}")
