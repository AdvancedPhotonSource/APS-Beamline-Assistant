from typing import Any, Dict, List, Optional
import json
import sys
import os
from pathlib import Path
import numpy as np
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server for ML models
mcp = FastMCP("ml-beamline-models")

def format_result(result: dict) -> str:
    """Format results into JSON string"""
    return json.dumps(result, indent=2)

# Import ML/DL libraries with error handling
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    PYTORCH_AVAILABLE = True
    print("PyTorch available", file=sys.stderr)
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available", file=sys.stderr)

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow available", file=sys.stderr)
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available", file=sys.stderr)

try:
    from PIL import Image
    import cv2
    IMAGE_LIBS_AVAILABLE = True
except ImportError:
    IMAGE_LIBS_AVAILABLE = False

try:
    from scipy import ndimage
    import skimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class UNetModel:
    """Wrapper for U-Net segmentation model"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        
    def load(self):
        """Load trained U-Net model"""
        if PYTORCH_AVAILABLE:
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
            return True
        return False
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference on image"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Preprocess
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Postprocess
        prediction = output.squeeze().cpu().numpy()
        return prediction


class XRDPhasePredictor:
    """ML model for XRD phase prediction from patterns"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        
    def load(self):
        """Load trained phase prediction model"""
        try:
            if PYTORCH_AVAILABLE:
                checkpoint = torch.load(self.model_path)
                self.model = checkpoint.get('model')
                self.scaler = checkpoint.get('scaler')
                self.model.eval()
                return True
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
        return False
    
    def predict(self, pattern: np.ndarray) -> Dict[str, float]:
        """Predict phases from 1D XRD pattern"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Normalize pattern
        if self.scaler:
            pattern_scaled = self.scaler.transform(pattern.reshape(1, -1))
        else:
            pattern_scaled = (pattern - pattern.mean()) / pattern.std()
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(pattern_scaled)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
        
        # Map to phase names
        phase_names = ["austenite", "ferrite", "martensite", "cementite", "other"]
        predictions = {
            name: float(prob) 
            for name, prob in zip(phase_names, probabilities[0].numpy())
        }
        
        return predictions


@mcp.tool()
async def segment_xct_image(
    image_path: str,
    model_path: str,
    output_path: str = None,
    model_type: str = "unet",
    device: str = "cpu"
) -> str:
    """Segment XCT (X-ray computed tomography) images using trained U-Net model.
    
    This tool applies deep learning segmentation to identify features in XCT data,
    such as pores, cracks, phases, or grain boundaries.
    
    Args:
        image_path: Path to XCT image (TIFF, PNG, NPY)
        model_path: Path to trained model weights (.pth, .h5)
        output_path: Path to save segmentation result
        model_type: Model architecture ("unet", "unet++", "attention_unet")
        device: Computing device ("cpu", "cuda", "mps")
    """
    try:
        if not PYTORCH_AVAILABLE and not TENSORFLOW_AVAILABLE:
            return json.dumps({
                "error": "No ML framework available (PyTorch or TensorFlow required)",
                "status": "failed"
            }, indent=2)
        
        image_file = Path(image_path).expanduser()
        model_file = Path(model_path).expanduser()
        
        if not image_file.exists():
            return json.dumps({
                "error": f"Image not found: {image_file}",
                "status": "failed"
            }, indent=2)
        
        if not model_file.exists():
            return json.dumps({
                "error": f"Model not found: {model_file}",
                "status": "failed"
            }, indent=2)
        
        # Load image
        if image_path.endswith('.npy'):
            image = np.load(image_file)
        else:
            from PIL import Image as PILImage
            image = np.array(PILImage.open(image_file))
        
        print(f"Loaded image: {image.shape}", file=sys.stderr)
        
        # Load and run model
        model = UNetModel(model_file, device)
        if not model.load():
            return json.dumps({
                "error": "Failed to load model",
                "status": "failed"
            }, indent=2)
        
        print("Running segmentation...", file=sys.stderr)
        segmentation = model.predict(image)
        
        # Save output
        if output_path:
            output_file = Path(output_path).expanduser()
            np.save(output_file, segmentation)
        else:
            output_file = image_file.parent / f"{image_file.stem}_segmented.npy"
            np.save(output_file, segmentation)
        
        # Calculate metrics
        unique_labels = np.unique(segmentation)
        label_counts = {int(label): int(np.sum(segmentation == label)) 
                       for label in unique_labels}
        
        results = {
            "tool": "segment_xct_image",
            "status": "completed",
            "input_image": str(image_file),
            "output_segmentation": str(output_file),
            "model_type": model_type,
            "image_shape": list(image.shape),
            "segmentation_shape": list(segmentation.shape),
            "unique_labels": len(unique_labels),
            "label_distribution": label_counts,
            "device": device,
            "model_path": str(model_file)
        }
        
        return format_result(results)
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "status": "error"
        }, indent=2)


@mcp.tool()
async def predict_xrd_phases_ml(
    pattern_file: str,
    model_path: str,
    confidence_threshold: float = 0.1
) -> str:
    """Predict crystalline phases from XRD pattern using trained ML model.
    
    Uses a neural network trained on labeled XRD patterns to predict phase composition.
    Complements traditional crystallographic matching with data-driven predictions.
    
    Args:
        pattern_file: Path to 1D XRD pattern (.dat, .xy, .npy)
        model_path: Path to trained phase prediction model
        confidence_threshold: Minimum confidence to report phase (0-1)
    """
    try:
        if not PYTORCH_AVAILABLE:
            return json.dumps({
                "error": "PyTorch required for this model",
                "status": "failed"
            }, indent=2)
        
        pattern_path = Path(pattern_file).expanduser()
        model_file = Path(model_path).expanduser()
        
        if not pattern_path.exists():
            return json.dumps({
                "error": f"Pattern file not found: {pattern_path}",
                "status": "failed"
            }, indent=2)
        
        # Load pattern
        if pattern_file.endswith('.npy'):
            pattern = np.load(pattern_path)
        else:
            data = np.loadtxt(pattern_path)
            pattern = data[:, 1] if data.ndim > 1 else data
        
        print(f"Loaded pattern: {pattern.shape}", file=sys.stderr)
        
        # Load model
        predictor = XRDPhasePredictor(model_file)
        if not predictor.load():
            return json.dumps({
                "error": "Failed to load phase prediction model",
                "status": "failed"
            }, indent=2)
        
        # Predict phases
        print("Running phase prediction...", file=sys.stderr)
        predictions = predictor.predict(pattern)
        
        # Filter by confidence threshold
        detected_phases = {
            phase: conf 
            for phase, conf in predictions.items() 
            if conf >= confidence_threshold
        }
        
        # Sort by confidence
        sorted_phases = sorted(detected_phases.items(), 
                             key=lambda x: x[1], 
                             reverse=True)
        
        results = {
            "tool": "predict_xrd_phases_ml",
            "status": "completed",
            "input_pattern": str(pattern_path),
            "model_path": str(model_file),
            "confidence_threshold": confidence_threshold,
            "all_predictions": predictions,
            "detected_phases": [
                {
                    "phase": phase,
                    "confidence": float(conf),
                    "percentage": float(conf * 100)
                }
                for phase, conf in sorted_phases
            ],
            "primary_phase": sorted_phases[0][0] if sorted_phases else "unknown",
            "primary_confidence": sorted_phases[0][1] if sorted_phases else 0.0
        }
        
        return format_result(results)
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "status": "error"
        }, indent=2)


@mcp.tool()
async def denoise_diffraction_image(
    image_path: str,
    model_path: str,
    output_path: str = None,
    method: str = "deep_learning"
) -> str:
    """Denoise diffraction images using trained denoising model.
    
    Applies ML-based denoising to improve quality of diffraction patterns,
    especially useful for low-dose or fast acquisition data.
    
    Args:
        image_path: Path to noisy diffraction image
        model_path: Path to trained denoising model (can be "classical" for non-ML)
        output_path: Path to save denoised image
        method: Denoising method ("deep_learning", "classical", "wavelet")
    """
    try:
        image_file = Path(image_path).expanduser()
        
        if not image_file.exists():
            return json.dumps({
                "error": f"Image not found: {image_file}",
                "status": "failed"
            }, indent=2)
        
        # Load image
        if image_path.endswith('.npy'):
            image = np.load(image_file)
        else:
            from PIL import Image as PILImage
            image = np.array(PILImage.open(image_file))
        
        print(f"Loaded noisy image: {image.shape}", file=sys.stderr)
        
        if method == "deep_learning" and PYTORCH_AVAILABLE:
            # Load denoising model
            model_file = Path(model_path).expanduser()
            if not model_file.exists():
                return json.dumps({
                    "error": f"Model not found: {model_file}",
                    "status": "failed"
                }, indent=2)
            
            model = torch.load(model_file, map_location='cpu')
            model.eval()
            
            # Prepare input
            transform = transforms.ToTensor()
            input_tensor = transform(image).unsqueeze(0)
            
            # Denoise
            with torch.no_grad():
                denoised_tensor = model(input_tensor)
            
            denoised = denoised_tensor.squeeze().numpy()
            
        elif method == "classical" and SCIPY_AVAILABLE:
            # Classical denoising (non-ML)
            from scipy.ndimage import gaussian_filter, median_filter
            
            # Apply combination of filters
            denoised = gaussian_filter(image, sigma=1.0)
            denoised = median_filter(denoised, size=3)
            
        else:
            return json.dumps({
                "error": f"Method '{method}' not available with current libraries",
                "status": "failed"
            }, indent=2)
        
        # Calculate improvement metrics
        if SCIPY_AVAILABLE:
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity
            
            psnr = peak_signal_noise_ratio(image, denoised)
            ssim = structural_similarity(image, denoised)
        else:
            psnr = float(np.mean((image - denoised) ** 2))
            ssim = 0.0
        
        # Save output
        if output_path:
            output_file = Path(output_path).expanduser()
        else:
            output_file = image_file.parent / f"{image_file.stem}_denoised.npy"
        
        np.save(output_file, denoised)
        
        results = {
            "tool": "denoise_diffraction_image",
            "status": "completed",
            "input_image": str(image_file),
            "output_image": str(output_file),
            "method": method,
            "image_shape": list(image.shape),
            "quality_metrics": {
                "psnr": float(psnr),
                "ssim": float(ssim),
                "noise_reduction": float(np.std(image) - np.std(denoised))
            }
        }
        
        return format_result(results)
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "status": "error"
        }, indent=2)


@mcp.tool()
async def classify_diffraction_quality(
    image_path: str,
    model_path: str
) -> str:
    """Classify diffraction image quality using trained classifier.
    
    Automatically assess if a diffraction pattern is good/acceptable/poor
    for further analysis. Useful for automated data filtering.
    
    Args:
        image_path: Path to diffraction image
        model_path: Path to trained quality classifier model
    """
    try:
        if not PYTORCH_AVAILABLE:
            return json.dumps({
                "error": "PyTorch required",
                "status": "failed"
            }, indent=2)
        
        image_file = Path(image_path).expanduser()
        model_file = Path(model_path).expanduser()
        
        if not image_file.exists():
            return json.dumps({
                "error": f"Image not found: {image_file}",
                "status": "failed"
            }, indent=2)
        
        # Load image
        if image_path.endswith('.npy'):
            image = np.load(image_file)
        else:
            from PIL import Image as PILImage
            image = np.array(PILImage.open(image_file))
        
        # Load classifier
        model = torch.load(model_file, map_location='cpu')
        model.eval()
        
        # Preprocess
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        
        # Classify
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
        
        # Quality classes
        quality_labels = ["poor", "acceptable", "good", "excellent"]
        quality_probs = {
            label: float(prob)
            for label, prob in zip(quality_labels, probabilities[0].numpy())
        }
        
        predicted_quality = quality_labels[torch.argmax(probabilities).item()]
        confidence = float(torch.max(probabilities).item())
        
        results = {
            "tool": "classify_diffraction_quality",
            "status": "completed",
            "input_image": str(image_file),
            "predicted_quality": predicted_quality,
            "confidence": confidence,
            "quality_probabilities": quality_probs,
            "recommendation": (
                "Proceed with analysis" if predicted_quality in ["good", "excellent"]
                else "Consider re-acquisition or longer exposure"
            )
        }
        
        return format_result(results)
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "status": "error"
        }, indent=2)


@mcp.tool()
async def predict_material_properties(
    diffraction_pattern: str,
    model_path: str,
    properties: List[str] = None
) -> str:
    """Predict material properties from diffraction pattern using ML model.
    
    Predict properties like grain size, texture, residual stress, or hardness
    directly from diffraction data using trained regression models.
    
    Args:
        diffraction_pattern: Path to diffraction pattern file
        model_path: Path to trained property prediction model
        properties: List of properties to predict (e.g., ["grain_size", "hardness"])
    """
    try:
        if properties is None:
            properties = ["grain_size", "texture_index", "residual_stress"]
        
        pattern_file = Path(diffraction_pattern).expanduser()
        model_file = Path(model_path).expanduser()
        
        if not pattern_file.exists():
            return json.dumps({
                "error": f"Pattern not found: {pattern_file}",
                "status": "failed"
            }, indent=2)
        
        # Load pattern
        if diffraction_pattern.endswith('.npy'):
            pattern = np.load(pattern_file)
        else:
            data = np.loadtxt(pattern_file)
            pattern = data[:, 1] if data.ndim > 1 else data
        
        # Load regression model
        if PYTORCH_AVAILABLE:
            model = torch.load(model_file, map_location='cpu')
            model.eval()
            
            # Prepare input
            input_tensor = torch.FloatTensor(pattern).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                predictions = model(input_tensor)
            
            predicted_values = predictions.squeeze().numpy()
        else:
            return json.dumps({
                "error": "PyTorch required",
                "status": "failed"
            }, indent=2)
        
        # Map predictions to property names
        property_predictions = {
            prop: float(val)
            for prop, val in zip(properties, predicted_values)
        }
        
        results = {
            "tool": "predict_material_properties",
            "status": "completed",
            "input_pattern": str(pattern_file),
            "model_path": str(model_file),
            "predicted_properties": property_predictions,
            "properties_predicted": properties
        }
        
        return format_result(results)
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "status": "error"
        }, indent=2)


@mcp.tool()
async def list_available_models(
    models_directory: str = "~/ml_models/beamline"
) -> str:
    """List all available trained ML models for beamline data analysis.
    
    Args:
        models_directory: Directory containing trained models
    """
    try:
        models_dir = Path(models_directory).expanduser()
        
        if not models_dir.exists():
            return json.dumps({
                "error": f"Models directory not found: {models_dir}",
                "available_models": [],
                "status": "not_found"
            }, indent=2)
        
        # Find all model files
        model_files = []
        for ext in ['.pth', '.pt', '.h5', '.pkl', '.onnx']:
            model_files.extend(models_dir.glob(f'**/*{ext}'))
        
        # Organize by type
        models_by_type = {
            "segmentation": [],
            "classification": [],
            "regression": [],
            "denoising": [],
            "other": []
        }
        
        for model_file in model_files:
            model_info = {
                "name": model_file.stem,
                "path": str(model_file),
                "size_mb": model_file.stat().st_size / (1024 * 1024),
                "format": model_file.suffix
            }
            
            # Categorize by name
            name_lower = model_file.stem.lower()
            if 'segment' in name_lower or 'unet' in name_lower:
                models_by_type["segmentation"].append(model_info)
            elif 'classif' in name_lower or 'quality' in name_lower:
                models_by_type["classification"].append(model_info)
            elif 'denoise' in name_lower:
                models_by_type["denoising"].append(model_info)
            elif 'predict' in name_lower or 'regress' in name_lower:
                models_by_type["regression"].append(model_info)
            else:
                models_by_type["other"].append(model_info)
        
        results = {
            "tool": "list_available_models",
            "status": "completed",
            "models_directory": str(models_dir),
            "total_models": len(model_files),
            "models_by_type": models_by_type,
            "frameworks": {
                "pytorch": PYTORCH_AVAILABLE,
                "tensorflow": TENSORFLOW_AVAILABLE
            }
        }
        
        return format_result(results)
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "status": "error"
        }, indent=2)


if __name__ == "__main__":
    print("Starting ML Models FastMCP Server for Beamline Data...", file=sys.stderr)
    print("Available tools:", file=sys.stderr)
    print("  - segment_xct_image", file=sys.stderr)
    print("  - predict_xrd_phases_ml", file=sys.stderr)
    print("  - denoise_diffraction_image", file=sys.stderr)
    print("  - classify_diffraction_quality", file=sys.stderr)
    print("  - predict_material_properties", file=sys.stderr)
    print("  - list_available_models", file=sys.stderr)
    mcp.run(transport='stdio')