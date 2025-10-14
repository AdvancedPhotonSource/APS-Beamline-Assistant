# ML Models MCP Server - Deployment Guide

## Overview

This MCP server provides access to trained ML models for:
- **XRD**: Phase prediction, pattern denoising, quality classification
- **XCT**: Image segmentation (pores, cracks, phases)
- **Material Properties**: Grain size, texture, stress prediction
- **Quality Control**: Automated data filtering

## Installation

### 1. Install Dependencies

```bash
# Core ML frameworks
uv add torch torchvision  # PyTorch
# OR
uv add tensorflow  # TensorFlow

# Image processing
uv add pillow opencv-python scikit-image

# Scientific computing
uv add scipy scikit-learn

# Optional: ONNX for model portability
uv add onnx onnxruntime
```

### 2. Set Up Model Directory

```bash
# Create models directory
mkdir -p ~/ml_models/beamline/{xrd,xct,quality,properties}

# Example structure:
~/ml_models/beamline/
├── xrd/
│   ├── phase_classifier_v1.pth
│   ├── pattern_denoiser_v2.pth
│   └── quality_classifier.pth
├── xct/
│   ├── unet_segmentation.pth
│   ├── pore_detector.pth
│   └── crack_segmenter.pth
├── quality/
│   └── diffraction_quality_classifier.pth
└── properties/
    ├── grain_size_predictor.pth
    └── stress_predictor.pth
```

### 3. Add to Your Client

```bash
# Start client with ML server
uv run argo_mcp_client.py \
    midas:fastmcp_midas_server.py \
    ml:ml_model_server.py \
    executor:command_executor_server.py \
    filesystem:filesystem_server.py
```

## Available Tools

### 1. **segment_xct_image** - XCT Segmentation

Segment tomography images to identify features.

**Use Cases:**
- Pore detection in metals
- Crack segmentation
- Phase separation
- Grain boundary detection

**Example:**
```bash
Beamline> Segment the XCT image /data/sample_001.tif to find pores

# AI will call:
TOOL_CALL: ml_segment_xct_image
ARGUMENTS: {
  "image_path": "/data/sample_001.tif",
  "model_path": "~/ml_models/beamline/xct/pore_detector.pth",
  "model_type": "unet",
  "device": "cuda"
}
```

**Output:**
```json
{
  "status": "completed",
  "input_image": "/data/sample_001.tif",
  "output_segmentation": "/data/sample_001_segmented.npy",
  "unique_labels": 3,
  "label_distribution": {
    "0": 1500000,  // Background
    "1": 50000,    // Pores
    "2": 200000    // Material
  }
}
```

### 2. **predict_xrd_phases_ml** - ML-Based Phase Prediction

Predict phases using trained neural network.

**Use Cases:**
- Quick phase screening
- Complex multi-phase systems
- When traditional matching fails
- Real-time analysis

**Example:**
```bash
Beamline> Use ML to predict phases from pattern /data/steel_RT.dat

TOOL_CALL: ml_predict_xrd_phases_ml
ARGUMENTS: {
  "pattern_file": "/data/steel_RT.dat",
  "model_path": "~/ml_models/beamline/xrd/phase_classifier_v1.pth",
  "confidence_threshold": 0.1
}
```

**Output:**
```json
{
  "status": "completed",
  "detected_phases": [
    {
      "phase": "austenite",
      "confidence": 0.85,
      "percentage": 85.0
    },
    {
      "phase": "ferrite",
      "confidence": 0.12,
      "percentage": 12.0
    }
  ],
  "primary_phase": "austenite",
  "primary_confidence": 0.85
}
```

### 3. **denoise_diffraction_image** - ML Denoising

Remove noise from diffraction patterns.

**Use Cases:**
- Low-dose acquisitions
- Fast time-resolved measurements
- Improve peak detection
- Enhance weak features

**Example:**
```bash
Beamline> Denoise the diffraction image /data/noisy_pattern.tif

TOOL_CALL: ml_denoise_diffraction_image
ARGUMENTS: {
  "image_path": "/data/noisy_pattern.tif",
  "model_path": "~/ml_models/beamline/xrd/pattern_denoiser_v2.pth",
  "method": "deep_learning"
}
```

### 4. **classify_diffraction_quality** - Quality Assessment

Automatically classify data quality.

**Use Cases:**
- Automated data filtering
- Real-time quality monitoring
- Decide if re-measurement needed
- High-throughput screening

**Example:**
```bash
Beamline> Check quality of /data/experiment_001.tif

TOOL_CALL: ml_classify_diffraction_quality
ARGUMENTS: {
  "image_path": "/data/experiment_001.tif",
  "model_path": "~/ml_models/beamline/quality/diffraction_quality_classifier.pth"
}
```

**Output:**
```json
{
  "predicted_quality": "good",
  "confidence": 0.92,
  "quality_probabilities": {
    "poor": 0.02,
    "acceptable": 0.06,
    "good": 0.92,
    "excellent": 0.00
  },
  "recommendation": "Proceed with analysis"
}
```

### 5. **predict_material_properties** - Property Prediction

Predict material properties directly from diffraction.

**Use Cases:**
- Grain size estimation
- Texture quantification
- Residual stress
- Hardness prediction

**Example:**
```bash
Beamline> Predict grain size and stress from /data/sample.dat

TOOL_CALL: ml_predict_material_properties
ARGUMENTS: {
  "diffraction_pattern": "/data/sample.dat",
  "model_path": "~/ml_models/beamline/properties/grain_size_predictor.pth",
  "properties": ["grain_size", "residual_stress", "texture_index"]
}
```

**Output:**
```json
{
  "predicted_properties": {
    "grain_size": 45.3,        // nm
    "residual_stress": -125.6,  // MPa
    "texture_index": 1.8
  },
  "properties_predicted": ["grain_size", "residual_stress", "texture_index"]
}
```

### 6. **list_available_models** - Model Inventory

List all trained models available.

**Example:**
```bash
Beamline> What ML models are available?

TOOL_CALL: ml_list_available_models
ARGUMENTS: {
  "models_directory": "~/ml_models/beamline"
}
```

## Training Your Own Models

### Example 1: U-Net for XCT Segmentation

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class UNet(nn.Module):
    """Simple U-Net for segmentation"""
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        
        # Decoder
        self.dec3 = self.conv_block(256, 128)
        self.dec2 = self.conv_block(128, 64)
        self.dec1 = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Decoder
        d3 = self.dec3(self.upsample(e3))
        d2 = self.dec2(self.upsample(d3))
        d1 = self.dec1(d2)
        
        return torch.sigmoid(d1)

# Training
model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Train loop
for epoch in range(100):
    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

# Save model
torch.save(model, '~/ml_models/beamline/xct/unet_segmentation.pth')
```

### Example 2: XRD Phase Classifier

```python
import torch
import torch.nn as nn

class XRDPhaseClassifier(nn.Module):
    """Neural network for phase classification from XRD patterns"""
    def __init__(self, input_size=2048, num_phases=5):
        super(XRDPhaseClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, num_phases)
        )
    
    def forward(self, x):
        return self.network(x)

# Training
model = XRDPhaseClassifier()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(50):
    for patterns, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(patterns)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Save with scaler
torch.save({
    'model': model,
    'scaler': scaler  # StandardScaler from sklearn
}, '~/ml_models/beamline/xrd/phase_classifier_v1.pth')
```

### Example 3: Denoising Autoencoder

```python
class DenoisingAutoencoder(nn.Module):
    """Autoencoder for denoising diffraction images"""
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Training with noisy data
model = DenoisingAutoencoder()
criterion = nn.MSELoss()

for epoch in range(100):
    for clean_images in train_loader:
        # Add noise
        noisy_images = clean_images + torch.randn_like(clean_images) * 0.1
        
        outputs = model(noisy_images)
        loss = criterion(outputs, clean_images)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model, '~/ml_models/beamline/xrd/pattern_denoiser_v2.pth')
```

## Integration with Your Beamline Assistant

### Update Client Configuration

```python
# In your client startup
server_configs = [
    {"name": "midas", "script_path": "fastmcp_midas_server.py"},
    {"name": "ml", "script_path": "ml_model_server.py"},  # Add ML server
    {"name": "executor", "script_path": "command_executor_server.py"},
    {"name": "filesystem", "script_path": "filesystem_server.py"}
]
```

### Natural Language Queries

```bash
# XCT Segmentation
Beamline> Segment pores in /data/aluminum_xct.tif using the U-Net model

# Phase Prediction
Beamline> Use ML to predict phases from the XRD pattern at /data/sample.dat

# Quality Check
Beamline> Is /data/diffraction_001.tif good quality for analysis?

# Denoising
Beamline> Denoise the noisy pattern /data/fast_acquisition.tif

# Property Prediction
Beamline> Predict grain size from /data/steel_pattern.dat
```

## Workflow Examples

### Workflow 1: Automated XCT Analysis Pipeline

```bash
# Step 1: Quality check
Beamline> Check quality of /data/sample_xct.tif

# Step 2: Denoise if needed
Beamline> Denoise /data/sample_xct.tif

# Step 3: Segment features
Beamline> Segment pores in /data/sample_xct_denoised.tif

# Step 4: Analyze results
Beamline> Calculate pore volume fraction from segmentation
```

### Workflow 2: High-Throughput XRD Screening

```bash
# Process multiple samples
Beamline> For each pattern in /data/batch1/:
1. Check quality
2. If quality > acceptable, predict phases with ML
3. If confidence > 0.8, proceed with full MIDAS analysis
```

### Workflow 3: Real-Time Quality Monitoring

```bash
# During data collection
Beamline> Monitor quality of incoming images in /data/realtime/
- Classify each new image
- Alert if quality < acceptable
- Suggest adjustment if needed
```

## Model Performance Tips

### 1. Data Preparation
- **Normalize patterns**: Same scale as training data
- **Augmentation**: Rotation, flipping for robustness
- **Balance classes**: Equal representation of phases

### 2. Model Selection
- **U-Net**: Best for segmentation tasks
- **ResNet/EfficientNet**: Good for classification
- **CNN-LSTM**: For time-series diffraction data
- **Transformers**: For complex pattern recognition

### 3. Hardware Acceleration
```python
# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Example with CUDA
TOOL_CALL: ml_segment_xct_image
ARGUMENTS: {
  "image_path": "/data/large_volume.tif",
  "model_path": "~/ml_models/beamline/xct/unet_segmentation.pth",
  "device": "cuda"  # Use GPU
}
```

### 4. Batch Processing
```python
# Process multiple images efficiently
images = [f"/data/sample_{i:03d}.tif" for i in range(100)]

for img in images:
    # Segment with ML
    result = segment_xct_image(img, model_path, device="cuda")
```

## Advanced Features

### 1. Ensemble Predictions
```python
# Use multiple models for better accuracy
models = [
    "phase_classifier_v1.pth",
    "phase_classifier_v2.pth",
    "phase_classifier_v3.pth"
]

# Average predictions
predictions = []
for model_path in models:
    pred = predict_xrd_phases_ml(pattern, model_path)
    predictions.append(pred)

# Ensemble result
ensemble_pred = average_predictions(predictions)
```

### 2. Transfer Learning
```python
# Fine-tune pre-trained model on your data
pretrained_model = torch.load('generic_xrd_model.pth')

# Freeze early layers
for param in pretrained_model.encoder.parameters():
    param.requires_grad = False

# Train only classifier
optimizer = torch.optim.Adam(
    pretrained_model.classifier.parameters(),
    lr=0.0001
)
```

### 3. Model Versioning
```bash
# Organize models by version
~/ml_models/beamline/
├── xrd/
│   ├── phase_classifier_v1.pth      # Original
│   ├── phase_classifier_v2.pth      # Improved with more data
│   ├── phase_classifier_v3_steel.pth # Specialized for steel
│   └── phase_classifier_latest.pth  # Symlink to current best
```

### 4. Uncertainty Quantification
```python
# Monte Carlo dropout for uncertainty estimation
class UncertaintyModel(nn.Module):
    def predict_with_uncertainty(self, x, n_samples=100):
        self.train()  # Enable dropout
        predictions = []
        
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std
```

## Troubleshooting

### Issue: Model not loading
```bash
Error: "Failed to load model"

Solutions:
1. Check PyTorch version compatibility
2. Verify model file not corrupted
3. Try loading with map_location='cpu'
4. Check model architecture matches saved weights
```

### Issue: Out of memory
```bash
Error: "CUDA out of memory"

Solutions:
1. Reduce batch size
2. Use smaller model
3. Process in patches/tiles
4. Use gradient checkpointing
```

### Issue: Poor predictions
```bash
Problem: Low confidence or wrong predictions

Solutions:
1. Check input normalization matches training
2. Verify image/pattern format correct
3. Ensure model trained on similar data
4. Consider retraining with more diverse dataset
```

## Best Practices

1. **Version Control**: Track model versions with metadata
2. **Validation**: Always validate on held-out test set
3. **Documentation**: Document training data and parameters
4. **Monitoring**: Log predictions for quality control
5. **Fallback**: Have traditional analysis as backup

## Example: Complete ML-Assisted Analysis

```bash
Beamline> Analyze /data/new_sample.tif with ML assistance

# System will:
# 1. Check quality with classifier
# 2. Denoise if quality < excellent
# 3. Detect rings with MIDAS
# 4. Integrate to 1D
# 5. Predict phases with ML
# 6. Confirm with traditional matching
# 7. Predict material properties
# 8. Generate comprehensive report
```

## Summary

The ML Models MCP Server enables:
- ✅ **Faster analysis** with ML predictions
- ✅ **Better quality** through denoising
- ✅ **Automated screening** via quality classification  
- ✅ **Novel insights** from property prediction
- ✅ **Seamless integration** with existing tools

Deploy your trained models and let AI assist your beamline experiments!