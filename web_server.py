#!/usr/bin/env python3
"""
Beamline Assistant Web Server
Integrates the web UI with MCP servers for diffraction analysis
Enhanced with image viewer capabilities for TIFF/GE diffraction images
"""

import asyncio
import json
import os
import tempfile
import io
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import uuid

import numpy as np
from PIL import Image
import tifffile
from scipy import ndimage

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import your MCP client
from argo_mcp_client import APEXAClient

app = FastAPI(title="Beamline Assistant API", version="0.1.0")

# CORS middleware for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global MCP client instance
mcp_client: Optional[APEXAClient] = None
upload_dir = Path("uploads")
upload_dir.mkdir(exist_ok=True)

# Image cache for viewer
image_cache: Dict[str, np.ndarray] = {}
image_paths: Dict[str, str] = {}  # Maps file_id to actual file path
calibration_cache: Dict[str, Dict[str, Any]] = {}


# ==================== Image Processing Functions ====================

def load_diffraction_image(file_path: str) -> np.ndarray:
    """Load TIFF, GE, or standard image formats with proper intensity handling"""
    path = Path(file_path)

    if path.suffix.lower() in ['.tif', '.tiff']:
        img = tifffile.imread(str(path))
        return np.array(img, dtype=np.float32)

    elif path.suffix.lower() in ['.ge', '.ge2', '.ge3', '.ge4', '.ge5']:
        # GE format: 8192 byte header, then 2-byte unsigned integers
        with open(path, 'rb') as f:
            f.seek(8192)
            data = np.fromfile(f, dtype=np.uint16)
            size = int(np.sqrt(len(data)))
            img = data.reshape(size, size)
            return np.array(img, dtype=np.float32)
    else:
        img = Image.open(path)
        return np.array(img, dtype=np.float32)


def apply_contrast(img: np.ndarray, vmin: float = None, vmax: float = None,
                   gamma: float = 1.0) -> np.ndarray:
    """Apply contrast adjustment with gamma correction"""
    if vmin is None:
        vmin = np.percentile(img, 1)
    if vmax is None:
        vmax = np.percentile(img, 99)

    img_norm = np.clip(img, vmin, vmax)
    img_norm = (img_norm - vmin) / (vmax - vmin + 1e-10)

    if gamma != 1.0:
        img_norm = np.power(img_norm, gamma)

    return img_norm


def apply_colormap(img: np.ndarray, colormap: str = 'gray') -> np.ndarray:
    """Apply colormap to grayscale image"""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(colormap)
    colored = cmap(img)
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    return rgb


def image_to_base64(img: np.ndarray, format: str = 'png') -> str:
    """Convert numpy array to base64 encoded image"""
    if img.dtype != np.uint8:
        img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-10) * 255).astype(np.uint8)
    else:
        img_norm = img

    if len(img_norm.shape) == 2:
        pil_img = Image.fromarray(img_norm, mode='L')
    else:
        pil_img = Image.fromarray(img_norm, mode='RGB')

    buffer = io.BytesIO()
    pil_img.save(buffer, format=format.upper())
    buffer.seek(0)

    img_base64 = base64.b64encode(buffer.read()).decode()
    return f"data:image/{format};base64,{img_base64}"


def calculate_radial_profile(img: np.ndarray, center: Tuple[int, int],
                             num_bins: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate radial intensity profile from center point"""
    y, x = np.indices(img.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    max_r = min(num_bins, int(r.max()))
    radial_profile = np.zeros(max_r)
    radial_counts = np.zeros(max_r)

    for i in range(max_r):
        mask = (r == i)
        if mask.any():
            radial_profile[i] = img[mask].mean()
            radial_counts[i] = mask.sum()

    radii = np.arange(max_r)
    return radii, radial_profile

def load_midas_calibration(cal_file: str) -> Dict[str, Any]:
    """Load MIDAS calibration file (.txt) and extract ring information"""
    calibration = {
        "beam_center": [0, 0],
        "pixel_size": 200.0,  # microns
        "detector_distance": 1000.0,  # mm
        "wavelength": 0.0,  # Angstroms
        "rings": []  # List of ring radii in pixels
    }

    try:
        with open(cal_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line.startswith('BeamCenter'):
                parts = line.split()
                if len(parts) >= 3:
                    calibration["beam_center"] = [float(parts[1]), float(parts[2])]
            elif line.startswith('PixelSize'):
                parts = line.split()
                if len(parts) >= 2:
                    calibration["pixel_size"] = float(parts[1])
            elif line.startswith('Distance'):
                parts = line.split()
                if len(parts) >= 2:
                    calibration["detector_distance"] = float(parts[1])
            elif line.startswith('Wavelength'):
                parts = line.split()
                if len(parts) >= 2:
                    calibration["wavelength"] = float(parts[1])
            elif line.startswith('Ring'):
                parts = line.split()
                if len(parts) >= 2:
                    calibration["rings"].append(float(parts[1]))

    except Exception as e:
        print(f"Error loading calibration file: {e}")

    return calibration

def calculate_azimuthal_profile(img: np.ndarray, center: Tuple[int, int], radius: float, width: float = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate azimuthal (angular) profile around a ring"""
    cy, cx = center
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]

    # Calculate distance and angle from center
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    theta = np.arctan2(y - cy, x - cx)  # Returns angle in radians (-π to π)
    theta_degrees = np.degrees(theta) % 360  # Convert to 0-360 degrees

    # Create mask for pixels within the ring (radius ± width/2)
    mask = np.abs(r - radius) <= (width / 2)

    if not mask.any():
        return np.array([]), np.array([])

    # Bin by angle (0-360 degrees, 360 bins = 1 degree per bin)
    num_bins = 360
    azimuthal_profile = np.zeros(num_bins)
    azimuthal_counts = np.zeros(num_bins)

    for i in range(num_bins):
        angle_mask = mask & (theta_degrees >= i) & (theta_degrees < i + 1)
        if angle_mask.any():
            azimuthal_profile[i] = img[angle_mask].mean()
            azimuthal_counts[i] = angle_mask.sum()

    # Handle bins with no data by interpolating
    angles = np.arange(num_bins)
    valid = azimuthal_counts > 0
    if valid.sum() > 0:
        azimuthal_profile[~valid] = np.interp(angles[~valid], angles[valid], azimuthal_profile[valid])

    return angles, azimuthal_profile

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except:
            pass

    async def broadcast(self, message: dict):
        for connection in self.active_connections[:]:
            try:
                await connection.send_text(json.dumps(message))
            except:
                self.active_connections.remove(connection)

manager = ConnectionManager()

async def initialize_mcp_client():
    """Initialize the MCP client with server configurations from servers.config"""
    global mcp_client

    try:
        mcp_client = APEXAClient()

        # Read server configurations from servers.config
        server_configs = []
        config_file = Path("servers.config")

        if config_file.exists():
            with open(config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith('#'):
                        if ':' in line:
                            name, script_path = line.split(':', 1)
                            server_configs.append({
                                "name": name.strip(),
                                "script_path": script_path.strip()
                            })
            print(f"Loaded {len(server_configs)} server(s) from servers.config")
        else:
            print("⚠️  Warning: servers.config not found, using fallback configuration")
            server_configs = [
                {"name": "midas", "script_path": "./midas_comprehensive_server.py"},
                {"name": "filesystem", "script_path": "./filesystem_server.py"},
                {"name": "executor", "script_path": "./command_executor_server.py"}
            ]

        await mcp_client.connect_to_multiple_servers(server_configs)
        print("MCP client initialized successfully")

    except Exception as e:
        print(f"Failed to initialize MCP client: {e}")
        mcp_client = None

# Use startup event handler for now (we'll fix deprecation later)
@app.on_event("startup")
async def startup_event():
    """Initialize MCP client on startup"""
    await initialize_mcp_client()

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown"""
    if mcp_client:
        await mcp_client.cleanup()

@app.get("/", response_class=HTMLResponse)
async def serve_web_ui():
    """Serve the main web UI"""
    html_file = Path("beamline_web_ui.html")
    if html_file.exists():
        return FileResponse(html_file)
    else:
        return HTMLResponse("""
        <html><body>
        <h1>Beamline Assistant</h1>
        <p>Web UI file not found. Please ensure beamline_web_ui.html is in the current directory.</p>
        </body></html>
        """)

@app.get("/test_viewer.html", response_class=HTMLResponse)
async def serve_test_viewer():
    """Serve the image viewer test page"""
    test_file = Path("test_viewer.html")
    if test_file.exists():
        return FileResponse(test_file)
    else:
        raise HTTPException(status_code=404, detail="Test viewer not found")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload"""
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        saved_filename = f"{file_id}{file_extension}"
        file_path = upload_dir / saved_filename
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "saved_path": str(file_path),
            "size": len(content),
            "type": file.content_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/analyze")
async def analyze_file(request: Dict[str, Any]):
    """Start diffraction analysis"""
    if not mcp_client:
        raise HTTPException(status_code=503, detail="MCP client not available")

    # Extract parameters from request
    file_id = request.get("file_id")
    analysis_type = request.get("analysis_type", "comprehensive")
    parameters = request.get("parameters")
    
    try:
        # Find uploaded file
        file_path = None
        for ext in ['.tif', '.tiff', '.png', '.dat', '.xy', '.txt']:
            potential_path = upload_dir / f"{file_id}{ext}"
            if potential_path.exists():
                file_path = potential_path
                break
        
        if not file_path:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Broadcast analysis start
        await manager.broadcast(json.dumps({
            "type": "analysis_start",
            "file_id": file_id,
            "analysis_type": analysis_type
        }))
        
        # Run analysis based on file type
        if file_path.suffix.lower() in ['.tif', '.tiff', '.png']:
            # 2D image analysis
            result = await analyze_2d_image(str(file_path), parameters or {})
        else:
            # 1D pattern analysis  
            result = await analyze_1d_pattern(str(file_path), parameters or {})
        
        # Broadcast results
        await manager.broadcast(json.dumps({
            "type": "analysis_complete",
            "file_id": file_id,
            "results": result
        }))
        
        return result
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        await manager.broadcast(json.dumps({
            "type": "analysis_error", 
            "file_id": file_id,
            "error": error_msg
        }))
        raise HTTPException(status_code=500, detail=error_msg)

async def analyze_2d_image(image_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze 2D diffraction image"""
    results = {}
    
    # Step 1: Detect rings
    await manager.broadcast(json.dumps({
        "type": "analysis_progress",
        "step": "Detecting diffraction rings",
        "progress": 20
    }))
    
    ring_result = await mcp_client.execute_tool_call(
        "midas_detect_diffraction_rings",
        {
            "image_path": image_path,
            "detector_distance": parameters.get("detector_distance", 1000.0),
            "wavelength": parameters.get("wavelength", 0.2066)
        }
    )
    results["ring_detection"] = json.loads(ring_result)
    
    # Step 2: Integrate to 1D
    await manager.broadcast(json.dumps({
        "type": "analysis_progress", 
        "step": "Integrating 2D to 1D pattern",
        "progress": 40
    }))
    
    integration_result = await mcp_client.execute_tool_call(
        "midas_integrate_2d_to_1d",
        {
            "image_path": image_path,
            "unit": "2th_deg",
            "detector_distance": parameters.get("detector_distance", 1000.0),
            "wavelength": parameters.get("wavelength", 0.2066)
        }
    )
    results["integration"] = json.loads(integration_result)
    
    # Step 3: Find peaks
    await manager.broadcast(json.dumps({
        "type": "analysis_progress",
        "step": "Analyzing diffraction peaks", 
        "progress": 60
    }))
    
    peak_result = await mcp_client.execute_tool_call(
        "midas_analyze_diffraction_peaks",
        {"pattern_file": f"{image_path}_integrated.dat"}
    )
    results["peak_analysis"] = json.loads(peak_result)
    
    # Step 4: Identify phases
    await manager.broadcast(json.dumps({
        "type": "analysis_progress",
        "step": "Identifying crystalline phases",
        "progress": 80
    }))
    
    # Extract peak positions from peak analysis
    peak_data = results["peak_analysis"].get("peak_data", [])
    peak_positions = [peak["position_2theta"] for peak in peak_data]
    
    if peak_positions:
        phase_result = await mcp_client.execute_tool_call(
            "midas_identify_crystalline_phases",
            {
                "peak_positions": peak_positions,
                "material_system": parameters.get("material_system", "unknown"),
                "temperature": parameters.get("temperature", 25.0)
            }
        )
        results["phase_identification"] = json.loads(phase_result)
    
    await manager.broadcast(json.dumps({
        "type": "analysis_progress",
        "step": "Analysis complete",
        "progress": 100
    }))
    
    return results

async def analyze_1d_pattern(pattern_path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze 1D diffraction pattern"""
    results = {}
    
    # Step 1: Peak analysis
    await manager.broadcast(json.dumps({
        "type": "analysis_progress",
        "step": "Analyzing diffraction peaks",
        "progress": 40
    }))
    
    peak_result = await mcp_client.execute_tool_call(
        "midas_analyze_diffraction_peaks",
        {
            "pattern_file": pattern_path,
            "min_peak_height": parameters.get("min_peak_height", 0.05)
        }
    )
    results["peak_analysis"] = json.loads(peak_result)
    
    # Step 2: Phase identification
    await manager.broadcast(json.dumps({
        "type": "analysis_progress",
        "step": "Identifying crystalline phases",
        "progress": 80
    }))
    
    peak_data = results["peak_analysis"].get("peak_data", [])
    peak_positions = [peak["position_2theta"] for peak in peak_data]
    
    if peak_positions:
        phase_result = await mcp_client.execute_tool_call(
            "midas_identify_crystalline_phases",
            {
                "peak_positions": peak_positions,
                "material_system": parameters.get("material_system", "unknown"),
                "temperature": parameters.get("temperature", 25.0)
            }
        )
        results["phase_identification"] = json.loads(phase_result)
    
    await manager.broadcast(json.dumps({
        "type": "analysis_progress",
        "step": "Analysis complete", 
        "progress": 100
    }))
    
    return results

@app.post("/api/quick_analysis")
async def quick_phase_identification(
    peak_positions: List[float],
    material_system: str = "unknown",
    temperature: float = 25.0
):
    """Quick phase identification from peak positions"""
    if not mcp_client:
        raise HTTPException(status_code=503, detail="MCP client not available")
    
    try:
        result = await mcp_client.execute_tool_call(
            "midas_identify_crystalline_phases",
            {
                "peak_positions": peak_positions,
                "material_system": material_system,
                "temperature": temperature
            }
        )
        return json.loads(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick analysis failed: {str(e)}")

@app.post("/api/chat")
async def chat_with_assistant(
    message: str = Form(...),
    file_id: Optional[str] = Form(None),
    model: str = Form("gpt4o")
):
    """Chat with the AI assistant"""
    if not mcp_client:
        raise HTTPException(status_code=503, detail="MCP client not available")
    
    try:
        # Set the model if different
        if model != mcp_client.selected_model:
            mcp_client.selected_model = model
        
        # Find file path if file_id provided
        image_path = None
        if file_id:
            for ext in ['.tif', '.tiff', '.png', '.dat', '.xy', '.txt']:
                potential_path = upload_dir / f"{file_id}{ext}"
                if potential_path.exists():
                    image_path = str(potential_path)
                    break
        
        # Process the query
        response = await mcp_client.process_diffraction_query(
            message, 
            image_path=image_path
        )
        
        return {"response": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket)
    print(f"WebSocket client connected. Total connections: {len(manager.active_connections)}")
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            print(f"Received WebSocket message: {message_data}")
            
            # Handle different message types
            if message_data["type"] == "chat":
                if mcp_client:
                    try:
                        # Build context about currently loaded images
                        context = ""
                        if image_cache or calibration_cache:
                            context += f"\n\n==== IMPORTANT CONTEXT ====\n"
                            context += "The user has loaded files in the image viewer. When they say 'this image' or 'the uploaded file', they mean:\n\n"

                        if image_cache:
                            context += f"LOADED DIFFRACTION IMAGES:\n"
                            for file_id in image_cache.keys():
                                file_path = image_paths.get(file_id, "unknown")
                                img_shape = image_cache[file_id].shape
                                context += f"  • File: {file_id}\n"
                                context += f"    Full path: {file_path}\n"
                                context += f"    Size: {img_shape[1]}x{img_shape[0]} pixels\n"
                                context += f"    USE THIS FILE PATH when running MIDAS tools\n\n"

                        if calibration_cache:
                            context += f"LOADED CALIBRATION FILES:\n"
                            for cal_id, cal_data in calibration_cache.items():
                                rings = cal_data.get('rings', [])
                                center = cal_data.get('beam_center', [0, 0])
                                wavelength = cal_data.get('wavelength', 0)
                                distance = cal_data.get('detector_distance', 0)
                                context += f"  • Calibration: {cal_id}\n"
                                context += f"    Beam center: ({center[0]:.1f}, {center[1]:.1f})\n"
                                context += f"    Number of rings: {len(rings)}\n"
                                if wavelength > 0:
                                    context += f"    Wavelength: {wavelength:.4f} Å\n"
                                if distance > 0:
                                    context += f"    Detector distance: {distance:.1f} mm\n"
                                if rings:
                                    context += f"    Ring radii: {', '.join([f'{r:.1f}' for r in rings[:5]])} pixels\n"
                                context += "\n"

                        # Append context to user message
                        user_message = message_data["message"]
                        if context:
                            context += "When the user asks to analyze 'this image' or 'the uploaded file', use the file path(s) shown above.\n"
                            context += "==== END CONTEXT ====\n"
                            user_message += context

                        print(f"Sending to AI with context: {user_message[:500]}...")  # Debug log

                        response = await mcp_client.process_diffraction_query(user_message)
                        await manager.send_personal_message({
                            "type": "chat_response",
                            "message": response
                        }, websocket)
                    except Exception as e:
                        await manager.send_personal_message({
                            "type": "error", 
                            "message": f"Chat processing failed: {str(e)}"
                        }, websocket)
                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": "MCP client not available"
                    }, websocket)
            
            elif message_data["type"] == "change_model":
                if mcp_client:
                    mcp_client.selected_model = message_data["model"]
                    await manager.send_personal_message({
                        "type": "model_changed",
                        "model": message_data["model"]
                    }, websocket)
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"WebSocket client disconnected. Total connections: {len(manager.active_connections)}")

@app.get("/api/status")
async def get_status():
    """Get server status"""
    return {
        "status": "running",
        "mcp_client_connected": mcp_client is not None,
        "connected_servers": list(mcp_client.sessions.keys()) if mcp_client else [],
        "active_connections": len(manager.active_connections),
        "upload_directory": str(upload_dir),
        "available_models": list(mcp_client.available_models.keys()) if mcp_client else []
    }

@app.get("/api/files")
async def list_uploaded_files():
    """List uploaded files"""
    files = []
    for file_path in upload_dir.glob("*"):
        if file_path.is_file():
            stat = file_path.stat()
            files.append({
                "file_id": file_path.stem,
                "filename": file_path.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "extension": file_path.suffix
            })
    return {"files": files}

@app.get("/api/viewer/status")
async def get_viewer_status():
    """Get current viewer state for debugging"""
    return {
        "loaded_images": list(image_cache.keys()),
        "image_paths": image_paths,
        "loaded_calibrations": list(calibration_cache.keys())
    }


# ==================== Image Viewer Endpoints ====================

@app.post("/api/viewer/load")
async def load_viewer_image(file: UploadFile = File(...)):
    """Load image for viewer with proper handling"""
    try:
        content = await file.read()
        temp_path = upload_dir / file.filename
        with open(temp_path, "wb") as f:
            f.write(content)

        img = load_diffraction_image(str(temp_path))
        file_id = file.filename
        image_cache[file_id] = img
        image_paths[file_id] = str(temp_path)  # Store the file path

        stats = {
            "shape": list(img.shape),
            "dtype": str(img.dtype),
            "min": float(img.min()),
            "max": float(img.max()),
            "mean": float(img.mean()),
            "std": float(img.std())
        }

        img_preview = apply_contrast(img)
        preview_base64 = image_to_base64(img_preview)

        return JSONResponse({
            "success": True,
            "file_id": file_id,
            "stats": stats,
            "preview": preview_base64
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/viewer/adjust")
async def adjust_viewer_image(
    file_id: str = Form(...),
    vmin: Optional[float] = Form(None),
    vmax: Optional[float] = Form(None),
    gamma: float = Form(1.0),
    colormap: str = Form('gray')
):
    """Apply contrast/colormap adjustments"""
    if file_id not in image_cache:
        raise HTTPException(status_code=404, detail="Image not loaded")

    try:
        img = image_cache[file_id]
        img_adjusted = apply_contrast(img, vmin, vmax, gamma)

        if colormap != 'gray':
            img_colored = apply_colormap(img_adjusted, colormap)
        else:
            img_colored = (img_adjusted * 255).astype(np.uint8)

        img_base64 = image_to_base64(img_colored)

        return JSONResponse({
            "success": True,
            "image": img_base64
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/viewer/radial_profile")
async def get_viewer_radial_profile(
    file_id: str = Form(...),
    center_x: int = Form(...),
    center_y: int = Form(...),
    num_bins: int = Form(1000)
):
    """Calculate radial profile"""
    if file_id not in image_cache:
        raise HTTPException(status_code=404, detail="Image not loaded")

    try:
        img = image_cache[file_id]
        radii, profile = calculate_radial_profile(img, (center_x, center_y), num_bins)

        return JSONResponse({
            "success": True,
            "radii": radii.tolist(),
            "intensity": profile.tolist()
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/viewer/azimuthal_profile")
async def get_viewer_azimuthal_profile(
    file_id: str = Form(...),
    center_x: int = Form(...),
    center_y: int = Form(...),
    radius: float = Form(...),
    width: float = Form(5.0)
):
    """Calculate azimuthal (angular) profile around a ring"""
    if file_id not in image_cache:
        raise HTTPException(status_code=404, detail="Image not loaded")

    try:
        img = image_cache[file_id]
        angles, profile = calculate_azimuthal_profile(img, (center_x, center_y), radius, width)

        return JSONResponse({
            "success": True,
            "angles": angles.tolist(),
            "intensity": profile.tolist()
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/viewer/load_calibration")
async def load_calibration_file(file: UploadFile = File(...)):
    """Load MIDAS calibration file and cache it"""
    try:
        content = await file.read()
        temp_path = upload_dir / file.filename
        with open(temp_path, "wb") as f:
            f.write(content)

        calibration = load_midas_calibration(str(temp_path))
        file_id = file.filename
        calibration_cache[file_id] = calibration

        return JSONResponse({
            "success": True,
            "file_id": file_id,
            "calibration": calibration
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/viewer/colormaps")
async def get_colormaps():
    """Get list of available colormaps"""
    return JSONResponse({
        "colormaps": [
            "gray", "viridis", "plasma", "inferno", "magma",
            "jet", "hot", "cool", "spring", "summer",
            "autumn", "winter", "bone", "copper"
        ]
    })


if __name__ == "__main__":
    print("Starting Beamline Assistant Web Server...")
    print("Make sure you have the following files in the current directory:")
    print("- beamline_web_ui.html")
    print("- argo_mcp_client.py") 
    print("- fastmcp_midas_server.py")
    print("- filesystem_server.py")
    print("- command_executor_server.py")
    print("- .env file with ANL_USERNAME and ARGO_MODEL")
    print("")
    print("Dependencies should be installed with uv:")
    print("  uv add fastapi uvicorn websockets python-multipart")
    
    uvicorn.run(
        "web_server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )