#!/usr/bin/env python3
"""
Beamline Assistant Web Server
Integrates the web UI with MCP servers for diffraction analysis
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import your MCP client
from argo_mcp_client import ArgoMCPClient

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
mcp_client: Optional[ArgoMCPClient] = None
upload_dir = Path("uploads")
upload_dir.mkdir(exist_ok=True)

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
    """Initialize the MCP client with server configurations"""
    global mcp_client
    
    try:
        mcp_client = ArgoMCPClient()
        
        # Configure your MCP servers
        server_configs = [
            {
                "name": "midas",
                "script_path": "./fastmcp_midas_server.py"
            },
            {
                "name": "filesystem", 
                "script_path": "./filesystem_server.py"
            },
            {
                "name": "executor",
                "script_path": "./command_executor_server.py"
            }
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
async def analyze_file(
    file_id: str,
    analysis_type: str = "comprehensive",
    parameters: Optional[Dict[str, Any]] = None
):
    """Start diffraction analysis"""
    if not mcp_client:
        raise HTTPException(status_code=503, detail="MCP client not available")
    
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
                        response = await mcp_client.process_diffraction_query(
                            message_data["message"]
                        )
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