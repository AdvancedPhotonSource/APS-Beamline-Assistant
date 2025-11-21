# APEXA Web User Interface Guide

**Advanced Photon EXperiment Assistant - Browser-Based Analysis Platform**

---

## Overview

The APEXA Web UI provides a modern, browser-based interface for real-time diffraction data analysis. Perfect for demos, collaboration, and quick visualization during beamtime.

### Key Features
- 📊 **Interactive Visualizations** - Real-time plotting with Plotly.js
- 🤖 **AI Chat Assistant** - Natural language analysis queries
- 🖼️ **2D Image Viewer** - Advanced diffraction pattern visualization
- 🔬 **Automated Analysis** - One-click comprehensive analysis
- 💬 **Multi-Model Support** - GPT-4o, Claude, Gemini
- 📈 **Radial/Azimuthal Profiles** - Advanced image analysis tools

---

## Getting Started

### Launch the Web UI

```bash
./start_web_viewer.sh
```

**Access at:** http://localhost:8001

The interface opens automatically in your default browser with three main sections:
- **Left Sidebar** - File management and analysis controls
- **Center Panel** - Visualizations and results
- **Right Panel** - AI chat assistant

---

## Interface Layout

### Header Controls

**Connection Status**
- 🟢 **Connected** - Backend server running
- 🟡 **Demo Mode** - Simulated analysis (backend offline)

**Model Selector**
Choose your AI model:
- **GPT-4o** - Fast, general-purpose (default)
- **Claude Sonnet 4** - Best reasoning and detailed explanations
- **Gemini 2.5 Pro** - Long context, complex analysis

---

## Main Features

### 1. File Upload & Management

**Upload Methods:**
- **Drag & Drop** - Simply drag files onto the upload area
- **Click to Browse** - Click the upload box to select files
- **Recent Files** - Quick access to previously analyzed files

**Supported Formats:**
- **2D Diffraction Images:** `.tif`, `.tiff`, `.ge2`, `.ge3`, `.ge4`, `.ge5`, `.png`
- **1D Patterns:** `.dat`, `.xy`, `.txt`

**Upload Process:**
1. Drag/select your diffraction file
2. File appears in **Recent Files** list
3. Click to select for analysis
4. Use analysis buttons or chat with AI

---

### 2. Analysis Tools

#### 🔬 Start Analysis
**Comprehensive diffraction analysis pipeline:**
- Automatic peak detection
- Ring identification
- Phase identification
- Crystallographic analysis
- Statistical reporting

**Usage:**
```
1. Upload/select diffraction file
2. Click "Start Analysis" button
3. Watch progress bar in Results tab
4. View comprehensive results
```

**Analysis Steps:**
- Uploading file → 10%
- Loading image → 20%
- Detecting rings → 40%
- Integrating pattern → 60%
- Finding peaks → 80%
- Identifying phases → 100%

#### ⚡ Quick Phase ID
**Rapid phase identification from known peak positions:**
- Fast processing (< 1 second)
- Peak matching against databases
- Phase fraction estimation
- Confidence scoring

**Usage:**
```
1. Click "Quick Phase ID" button
2. View identified phases in chat
3. Get immediate crystallographic data
```

**Example Output:**
```
PHASE IDENTIFICATION RESULTS
================================

IDENTIFIED PHASES:
------------------
1. Austenite (γ-Fe)
   Crystal System: Cubic
   Space Group: Fm-3m
   Phase Fraction: 65.0%
   Confidence: 92%

2. Ferrite (α-Fe)
   Crystal System: Cubic  
   Space Group: Im-3m
   Phase Fraction: 35.0%
   Confidence: 88%
```

#### 📊 Sample Data
**Generate demonstration diffraction pattern:**
- Realistic steel sample pattern
- 5 characteristic peaks
- Instant visualization
- Perfect for testing and demos

---

### 3. Visualization Tab

#### Diffraction Pattern Plot
**Interactive 1D pattern visualization:**
- Zoomable/pannable with Plotly
- 2θ (degrees) vs. Intensity (counts)
- Automatic scaling
- Hover for data values

**Controls:**
- **Sample Data** - Generate demo pattern
- **Export** - Download plot as PNG

#### Peak Analysis Plot
**Bar chart of identified peaks:**
- Peak positions (2θ)
- Peak intensities
- Phase assignments
- Miller indices labels

---

### 4. Results Tab

**Comprehensive Analysis Output:**

**Phase Information:**
- Phase names and formulas
- Crystal systems
- Space groups
- Phase fractions (%)
- Confidence scores

**Peak Statistics:**
- Total peaks found
- Peak positions
- Peak intensities
- Ring assignments

**Analysis Summary:**
- Rings detected
- Phases identified
- Match quality
- Data quality metrics

---

### 5. Images Tab

**Advanced 2D Diffraction Image Viewer**

#### Image Upload & Display
- Load TIFF, GE, PNG detector images
- Auto-detect image center
- Real-time statistics display

#### Image Statistics
Displayed automatically upon load:
- **Min Intensity** - Lowest pixel value
- **Max Intensity** - Highest pixel value
- **Mean** - Average intensity
- **Std Dev** - Standard deviation

#### Image Controls

**Contrast Adjustment:**
- **Min Intensity (%)** - Lower bound (0-100%)
- **Max Intensity (%)** - Upper bound (0-100%)
- Real-time slider updates

**Gamma Correction:**
- Range: 0.1 to 3.0
- Default: 1.0 (linear)
- < 1.0: Brightens dark features
- \> 1.0: Enhances bright features

**Colormaps:**
Choose from 8 color schemes:
- **Grayscale** - Traditional (default)
- **Viridis** - Perceptually uniform
- **Plasma** - High contrast
- **Inferno** - Warm colors
- **Jet** - Rainbow (classic)
- **Hot** - Heat map
- **Cool** - Cold colors
- **Bone** - Light  grayscale

#### MIDAS Calibration Overlay
**Load and visualize calibration rings:**
1. Click "Load Calibration" button
2. Select MIDAS `Parameters.txt` file
3. Calibration rings overlay on image
4. View beam center and tilts

#### Radial Profile
**1D intensity vs. radius analysis:**
- Set beam center (X, Y coordinates)
- Click "Calculate Profile"
- View azimuthally-averaged intensity
- Perfect for ring quality assessment

**Usage:**
```
1. Set Center X and Y (auto-detected default)
2. Click "Calculate Profile"
3. View plot below controls
```

#### Azimuthal Profile
**Texture and orientation analysis:**
- Select radius and width
- Set beam center
- Extract intensity around ring
- Detect preferred orientations

**Parameters:**
- **Center X, Y** - Beam center coordinates
- **Radius** - Distance from center (pixels)
- **Width** - Integration width (pixels)

---

### 6. AI Chat Assistant

**Natural Language Interface**

**Ask About:**
- Phase identification
- Peak analysis
- Data quality
- Experimental parameters
- Material properties
- Temperature effects
- Analysis interpretation

**Example Questions:**
```
"What phases do you see in this pattern?"
"What's the confidence on the austenite phase?"
"How does temperature affect these peaks?"
"Is my data quality good?"
"What should I do next?"
```

**Chat Features:**
- Context-aware responses
- File-specific answers
- Analysis suggestions
- Real-time typing
- Message history
- Timestamped messages

**Model-Specific Strengths:**
- **GPT-4o** - Fast, general questions
- **Claude Sonnet 4** - Detailed explanations, literature context
- **Gemini 2.5 Pro** - Complex multi-step reasoning

---

## Workflows

### Basic Analysis Workflow

**For 1D Diffraction Patterns (.dat, .xy):**
```
1. Upload file via drag-and-drop
2. File auto-plots in Visualization tab
3. Click "Start Analysis"
4. View results in Results tab
5. Ask AI questions in chat
```

### 2D Image Analysis Workflow

**For Detector Images (.tif, .ge):**
```
1. Go to Images tab
2. Click "Upload Image"
3. Select TIFF/GE file
4. Adjust contrast/gamma if needed
5. Calculate radial profile
6. Load calibration overlay
7. Extract azimuthal profile for texture
```

### Quick Phase ID Workflow

**For Known Peak Lists:**
```
1. Click "Quick Phase ID"
2. AI identifies phases from demo peaks
3. View results in chat
4. Ask follow-up questions
```

### Demo/Presentation Workflow

**For Live Demonstrations:**
```
1. Click "Sample Data" button
2. Instant steel pattern generation
3. Show 1D and peak plots
4. Click "Start Analysis"
5. Discuss results as they appear
6. Use chat for Q&A
```

---

## Advanced Features

### Progress Tracking
Real-time analysis progress bar with percentage:
- Visual progress indicator
- Status messages
- Chat notifications
- Automatic tab switching

### File History
Recent files panel:
- Last 10 uploaded files
- Click to reselect
- Visual selection highlight
- One-click switching

### Export Capabilities
- **Download Plots** - PNG export from Plotly
- **Save Results** - Copy from Results tab
- **Chat History** - Complete conversation log

### Responsive Design
Works on all screen sizes:
- **Desktop** - Full 3-column layout
- **Tablet** - Collapsible sidebar
- **Mobile** - Stacked panels

---

## Integration with APEXA

### Backend Connection
Web UI connects to `web_server.py` backend:
- **Port:** 8001
- **API Endpoints:** `/api/upload`, `/api/analyze`, `/api/chat`, `/api/viewer/*`
- **Auto-fallback:** Demo mode if backend unavailable

### Shared Analysis Engine
Uses same MCP servers as CLI:
- `midas_comprehensive_server.py` - MIDAS tools
- `filesystem_server.py` - File operations
- `command_executor_server.py` - Command execution

### Model Configuration
Reads from `.env` file:
```bash
ARGO_MODEL=gpt4o  # Default model
ANL_USERNAME=your_username
```

---

## Troubleshooting

### Connection Issues

**"Demo Mode" Status:**
- Backend server not running
- Start with `./start_web_viewer.sh`
- Check port 8001 not in use

**File Upload Fails:**
- Check file format is supported
- Verify file size < 100MB
- Try different browser

### Display Issues

**Image Not Showing:**
- Wait for upload to complete
- Check browser console (F12)
- Try refresh (Ctrl+R)

**Plot Not Rendering:**
- Verify Plotly CDN loaded
- Check JavaScript errors
- Try different browser

### Analysis Not Starting

**Button Disabled:**
- Select a file first
- Wait for previous analysis to complete
- Check backend connection

**No Results Shown:**
- Switch to Results tab
- Check chat for error messages
- Verify file format compatibility

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Enter** | Send chat message |
| **Shift+Enter** | New line in chat |
| **Drag** | Pan plots |
| **Scroll** | Zoom plots |

---

## Performance Tips

### For Fast Analysis
1. Use "Quick Phase ID" for rapid results
2. Preload files before beamtime
3. Keep browser tab active
4. Close other heavy applications

### For Demos
1. Load "Sample Data" before presentation
2. Have key files pre-uploaded
3. Test backend connection beforehand
4. Prepare chat questions in advance

### For Image Viewing
1. Adjust contrast before zooming
2. Use grayscale for speed
3. Calculate radial profile at center first
4. Load calibration after image settles

---

## API Endpoints

For developers and advanced users:

### Viewer API (`/api/viewer/`)
- **POST `/load`** - Upload and load 2D image
- **POST `/contrast`** - Apply contrast adjustment
- **POST `/radial_profile`** - Calculate radial integration
- **POST `/azimuthal_profile`** - Calculate azimuthal integration
- **POST `/overlay_calibration`** - Add MIDAS calibration rings

### Analysis API
- **POST `/api/upload`** - Upload file to backend
- **POST `/api/analyze`** - Run comprehensive analysis
- **POST `/api/quick_analysis`** - Quick phase ID
- **POST `/api/chat`** - Send message to AI
- **GET `/api/status`** - Backend health check

---

## Best Practices

### During Beamtime
✅ Keep web UI open in dedicated browser window
✅ Pre-configure model selection
✅ Test backend connection before scanning
✅ Save plots periodically
✅ Document analysis in chat

### For Collaboration
✅ Share screen during remote analysis
✅ Use chat for annotation
✅ Export plots for reports
✅ Document parameters in messages

### For Data Quality
✅ Check image statistics immediately
✅ Verify radial profile smoothness
✅ Assess peak-to-background ratio
✅ Monitor saturation warnings

---

## Comparison: Web UI vs. CLI

| Feature | Web UI | CLI |
|---------|--------|-----|
| **Interface** | Visual, mouse-driven | Text, keyboard-driven |
| **Best For** | Demos, collaboration | Batch processing, scripting |
| **Learning Curve** | Gentle | Steep |
| **Speed** | Click-based | Command-based (faster) |
| **Visualization** | Built-in interactive | External plotting |
| **Chat** | Integrated panel | Full terminal |
| **File Handling** | Drag-and-drop | Path-based |
| **Ideal Use** | Beamtime, QA, demos | Automation, large datasets |

**Recommendation:** Use Web UI for interactive analysis and demos, CLI for automated workflows and batch processing.

---

## Credits

**Web UI Development:**
- Frontend: Vanilla JavaScript + Plotly.js
- Backend: Python Flask/FastAPI
- Design: Material Design inspired
- Integration: APEXA MCP architecture

**Dependencies:**
- Plotly.js 2.26.0 - Interactive plotting
- Python 3.10+ - Backend server
- Modern browser - Chrome, Firefox, Edge, Safari

---

## Version History

**Current Version:** 1.0.0  
**Last Updated:** November 2024  
**Compatibility:** APEXA 1.0.0+, All major browsers

---

**Ready to analyze! 🔬✨**

For CLI version, see [USER_MANUAL.md](file:///Users/b324240/Git/beamline-assistant-dev/USER_MANUAL.md)  
For smart features, see [APEXA_SMART_FEATURES_MANUAL.md](file:///Users/b324240/Git/beamline-assistant-dev/APEXA_SMART_FEATURES_MANUAL.md)
