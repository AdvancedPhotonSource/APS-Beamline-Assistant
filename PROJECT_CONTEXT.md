# APEXA - Advanced Photon EXperiment Assistant

## Project Overview
AI-powered beamline scientist assistant for High Energy Diffraction Microscopy (HEDM) at Advanced Photon Source (APS).

## Core Goal
Provide an interactive web-based interface for beamline scientists to:
- Upload and visualize diffraction data (GE, TIFF, HDF5 formats)
- Run MIDAS calibration and analysis workflows
- Integrate with FF-HEDM tools for grain mapping and strain analysis
- Chat with AI assistant for workflow guidance

## Technical Stack
- **Backend**: FastAPI (web_server.py) with WebSocket for real-time chat
- **Frontend**: Single-page HTML/JavaScript UI (beamline_web_ui.html)
- **Environment**: UV for APEXA dependencies, separate MIDAS conda environment
- **Analysis Tools**: MIDAS package (AutoCalibrateZarr.py, integrator.py, etc.)

## Key Architecture Decisions

### Dual Environment Setup
- **APEXA UV environment**: FastAPI, websockets, web server dependencies
- **MIDAS conda environment**: zarr, diplib, numba, h5py, skimage
- Bridge via `find_midas_python()` and `get_midas_env()` in midas_comprehensive_server.py

### File Handling
- Upload directory: `/Users/b324240/Git/beamline-assistant-dev/uploads/`
- Image paths stored in `image_paths` dict with file IDs
- Support for GE binary, TIFF, HDF5, Zarr formats

## Implemented Features

### 1. Data Upload & Visualization
- Multi-format support (GE, TIFF, HDF5)
- Interactive viewer with zoom, pan, pixel intensity readout
- Calibration ring overlay visualization
- Brightest pixel detection

### 2. MIDAS Auto-Calibration
- Full AutoCalibrateZarr.py parameter support
- Automatic image dimension detection
- Parameter file loading/saving
- Mask file upload (.bin, .tif, .npy, .h5)
- Real-time progress with matplotlib plots
- Outputs refined_MIDAS_params.txt

### 3. MIDAS Integration (2D→1D Caking)
- Complete integrator.py parameter support
- File pattern processing
- Multi-CPU batch processing
- HDF5 data location specification
- Parameter file reuse from calibration

### 4. AI Chat Interface
- WebSocket-based real-time chat
- Context-aware responses
- Workflow guidance
- NOT connected to MCP servers (standalone web chat)

### 5. Knowledge Base & Domain Expertise (NEW)
- RAG-powered semantic search across papers, logbooks, books
- Materials Project integration for crystallographic data
- Typical HEDM parameter guidelines
- Lsd/BC estimation from ring positions
- Sub-second query performance

## MCP Server Architecture

APEXA uses a modular MCP (Model Context Protocol) server architecture:

### Active Servers (configured in servers.config):
1. **midas_comprehensive_server.py** - Primary HEDM analysis server
   - FF-HEDM workflows (calibration, integration, grain tracking)
   - NF-HEDM reconstruction and DREAM.3D conversion
   - PF-HEDM workflows
   - Advanced analysis (misorientation, forward simulation)
   - Data management (GE→TIFF conversion, parameter files)
   - **Knowledge Base** (query_hedm_knowledge, get_material_properties, estimate_parameters_from_image)

2. **filesystem_server.py** - File operations
   - Directory listing, file reading, searching
   - Used by all other servers for file access

3. **command_executor_server.py** - System commands
   - Execute shell commands
   - Environment management

### Inactive/Optional Servers (commented in servers.config):
- **analysis_utilities_server.py** - Quick diagnostics (non-official MIDAS)
- **gsas2_server.py** - Rietveld refinement
- **maud_server.py** - Texture analysis
- **pyfai_server.py** - Fast azimuthal integration
- **dioptas_server.py** - Interactive data processing

### Knowledge Base Components:
- **knowledge_base/** - Domain expertise directory
  - papers/ - Research PDFs (gitignored)
  - logbooks/ - Experimental notes (gitignored)
  - books/ - Textbooks (gitignored)
  - data/ - JSON databases (tracked)
    - materials.json - Crystallographic data from Materials Project
    - typical_parameters.json - HEDM best practices
  - chroma_db/ - Vector database (gitignored, auto-generated)
  - index_knowledge.py - Indexing script
  - fetch_materials_from_mp.py - Materials Project API integration

## Recent Development Sessions

### Session 2025-12-05 (Part 2): Gradio Conversational UI
**Added modern AI-driven chat interface**

#### New Features:
1. **Gradio Web UI** (localhost:7860)
   - Conversational chat interface (primary interaction method)
   - Drag-and-drop file uploads
   - Embedded visualizations in chat thread
   - Example prompts for quick start
   - Real-time progress tracking

2. **Infinite Loop Protection**
   - Detects repeated tool calls (3+ times with same arguments)
   - Breaks loop and forces AI to proceed with available info
   - Prevents "filesystem_list_directory" trap

3. **Three UI Options**:
   - **Gradio UI** (recommended): AI-driven chat, drag-and-drop, embedded plots
   - **CLI**: Terminal-based, power users, scripting
   - **Web UI**: Traditional forms, direct parameter control

#### Architecture:
- `gradio_ui.py`: Wraps existing ArgoMCPClient with Gradio interface
- `start_gradio_ui.sh`: Startup script for Gradio UI
- Connects to same MCP servers (filesystem, executor, midas)
- All MIDAS tools accessible via natural language

#### Why Gradio:
- ✅ Built-in ChatInterface (perfect for conversational AI)
- ✅ 20 lines of code to prototype
- ✅ Plays well with existing FastAPI backend
- ✅ Easy to embed plots in chat responses
- ✅ Modern, professional UI out-of-the-box

#### User Experience Improvements:
- No need to know parameter names
- Progressive disclosure (simple questions → advanced control)
- Contextual suggestions ("Would you like to integrate?")
- Natural language error messages
- Visual feedback for long-running operations

### Session 2025-12-05 (Part 1): Knowledge Base & RAG Integration
**Implemented PhD-level domain expertise system**

#### New Features:
1. **RAG-Powered Knowledge Base**
   - ChromaDB vector database for semantic search
   - Sentence Transformers (all-MiniLM-L6-v2) for embeddings
   - Sub-second query performance across 166+ chunks
   - Indexed: 4 research papers, 10 MIDAS logbooks

2. **Materials Project Integration**
   - API integration via mp-api package
   - Fetch authoritative crystallographic data (DFT-calculated)
   - Properties: lattice parameters, space groups, density, structure

3. **MCP Tools Added to midas_comprehensive_server.py**:
   - `query_hedm_knowledge`: Semantic search papers/logbooks/books
   - `get_material_properties`: Materials Project crystallographic lookup
   - `get_typical_hedm_parameters`: HEDM best practices and thresholds
   - `estimate_parameters_from_image`: Lsd/BC estimation from ring positions using Bragg's law

4. **Knowledge Base Infrastructure**:
   - `knowledge_base/index_knowledge.py`: Document indexing with chunk overlap
   - `knowledge_base/fetch_materials_from_mp.py`: Materials Project data fetching
   - `knowledge_base/data/typical_parameters.json`: Quality thresholds, parameter ranges

#### Architecture Decisions:
- **Lazy loading**: Knowledge base initialized on first query (not startup)
- **Separation of concerns**: ChromaDB for semantic search, JSON for structured data
- **Privacy-first**: User documents (PDFs, logbooks) gitignored
- **Dual deployment**: Code tracked in git, data copied separately to beamline
- **Local-first**: No cloud dependencies, runs entirely offline

#### Dependencies Added:
- chromadb>=0.4.0 (vector database)
- sentence-transformers>=2.2.0 (embeddings)
- pypdf2>=3.0.0 (PDF text extraction)
- mp-api>=0.41.0 (Materials Project)

#### Documentation Updates:
- USER_MANUAL.md: Added knowledge base setup and usage section
- PROJECT_CONTEXT.md: Now tracked in git (removed from .gitignore)
- .gitignore: Configured to exclude user documents but track infrastructure

### Session 2025-12-04: Calibration & Integration UI
**Fixed core MIDAS workflows and added integration UI**

#### Issues Resolved:
1. **Calibration dimension mismatch**: Fixed by auto-detecting image dimensions and adding NrPixelsY/NrPixelsZ to parameter files
2. **MIDAS environment errors**: Properly using find_midas_python() to call MIDAS scripts
3. **Integration UI**: Added complete integration section to web UI

#### Latest Successful Test:
- Image: CeO2_650mm_61p332keV_2DFocused_0p1s_att200_004018.tif (1679×1475 px)
- Calibration: Successfully converged with mean strain 0.0003
- Parameters: Lsd=651.12mm, BC=[702.87, 812.52]px

## File Structure
```
beamline-assistant-dev/
├── web_server.py                    # Main FastAPI server
├── beamline_web_ui.html            # Frontend UI
├── argo_mcp_client.py              # MCP client for Claude Desktop
├── start_beamline_assistant.sh     # Startup script (adds ~/.local/bin to PATH)
├── servers.config                   # MCP server configuration
├── midas_comprehensive_server.py   # Primary HEDM MCP server + Knowledge Base
├── filesystem_server.py            # File operations MCP server
├── command_executor_server.py      # System commands MCP server
├── knowledge_base/                 # Domain expertise (NEW)
│   ├── papers/                     # Research PDFs (gitignored)
│   ├── logbooks/                   # Experimental notes (gitignored)
│   ├── books/                      # Textbooks (gitignored)
│   ├── data/                       # Structured data (tracked)
│   │   ├── materials.json          # Materials Project crystallographic data
│   │   └── typical_parameters.json # HEDM best practices
│   ├── chroma_db/                  # Vector database (gitignored, auto-generated)
│   ├── index_knowledge.py          # Document indexing script
│   └── fetch_materials_from_mp.py  # Materials Project API integration
├── uploads/                         # Uploaded data files
├── test2/                          # Test data directory
├── USER_MANUAL.md                  # End-user documentation
└── PROJECT_CONTEXT.md              # AI development context (this file)
```

## Common Workflows

### Calibration Workflow
1. Upload calibration image (TIFF/GE/HDF5)
2. Upload or create parameter file with SpaceGroup, px, Wavelength, LatticeParameter
3. Optional: Upload mask file
4. Set initial guesses for Lsd and BC
5. Run Auto-Calibration
6. Review refined parameters

### Integration Workflow
1. Use refined parameters from calibration
2. Specify data file pattern and range
3. Set processing options (CPUs, chunks)
4. Run MIDAS Integration
5. Review 1D integrated spectra

## Known Limitations
- Web UI chat NOT connected to MCP servers (separate AI endpoint)
- Large HDF5 files may timeout (current: 30min limit)
- Knowledge base requires manual setup on each deployment (copy documents, run indexing)
- Materials Project API requires free account/key for crystallographic data

## Key Parameters Reference

### Detector
- Pixel size (px): 172.0 µm (typical GE detector)
- Dimensions: Variable (auto-detected from image)

### Typical Materials
- CeO2: SpaceGroup 225, a=5.4116 Å
- LaB6: SpaceGroup 221, a=4.1569 Å
- Si: SpaceGroup 227, a=5.4309 Å

## Development Notes
- Always read files before editing
- Use proper Python environment (UV for server, conda for MIDAS)
- Test with files in test2/ directory before production
- Check autocal.log for calibration debugging
