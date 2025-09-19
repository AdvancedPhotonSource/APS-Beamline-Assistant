# Beamline Assistant Documentation

**AI-Powered Synchrotron X-ray Diffraction Analysis System**

Version 1.0 | Argonne National Laboratory

## Overview

Beamline Assistant is an AI-powered system that provides real-time analysis of synchrotron X-ray diffraction data. It combines the MIDAS (Microstructure-Informed Data Analysis Suite) with advanced AI models through Argonne's Argo API gateway, enabling automated diffraction pattern analysis, phase identification, and scientific interpretation.

## System Requirements

### Prerequisites
- macOS 10.15+ or Linux
- Python 3.9+
- Access to Argonne internal network (VPN or on-site)
- ANL domain account with Argo API access

### Dependencies
- UV package manager
- MCP (Model Context Protocol) libraries
- MIDAS diffraction analysis tools
- Scientific Python stack (numpy, scipy, matplotlib)

## Installation

### 1. Install UV Package Manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc  # or ~/.bashrc
```

### 2. Clone and Setup Project
```bash
mkdir beamline-assistant && cd beamline-assistant
uv init
```

### 3. Install Dependencies
```bash
uv add mcp httpx python-dotenv numpy scipy matplotlib pandas
uv add Pillow scikit-image pyFAI fabio plotly seaborn lmfit
uv add orjson rich typer fastapi uvicorn websockets
uv add mcp httpx python-dotenv asyncio-compat
```

### 4. Configure Environment
Create a `.env` file:
```bash
# Required: Your ANL domain username
ANL_USERNAME=your_anl_username

# Default AI model
ARGO_MODEL=gpt4o

# Optional: Path to MIDAS installation
MIDAS_PATH=./MIDAS/utils
```

### 5. Make Scripts Executable
```bash
chmod +x *.py
```

## System Architecture

The system consists of multiple FastMCP servers connected through a central client:

- **MIDAS Server**: Diffraction analysis tools
- **Command Executor**: System command execution
- **Filesystem Server**: File operations
- **Argo Client**: AI integration and user interface

## Starting the System

### Basic Usage (MIDAS only)
```bash
uv run argo_mcp_client.py fastmcp_midas_server.py
```

### Full System (All servers)
```bash
uv run argo_mcp_client.py midas:fastmcp_midas_server.py executor:command_executor_server.py filesystem:filesystem_server.py
```

### System Status Check
After startup, verify all servers are connected:
```bash
Beamline> servers
Connected: ['midas', 'executor', 'filesystem']
```

## Available Commands

### Interactive Commands

| Command | Description | Example |
|---------|-------------|---------|
| `analyze <file>` | Full diffraction analysis | `analyze /data/steel.tif` |
| `models` | List available AI models | `models` |
| `model <name>` | Switch AI model | `model claudesonnet4` |
| `tools` | Show all analysis tools | `tools` |
| `servers` | Show connected servers | `servers` |
| `ls <path>` | List directory contents | `ls /data` |
| `run <command>` | Execute system command | `run python --version` |
| `help` | Show help information | `help` |
| `quit` | Exit the system | `quit` |

### Natural Language Queries

The system understands natural language queries about diffraction data:

```bash
# Phase identification
Beamline> I have peaks at 12.5, 18.2, 25.8, 31.4 degrees. What phases are these?

# Data quality assessment
Beamline> Check the quality of my diffraction pattern at /data/sample001.tif

# Pattern analysis
Beamline> Integrate the 2D pattern and find all peaks above 5% intensity

# General questions
Beamline> What's the difference between austenite and ferrite?
```

## Core Analysis Workflows

### 1. Phase Identification from Peak Positions

**Input**: List of 2θ peak positions
**Process**: Automatic crystallographic database matching
**Output**: Formatted phase analysis with confidence scores

```bash
Beamline> I have peaks at 12.5, 18.2, 25.8, 31.4 degrees. What phases are these?
```

**Expected Output**:
```
PHASE IDENTIFICATION RESULTS
==================================================
Input peaks analyzed: 4
Peak positions: [12.5, 18.2, 25.8, 31.4]

IDENTIFIED PHASES:
--------------------
1. Austenite (γ-Fe)
   Crystal System: Cubic
   Space Group: Fm-3m
   Phase Fraction: 65.0%
   Confidence: 92%
   Matched Peaks:
     • 12.5° → (111) (calc: 12.47°, intensity: 100%)
     • 18.2° → (200) (calc: 18.15°, intensity: 60%)
     • 25.8° → (220) (calc: 25.84°, intensity: 40%)

ANALYSIS SUMMARY:
-----------------
Total phases found: 1
Peaks matched: 3/4
Match quality: Excellent
```

### 2. Complete Diffraction Pattern Analysis

```bash
Beamline> analyze /beamline/data/steel_sample_RT.tif
```

**Analysis Steps**:
1. File existence verification
2. Ring detection and quality assessment
3. 2D to 1D pattern integration
4. Peak finding and fitting
5. Phase identification
6. Scientific interpretation

### 3. Data Quality Assessment

```bash
Beamline> Check the quality of /data/experiment_001.tif
```

**Assessment Includes**:
- Signal-to-noise ratio
- Background uniformity
- Peak resolution
- Detector performance metrics
- Recommendations for improvement

## AI Models Available

### Production Models (Stable)
- `gpt4o` - GPT-4o (recommended for most tasks)
- `gpt4turbo` - GPT-4 Turbo
- `gpt4` - Standard GPT-4

### Development Models (Latest Features)
- `claudesonnet4` - Claude Sonnet 4
- `gemini25pro` - Gemini 2.5 Pro
- `gpt5` - GPT-5 (when available)

### Switching Models
```bash
Beamline> model claudesonnet4
✅ Switched to model: claudesonnet4
   Provider: Anthropic
   Description: Claude Sonnet 4 (200K input, dev only)
```

## File Operations

### Supported File Formats
- **Images**: TIFF, PNG, JPG (2D diffraction patterns)
- **Data**: DAT, XY, TXT, CSV (1D integrated patterns)
- **Configurations**: JSON, YAML

### File System Commands
```bash
# List beamline data directory
Beamline> ls /beamline/data

# Check file information
Beamline> Get info about /data/sample.tif

# Read measurement log
Beamline> Read the file /beamline/logs/experiment.txt
```

## Command Execution

### Safe Command Execution
The system allows execution of whitelisted commands for safety:

**Allowed Commands**:
- File operations: `ls`, `cat`, `head`, `tail`, `find`
- Python tools: `python`, `pip`, `uv`
- System info: `ps`, `df`, `whoami`, `env`
- Development: `git`, `which`

```bash
# Check Python environment
Beamline> run python --version

# List large files
Beamline> run find /data -size +100M -type f

# Check disk usage
Beamline> run df -h
```

## Advanced Features

### Material-Specific Analysis

The system recognizes common material systems:

```bash
# Steel analysis
Beamline> I have a duplex steel sample with peaks at 12.5, 31.4, 44.7 degrees

# Aluminum alloy
Beamline> Analyze this aluminum alloy pattern with texture

# Ceramic analysis  
Beamline> What oxide phases could give peaks at 25.3, 35.1, 43.2 degrees?
```

### Temperature-Dependent Analysis

```bash
Beamline> I have peaks at 800°C: 12.1, 17.9, 25.5 degrees. What phases at this temperature?
```

### Batch Processing Guidance

```bash
Beamline> How can I analyze 100 similar steel samples automatically?
```

## Troubleshooting

### Common Issues

**Connection Problems**:
```bash
# Check server status
Beamline> servers

# Restart with single server
uv run argo_mcp_client.py midas:fastmcp_midas_server.py
```

**File Access Issues**:
```bash
# Verify file exists
Beamline> ls /path/to/your/file

# Check permissions
Beamline> run ls -la /path/to/file
```

**Analysis Errors**:
```bash
# Check MIDAS dependencies
Beamline> run python -c "import fabio, pyFAI; print('OK')"

# Test with simple data
Beamline> I have peaks at 12.5, 18.2 degrees. What could these be?
```

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "Server not connected" | MCP server failed | Restart with single server |
| "File not found" | Invalid file path | Use `ls` to verify path |
| "Command not allowed" | Restricted command | Use whitelisted commands only |
| "Argo API error" | Network/auth issue | Check VPN connection |

## Performance Optimization

### For Best Results

1. **Use specific queries**: "Peaks at 12.5, 18.2 degrees" vs "Analyze my data"
2. **Include context**: Mention material type, temperature, experimental conditions
3. **Choose appropriate models**: 
   - `gpt4o` for fast, accurate analysis
   - `claudesonnet4` for detailed scientific interpretation
   - `gemini25pro` for complex multi-step analysis

### Memory and Processing

- Large files (>50MB): Use `analyze` command with automatic chunking
- Multiple samples: Process individually for better accuracy
- Complex materials: Provide expected phases or material system

## Integration with Beamline Operations

### Real-time Analysis Workflow

1. **Data Collection**: Collect diffraction pattern
2. **Quick Check**: `Beamline> analyze /path/to/new/pattern.tif`
3. **Phase Verification**: Ask specific questions about results
4. **Parameter Optimization**: Get AI recommendations for next measurement

### Experiment Planning

```bash
# Get measurement suggestions
Beamline> I want to study austenite decomposition in steel. What peaks should I monitor?

# Optimize conditions
Beamline> My peaks are weak at 2θ = 45°. How can I improve the signal?
```

### Data Logging

```bash
# Document results
Beamline> Summarize today's phase analysis results for my lab notebook
```

## Safety and Security

### Data Security
- All analysis occurs locally on Argonne network
- No diffraction data sent to external services
- User authentication through ANL domain credentials

### Command Restrictions
- Only whitelisted system commands allowed
- No file modification commands in restricted directories
- Automatic timeout for long-running processes

## Support and Maintenance

### Getting Help
- **Technical Issues**: Contact system administrator
- **Analysis Questions**: Ask Beamline Assistant directly
- **Feature Requests**: Submit through Vector ticketing system

### Updates
```bash
# Update dependencies
uv sync --upgrade

# Check for system updates
Beamline> run uv list --outdated
```

### Logs and Diagnostics
```bash
# Check system status
Beamline> run ps aux | grep midas

# View recent errors  
Beamline> Check system logs for any errors
```

## Best Practices

### Query Formation
- Be specific with peak positions: "12.5, 18.2, 25.8 degrees"
- Include experimental context: "steel sample at 800°C" 
- Ask follow-up questions for clarification

### Data Management
- Use consistent file naming conventions
- Organize data by experiment date/type
- Keep measurement logs accessible

### Model Selection
- Start with `gpt4o` for general analysis
- Use `claudesonnet4` for detailed interpretation
- Try `gemini25pro` for complex multi-phase systems

### Workflow Efficiency
1. Quick phase ID first: check major phases
2. Detailed analysis second: quantitative results  
3. Interpretation last: scientific significance

This system transforms synchrotron diffraction analysis from manual interpretation to AI-assisted scientific discovery, enabling faster insights and better experimental decisions at the beamline.
