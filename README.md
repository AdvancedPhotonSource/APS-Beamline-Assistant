# APEXA - Advanced Photon EXperiment Assistant

AI-powered beamline scientist for synchrotron X-ray diffraction analysis at Argonne National Laboratory's Advanced Photon Source.

## Quick Start

```bash
./setup_user.sh                  # One-time setup
./start_beamline_assistant.sh    # Start analyzing
```

## Features

- **AI-Powered** - Natural language via Argo Gateway (GPT-4o, Claude, Gemini)
- **Modular** - Easy to add analysis tools (GSAS-II, MAUD, PyFAI, etc.)
- **Context-Aware** - Remembers previous queries in conversation
- **Multi-User** - Separate configuration per user
- **Auto-Detection** - Finds tools automatically at standard locations
- **20+ Tools** - MIDAS FF/NF-HEDM analysis suite

## Documentation

### Essential
- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 2 minutes
- **[BEAMLINE_DEPLOYMENT.md](BEAMLINE_DEPLOYMENT.md)** - Multi-user deployment

### For Developers
- **[ADDING_NEW_SERVERS.md](ADDING_NEW_SERVERS.md)** - Add new tools (5 min)
- **[MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md)** - How it works
- **[docs/development/](docs/development/)** - Technical details

### Archive
- **[docs/archive/](docs/archive/)** - Old documentation (reference)

## Usage

```
APEXA> integrate the .tiff file from 2D to 1D
APEXA> integrate data.ge5 with dark file dark.ge5 using calib.txt
APEXA> plot 2d sample.ge5
APEXA> plot radial data.tiff
APEXA> run FF-HEDM workflow on /data/experiment
APEXA> list files in /data
APEXA> read the Parameters.txt file there
```

### ✨ Smart Features

- **🔍 Multimodal Image Analysis**: AI can "see" and analyze your diffraction images directly
- **📈 Advanced Plotting**: 2D images, radial profiles, 1D patterns, multi-pattern comparisons
- **🔄 Real-time Monitoring**: Automatic alerts during beamtime for quality issues
- **📦 Batch Processing**: Process hundreds of files with one command
- **🛡️ Error Prevention**: Validate parameters before execution
- **🌑 Dark File Subtraction**: Automatic background correction
- **📁 Multi-Format Support**: TIFF, GE2/GE5, ED5, EDF detector formats
- **💡 Smart Suggestions**: Proactive next-step recommendations
- **💾 Session Persistence**: Automatic tracking of experimental workflows
- **⚡ Smart Caching**: Faster repeated operations, lower AI costs

## Adding Analysis Tools

```bash
cp servers/gsas2_server.py servers/mynew_server.py  # Copy template
nano servers/mynew_server.py                        # Edit & add @mcp.tool()
echo "mynew:servers/mynew_server.py" >> servers.config  # Enable
uv add required-packages                            # Install dependencies
./start_beamline_assistant.sh                      # Restart
```

See [ADDING_NEW_SERVERS.md](ADDING_NEW_SERVERS.md) for complete guide.

## Configuration

**User** (.env):
```bash
ANL_USERNAME=your_username
ARGO_MODEL=gpt4o
MIDAS_PATH=~/.MIDAS  # Optional - auto-detected
```

**Servers** (servers.config):
```bash
filesystem:filesystem_server.py
executor:command_executor_server.py
midas:midas_comprehensive_server.py
# gsas2:servers/gsas2_server.py  # Uncomment to enable
```

## Architecture

```
argo_mcp_client.py (AI Client)
    ↓
servers.config
    ↓
├── midas_comprehensive_server.py (20+ HEDM tools)
├── filesystem_server.py (file operations)
├── command_executor_server.py (commands)
└── servers/
    ├── gsas2_server.py (Rietveld - template)
    └── maud_server.py (Texture - template)
```

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager  
- ANL network access
- Analysis tools at standard locations

## Troubleshooting

**Tool not found?** → `export TOOL_PATH=/path/to/tool`  
**Import error?** → `uv add package-name`  
**Server not loading?** → Check `servers.config` syntax

## Project Structure

```
beamline-assistant/
├── argo_mcp_client.py            # AI client
├── servers.config                 # Server configuration
├── start_beamline_assistant.sh   # Startup
├── setup_user.sh                  # User setup
├── Core servers/
│   ├── midas_comprehensive_server.py
│   ├── filesystem_server.py
│   └── command_executor_server.py
├── servers/                       # Additional servers
│   ├── gsas2_server.py
│   └── maud_server.py
└── docs/                          # Documentation
    ├── development/               # Technical
    └── archive/                   # Old docs
```

## Acknowledgments

- [MIDAS](https://github.com/marinerhemant/MIDAS) by Hemant Sharma
- [FastMCP](https://github.com/jlowin/fastmcp) by Marvin
- [uv](https://github.com/astral-sh/uv) by Astral
- Argo Gateway by ANL

---

**Quick Links:**
[Get Started](QUICKSTART.md) | 
[Deploy](BEAMLINE_DEPLOYMENT.md) | 
[Add Tools](ADDING_NEW_SERVERS.md) | 
[Architecture](MODULAR_ARCHITECTURE.md)
