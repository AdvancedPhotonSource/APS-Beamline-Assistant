# Modular Architecture for Extensible Analysis Platform

## Overview

The Beamline Assistant now features a **modular architecture** that makes it easy to add new analysis tools (GSAS-II, MAUD, PyFAI, DIOPTAS, etc.) without modifying core code.

## Key Features

✓ **Dynamic Server Loading** - Servers configured in `servers.config`
✓ **Auto-Detection** - Tools automatically found at standard locations
✓ **Easy Extension** - Add new tools by creating a server file
✓ **uv Package Manager** - Modern Python package management
✓ **No Core Changes** - Add tools without touching argo_mcp_client.py
✓ **Enable/Disable** - Comment/uncomment servers in config file

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  argo_mcp_client.py                     │
│              (AI Client - No Changes Needed)            │
└──────────────────────────┬──────────────────────────────┘
                           │
                           │ Reads servers.config
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    servers.config                       │
│    ┌────────────────────────────────────────────┐      │
│    │ filesystem:filesystem_server.py            │      │
│    │ executor:command_executor_server.py        │      │
│    │ midas:midas_comprehensive_server.py        │      │
│    │ gsas2:servers/gsas2_server.py              │      │
│    │ maud:servers/maud_server.py                 │      │
│    │ # pyfai:servers/pyfai_server.py (disabled) │      │
│    └────────────────────────────────────────────┘      │
└──────────────────────────┬──────────────────────────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
┌───────────▼────┐  ┌──────▼──────┐  ┌───▼─────────┐
│ MIDAS Server   │  │ GSAS-II     │  │ MAUD        │
│ (FF-HEDM,      │  │ Server      │  │ Server      │
│  NF-HEDM,      │  │ (Rietveld)  │  │ (Texture)   │
│  PF-HEDM)      │  │             │  │             │
└────────────────┘  └─────────────┘  └─────────────┘
```

## Files Structure

```
beamline-assistant/
├── argo_mcp_client.py            # AI client (core - no changes)
├── servers.config                 # Server configuration (edit this!)
├── start_beamline_assistant.sh   # Startup script (now dynamic)
├── .env.template                  # Configuration template
├── ADDING_NEW_SERVERS.md         # Developer guide
├── MODULAR_ARCHITECTURE.md       # This file
│
├── servers/                       # New analysis servers go here
│   ├── gsas2_server.py           # GSAS-II integration
│   ├── maud_server.py            # MAUD integration
│   ├── pyfai_server.py           # PyFAI integration (future)
│   └── dioptas_server.py         # DIOPTAS integration (future)
│
└── Core servers (keep in root):
    ├── midas_comprehensive_server.py
    ├── filesystem_server.py
    └── command_executor_server.py
```

## How It Works

### 1. Configuration (servers.config)

```bash
# Core servers (always recommended)
filesystem:filesystem_server.py
executor:command_executor_server.py

# MIDAS - High Energy Diffraction
midas:midas_comprehensive_server.py

# GSAS-II - Rietveld refinement
gsas2:servers/gsas2_server.py

# MAUD - Texture analysis
# maud:servers/maud_server.py    # Disabled (commented out)
```

### 2. Dynamic Loading (start_beamline_assistant.sh)

The startup script now:
1. Reads `servers.config`
2. Parses each line for `name:path`
3. Checks if server file exists
4. Builds argument list dynamically
5. Starts argo_mcp_client.py with all enabled servers

**Output:**
```
Loading analysis servers:
  ✓ filesystem (filesystem_server.py)
  ✓ executor (command_executor_server.py)
  ✓ midas (midas_comprehensive_server.py)
  ✓ gsas2 (servers/gsas2_server.py)
  ✗ maud (not found: servers/maud_server.py)

Active Servers: filesystem executor midas gsas2
```

### 3. Server Auto-Detection

Each server finds its tool automatically:

```python
def find_tool_installation() -> Optional[Path]:
    """Find tool installation."""

    # 1. Check environment variable
    if "TOOL_PATH" in os.environ:
        path = Path(os.environ["TOOL_PATH"]).expanduser()
        if path.exists():
            return path

    # 2. Check common locations
    common_paths = [
        Path.home() / ".tool",
        Path.home() / "tool",
        Path("/opt/tool")
    ]

    for path in common_paths:
        if path.exists():
            return path

    return None
```

## Adding a New Tool

### Quick Start (5 minutes)

```bash
# 1. Copy template
cp servers/gsas2_server.py servers/mynew_server.py

# 2. Edit the server
nano servers/mynew_server.py
# - Change server name
# - Add your tools with @mcp.tool()
# - Update installation detection

# 3. Add to configuration
echo "mynew:servers/mynew_server.py" >> servers.config

# 4. Install dependencies
uv add required-packages

# 5. Restart
./start_beamline_assistant.sh
```

**Done!** Your server is now integrated.

### Detailed Steps

See [ADDING_NEW_SERVERS.md](ADDING_NEW_SERVERS.md) for:
- Complete server template
- Best practices
- Error handling
- Testing procedures
- Multiple examples

## Example Servers

### GSAS-II Server (Rietveld Refinement)

**Location:** `servers/gsas2_server.py`

**Features:**
- Auto-detects GSAS-II at ~/GSASII, /opt/GSASII
- Installation checker
- Placeholder tools for future implementation
- Ready to extend

**Tools:**
- `check_gsas2_installation` - Verify GSAS-II is installed
- `run_rietveld_refinement` - Run Rietveld refinement (placeholder)

**Usage:**
```bash
# Enable in servers.config
gsas2:servers/gsas2_server.py

# Set path (optional, auto-detected)
export GSAS2_PATH=~/GSASII

# Use
Beamline> check GSAS-II installation
```

### MAUD Server (Texture Analysis)

**Location:** `servers/maud_server.py`

**Features:**
- Auto-detects MAUD at ~/MAUD, /opt/MAUD
- Texture analysis placeholder
- Quantitative phase analysis placeholder

**Tools:**
- `check_maud_installation` - Verify MAUD is installed
- `run_texture_analysis` - Texture ODF/pole figures (placeholder)
- `run_quantitative_phase_analysis` - Phase fractions (placeholder)

## Benefits

### For Users

**Before (monolithic):**
- All tools bundled together
- Can't disable unused tools
- Hard to add new capabilities

**After (modular):**
- Enable only what you need
- Easy to try new tools
- Faster startup (fewer servers)
- Clear organization

### For Administrators

**Before:**
- One-size-fits-all configuration
- Hard to customize per beamline
- Tool conflicts possible

**After:**
- Customize per beamline:
  ```bash
  # Beamline 1: HEDM only
  cp configs/hedm.config servers.config

  # Beamline 2: Powder diffraction
  cp configs/powder.config servers.config
  ```
- No conflicts - servers isolated
- Easy maintenance

### For Developers

**Before:**
- Modify core code to add tools
- Risk breaking existing features
- Tightly coupled

**After:**
- No core code changes
- Independent development
- Easy testing
- Clean interfaces

## Configuration Profiles

Create different profiles for different use cases:

### configs/hedm.config
```bash
filesystem:filesystem_server.py
executor:command_executor_server.py
midas:midas_comprehensive_server.py
```

### configs/powder.config
```bash
filesystem:filesystem_server.py
executor:command_executor_server.py
gsas2:servers/gsas2_server.py
maud:servers/maud_server.py
```

### configs/all.config
```bash
filesystem:filesystem_server.py
executor:command_executor_server.py
midas:midas_comprehensive_server.py
gsas2:servers/gsas2_server.py
maud:servers/maud_server.py
pyfai:servers/pyfai_server.py
```

**Switch profiles:**
```bash
cp configs/powder.config servers.config
./start_beamline_assistant.sh
```

## Package Management with uv

The project uses [uv](https://github.com/astral-sh/uv) for fast, reliable package management.

### Why uv?

- ⚡ **Fast** - 10-100x faster than pip
- 🔒 **Deterministic** - Reproducible installs
- 🎯 **Simple** - Drop-in pip replacement
- 📦 **Modern** - Written in Rust

### Common Commands

```bash
# Install project
uv sync

# Add dependency
uv add numpy scipy matplotlib

# Add development dependency
uv add --dev pytest black

# Install from requirements.txt
uv pip install -r requirements.txt

# Run command
uv run python script.py

# Create virtual environment
uv venv
```

### Adding Server Dependencies

When creating a new server:

```bash
# Add required packages
uv add package1 package2

# Or edit pyproject.toml and sync
uv sync
```

## Testing

### Test Individual Server

```bash
# Run server standalone
uv run servers/gsas2_server.py

# Should start without errors
```

### Test Integration

```bash
# Enable server
nano servers.config  # Add your server

# Start system
./start_beamline_assistant.sh

# Check it loaded
# Should see: ✓ myserver (servers/myserver.py)
```

### Test Tools

```bash
Beamline> tools
# Should list your tools

Beamline> check myserver installation
# Test a tool
```

## Migration from Old System

### Old (hardcoded):

```bash
#!/bin/bash
uv run argo_mcp_client.py \
  midas:midas_comprehensive_server.py \
  executor:command_executor_server.py \
  filesystem:filesystem_server.py
```

### New (dynamic):

```bash
#!/bin/bash
# Reads servers.config automatically
SERVER_ARGS=$(parse_servers_config)
uv run argo_mcp_client.py $SERVER_ARGS
```

**Migration:** Just run the new script - it auto-creates `servers.config` with defaults!

## Future Analysis Tools

### Planned Integrations

- [x] MIDAS (FF/NF-HEDM) - Complete
- [x] GSAS-II (Rietveld) - Template ready
- [x] MAUD (Texture) - Template ready
- [ ] PyFAI (Integration) - Coming soon
- [ ] DIOPTAS (Interactive) - Coming soon
- [ ] Fit2D (2D integration) - Planned
- [ ] DAWN (Data processing) - Planned
- [ ] Custom tools - Easy to add!

### Community Contributions

Want to add a tool?

1. Create server using template
2. Test locally
3. Document it
4. Share with community
5. Add to official repository

## Best Practices

### 1. Server Naming

- Use descriptive names: `gsas2`, `maud`, `pyfai`
- Lowercase with underscores: `my_tool_server`
- Add `_server.py` suffix

### 2. Organization

- Put new servers in `servers/` directory
- Keep core servers in root
- Use clear documentation

### 3. Dependencies

- Document required packages
- Use `uv add` for installation
- Keep dependencies minimal

### 4. Testing

- Test standalone first
- Test integration second
- Provide example data

### 5. Documentation

- Add to ADDING_NEW_SERVERS.md
- Include usage examples
- Document environment variables

## Troubleshooting

### Server Not Loading

```
✗ myserver (not found: servers/myserver.py)
```

**Fix:** Check file exists at specified path

### Import Errors

```
ModuleNotFoundError: No module named 'mytool'
```

**Fix:** `uv add mytool`

### Tool Not Appearing

```bash
Beamline> tools
# Tool missing
```

**Fix:** Check `@mcp.tool()` decorator present

## Summary

The modular architecture provides:

✓ **Easy extension** - Add tools without core changes
✓ **Flexible configuration** - Enable/disable as needed
✓ **Auto-detection** - Tools found automatically
✓ **Clean separation** - Independent servers
✓ **uv package management** - Fast, reliable dependencies
✓ **Future-proof** - Easy to add new capabilities

**Add a new analysis tool in 5 minutes!**

See [ADDING_NEW_SERVERS.md](ADDING_NEW_SERVERS.md) for complete guide.
