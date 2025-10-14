# Beamline Deployment Guide

This guide explains how to deploy the Beamline Assistant on a Linux beamline system where multiple users may need access.

## Prerequisites

- Linux system (tested on RHEL, CentOS, Ubuntu)
- Access to Argonne National Laboratory network
- MIDAS installed (typically at `~/.MIDAS` or `/opt/MIDAS`)
- Python 3.10+ with `uv` package manager

## Installation Locations

### MIDAS Installation

The system automatically searches for MIDAS in the following order:

1. `$MIDAS_PATH` environment variable (highest priority)
2. `~/.MIDAS` (recommended for beamline)
3. `~/MIDAS`
4. `~/opt/MIDAS` (macOS/development)
5. `/opt/MIDAS` (system-wide)
6. `./MIDAS` (current directory)

**Recommendation:** Install MIDAS at `~/.MIDAS` for each beamline user.

### Beamline Assistant

Clone to a shared or user-specific location:

```bash
# Option 1: User-specific installation
cd ~
git clone <repository-url> beamline-assistant
cd beamline-assistant

# Option 2: Shared installation (requires write permissions)
cd /opt
sudo git clone <repository-url> beamline-assistant
sudo chown -R beamline-users:beamline-users beamline-assistant
cd beamline-assistant
```

## Configuration for Each User

Each user needs their own `.env` file with their ANL credentials:

```bash
# Copy template
cp .env.template .env

# Edit with your credentials
nano .env
```

**.env file contents:**
```bash
# Your ANL username
ANL_USERNAME=your_anl_username

# AI model (optional, defaults to gpt4o)
ARGO_MODEL=gpt4o

# MIDAS path (optional, auto-detected if not set)
# MIDAS_PATH=~/.MIDAS
```

**Important:** Each user should create their own `.env` file. Never share credentials.

## Multi-User Setup

### Option 1: User-Specific Installations (Recommended)

Each user has their own copy:

```bash
# User 1
cd ~/beamline-assistant
cp .env.template .env
# Edit .env with user1 credentials
./start_beamline_assistant.sh

# User 2
cd ~/beamline-assistant
cp .env.template .env
# Edit .env with user2 credentials
./start_beamline_assistant.sh
```

**Advantages:**
- Each user has their own conversation history
- No permission conflicts
- Easy to customize per user

### Option 2: Shared Installation with User-Specific .env

Share code, but keep credentials separate:

```bash
# Shared installation at /opt/beamline-assistant

# User 1
cd /opt/beamline-assistant
cp .env.template ~/.beamline-assistant.env
nano ~/.beamline-assistant.env
# Create symlink
ln -sf ~/.beamline-assistant.env .env
./start_beamline_assistant.sh

# User 2
cd /opt/beamline-assistant
cp .env.template ~/.beamline-assistant.env
nano ~/.beamline-assistant.env
# Create symlink
ln -sf ~/.beamline-assistant.env .env
./start_beamline_assistant.sh
```

### Option 3: Environment Variables

Set environment variables in user's shell profile:

```bash
# Add to ~/.bashrc or ~/.bash_profile
export ANL_USERNAME=your_anl_username
export ARGO_MODEL=gpt4o
export MIDAS_PATH=~/.MIDAS
```

Then run without `.env` file:
```bash
cd /opt/beamline-assistant
./start_beamline_assistant.sh
```

## MIDAS Installation Per User

If each user needs their own MIDAS:

```bash
# Clone MIDAS
cd ~
git clone https://github.com/marinerhemant/MIDAS.git .MIDAS
cd .MIDAS

# Build MIDAS
mkdir build && cd build
cmake ..
make -j$(nproc)

# Verify installation
ls build/bin/  # Should show executables
```

The Beamline Assistant will automatically find MIDAS at `~/.MIDAS`.

## Startup

### Standard Startup
```bash
cd /path/to/beamline-assistant
./start_beamline_assistant.sh
```

### Custom Startup with Specific MIDAS Path
```bash
MIDAS_PATH=/custom/path/to/MIDAS ./start_beamline_assistant.sh
```

### Verify Startup
After starting, you should see:
```
======================================================================
  Starting Beamline Assistant with MIDAS Integration
======================================================================

MIDAS Path: /home/username/.MIDAS
Available Tools:
  - 20+ MIDAS analysis tools (FF-HEDM, NF-HEDM, PF-HEDM)
  - Filesystem operations
  - Command execution

AI Models available via Argo Gateway:
  - gpt4o (default)
  - claudesonnet4
  - gemini25pro

======================================================================

Found MIDAS installation at: /home/username/.MIDAS
✓ Connected to midas server with tools: [...]
✓ Connected to executor server with tools: [...]
✓ Connected to filesystem server with tools: [...]

Beamline Assistant - AI Diffraction Analysis
============================================================
Current AI Model: gpt4o
ANL User: username
Connected Servers: ['midas', 'executor', 'filesystem']

Commands: analyze, models, tools, servers, ls, run, clear, help, quit

Beamline>
```

## Network Requirements

The Beamline Assistant requires access to:

1. **Argo Gateway:** `https://apps.inside.anl.gov/argoapi/`
   - Must be on ANL network or VPN
   - Requires valid ANL credentials

2. **MIDAS Installation:** Local filesystem access

3. **Data Files:** Typically on beamline storage

### Testing Network Access

```bash
# Test Argo Gateway connectivity
curl -I https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/

# Should return HTTP 200 or redirect (not connection refused)
```

## Troubleshooting

### MIDAS Not Found

```
WARNING: MIDAS not found, using default: /home/username/.MIDAS
```

**Solutions:**
1. Install MIDAS at `~/.MIDAS`
2. Set `MIDAS_PATH` in `.env`
3. Create symlink: `ln -s /actual/path ~/.MIDAS`

### ANL Authentication Failed

```
Error calling Argo API: 401 Unauthorized
```

**Solutions:**
1. Verify `ANL_USERNAME` is correct in `.env`
2. Ensure you're on ANL network/VPN
3. Check Argo Gateway access permissions

### Permission Denied

```
Permission denied: .env
```

**Solutions:**
1. Check file ownership: `ls -l .env`
2. Fix permissions: `chmod 600 .env`
3. Ensure you own the file: `chown $USER .env`

### Tool Not Found

```
Tool 'integrate_2d_to_1d' not listed
```

**Solutions:**
1. Restart the assistant to reload servers
2. Check MIDAS installation is complete
3. Verify Python dependencies: `uv pip list`

## Security Best Practices

1. **Never commit .env files** - They contain credentials
2. **Use file permissions** - `chmod 600 .env` (owner read/write only)
3. **Separate credentials** - Each user has their own .env
4. **Audit access** - Monitor who uses the system
5. **Use VPN** - When accessing from outside ANL network

## Example Beamline Workflow

```bash
# User arrives at beamline
cd ~/beamline-assistant

# Start assistant
./start_beamline_assistant.sh

# Analyze experiment data
Beamline> List files in /data/2024-10/run_042
Beamline> Integrate the .tiff files from 2D to 1D
Beamline> Run FF-HEDM workflow on /data/2024-10/run_042

# Clear history before next experiment
Beamline> clear

# Exit
Beamline> quit
```

## Updating the System

```bash
# Shared installation
cd /opt/beamline-assistant
sudo git pull
sudo chown -R beamline-users:beamline-users .

# User installation
cd ~/beamline-assistant
git pull
```

## Support

For issues or questions:
1. Check this documentation
2. Review error messages carefully
3. Contact beamline support
4. File issues at: `<repository-url>/issues`
