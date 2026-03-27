# APEXA - Quick Start

## Setup (2 minutes)

### 1. Run Setup Script
```bash
cd beamline-assistant
./setup_user.sh
```

The script will ask for:
- Your ANL username
- Preferred AI model (default: gpt4o)
- MIDAS path (optional -- auto-detected)

### 2. Start APEXA
```bash
./start_beamline_assistant.sh
```

You should see:
```
  core: 9 tools
  midas: 21 tools

  APEXA - Advanced Photon EXperiment Assistant
  Model: gpt4o (PROD)  |  30 tools  |  Servers: core, midas
```

### 3. Start Analyzing
```
APEXA> list files in /data/experiment
APEXA> calibrate the CeO2 image
APEXA> integrate the .tif file
APEXA> show me the lineout
```

## Requirements

- **Python:** 3.13+ (with `uv` package manager)
- **Network:** ANL network or VPN
- **MIDAS:** v11, auto-detected from standard locations

## MIDAS Auto-Detection

MIDAS is found automatically from (in order):

1. `$MIDAS_PATH` environment variable
2. `~/Git/MIDAS`
3. `~/opt/MIDAS`
4. `/home/beams/S*USER/opt/MIDAS` (beamline systems)
5. `~/MIDAS`
6. `/opt/MIDAS`
7. `~/.MIDAS`

No configuration needed if MIDAS is in a standard location.

## Manual Configuration

Edit `.env` directly:

```bash
cp .env.template .env
nano .env
```

```bash
ANL_USERNAME=your_anl_username
ARGO_MODEL=gpt4o                    # or claudesonnet4, gemini25pro
# MIDAS_PATH=~/Git/MIDAS            # only if auto-detection fails
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `models` | Show available AI models |
| `model <name>` | Switch AI model |
| `timing` | Toggle API response time display |
| `tools` | List all analysis tools |
| `servers` | Show connected servers |
| `ls <path>` | List directory |
| `clear` | Clear conversation history |
| `help` | Show help |
| `quit` | Exit |

## Natural Language Examples

Just ask in plain English:

```
APEXA> calibrate the CeO2 image in test1
APEXA> integrate CeO2_000001.tif using refined parameters
APEXA> show me the lineout for CeO2
APEXA> convert 61.332 keV to wavelength
APEXA> what is the d-spacing for Fe (110)?
APEXA> run FF-HEDM workflow on /data/experiment
APEXA> explain what beam center calibration does
```

## Switching AI Models

```
APEXA> models
[Shows available models]

APEXA> model claudesonnet4
Switched to: claudesonnet4 (using DEV environment)
```

## Troubleshooting

**MIDAS Not Found:**
```
WARNING: MIDAS not found
```
Fix: Set `MIDAS_PATH` in `.env`

**Authentication Error:**
```
Error calling Argo API: 401 Unauthorized
```
Fix: Check `ANL_USERNAME` in `.env` and network connection

## Multi-User Environments

Each user should:
1. Run `./setup_user.sh` with their own ANL username
2. Keep `.env` private (`chmod 600 .env`)

---

See [USER_MANUAL.md](../../USER_MANUAL.md) for complete documentation.
