# Beamline Assistant - Quick Start

## New User Setup (2 minutes)

### 1. Run Setup Script
```bash
cd beamline-assistant
./setup_user.sh
```

The script will ask for:
- Your ANL username
- Preferred AI model (default: gpt4o)
- MIDAS path (optional - auto-detected if not specified)

### 2. Start the Assistant
```bash
./start_beamline_assistant.sh
```

### 3. Start Analyzing!
```
Beamline> List files in /data/experiment_042
Beamline> Integrate the .tiff files from 2D to 1D
Beamline> Run FF-HEDM workflow on /data/experiment_042
```

## Requirements

- **Python:** 3.10+ (with `uv` package manager)
- **Network:** ANL network or VPN connection
- **MIDAS:** Auto-detected from standard locations (see below)

## MIDAS Auto-Detection

**MIDAS is auto-detected from these locations (in order):**

1. `$MIDAS_PATH` environment variable (highest priority)
2. `~/Git/MIDAS` (common for development)
3. `~/opt/MIDAS` (beamline recommended)
4. `/home/beams/S*USER/opt/MIDAS` (APS beamlines)
5. `~/MIDAS`
6. `/opt/MIDAS`
7. `~/.MIDAS`

**No configuration needed if MIDAS is in a standard location!**

## Manual Configuration

If you prefer to edit `.env` manually:

```bash
cp .env.template .env
nano .env
```

Set these variables:
```bash
ANL_USERNAME=your_anl_username
ARGO_MODEL=gpt4o
# MIDAS_PATH=~/.MIDAS  # Optional - only if auto-detection fails
```

## Commands

Once started, you can use:

| Command | Description |
|---------|-------------|
| `models` | Show available AI models |
| `model <name>` | Switch AI model |
| `tools` | List all analysis tools |
| `servers` | Show connected servers |
| `ls <path>` | List directory contents |
| `clear` | Clear conversation history |
| `help` | Show help |
| `quit` | Exit |

## Natural Language Queries

Just ask in plain English:

- "List files in /data/2024-10/run_042"
- "Integrate the .tiff file from 2D to 1D"
- "Run FF-HEDM workflow on this directory"
- "Read the Parameters.txt file there" (remembers context!)
- "What phases are in this sample?"

## Switching AI Models

```
Beamline> models
[Shows available models]

Beamline> model claudesonnet4
✓ Switched to: claudesonnet4
```

## Troubleshooting

### MIDAS Not Found
```
WARNING: MIDAS not found, using default: ~/.MIDAS
```
**Fix:** Install MIDAS or set `MIDAS_PATH` in `.env`

### Authentication Error
```
Error calling Argo API: 401 Unauthorized
```
**Fix:** Check ANL_USERNAME in `.env` and network connection

### Need Help?
```
Beamline> help
```

## Multi-User Environments

Each user should:
1. Run `./setup_user.sh` with their own ANL username
2. Keep their `.env` file private (chmod 600)
3. Never share credentials

The system supports multiple users with separate configurations and conversation histories.

## Advanced Usage

See detailed documentation:
- [BEAMLINE_DEPLOYMENT.md](BEAMLINE_DEPLOYMENT.md) - Full deployment guide
- [IMPROVEMENTS.md](IMPROVEMENTS.md) - Recent improvements
- [TOOL_FIX.md](TOOL_FIX.md) - Integration tool details

## Example Session

```bash
$ ./start_beamline_assistant.sh
Found MIDAS installation at: /home/username/.MIDAS
✓ Connected to midas server
✓ Connected to executor server
✓ Connected to filesystem server

Beamline> List files in /Users/b324240/opt/MIDAS/FF_HEDM/Example

→ Filesystem List Directory

Files in /Users/b324240/opt/MIDAS/FF_HEDM/Example:
- Parameters.txt (4.2 KB)
- ff_011276_ge2_0001.tiff (8.4 MB)
- GrainsSim.csv (125 KB)

Beamline> Read the Parameters.txt file there

→ Filesystem Read File

The Parameters.txt file contains detailed configuration parameters...

Beamline> Integrate the .tiff file from 2D to 1D

→ Filesystem List Directory
→ Integrate 2D To 1D

Successfully integrated ff_011276_ge2_0001.tiff to 1D pattern
- Output: ff_011276_ge2_0001_1d.dat
- Detected 8 peaks
- Signal-to-noise ratio: 45.2

Beamline> quit
Goodbye!
```

## Security Note

Your `.env` file contains your ANL credentials. Keep it secure:
- Never commit to git (already in .gitignore)
- Set permissions: `chmod 600 .env`
- Don't share with others
- Each user should have their own
