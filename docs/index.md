 # APEXA - Advanced Photon EXperiment Assistant

AI-powered beamline scientist for real-time HEDM data analysis at Argonne National Laboratory's Advanced Photon Source.

---

## 🚀 Quick Start

### Command-Line Interface (CLI)
```bash
./setup_user.sh                  # One-time setup
./start_beamline_assistant.sh    # Start APEXA CLI
```

### Web User Interface (Web UI)
```bash
./start_web_viewer.sh            # Start Web UI
```
Then open: **http://localhost:8001**

**That's it!** Natural language interface ready for:
- ✅ Detector calibration (CeO2, LaB6, Si)
- ✅ 2D→1D integration with dark subtraction
- ✅ FF-HEDM grain reconstruction
- ✅ NF-HEDM microstructure mapping
- ✅ Phase identification
- ✅ Batch processing
- ✅ Real-time monitoring

---

## 📖 Documentation

### 🌐 Interactive Documentation Site
For the best experience, view our **searchable documentation website**:
```bash
./serve_docs.sh  # Opens at http://localhost:8000
```

Features:
- 🔍 Full-text search across all docs
- 📱 Mobile-friendly responsive design
- 🌓 Dark/light mode toggle
- 📑 Organized navigation and table of contents

### For Users
- **[USER_MANUAL.md](USER_MANUAL.md)** - Complete guide with examples and tutorials
- **[WEB_UI_GUIDE.md](WEB_UI_GUIDE.md)** - Browser-based interface for demos and collaboration
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command cheat sheet for demos

### For Developers & Advanced Users
- **[APEXA_SMART_FEATURES_MANUAL.md](APEXA_SMART_FEATURES_MANUAL.md)** - Smart features and AI capabilities
- **[MIDAS_WORKFLOWS_REFERENCE.md](MIDAS_WORKFLOWS_REFERENCE.md)** - Technical MIDAS workflow details

---

## 💬 Example Usage

```
APEXA> calibrate using the CeO2 image in this directory
→ Auto Calibrate
✓ Refined parameters saved to refined_MIDAS_params.txt

APEXA> integrate sample.tif using those parameters
→ Integrate 2D To 1D
✓ Output: sample.dat

APEXA> identify phases from the peaks
→ Phase Identification
Detected phases: Ti (α-phase), TiO₂ (rutile)

APEXA> run FF-HEDM workflow on /data/experiment
→ FF-HEDM Full Workflow
✓ Found 2,347 grains
```

---

## 🎯 Key Features

### Smart & Conversational
No need to memorize commands - just describe what you want:
- "Calibrate using the ceria file"
- "Integrate with dark subtraction"
- "Run FF-HEDM with 32 CPUs"

### Context-Aware
Remembers your session:
- Previous files and directories
- Analysis history
- Conversation context

### Proactive
Suggests next steps after each analysis:
- "📊 Suggested next steps: Integrate rings to 1D pattern"
- Auto-validates parameters before execution
- Real-time alerts during beamtime

See [APEXA_SMART_FEATURES_MANUAL.md](APEXA_SMART_FEATURES_MANUAL.md) for details on all smart features.

### Extensible
Add new analysis tools in 5 minutes - see USER_MANUAL.md for details.

---

## 🛠️ System Architecture

```
User → Argo-AI (GPT-4o/Claude/Gemini)
         ↓
    MCP Client (argo_mcp_client.py)
         ↓
   ┌─────────┬─────────┬─────────┐
   │filesystem│  midas  │ executor│
   └─────────┴─────────┴─────────┘
```

---

## ⚙️ Configuration

**User Settings** (`.env`):
```bash
ANL_USERNAME=your_username
ARGO_MODEL=gpt4o              # or claudesonnet4, gemini25pro
MIDAS_PATH=~/Git/MIDAS        # Optional - auto-detected
```

**Server Configuration** (`servers.config`):
```bash
filesystem:filesystem_server.py
executor:command_executor_server.py
midas:midas_comprehensive_server.py
```

---

## 📋 Requirements

- **Python:** 3.10+
- **Package Manager:** [uv](https://github.com/astral-sh/uv)
- **Network:** ANL access for Argo-AI
- **MIDAS:** Installed with conda environment
- **Memory:** 16+ GB RAM (64+ GB recommended for FF-HEDM)

---

## 🔧 Troubleshooting

**MIDAS not detected?**

APEXA automatically searches for MIDAS in this order:
1. `$MIDAS_PATH` environment variable
2. `~/Git/MIDAS`
3. `~/opt/MIDAS`
4. `/home/beams/S*USER/opt/MIDAS` (beamline systems)
5. `~/MIDAS`
6. `/opt/MIDAS`
7. `~/.MIDAS`

To override auto-detection:
```bash
export MIDAS_PATH=/path/to/MIDAS
```

**Tool warnings?**
Restart the assistant - warnings are cosmetic.

**Need help?**
```
APEXA> help
APEXA> what can you do?
APEXA> how do I calibrate?
```

See [USER_MANUAL.md](USER_MANUAL.md#troubleshooting) for detailed troubleshooting.

---

## 🎓 Credits

**Development:**
- Pawan Tripathi - Lead Developer
- Advanced Photon Source, Argonne National Laboratory

**Core Dependencies:**
- [MIDAS](https://github.com/marinerhemant/MIDAS) - Hemant Sharma
- [FastMCP](https://github.com/jlowin/fastmcp) - Marvin
- [uv](https://github.com/astral-sh/uv) - Astral
- Argo Gateway - Argonne National Laboratory

---

## 📄 License

Copyright © 2024 UChicago Argonne, LLC  
See [LICENSE](LICENSE) for details.

---

**Ready to analyze? Start with:**
```bash
./start_beamline_assistant.sh
```

**Documentation Map:**
- **New user?** → [User Manual](USER_MANUAL.md)
- **Demo/presentation?** → [Quick Reference](QUICK_REFERENCE.md)
- **MIDAS expert?** → [MIDAS Workflows](MIDAS_WORKFLOWS_REFERENCE.md)
- **Want advanced features?** → [Smart Features](APEXA_SMART_FEATURES_MANUAL.md)
