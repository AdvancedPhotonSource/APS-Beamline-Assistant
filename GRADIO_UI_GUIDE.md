# APEXA Gradio UI Guide

**Conversational AI Interface for Beamline Analysis**

---

## What is the Gradio UI?

A modern, chat-based interface for APEXA that replaces static forms with natural language conversation. Think of it as ChatGPT for your beamline data analysis.

## Quick Start

```bash
./start_gradio_ui.sh
```

Opens at: **http://localhost:7860**

## Features

### 💬 Natural Language Commands
No need to remember parameter names or button locations. Just ask:

```
"Calibrate the CeO2 image in test5 folder with stopping strain 0.003"
"Show me what files are in the current directory"
"What's a good calibration strain value?"
"Integrate using the refined parameters"
```

### 📁 Drag-and-Drop Upload
- Drop TIFF, GE, HDF5 files directly into the sidebar
- Files automatically copied to `uploads/` directory
- Reference them in chat by filename

### 📊 Embedded Visualizations
- Calibration plots appear inline in chat
- Ring overlays shown automatically
- Results formatted for easy reading

### 🎯 Example Prompts
Click pre-written examples to get started:
- Calibration workflows
- File exploration
- Knowledge base queries
- Integration tasks

### ⚡ Real-Time Progress
Watch as APEXA:
- Detects files
- Runs calibration iterations
- Processes integrations
- Searches knowledge base

---

## Interface Layout

```
┌─────────────────────────────────────────────────────────┐
│  APEXA - Advanced Photon EXperiment Assistant          │
├──────────────────────────┬──────────────────────────────┤
│  Chat Thread             │  Quick Actions Sidebar       │
│  ┌────────────────────┐  │  📁 Upload Files             │
│  │ User: calibrate... │  │  🔧 Available Tools          │
│  └────────────────────┘  │  📊 Status                   │
│  ┌────────────────────┐  │                              │
│  │ APEXA: Running...  │  │                              │
│  │ [Plot appears]     │  │                              │
│  │ ✓ Done!           │  │                              │
│  └────────────────────┘  │                              │
│                          │                              │
│  Example Prompts Below   │                              │
└──────────────────────────┴──────────────────────────────┘
```

---

## Example Workflows

### Calibration
```
You: "Calibrate the CeO2 image in test5 folder with stopping strain 0.003"

APEXA: "I'll run auto-calibration on the CeO2 image.

→ Filesystem List Directory
   Found: CeO2_650mm_61p332keV_2DFocused_0p1s_att200_004018.tif

→ MIDAS Auto Calibrate
   Running AutoCalibrateZarr.py...
   Iteration 1: strain=0.0015
   Iteration 2: strain=0.0008
   ...
   ✓ Converged! Mean strain: 0.0003

Results:
- Lsd: 651.12mm
- Beam Center: [702.87, 812.52]px
- Quality: Excellent (strain < 0.0005)

Saved to: refined_MIDAS_params.txt

Would you like to integrate using these parameters?"
```

### Knowledge Base Query
```
You: "What is a good calibration strain value?"

APEXA: "Let me search the knowledge base for calibration quality metrics.

→ Query HEDM Knowledge
   Searching papers and logbooks...

Based on HEDM-II.pdf:
'Calibration quality is typically assessed by mean strain values.
Excellent calibrations achieve strain < 0.0005, good calibrations
are < 0.001, and acceptable calibrations are < 0.003.'

From FF_autocalibrate.md logbook:
'For CeO2 at 61keV, we typically see converged strains around 0.0002-0.0004
with proper initial guesses. Higher values may indicate poor ring detection
or incorrect initial parameters.'

**Recommendation**: Aim for strain < 0.0005 for high-quality calibrations."
```

### File Exploration
```
You: "Show me files in test5"

APEXA: "

→ Filesystem List Directory
   Path: test5/

Files found:
1. CeO2_650mm_61p332keV_2DFocused_0p1s_att200_004018.tif (2.3 MB)
2. Parameters.txt (1.2 KB)
3. dark_CeO2_650mm.tif (2.3 MB)

This looks like a calibration dataset with:
- Calibrant image (CeO2)
- Parameter file
- Dark frame for background subtraction

Would you like me to run calibration on this data?"
```

---

## Tips for Effective Use

### Be Specific with Paths
❌ "calibrate the file"
✅ "calibrate the CeO2 image in test5 folder"

### Provide Context When Needed
❌ "integrate"
✅ "integrate data.tif using refined_MIDAS_params.txt"

### Ask Follow-up Questions
APEXA remembers conversation history:
```
You: "What files are in test5?"
APEXA: [shows list]
You: "calibrate the CeO2 one"  ← References previous response
```

### Use Natural Language
You don't need to format commands perfectly:
- "run calibration on the ceria image"
- "integrate the tiff file"
- "what's a good strain value for calibration?"

All work the same!

---

## Advantages Over Command Line

| Feature | Gradio UI | CLI |
|---------|-----------|-----|
| **Visuals** | ✅ Plots inline | ❌ Terminal only |
| **File Upload** | ✅ Drag & drop | ❌ Manual path |
| **History** | ✅ Scrollable chat | ⚠️ Limited scrollback |
| **Examples** | ✅ Clickable | ❌ Manual typing |
| **Multi-user** | ✅ Web accessible | ❌ SSH only |
| **Learning Curve** | ✅ Very low | ⚠️ Medium |

---

## Advantages Over Web UI (Forms)

| Feature | Gradio UI | Web UI |
|---------|-----------|--------|
| **Flexibility** | ✅ Any request | ⚠️ Fixed forms |
| **Discovery** | ✅ Ask what's possible | ❌ Must know UI |
| **Complexity** | ✅ Hidden until needed | ⚠️ All options shown |
| **AI Assistance** | ✅ Built-in | ⚠️ Separate chat |

---

## Troubleshooting

### "APEXA not initialized"
**Cause**: MCP servers failed to connect
**Fix**: Check `servers.config` exists and server paths are correct

### Slow Response Times
**Cause**: Large file processing or complex queries
**Solution**: This is normal. Watch for tool call progress indicators.

### "Tool result" Appearing in Chat
**Cause**: AI is showing raw tool output
**Fix**: This is intentional for transparency. The next message will summarize.

### Plots Not Showing
**Cause**: Plot generation failed or format not supported
**Fix**: Check logs in terminal for errors. Try different visualization request.

---

## Configuration

### Change Port
Edit `start_gradio_ui.sh`:
```python
demo.launch(server_port=7860)  # Change 7860 to desired port
```

### Enable Public Sharing
Edit `gradio_ui.py`:
```python
demo.launch(share=True)  # Creates public Gradio link
```

### Customize Theme
Edit `gradio_ui.py`:
```python
gr.Blocks(theme=gr.themes.Soft())  # Try: Base, Default, Glass, Monochrome
```

---

## Keyboard Shortcuts

- **Enter**: Send message (Shift+Enter for new line)
- **Ctrl+C**: Stop server (in terminal)
- **Up/Down**: Navigate example prompts

---

## Next Steps

1. **Try Example Prompts**: Click the examples below the chat box
2. **Upload Test Data**: Drag a TIFF or GE file to the sidebar
3. **Ask Questions**: "What can you do?" or "Explain FF-HEDM workflow"
4. **Run Calibration**: "Calibrate the CeO2 image in [your folder]"
5. **Explore Knowledge Base**: "What materials are used for calibration?"

---

## Architecture (For Developers)

```
User Browser (localhost:7860)
    ↓
Gradio UI (gradio_ui.py)
    ↓
ArgoMCPClient (argo_mcp_client.py)
    ↓ ↓ ↓
[filesystem_server] [executor_server] [midas_comprehensive_server]
    ↓
MIDAS Tools (AutoCalibrateZarr.py, integrator.py, etc.)
```

**Key Points**:
- Gradio provides the web interface
- ArgoMCPClient handles AI logic and tool orchestration
- MCP servers expose MIDAS functionality as tools
- Everything runs locally (no cloud dependencies)

---

**Questions?** Check [USER_MANUAL.md](USER_MANUAL.md) or ask APEXA directly!
