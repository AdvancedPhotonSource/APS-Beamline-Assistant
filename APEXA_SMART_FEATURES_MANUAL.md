# APEXA Smart Features Manual
## Advanced Photon EXperiment Assistant

**Version:** 1.0
**Author:** Pawan Tripathi
**Organization:** Argonne National Laboratory - Advanced Photon Source

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Session Persistence](#session-persistence)
3. [Proactive Analysis Suggestions](#proactive-analysis-suggestions)
4. [Batch Processing](#batch-processing)
5. [Error Prevention & Validation](#error-prevention--validation)
6. [Smart Caching](#smart-caching)
7. [Workflow Builder](#workflow-builder)
8. [Smart Context Management](#smart-context-management)
9. [Complete Workflow Examples](#complete-workflow-examples)
10. [Advanced Features](#advanced-features)

---

## 1. Introduction

APEXA (Advanced Photon EXperiment Assistant) is an AI-powered scientist that assists with synchrotron X-ray diffraction analysis at the Advanced Photon Source. Unlike traditional tools that simply execute commands, APEXA:

- **Remembers** your experiment context across sessions
- **Suggests** smart next steps after each analysis
- **Learns** from your workflow patterns
- **Guides** you through complex multi-step analyses
- **Automates** repetitive tasks

Think of APEXA as your AI colleague who's always watching your back!

---

## 2. Session Persistence

### 2.1 What is Session Persistence?

APEXA automatically tracks and saves your entire experimental workflow, so you can:
- Resume work after a break
- Share experiments with colleagues
- Review what you did last week
- Document your methods for publications

### 2.2 What Gets Saved?

Every session automatically tracks:

| **Category** | **Details** |
|--------------|-------------|
| **Sample Info** | Sample name, beamline, experiment ID |
| **Analysis History** | Every tool used, with timestamps |
| **Key Findings** | Important discoveries marked during analysis |
| **Active Files** | Files and directories you're working with |
| **Context** | Current working directory, user, model used |

### 2.3 Session Commands

#### Auto-Save (Automatic)
APEXA automatically saves your context as you work. No manual intervention needed!

```python
# Session is auto-saved after each analysis
APEXA> identify phases from peaks at 12.5, 18.2, 25.8 degrees
# ✓ Analysis recorded in session
```

#### Manual Save (On Demand)
```python
APEXA> save session as "ni-alloy-thermal-study"
✓ Session saved to: ~/.apexa/sessions/ni-alloy-thermal-study.json
```

#### Load Previous Session
```python
APEXA> load session "ni-alloy-thermal-study"
✓ Loaded session from 2025-10-14
✓ Sample: Ni-Alloy-Test-01
✓ 15 analyses previously performed
```

#### List All Sessions
```python
APEXA> list sessions
Available sessions:
  1. ni-alloy-thermal-study (2025-10-14 14:23)
  2. steel-deformation-series (2025-10-13 09:15)
  3. ceramic-calibration (2025-10-12 16:45)
```

### 2.4 Session Storage

**Location:** `~/.apexa/sessions/`

**Format:** JSON (human-readable, can be edited)

**Example session file:**
```json
{
  "experiment_id": "APS-2025-10-14",
  "sample_name": "Ni-Alloy-Test-01",
  "beamline": "6-ID-D",
  "start_time": "2025-10-14T14:23:15",
  "user": "ptripathi",
  "analysis_history": [
    {
      "timestamp": "2025-10-14T14:25:30",
      "type": "midas_identify_crystalline_phases",
      "result": "Found austenite (γ-Fe) FCC phase..."
    }
  ],
  "key_findings": [
    {
      "timestamp": "2025-10-14T14:30:00",
      "finding": "Austenite stable at room temperature"
    }
  ]
}
```

### 2.5 Example: Resuming After Lunch

**Morning Session:**
```
APEXA> I'm analyzing Ni-alloy sample 42 from beamline 6-ID-D
APEXA> detect rings in ./data/sample42.tiff
APEXA> identify phases from peaks
# Found austenite + ferrite

[You go to lunch...]
```

**Afternoon Session:**
```
APEXA> what was I working on?
📋 Current Experiment:
   Sample: Ni-alloy sample 42
   Beamline: 6-ID-D
   Analyses: 3 (ring detection, phase ID, integration)
   Last finding: Found austenite + ferrite phases

APEXA> continue tracking these grains through temperature
# APEXA remembers your sample and context!
```

---

## 3. Proactive Analysis Suggestions

### 3.1 What Are Proactive Suggestions?

After every analysis, APEXA automatically suggests **scientifically relevant next steps** based on:
- What analysis you just performed
- The results obtained
- Common scientific workflows
- Best practices in the field

### 3.2 When Do Suggestions Appear?

Suggestions appear automatically after these analyses:

| **Analysis Type** | **When Triggered** |
|-------------------|-------------------|
| Phase Identification | After identifying phases |
| Ring Detection | After detecting diffraction rings |
| FF-HEDM Reconstruction | After grain reconstruction |
| 2D to 1D Integration | After pattern integration |
| Grain Tracking | After tracking analysis |

### 3.3 Examples by Analysis Type

#### Example 1: After Phase Identification

**Your Query:**
```
APEXA> identify phases from peaks at 12.5, 18.2, 25.8 degrees
```

**APEXA Response:**
```
✓ Identified Phase: Austenite (γ-Fe)
  Crystal Structure: FCC
  Space Group: Fm-3m (225)

Peak Matching:
| Observed 2θ | Calculated 2θ | (hkl) |
|------------|---------------|-------|
| 12.5       | 12.47        | (111) |
| 18.2       | 18.15        | (200) |
| 25.8       | 25.84        | (220) |

📊 **Suggested next steps:**
• Quantify phase fraction using Rietveld refinement
• Check for preferred orientation (texture analysis)
• Calculate lattice parameters and compare to literature
```

#### Example 2: After Ring Detection

**Your Query:**
```
APEXA> detect rings in sample_0001.tiff
```

**APEXA Response:**
```
✓ Detected 8 diffraction rings
  Ring radii: [245, 356, 435, 612, 701, 845, 923, 1012] pixels
  Quality: Excellent (circular, sharp)

📊 **Suggested next steps:**
• Integrate rings to 1D pattern for phase ID
• Check calibration quality (ring circularity)
• Perform full FF-HEDM reconstruction
```

#### Example 3: After FF-HEDM Reconstruction

**Your Query:**
```
APEXA> run FF-HEDM workflow on ./reconstruction_data/
```

**APEXA Response:**
```
✓ FF-HEDM Reconstruction Complete
  Grains found: 1,247
  Completeness: 94.3%
  Processing time: 2.3 hours

📊 **Suggested next steps:**
• Analyze grain size distribution (1,247 grains found)
• Calculate grain orientations and texture
• Track grains through deformation series (if applicable)
• Export to DREAM.3D for visualization
• Calculate misorientation statistics
```

#### Example 4: After 2D to 1D Integration

**Your Query:**
```
APEXA> integrate pattern from sample_0001.tiff
```

**APEXA Response:**
```
✓ Integration Complete
  Output: sample_0001_1D.chi
  Q-range: 0.5 - 8.2 Å⁻¹
  Peak count: 12 strong reflections

📊 **Suggested next steps:**
• Identify phases from peak positions
• Perform Rietveld refinement
• Check for peak splitting (sample stress/strain)
• Compare with reference patterns
```

### 3.4 Following Suggestions

You can directly respond to suggestions:

```
APEXA> identify phases from peaks at 12.5, 18.2, 25.8 degrees
# ... phase identification results ...
# ... suggestions appear ...

APEXA> yes, check for texture
# APEXA understands and runs texture analysis!
```

Or pick specific suggestions:

```
APEXA> do the second suggestion
# APEXA runs the second suggested analysis
```

### 3.5 Disabling Suggestions (Optional)

If you prefer no suggestions:

```
APEXA> disable suggestions
✓ Proactive suggestions disabled
```

To re-enable:
```
APEXA> enable suggestions
✓ Proactive suggestions enabled
```

---

## 4. Batch Processing

### 4.1 What is Batch Processing?

Process dozens or hundreds of diffraction images automatically with a single command! APEXA's batch processor handles:
- Multiple files with wildcard patterns
- Dark file subtraction for all images
- Progress tracking and error handling
- Summary reports of success/failure

### 4.2 Batch Integration Command

**Syntax:**
```bash
batch integrate <pattern> with <calibration_file> [dark_file]
```

**Example 1: Process all GE5 files**
```
APEXA> batch integrate *.ge5 with calib.txt dark.ge5
Found 147 files to process
Process all 147 files? (yes/no): yes

[1/147] Processing: data_0001.ge5
→ Integrate 2D To 1D
✓ Success

[2/147] Processing: data_0002.ge5
→ Integrate 2D To 1D
✓ Success

... (progress continues)

============================================================
Batch Processing Complete:
  Total: 147
  ✓ Successful: 145
  ✗ Failed: 2
============================================================
```

**Example 2: Process specific range**
```bash
# Unix systems
APEXA> batch integrate data_00[0-9][0-9].ge5 with calib.txt

# Processes: data_0001.ge5 through data_0099.ge5
```

**Example 3: With dark subtraction**
```
APEXA> batch integrate sample_*.tiff with geometry.txt dark_avg.tiff
Found 50 files to process
Process all 50 files? (yes/no): yes

... processing with automatic dark subtraction ...
```

### 4.3 Batch Results

Results for each file are saved as:
- `filename_1d.dat` - Integrated 1D pattern
- Batch summary in terminal
- Individual errors logged for failed files

### 4.4 Use Cases

**Time Series Analysis:**
```bash
# Process entire in-situ heating experiment
batch integrate heating_*K.ge5 with calib.txt dark.ge5
# Processes: heating_300K.ge5, heating_400K.ge5, ..., heating_1200K.ge5
```

**Deformation Series:**
```bash
# Process compression experiment at different strains
batch integrate strain_*.tiff with setup.txt
```

**Multiple Samples:**
```bash
# Process all samples from a batch
batch integrate sample_*.ge2 with calibration.txt dark_combined.ge2
```

---

## 5. Error Prevention & Validation

### 5.1 What is Error Prevention?

APEXA validates all parameters BEFORE executing expensive operations. This saves time and prevents:
- Running analysis on non-existent files
- Using wrong file formats
- Missing required parameters
- Invalid parameter values

### 5.2 Validation Checks

**File Existence:**
```
APEXA> integrate missing_file.ge5 with calib.txt
✗ Validation Error: Image file not found: missing_file.ge5
💡 Suggestion: Please check your parameters and try again
```

**File Format Validation:**
```
APEXA> integrate data.txt with calib.txt
✗ Validation Error: Unsupported image format. Use: .tif, .tiff, .ge2, .ge5, .ed5, .edf
💡 Suggestion: Please check your parameters and try again
```

**Missing Parameters:**
```
APEXA> integrate data.ge5
✗ Validation Error: Either calibration_file OR all manual parameters (wavelength, detector_distance, beam_center_x, beam_center_y) must be provided
💡 Suggestion: Please check your parameters and try again
```

**Parameter Range Validation:**
```
APEXA> integrate data.ge5 with wavelength -0.5 distance 1000 center 1024 1024
✗ Validation Error: Wavelength must be positive, got: -0.5
💡 Suggestion: Please check your parameters and try again
```

**Dark File Dimension Mismatch:**
```
APEXA> integrate data_2048x2048.ge5 with calib.txt dark_1024x1024.tiff
✗ Validation Error: Image and dark file dimensions don't match: (2048, 2048) vs (1024, 1024)
💡 Suggestion: Please check your parameters and try again
```

### 5.3 FF-HEDM Validation

**Directory Validation:**
```
APEXA> run FF-HEDM on /nonexistent/path
✗ Validation Error: Directory not found: /nonexistent/path
💡 Suggestion: Please check your parameters and try again
```

**Missing Parameters.txt:**
```
APEXA> run FF-HEDM on ./my_data/
✗ Validation Error: Parameters.txt not found in ./my_data/
💡 Suggestion: Please check your parameters and try again
```

### 5.4 Benefits

✅ **Fail Fast:** Errors caught immediately, not after hours of processing
✅ **Clear Messages:** Know exactly what's wrong and how to fix it
✅ **Time Saved:** No wasted beamtime on incorrect setups
✅ **Confidence:** Guaranteed valid operations

---

## 6. Smart Caching

### 6.1 What is Smart Caching?

APEXA automatically caches expensive read operations to speed up repeated queries and reduce AI costs.

### 6.2 What Gets Cached?

**File System Operations:**
- Directory listings
- File reads
- Parameter file contents

**Behavior:**
```
APEXA> list files in /data/experiment
→ Filesystem List Directory
... (takes 0.5s) ...

APEXA> list files in /data/experiment
→ Filesystem List Directory (from cache)
... (instant!) ...
```

### 6.3 Cache Location

- **Memory Cache:** Fast, session-only
- **Disk Cache:** `~/.apexa/cache/` - persists between sessions

### 6.4 Cache Invalidation

Cache automatically invalidates when:
- Files are modified
- 24 hours have passed
- APEXA is restarted

Manual cache clear:
```
APEXA> clear cache
✓ Cache cleared
```

### 6.5 Benefits

⚡ **Faster:** Repeated operations are instant
💰 **Cost Savings:** Fewer AI API calls
🔋 **Efficiency:** Less network traffic

---

## 7. Workflow Builder

### 7.1 What are Workflows?

Pre-defined sequences of analysis steps for common experimental scenarios. Think of them as "recipes" for analysis.

### 7.2 Available Workflows

**List all workflows:**
```
APEXA> workflow list

Available Workflows:
==================================================

phase_analysis:
  1. Integrate 2D image to 1D pattern
  2. Identify phases from peaks

full_hedm:
  1. Check data directory
  2. Run FF-HEDM reconstruction

calibration_check:
  1. Detect rings for calibration
  2. Integrate to verify calibration
```

### 7.3 Using Workflows

**Natural Language (Recommended):**
```
APEXA> run phase analysis workflow on sample_001.ge5
✓ Step 1: Integrating 2D image to 1D pattern
✓ Step 2: Identifying phases from peaks
✓ Workflow complete! Found: γ-Fe (austenite)
```

**Direct Command:**
```
APEXA> workflow phase_analysis

Executing workflow: phase_analysis
==================================================

Step 1: Integrate 2D image to 1D pattern
  Tool: midas_integrate_2d_to_1d

Step 2: Identify phases from peaks
  Tool: midas_identify_crystalline_phases

Note: Use natural language queries to execute workflows with your data
```

### 7.4 Workflow Suggestions

APEXA automatically suggests workflows:

```
APEXA> I want to calibrate my detector
💡 Suggested workflow: calibration_check
  1. Detect diffraction rings
  2. Integrate to verify calibration

Would you like to run this workflow? (yes/no):
```

### 7.5 Custom Workflows

*Coming soon:* Define your own workflows!

---

## 8. Smart Context Management

### 4.1 What is Smart Context?

APEXA maintains a "mental model" of your experiment:
- Files you're working with
- Previous analyses performed
- Sample characteristics discovered
- Current experimental goals

This allows natural conversations like:

```
APEXA> list files here
# APEXA sees: sample_0001.tiff, Parameters.txt, ...

APEXA> detect rings in that first tiff
# APEXA knows "that first tiff" = sample_0001.tiff

APEXA> what parameters are being used?
# APEXA knows to read Parameters.txt
```

### 4.2 Context-Aware Commands

#### Referencing Previous Results
```
APEXA> identify phases from peaks at 12.5, 18.2 degrees
# Found: Austenite

APEXA> calculate the lattice parameter for that phase
# APEXA knows "that phase" = Austenite
```

#### Remembering Files
```
APEXA> ls ./data/
# Shows: run001.tiff, run002.tiff, run003.tiff

APEXA> integrate all those tiff files
# APEXA remembers the list and processes all 3
```

#### Directory Context
```
APEXA> cd /path/to/experiment/
APEXA> what's in the parameters file?
# APEXA looks in current directory

APEXA> run workflow here
# Uses files in /path/to/experiment/
```

### 4.3 Experiment Metadata

Set metadata to help APEXA understand your experiment:

```
APEXA> set sample name to "Ti-6Al-4V-Tensile-Test"
✓ Sample name updated

APEXA> set beamline to "6-ID-D"
✓ Beamline updated

APEXA> set experiment ID to "APS-2025-3-User-Proposal-12345"
✓ Experiment ID updated
```

Later, APEXA uses this in suggestions:

```
APEXA> You're analyzing Ti-6Al-4V-Tensile-Test from beamline 6-ID-D.
       Would you like to check for α/β phase fractions typical in Ti alloys?
```

---

## 5. Workflow Examples

### 5.1 Complete Workflow: Unknown Sample Phase ID

**Scenario:** You have a mysterious sample and need to identify what it is.

```
# Step 1: Start session
APEXA> I have an unknown steel sample, need to identify phases

APEXA> Sample noted! Let's start with diffraction analysis.

# Step 2: Detect rings
APEXA> detect rings in ./data/unknown_sample.tiff

APEXA> ✓ Detected 6 diffraction rings
       📊 Suggested: Integrate to 1D for phase ID

# Step 3: Follow suggestion
APEXA> yes, integrate to 1D

APEXA> ✓ Integration complete
       📊 Suggested: Identify phases from peaks

# Step 4: Phase ID
APEXA> do that

APEXA> ✓ Found: Ferrite (α-Fe, BCC) + Austenite (γ-Fe, FCC)
       📊 Suggested: Quantify phase fractions

# Step 5: Quantification
APEXA> quantify the phases

APEXA> ✓ Phase fractions:
       - Ferrite: 73%
       - Austenite: 27%

# Step 6: Save for publication
APEXA> save session as "unknown-steel-analysis"
APEXA> export results as publication table

APEXA> ✓ Session saved
       ✓ Results exported to: steel_analysis_table.csv
```

### 5.2 Multi-Day Experiment: Grain Tracking

**Day 1: Initial Reconstruction**
```
APEXA> Starting Ti-alloy deformation study at 6-ID-D

APEXA> run FF-HEDM on ./as-received/

APEXA> ✓ 842 grains reconstructed
       ✓ Session auto-saved

[End of Day 1]
```

**Day 2: First Deformation Step**
```
APEXA> load yesterday's session

APEXA> ✓ Loaded: Ti-alloy deformation study
       ✓ Last step: 842 grains reconstructed

APEXA> run FF-HEDM on ./5percent-strain/

APEXA> ✓ 839 grains found

APEXA> track grains from as-received to 5percent-strain

APEXA> ✓ Tracked: 98.6% success rate
       📊 Suggested: Analyze grain rotations

[Automatically saved]
```

**Day 3: Continue Series**
```
APEXA> continue tracking to 10 percent strain

APEXA> ✓ Loaded previous tracking data
       ✓ Processing 10% strain dataset...

# APEXA remembers entire deformation series!
```

### 5.3 Batch Processing: Temperature Series

```
APEXA> I have 20 datasets from 25°C to 500°C in 25°C steps

APEXA> Great! Would you like me to:
       1. Process all automatically
       2. Show me the workflow first
       3. Process one as example

APEXA> process all automatically

APEXA> ✓ Batch processing started...
       [1/20] 25°C: Phase ID → Austenite
       [2/20] 50°C: Phase ID → Austenite
       ...
       [15/20] 375°C: Phase ID → Austenite + Ferrite ⚠️
       ...
       [20/20] 500°C: Phase ID → Ferrite

       📊 Key Finding: Phase transformation at ~375°C
       📊 Suggested: Plot phase fraction vs temperature

APEXA> yes please

APEXA> ✓ Plot saved: phase_vs_temperature.png
```

---

## 6. Advanced Features

### 6.1 Natural Language Workflows

Describe what you want in plain English:

```
APEXA> For each file in ./temperature-series/:
       1. Detect rings
       2. If more than 8 rings, run FF-HEDM
       3. Otherwise, just do phase ID
       4. Save results to a summary table

APEXA> ✓ Understood! Processing 15 files...
       ✓ 12 files → FF-HEDM (>8 rings)
       ✓ 3 files → Phase ID only
       ✓ Results: temperature_series_summary.csv
```

### 6.2 Smart Error Recovery

APEXA catches and fixes common mistakes:

```
APEXA> run FF-HEDM on ./data/
⚠️  Warning: No Parameters.txt found
    Do you want me to:
    1. Create one from template
    2. Use parameters from similar previous run
    3. Guide you through manual setup

APEXA> use parameters from yesterday

APEXA> ✓ Using parameters from: Ti-alloy-day1
       ✓ FF-HEDM started...
```

### 6.3 Multi-Model Support

Use different AI models for different tasks:

```
# Fast model for simple tasks
APEXA> model gpt4o
APEXA> list files here
# Quick response

# Smart model for complex analysis
APEXA> model claudesonnet45
APEXA> analyze this complex microstructure and suggest experiments
# Detailed, thoughtful response
```

### 6.4 Collaborative Features

Share context with team members:

```
APEXA> export session for collaborator

APEXA> ✓ Session package created: experiment_share.zip
       Contains:
       - Session metadata
       - Analysis history
       - Key findings
       - File references

# Colleague loads it:
APEXA> import session experiment_share.zip
APEXA> what did they find?

APEXA> Previous user (ptripathi) found:
       - Austenite phase at room temp
       - 842 grains reconstructed
       - Grain size: 45 μm average
```

### 6.5 Publication-Ready Output

```
APEXA> generate methods section

APEXA> **Methods:**
       Synchrotron X-ray diffraction data were collected at
       beamline 6-ID-D of the Advanced Photon Source
       (Argonne National Laboratory). The sample (Ti-6Al-4V)
       was analyzed using far-field high-energy diffraction
       microscopy (FF-HEDM) with a photon energy of 71.7 keV
       (λ = 0.173 Å)...

APEXA> export all figures with captions

APEXA> ✓ Exported:
       - Figure1_diffraction_rings.png
       - Figure2_phase_map.png
       - Figure3_grain_orientations.png
       ✓ Captions saved: figure_captions.txt
```

---

## 7. Tips & Best Practices

### 7.1 Naming Conventions

Good session names help future you:

```
✓ Good: "ni-alloy-thermal-expansion-2025-10"
✓ Good: "steel-deformation-series-run3"
✗ Bad: "test"
✗ Bad: "data"
```

### 7.2 Regular Saves

For long experiments, save periodically:

```
APEXA> save checkpoint

APEXA> ✓ Checkpoint saved: session_20251014_1430
```

### 7.3 Use Metadata

Set experiment info early:

```
APEXA> set sample name to "Sample-ID-12345"
APEXA> set beamline to "6-ID-D"
APEXA> set experiment ID to "APS-2025-3-12345"
```

### 7.4 Follow Suggestions

APEXA's suggestions are based on scientific best practices. Following them helps ensure complete analysis!

---

## 8. Troubleshooting

### Session not loading?

```
APEXA> list sessions
# Check if session name is correct

APEXA> load session "exact-name-here"
```

### Suggestions not appearing?

```
APEXA> are suggestions enabled?

APEXA> enable suggestions
```

### Want to start fresh?

```
APEXA> new session
# Starts clean session (old one auto-saved)
```

---

## 9. Contact & Support

**Questions?** Contact: ptripathi@anl.gov

**Issues?** GitHub: https://github.com/AdvancedPhotonSource/APS-Beamline-Assistant

**Documentation:** This manual + inline help (`APEXA> help`)

---

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-15 | Initial release with session persistence and proactive suggestions |

---

**APEXA** - Making your beamtime more productive, one suggestion at a time! 🚀
