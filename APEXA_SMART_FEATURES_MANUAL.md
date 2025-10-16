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
9. [Multimodal Image Analysis](#multimodal-image-analysis)
10. [Real-time Feedback During Beamtime](#real-time-feedback-during-beamtime)
11. [Advanced Plotting & Visualization](#advanced-plotting--visualization)
12. [Troubleshooting](#troubleshooting)

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

## 9. Multimodal Image Analysis

### 9.1 What is Multimodal Analysis?

APEXA can "see" and understand your diffraction images! No need to describe what you see - just point APEXA at the image and it will analyze:
- Image quality (signal, noise, saturation)
- Diffraction ring detection
- Hot pixel detection
- Overall assessment

### 9.2 Quick Image Check

**Check image quality before running expensive analysis:**
```
APEXA> image quality /data/sample_001.ge5

🔍 Quality check: sample_001.ge5
  Overall Quality: Excellent
  Signal-to-Noise: 45.3
  Saturation: 0.02%
```

**Detect rings to verify calibration:**
```
APEXA> image rings /data/CeO2_calibration.ge5

🔍 Ring detection: CeO2_calibration.ge5
  Rings Detected: 12
  Ring Radii (pixels): [245, 356, 435, 612, 701, 845, ...]
  Assessment: Good
```

**Full analysis with AI summary:**
```
APEXA> image analyze /data/sample_001.ge5

📸 Image Analysis: sample_001.ge5

Image Properties:
  Dimensions: (2048, 2048)
  Signal-to-Noise: 45.3
  Overall Quality: Excellent

Quality Issues:
  ✓ No issues detected

Diffraction Rings:
  Rings Detected: 8
  Ring Radii: [245, 356, 435, 612, 701, 845, 923, 1012]
  Assessment: Good

Statistics:
  Min/Max Intensity: 50 / 12543
  Mean Intensity: 325
  Saturation: 0.02%
  Hot Pixels: 15
```

### 9.3 Use Cases

**Before Integration:**
```
# Check if image quality is good enough
APEXA> image quality sample.ge5
# If quality is poor, adjust exposure time or detector settings
```

**Calibration Verification:**
```
# Detect rings to check calibration
APEXA> image rings CeO2_standard.ge5
# If few rings detected, check sample/beam alignment
```

**Batch Quality Control:**
```
# Check multiple images quickly
APEXA> image quality data_0001.ge5
APEXA> image quality data_0002.ge5
APEXA> image quality data_0003.ge5
```

### 9.4 What Gets Analyzed?

| **Metric** | **What It Tells You** |
|------------|----------------------|
| **Signal-to-Noise** | Data quality - higher is better (>10 is good) |
| **Saturation** | Detector overexposure - should be <1% |
| **Hot Pixels** | Bad detector pixels - indicates issues if >0.1% |
| **Ring Count** | Calibration quality - more rings = better |
| **Overall Quality** | Excellent / Good / Fair / Poor |

### 9.5 Natural Language Integration

You can also just ask APEXA naturally:

```
APEXA> What's the quality of sample_001.ge5?
# APEXA will automatically run image quality check

APEXA> Check if there are diffraction rings in the calibration image
# APEXA will run ring detection

APEXA> Is the signal good enough in this image?
# APEXA will analyze and tell you
```

---

## 10. Real-time Feedback During Beamtime

### 10.1 What is Real-time Monitoring?

During beamtime, APEXA can watch your data directory and automatically:
- Detect new images as they're collected
- Analyze quality instantly
- Alert you to problems (saturation, poor signal, etc.)
- Track statistics across the entire run

### 10.2 Start Monitoring

**Monitor a directory:**
```
APEXA> monitor start /data/experiment_run1

🔄 Real-time monitoring active on /data/experiment_run1
   Checking every 5 seconds
   Press Ctrl+C to stop or use 'monitor stop'
```

APEXA now watches the directory and automatically analyzes new images!

### 10.3 Real-time Alerts

**Example alerts during beamtime:**

```
🆕 Found 1 new file(s):

  📁 sample_0042.ge5
     Quality: Poor
     Rings: 2
     ⚠️  WARNING: Poor image quality detected in sample_0042.ge5
     ⚠️  WARNING: Few diffraction rings in sample_0042.ge5
```

```
🆕 Found 1 new file(s):

  📁 sample_0055.ge5
     Quality: Poor
     Rings: 8
     🚨 CRITICAL: Detector saturation in sample_0055.ge5
```

**Action immediately!** APEXA caught the problem before you wasted time on bad data.

### 10.4 Check Status

**See monitoring statistics:**
```
APEXA> monitor status

📊 Monitoring Status:
   Active: True
   Directory: /data/experiment_run1
   Files Processed: 127
   Total Alerts: 8
     ⚠️  Warnings: 6
     🚨 Critical: 2

   Recent Alerts:
     🚨 Detector saturation in sample_0055.ge5
     ⚠️  Poor image quality detected in sample_0042.ge5
     ⚠️  Few diffraction rings in sample_0038.ge5
```

### 10.5 Manual Check

**Check for new files without auto-monitoring:**
```
APEXA> monitor check

🆕 Found 3 new file(s):

  📁 sample_0128.ge5
     Quality: Excellent
     Rings: 12
     ✓ No alerts

  📁 sample_0129.ge5
     Quality: Excellent
     Rings: 12
     ✓ No alerts
```

### 10.6 Stop Monitoring

```
APEXA> monitor stop

⏹️  Monitoring stopped
   Files processed: 135
   Alerts generated: 8
```

### 10.7 Beamtime Workflow

**Complete beamtime monitoring workflow:**

```bash
# 1. Start your experiment
APEXA> monitor start /data/my_experiment

# 2. Begin data collection at beamline
# APEXA watches in background and alerts you to issues

# 3. Check status periodically
APEXA> monitor status

# 4. If you see critical alerts, adjust detector/beam settings immediately

# 5. After run is complete
APEXA> monitor stop
APEXA> session save "experiment_2025_10_15"
```

### 10.8 Alert Levels

| **Level** | **Icon** | **Meaning** | **Action** |
|-----------|----------|------------|------------|
| **CRITICAL** | 🚨 | Major problem (saturation, etc.) | Fix immediately! |
| **WARNING** | ⚠️  | Quality issue | Investigate |
| **INFO** | ℹ️  | FYI notification | Note for later |

### 10.9 Benefits

✅ **Catch problems early** - Before wasting beamtime
✅ **Automatic QC** - No manual checking needed
✅ **Complete log** - Track everything that happened
✅ **Peace of mind** - APEXA is watching while you focus on science

---

## 11. Advanced Plotting & Visualization

### 11.1 Overview

APEXA includes a powerful plotting engine built on matplotlib that creates publication-quality visualizations of your diffraction data. All plots are automatically saved to `~/.apexa/plots/` for easy access.

**What can you plot?**
- 📸 2D diffraction images (linear and log scale)
- 📊 Radial intensity profiles with peak detection
- 📈 1D integrated patterns with peak identification
- 🔄 Multi-pattern comparisons (overlay multiple datasets)

### 11.2 2D Image Plotting

**Visualize raw 2D diffraction images with dual-scale display.**

#### Basic Usage

```
APEXA> plot 2d sample.ge5
```

**Output:**
```
📊 Plotting 2D image: sample.ge5
✓ Plot saved: /home/username/.apexa/plots/sample_2d.png
  Statistics:
    Mean: 342.5
    Max: 65535.0
    Std: 1234.2
```

#### What You Get

The plot includes:
- **Left panel**: Linear scale - See overall intensity distribution
- **Right panel**: Log scale - Reveal weak features and rings
- **Colorbars**: Intensity values for both scales
- **Statistics**: Mean, max, standard deviation of pixel intensities

#### Supported Formats

All detector formats work:
- TIFF (.tif, .tiff)
- GE detectors (.ge2, .ge5)
- Mar detectors (.ed5)
- ESRF (.edf)

#### Example Use Cases

```
# Quick visual check before integration
APEXA> plot 2d data/fresh_sample.ge5

# Compare with reference
APEXA> plot 2d reference_material.tiff

# Check dark image quality
APEXA> plot 2d dark_background.ge2
```

### 11.3 Radial Profile Plotting

**Extract and visualize radial intensity distribution with automatic ring detection.**

#### Basic Usage

```
APEXA> plot radial sample.ge5
```

**Output:**
```
📊 Plotting radial profile: sample.ge5
✓ Radial profile plotted with 16 rings detected
  Plot saved: /home/username/.apexa/plots/sample_radial.png
```

#### What You Get

- **Radial profile curve**: Average intensity vs. radius from beam center
- **Peak markers**: Red dots at detected diffraction rings
- **Ring count**: Number of crystalline phases/d-spacings
- **Grid overlay**: Easy reading of radius and intensity values

#### How It Works

1. Calculates beam center (assumes image center)
2. Bins pixels by radial distance
3. Averages intensity in each radial bin
4. Detects peaks using scipy signal processing
5. Marks peaks with prominence > 1 standard deviation

#### Radial Profile Interpretation

**What the peaks tell you:**
- **Sharp, intense peaks** = Strong crystalline texture
- **Many peaks** = Multiple phases or polycrystalline
- **Broad peaks** = Small crystallites or strain
- **No peaks** = Amorphous material

#### Example Workflow

```
# Check image quality first
APEXA> image quality sample.ge5
# Output: 16 rings detected

# Visualize the rings
APEXA> plot radial sample.ge5
# Creates plot showing all 16 rings

# Now integrate to 1D for quantitative analysis
APEXA> integrate sample.ge5 to 1D using calib.txt
```

### 11.4 1D Pattern Plotting

**Visualize integrated diffraction patterns with automatic peak detection.**

#### Basic Usage

```
APEXA> plot 1d pattern.dat
```

**Output:**
```
📊 Plotting 1D pattern: pattern.dat
✓ 1D pattern plotted with 23 peaks detected
  Plot saved: /home/username/.apexa/plots/pattern_1d.png
```

#### What You Get

The plot shows **two panels**:
- **Top panel**: Linear scale - Overall pattern shape
- **Bottom panel**: Log scale - Reveals weak peaks
- **Peak markers**: Red dots at detected Bragg peaks
- **Smart axis labels**: Automatically detects Q (Å⁻¹) or 2θ (degrees)

#### Input File Formats

APEXA automatically handles:

**Two-column format** (most common):
```
# Q(A^-1)  Intensity
0.5  120.3
0.51 125.8
0.52 130.1
```

**Single-column format** (intensity only):
```
120.3
125.8
130.1
```

#### Peak Detection

Peaks are detected using:
- **Prominence threshold**: 2× standard deviation
- **Scipy find_peaks**: Robust peak finding
- **Smart filtering**: Avoids noise spikes

#### Example Use Cases

```
# View integrated pattern
APEXA> plot 1d integrated_pattern.dat

# Compare before/after processing
APEXA> plot 1d raw.chi
APEXA> plot 1d background_subtracted.chi

# Quick peak count
APEXA> plot 1d mystery_phase.xy
# Output: 23 peaks detected → Check against database
```

### 11.5 Multi-Pattern Comparison

**Overlay multiple 1D patterns to compare samples, temperatures, or time points.**

#### Basic Usage

```
APEXA> plot compare sample1.dat sample2.dat sample3.dat
```

**Output:**
```
📊 Comparing 3 patterns...
✓ Comparison plot created for 3 patterns
  Plot saved: /home/username/.apexa/plots/comparison_3patterns.png
```

#### What You Get

- **Normalized overlay**: All patterns scaled to same height for comparison
- **Color-coded**: Each pattern has distinct color from tab10 colormap
- **Two panels**: Linear (top) and log (bottom) scales
- **Legend**: Shows file names for each pattern
- **Aligned axes**: Easy to spot peak shifts

#### Normalization

All patterns are normalized to max=1.0 to enable fair comparison regardless of:
- Different counting times
- Different beam intensities
- Different sample amounts

#### Example: Temperature Series

```
# Compare phase transitions during heating
APEXA> plot compare 300K.dat 400K.dat 500K.dat 600K.dat
```

**What to look for:**
- **Peak shifts** → Thermal expansion
- **Peak splitting** → Phase transition
- **New peaks** → New phase formation
- **Peak disappearance** → Phase decomposition

#### Example: Time Series During Reaction

```
# Monitor in-situ reaction
APEXA> plot compare t0min.dat t5min.dat t10min.dat t15min.dat
```

**What to look for:**
- **Reactant peaks decreasing**
- **Product peaks growing**
- **Intermediate phases appearing/disappearing**

#### Example: Sample Comparison

```
# Compare different synthesis conditions
APEXA> plot compare as_synthesized.dat annealed_500C.dat annealed_800C.dat
```

### 11.6 Integration with Analysis Workflow

**Plotting works seamlessly with other APEXA features.**

#### Example 1: Monitor + Plot Pipeline

```
# Start monitoring beamtime
APEXA> monitor start /data/experiment

# When new file arrives, APEXA automatically analyzes it
# Then manually plot interesting ones:
APEXA> plot 2d /data/experiment/sample_0042.ge5
APEXA> plot radial /data/experiment/sample_0042.ge5
```

#### Example 2: Batch Integration + Comparison

```
# Integrate all files in a series
APEXA> batch integrate *.ge5 with dark.ge5 using calib.txt

# Compare all integrated patterns
APEXA> plot compare integrated_*.dat
```

#### Example 3: Image Analysis → Decision → Plot

```
# Quick quality check
APEXA> image quality sample.ge5
# Output: Excellent quality, 18 rings

# Good quality → Worth plotting
APEXA> plot 2d sample.ge5
APEXA> plot radial sample.ge5

# Bad quality → Skip and fix collection
APEXA> image quality bad_sample.ge5
# Output: Poor quality, saturation issues
# → Don't waste time plotting, fix detector setup
```

### 11.7 Customization & Advanced Usage

#### Plot Output Location

All plots are saved to:
```
~/.apexa/plots/
```

**Automatic naming:**
- 2D plots: `{filename}_2d.png`
- Radial: `{filename}_radial.png`
- 1D: `{filename}_1d.png`
- Comparison: `comparison_{N}patterns.png`

#### Resolution

All plots are saved at **150 DPI** - suitable for:
- ✅ PowerPoint presentations
- ✅ Lab notebooks
- ✅ Quick figures for papers
- ⚠️  For publication, you may want to re-generate at 300+ DPI

#### File Formats

Currently saves as **PNG** (portable, universal).

Future versions may support:
- PDF (vector graphics)
- SVG (editable in Inkscape/Illustrator)
- EPS (LaTeX compatibility)

### 11.8 Tips & Best Practices

#### ✅ DO

- **Plot 2D first** - Visual check before integration
- **Use radial plots** - Quick ring count and quality assessment
- **Compare normalized** - Use `plot compare` for multiple samples
- **Check both scales** - Linear and log reveal different features
- **Save important plots** - Copy from `~/.apexa/plots/` to your results folder

#### ❌ DON'T

- **Don't plot without checking quality** - Use `image quality` first
- **Don't compare too many** - More than 5-6 patterns gets cluttered
- **Don't ignore log scale** - Weak features often matter!
- **Don't forget to organize** - Plots accumulate quickly in `~/.apexa/plots/`

### 11.9 Troubleshooting Plots

#### "Error: File not found"

```
# Make sure file path is correct
APEXA> plot 2d /full/path/to/sample.ge5

# Or navigate to directory first
APEXA> cd /data/experiment
APEXA> plot 2d sample.ge5
```

#### "Error: Could not load image"

- **Check format**: APEXA supports TIFF, GE2, GE5, ED5, EDF
- **Check corruption**: Try opening in ImageJ or fabio
- **Check permissions**: Ensure file is readable

#### "No peaks detected"

This is normal for:
- Amorphous materials
- Very weak diffraction
- Highly textured samples (rings not visible radially)

#### Plot window not showing?

Plots are **saved automatically**, you don't need to see the window.

Check: `~/.apexa/plots/{filename}.png`

### 11.10 Example Session: Complete Analysis with Plotting

```
# 1. Check image quality
APEXA> image quality sample_RT.ge5
   → Excellent quality, 16 rings, SNR: 45.3

# 2. Visualize the 2D pattern
APEXA> plot 2d sample_RT.ge5
   → Saved to ~/.apexa/plots/sample_RT_2d.png

# 3. Check radial distribution
APEXA> plot radial sample_RT.ge5
   → 16 rings detected and plotted

# 4. Integrate to 1D
APEXA> integrate sample_RT.ge5 with dark_RT.ge5 using calib.txt
   → Created sample_RT_integrated.dat

# 5. Plot 1D pattern
APEXA> plot 1d sample_RT_integrated.dat
   → 23 peaks detected and marked

# 6. Heat sample and repeat
APEXA> integrate sample_500C.ge5 with dark_RT.ge5 using calib.txt
APEXA> integrate sample_800C.ge5 with dark_RT.ge5 using calib.txt

# 7. Compare temperature series
APEXA> plot compare sample_RT_integrated.dat sample_500C_integrated.dat sample_800C_integrated.dat
   → Comparison shows peak shift from thermal expansion!

# 8. Save plots to paper folder
APEXA> run bash cp ~/.apexa/plots/comparison_3patterns.png ~/paper/figures/
```

### 11.11 Benefits

✅ **Fast visualization** - No need to open separate software
✅ **Automatic peak detection** - Don't miss important features
✅ **Dual-scale display** - See both strong and weak features
✅ **Publication-ready** - 150 DPI PNG files
✅ **Integrated workflow** - Plot directly from APEXA prompt
✅ **Batch-friendly** - Easy to compare multiple patterns
✅ **Non-destructive** - Original data files never modified

---

## 12. Troubleshooting

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
