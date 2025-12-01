# APEXA User Manual
**Advanced Photon EXperiment Assistant**

Your AI beamline scientist for real-time HEDM data analysis at APS.

---

## Quick Start

```bash
./start_beamline_assistant.sh
```

That's it! The assistant will:
- ✅ Load all analysis servers automatically
- ✅ Connect to Argo-AI (GPT-4o, Claude, Gemini)
- ✅ Auto-detect MIDAS installation
- ✅ Ready for natural language commands

---

## What Can APEXA Do?

### 🔧 Detector Calibration
Automatically calibrate detector geometry using calibrant materials.

**Example Prompts:**
```
APEXA> calibrate using the CeO2 image in this directory
APEXA> auto-calibrate with LaB6_650mm.tif and use stopping strain 1e-3
APEXA> run calibration on ceria file with mask and initial parameters
```

**What It Does:**
- Finds calibrant image (CeO2, LaB6, Si)
- Uses MIDAS `AutoCalibrateZarr.py` workflow
- Iteratively refines: beam center, distance, tilts, distortion
- Outputs: `refined_MIDAS_params.txt`, `autocal.log`

**Supported Formats:** `.tif`, `.tiff`, `.ge2`, `.ge3`, `.ge4`, `.ge5`, `.h5`

---

### 📊 2D → 1D Integration
Integrate diffraction images to 1D patterns for phase analysis.

**Example Prompts:**
```
APEXA> integrate the .tif file to 1D
APEXA> integrate data.ge5 using Parameters.txt
APEXA> integrate with dark file subtraction using dark.ge5
APEXA> batch integrate all .tif files in /data/experiment
```

**What It Does:**
- Azimuthal integration using pyFAI or MIDAS Integrator
- Optional dark file subtraction for background correction
- Outputs: `.dat` file with Q vs Intensity

**Use Cases:**
- Phase identification (peak matching)
- Strain analysis
- Texture analysis
- Rietveld refinement preparation

---

### 🔬 FF-HEDM Full Workflow
Complete Far-Field HEDM grain reconstruction pipeline.

**Example Prompts:**
```
APEXA> run FF-HEDM workflow on /data/experiment
APEXA> run FF-HEDM with 32 CPUs on layers 1-10
APEXA> analyze the FF-HEDM data in ~/beamtime/sample1
```

**What It Does:**
1. Data conversion to Zarr format
2. HKL list generation
3. Peak search and fitting
4. Peak merging
5. Indexing (grain orientation)
6. Refinement (position/strain)
7. Post-processing

**Outputs:**
- `GrainsReconstructed.csv` - Grain orientations and positions
- `*.MIDAS.zip` - Zarr archive with all data

**Time:** ~10-60 minutes depending on dataset size

---

### 🗺️ NF-HEDM Reconstruction
Near-Field HEDM for high-resolution 3D microstructure mapping.

**Example Prompts:**
```
APEXA> run NF-HEDM reconstruction with FF seed orientations
APEXA> process NF data using grains from FF-HEDM
APEXA> reconstruct microstructure with 10 CPUs
```

**What It Does:**
- Image processing (dark/flat correction, spot finding)
- Voxel-by-voxel orientation fitting
- 3D grain map generation
- Strain tensor calculation

**Resolution:** ~1-5 μm (vs ~50-100 μm for FF-HEDM)

---

### 🔍 Phase Identification
Identify crystalline phases from diffraction peaks.

**Example Prompts:**
```
APEXA> identify phases from peaks at 12.5, 18.2, 25.8 degrees
APEXA> what phases match these peaks: 15.3, 22.1, 31.4
APEXA> match the peaks in pattern.dat to known phases
```

**What It Does:**
- Searches crystallography databases (COD, ICSD)
- Matches d-spacings to known phases
- Ranks candidates by match quality

**Common Calibrants:** CeO2, LaB6, Si, Al₂O₃, NAC

---

### 🧮 Crystallography Math Tools
**⚠️ IMPORTANT: Use these tools for math - LLMs are unreliable at calculations!**

APEXA includes verified crystallography calculators for accurate computations.

#### Calculate d-spacing from 2θ
```
APEXA> calculate d-spacing for 2-theta 10.5 degrees with wavelength 0.2066 angstroms
APEXA> what's the d-spacing at 3.79 degrees using lambda 0.2021?
```

**Formula:** d = λ / (2·sin(θ))

**Returns:**
- d-spacing in Ångströms
- Formula used
- Step-by-step calculation
- Full transparency for verification

---

#### Calculate 2θ from d-spacing
```
APEXA> calculate 2-theta for d-spacing 3.124 angstroms with wavelength 0.2021
APEXA> what angle does the (111) peak appear at? d=3.12 Å, λ=0.2021 Å
```

**Formula:** 2θ = 2·arcsin(λ/(2d))

**Use Case:** Predicting peak positions for phase identification

---

#### Convert Energy ↔ Wavelength
```
APEXA> convert 61.332 keV to wavelength
APEXA> what's the wavelength for 71.2 keV X-rays?
APEXA> convert 0.2021 angstroms to energy
```

**Formula:** λ(Å) = 12.398 / E(keV)

**Common Energies:**
- APS 1-ID: 61.332 keV (λ = 0.2021 Å)
- APS 1-ID: 71.676 keV (λ = 0.1729 Å)

---

#### Calculate Lattice Strain
```
APEXA> calculate strain for measured d=3.155 and reference d=3.124
APEXA> what's the strain if d changed from 3.120 to 3.108?
```

**Formula:** ε = (d_measured - d_reference) / d_reference

**Returns:**
- Strain (absolute and percentage)
- Strain type (tensile/compressive)
- Full calculation shown

**Use Case:** Residual stress analysis, thermal expansion studies

---

#### Calculate Detector Distance
```
APEXA> calculate detector distance from ring at 512 pixels,
       d=3.124 angstroms, lambda=0.2021, pixel size 200 microns
```

**Formula:** L = r / tan(θ), where 2θ = 2·arcsin(λ/(2d))

**Use Case:** Verifying calibration, detector setup validation

---

### 📁 File Operations
Navigate, read, and search your data directories.

**Example Prompts:**
```
APEXA> list files in /data/experiment
APEXA> read the Parameters.txt file
APEXA> find all .ge5 files in this directory
APEXA> show me the calibration parameters
```

**Available Operations:**
- List directories (with filters)
- Read text files
- Search for files (glob patterns)
- Get file metadata
- Navigate directory trees

---

### 📈 Data Visualization
Plot diffraction data for quality checks.

**Example Prompts:**
```
APEXA> plot 2D image sample.tif
APEXA> plot radial profile of data.ge5
APEXA> compare these three 1D patterns
APEXA> show me the integrated pattern
```

**Plot Types:**
- 2D images (linear + log scale)
- Radial profiles (with peak detection)
- 1D patterns (Q vs Intensity)
- Multi-pattern comparisons

**Outputs:** PNG files saved to `~/.apexa/plots/`

---

### ⚡ Batch Processing
Process hundreds of files with one command.

**Example Prompts:**
```
APEXA> batch integrate all .tif files
APEXA> process all images in /data using Parameters.txt
APEXA> convert all GE files to TIFF
```

**What It Does:**
- Automatically finds matching files
- Processes in parallel (when possible)
- Progress tracking
- Error reporting for failed files

**Use Cases:**
- Time-resolved experiments
- Temperature series
- Deformation series
- Mapping experiments

---

### 🎯 Real-Time Monitoring
Monitor data quality during beamtime.

**Example Prompts:**
```
APEXA> start monitoring /data/live for new images
APEXA> alert me if detector saturates
APEXA> check image quality every 5 seconds
```

**What It Does:**
- Watches directory for new files
- Automatic quality checks:
  - Signal-to-noise ratio
  - Detector saturation
  - Hot pixel detection
  - Ring visibility
- Real-time alerts for issues

**Perfect For:** Optimizing acquisition during beamtime

---

## Example Demo Session

### Setup Check
```
APEXA> what tools are available?
APEXA> check if MIDAS is installed
APEXA> list the current directory
```

### Standard Calibration Workflow
```
APEXA> I have a CeO2 calibration image. Let's calibrate the detector.

→ AI finds CeO2.tif and Parameters.txt
→ Runs AutoCalibrateZarr.py
→ Outputs refined_MIDAS_params.txt

APEXA> show me the calibrated beam center and distance

→ AI reads refined_MIDAS_params.txt
→ Reports: BC = (1024.3, 1048.7), Lsd = 652.4 mm
```

### Data Analysis Workflow
```
APEXA> integrate sample_001.tif using the calibrated parameters

→ Runs pyFAI integration
→ Outputs sample_001.dat

APEXA> plot the integrated pattern

→ Creates 1D plot with peak detection
→ Saves to ~/.apexa/plots/sample_001_1d.png

APEXA> identify the phases from the peaks

→ Matches peaks to crystallography databases
→ Reports: "Likely phases: Ti (α-phase), TiO₂ (rutile)"
```

### FF-HEDM Reconstruction
```
APEXA> run FF-HEDM on this dataset with 20 CPUs

→ Executes full ff_MIDAS.py workflow
→ Processes layers 1-10
→ Finds 2,347 grains

APEXA> what's the average grain size?

→ Analyzes GrainsReconstructed.csv
→ Reports: "Average grain diameter: 45 μm"
```

---

## Tips for Effective Use

### 🎯 Be Conversational
You don't need to memorize commands. Just describe what you want:

**Good:**
- "Calibrate using the ceria file"
- "Integrate this image with dark subtraction"
- "Find all gold data in the directory"

**Also Works:**
- "midas_auto_calibrate --image CeO2.tif --params Parameters.txt"
- But why type all that? 😊

---

### 📂 Use Relative Paths
The assistant understands context:

```
APEXA> list files here
APEXA> read the parameters file there
APEXA> integrate the .tif file
```

It remembers the current directory and previous file references.

---

### 🔄 Conversation History
The assistant remembers your session:

```
APEXA> list files in /data/sample1
→ Shows: CeO2.tif, Parameters.txt, dark.tif

APEXA> calibrate using the first file
→ Knows you mean CeO2.tif

APEXA> now integrate with that dark file
→ Remembers dark.tif from the listing
```

---

### 💡 Ask for Help
Not sure what to do next?

```
APEXA> what should I do after calibration?
APEXA> how do I run FF-HEDM?
APEXA> what's the difference between FF and NF-HEDM?
```

The AI provides guidance and suggests next steps.

---

## Advanced Features

### Smart Suggestions
After each analysis, APEXA suggests next steps:

```
APEXA> integrate sample.tif

→ Integration complete

📊 Suggested next steps:
• Identify phases from peak positions
• Perform Rietveld refinement
• Check for peak splitting (sample stress/strain)
```

---

### Error Prevention
APEXA validates parameters before execution:

```
APEXA> integrate missing_file.tif

✗ Validation Error: Image file not found: missing_file.tif
→ Prevented wasted computation
```

**Checks:**
- File existence
- File format compatibility
- Required parameters present
- Parameter value ranges

---

### Smart Caching
Frequently accessed data is cached:

```
APEXA> list files in /data

→ Lists files

APEXA> list files in /data again

→ Returns instantly (from cache)
```

Reduces latency for repeated operations.

---

## Model Selection

Choose your preferred AI model:

```bash
# In .env file:
ARGO_MODEL=gpt4o          # Default - Fast, capable
ARGO_MODEL=claudesonnet4  # Best reasoning
ARGO_MODEL=gemini25pro    # Longest context
```

**Available Models:**
- **OpenAI:** gpt4o, gpt4turbo, gpt4, gpt35
- **Anthropic:** claudesonnet4, claudeopus4
- **Google:** gemini25pro, gemini25flash

**Switch On-the-Fly:**
```
APEXA> models
APEXA> switch to claudesonnet4
```

---

## Troubleshooting

### "Tool not found" Warning
```
WARNING  Tool 'midas_auto_calibrate' not listed
```

**Fix:** Restart the assistant. This warning is cosmetic and doesn't affect functionality.

---

### MIDAS Not Detected
```
⚠ No valid MIDAS installation found
```

**Fix:** Set the path manually:
```bash
# In .env:
export MIDAS_PATH=/path/to/MIDAS
```

**Or activate MIDAS conda environment first:**
```bash
conda activate midas_env
./start_beamline_assistant.sh
```

---

### Calibration Fails to Converge
```
✗ Convergence failed after 50 iterations
```

**Common Causes:**
1. **Bad initial guess** - Check `Lsd` and `BC` in Parameters.txt
2. **Wrong calibrant** - Verify material (CeO2 vs LaB6)
3. **Low-quality image** - Check saturation and SNR

**Fix:** Adjust starting parameters or use `--mult_factor 3.5` for more aggressive refinement.

---

### Integration Produces Noisy Pattern
```
Pattern has low signal-to-noise ratio
```

**Solutions:**
1. **Use dark file subtraction:**
   ```
   APEXA> integrate with dark file dark_001.tif
   ```

2. **Check detector mask:**
   Mask saturated/dead pixels before integration

3. **Increase exposure time** (if possible during collection)

---

## File Formats

### Supported Image Formats
- **TIFF:** `.tif`, `.tiff` (most common)
- **GE Detectors:** `.ge2`, `.ge3`, `.ge4`, `.ge5`
- **Eiger/Pilatus:** `.h5`, `.hdf5`
- **European Detectors:** `.edf`

### Parameter Files
MIDAS uses `Parameters.txt` format:
```
Lsd 650000         # Sample-to-detector distance (μm)
BC 1024 1024       # Beam center (pixels)
tx 0 ty 0 tz 0     # Detector tilts (degrees)
px 200             # Pixel size (μm)
Wavelength 0.1777  # X-ray wavelength (Å)
```

### Output Files
- **Calibration:** `refined_MIDAS_params.txt`, `autocal.log`
- **Integration:** `*.dat` (Q, Intensity)
- **FF-HEDM:** `GrainsReconstructed.csv`, `*.MIDAS.zip`
- **NF-HEDM:** `Grains.csv`, `Microstructure.h5`

---

## System Requirements

### Minimum
- **OS:** Linux (Ubuntu 20.04+, RHEL 8+)
- **Python:** 3.10+
- **Memory:** 16 GB RAM
- **Storage:** 100 GB for typical datasets

### Recommended
- **CPU:** 32+ cores (for FF-HEDM)
- **Memory:** 64 GB RAM
- **GPU:** NVIDIA (for future deep learning features)
- **Storage:** 1 TB SSD

### Network
- **ANL Network Access** required for Argo-AI
- Alternative: Use local LLM (configuration needed)

---

## Keyboard Shortcuts

- **Ctrl+C:** Interrupt current operation
- **Ctrl+D:** Exit assistant
- **↑/↓:** Navigate command history
- **Tab:** Auto-complete (if supported by terminal)

---

## Getting Help

### Built-in Help
```
APEXA> help
APEXA> what can you do?
APEXA> how do I calibrate?
APEXA> explain FF-HEDM workflow
```

### Documentation
- This manual: `USER_MANUAL.md`
- MIDAS docs: https://github.com/marinerhemant/MIDAS
- Argo Gateway: (ANL internal)

### Support
- **Issues:** Report via GitHub
- **Beamline:** Contact beamline scientist
- **Email:** [your support email]

---

## Credits

**APEXA Development:**
- Pawan Tripathi (Lead Developer)
- Advanced Photon Source, Argonne National Laboratory

**Core Dependencies:**
- **MIDAS:** Hemant Sharma (github.com/marinerhemant/MIDAS)
- **FastMCP:** Marvin (github.com/jlowin/fastmcp)
- **Argo Gateway:** Argonne National Laboratory

---

## Version

**Current Version:** 1.0.0  
**Last Updated:** November 2024  
**Compatibility:** MIDAS 2024.11+, Python 3.10+

---

**Happy Analyzing! 🔬✨**
