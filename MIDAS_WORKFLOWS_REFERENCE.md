# MIDAS Workflows Reference for APEXA

Comprehensive reference for all MIDAS workflows based on official manuals.

## 1. FF-HEDM Calibration Workflow (AutoCalibrateZarr.py)

**Manual:** https://github.com/marinerhemant/MIDAS/blob/master/manuals/FF_autocalibrate.md

**Purpose:** Automatically determine detector geometry from calibrant (CeO2, LaB6, etc.)

**Tool:** `midas_auto_calibrate`

**Required Inputs:**
- 2D diffraction image of calibrant (.tif, .ge2-5, .h5, .zarr.zip)
- Parameter file with: SpaceGroup, LatticeParameter, Wavelength, px

**Key Parameters:**
- `stopping_strain` (0.00004): Convergence criterion
- `mult_factor` (2.5): Outlier rejection threshold
- `lsd_guess` (µm): Initial detector distance
- `bc_x_guess`, `bc_y_guess` (pixels): Initial beam center
- `image_transform`: Flip/rotate detector ("0"=none, "1"=flip LR, "2"=flip UD, "3"=transpose)

**Workflow:**
1. Convert input to Zarr format
2. Apply median filter for background subtraction
3. Auto-detect beam center from ring geometry (or use guess)
4. Estimate Lsd from ring radius ratios (or use guess)
5. **Iterative refinement loop:**
   - Run CalibrantOMP (least-squares fitting)
   - Calculate pseudo-strain for each ring
   - Reject outliers (strain > mult_factor × median)
   - Update geometric parameters
   - Repeat until: no new outliers AND mean_strain < stopping_strain

**Outputs:**
- `refined_MIDAS_params.txt` - Final geometric parameters (BC, Lsd, tx, ty, tz, p0-p3)
- `autocal.log` - Iteration history with convergence metrics
- `calibrant_screen_out.csv` - Raw CalibrantOMP output

**Example:**
```python
{
    "image_file": "/data/CeO2_61keV_650mm.tif",
    "parameters_file": "/data/Params_CeO2.txt",
    "lsd_guess": 650000,
    "stopping_strain": 0.0001,
    "image_transform": "2"
}
```

---

## 2. FF-HEDM 2D→1D Integration

**Manual:** https://github.com/marinerhemant/MIDAS/blob/master/manuals/MIDAS_RADINT.md

**Purpose:** Azimuthal integration of 2D detector images to 1D patterns

### Method A: MIDAS Integrator (Simple, CPU-based)

**Tool:** `midas_integrate_2d_to_1d` (current implementation)

**Executable:** `MIDAS/FF_HEDM/bin/Integrator`

**Command:** `Integrator ParamFN ImageName [DarkName]`

**Inputs:**
- Calibrated parameter file (from auto-calibration)
- 2D diffraction image
- Optional dark file

**Outputs:**
- `ImageName_1d.dat` - Integrated 1D pattern (2θ, intensity)

### Method B: RADINT GPU Pipeline (Real-time, Production)

**Components:**
1. **DetectorMapper** (offline, once per geometry)
   - Pre-computes pixel→polar mapping
   - Outputs: `Map.bin`, `nMap.bin`

2. **IntegratorFitPeaksGPUStream** (real-time GPU processing)
   - Socket-based streaming integration
   - Performs peak fitting with NLopt
   - Outputs: `lineout.bin`, `fit.bin`

3. **integrator_batch_process.py** (orchestrator)
   - Manages DetectorMapper + IntegratorFitPeaksGPUStream + data feeder
   - Converts output to HDF5

**Workflow:**
```bash
python integrator_batch_process.py \
    --param-file setup.txt \
    --folder /data/scan \
    --dark /data/dark.bin \
    --output-h5 integrated.h5
```

**Status:** Not yet implemented in APEXA (requires GPU + complex setup)

---

## 3. FF-HEDM Full Analysis Workflow (ff_MIDAS.py)

**Manual:** https://github.com/marinerhemant/MIDAS/blob/master/manuals/FF_Analysis.md

**Purpose:** Complete FF-HEDM pipeline from raw data to grain reconstruction

**Tool:** `midas_run_ff_hedm_full_workflow` (currently limited implementation)

**Workflow Steps:**
1. **Data Conversion** (`ffGenerateZipRefactor.py`): Raw → Zarr
2. **HKL Generation** (`GetHKLList`): Calculate theoretical reflections
3. **Peak Search** (`PeaksFittingOMPZarrRefactor`): Find diffraction peaks
4. **Peak Merging** (`MergeOverlappingPeaksAllZarr`): Consolidate overlaps
5. **Binning** (`SaveBinData`): Prepare for indexing
6. **Indexing** (`IndexerOMP`): Find grain orientations
7. **Refinement** (`FitPosOrStrainsOMP`): Refine position + strain
8. **Post-processing** (`ProcessGrainsZarr`): Output `Grains.csv`

**Key Arguments:**
- `-paramFN`: Main parameter file
- `-resultFolder`: Output directory
- `-machineName`: Execution environment (local, purdue, umich, etc.)
- `-nCPUs` / `-nNodes`: Parallelization
- `-startLayerNr` / `-endLayerNr`: Layer range
- `-convertFiles`: Skip if Zarr exists (0/1)
- `-doPeakSearch`: Skip if peaks exist (0/1)

**Example:**
```bash
python ff_MIDAS.py \
    -paramFN Ti_params.txt \
    -resultFolder /results/Ti_analysis \
    -machineName local \
    -nCPUs 16 \
    -startLayerNr 1 \
    -endLayerNr 5
```

**Output:** `Grains.csv` with grain positions, orientations (Rodrigues vectors), strain tensors

**Status:** Needs full implementation in APEXA

---

## 4. NF-HEDM Calibration (GUI-based)

**Manual:** https://github.com/marinerhemant/MIDAS/blob/master/manuals/NF_calibration.md

**Purpose:** Calibrate multiple detector distances for near-field microscopy

**Tool:** `python ~/opt/MIDAS/gui/nf.py`

**Workflow:**
1. **Part I: Beam Center Determination**
   - Load DetZBeamPosScan or Au scan
   - Find left/right edge centers (horizontal + vertical)
   - Average to get beam center for each detector distance

2. **Part II: Detector Position Calculation**
   - Load Au calibration scan
   - Enter beam centers from Part I
   - Select same diffraction spot on all 3 detector distances
   - Calculate precise Lsd values

**Outputs:** Calibrated detector distances and Y-positions

**Status:** GUI-based, not suitable for automated APEXA workflow

---

## 5. NF-HEDM Full Analysis (nf_MIDAS.py)

**Manual:** https://github.com/marinerhemant/MIDAS/blob/master/manuals/NF_Analysis.md

**Purpose:** 3D microstructure reconstruction from multi-distance NF-HEDM data

**Workflow:**
1. **Pre-processing:**
   - HKL generation (`GetHKLListNF`)
   - Seed orientations (`GenSeedOrientationsFF2NFHEDM` if using FF seeds)
   - Grid generation (`MakeHexGrid`)
   - Simulate diffraction spots (`MakeDiffrSpots`)

2. **Image Processing:**
   - Median calculation (`MedianImageLibTiff`)
   - Background subtraction (`ImageProcessingLibTiffOMP`)

3. **Fitting:**
   - Memory map data (`MMapImageInfo`)
   - Copy to shared memory (/dev/shm)
   - Orientation fitting (`FitOrientationOMP`)
   - Parse results (`ParseMic`)

**Key Arguments:**
- `-paramFN`: NF parameter file
- `-machineName`: Cluster configuration
- `-nNodes` / `-nCPUs`: Parallelization
- `-ffSeedOrientations`: Use FF Grains.csv (0/1)
- `-refineParameters`: Geometry refinement mode (0/1)
- `-doImageProcessing`: Skip if already done (0/1)

**Output:** `Grains.mic` with 3D voxel-by-voxel orientations

**Status:** Complex HPC workflow, needs implementation

---

## Priority Implementation Order for Beamline

### Immediate Priority (Current Need)
1. ✅ **Auto-calibration** (`midas_auto_calibrate`) - DONE
2. ⚠️  **2D→1D Integration** (`midas_integrate_2d_to_1d`) - Needs improvement

### Medium Priority
3. **Parameter file creation** - Add validation and templates
4. **FF-HEDM peak search only** - For quick data quality checks
5. **Simple FF-HEDM workflow** - Local analysis with limited parameters

### Low Priority (Advanced/HPC)
6. RADINT GPU pipeline - Requires GPU infrastructure
7. Full FF-HEDM workflow with Parsl - HPC cluster integration
8. NF-HEDM workflows - Specialized experiments

---

## Common Parameter File Format

All MIDAS tools use a text-based parameter file:

```
# Material Properties
SpaceGroup 225                # CeO2 cubic Fm-3m
LatticeParameter 5.411 5.411 5.411 90 90 90  # a b c alpha beta gamma in Å
Wavelength 0.2021            # X-ray wavelength in Å

# Detector Geometry (from calibration)
Lsd 650118.126               # Sample-to-detector distance in µm
BC 702.863 865.468           # Beam center Y X in pixels
px 172.0                     # Pixel size in µm

# Detector Tilts (from calibration)
tx 0.000123                  # Rotation around X in rad
ty -0.000045                 # Rotation around Y in rad
tz 0.000089                  # Rotation around Z in rad

# Distortion Correction (from calibration)
p0 0.0                       # Radial distortion coefficients
p1 0.0
p2 0.0
p3 0.0
RhoD 650118.126              # Normalization for distortion (usually = Lsd)

# Integration Parameters
OmegaStep 0.25               # Rotation step in degrees
OmegaFirstFile 0.0           # Starting omega angle

# Data Paths
RawFolder /path/to/data
FileStem sample_
Ext .tif
StartFileNrFirstLayer 1
NrFilesPerSweep 1440

# Hardware
NrPixels 2048 2048          # Detector dimensions
```

---

## Tool Status in APEXA

| Tool | Status | Notes |
|------|--------|-------|
| `midas_auto_calibrate` | ✅ Implemented | Full parameter support, convergence metrics |
| `midas_integrate_2d_to_1d` | ⚠️  Partial | Uses Integrator executable, needs improvement |
| `midas_create_parameter_file` | ⚠️  Basic | Needs validation and templates |
| `midas_run_ff_hedm_full_workflow` | ⚠️  Stub | Needs full implementation |
| RADINT tools | ❌ Not implemented | Requires GPU + complex orchestration |
| NF workflows | ❌ Not implemented | HPC-focused, lower priority |

---

## References

All information from official MIDAS manuals:
- https://github.com/marinerhemant/MIDAS/tree/master/manuals
