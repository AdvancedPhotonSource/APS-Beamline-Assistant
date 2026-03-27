# APEXA Command Reference — Cheat Sheet

> Type these queries at the `APEXA>` prompt. APEXA routes to the correct agent automatically.

---

## Calibration

| What you want | What to type |
|---|---|
| Calibrate with auto-detect | `Calibrate the CeO2 image in test5` |
| Calibrate with energy | `Calibrate the CeO2 data at 61.332 keV in test5` |
| Calibrate with explicit files | `Calibrate test5/CeO_000001.tif.ge using test5/Parameters.txt with dark file test5/dark_CeO_000001.tif.ge` |
| Calibrate with initial guesses | `Calibrate test5/CeO_000001.tif.ge with Lsd guess 1200000 microns, beam center 1024 1024` |
| List calibrant materials | `What calibrants are available?` |
| Validate beamline params | `Validate: energy 61.332 keV, detector distance 650 mm, beam center 1024 1024, pixel size 200 microns` |

---

## Integration (2D to 1D)

| What you want | What to type |
|---|---|
| Basic integration | `Integrate the diffraction image in test5` |
| With explicit files | `Integrate test5/CeO_000001.tif.ge using calibration file test5/refined_MIDAS_params_CeO.txt` |
| With dark subtraction | `Integrate test5/CeO_000001.tif.ge with dark file test5/dark_CeO_000001.tif.ge` |
| Batch (multi-frame HDF5) | `Batch integrate data/sample_003083.ge1.h5 frames 3083 to 3085 using dark file data/dark_003084.ge1.h5 and parameter file refined_MIDAS_params.txt with 80 CPUs` |

---

## HEDM Workflows

| What you want | What to type |
|---|---|
| FF-HEDM reconstruction | `Run FF-HEDM reconstruction on test5` |
| FF-HEDM with GPU (v10) | `Run FF-HEDM on test5 with GPU enabled` |
| FF-HEDM resume checkpoint | `Resume FF-HEDM workflow from checkpoint file test5/checkpoint.h5` |
| FF-HEDM restart from step | `Restart FF-HEDM from the indexing step in test5` |
| NF-HEDM reconstruction | `Run NF-HEDM reconstruction using test5/Parameters.txt` |
| NF-HEDM with FF seeds | `Run NF-HEDM using test5/Parameters.txt with FF grains from test5/Grains.csv` |
| PF-HEDM workflow | `Run PF-HEDM workflow using Parameters.txt with positions file scan_positions.csv` |

---

## Grain Analysis & Post-Processing

| What you want | What to type |
|---|---|
| Match grains across steps | `Match grains between step1/Grains.csv and step2/Grains.csv with position tolerance 100 microns and orientation tolerance 2 degrees` |
| Calculate misorientation | `Calculate misorientation between grain 1 and grain 5 in test5/Grains.csv for FCC (space group 225)` |
| Extract grain centroids | `Extract grain centroids from test5/Grains.mic with minimum grain size 100 voxels` |
| Forward simulation | `Run forward simulation using test5/Grains.csv and test5/Parameters.txt` |
| Export to Dream3D | `Convert NF-HEDM results to Dream3D format` |
| Overlay FF + NF results | `Overlay FF-HEDM and NF-HEDM results from test5` |

---

## X-ray Calculations

| What you want | What to type |
|---|---|
| Energy to wavelength | `Convert 61.332 keV to wavelength` |
| Wavelength to energy | `Convert 0.2022 angstroms to energy in keV` |
| d-spacing from hkl | `Calculate d-spacing for (110) plane in bcc iron` |
| d-spacing from 2-theta | `Calculate d-spacing for 2-theta of 12.5 degrees at wavelength 0.202 angstroms` |
| 2-theta from d-spacing | `What is the 2-theta angle for d-spacing 2.03 angstroms at 61.332 keV?` |
| Strain calculation | `Calculate strain: measured d-spacing 2.035, reference d-spacing 2.028` |
| List materials | `List all available materials in the database` |

---

## Knowledge & Material Properties

| What you want | What to type |
|---|---|
| Material properties | `What are the material properties of CeO2?` |
| Lattice parameters | `Give me the lattice parameters for LaB6` |
| Space group | `What is the space group of titanium?` |
| HEDM methodology | `What is the difference between FF-HEDM and NF-HEDM?` |
| Best practices | `What are best practices for calibration at high energy?` |
| Typical parameters | `What are typical HEDM parameters for 61 keV?` |
| Estimate Lsd from rings | `Estimate detector distance from ring radii [412, 478, 675] pixels at wavelength 0.2022 angstroms with pixel size 200 microns for CeO2` |

---

## Visualization

| What you want | What to type |
|---|---|
| Raw 2D diffraction image | `Show me the diffraction image in test5` |
| Lineout (1D profile) | `Plot the lineout for test5 integration` |
| Caked image | `Show the caked output in test5/integration` |
| Calibration fit results | `Plot the calibration results in test5` |
| Live viewer (GPU) | `Launch the live viewer for test5/CeO_000001_lineout.bin with 2000 radial bins` |
| FF grain map (Dash app) | `Show the FF-HEDM grain results in test5` |
| NF microstructure map | `View the NF-HEDM microstructure map test5/Grains.mic` |
| Compare lineouts | `Compare lineouts from test5/CeO_lineout.xy and test6/CeO_lineout.xy` |
| Caked peaks | `Plot caked peaks from test5/CeO_caked_peaks.h5` |

---

## Motor Control (EPICS)

| What you want | What to type |
|---|---|
| Read position | `What is the position of motor m1?` |
| Full status | `Show me the status of motor m1` |
| List multiple motors | `Show positions of motors m1, m2, m3, m4` |
| Absolute move | `Move m1 to 25.3` |
| Relative move | `Move m1 by +0.5` |
| Tweak (fine step) | `Tweak m1 forward by 0.01` |
| Jog (timed move) | `Jog m1 forward for 2 seconds` |
| Stop immediately | `Stop motor m1!` |
| Set velocity | `Set motor m1 velocity to 5.0` |
| Check limits | `What are the limits of m1?` |
| Set soft limits | `Set high limit of m1 to 200.0 and low limit to -50.0` |
| Home motor | `Home motor m1 in the forward direction` |

---

## File Operations

| What you want | What to type |
|---|---|
| List directory | `List files in test5` |
| Read file | `Read the Parameters.txt file in test5` |
| File info | `Get file info for test5/CeO_000001.tif.ge` |
| Convert GE to TIFF | `Convert all GE files in test5 to TIFF format` |
| Check environment | `Check the current environment` |

---

## Multi-Turn Workflow Example

```
APEXA> Calibrate the CeO2 data in test5 at 61.332 keV
APEXA> Now integrate that data
APEXA> Show me the lineout
APEXA> What phases match these peaks?
APEXA> Run FF-HEDM on this data
APEXA> How many grains were found?
APEXA> Track grains between step 1 and step 2
APEXA> Export the results to Dream3D
```

---

## System Commands

| Command | Description |
|---|---|
| `help` | Show available commands |
| `models` | List all available AI models |
| `servers` | Show connected MCP servers |
| `clear` | Clear conversation history |
| `quit` | Exit APEXA |
