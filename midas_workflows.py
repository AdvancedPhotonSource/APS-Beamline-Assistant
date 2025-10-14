"""
MIDAS Workflow Manager
Orchestrates complex multi-step MIDAS workflows for beamline operations

Author: Beamline Assistant Team
Organization: Argonne National Laboratory
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
import subprocess
import json
from datetime import datetime


class MIDASWorkflowManager:
    """Orchestrate complex multi-step MIDAS workflows with progress tracking."""

    def __init__(self, midas_path: Path = None):
        """Initialize workflow manager.

        Args:
            midas_path: Path to MIDAS installation root
        """
        if midas_path is None:
            midas_path = Path("/Users/b324240/opt/MIDAS")

        self.midas_path = Path(midas_path)
        self.bin_path = self.midas_path / "build" / "bin"
        self.ff_bin = self.midas_path / "FF_HEDM" / "bin"
        self.nf_bin = self.midas_path / "NF_HEDM" / "bin"
        self.ff_v7 = self.midas_path / "FF_HEDM" / "v7"
        self.nf_v7 = self.midas_path / "NF_HEDM" / "v7"
        self.utils = self.midas_path / "utils"

        self.workflow_state = {
            "status": "idle",
            "current_step": None,
            "completed_steps": [],
            "errors": [],
            "start_time": None,
            "end_time": None
        }

    async def run_ff_workflow(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full FF-HEDM workflow with progress tracking.

        Args:
            config: Configuration dictionary with workflow parameters

        Returns:
            Dictionary with workflow results and status
        """
        self.workflow_state["status"] = "running"
        self.workflow_state["start_time"] = datetime.now().isoformat()
        self.workflow_state["completed_steps"] = []
        self.workflow_state["errors"] = []

        result_folder = Path(config["result_folder"])
        result_folder.mkdir(parents=True, exist_ok=True)

        workflow_log = result_folder / "workflow_log.json"

        try:
            # Step 1: Data validation
            await self._update_step("Validating input data")
            validation = await self._validate_ff_inputs(config)
            if not validation["success"]:
                raise Exception(f"Validation failed: {validation['error']}")

            # Step 2: Data conversion (if needed)
            if config.get("convert_data", True):
                await self._update_step("Converting data to Zarr format")
                conversion = await self._convert_ff_data(config)
                if not conversion["success"]:
                    raise Exception(f"Data conversion failed: {conversion['error']}")

            # Step 3: Generate HKL list
            await self._update_step("Generating HKL list")
            hkl_gen = await self._run_executable("GetHKLListZarr",
                                                config["param_file"],
                                                str(result_folder))
            if not hkl_gen["success"]:
                raise Exception("HKL generation failed")

            # Step 4: Peak search (if requested)
            if config.get("do_peak_search", True):
                await self._update_step("Searching for diffraction peaks")
                peak_search = await self._run_executable("PeaksFittingOMPZarrRefactor",
                                                        config["param_file"],
                                                        str(result_folder))
                if not peak_search["success"]:
                    raise Exception("Peak search failed")

                # Step 5: Merge overlapping peaks
                await self._update_step("Merging overlapping peaks")
                merge = await self._run_executable("MergeOverlappingPeaksAllZarr",
                                                  config["param_file"],
                                                  str(result_folder))
                if not merge["success"]:
                    raise Exception("Peak merging failed")

            # Step 6: Calculate radii and prepare for indexing
            await self._update_step("Calculating peak radii")
            calc_radius = await self._run_executable("CalcRadiusAllZarr",
                                                    config["param_file"],
                                                    str(result_folder))

            # Step 7: Indexing
            await self._update_step("Indexing grain orientations")
            indexing = await self._run_executable("IndexerOMP",
                                                 config["param_file"],
                                                 str(result_folder),
                                                 timeout=3600)
            if not indexing["success"]:
                raise Exception("Indexing failed")

            # Step 8: Refinement
            await self._update_step("Refining grain positions and strains")
            refinement = await self._run_executable("FitPosOrStrainsOMP",
                                                   config["param_file"],
                                                   str(result_folder),
                                                   timeout=3600)
            if not refinement["success"]:
                self.workflow_state["errors"].append("Refinement failed, using indexed results")

            # Step 9: Post-processing
            await self._update_step("Processing final grain results")
            postproc = await self._run_executable("ProcessGrainsZarr",
                                                 config["param_file"],
                                                 str(result_folder))

            # Finalize
            await self._update_step("Workflow completed")
            self.workflow_state["status"] = "completed"
            self.workflow_state["end_time"] = datetime.now().isoformat()

            # Collect results
            results = await self._collect_ff_results(result_folder, config)

            # Save workflow log
            with open(workflow_log, 'w') as f:
                json.dump({
                    "workflow": "FF-HEDM",
                    "config": config,
                    "state": self.workflow_state,
                    "results": results
                }, f, indent=2)

            return {
                "success": True,
                "workflow": "FF-HEDM",
                "state": self.workflow_state,
                "results": results
            }

        except Exception as e:
            self.workflow_state["status"] = "failed"
            self.workflow_state["end_time"] = datetime.now().isoformat()
            self.workflow_state["errors"].append(str(e))

            # Save error log
            with open(workflow_log, 'w') as f:
                json.dump({
                    "workflow": "FF-HEDM",
                    "config": config,
                    "state": self.workflow_state,
                    "error": str(e)
                }, f, indent=2)

            return {
                "success": False,
                "workflow": "FF-HEDM",
                "state": self.workflow_state,
                "error": str(e)
            }

    async def run_nf_workflow(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full NF-HEDM workflow with progress tracking.

        Args:
            config: Configuration dictionary with workflow parameters

        Returns:
            Dictionary with workflow results and status
        """
        self.workflow_state["status"] = "running"
        self.workflow_state["start_time"] = datetime.now().isoformat()
        self.workflow_state["completed_steps"] = []
        self.workflow_state["errors"] = []

        param_file = Path(config["param_file"])
        work_dir = param_file.parent

        workflow_log = work_dir / "nf_workflow_log.json"

        try:
            # Step 1: Pre-processing
            await self._update_step("NF-HEDM pre-processing")

            # Generate HKL list
            hkl_gen = await self._run_executable("GetHKLListNF",
                                                str(param_file),
                                                str(work_dir))
            if not hkl_gen["success"]:
                raise Exception("NF HKL generation failed")

            # Generate seed orientations from FF if requested
            if config.get("ff_seed_orientations", True) and config.get("ff_grains_file"):
                await self._update_step("Generating seed orientations from FF-HEDM")
                seed_gen = await self._run_executable("GenSeedOrientationsFF2NFHEDM",
                                                     str(param_file),
                                                     str(work_dir))

            # Create reconstruction grid
            await self._update_step("Creating reconstruction grid")
            grid = await self._run_executable("MakeHexGrid",
                                             str(param_file),
                                             str(work_dir))

            # Simulate diffraction spots
            await self._update_step("Simulating diffraction spots")
            spots = await self._run_executable("MakeDiffrSpots",
                                              str(param_file),
                                              str(work_dir))

            # Step 2: Image processing (if requested)
            if config.get("do_image_processing", True):
                await self._update_step("Processing images (median filter)")
                median = await self._run_executable("MedianImageLibTiff",
                                                   str(param_file),
                                                   str(work_dir))

                await self._update_step("Background subtraction")
                bg_sub = await self._run_executable("ImageProcessingLibTiffOMP",
                                                   str(param_file),
                                                   str(work_dir))

            # Step 3: Orientation fitting
            await self._update_step("Fitting voxel orientations")
            fitting = await self._run_executable("FitOrientationOMP",
                                                str(param_file),
                                                str(work_dir),
                                                timeout=7200)
            if not fitting["success"]:
                raise Exception("Orientation fitting failed")

            # Step 4: Post-processing
            await self._update_step("Parsing reconstruction results")
            parse = await self._run_executable("ParseMic",
                                              str(param_file),
                                              str(work_dir))

            # Finalize
            await self._update_step("NF-HEDM workflow completed")
            self.workflow_state["status"] = "completed"
            self.workflow_state["end_time"] = datetime.now().isoformat()

            # Collect results
            results = await self._collect_nf_results(work_dir, config)

            # Save workflow log
            with open(workflow_log, 'w') as f:
                json.dump({
                    "workflow": "NF-HEDM",
                    "config": config,
                    "state": self.workflow_state,
                    "results": results
                }, f, indent=2)

            return {
                "success": True,
                "workflow": "NF-HEDM",
                "state": self.workflow_state,
                "results": results
            }

        except Exception as e:
            self.workflow_state["status"] = "failed"
            self.workflow_state["end_time"] = datetime.now().isoformat()
            self.workflow_state["errors"].append(str(e))

            # Save error log
            with open(workflow_log, 'w') as f:
                json.dump({
                    "workflow": "NF-HEDM",
                    "config": config,
                    "state": self.workflow_state,
                    "error": str(e)
                }, f, indent=2)

            return {
                "success": False,
                "workflow": "NF-HEDM",
                "state": self.workflow_state,
                "error": str(e)
            }

    async def run_combined_ff_nf(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run FF-HEDM then seed NF-HEDM from results.

        Args:
            config: Configuration dictionary with both FF and NF parameters

        Returns:
            Dictionary with combined workflow results
        """
        self.workflow_state["status"] = "running"
        self.workflow_state["start_time"] = datetime.now().isoformat()

        try:
            # Run FF-HEDM first
            await self._update_step("Running FF-HEDM workflow")
            ff_result = await self.run_ff_workflow(config["ff_config"])

            if not ff_result["success"]:
                raise Exception("FF-HEDM workflow failed")

            # Find FF grains file
            ff_grains = None
            for layer_info in ff_result["results"].get("layer_outputs", []):
                if "grains_file" in layer_info:
                    ff_grains = layer_info["grains_file"]
                    break

            if not ff_grains:
                raise Exception("No FF-HEDM grains file found")

            # Run NF-HEDM with FF seeds
            await self._update_step("Running NF-HEDM workflow with FF seeds")
            nf_config = config["nf_config"]
            nf_config["ff_seed_orientations"] = True
            nf_config["ff_grains_file"] = ff_grains

            nf_result = await self.run_nf_workflow(nf_config)

            # Finalize
            await self._update_step("Combined FF-NF workflow completed")
            self.workflow_state["status"] = "completed"
            self.workflow_state["end_time"] = datetime.now().isoformat()

            return {
                "success": True,
                "workflow": "Combined FF-NF-HEDM",
                "state": self.workflow_state,
                "ff_results": ff_result,
                "nf_results": nf_result
            }

        except Exception as e:
            self.workflow_state["status"] = "failed"
            self.workflow_state["end_time"] = datetime.now().isoformat()
            self.workflow_state["errors"].append(str(e))

            return {
                "success": False,
                "workflow": "Combined FF-NF-HEDM",
                "state": self.workflow_state,
                "error": str(e)
            }

    async def _update_step(self, step_name: str):
        """Update current workflow step."""
        if self.workflow_state["current_step"]:
            self.workflow_state["completed_steps"].append(self.workflow_state["current_step"])
        self.workflow_state["current_step"] = step_name
        print(f"[{datetime.now().isoformat()}] {step_name}", flush=True)

    async def _run_executable(self, exe_name: str, param_file: str, cwd: str,
                             timeout: int = 1800) -> Dict[str, Any]:
        """Run a MIDAS executable asynchronously."""
        # Try multiple possible locations
        possible_paths = [
            self.bin_path / exe_name,
            self.ff_bin / exe_name,
            self.nf_bin / exe_name
        ]

        exe_path = None
        for p in possible_paths:
            if p.exists():
                exe_path = p
                break

        if not exe_path:
            return {
                "success": False,
                "error": f"Executable not found: {exe_name}",
                "searched_paths": [str(p) for p in possible_paths]
            }

        try:
            process = await asyncio.create_subprocess_exec(
                str(exe_path), param_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            return {
                "success": process.returncode == 0,
                "return_code": process.returncode,
                "stdout": stdout.decode(),
                "stderr": stderr.decode()
            }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Execution timed out after {timeout}s"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _validate_ff_inputs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate FF-HEDM workflow inputs."""
        errors = []

        # Check parameter file
        param_file = Path(config.get("param_file", ""))
        if not param_file.exists():
            errors.append(f"Parameter file not found: {param_file}")

        # Check data file
        data_file = Path(config.get("data_file", ""))
        if not data_file.exists():
            errors.append(f"Data file not found: {data_file}")

        # Check result folder can be created
        result_folder = Path(config.get("result_folder", ""))
        try:
            result_folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create result folder: {e}")

        return {
            "success": len(errors) == 0,
            "errors": errors
        }

    async def _convert_ff_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert FF-HEDM data to Zarr format."""
        script = self.utils / "ffGenerateZip.py"

        if not script.exists():
            return {
                "success": False,
                "error": "Data conversion script not found"
            }

        try:
            process = await asyncio.create_subprocess_exec(
                "python", str(script),
                "-resultFolder", config["result_folder"],
                "-paramFN", config["param_file"],
                "-dataFN", config["data_file"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=config["result_folder"]
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=600
            )

            return {
                "success": process.returncode == 0,
                "return_code": process.returncode
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _collect_ff_results(self, result_folder: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Collect FF-HEDM workflow results."""
        results = {
            "layer_outputs": [],
            "total_grains": 0
        }

        # Check each layer
        for layer in range(config.get("start_layer", 1), config.get("end_layer", 1) + 1):
            layer_dir = result_folder / f"LayerNr_{layer}"
            if layer_dir.exists():
                grains_file = layer_dir / "GrainsReconstructed.csv"
                if grains_file.exists():
                    # Count grains
                    try:
                        with open(grains_file, 'r') as f:
                            n_grains = sum(1 for line in f) - 1
                    except:
                        n_grains = 0

                    results["layer_outputs"].append({
                        "layer": layer,
                        "grains_file": str(grains_file),
                        "n_grains": n_grains
                    })
                    results["total_grains"] += n_grains

        return results

    async def _collect_nf_results(self, work_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Collect NF-HEDM workflow results."""
        results = {
            "mic_file": None,
            "n_voxels": 0
        }

        # Check for MIC file
        mic_file = work_dir / "Grains.mic"
        if mic_file.exists():
            results["mic_file"] = str(mic_file)

            # Count voxels
            try:
                with open(mic_file, 'r') as f:
                    results["n_voxels"] = sum(1 for line in f if not line.startswith('%'))
            except:
                pass

        return results

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return self.workflow_state.copy()


# Example usage
if __name__ == "__main__":
    import asyncio

    async def example():
        manager = MIDASWorkflowManager()

        # FF-HEDM workflow
        ff_config = {
            "result_folder": "./ff_results",
            "param_file": "./Parameters.txt",
            "data_file": "./raw_data.zip",
            "start_layer": 1,
            "end_layer": 1,
            "do_peak_search": True
        }

        result = await manager.run_ff_workflow(ff_config)
        print(json.dumps(result, indent=2))

    # asyncio.run(example())
