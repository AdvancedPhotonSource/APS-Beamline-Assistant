#!/usr/bin/env python3
"""
APEXA - Advanced Photon EXperiment Assistant
AI-powered beamline scientist for synchrotron X-ray diffraction analysis

Developed for: Advanced Photon Source, Argonne National Laboratory
Author: Pawan Tripathi
"""

import asyncio
import json
import os
import re
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import httpx
from dotenv import load_dotenv

load_dotenv()

class ExperimentContext:
    """Smart context manager for APEXA sessions"""
    def __init__(self, session_dir: Path = None):
        self.session_dir = session_dir or Path.home() / ".apexa" / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = {
            "experiment_id": None,
            "sample_name": None,
            "beamline": None,
            "start_time": datetime.now().isoformat(),
            "user": os.getenv("ANL_USERNAME", "unknown"),
            "current_directory": str(Path.cwd()),
            "analysis_history": [],
            "key_findings": [],
            "active_files": []
        }

    def update(self, key: str, value: Any):
        """Update experiment metadata"""
        self.metadata[key] = value

    def add_analysis(self, analysis_type: str, result: str):
        """Record analysis performed"""
        self.metadata["analysis_history"].append({
            "timestamp": datetime.now().isoformat(),
            "type": analysis_type,
            "result": result[:500]  # Truncate long results
        })

    def add_finding(self, finding: str):
        """Record key scientific finding"""
        self.metadata["key_findings"].append({
            "timestamp": datetime.now().isoformat(),
            "finding": finding
        })

    def save_session(self, session_name: str = None):
        """Save current session to disk"""
        if not session_name:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        session_file = self.session_dir / f"{session_name}.json"
        with open(session_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        return session_file

    def load_session(self, session_name: str):
        """Load a previous session"""
        session_file = self.session_dir / f"{session_name}.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                self.metadata = json.load(f)
            return True
        return False

    def list_sessions(self) -> List[str]:
        """List all available sessions"""
        return [f.stem for f in self.session_dir.glob("*.json")]

    def get_summary(self) -> str:
        """Get a summary of current experiment"""
        summary = f"Experiment: {self.metadata.get('experiment_id', 'Unnamed')}\n"
        summary += f"Sample: {self.metadata.get('sample_name', 'N/A')}\n"
        summary += f"Analyses performed: {len(self.metadata['analysis_history'])}\n"
        summary += f"Key findings: {len(self.metadata['key_findings'])}\n"
        return summary

class ProactiveSuggestions:
    """Generate smart next-step suggestions based on analysis results"""

    @staticmethod
    def suggest_after_phase_id(phases_found: List[str]) -> str:
        """Suggest next steps after phase identification"""
        suggestions = []

        if len(phases_found) == 1:
            suggestions.append("📊 **Suggested next steps:**")
            suggestions.append("• Quantify phase fraction using Rietveld refinement")
            suggestions.append("• Check for preferred orientation (texture analysis)")
            suggestions.append("• Calculate lattice parameters and compare to literature")
        elif len(phases_found) > 1:
            suggestions.append("📊 **Suggested next steps:**")
            suggestions.append("• Quantify relative phase fractions")
            suggestions.append("• Map phase distribution (if using HEDM)")
            suggestions.append("• Analyze phase transformation conditions")

        return "\n".join(suggestions)

    @staticmethod
    def suggest_after_ring_detection(num_rings: int) -> str:
        """Suggest next steps after ring detection"""
        suggestions = ["📊 **Suggested next steps:**"]

        if num_rings > 5:
            suggestions.append("• Integrate rings to 1D pattern for phase ID")
            suggestions.append("• Check calibration quality (ring circularity)")
            suggestions.append("• Perform full FF-HEDM reconstruction")
        else:
            suggestions.append("• Check if sample is single crystal (few rings)")
            suggestions.append("• Verify detector calibration")
            suggestions.append("• Consider if more exposure time needed")

        return "\n".join(suggestions)

    @staticmethod
    def suggest_after_ff_hedm(num_grains: int) -> str:
        """Suggest next steps after FF-HEDM reconstruction"""
        suggestions = ["📊 **Suggested next steps:**"]
        suggestions.append(f"• Analyze grain size distribution ({num_grains} grains found)")
        suggestions.append("• Calculate grain orientations and texture")
        suggestions.append("• Track grains through deformation series (if applicable)")
        suggestions.append("• Export to DREAM.3D for visualization")
        suggestions.append("• Calculate misorientation statistics")

        return "\n".join(suggestions)

    @staticmethod
    def suggest_after_integration() -> str:
        """Suggest next steps after 2D to 1D integration"""
        return """📊 **Suggested next steps:**
• Identify phases from peak positions
• Perform Rietveld refinement
• Check for peak splitting (sample stress/strain)
• Compare with reference patterns"""

    @staticmethod
    def get_suggestion(tool_name: str, result: str) -> Optional[str]:
        """Get proactive suggestion based on tool used"""

        # Parse result to extract key info
        if "identify_crystalline_phases" in tool_name:
            # Count phases mentioned in result
            phases = []
            if "phase" in result.lower():
                return ProactiveSuggestions.suggest_after_phase_id(["phase"])

        elif "detect_diffraction_rings" in tool_name:
            # Try to extract number of rings
            import re
            match = re.search(r'(\d+)\s+rings?', result.lower())
            num_rings = int(match.group(1)) if match else 5
            return ProactiveSuggestions.suggest_after_ring_detection(num_rings)

        elif "run_ff_hedm" in tool_name:
            match = re.search(r'(\d+)\s+grains?', result.lower())
            num_grains = int(match.group(1)) if match else 0
            return ProactiveSuggestions.suggest_after_ff_hedm(num_grains)

        elif "integrate_2d_to_1d" in tool_name:
            return ProactiveSuggestions.suggest_after_integration()

        return None

class BatchProcessor:
    """Smart batch processing for multiple files"""

    @staticmethod
    async def process_batch(client, operation: str, files: List[str], **kwargs) -> Dict[str, Any]:
        """Process multiple files with the same operation

        Args:
            client: APEXAClient instance
            operation: Tool name to execute
            files: List of file paths
            **kwargs: Additional arguments for the tool

        Returns:
            Dictionary with results for each file
        """
        results = {
            "operation": operation,
            "total_files": len(files),
            "successful": 0,
            "failed": 0,
            "results": []
        }

        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing: {Path(file_path).name}")

            try:
                # Merge file path with other arguments
                args = {**kwargs, "image_path": file_path}
                result = await client.execute_tool_call(operation, args)

                results["results"].append({
                    "file": file_path,
                    "status": "success",
                    "result": result
                })
                results["successful"] += 1

            except Exception as e:
                results["results"].append({
                    "file": file_path,
                    "status": "failed",
                    "error": str(e)
                })
                results["failed"] += 1

        return results

class ErrorPreventor:
    """Validate inputs and prevent common errors before execution"""

    @staticmethod
    def validate_integration_params(args: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate parameters for 2D to 1D integration

        Returns:
            (is_valid, error_message)
        """
        image_path = args.get("image_path")
        calibration_file = args.get("calibration_file")
        wavelength = args.get("wavelength")
        detector_distance = args.get("detector_distance")
        beam_center_x = args.get("beam_center_x")
        beam_center_y = args.get("beam_center_y")
        dark_file = args.get("dark_file")

        # Check image exists
        if image_path:
            img_path = Path(image_path).expanduser()
            if not img_path.exists():
                return False, f"Image file not found: {image_path}"

            # Check file extension
            valid_extensions = ['.tif', '.tiff', '.ge2', '.ge5', '.ed5', '.edf']
            if not any(str(img_path).lower().endswith(ext) for ext in valid_extensions):
                return False, f"Unsupported image format. Use: {', '.join(valid_extensions)}"

        # Check dark file if provided
        if dark_file:
            dark_path = Path(dark_file).expanduser()
            if not dark_path.exists():
                return False, f"Dark file not found: {dark_file}"

        # Check calibration or manual parameters
        has_calib = calibration_file is not None
        has_manual = all([wavelength, detector_distance, beam_center_x, beam_center_y])

        if not has_calib and not has_manual:
            return False, "Either calibration_file OR all manual parameters (wavelength, detector_distance, beam_center_x, beam_center_y) must be provided"

        # Validate calibration file if provided
        if calibration_file:
            cal_path = Path(calibration_file).expanduser()
            if not cal_path.exists():
                return False, f"Calibration file not found: {calibration_file}"

        # Validate manual parameters if provided
        if has_manual:
            if wavelength <= 0:
                return False, f"Wavelength must be positive, got: {wavelength}"
            if detector_distance <= 0:
                return False, f"Detector distance must be positive, got: {detector_distance}"
            if beam_center_x < 0 or beam_center_y < 0:
                return False, f"Beam center coordinates must be positive"

        return True, None

    @staticmethod
    def validate_ff_hedm_params(args: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate parameters for FF-HEDM workflow"""
        example_dir = args.get("example_dir")

        if not example_dir:
            return False, "example_dir parameter is required"

        dir_path = Path(example_dir).expanduser()
        if not dir_path.exists():
            return False, f"Directory not found: {example_dir}"

        if not dir_path.is_dir():
            return False, f"Path is not a directory: {example_dir}"

        # Check for Parameters.txt
        param_file = dir_path / "Parameters.txt"
        if not param_file.exists():
            return False, f"Parameters.txt not found in {example_dir}"

        return True, None

class WorkflowBuilder:
    """Natural language workflow builder for complex analysis sequences"""

    def __init__(self):
        self.workflows = {
            "phase_analysis": [
                {"tool": "midas_integrate_2d_to_1d", "description": "Integrate 2D image to 1D pattern"},
                {"tool": "midas_identify_crystalline_phases", "description": "Identify phases from peaks"}
            ],
            "full_hedm": [
                {"tool": "filesystem_list_directory", "description": "Check data directory"},
                {"tool": "midas_run_ff_hedm_full_workflow", "description": "Run FF-HEDM reconstruction"}
            ],
            "calibration_check": [
                {"tool": "midas_detect_diffraction_rings", "description": "Detect rings for calibration"},
                {"tool": "midas_integrate_2d_to_1d", "description": "Integrate to verify calibration"}
            ]
        }

    def get_workflow(self, workflow_name: str) -> Optional[List[Dict[str, str]]]:
        """Get predefined workflow steps"""
        return self.workflows.get(workflow_name)

    def suggest_workflow(self, user_query: str) -> Optional[str]:
        """Suggest appropriate workflow based on user query"""
        query_lower = user_query.lower()

        if "phase" in query_lower and "identif" in query_lower:
            return "phase_analysis"
        elif "ff-hedm" in query_lower or "hedm" in query_lower:
            return "full_hedm"
        elif "calibrat" in query_lower:
            return "calibration_check"

        return None

class ImageAnalyzer:
    """Multimodal image analysis - AI can see and analyze diffraction images"""

    @staticmethod
    def analyze_image_quality(image_path: str) -> Dict[str, Any]:
        """Analyze diffraction image quality using vision

        Args:
            image_path: Path to diffraction image

        Returns:
            Dictionary with quality metrics and AI observations
        """
        try:
            import fabio
            import numpy as np
            from scipy import ndimage

            img = fabio.open(image_path)
            data = img.data.astype(float)

            # Calculate quality metrics
            metrics = {
                "image_path": image_path,
                "dimensions": data.shape,
                "data_type": str(data.dtype),
                "statistics": {
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "mean": float(np.mean(data)),
                    "median": float(np.median(data)),
                    "std": float(np.std(data))
                }
            }

            # Signal-to-noise ratio
            background = np.percentile(data, 10)
            signal = np.percentile(data, 99)
            snr = (signal - background) / np.std(data[data < np.percentile(data, 20)])
            metrics["signal_to_noise"] = float(snr)

            # Detect saturation
            max_val = np.max(data)
            if data.dtype == np.uint16:
                saturation_threshold = 65535 * 0.95
            else:
                saturation_threshold = max_val * 0.95

            saturated_pixels = np.sum(data >= saturation_threshold)
            saturation_percent = (saturated_pixels / data.size) * 100
            metrics["saturation_percent"] = float(saturation_percent)

            # Hot pixel detection
            median_filtered = ndimage.median_filter(data, size=3)
            diff = np.abs(data - median_filtered)
            hot_pixels = np.sum(diff > 10 * np.std(diff))
            metrics["hot_pixels"] = int(hot_pixels)

            # Overall quality assessment
            quality = "Excellent"
            issues = []

            if snr < 5:
                quality = "Poor"
                issues.append("Low signal-to-noise ratio")
            elif snr < 10:
                quality = "Fair"
                issues.append("Moderate signal-to-noise ratio")

            if saturation_percent > 1:
                quality = "Poor" if quality != "Poor" else quality
                issues.append(f"Saturation detected ({saturation_percent:.1f}% pixels)")

            if hot_pixels > data.size * 0.01:
                issues.append(f"Many hot pixels detected ({hot_pixels})")

            metrics["overall_quality"] = quality
            metrics["issues"] = issues

            return metrics

        except Exception as e:
            return {
                "error": str(e),
                "image_path": image_path
            }

    @staticmethod
    def detect_rings_visual(image_path: str) -> Dict[str, Any]:
        """Detect diffraction rings visually

        Args:
            image_path: Path to diffraction image

        Returns:
            Ring detection results
        """
        try:
            import fabio
            import numpy as np
            from scipy import ndimage

            img = fabio.open(image_path)
            data = img.data.astype(float)

            # Simple ring detection via radial integration
            center_y, center_x = np.array(data.shape) // 2
            y, x = np.indices(data.shape)
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)

            # Radial profile
            radial_profile = ndimage.mean(data, labels=r, index=np.arange(0, r.max()))

            # Find peaks in radial profile
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(radial_profile,
                                          height=np.percentile(radial_profile, 75),
                                          distance=20)

            ring_radii = peaks.tolist()
            ring_intensities = [float(radial_profile[p]) for p in peaks]

            return {
                "image_path": image_path,
                "rings_detected": len(ring_radii),
                "ring_radii_pixels": ring_radii,
                "ring_intensities": ring_intensities,
                "center_position": [int(center_x), int(center_y)],
                "quality": "Good" if len(ring_radii) > 3 else "Check calibration"
            }

        except Exception as e:
            return {
                "error": str(e),
                "image_path": image_path
            }

    @staticmethod
    def create_image_summary(image_path: str) -> str:
        """Create human-readable summary for AI vision

        Args:
            image_path: Path to image

        Returns:
            Text summary suitable for AI multimodal understanding
        """
        quality = ImageAnalyzer.analyze_image_quality(image_path)
        rings = ImageAnalyzer.detect_rings_visual(image_path)

        summary = f"""
📸 Image Analysis: {Path(image_path).name}

Image Properties:
  Dimensions: {quality.get('dimensions', 'N/A')}
  Signal-to-Noise: {quality.get('signal_to_noise', 0):.1f}
  Overall Quality: {quality.get('overall_quality', 'Unknown')}

Quality Issues:
{chr(10).join('  • ' + issue for issue in quality.get('issues', [])) if quality.get('issues') else '  ✓ No issues detected'}

Diffraction Rings:
  Rings Detected: {rings.get('rings_detected', 0)}
  Ring Radii: {rings.get('ring_radii_pixels', [])}
  Assessment: {rings.get('quality', 'N/A')}

Statistics:
  Min/Max Intensity: {quality.get('statistics', {}).get('min', 0):.0f} / {quality.get('statistics', {}).get('max', 0):.0f}
  Mean Intensity: {quality.get('statistics', {}).get('mean', 0):.0f}
  Saturation: {quality.get('saturation_percent', 0):.2f}%
  Hot Pixels: {quality.get('hot_pixels', 0)}
"""
        return summary

class RealtimeFeedback:
    """Real-time experiment feedback during beamtime"""

    def __init__(self):
        self.monitoring = False
        self.watch_directory = None
        self.last_check_time = None
        self.processed_files = set()
        self.alerts = []

    def start_monitoring(self, directory: str, check_interval: int = 5):
        """Start monitoring directory for new diffraction images

        Args:
            directory: Directory to watch
            check_interval: Check for new files every N seconds
        """
        self.monitoring = True
        self.watch_directory = Path(directory)
        self.last_check_time = datetime.now()
        self.check_interval = check_interval

        return {
            "status": "monitoring_started",
            "directory": str(self.watch_directory),
            "check_interval": check_interval,
            "message": f"Real-time monitoring active on {directory}"
        }

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring = False
        return {
            "status": "monitoring_stopped",
            "files_processed": len(self.processed_files),
            "alerts_generated": len(self.alerts)
        }

    def check_new_files(self) -> List[Dict[str, Any]]:
        """Check for new diffraction images and analyze them

        Returns:
            List of new file analyses
        """
        if not self.monitoring or not self.watch_directory:
            return []

        new_analyses = []

        # Find new image files
        for ext in ['.tif', '.tiff', '.ge2', '.ge5', '.ed5', '.edf']:
            for img_file in self.watch_directory.glob(f'*{ext}'):
                if img_file.stat().st_mtime > self.last_check_time.timestamp():
                    if str(img_file) not in self.processed_files:
                        # New file detected! Analyze it
                        analysis = self._analyze_and_alert(img_file)
                        new_analyses.append(analysis)
                        self.processed_files.add(str(img_file))

        self.last_check_time = datetime.now()
        return new_analyses

    def _analyze_and_alert(self, image_path: Path) -> Dict[str, Any]:
        """Analyze new image and generate alerts if needed

        Args:
            image_path: Path to new image

        Returns:
            Analysis with alerts
        """
        quality = ImageAnalyzer.analyze_image_quality(str(image_path))
        rings = ImageAnalyzer.detect_rings_visual(str(image_path))

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "file": image_path.name,
            "quality": quality,
            "rings": rings,
            "alerts": []
        }

        # Generate alerts for issues
        if quality.get('overall_quality') == 'Poor':
            alert = {
                "level": "WARNING",
                "message": f"Poor image quality detected in {image_path.name}",
                "details": quality.get('issues', [])
            }
            analysis["alerts"].append(alert)
            self.alerts.append(alert)

        if quality.get('saturation_percent', 0) > 1:
            alert = {
                "level": "CRITICAL",
                "message": f"Detector saturation in {image_path.name}",
                "details": f"{quality.get('saturation_percent', 0):.1f}% pixels saturated"
            }
            analysis["alerts"].append(alert)
            self.alerts.append(alert)

        if rings.get('rings_detected', 0) < 3:
            alert = {
                "level": "INFO",
                "message": f"Few diffraction rings in {image_path.name}",
                "details": f"Only {rings.get('rings_detected', 0)} rings detected"
            }
            analysis["alerts"].append(alert)

        return analysis

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring session

        Returns:
            Session statistics
        """
        return {
            "monitoring_active": self.monitoring,
            "directory": str(self.watch_directory) if self.watch_directory else None,
            "files_processed": len(self.processed_files),
            "total_alerts": len(self.alerts),
            "critical_alerts": len([a for a in self.alerts if a['level'] == 'CRITICAL']),
            "warning_alerts": len([a for a in self.alerts if a['level'] == 'WARNING']),
            "recent_alerts": self.alerts[-5:] if self.alerts else []
        }

class PlottingEngine:
    """Advanced plotting for diffraction data visualization"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path.home() / ".apexa" / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_2d_image(self, image_path: str, scale: str = "linear", save: bool = True, show: bool = False) -> Dict[str, Any]:
        """Plot 2D diffraction image with enhancements

        Args:
            image_path: Path to diffraction image
            scale: "linear" or "log" for intensity scale
            save: Save plot to file
            show: Display plot interactively

        Returns:
            Dictionary with plot info and statistics
        """
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            import fabio

            # Load image
            img = fabio.open(str(Path(image_path).expanduser().absolute()))
            data = img.data.astype(float)

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Linear scale
            im1 = ax1.imshow(data, cmap='viridis', origin='lower')
            ax1.set_title(f'{Path(image_path).name} - Linear Scale')
            ax1.set_xlabel('X (pixels)')
            ax1.set_ylabel('Y (pixels)')
            plt.colorbar(im1, ax=ax1, label='Intensity')

            # Log scale
            data_log = np.copy(data)
            data_log[data_log <= 0] = 1  # Avoid log(0)
            im2 = ax2.imshow(data_log, cmap='viridis', norm=colors.LogNorm(), origin='lower')
            ax2.set_title(f'{Path(image_path).name} - Log Scale')
            ax2.set_xlabel('X (pixels)')
            ax2.set_ylabel('Y (pixels)')
            plt.colorbar(im2, ax=ax2, label='Intensity (log)')

            plt.tight_layout()

            # Save or show
            output_path = None
            if save:
                output_path = self.output_dir / f"{Path(image_path).stem}_2d.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')

            if show:
                plt.show()
            else:
                plt.close()

            # Statistics
            stats = {
                "mean": float(np.mean(data)),
                "max": float(np.max(data)),
                "min": float(np.min(data)),
                "std": float(np.std(data))
            }

            return {
                "status": "success",
                "plot_saved": str(output_path) if output_path else None,
                "statistics": stats,
                "message": f"2D plot created for {Path(image_path).name}"
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def plot_radial_profile(self, image_path: str, save: bool = True, show: bool = False) -> Dict[str, Any]:
        """Plot radial intensity profile with peak detection

        Args:
            image_path: Path to diffraction image
            save: Save plot to file
            show: Display plot interactively

        Returns:
            Dictionary with plot info and detected peaks
        """
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import fabio
            from scipy import signal

            # Load image
            img = fabio.open(str(Path(image_path).expanduser().absolute()))
            data = img.data.astype(float)

            # Calculate center
            center_y, center_x = np.array(data.shape) / 2

            # Create radial profile
            y, x = np.indices(data.shape)
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            r = r.astype(int)

            # Bin by radius
            tbin = np.bincount(r.ravel(), data.ravel())
            nr = np.bincount(r.ravel())
            radial_prof = tbin / nr

            # Remove NaN values
            radial_prof = radial_prof[~np.isnan(radial_prof)]
            radii = np.arange(len(radial_prof))

            # Detect peaks
            peaks, properties = signal.find_peaks(radial_prof, prominence=np.std(radial_prof))

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(radii, radial_prof, 'b-', linewidth=1, label='Radial Profile')
            ax.plot(peaks, radial_prof[peaks], 'ro', markersize=8, label=f'Peaks ({len(peaks)} found)')

            ax.set_xlabel('Radius (pixels)')
            ax.set_ylabel('Average Intensity')
            ax.set_title(f'Radial Profile - {Path(image_path).name}')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save or show
            output_path = None
            if save:
                output_path = self.output_dir / f"{Path(image_path).stem}_radial.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')

            if show:
                plt.show()
            else:
                plt.close()

            return {
                "status": "success",
                "plot_saved": str(output_path) if output_path else None,
                "peaks_detected": len(peaks),
                "peak_positions": peaks.tolist(),
                "message": f"Radial profile plotted with {len(peaks)} rings detected"
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def plot_1d_pattern(self, pattern_file: str, save: bool = True, show: bool = False) -> Dict[str, Any]:
        """Plot 1D integrated diffraction pattern

        Args:
            pattern_file: Path to 1D pattern file (.dat, .xy, .chi)
            save: Save plot to file
            show: Display plot interactively

        Returns:
            Dictionary with plot info and peak information
        """
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy import signal

            # Load 1D pattern
            pattern_path = Path(pattern_file).expanduser().absolute()
            data = np.loadtxt(pattern_path)

            if data.ndim == 1:
                # Single column - assume it's intensity only
                q = np.arange(len(data))
                intensity = data
            else:
                # Two columns - Q and intensity
                q = data[:, 0]
                intensity = data[:, 1]

            # Detect peaks
            peaks, properties = signal.find_peaks(intensity, prominence=np.std(intensity)*2)

            # Create plot with two subplots (linear and log)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # Linear scale
            ax1.plot(q, intensity, 'b-', linewidth=1, label='Integrated Pattern')
            ax1.plot(q[peaks], intensity[peaks], 'ro', markersize=6, label=f'Peaks ({len(peaks)})')
            ax1.set_ylabel('Intensity')
            ax1.set_title(f'1D Pattern - {Path(pattern_file).name} (Linear)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Log scale
            ax2.semilogy(q, intensity, 'b-', linewidth=1, label='Integrated Pattern')
            ax2.semilogy(q[peaks], intensity[peaks], 'ro', markersize=6, label=f'Peaks ({len(peaks)})')
            ax2.set_xlabel('Q (Å⁻¹)' if q.max() < 20 else '2θ (degrees)')
            ax2.set_ylabel('Intensity (log)')
            ax2.set_title(f'1D Pattern - {Path(pattern_file).name} (Log)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save or show
            output_path = None
            if save:
                output_path = self.output_dir / f"{Path(pattern_file).stem}_1d.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')

            if show:
                plt.show()
            else:
                plt.close()

            return {
                "status": "success",
                "plot_saved": str(output_path) if output_path else None,
                "peaks_detected": len(peaks),
                "peak_positions": q[peaks].tolist(),
                "message": f"1D pattern plotted with {len(peaks)} peaks detected"
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def plot_comparison(self, files: list, labels: list = None, save: bool = True, show: bool = False) -> Dict[str, Any]:
        """Compare multiple 1D patterns in one plot

        Args:
            files: List of pattern file paths
            labels: Optional custom labels for each pattern
            save: Save plot to file
            show: Display plot interactively

        Returns:
            Dictionary with plot info
        """
        try:
            import numpy as np
            import matplotlib.pyplot as plt

            if not labels:
                labels = [Path(f).stem for f in files]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            colors = plt.cm.tab10(np.linspace(0, 1, len(files)))

            for i, (file, label, color) in enumerate(zip(files, labels, colors)):
                # Load pattern
                data = np.loadtxt(Path(file).expanduser().absolute())

                if data.ndim == 1:
                    q = np.arange(len(data))
                    intensity = data
                else:
                    q = data[:, 0]
                    intensity = data[:, 1]

                # Normalize for comparison
                intensity_norm = intensity / np.max(intensity)

                # Plot
                ax1.plot(q, intensity_norm, color=color, linewidth=1.5,
                        label=label, alpha=0.8)
                ax2.semilogy(q, intensity_norm, color=color, linewidth=1.5,
                           label=label, alpha=0.8)

            # Linear scale
            ax1.set_ylabel('Normalized Intensity')
            ax1.set_title('Pattern Comparison (Linear)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Log scale
            ax2.set_xlabel('Q (Å⁻¹)' if q.max() < 20 else '2θ (degrees)')
            ax2.set_ylabel('Normalized Intensity (log)')
            ax2.set_title('Pattern Comparison (Log)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save or show
            output_path = None
            if save:
                output_path = self.output_dir / f"comparison_{len(files)}patterns.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')

            if show:
                plt.show()
            else:
                plt.close()

            return {
                "status": "success",
                "plot_saved": str(output_path) if output_path else None,
                "patterns_compared": len(files),
                "message": f"Comparison plot created for {len(files)} patterns"
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

class SmartCache:
    """Cache expensive operations to reduce AI costs and improve speed"""

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path.home() / ".apexa" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}

    def get_cache_key(self, operation: str, args: Dict[str, Any]) -> str:
        """Generate cache key from operation and arguments"""
        import hashlib
        # Sort args for consistent hashing
        sorted_args = json.dumps(args, sort_keys=True)
        key_str = f"{operation}:{sorted_args}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, operation: str, args: Dict[str, Any]) -> Optional[Any]:
        """Get cached result if available"""
        cache_key = self.get_cache_key(operation, args)

        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                self.memory_cache[cache_key] = cached_data
                return cached_data

        return None

    def set(self, operation: str, args: Dict[str, Any], result: Any):
        """Cache result for future use"""
        cache_key = self.get_cache_key(operation, args)

        # Save to memory
        self.memory_cache[cache_key] = result

        # Save to disk
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        except Exception:
            pass  # Non-critical if caching fails

class APEXAClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.sessions = {}
        self.exit_stack = AsyncExitStack()

        # Smart context manager for session persistence
        self.context = ExperimentContext()

        # Initialize smart features
        self.batch_processor = BatchProcessor()
        self.error_preventor = ErrorPreventor()
        self.workflow_builder = WorkflowBuilder()
        self.cache = SmartCache()
        self.image_analyzer = ImageAnalyzer()
        self.realtime_feedback = RealtimeFeedback()
        self.plotting = PlottingEngine()

        # Determine environment based on model (dev models require dev endpoint)
        self.anl_username = os.getenv("ANL_USERNAME")
        self.selected_model = os.getenv("ARGO_MODEL", "gpt4o")

        # Models only available in DEV environment
        self.dev_only_models = {
            "gpt5", "gpt5mini", "gpt5nano",
            "gemini25pro", "gemini25flash",
            "claudeopus41", "claudeopus4", "claudesonnet45", "claudesonnet4", "claudesonnet37",
            "gpto1", "gpto3mini", "gpto4mini", "gpt41", "gpt41mini", "gpt41nano"
        }

        # Use DEV endpoint if model requires it
        if self.selected_model in self.dev_only_models:
            self.argo_chat_url = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"
            self.argo_embed_url = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/embed/"
            self.environment = "DEV"
        else:
            self.argo_chat_url = "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/"
            self.argo_embed_url = "https://apps.inside.anl.gov/argoapi/api/v1/resource/embed/"
            self.environment = "PROD"

        self.http_client = httpx.AsyncClient(timeout=120.0)

        # Conversation history for interactive sessions
        self.conversation_history = []

        if not self.anl_username:
            raise ValueError("ANL_USERNAME must be set in environment (.env file)")

        self.available_models = {
            "OpenAI (PROD)": {
                "gpt35": "GPT-3.5 Turbo (4K tokens)",
                "gpt4": "GPT-4 (8K tokens)",
                "gpt4turbo": "GPT-4 Turbo (128K input)",
                "gpt4o": "GPT-4o (128K input, 16K output)"
            },
            "OpenAI (DEV only)": {
                "gpt5": "GPT-5 (272K input, 128K output)",
                "gpt5mini": "GPT-5 Mini (272K input, 128K output)",
                "gpt5nano": "GPT-5 Nano (272K input, 128K output)"
            },
            "Google (DEV only)": {
                "gemini25pro": "Gemini 2.5 Pro (1M tokens)",
                "gemini25flash": "Gemini 2.5 Flash (1M tokens)"
            },
            "Anthropic (DEV only)": {
                "claudeopus41": "Claude Opus 4.1 (200K input)",
                "claudeopus4": "Claude Opus 4 (200K input)",
                "claudesonnet45": "Claude Sonnet 4.5 (200K input)",
                "claudesonnet4": "Claude Sonnet 4 (200K input)",
                "claudesonnet37": "Claude Sonnet 3.7 (200K input)"
            }
        }

    async def connect_to_multiple_servers(self, server_configs: List[Dict[str, str]]):
        self.sessions = {}
        
        for config in server_configs:
            name = config["name"]
            script_path = config["script_path"]
            
            try:
                command = "python" if script_path.endswith('.py') else "node"
                server_params = StdioServerParameters(
                    command=command,
                    args=[script_path],
                    env=None
                )
                
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                stdio, write = stdio_transport
                session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                await session.initialize()
                
                self.sessions[name] = session
                
                response = await session.list_tools()
                tools = response.tools
                print(f"✓ Connected to {name} server with tools: {[tool.name for tool in tools]}")
                
            except Exception as e:
                print(f"✗ Failed to connect to {name} server: {e}")
        
        if "midas" in self.sessions:
            self.session = self.sessions["midas"]

    def _convert_tools_to_claude_format(self, openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Claude tool format"""
        claude_tools = []
        for tool in openai_tools:
            if tool.get("type") == "function":
                func = tool["function"]
                claude_tool = {
                    "name": func["name"],
                    "description": func["description"],
                    "input_schema": func.get("parameters", {})
                }
                claude_tools.append(claude_tool)
        return claude_tools

    def _prepare_argo_payload(self, messages: List[Dict[str, str]], model: str, tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "user": self.anl_username,
            "model": model,
            "messages": messages
        }

        # Claude Sonnet 4.5 does NOT accept both temperature and top_p
        # Other Claude models require both
        if model == "claudesonnet45":
            payload["temperature"] = 0.7
            # Do not include top_p for Claude Sonnet 4.5
        elif model.startswith("claude"):
            payload["temperature"] = 0.7
            payload["top_p"] = 0.9
        else:
            # OpenAI and Google models accept both
            payload["temperature"] = 0.7
            payload["top_p"] = 0.9

        # Set max_tokens based on model
        if model.startswith("claude"):
            payload["max_tokens"] = 21000
        elif model.startswith("gpt4o"):
            payload["max_tokens"] = 16000
        else:
            payload["max_tokens"] = 4000

        if tools:
            # Claude models use a different tool format than OpenAI
            if model.startswith("claude"):
                payload["tools"] = self._convert_tools_to_claude_format(tools)
                payload["tool_choice"] = {"type": "auto"}
            else:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

        return payload

    async def call_argo_chat_api(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = self._prepare_argo_payload(messages, self.selected_model, tools)

        try:
            response = await self.http_client.post(
                self.argo_chat_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            response.raise_for_status()
            result = response.json()
            
            # Removed debug output for cleaner interface
            # print(f"\n🔧 DEBUG: Argo API Response Structure")
            # print(f"  Response keys: {list(result.keys())}")
            
            # Handle Argo's actual format: {"response": {"content": ..., "tool_calls": [...]}}
            if 'response' in result and isinstance(result['response'], dict):
                response_obj = result['response']
                # print(f"  Response object keys: {list(response_obj.keys())}")

                # Check for tool calls in Argo format
                if 'tool_calls' in response_obj and response_obj['tool_calls']:
                    pass # print(f"  ✅ Argo native tool calls found: {len(response_obj['tool_calls'])}")
                    
                    # Convert Argo format to standard format for consistency
                    return {
                        'choices': [{
                            'message': {
                                'role': 'assistant',
                                'content': response_obj.get('content'),
                                'tool_calls': response_obj['tool_calls']
                            }
                        }]
                    }
                else:
                    # No tool calls, just content
                    pass # print(f"  ✗ No tool calls in Argo response")
                    
                    return {
                        'choices': [{
                            'message': {
                                'role': 'assistant',
                                'content': response_obj.get('content', ''),
                                '_argo_format': True
                            }
                        }]
                    }
            
            # Fallback: old format (single "response" string)
            elif 'response' in result and isinstance(result['response'], str):
                pass # print(f"  ⚠️  Legacy string format")
                
                return {
                    'choices': [{
                        'message': {
                            'role': 'assistant',
                            'content': result['response'],
                            '_legacy_format': True
                        }
                    }]
                }
            
            # Standard OpenAI format (shouldn't happen with Argo)
            elif 'choices' in result:
                return result

            else:
                pass # print(f"  ⚠️  Unexpected response format")
                return {
                    'choices': [{
                        'message': {
                            'role': 'assistant',
                            'content': str(result)
                        }
                    }]
                }
            
        except Exception as e:
            print(f"\n✗ Error calling Argo API: {str(e)}")
            raise Exception(f"Error calling Argo API: {str(e)}")

    async def get_all_available_tools(self) -> List[Dict[str, Any]]:
        all_tools = []
        
        for server_name, session in self.sessions.items():
            try:
                response = await session.list_tools()
                for tool in response.tools:
                    tool_info = {
                        "type": "function",
                        "function": {
                            "name": f"{server_name}_{tool.name}",
                            "description": f"[{server_name.upper()}] {tool.description}",
                            "parameters": tool.inputSchema
                        },
                        "server": server_name,
                        "original_name": tool.name
                    }
                    all_tools.append(tool_info)
            except Exception as e:
                print(f"✗ Error getting tools from {server_name}: {e}")
        
        # Removed debug output for cleaner interface
        # print(f"\n🔧 DEBUG: Available tools: {len(all_tools)}")
        # midas_tools = [t for t in all_tools if t['function']['name'].startswith('midas_')]
        # print(f"  MIDAS tools ({len(midas_tools)}):")
        # for tool in midas_tools:
        #     print(f"    - {tool['function']['name']}")
        # other_tools = [t for t in all_tools if not t['function']['name'].startswith('midas_')]
        # print(f"  Other tools ({len(other_tools)}): {[t['function']['name'] for t in other_tools[:5]]}")
        
        return all_tools

    async def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        # Clean output - just show what we're doing
        print(f"\n→ {tool_name.replace('midas_', '').replace('_', ' ').title()}")

        # ===== ERROR PREVENTION =====
        # Validate parameters before execution
        if "integrate_2d_to_1d" in tool_name:
            is_valid, error_msg = self.error_preventor.validate_integration_params(arguments)
            if not is_valid:
                print(f"✗ Validation Error: {error_msg}")
                return json.dumps({
                    "status": "validation_error",
                    "error": error_msg,
                    "suggestion": "Please check your parameters and try again"
                })

        elif "ff_hedm" in tool_name:
            is_valid, error_msg = self.error_preventor.validate_ff_hedm_params(arguments)
            if not is_valid:
                print(f"✗ Validation Error: {error_msg}")
                return json.dumps({
                    "status": "validation_error",
                    "error": error_msg,
                    "suggestion": "Please check your parameters and try again"
                })

        # ===== SMART CACHING =====
        # Check cache for expensive read-only operations
        cacheable_operations = ["filesystem_read_file", "filesystem_list_directory"]
        if tool_name in cacheable_operations:
            cached_result = self.cache.get(tool_name, arguments)
            if cached_result:
                print(" (from cache)")
                return cached_result

        # Parse server_name and tool_name
        # Tool names might already include the server prefix (e.g., "midas_auto_calibrate")
        # We need to identify the correct server and pass the full tool name to it
        server_name = None
        original_tool_name = tool_name

        # Try to match against known server names
        for srv_name in self.sessions.keys():
            if tool_name.startswith(f"{srv_name}_"):
                server_name = srv_name
                # Keep the full tool name as registered in the MCP server
                # Don't strip the prefix because the tool is registered with it
                original_tool_name = tool_name
                break

        if not server_name:
            # Fallback: assume it's a midas tool
            server_name = "midas"
            original_tool_name = tool_name

        if server_name not in self.sessions:
            return f"Error: Server '{server_name}' not connected"

        try:
            session = self.sessions[server_name]
            result = await session.call_tool(original_tool_name, arguments)
            result_text = str(result.content[0].text if result.content else "No result")

            # Cache result if applicable
            if tool_name in cacheable_operations:
                self.cache.set(tool_name, arguments, result_text)

            # Record analysis in context
            self.context.add_analysis(tool_name, result_text)

            # Add proactive suggestion to result
            suggestion = ProactiveSuggestions.get_suggestion(tool_name, result_text)
            if suggestion:
                result_text += f"\n\n{suggestion}"

            return result_text
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"✗ {error_msg}")
            return error_msg

    def _extract_peak_positions(self, text: str) -> List[float]:
        """Extract peak positions from text"""
        patterns = [
            r'\[([\d.,\s]+)\]',
            r'(?:peaks?\s+at\s+|positions?\s+)([\d.,\s]+)(?:\s+degrees?)?',
            r'((?:\d+\.?\d*[,\s]+)+\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                numbers_str = match.group(1)
                numbers = re.findall(r'\d+\.?\d*', numbers_str)
                try:
                    return [float(num) for num in numbers if float(num) > 0]
                except ValueError:
                    continue
        
        return []

    async def process_diffraction_query(self, query: str, image_path: str = None, experimental_params: Dict[str, Any] = None, max_iterations: int = 30, use_history: bool = True) -> str:
        """Process query with automatic tool calling loop"""

        if not self.sessions:
            return "Error: Not connected to any MCP servers."

        available_tools = await self.get_all_available_tools()

        system_prompt = """You are APEXA (Advanced Photon EXperiment Assistant), an expert AI scientist for synchrotron X-ray diffraction analysis at Argonne National Laboratory's Advanced Photon Source.
You maintain context across the conversation. When users refer to previous files, directories, or results using words like "there", "it", "that file", etc., use the conversation history to understand what they're referring to.

⚠️ CRITICAL INSTRUCTIONS ⚠️

1. WHEN TO USE TOOLS:
   - User gives a COMMAND: "integrate the file", "run workflow", "list files"
   - User provides DATA for analysis: "I have peaks at 12.5, 18.2 degrees"

2. WHEN NOT TO USE TOOLS:
   - User asks HOW: "how do you run analysis", "how does this work"
   - User asks WHAT: "what can you do", "what tools are available"
   - User needs EXPLANATION: "explain the workflow", "what is FF-HEDM"
   - General conversation: "hello", "thank you", etc.

3. TOOL CALLING FORMAT:
   Your API does NOT support native tool calling. You MUST use this special format:

   TOOL_CALL: exact_tool_name
   ARGUMENTS: {json_arguments}

   DO NOT just describe what you would do. DO NOT say "I would use the tool".
   ONLY use this format when user gives a COMMAND, not when they ask a QUESTION.

📋 EXAMPLES OF CORRECT RESPONSES:

Example 1 - Question (NO TOOL):
User: "how do you run the analysis"
Your response:
"To run analysis with the Beamline Assistant, you can:

1. FF-HEDM Full Workflow: Provide a directory with Parameters.txt and data files
2. 2D to 1D Integration: Provide image file (TIFF, GE2/GE5, ED5, EDF) and calibration parameters
   - Supports dark image subtraction for background correction
   - Can use calibration file or explicit geometry parameters
3. Phase Identification: Provide peak positions in degrees 2theta

For example, you can say 'Run FF-HEDM workflow on /path/to/data' or 'Integrate the .ge5 file from 2D to 1D with dark file'.

What specific analysis would you like to perform?"

Example 2 - Phase Identification Command (USE TOOL):
User: "I have peaks at 12.5, 18.2, 25.8 degrees. What phases?"
Your response:
"I'll identify the crystalline phases from these peak positions.

TOOL_CALL: midas_identify_crystalline_phases
ARGUMENTS: {"peak_positions": [12.5, 18.2, 25.8]}
"

Example 3 - FF-HEDM Workflow Command (USE TOOL):
User: "Run FF-HEDM workflow on ~/opt/MIDAS/FF_HEDM/Example"
Your response:
"I'll run the FF-HEDM full workflow.

TOOL_CALL: midas_run_ff_hedm_full_workflow
ARGUMENTS: {"example_dir": "~/opt/MIDAS/FF_HEDM/Example"}
"

Example 4 - 2D to 1D Integration Command (USE TOOL):
User: "Integrate the .tiff file from 2D to 1D in the current directory"
Your response:
"I'll integrate the 2D diffraction image to 1D.

TOOL_CALL: filesystem_list_directory
ARGUMENTS: {"path": "."}

[After seeing ff_011276_ge2_0001.tiff in results]

TOOL_CALL: midas_integrate_2d_to_1d
ARGUMENTS: {"image_path": "./ff_011276_ge2_0001.tiff", "calibration_file": "./Parameters.txt"}
"

Example 5 - Integration with Dark File Command (USE TOOL):
User: "I want to run MIDAS integration for /path/data.ge5 using /path/dark.ge5 as dark and /path/calib.txt"
Your response:
"I'll integrate the diffraction image with dark file subtraction.

TOOL_CALL: midas_integrate_2d_to_1d
ARGUMENTS: {"image_path": "/path/data.ge5", "calibration_file": "/path/calib.txt", "dark_file": "/path/dark.ge5"}
"

🔧 AVAILABLE TOOLS (MIDAS-NATIVE WORKFLOW):

⚠️  CRITICAL: Use MIDAS tools (not pyFAI) for calibration and integration!

- midas_auto_calibrate ⭐ PRIMARY CALIBRATION TOOL - Iterative geometric refinement
  Required: {"image_file": "CeO2.tif", "parameters_file": "Params.txt"}
  Optional: {"lsd_guess": 650000, "stopping_strain": 0.0001, "mult_factor": 2.5, "dark_file": "dark.tif"}
  Advanced: {"bc_x_guess": 1024, "bc_y_guess": 1024, "first_ring_nr": 1, "eta_bin_size": 5.0, "threshold": 500, "save_plots_hdf": "diag.h5", "image_transform": "2"}
  Uses: MIDAS AutoCalibrateZarr.py → CalibrantOMP (least-squares fitting with outlier rejection)
  Returns: Refined BC, Lsd, tilts (tx/ty/tz), distortion (p0-p3), convergence metrics
  Outputs: refined_MIDAS_params.txt (use for integration), autocal.log (iteration history)
  Defaults: stopping_strain=0.00004, mult_factor=2.5, auto-detect BC and Lsd from rings

- midas_integrate_2d_to_1d ⭐ UPDATED - Now uses MIDAS Integrator (not pyFAI!)
  Args: {"image_path": "/path/image.tif", "calibration_file": "calib.txt"}
  Or: {"image_path": "/path/image.tif", "wavelength": 0.22, "detector_distance": 1000, "beam_center_x": 1024, "beam_center_y": 1024}
  With dark subtraction: {"image_path": "/path/image.tif", "calibration_file": "calib.txt", "dark_file": "/path/dark.tif"}
  Uses: MIDAS Integrator executable (native MIDAS workflow)

- midas_identify_crystalline_phases
  Args: {"peak_positions": [12.5, 18.2]}

- midas_run_ff_hedm_full_workflow
  Args: {"example_dir": "~/path", "n_cpus": 20}

- midas_detect_diffraction_rings
  Args: {"image_path": "/path/image.tif"}

- filesystem_read_file
  Args: {"file_path": "/path/file"}

- filesystem_list_directory
  Args: {"path": "/path/dir"}

- executor_run_command
  Args: {"command": "ls -la"}

📊 TYPICAL MIDAS WORKFLOW:
1. Auto-calibrate with standard (CeO2, LaB6): midas_auto_calibrate
2. Get refined parameters (BC, distance, tilts)
3. Integrate 2D → 1D with calibrated params: midas_integrate_2d_to_1d
4. Analyze 1D pattern in GSAS-II or identify phases

⚠️ REMEMBER:
- ALWAYS use "TOOL_CALL:" and "ARGUMENTS:" format
- NEVER just describe what you would do
- NEVER say "I don't have access" - you DO have access via the TOOL_CALL format
- When user says "I want to run", "integrate", "calibrate", "analyze" - they are giving a COMMAND, USE TOOLS
- Tool names must be EXACT (case-sensitive)
- Arguments must be valid JSON
- If user provides file paths and asks to integrate/calibrate, IMMEDIATELY call midas_integrate_2d_to_1d"""

        system_prompt += f"\n\nCurrent Model: {self.selected_model} via Argo Gateway"

        # Build messages with conversation history
        if use_history and self.conversation_history:
            # Start with system prompt + history + new query
            messages = [{"role": "system", "content": system_prompt}] + self.conversation_history.copy()

            # Add current user query
            user_content = query
            if image_path:
                user_content += f"\n\nImage: {image_path}"
            if experimental_params:
                user_content += f"\n\nParameters: {json.dumps(experimental_params)}"

            messages.append({"role": "user", "content": user_content})
        else:
            # No history - fresh conversation
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]

            if image_path:
                messages[-1]["content"] += f"\n\nImage: {image_path}"
            if experimental_params:
                messages[-1]["content"] += f"\n\nParameters: {json.dumps(experimental_params)}"

        # Cleaner output
        # print(f"\n{'='*60}")
        # print(f"Processing query: {query[:100]}...")
        # print(f"{'='*60}")

        iteration = 0
        final_response = ""
        
        while iteration < max_iterations:
            iteration += 1
            # print(f"\n--- Iteration {iteration} ---")
            
            response_data = await self.call_argo_chat_api(messages, available_tools)
            
            if 'choices' not in response_data or not response_data['choices']:
                return f"Unexpected response format: {response_data}"
                
            choice = response_data['choices'][0]
            
            if 'message' not in choice:
                return f"Unexpected choice format: {choice}"
            
            message = choice['message']
            
            # Check if this is legacy format (no native tool calling)
            is_legacy = message.get('_legacy_format', False)
            is_argo = message.get('_argo_format', False)
            
            # Check for native tool calls (works with Argo format now)
            tool_calls = message.get('tool_calls', [])
            
            if tool_calls and not is_legacy:
                # Native tool calling! Process them
                # print(f"\n🔧 Processing {len(tool_calls)} native tool call(s)...")

                # For Claude, need to convert message format from OpenAI-style to Claude-style
                if self.selected_model.startswith("claude"):
                    # Convert tool_calls to content blocks for Claude
                    content_blocks = []

                    # Add any text content first
                    if message.get('content'):
                        content_blocks.append({
                            "type": "text",
                            "text": message['content']
                        })

                    # Add tool_use blocks
                    for tool_call in tool_calls:
                        tool_id = tool_call.get('id', 'unknown')

                        # Extract tool info
                        if 'function' in tool_call:
                            tool_name = tool_call['function'].get('name')
                            try:
                                tool_input = json.loads(tool_call['function'].get('arguments', '{}'))
                            except json.JSONDecodeError:
                                tool_input = {}
                        elif 'input' in tool_call:
                            tool_name = tool_call.get('name')
                            tool_input = tool_call['input']
                        else:
                            continue

                        content_blocks.append({
                            "type": "tool_use",
                            "id": tool_id,
                            "name": tool_name,
                            "input": tool_input
                        })

                    # Create properly formatted message for Claude
                    claude_message = {
                        "role": "assistant",
                        "content": content_blocks
                    }
                    messages.append(claude_message)
                else:
                    # OpenAI/Gemini: use message as-is
                    messages.append(message)

                for tool_call in tool_calls:
                    # Handle different formats from different providers
                    # OpenAI format: {"id": "...", "function": {"name": "...", "arguments": "{...}"}}
                    # Anthropic format: {"id": "...", "input": {...}}
                    # Gemini format: {"id": null, "args": {...}}
                    
                    tool_id = tool_call.get('id', 'unknown')
                    tool_name = None
                    arguments = {}
                    
                    if 'function' in tool_call:
                        # OpenAI format
                        function = tool_call['function']
                        tool_name = function.get('name')
                        try:
                            arguments = json.loads(function.get('arguments', '{}'))
                        except json.JSONDecodeError:
                            arguments = {}
                    elif 'input' in tool_call:
                        # Anthropic format
                        tool_name = tool_call.get('name')
                        arguments = tool_call['input']
                    elif 'args' in tool_call:
                        # Gemini format
                        tool_name = tool_call.get('name')
                        arguments = tool_call['args']
                    
                    if not tool_name:
                        print(f"  ⚠️  Could not extract tool name from: {tool_call}")
                        continue
                    
                    print(f"\n  Calling: {tool_name}")
                    print(f"  Arguments: {json.dumps(arguments, indent=4)}")
                    
                    # Execute the tool
                    tool_result = await self.execute_tool_call(tool_name, arguments)

                    # Add tool result to conversation
                    # Claude expects different format than OpenAI
                    if self.selected_model.startswith("claude"):
                        # Claude format: role=user with tool_result content block
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": tool_result
                                }
                            ]
                        })
                    else:
                        # OpenAI/Gemini format: role=tool with simple content
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": tool_result
                        })
                
                # Continue loop to get AI's interpretation
                continue
            
            if is_legacy or 'tool_calls' not in message:
                # Parse text-based tool calls
                content = message.get('content', '')
                
                # Look for TOOL_CALL format
                if 'TOOL_CALL:' in content:
                    # print(f"  💡 Detected text-based tool call")
                    
                    # Extract tool name and arguments
                    tool_match = re.search(r'TOOL_CALL:\s*(\w+)', content)
                    args_match = re.search(r'ARGUMENTS:\s*({[^}]+})', content)
                    
                    if tool_match:
                        tool_name = tool_match.group(1)
                        
                        if args_match:
                            try:
                                arguments = json.loads(args_match.group(1))
                            except json.JSONDecodeError:
                                arguments = {}
                        else:
                            arguments = {}
                        
                        print(f"  Extracted tool: {tool_name}")
                        print(f"  Arguments: {arguments}")
                        
                        # Execute the tool
                        tool_result = await self.execute_tool_call(tool_name, arguments)
                        
                        # Add assistant message and tool result
                        messages.append(message)
                        messages.append({
                            "role": "user",
                            "content": f"Tool result:\n{tool_result}\n\nPlease provide a natural language summary of these results."
                        })
                        
                        # Continue loop to get interpretation
                        continue
                
                # Fallback: Try to detect tool intent from text (ONLY if very specific)
                # This should be rare - AI should use TOOL_CALL format

                # Check if AI mentioned a specific tool with clear action intent
                tool_intent = None
                tool_args = {}

                # Only trigger fallback if it's a CLEAR command-like statement from the AI
                # Not if it's explaining or asking questions
                is_explanation = any(word in content.lower() for word in [
                    'how to', 'you can', 'would', 'could', 'should', 'explain',
                    'here\'s', 'let me', 'i can', 'to run', 'please', '?'
                ])

                if not is_explanation:
                    # Pattern 1: Very specific FF-HEDM command
                    if re.search(r'run.*ff[_-]?hedm.*workflow.*on', content.lower()):
                        tool_intent = 'midas_run_ff_hedm_full_workflow'
                        # Try to extract directory
                        dir_match = re.search(r'on\s+([~/][\w/.-]+)', content)
                        if dir_match:
                            tool_args = {"example_dir": dir_match.group(1)}

                    # Pattern 2: Identify phases with peak data
                    elif re.search(r'identif.*phase.*from.*peak', content.lower()):
                        # Extract peak positions from earlier messages
                        for msg in reversed(messages):
                            if msg['role'] == 'user':
                                peaks = self._extract_peak_positions(msg['content'])
                                if peaks:
                                    tool_intent = 'midas_identify_crystalline_phases'
                                    tool_args = {"peak_positions": peaks}
                                    break

                    # Pattern 3: Integrate specific file
                    elif re.search(r'integrat.*(?:file|image).*(?:from|to)', content.lower()):
                        # Look for image path in recent messages
                        for msg in reversed(messages):
                            if msg['role'] == 'user':
                                image_match = re.search(r'([~/.\w/-]+\.(?:tiff?|ge2|edf))', msg['content'])
                                if image_match:
                                    tool_intent = 'midas_integrate_2d_to_1d'
                                    tool_args['image_path'] = image_match.group(1)
                                    # Look for calibration file
                                    calib_match = re.search(r'(?:with|using)\s+([\w.-]+\.txt)', msg['content'])
                                    if calib_match:
                                        tool_args['calibration_file'] = calib_match.group(1)
                                    break

                if tool_intent and tool_args:  # Only execute if we have both intent AND arguments
                    # Comment out debug output for cleaner interface
                    # print(f"  💡 Detected tool intent: {tool_intent}")
                    # print(f"     Extracted args: {tool_args}")
                    # print(f"  ⚠️  AI didn't use TOOL_CALL format - executing anyway...")

                    # Execute the tool
                    tool_result = await self.execute_tool_call(tool_intent, tool_args)

                    # Add assistant message and tool result
                    messages.append(message)
                    messages.append({
                        "role": "user",
                        "content": f"Tool result:\n{tool_result}\n\nPlease provide a natural language summary."
                    })

                    # Continue loop
                    continue
                
                # No tool call detected - this is final response
                final_response = content
                # print(f"\n✓ Response complete")
                break
            
            # Native tool calls (if Argo supports them)
            messages.append(message)
            tool_calls = message.get('tool_calls', [])
            
            if not tool_calls:
                final_response = message.get('content', '')
                print(f"\n✓ Final response received (no more tool calls)")
                break
            
            # Execute all tool calls
            print(f"\n🔧 Processing {len(tool_calls)} tool call(s)...")
            
            for tool_call in tool_calls:
                tool_id = tool_call.get('id', 'unknown')
                function = tool_call.get('function', {})
                tool_name = function.get('name')
                
                try:
                    arguments = json.loads(function.get('arguments', '{}'))
                except json.JSONDecodeError:
                    arguments = {}
                
                print(f"\n  Calling: {tool_name}")
                
                # Execute the tool
                tool_result = await self.execute_tool_call(tool_name, arguments)
                
                # Add tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": tool_result
                })
        
        if iteration >= max_iterations:
            print(f"\n⚠️  Reached maximum iterations ({max_iterations})")

        # Get final response
        final_text = final_response if final_response else messages[-1].get('content', 'No response generated')

        # Update conversation history (keep last 10 exchanges to avoid token overflow)
        if use_history:
            # Add user query
            user_msg = {"role": "user", "content": query}
            if image_path:
                user_msg["content"] += f"\n\nImage: {image_path}"
            if experimental_params:
                user_msg["content"] += f"\n\nParameters: {json.dumps(experimental_params)}"

            # Add assistant response
            assistant_msg = {"role": "assistant", "content": final_text}

            self.conversation_history.append(user_msg)
            self.conversation_history.append(assistant_msg)

            # Keep only last 20 messages (10 exchanges) to avoid context overflow
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

        return final_text

    def show_available_models(self):
        print("\nAvailable Argo Models:")
        print("=" * 50)
        
        for provider, models in self.available_models.items():
            print(f"\n{provider}:")
            for model_id, description in models.items():
                status = "✅" if model_id == self.selected_model else "  "
                print(f"{status} {model_id:15} - {description}")

    def _is_valid_model(self, model_name: str) -> bool:
        for provider, models in self.available_models.items():
            if model_name in models:
                return True
        return False

    async def interactive_analysis_session(self):
        print(f"\n╔══════════════════════════════════════════════════════════════╗")
        print(f"║  APEXA - Advanced Photon EXperiment Assistant               ║")
        print(f"║  Your AI Scientist at the Beamline                          ║")
        print(f"╚══════════════════════════════════════════════════════════════╝")
        print()
        print(f"🤖 AI Model: {self.selected_model}")
        print(f"👤 User: {self.anl_username}")
        print(f"🔌 Servers: {', '.join(list(self.sessions.keys()))}")
        print()
        print("Commands: analyze, batch, workflow, session, image, monitor, models, tools, clear, help, quit")
        print()
        
        # Command history
        history = []
        history_index = -1
        
        while True:
            try:
                # Use input with readline support for history and tab completion
                import readline
                
                # Set up history
                readline.clear_history()
                for cmd in history:
                    readline.add_history(cmd)
                
                user_input = input("APEXA> ").strip()
                
                if not user_input:
                    continue
                    
                # Add to history
                if user_input and (not history or history[-1] != user_input):
                    history.append(user_input)
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("✓ Conversation history cleared")
                elif user_input.lower() == 'models':
                    self.show_available_models()
                elif user_input.lower() == 'servers':
                    print(f"Connected: {list(self.sessions.keys())}")

                # ===== NEW SMART COMMANDS =====
                elif user_input.lower().startswith('batch '):
                    # Batch processing command
                    # Example: batch integrate *.ge5 with calib.txt dark.tif
                    parts = user_input[6:].strip().split()
                    if len(parts) < 2:
                        print("Usage: batch integrate <pattern> with <calibration_file> [dark_file]")
                        continue

                    # operation = parts[0]  # e.g., "integrate" - reserved for future
                    pattern = parts[1]    # e.g., "*.ge5"

                    # Parse additional arguments
                    calibration_file = None
                    dark_file = None

                    if 'with' in parts:
                        with_idx = parts.index('with')
                        calibration_file = parts[with_idx + 1] if len(parts) > with_idx + 1 else None
                        dark_file = parts[with_idx + 2] if len(parts) > with_idx + 2 else None

                    # Find files matching pattern
                    from glob import glob
                    files = glob(pattern)

                    if not files:
                        print(f"No files found matching: {pattern}")
                        continue

                    print(f"Found {len(files)} files to process")
                    confirm = input(f"Process all {len(files)} files? (yes/no): ")

                    if confirm.lower() in ['yes', 'y']:
                        kwargs = {}
                        if calibration_file:
                            kwargs['calibration_file'] = calibration_file
                        if dark_file:
                            kwargs['dark_file'] = dark_file

                        results = await self.batch_processor.process_batch(
                            self,
                            "midas_integrate_2d_to_1d",
                            files,
                            **kwargs
                        )

                        print(f"\n{'='*60}")
                        print(f"Batch Processing Complete:")
                        print(f"  Total: {results['total_files']}")
                        print(f"  ✓ Successful: {results['successful']}")
                        print(f"  ✗ Failed: {results['failed']}")
                        print(f"{'='*60}\n")

                elif user_input.lower().startswith('workflow '):
                    # Workflow command
                    # Example: workflow phase_analysis or workflow list
                    workflow_cmd = user_input[9:].strip()

                    if workflow_cmd == 'list':
                        print("\nAvailable Workflows:")
                        print("="*50)
                        for name, steps in self.workflow_builder.workflows.items():
                            print(f"\n{name}:")
                            for i, step in enumerate(steps, 1):
                                print(f"  {i}. {step['description']}")
                    else:
                        workflow = self.workflow_builder.get_workflow(workflow_cmd)
                        if workflow:
                            print(f"\nExecuting workflow: {workflow_cmd}")
                            print("="*50)
                            for i, step in enumerate(workflow, 1):
                                print(f"\nStep {i}: {step['description']}")
                                # Note: Would need user input for arguments
                                print(f"  Tool: {step['tool']}")
                            print("\nNote: Use natural language queries to execute workflows with your data")
                        else:
                            print(f"Unknown workflow: {workflow_cmd}")
                            print("Use 'workflow list' to see available workflows")

                elif user_input.lower().startswith('session '):
                    # Session management
                    # Example: session save my_experiment, session load my_experiment, session list
                    session_cmd = user_input[8:].strip().split()

                    if not session_cmd:
                        print("Usage: session <save|load|list> [name]")
                        continue

                    action = session_cmd[0]

                    if action == 'save':
                        session_name = session_cmd[1] if len(session_cmd) > 1 else None
                        saved_file = self.context.save_session(session_name)
                        print(f"✓ Session saved: {saved_file}")

                    elif action == 'load':
                        if len(session_cmd) < 2:
                            print("Usage: session load <session_name>")
                            continue
                        session_name = session_cmd[1]
                        if self.context.load_session(session_name):
                            print(f"✓ Session loaded: {session_name}")
                            print(self.context.get_summary())
                        else:
                            print(f"✗ Session not found: {session_name}")

                    elif action == 'list':
                        sessions = self.context.list_sessions()
                        if sessions:
                            print("\nSaved Sessions:")
                            for session in sessions:
                                print(f"  - {session}")
                        else:
                            print("No saved sessions found")

                    elif action == 'summary':
                        print("\nCurrent Session:")
                        print(self.context.get_summary())

                    else:
                        print(f"Unknown session command: {action}")
                        print("Available: save, load, list, summary")

                elif user_input.lower().startswith('image '):
                    # Image analysis command
                    # Example: image analyze sample.ge5, image quality sample.ge5
                    # Also handles: "image quality of the file.ge5", "image quality for file.ge5"
                    cmd_text = user_input[6:].strip()

                    # Extract action (analyze, quality, rings)
                    action = None
                    for possible_action in ['analyze', 'quality', 'rings']:
                        if cmd_text.lower().startswith(possible_action):
                            action = possible_action
                            # Remove action from text
                            cmd_text = cmd_text[len(possible_action):].strip()
                            break

                    if not action:
                        print("Usage: image <analyze|quality|rings> <image_path>")
                        print("Examples:")
                        print("  image quality sample.ge5")
                        print("  image quality of the .tiff file")
                        print("  image analyze data.ge2")
                        continue

                    # Remove common filler words to find the actual file
                    filler_words = ['of', 'the', 'for', 'in', 'file', 'image', 'this', 'directory', 'a', 'an']
                    words = cmd_text.split()

                    # Find file extensions in the text
                    image_path = None
                    for word in words:
                        # Check if it looks like a file path or has an extension
                        if '.' in word and any(word.endswith(ext) for ext in ['.tif', '.tiff', '.ge2', '.ge5', '.ed5', '.edf']):
                            image_path = word
                            break
                        # Check if it contains a path separator
                        if '/' in word or word.startswith('~'):
                            image_path = word
                            break

                    # If no explicit path found, try to find files with mentioned extension
                    if not image_path:
                        # Look for extension mentions like ".tiff" or ".ge5"
                        for word in words:
                            if word.startswith('.'):
                                # Find files with this extension in current directory
                                from glob import glob
                                ext = word
                                matching_files = glob(f'*{ext}')
                                if matching_files:
                                    image_path = matching_files[0]
                                    print(f"Found: {image_path}")
                                    break

                    if not image_path:
                        # Try to use any word that's not a filler word
                        for word in words:
                            if word.lower() not in filler_words:
                                image_path = word
                                break

                    if not image_path:
                        print("Could not find image file in command")
                        print("Please specify the image file name")
                        print("Examples:")
                        print("  image quality sample.ge5")
                        print("  image quality /path/to/data.tiff")
                        continue

                    if action == 'analyze':
                        print(f"\n📸 Analyzing image: {image_path}")
                        summary = self.image_analyzer.create_image_summary(image_path)
                        print(summary)

                    elif action == 'quality':
                        print(f"\n🔍 Quality check: {image_path}")
                        quality = self.image_analyzer.analyze_image_quality(image_path)
                        if 'error' in quality:
                            print(f"✗ Error: {quality['error']}")
                        else:
                            print(f"  Overall Quality: {quality['overall_quality']}")
                            print(f"  Signal-to-Noise: {quality['signal_to_noise']:.1f}")
                            print(f"  Saturation: {quality['saturation_percent']:.2f}%")
                            if quality['issues']:
                                print(f"  Issues:")
                                for issue in quality['issues']:
                                    print(f"    • {issue}")

                    elif action == 'rings':
                        print(f"\n🔍 Ring detection: {image_path}")
                        rings = self.image_analyzer.detect_rings_visual(image_path)
                        if 'error' in rings:
                            print(f"✗ Error: {rings['error']}")
                        else:
                            print(f"  Rings Detected: {rings['rings_detected']}")
                            print(f"  Ring Radii (pixels): {rings['ring_radii_pixels']}")
                            print(f"  Assessment: {rings['quality']}")

                    else:
                        print(f"Unknown image command: {action}")
                        print("Available: analyze, quality, rings")

                elif user_input.lower().startswith('monitor '):
                    # Real-time monitoring command
                    # Example: monitor start /data/experiment, monitor stop, monitor status
                    parts = user_input[8:].strip().split()
                    if not parts:
                        print("Usage: monitor <start|stop|status|check> [directory]")
                        continue

                    action = parts[0]

                    if action == 'start':
                        if len(parts) < 2:
                            print("Usage: monitor start <directory>")
                            continue
                        directory = parts[1]
                        result = self.realtime_feedback.start_monitoring(directory)
                        print(f"\n🔄 {result['message']}")
                        print(f"   Checking every {result['check_interval']} seconds")
                        print(f"   Press Ctrl+C to stop or use 'monitor stop'")

                    elif action == 'stop':
                        result = self.realtime_feedback.stop_monitoring()
                        print(f"\n⏹️  Monitoring stopped")
                        print(f"   Files processed: {result['files_processed']}")
                        print(f"   Alerts generated: {result['alerts_generated']}")

                    elif action == 'status':
                        summary = self.realtime_feedback.get_session_summary()
                        print(f"\n📊 Monitoring Status:")
                        print(f"   Active: {summary['monitoring_active']}")
                        if summary['monitoring_active']:
                            print(f"   Directory: {summary['directory']}")
                        print(f"   Files Processed: {summary['files_processed']}")
                        print(f"   Total Alerts: {summary['total_alerts']}")
                        print(f"     ⚠️  Warnings: {summary['warning_alerts']}")
                        print(f"     🚨 Critical: {summary['critical_alerts']}")

                        if summary['recent_alerts']:
                            print(f"\n   Recent Alerts:")
                            for alert in summary['recent_alerts']:
                                icon = "🚨" if alert['level'] == 'CRITICAL' else "⚠️" if alert['level'] == 'WARNING' else "ℹ️"
                                print(f"     {icon} {alert['message']}")

                    elif action == 'check':
                        new_files = self.realtime_feedback.check_new_files()
                        if not new_files:
                            print("\n✓ No new files detected")
                        else:
                            print(f"\n🆕 Found {len(new_files)} new file(s):")
                            for analysis in new_files:
                                print(f"\n  📁 {analysis['file']}")
                                print(f"     Quality: {analysis['quality']['overall_quality']}")
                                print(f"     Rings: {analysis['rings']['rings_detected']}")
                                if analysis['alerts']:
                                    for alert in analysis['alerts']:
                                        icon = "🚨" if alert['level'] == 'CRITICAL' else "⚠️" if alert['level'] == 'WARNING' else "ℹ️"
                                        print(f"     {icon} {alert['message']}")

                    else:
                        print(f"Unknown monitor command: {action}")
                        print("Available: start, stop, status, check")

                elif user_input.lower().startswith('plot '):
                    # Plotting command
                    # Examples: plot 2d sample.ge5, plot radial data.tiff, plot 1d pattern.dat
                    #           plot compare file1.dat file2.dat file3.dat
                    cmd_text = user_input[5:].strip()

                    # Extract plot type
                    plot_type = None
                    for possible_type in ['2d', 'radial', '1d', 'pattern', 'compare', 'comparison']:
                        if cmd_text.lower().startswith(possible_type):
                            plot_type = possible_type
                            cmd_text = cmd_text[len(possible_type):].strip()
                            break

                    if not plot_type:
                        print("Usage: plot <2d|radial|1d|compare> <file(s)>")
                        print("Examples:")
                        print("  plot 2d sample.ge5           - Plot 2D diffraction image")
                        print("  plot radial data.tiff        - Plot radial intensity profile")
                        print("  plot 1d pattern.dat          - Plot 1D integrated pattern")
                        print("  plot compare file1.dat file2.dat - Compare multiple patterns")
                        continue

                    # Parse file path(s)
                    files = cmd_text.split()
                    if not files:
                        print("Please specify file(s) to plot")
                        continue

                    # Handle different plot types
                    if plot_type == '2d':
                        if len(files) != 1:
                            print("2D plot requires exactly one image file")
                            continue

                        print(f"\n📊 Plotting 2D image: {files[0]}")
                        result = self.plotting.plot_2d_image(files[0])

                        if result['status'] == 'success':
                            print(f"✓ Plot saved: {result['plot_saved']}")
                            print(f"  Statistics:")
                            print(f"    Mean: {result['statistics']['mean']:.1f}")
                            print(f"    Max: {result['statistics']['max']:.1f}")
                            print(f"    Std: {result['statistics']['std']:.1f}")
                        else:
                            print(f"✗ Error: {result['error']}")

                    elif plot_type == 'radial':
                        if len(files) != 1:
                            print("Radial plot requires exactly one image file")
                            continue

                        print(f"\n📊 Plotting radial profile: {files[0]}")
                        result = self.plotting.plot_radial_profile(files[0])

                        if result['status'] == 'success':
                            print(f"✓ {result['message']}")
                            print(f"  Plot saved: {result['plot_saved']}")
                        else:
                            print(f"✗ Error: {result['error']}")

                    elif plot_type in ['1d', 'pattern']:
                        if len(files) != 1:
                            print("1D pattern plot requires exactly one data file")
                            continue

                        print(f"\n📊 Plotting 1D pattern: {files[0]}")
                        result = self.plotting.plot_1d_pattern(files[0])

                        if result['status'] == 'success':
                            print(f"✓ {result['message']}")
                            print(f"  Plot saved: {result['plot_saved']}")
                        else:
                            print(f"✗ Error: {result['error']}")

                    elif plot_type in ['compare', 'comparison']:
                        if len(files) < 2:
                            print("Comparison requires at least 2 pattern files")
                            continue

                        print(f"\n📊 Comparing {len(files)} patterns...")
                        result = self.plotting.plot_comparison(files)

                        if result['status'] == 'success':
                            print(f"✓ {result['message']}")
                            print(f"  Plot saved: {result['plot_saved']}")
                        else:
                            print(f"✗ Error: {result['error']}")

                elif user_input.lower() == 'tools':
                    tools = await self.get_all_available_tools()
                    print(f"\nAvailable tools ({len(tools)}):")
                    for tool in tools:
                        print(f"  - {tool['function']['name']}: {tool['function']['description'][:80]}")
                elif user_input.lower() == 'help':
                    print("""
APEXA Smart Commands:

📊 Analysis & Processing:
  analyze <query>                      - Run AI-powered analysis
  batch integrate <pattern> with ...   - Process multiple files at once
  workflow list                        - Show available workflows
  workflow <name>                      - Execute predefined workflow

📸 Image Analysis (Multimodal):
  image analyze <file>                 - Full image analysis with AI
  image quality <file>                 - Check signal, noise, saturation
  image rings <file>                   - Detect diffraction rings

📈 Plotting & Visualization:
  plot 2d <file>                       - Plot 2D diffraction image
  plot radial <file>                   - Plot radial intensity profile
  plot 1d <file>                       - Plot 1D integrated pattern
  plot compare <file1> <file2> ...     - Compare multiple patterns

🔄 Real-time Monitoring:
  monitor start <directory>            - Start watching for new images
  monitor stop                         - Stop monitoring
  monitor status                       - Show monitoring stats
  monitor check                        - Check for new files now

💾 Session Management:
  session save [name]                  - Save current session
  session load <name>                  - Load saved session
  session list                         - List all saved sessions
  session summary                      - Show current session info

🔧 Tools & Configuration:
  models                               - Show available AI models
  model <name>                         - Switch AI model
  tools                                - List all analysis tools
  servers                              - Show connected servers
  clear                                - Clear conversation history
  help                                 - Show this help
  quit                                 - Exit APEXA

💡 Natural Language Examples:
  • "Integrate data.ge5 with dark.ge5 using calib.txt"
  • "I have peaks at 12.5, 18.2, 25.8 degrees. What phases?"
  • "Run FF-HEDM workflow in /path/to/data"
  • "Analyze the diffraction rings in image.tif"
  • "Plot the radial profile of sample.ge5"

✨ Smart Features:
  • Multimodal image analysis - AI can "see" your images!
  • Advanced plotting & visualization with matplotlib
  • Real-time feedback during beamtime
  • Automatic error prevention and validation
  • Proactive next-step suggestions after each analysis
  • Session persistence with auto-save
  • Smart caching for faster repeated operations
  • Batch processing for multiple files

Use ↑/↓ arrows for command history | Tab for completion
                    """)
                elif user_input.startswith('model '):
                    model_name = user_input[6:].strip()
                    if self._is_valid_model(model_name):
                        self.selected_model = model_name

                        # Update endpoint based on model environment requirements
                        if self.selected_model in self.dev_only_models:
                            self.argo_chat_url = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"
                            self.argo_embed_url = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/embed/"
                            self.environment = "DEV"
                            print(f"✅ Switched to: {model_name} (using DEV environment)")
                        else:
                            self.argo_chat_url = "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/"
                            self.argo_embed_url = "https://apps.inside.anl.gov/argoapi/api/v1/resource/embed/"
                            self.environment = "PROD"
                            print(f"✅ Switched to: {model_name} (using PROD environment)")
                    else:
                        print(f"✗ Invalid model: {model_name}")
                        print("Available models: models")
                elif user_input.startswith('run '):
                    command = user_input[4:].strip()
                    
                    # Check if this is a special command like FF-HEDM
                    if 'ff-hedm' in command.lower() or 'ff_hedm' in command.lower():
                        # Extract directory if provided
                        dir_match = re.search(r'in\s+([~/.\w/-]+)', command)
                        if dir_match:
                            example_dir = dir_match.group(1)
                        else:
                            example_dir = "~/opt/MIDAS/FF_HEDM/Example"
                        
                        # Extract CPU count if provided  
                        cpu_match = re.search(r'(\d+)\s*cpu', command.lower())
                        n_cpus = int(cpu_match.group(1)) if cpu_match else None
                        
                        print(f"Running FF-HEDM workflow in {example_dir}...")
                        if "midas" in self.sessions:
                            result = await self.execute_tool_call(
                                "midas_run_ff_hedm_full_workflow",
                                {"example_dir": example_dir, "n_cpus": n_cpus}
                            )
                            print(f"\n{result}\n")
                        else:
                            print("MIDAS server not connected")
                    
                    elif 'integrator' in command.lower() or 'batch' in command.lower():
                        # Integrator batch command
                        print("Use the interactive query instead:")
                        print('Beamline> Run batch integration on /path/to/data with calib_file.txt')
                    
                    else:
                        # Regular shell command
                        if "executor" in self.sessions:
                            result = await self.execute_tool_call("executor_run_command", {"command": command})
                            print(f"\n{result}\n")
                        else:
                            print("Executor server not connected")
                elif user_input.startswith('ls '):
                    path = user_input[3:].strip()
                    if "filesystem" in self.sessions:
                        result = await self.execute_tool_call("filesystem_list_directory", {"path": path})
                        print(f"\n{result}\n")
                    else:
                        print("Filesystem server not connected")
                elif user_input:
                    response = await self.process_diffraction_query(user_input)
                    print(f"\n{response}\n")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except EOFError:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

    async def cleanup(self):
        await self.http_client.aclose()
        await self.exit_stack.aclose()

async def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python argo_mcp_client.py <server_configs...>")
        sys.exit(1)

    client = APEXAClient()
    
    try:
        server_configs = []
        
        for arg in sys.argv[1:]:
            if ":" in arg:
                name, script_path = arg.split(":", 1)
                server_configs.append({"name": name, "script_path": script_path})
            else:
                server_configs.append({"name": "midas", "script_path": arg})
        
        await client.connect_to_multiple_servers(server_configs)
        
        if not client.sessions:
            print("Failed to connect to any servers")
            sys.exit(1)
            
        await client.interactive_analysis_session()
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())