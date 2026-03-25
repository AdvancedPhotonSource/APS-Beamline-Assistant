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
import sys
import re
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
from apexa_agents import ArgoProvider, OrchestratorAgent

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
        """Validate parameters for midas_integrate_2d_to_1d (v10).

        Returns:
            (is_valid, error_message)
        """
        image_file = args.get("image_file")
        calibration_file = args.get("calibration_file")
        dark_file = args.get("dark_file")

        # Check image exists
        if image_file:
            img_path = Path(image_file).expanduser()
            if not img_path.exists():
                return False, f"Image file not found: {image_file}"
            valid_extensions = ['.tif', '.tiff', '.ge', '.ge1', '.ge2', '.ge3', '.ge4', '.ge5',
                                 '.h5', '.hdf5', '.hdf', '.nxs', '.zip']
            if not any(str(img_path).lower().endswith(ext) for ext in valid_extensions):
                return False, f"Unsupported image format. Use: {', '.join(valid_extensions)}"

        # Calibration file is optional — auto-detected if omitted
        if calibration_file:
            cal_path = Path(calibration_file).expanduser()
            if not cal_path.exists():
                return False, f"Calibration file not found: {calibration_file}"

        if dark_file:
            dark_path = Path(dark_file).expanduser()
            if not dark_path.exists():
                return False, f"Dark file not found: {dark_file}"

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

        self.environment = "DEV" if self.selected_model in self.dev_only_models else "PROD"

        # Phase 2: tool registry (built at connection time, not per-query)
        self._tool_registry:   Dict[str, str]       = {}   # bare_tool_name → server_name
        self._available_tools: List[Dict[str, Any]] = []   # pre-built tool definitions
        self.orchestrator:     Optional[OrchestratorAgent] = None

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
                # Use venv Python if available, otherwise fall back to system python
                if script_path.endswith('.py'):
                    venv_python = Path(".venv/bin/python3")
                    command = str(venv_python) if venv_python.exists() else "python3"
                else:
                    command = "node"

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

                # Build tool registry for this server in same pass
                response = await session.list_tools()
                for tool in response.tools:
                    self._tool_registry[tool.name] = name
                    self._available_tools.append({
                        "type": "function",
                        "function": {
                            "name":        tool.name,
                            "description": f"[{name.upper()}] {tool.description}",
                            "parameters":  tool.inputSchema,
                        },
                    })
                print(f"  {name}: {len(response.tools)} tools")

            except Exception as e:
                print(f"  {name}: FAILED ({e})")

        if "midas" in self.sessions:
            self.session = self.sessions["midas"]

        # Initialise the orchestrator (Phase 2 core)
        self.orchestrator = OrchestratorAgent(
            execute_tool_fn=self.execute_tool_call,
            all_tools=self._available_tools,
            context=self.context,
        )

    async def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:

        # ===== ERROR PREVENTION =====
        # Validate parameters before execution
        if "ff_hedm" in tool_name:
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

        # Registry lookup (built once at connection time — O(1), no live RPCs)
        server_name = self._tool_registry.get(tool_name, "midas")

        if server_name not in self.sessions:
            return f"Error: Server '{server_name}' not connected"

        try:
            session = self.sessions[server_name]
            result = await session.call_tool(tool_name, arguments)
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

    async def run_query(self, query: str, use_history: bool = True) -> str:
        """Route query through the multi-agent orchestrator (Phase 2 entry point)."""
        if not self.orchestrator:
            return "Error: Not connected to any MCP servers."
        provider = ArgoProvider(self.anl_username, self.selected_model)
        return await self.orchestrator.process(query, provider, use_history)


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
        n_tools = len(self._available_tools)
        servers = ', '.join(self.sessions.keys())
        print(f"\n  APEXA - Advanced Photon EXperiment Assistant")
        print(f"  Model: {self.selected_model} ({self.environment})  |  {n_tools} tools  |  Servers: {servers}")
        print(f"  Type 'help' for commands, 'quit' to exit\n")
        
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

                # Get input and clean any embedded newlines from paste
                user_input = input("APEXA> ").strip().replace('\n', '').replace('\r', '')
                
                if not user_input:
                    continue
                    
                # Add to history
                if user_input and (not history or history[-1] != user_input):
                    history.append(user_input)
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'clear':
                    if self.orchestrator:
                        self.orchestrator.clear_history()
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
                    tools = self._available_tools
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
                            self.environment = "DEV"
                            print(f"✅ Switched to: {model_name} (using DEV environment)")
                        else:
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
                                "run_ff_hedm_full_workflow",
                                {"example_dir": example_dir, "n_cpus": n_cpus}
                            )
                            print(f"\n{result}\n")
                        else:
                            print("MIDAS server not connected")

                    elif 'integrator' in command.lower() or 'batch' in command.lower():
                        # Integrator batch command
                        print("Use the interactive query instead:")
                        print('APEXA> Run batch integration on /path/to/data with calib_file.txt')

                    else:
                        # Regular shell command via core server
                        if "core" in self.sessions:
                            result = await self.execute_tool_call("run_command", {"command": command})
                            print(f"\n{result}\n")
                        else:
                            print("Core server not connected")
                elif user_input.startswith('ls '):
                    path = user_input[3:].strip()
                    if "core" in self.sessions:
                        result = await self.execute_tool_call("list_directory", {"path": path})
                        print(f"\n{result}\n")
                    else:
                        print("Core server not connected")
                elif user_input:
                    response = await self.run_query(user_input)
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