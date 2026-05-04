#!/usr/bin/env python3
"""
APEXA Multi-Agent Orchestration Layer  (Phase 2)

Replaces the monolithic APEXAClient agentic loop with:
  - ArgoProvider  : single class for all Argo Gateway HTTP calls
  - APEXAAgent    : lightweight agent definition (instructions + tool filter)
  - AgentRunner   : clean 10-iteration loop, native tool calling only
  - OrchestratorAgent : keyword-based routing to specialist agents

What this eliminates from argo_mcp_client.py:
  - call_argo_chat_api()          (~82 lines)
  - _prepare_argo_payload()       (~68 lines)
  - _convert_tools_to_claude_format() (13 lines)
  - get_all_available_tools()     (40 lines)
  - process_diffraction_query()   (~570 lines)
  - _extract_peak_positions()     (18 lines)
  - CALCULATION_KEYWORDS / MAP dicts (28 lines)
  - _needs_calculation_tool()     (5 lines)
  - _detect_required_calculation_tool() (13 lines)

What stays in argo_mcp_client.py (unchanged):
  - All MCP server connection logic
  - execute_tool_call() — now uses _tool_registry for routing
  - ExperimentContext, ImageAnalyzer, PlottingEngine, etc.
  - interactive_analysis_session() CLI loop
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
import httpx
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Awaitable
from interaction_logger import InteractionLogger, InteractionEntry

# ── Compact directory listing ───────────────────────────────────────────────

_EXT_GROUPS = {
    "Data":    {".tif",".tiff",".ge",".ge1",".ge2",".ge3",".ge4",".ge5",
                ".h5",".hdf",".hdf5",".zarr",".nxs",".cbf",".zip"},
    "Config":  {".txt",".toml",".yaml",".yml",".json",".cfg",".ini",".env"},
    "Results": {".csv",".dat",".xy",".bin",".mic",".out"},
    "Scripts": {".py",".sh",".bash"},
    "Docs":    {".md",".rst",".pdf",".log"},
}

def _compact_listing(parsed: dict, max_preview: int = 3) -> str:
    """Build a grouped compact summary from list_directory JSON result.

    Shows full listing if ≤20 files, otherwise groups by file type
    with preview filenames and a hint to use 'ls' for the full listing.
    """
    DIM = "\033[2m"
    BOLD = "\033[1m"
    BLUE = "\033[1;34m"
    RESET = "\033[0m"

    path = parsed.get("path", "")
    dirs = parsed.get("dirs", [])
    files = parsed.get("files", [])

    if not files and not dirs:
        return parsed.get("listing", "")

    if len(files) <= 20:
        return parsed.get("listing", "")

    lines = [f"{BOLD}{path}{RESET}"]

    if dirs:
        dir_strs = [f"{BLUE}{BOLD}{d}{RESET}" for d in dirs]
        lines.append("  " + "  ".join(dir_strs))
        lines.append("")

    groups: dict = {cat: [] for cat in _EXT_GROUPS}
    groups["Other"] = []

    for fname in files:
        ext = Path(fname).suffix.lower()
        placed = False
        for cat, exts in _EXT_GROUPS.items():
            if ext in exts:
                groups[cat].append(fname)
                placed = True
                break
        if not placed:
            groups["Other"].append(fname)

    for cat, flist in groups.items():
        if not flist:
            continue
        preview = flist[:max_preview]
        preview_str = "  ".join(preview)
        if len(flist) > max_preview:
            preview_str += f"  {DIM}+{len(flist) - max_preview} more{RESET}"
        lines.append(f"  {cat} ({len(flist)}):  {preview_str}")

    hint_path = Path(path).name or path
    lines.append(f"  {DIM}{len(dirs)} directories, {len(files)} files — type 'ls {hint_path}' for full listing{RESET}")

    return "\n".join(lines)


# ── Constants ───────────────────────────────────────────────────────────────

PROD_URL = "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/"
DEV_URL  = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"

# Models that require the DEV endpoint (add new beta models here)
DEV_ONLY_MODELS: set = set()

# ── Data Types ──────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class AgentResponse:
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"


# ── Argo Provider ────────────────────────────────────────────────────────────

class ArgoProvider:
    """
    Single class handling all Argo Gateway communication.

    Normalises per-model format differences (Claude / OpenAI / Gemini) in
    one place instead of scattered if/else branches across the codebase.

    Created fresh per query (cheap — just config) so model switching is free.
    """

    def __init__(self, username: str, model: str):
        self.username = username
        self.model    = model
        self.url      = DEV_URL if model in DEV_ONLY_MODELS else PROD_URL
        self._client  = httpx.AsyncClient(timeout=120.0)


    # ── Payload builder ─────────────────────────────────────────────────────

    def _build_payload(self, messages: List[Dict],
                       temperature: float) -> Dict:
        payload: Dict[str, Any] = {
            "user":        self.username,
            "model":       self.model,
            "messages":    messages,
            "temperature": temperature,
        }

        # Max tokens and params per model family
        if self.model.startswith("claude"):
            payload["max_tokens"] = 21000
            # Haiku 4.5 and Sonnet 4.5/4.6 reject both temperature+top_p
            if self.model not in ("claudesonnet45", "claudesonnet46", "claudehaiku45"):
                payload["top_p"] = 0.9
        elif self.model.startswith("gpto"):
            # o-series: use max_completion_tokens, no temperature/top_p
            payload.pop("temperature", None)
            payload["max_completion_tokens"] = 16000
        elif self.model.startswith("gpt"):
            # All GPT models (gpt4o, gpt41, gpt5, etc.): max_completion_tokens
            payload["max_completion_tokens"] = 16000
            payload["top_p"] = 0.9
        elif self.model.startswith("gemini"):
            payload["max_tokens"] = 16000
        else:
            payload["max_completion_tokens"] = 16000

        # NOTE: Do NOT pass tools in the API payload.
        # Argo Gateway returns string responses (not dict) which strips
        # native tool_calls. Tools are listed in the system prompt instead,
        # and the model uses TOOL_CALL: / ARGUMENTS: text format.

        return payload

    # ── Response parsing ────────────────────────────────────────────────────

    def _parse_tool_calls(self, raw: List[Dict]) -> List[ToolCall]:
        calls: List[ToolCall] = []
        for i, tc in enumerate(raw):
            if "function" in tc:               # OpenAI format
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
            elif "input" in tc:                # Claude format
                name = tc.get("name", "")
                args = tc["input"]
            elif "args" in tc:                 # Gemini format
                name = tc.get("name", "")
                args = tc["args"]
            else:
                continue
            calls.append(ToolCall(
                id=tc.get("id", f"tool_{i}"),
                name=name,
                arguments=args,
            ))
        return calls

    def _parse_response(self, data: Dict) -> AgentResponse:
        # Argo wraps response in {"response": {"content": ..., "tool_calls": [...]}}
        if "response" in data and isinstance(data["response"], dict):
            resp      = data["response"]
            content   = resp.get("content", "") or ""
            raw_calls = resp.get("tool_calls", []) or []
        elif "choices" in data:
            msg       = data["choices"][0]["message"]
            content   = msg.get("content", "") or ""
            raw_calls = msg.get("tool_calls", []) or []
        else:
            content   = str(data.get("response", ""))
            raw_calls = []

        tool_calls = self._parse_tool_calls(raw_calls)
        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
        )

    # ── Public API ──────────────────────────────────────────────────────────

    async def chat(self, messages: List[Dict],
                   temperature: float = 0.7) -> AgentResponse:
        payload  = self._build_payload(messages, temperature)
        if os.environ.get("APEXA_DEBUG"):
            debug_payload = {k: v for k, v in payload.items() if k != "messages"}
            print(f"  [debug] Argo payload: {debug_payload}", file=sys.stderr)

        retries = 3
        for attempt in range(retries):
            try:
                t0 = time.monotonic()
                response = await self._client.post(
                    self.url, json=payload,
                    headers={"Content-Type": "application/json"},
                )
                elapsed = time.monotonic() - t0
                if os.environ.get("APEXA_SHOW_TIMING"):
                    print(f"  \033[2m⏱ {self.model} responded in {elapsed:.1f}s\033[0m", flush=True)
                if response.status_code in (502, 503, 429):
                    wait = 2 ** attempt
                    print(f"  \033[33m⚠ Argo {response.status_code}, retrying in {wait}s ({attempt+1}/{retries})\033[0m")
                    await asyncio.sleep(wait)
                    continue
                if response.status_code != 200:
                    print(f"  Argo API error ({response.status_code}): {response.text[:500]}", file=sys.stderr)
                    response.raise_for_status()
                return self._parse_response(response.json())
            except httpx.TimeoutException:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    print(f"  \033[33m⚠ Argo timeout, retrying in {wait}s ({attempt+1}/{retries})\033[0m")
                    await asyncio.sleep(wait)
                else:
                    raise
        response.raise_for_status()
        return self._parse_response(response.json())

    async def close(self):
        await self._client.aclose()


# ── Agent Definition ─────────────────────────────────────────────────────────

@dataclass
class APEXAAgent:
    name:         str
    instructions: str
    tool_names:   List[str]   # bare MCP tool names; empty list = all tools
    temperature:  float = 0.7


# ── Specialist Agents ────────────────────────────────────────────────────────

CALIBRATION_AGENT = APEXAAgent(
    name        = "CalibrationAgent",
    temperature = 0.3,   # low — calibration needs deterministic output
    tool_names  = [
        "midas_auto_calibrate",          # primary: AutoCalibrateZarr.py workflow
        "run_ff_calibration",            # FF-HEDM detector geometry calibration
        "xray_calculate",
        "validate_beamline_parameters",
        "list_common_calibrants",
        "list_directory",
        "read_file",
        "get_file_info",
        # Parameter validation
        "validate_parameter_file",
        "diagnose_parameter_file",
        "inspect_dataset_file",
        "enumerate_bragg_rings",
    ],
    instructions = """You are a detector calibration specialist for HEDM synchrotron experiments at APS.

Workflow — follow these steps IN ORDER, no confirmations needed:
1. Call list_directory ONCE to find the calibrant image file
2. IMMEDIATELY call midas_auto_calibrate with the full path — do NOT ask the user to confirm
3. parameters_file is OPTIONAL — omit it if not available. AutoCalibrateZarr.py auto-detects
   calibrant (CeO2/LaB6 from filename), energy (keV in filename), pixel size (from detector shape)
4. If the user provides energy in keV, call xray_calculate to convert to wavelength first

CRITICAL: After listing files, call midas_auto_calibrate IMMEDIATELY.
Never say "I found the file, shall I proceed?" — just run it.
Never call list_directory more than once per request.

After calibration report: refined BC, Lsd, tilts, and convergence quality.

PARAMETER VALIDATION:
- validate param file: validate_parameter_file (checks required keys, ranges, cross-field rules)
- diagnose param file: diagnose_parameter_file (LLM-ready diagnosis with fix suggestions)
- extract params from data: inspect_dataset_file (auto-detect from GE/HDF5/Zarr)
- Bragg ring listing: enumerate_bragg_rings (which rings hit the detector)

When the user asks to validate, diagnose, or check a parameter file:
1. Call validate_parameter_file or diagnose_parameter_file DIRECTLY with the directory path
   (e.g. param_file="test1", pipeline="ff"). The tool auto-finds Parameters.txt or refined_MIDAS_params*.txt.
   Do NOT call list_directory first — the tool handles file discovery.
2. Pipeline arg is required — infer from the user's query:
   - "calibrat" / "integrat" / "lineout" / "caked" → pipeline="ri" (radial integration)
   - "ff-hedm" / "far-field" / "reconstruction" → pipeline="ff"
   - "nf-hedm" / "near-field" / "microstructure" → pipeline="nf"
   - "pf-hedm" / "point-focus" → pipeline="pf"
   - Default to "ri" for calibration-related queries, "ff" for HEDM workflow queries.

NEVER mention pyFAI, .poni files, or azimuthalIntegrator. This system uses MIDAS exclusively.
Calibration output is refined_MIDAS_params*.txt — NOT .poni files.""",
)

ANALYSIS_AGENT = APEXAAgent(
    name        = "AnalysisAgent",
    temperature = 0.5,
    tool_names  = [
        # Integration
        "midas_integrate_2d_to_1d",
        "midas_batch_integrate",
        # GSAS-II refinement & live analysis
        "run_gsas_refinement",
        "run_live_analysis",
        "fetch_cif_from_mp",
        # FF/NF/PF-HEDM workflows
        "run_ff_hedm_full_workflow",
        "run_nf_hedm_reconstruction",
        "run_pf_hedm_workflow",
        # Post-processing
        "match_grains",
        "overlay_ff_nf_results",
        "calculate_misorientation",
        "run_forward_simulation",
        "extract_grain_centroids",
        "convert_nf_to_dream3d",
        # Status & utilities
        "get_midas_workflow_status",
        "create_midas_parameter_file",
        "validate_midas_installation",
        "batch_convert_ge_to_tiff",
        # Parameter validation (pre-workflow)
        "validate_parameter_file",
        "diagnose_parameter_file",
        "inspect_dataset_file",
        "enumerate_bragg_rings",
        # Stress/strain analysis (post-reconstruction)
        "compute_grain_stress",
        "get_material_stiffness",
        "correct_d0_equilibrium",
        "analyze_slip_systems",
        "read_grains_summary",
        # General tools
        "xray_calculate",
        "list_directory",
        "read_file",
        "write_file",
        "run_command",
        "get_file_info",
    ],
    instructions = """You are a HEDM data analysis specialist at APS.

When the user asks to integrate, reconstruct, refine, track grains, or run any workflow,
you MUST call the appropriate tool. Never describe steps in text — execute them.

Capabilities (use the matching tool for each):
- 2D → 1D integration: midas_integrate_2d_to_1d or midas_batch_integrate
- GSAS-II refinement: run_gsas_refinement (takes .zarr.zip + CIF files)
- Integration + refinement pipeline: run_live_analysis (batch or stream backend)
- FF-HEDM reconstruction: run_ff_hedm_full_workflow
- NF-HEDM mapping: run_nf_hedm_reconstruction
- PF-HEDM pole figures: run_pf_hedm_workflow
- Grain tracking/matching: run_ff_grain_tracking, match_grains (Hungarian algorithm)
- Misorientation: calculate_misorientation
- Dream3D export: convert_nf_to_dream3d
- X-ray calculations: xray_calculate (NEVER compute manually)
- File operations: list_directory, read_file, get_file_info
- Validate parameter file: validate_parameter_file (checks required keys, ranges, cross-field rules)
- Diagnose parameter file: diagnose_parameter_file (LLM-ready diagnosis with fix suggestions)
- Extract params from data: inspect_dataset_file (auto-detect from GE/HDF5/Zarr)
- Bragg rings: enumerate_bragg_rings (which rings hit the detector)
- Stress analysis: compute_grain_stress (Hooke's law + equilibrium from Grains.csv)
- Material lookup: get_material_stiffness (elastic constants for Au, Cu, Al, Fe, Ni, Ti, W, Si, CeO2)
- d0 correction: correct_d0_equilibrium (two-step isotropic strain + stress correction)
- Slip systems: analyze_slip_systems (Schmid factors, Taylor factor, yield proximity)
- Grain summary: read_grains_summary (statistics of a Grains.csv file)

PARAMETER VALIDATION — When the user asks to validate, diagnose, or check a parameter file:
1. Call validate_parameter_file or diagnose_parameter_file DIRECTLY with the directory path
   (e.g. param_file="test1", pipeline="ff"). The tool auto-finds Parameters.txt or refined_MIDAS_params*.txt.
   Do NOT call list_directory first — the tool handles file discovery.
2. Pipeline arg is required — infer from the user's query:
   - "calibrat" / "integrat" / "lineout" / "caked" → pipeline="ri" (radial integration)
   - "ff-hedm" / "far-field" / "reconstruction" → pipeline="ff"
   - "nf-hedm" / "near-field" / "microstructure" → pipeline="nf"
   - "pf-hedm" / "point-focus" → pipeline="pf"
   - Default to "ri" for integration/calibration queries, "ff" for HEDM workflow queries.

PRE-WORKFLOW VALIDATION — ONLY before heavyweight HEDM reconstruction (ff, nf, pf):
1. Call validate_parameter_file ONLY before run_ff_hedm_full_workflow, run_nf_hedm_reconstruction, or run_pf_hedm_workflow
2. Do NOT validate before simple operations like midas_integrate_2d_to_1d, midas_auto_calibrate, or run_gsas_refinement — just run them directly
3. If errors are found, call diagnose_parameter_file and fix issues before proceeding
4. If the user has a dataset file, call inspect_dataset_file to verify consistency

When the user says "retry", "rerun", or "redo" → call the requested tool IMMEDIATELY. Do NOT validate first.

Standard workflow:
  1. list_directory to find data files
  2. midas_integrate_2d_to_1d for 2D → 1D (produces .zarr.zip)
  4. run_gsas_refinement for peak fitting / lattice refinement on .zarr.zip
  5. Or run_live_analysis for combined integration + refinement in one step
  6. run_ff_hedm_full_workflow or run_nf_hedm_reconstruction
  7. Post-process: match_grains, run_ff_grain_tracking, overlay_ff_nf_results, extract_grain_centroids
  8. Export: convert_nf_to_dream3d

POST-RECONSTRUCTION STRESS ANALYSIS — After FF-HEDM or NF-HEDM completes:
1. Call read_grains_summary to understand the grain population
2. Call get_material_stiffness to look up the material (user must specify material)
3. Call compute_grain_stress with the Grains.csv and material name
4. If d0 correction is needed, call correct_d0_equilibrium
5. For plasticity analysis, call analyze_slip_systems with the load direction
Always report: grain count, mean/std von Mises stress, hydrostatic shift, d0 correction magnitude.

GSAS-II refinement workflow:
  1. If no CIF file → call fetch_cif_from_mp to download one (you have this tool)
  2. Read the CIF path from the fetch result
  3. IMMEDIATELY call run_gsas_refinement with:
     - data_file=<.zarr.zip path>
     - cif_files=[<CIF path>]
     - two_theta_limits=[2.0, 15.0] (ALWAYS set — without limits Rwp will be ~100%)
     - n_cpus=8 (parallelize across histograms)
  4. NEVER use run_command for GSAS-II — always use run_gsas_refinement

CRITICAL: After calling a tool, read the result carefully. Do NOT call list_directory
to verify files you already know about. Use the paths from the tool results directly.

Always report: grains found, convergence quality, Rwp, output file paths.

NEVER mention pyFAI, .poni files, or azimuthalIntegrator. This system uses MIDAS exclusively.
Only report data from actual tool results — never hallucinate file contents or parameters.""",
)

KNOWLEDGE_AGENT = APEXAAgent(
    name        = "KnowledgeAgent",
    temperature = 0.6,
    tool_names  = [
        "query_hedm_knowledge",
        "get_material_properties",
        "get_typical_hedm_parameters",
        "estimate_parameters_from_image",
        "list_common_calibrants",
        "xray_calculate",
        "fetch_cif_from_mp",
        "enumerate_bragg_rings",
        "get_material_stiffness",
    ],
    instructions = """You are an HEDM knowledge expert. You answer from indexed sources, not from memory.

MANDATORY: For ANY conceptual, methodology, or "what is / how does / explain" question,
your FIRST action MUST be a TOOL_CALL to query_hedm_knowledge. Do NOT answer first and
search later. Do NOT answer from your own knowledge if the tool returns matching excerpts.

After calling query_hedm_knowledge:
- If excerpts come back (similarity > 0.30), build your answer directly from them and
  cite each claim inline like "(Bernier 2020)" or "(Sharma 2012, p.694)". End the
  response with a "References:" section listing each source's full citation exactly as
  returned by the tool.
- If the tool returns no excerpts OR all similarities < 0.30, say so plainly:
  "No matching sources in the knowledge base — answering from general background:"
  and then give the answer. Do NOT pretend the knowledge base supports it.
- Never invent citations or paraphrase a source you didn't actually retrieve.

Other tools:
- get_material_properties for crystallographic data (lattice params, space groups, d-spacings)
- get_typical_hedm_parameters for recommended parameter ranges
- estimate_parameters_from_image to estimate beam parameters from diffraction images
- list_common_calibrants for calibrant materials
- xray_calculate for ANY calculation (NEVER compute manually)
- fetch_cif_from_mp to download CIF files from Materials Project for any material

When the user asks for a CIF file, call fetch_cif_from_mp IMMEDIATELY with the formula.
Report: formula, space group, crystal system, stability, and file path.

When in doubt, call the tool. A grounded "I don't have a source for that" beats a
fluent answer with no citation.""",
)

MOTOR_AGENT = APEXAAgent(
    name        = "MotorAgent",
    temperature = 0.2,   # very low — motor commands must be precise and deterministic
    tool_names  = [
        "get_motor_position",
        "get_motor_status",
        "move_motor_absolute",
        "move_motor_relative",
        "stop_motor",
        "set_motor_velocity",
        "jog_motor",
        "tweak_motor",
        "get_motor_limits",
        "set_motor_limits",
        "set_motor_description",
        "list_motors",
        "home_motor",
    ],
    instructions = """You are a motor control specialist for EPICS-based beamline instruments at APS.

Default IOC prefix is "20idMotSim". Motor PV names: "m1", "m2", etc.
The prefix parameter defaults automatically — you do NOT need to specify it.

MOTOR NAMES: Users can refer to motors by PV name (m1, m2, ...) OR by description
(e.g. "Sample X", "detector z"). The tools auto-resolve descriptions to PV names
via the EPICS DESC field. If the user uses a descriptive name, pass it directly
to the tool — it will resolve automatically.
If unsure which motor the user means, call list_motors first to see all descriptions.

⚠️ CRITICAL — ALWAYS call the tool, NEVER just describe what you would do:
1. User asks to MOVE → call move_motor_absolute (or move_motor_relative) IMMEDIATELY.
   The tool checks limits internally — do NOT call get_motor_status first.
2. User asks POSITION → call get_motor_position.
3. User asks STATUS → call get_motor_status.
4. User says STOP → call stop_motor IMMEDIATELY.
5. For small steps → use tweak_motor or move_motor_relative.
6. If a move is rejected for limits → call get_motor_limits to show the range.
7. NEVER say "I can move it" — CALL THE TOOL.
8. NEVER call get_motor_status before a move — it wastes a round-trip.
9. For multiple motors → call move_motor_absolute for EACH one. Do ALL of them.

After each move report: target, final RBV, and units.""",
)

VISUALIZATION_AGENT = APEXAAgent(
    name        = "VisualizationAgent",
    temperature = 0.3,
    tool_names  = [
        "run_midas_viewer",
        "list_directory",
        "get_file_info",
    ],
    instructions = """You are a visualization specialist for HEDM diffraction data at APS.

USE run_midas_viewer for ALL plotting. It handles MIDAS paths and Python automatically.
Do NOT use run_command or check_environment — run_midas_viewer does everything.
Do NOT read data files — the viewer GUI displays the data. Your job is to LAUNCH the viewer, not analyze data.

⚠️ CRITICAL: Call run_midas_viewer EXACTLY ONCE per request. Pick the single BEST viewer. NEVER launch multiple viewers.

STEP 1: Find the data file.
  - If the user says "integration results" or "caked output" → list the integration/ subdirectory first
  - Integration outputs (*_lineout.xy, *_caked.hdf.zarr.zip, *_lineout.bin) are in <dir>/integration/
  - Calibration outputs (*_corr.csv) are in the parent directory
  - Always prefer .zarr.zip over plain .hdf — the zarr archive is the complete output
  - Call list_directory on the CORRECT subdirectory, not just the parent
STEP 2: Match the file to the correct viewer — pick ONE:

| File pattern | viewer name | When to use |
|---|---|---|
| *_corr.csv | plot_calibrant_results | Calibration fit, calibration QC, lattice-vs-η |
| *_lineout.xy | plot_lineout_results | 1D diffraction profile, lineout, peaks |
| *_lineout.xy (compare) | plot_lineout_comparison | Compare calibrant vs integrator lineouts |
| *_lineout.bin (live) | live_viewer | Real-time GPU streaming monitor |
| *_caked.hdf.zarr.zip | plot_caked_peaks | Caked data, integrated image, 2D heatmap (PREFERRED for caked data) |
| *_caked_peaks.h5 | plot_caked_peaks | Peak fitting results |
| Raw .tif/.ge/.h5 | ff_asym_qt | Raw detector image, diffraction image, ring overlays |
| Grains.csv + .zarr | interactiveFFplotting | FF-HEDM grain map, grain results |
| .mic/.map (NF) | nf_qt | NF-HEDM microstructure |

DISAMBIGUATION — when the user request is ambiguous, pick ONE using these rules:
- "calibrated image" / "calibration results" / "calibration fit" → plot_calibrant_results
- "caked image" / "caked data" / "integrated data" / "integration result" → plot_caked_peaks
- "lineout" / "1D profile" / "diffraction pattern" → plot_lineout_results
- "raw image" / "diffraction image" / "detector image" → ff_asym_qt
- "grain map" / "grain results" / "FF results" → interactiveFFplotting
- "microstructure" / "NF results" → nf_qt
- For caked .zarr.zip files: ALWAYS use plot_caked_peaks (interactive Qt viewer with heatmap + profile + peak table). Do NOT use plot_integrator_peaks (that is a diagnostic scatter plot, not an interactive viewer).
- If still ambiguous, prefer the MOST PROCESSED result: caked > lineout > raw.

STEP 3: Call run_midas_viewer ONCE with the viewer name and data file path. That's it.

Example:
  User: "plot the calibration results in test1"
  → list_directory to find *_corr.csv
  → run_midas_viewer(viewer="plot_calibrant_results", data_file="/path/to/file_corr.csv")

Notes:
- Pass param_file if refined_MIDAS_params*.txt is available (enables 2θ/Q axes)
- For live_viewer: pass extra_args="--nRBins 2000" (capital R, capital B)
- viz_caking.py: DO NOT USE — use plot_caked_peaks instead
- plot_integrator_peaks: diagnostic scatter only — prefer plot_caked_peaks for interactive viewing

After launching, report ONE line: which viewer was launched and which file. Do NOT read or summarize the data — the GUI shows it.""",
)


# ── Tool-use system preamble ─────────────────────────────────────────────────

_TOOL_PREAMBLE = """⚠️ CRITICAL: YOU HAVE TOOLS — USE THEM.

You are APEXA, connected to live MCP servers with real analysis tools.
When the user gives a COMMAND (calibrate, integrate, list files, calculate,
run workflow), you MUST call the appropriate tool IMMEDIATELY.

⚠️ TOOL CALLING FORMAT — YOU MUST USE THIS EXACT FORMAT:

TOOL_CALL: tool_name
ARGUMENTS: {"param1": "value1", "param2": "value2"}

Examples of CORRECT behavior:

User: "Calculate d-spacing for (110) plane in bcc iron"
✅ CORRECT:
TOOL_CALL: xray_calculate
ARGUMENTS: {"calculation_type": "d_from_hkl", "h": 1, "k": 1, "l": 0, "material": "Fe"}

User: "List files in the current directory"
✅ CORRECT (use CWD as absolute path):
TOOL_CALL: list_directory
ARGUMENTS: {"path": "<CWD>"}

User: "Calibrate the CeO2 image in test5"
✅ CORRECT (prepend CWD to relative path):
TOOL_CALL: list_directory
ARGUMENTS: {"path": "<CWD>/test5"}

User: "Validate the parameter file in /home/user/data/scan1"
✅ CORRECT (absolute path — use EXACTLY as given, do NOT prepend CWD):
TOOL_CALL: list_directory
ARGUMENTS: {"path": "/home/user/data/scan1"}

User: "Convert 61.332 keV to wavelength"
✅ CORRECT:
TOOL_CALL: xray_calculate
ARGUMENTS: {"calculation_type": "energy_to_wavelength", "energy_kev": 61.332}

User: "Show me the lineout for CeO2 integration in test1"
✅ CORRECT:
TOOL_CALL: list_directory
ARGUMENTS: {"path": "<CWD>/test1/integration"}
[find *_lineout.xy, then:]
TOOL_CALL: run_midas_viewer
ARGUMENTS: {"viewer": "plot_lineout_results", "data_file": "/full/path/to/lineout.xy", "param_file": "/full/path/to/refined_MIDAS_params_CeO2.txt"}

User: "Plot the caked output"
✅ CORRECT (after finding *_caked.hdf.zarr.zip):
TOOL_CALL: run_midas_viewer
ARGUMENTS: {"viewer": "plot_caked_peaks", "data_file": "/full/path/to/file.caked.hdf.zarr.zip"}

User: "Plot calibration results in test1"
✅ CORRECT (after finding *_corr.csv):
TOOL_CALL: run_midas_viewer
ARGUMENTS: {"viewer": "plot_calibrant_results", "data_file": "/full/path/to/file_corr.csv"}

User: "Refine the caked output with GSAS-II using the CeO2 CIF"
✅ CORRECT (ALWAYS include two_theta_limits and n_cpus):
TOOL_CALL: run_gsas_refinement
ARGUMENTS: {"data_file": "/path/to/CeO2_caked.hdf.zarr.zip", "cif_files": ["/path/to/CeO2.cif"], "two_theta_limits": [2.0, 15.0], "n_cpus": 8}

User: "Fetch a CIF file for CeO2"
✅ CORRECT:
TOOL_CALL: fetch_cif_from_mp
ARGUMENTS: {"formula": "CeO2"}

User: "Run integration and refinement on the scan data"
✅ CORRECT:
TOOL_CALL: run_live_analysis
ARGUMENTS: {"backend": "batch", "param_file": "/path/to/params.txt", "data_file": "/path/to/data.h5", "cif_files": ["/path/to/phase.cif"]}

User: "Move motor m1 to 25.5"
✅ CORRECT:
TOOL_CALL: move_motor_absolute
ARGUMENTS: {"motor": "m1", "position": 25.5}

User: "Where is motor m1?"
✅ CORRECT:
TOOL_CALL: get_motor_position
ARGUMENTS: {"motor": "m1"}

❌ WRONG — NEVER do these:
- NEVER calculate d = a/√(h²+k²+l²) yourself — call xray_calculate
- NEVER say "you can use ls" or "here's how to do it in Python"
- NEVER say "Let me proceed" or "I can move it" without actually calling a tool
- NEVER describe what you WOULD do — DO IT with TOOL_CALL
- NEVER read_file to show plot data — launch the viewer with run_midas_viewer
- NEVER use run_command for MIDAS viewers — always use run_midas_viewer tool
- NEVER use run_command for GSAS-II, refinement, or peak fitting — always use run_gsas_refinement
- NEVER use run_command for integration + refinement pipelines — always use run_live_analysis
- NEVER try to construct Python paths manually — run_midas_viewer handles paths
- NEVER prepend CWD to an absolute path the user gave you
- NEVER switch back to CWD mid-chain — if the user said "/some/path", ALL subsequent
  tool calls for that request must use "/some/path" (not CWD)
- NEVER guess filenames — call list_directory on the target directory first, then use
  the EXACT filenames from the listing
- NEVER mention pyFAI, .poni files, or azimuthalIntegrator — this system uses MIDAS only
- NEVER hallucinate tools, files, or parameters that don't exist in tool results
- NEVER claim to have read data from a file you didn't actually call read_file on
- Only report information that came from actual tool results, not from your training data

⛔ ANTI-HALLUCINATION — READ THIS CAREFULLY:
NEVER generate fake tool results. NEVER fabricate parameter values (Lsd, Wavelength,
LatticeConstant, BC, PixelSize, etc.). NEVER produce validation reports, file contents,
or analysis results from your training data. If the user asks you to validate, diagnose,
read, or analyze a file — you MUST call the actual tool first. If you don't know a value,
call the tool to look it up. Your training data is NOT a substitute for real tool results.
Presenting fabricated data as real results is DANGEROUS at a beamline — wrong values can
damage equipment or ruin experiments.

❌ SPECIFIC WRONG EXAMPLES:
User: "run GSAS-II refinement on the integrated data"
WRONG: TOOL_CALL: run_command  ARGUMENTS: {"command": "GSAS-II ..."}
RIGHT: TOOL_CALL: run_gsas_refinement  ARGUMENTS: {"data_file": "/path/to.zarr.zip", "cif_files": ["/path/to.cif"], "two_theta_limits": [2.0, 15.0], "n_cpus": 8}

User: "refine the caked data"
WRONG: TOOL_CALL: run_command  ARGUMENTS: {"command": "python gsas_ii_refine.py ..."}
WRONG: TOOL_CALL: run_gsas_refinement  ARGUMENTS: {"data_file": "...", "cif_files": ["..."]}  ← missing two_theta_limits!
RIGHT: TOOL_CALL: run_gsas_refinement  ARGUMENTS: {"data_file": "/path/to.zarr.zip", "cif_files": ["/path/to.cif"], "two_theta_limits": [2.0, 15.0], "n_cpus": 8}

RULES:
1. For ANY X-ray calculation → TOOL_CALL: xray_calculate
2. For file listing → TOOL_CALL: list_directory
3. For reading files → TOOL_CALL: read_file
4. For calibration → TOOL_CALL: midas_auto_calibrate
5. For integration → TOOL_CALL: midas_integrate_2d_to_1d
6. For GSAS-II refinement → TOOL_CALL: run_gsas_refinement (needs .zarr.zip + CIF)
7. For combined integration + refinement → TOOL_CALL: run_live_analysis
8. For fetching CIF files → TOOL_CALL: fetch_cif_from_mp
9. For HEDM workflows → TOOL_CALL: the appropriate workflow tool
10. For visualization/plotting → TOOL_CALL: run_midas_viewer (pass viewer name + data file)
11. For motor control → TOOL_CALL: the appropriate motor tool (move, get_position, stop, etc.)

Only generate text WITHOUT a TOOL_CALL when:
- User says hello/greeting
- User asks a conceptual question ("what is HEDM?", "explain calibration")
- User asks what you can do

PATH HANDLING — CRITICAL:
- ALWAYS use ABSOLUTE paths in tool arguments
- When the user says "test5", they mean "<CWD>/test5" (where <CWD> is shown below)
- When the user says "current directory", they mean the CWD shown below
- Convert ALL relative paths to absolute by prepending the CWD
- If the user provides an absolute path (starts with /), use it as-is
- NEVER pass just a filename — always include the full directory path

"""

# ── Agent Runner ─────────────────────────────────────────────────────────────

ExecuteToolFn = Callable[[str, Dict], Awaitable[str]]
OnToolResultFn = Optional[Callable[[str, Dict, str], Awaitable[None]]]


class AgentRunner:
    """
    Agentic loop with dual-mode tool calling:
      1. Native API tool_calls (when Argo returns them)
      2. Text-based TOOL_CALL: / ARGUMENTS: parsing (fallback for string responses)

    Delegates tool *execution* back to APEXAClient.execute_tool_call() so that
    ErrorPreventor, SmartCache, ProactiveSuggestions, and ExperimentContext
    all continue to work without any changes.
    """

    # Regex for text-based tool calls
    _TOOL_CALL_RE = re.compile(
        r'TOOL_CALL:\s*(\S+)\s*\n\s*ARGUMENTS:\s*(\{.*?\})',
        re.DOTALL
    )

    def __init__(self, execute_tool_fn: ExecuteToolFn):
        self._execute = execute_tool_fn

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _filter_tools(self, tool_names: List[str],
                      all_tools: List[Dict]) -> List[Dict]:
        if not tool_names:
            return all_tools
        names = set(tool_names)
        return [t for t in all_tools if t["function"]["name"] in names]

    def _parse_text_tool_calls(self, text: str) -> List[ToolCall]:
        """Extract TOOL_CALL: / ARGUMENTS: pairs from model text output."""
        calls = []
        for i, match in enumerate(self._TOOL_CALL_RE.finditer(text)):
            name = match.group(1).strip()
            try:
                args = json.loads(match.group(2))
            except json.JSONDecodeError:
                continue
            calls.append(ToolCall(id=f"text_tc_{i}", name=name, arguments=args))
        return calls

    def _strip_tool_calls_from_text(self, text: str) -> str:
        """Remove TOOL_CALL/ARGUMENTS blocks from text to get the prose part."""
        clean = self._TOOL_CALL_RE.sub('', text).strip()
        # Also remove common preamble patterns the model adds before tool calls
        clean = re.sub(r'(?:I\'ll|Let me|Let\'s)\s+.*?(?:\.|:)\s*$', '', clean, flags=re.MULTILINE).strip()
        return clean

    def _assistant_message(self, resp: AgentResponse, model: str) -> Dict:
        """Format assistant message (with tool calls) for conversation history."""
        if model.startswith("claude"):
            blocks: List[Dict] = []
            if resp.content:
                blocks.append({"type": "text", "text": resp.content})
            for tc in resp.tool_calls:
                blocks.append({
                    "type":  "tool_use",
                    "id":    tc.id,
                    "name":  tc.name,
                    "input": tc.arguments,
                })
            return {"role": "assistant", "content": blocks}
        else:
            return {
                "role":    "assistant",
                "content": resp.content,
                "tool_calls": [
                    {
                        "id":   tc.id,
                        "type": "function",
                        "function": {
                            "name":      tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in resp.tool_calls
                ],
            }

    def _tool_result_message(self, tc: ToolCall, result: str,
                             model: str) -> Dict:
        """Format tool result for next API call (model-specific)."""
        if model.startswith("claude"):
            return {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tc.id, "content": result}
                ],
            }
        else:
            # For text-based tool calls, feed result back as a user message
            # so the model can process it in the next turn
            return {
                "role":    "user",
                "content": f"[Tool Result for {tc.name}]\n{result}",
            }

    @staticmethod
    def _looks_like_hallucinated_result(text: str) -> bool:
        """Detect model responses that look like fabricated tool output.

        Returns True when the text contains patterns characteristic of fake
        validation reports or parameter value listings that should only come
        from actual tool calls.
        """
        t = text.lower()
        hallucination_markers = [
            "validation result",
            "parameter file validation",
            "validation report",
            "diagnostic report",
            "parameter analysis",
        ]
        param_value_patterns = 0
        for kw in ["lsd:", "wavelength:", "latticeconstant:", "pixelsize:",
                    "bc_x:", "bc_y:", "spacegroupnum:", "ringthresh:",
                    "omegastart:", "omegastep:", "wedge:"]:
            if kw in t:
                param_value_patterns += 1

        has_marker = any(m in t for m in hallucination_markers)
        has_many_params = param_value_patterns >= 3

        return has_marker and has_many_params

    # ── Main loop ────────────────────────────────────────────────────────────

    async def run(self, agent: APEXAAgent, query: str,
                  provider: ArgoProvider, all_tools: List[Dict],
                  history: Optional[List[Dict]] = None,
                  max_iterations: int = 10,
                  log_entry: Optional[InteractionEntry] = None,
                  on_tool_result: OnToolResultFn = None) -> str:

        tools = self._filter_tools(agent.tool_names, all_tools)

        # Build system message: strong preamble + agent-specific instructions
        cwd = str(Path.cwd())
        system_content = _TOOL_PREAMBLE + f"\nCurrent working directory (CWD): {cwd}\n" + agent.instructions

        # Append tool catalog with parameters so the model knows what to call
        if tools:
            tool_entries = []
            for t in tools:
                fn = t["function"]
                name = fn["name"]
                desc = fn["description"][:120]
                params = fn.get("parameters", {})
                req = params.get("required", [])
                props = params.get("properties", {})
                if props:
                    param_strs = []
                    for pname, pinfo in list(props.items())[:6]:
                        ptype = pinfo.get("type", "string")
                        marker = " (required)" if pname in req else ""
                        param_strs.append(f"      {pname}: {ptype}{marker}")
                    param_block = "\n".join(param_strs)
                    tool_entries.append(f"  - {name}: {desc}\n    Parameters:\n{param_block}")
                else:
                    tool_entries.append(f"  - {name}: {desc}")
            system_content += f"\n\nYour available tools:\n" + "\n".join(tool_entries)

        messages = [{"role": "system", "content": system_content}]
        if history:
            # Motor/viz agents need less history (repetitive commands confuse the model)
            hist_limit = 4 if agent.name in ("MotorAgent", "VisualizationAgent") else 10
            messages.extend(history[-hist_limit:])
        messages.append({"role": "user", "content": query})

        last_tool_name = None          # track repeated tool calls
        _last_tool_args = None         # track repeated tool arguments
        repeat_count   = 0

        for _ in range(max_iterations):
            response = await provider.chat(messages, agent.temperature)

            # ── Mode 1: Native API tool_calls ──
            if response.tool_calls:
                messages.append(self._assistant_message(response, provider.model))
                for tc in response.tool_calls:
                    print(f"  \033[36m▸\033[0m \033[1m{tc.name}\033[0m")
                    t0 = time.monotonic()
                    result = await self._execute(tc.name, tc.arguments)
                    dur = int((time.monotonic() - t0) * 1000)
                    ok = "error" not in result.lower()[:100]
                    if log_entry:
                        log_entry.add_tool_call(tc.name, tc.arguments, result, ok, dur)
                    if on_tool_result:
                        try:
                            await on_tool_result(tc.name, tc.arguments, result)
                        except Exception:
                            pass
                    if tc.name == "list_directory":
                        try:
                            r = json.loads(result)
                            print(f"\n{_compact_listing(r)}\n")
                        except (json.JSONDecodeError, KeyError):
                            pass
                    messages.append(
                        self._tool_result_message(tc, result, provider.model)
                    )
                continue

            # ── Mode 2: Text-based TOOL_CALL: parsing ──
            text = response.content or ""
            text_calls = self._parse_text_tool_calls(text)

            if text_calls:
                # Add assistant text (with tool calls stripped) to history
                prose = self._strip_tool_calls_from_text(text)
                if prose:
                    messages.append({"role": "assistant", "content": prose})

                _once_per_response = set()

                for tc in text_calls:
                    # Detect repeated identical tool calls (loop bug)
                    # Compare both name AND arguments — same tool with different args
                    # is legitimate (e.g. moving 3 different motors)
                    tc_args_str = json.dumps(tc.arguments, sort_keys=True)
                    if tc.name == last_tool_name and tc_args_str == _last_tool_args:
                        repeat_count += 1
                    else:
                        last_tool_name = tc.name
                        _last_tool_args = tc_args_str
                        repeat_count = 0

                    if repeat_count >= 2:
                        # Model is looping with identical calls — force it to summarise
                        messages.append({
                            "role": "user",
                            "content": (
                                f"You already called {tc.name} with the same arguments and got the result above. "
                                "Do NOT call it again. Summarise the result for the user now."
                            ),
                        })
                        break

                    # Guard: tools that should only launch once per response
                    _ONCE_TOOLS = {"run_midas_viewer"}
                    if tc.name in _ONCE_TOOLS:
                        if tc.name in _once_per_response:
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"[Skipped duplicate {tc.name} — viewer already launched.]\n"
                                    "The GUI window is open. Do NOT launch another viewer. "
                                    "Report which viewer was launched and which file."
                                ),
                            })
                            continue
                        _once_per_response.add(tc.name)

                    # Guard: intercept run_command misuse for dedicated tools
                    if tc.name == "run_command":
                        cmd_str = str(tc.arguments.get("command", "")).lower()
                        if any(kw in cmd_str for kw in ["gsas", "refine", "rietveld", "gsas_ii_refine"]):
                            result = json.dumps({
                                "error": "Do NOT use run_command for GSAS-II refinement. "
                                         "Use TOOL_CALL: run_gsas_refinement with data_file (.zarr.zip) and cif_files.",
                                "correct_tool": "run_gsas_refinement",
                            })
                            messages.append({
                                "role": "user",
                                "content": f"[Tool Result for {tc.name}]\n{result}\n\n"
                                           "You used the WRONG tool. Use run_gsas_refinement instead of run_command. "
                                           "First call list_directory to find the .zarr.zip and .cif files, "
                                           "then call run_gsas_refinement with those paths.",
                            })
                            continue

                    print(f"  \033[36m▸\033[0m \033[1m{tc.name}\033[0m")
                    t0 = time.monotonic()
                    result = await self._execute(tc.name, tc.arguments)
                    dur = int((time.monotonic() - t0) * 1000)
                    ok = "error" not in result.lower()[:100]
                    if log_entry:
                        log_entry.add_tool_call(tc.name, tc.arguments, result, ok, dur)
                    if on_tool_result:
                        try:
                            await on_tool_result(tc.name, tc.arguments, result)
                        except Exception:
                            pass
                    if tc.name == "list_directory":
                        try:
                            r = json.loads(result)
                            print(f"\n{_compact_listing(r)}\n")
                        except (json.JSONDecodeError, KeyError):
                            pass

                    if len(result) > 8000:
                        result = result[:8000] + "\n... [truncated]"
                    # Build a context-aware follow-up prompt
                    if tc.name == "list_directory":
                        followup = (
                            "The directory listing is already displayed to the user. "
                            "Do NOT list files again. Do NOT summarize the listing. "
                            "IMMEDIATELY proceed with the user's original request — "
                            "call the appropriate tool using the file paths from the listing above. "
                            "Do NOT ask the user to confirm. Do NOT describe what you found. Just call the tool."
                        )
                    elif tc.name == "fetch_cif_from_mp":
                        followup = (
                            "CIF file downloaded. The file path is in the result above. "
                            "Now call run_gsas_refinement with the CIF path and the .zarr.zip data file. "
                            "Do NOT call list_directory or fetch_cif_from_mp again."
                        )
                    elif tc.name == "run_midas_viewer":
                        followup = (
                            "Viewer launched. Report ONE line: which viewer + which file. "
                            "Do NOT read the data file. Do NOT analyze or summarize data. "
                            "If the user asked for MULTIPLE plots (e.g. 'both', 'and', 'one by one'), "
                            "proceed to launch the NEXT viewer for the remaining request. "
                            "You may need to call list_directory on a subdirectory (e.g. integration/) to find the next file. "
                            "If the user asked for only ONE plot, do NOT call any more tools."
                        )
                    else:
                        followup = (
                            "Proceed with the user's request using the result above. "
                            "If the task is complete, summarize the results using markdown formatting: "
                            "bold **key values**, use bullet points for multiple items, "
                            "and keep it concise. Do NOT repeat the same tool call."
                        )
                    messages.append({
                        "role": "user",
                        "content": f"[Tool Result for {tc.name}]\n{result}\n\n{followup}",
                    })
                continue

            # ── No tool calls at all — check for hallucination, then return ──
            if text and self._looks_like_hallucinated_result(text):
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": (
                        "⚠️ STOP — you just generated what looks like a tool result "
                        "(validation report, parameter values, or file contents) WITHOUT "
                        "actually calling a tool. This is hallucinated data and may be WRONG.\n\n"
                        "You MUST call the actual tool to get real results. For example:\n"
                        "- To validate: TOOL_CALL: validate_parameter_file\n"
                        "- To diagnose: TOOL_CALL: diagnose_parameter_file\n"
                        "- To read a file: TOOL_CALL: read_file\n"
                        "- To list files: TOOL_CALL: list_directory\n\n"
                        "Call the appropriate tool NOW with the correct arguments."
                    ),
                })
                continue
            return text or "Analysis complete."

        # Extract last assistant text if iterations exhausted
        last = messages[-1]
        if isinstance(last.get("content"), str):
            return last["content"]
        return "Analysis reached maximum steps. Check tool outputs above."


# ── Orchestrator ─────────────────────────────────────────────────────────────

class OrchestratorAgent:
    """
    Routes user queries to the appropriate specialist agent.

    Replaces:
      - WorkflowBuilder (was a skeleton that never executed steps)
      - CALCULATION_KEYWORDS / _needs_calculation_tool() keyword detection
      - The manual per-query routing inside process_diffraction_query()

    Routing is keyword-score based (fast, deterministic).  The agent with the
    highest keyword score wins; ties and zero-scores default to AnalysisAgent,
    which is the most common operation at a beamline.
    """

    _ROUTES: Dict[str, APEXAAgent] = {
        "calibration":   CALIBRATION_AGENT,
        "analysis":      ANALYSIS_AGENT,
        "knowledge":     KNOWLEDGE_AGENT,
        "visualization": VISUALIZATION_AGENT,
        "motor":         MOTOR_AGENT,
    }

    _KEYWORDS: Dict[str, set] = {
        "calibration": {
            "calibrat", "ceo2", "lab6", "calibrant", "rings",
            "beam center", "detector distance", "lsd", "autocal",
            "stopping strain", "refined param", "bc_x", "bc_y",
            "tilt", "detector geometry",
            "validate param", "diagnose", "inspect dataset",
        },
        "analysis": {
            "integrat", "hedm", "ff-hedm", "nf-hedm", "pf-hedm", "grain",
            "phase", "workflow", "2d to 1d", "reconstruct",
            "microstructure", "orientation", "texture", "strain",
            "diffraction pattern", "peaks at", "identify",
            "calculate", "d-spacing", "d spacing", "wavelength",
            "energy", "bragg", "convert", "list file", "list dir",
            "show file", "current directory", "files here",
            "misorientation", "dream3d", "forward simulation",
            "gsas", "refine", "refinement", "rietveld", "rwp",
            "lattice param", "peak fit", "live analysis",
            "stress", "stiffness", "von mises", "schmid",
            "slip system", "d0 correct", "equilibrium",
            "plasticity", "taylor factor", "grains.csv",
            "validate param", "bragg ring",
            "calibrated file", "calibrated data", "calibrated image",
        },
        "knowledge": {
            "explain", "what is", "what's", "what are", "whats",
            "how does", "how do", "how is",
            "tell me", "describe", "definition", "define",
            "typical", "literature", "paper", "cite", "citation", "source",
            "reference", "knowledge base",
            "best practice", "recommend", "suggest", "look up",
            "material propert", "search", "parameter range",
            "cif file", "cif", "fetch cif", "download cif", "materials project",
            "crystal structure",
            # Domain-abbreviation conceptual queries (catch "what's HEDM?", "hedm overview")
            "hedm overview", "what hedm", "ff-hedm", "nf-hedm",
            "rietveld", "azimuthal integration overview",
        },
        "visualization": {
            "plot", "visualiz", "view", "show", "display", "see",
            "lineout", "caked", "heatmap", "chart", "graph",
            "live viewer", "overlay", "pattern", "diffraction image",
            "peak plot", "grain plot", "3d grain", "spots",
            "ring", "fit result", "caking", "zarr", "lineout.xy",
            # Compound keywords — disambiguation when "plot/show" + domain word
            "plot calibra", "show calibra", "view calibra", "display calibra",
            "plot the calibra", "show the calibra", "view the calibra",
            "display the calibra", "see the calibra",
            "calibration result", "calibrant result",
            "plot the lineout", "show the grain",
            "plot the caked", "show the caked", "plot the integration",
            "show the integration", "plot the raw", "show the raw",
        },
        "motor": {
            "motor", "move", "position", "caget", "caput", "epics",
            "ioc", "rbv", "readback", "jog", "tweak", "home motor",
            "stop motor", "velocity", "speed", "limit switch",
            "soft limit", "hls", "lls", "dmov", "pv", "channel access",
            "20idmotsim", "motorsim", "rename motor", "motor name", "desc",
            "samx", "samy", "samz", "detx", "dety", "detz",
            # PV name patterns — " m1", " m2", etc. (leading space avoids false matches)
            " m1", " m2", " m3", " m4", " m5", " m6", " m7", " m8",
        },
    }

    def __init__(self, execute_tool_fn: ExecuteToolFn,
                 all_tools: List[Dict], context=None):
        self.runner    = AgentRunner(execute_tool_fn)
        self.all_tools = all_tools
        self.context   = context
        self.conversation_history: List[Dict] = []
        self.logger    = InteractionLogger()
        self._last_agent: Optional[APEXAAgent] = None

    def clear_history(self):
        self.conversation_history = []
        self._last_agent = None

    def _route(self, query: str) -> APEXAAgent:
        q = query.lower()
        scores = {
            domain: sum(1 for kw in keywords if kw in q)
            for domain, keywords in self._KEYWORDS.items()
        }
        best = max(scores, key=scores.get)

        if scores[best] > 0:
            # Break ties: if analysis ties with another domain, prefer analysis
            # (it's the most general agent and handles post-calibration workflows)
            top_score = scores[best]
            tied = [d for d, s in scores.items() if s == top_score]
            if len(tied) > 1 and "analysis" in tied:
                best = "analysis"
            return self._ROUTES[best]      # strong keyword match → switch agent

        # No keywords matched — stay with current agent if we have one
        # This handles follow-ups like "yes", "ok", "fetch one for Ceria"
        if self._last_agent is not None:
            return self._last_agent

        return ANALYSIS_AGENT              # first query, no context → default

    async def process(self, query: str, provider: ArgoProvider,
                      use_history: bool = True,
                      on_tool_result: OnToolResultFn = None) -> str:
        agent   = self._route(query)
        self._last_agent = agent
        history = self.conversation_history if use_history else None

        log_entry = self.logger.start(query, model=provider.model)
        log_entry.set_agent(agent.name)

        result = await self.runner.run(
            agent, query, provider, self.all_tools, history,
            log_entry=log_entry,
            on_tool_result=on_tool_result,
        )

        # Detect if the agent looped (>3 calls to a single tool = loop)
        n_calls = len(log_entry.tool_calls)
        looped = n_calls > 3 and len(set(
            tc.name for tc in log_entry.tool_calls
        )) == 1
        log_entry.finish(result, iterations=len(log_entry.tool_calls), looped=looped)
        self.logger.save(log_entry)

        if use_history:
            self.conversation_history.extend([
                {"role": "user",      "content": query},
                {"role": "assistant", "content": result},
            ])
            # Keep last 20 messages (10 exchanges) to avoid token overflow
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

        if self.context:
            self.context.add_analysis(agent.name, result)

        return result
