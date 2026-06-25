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
from collections import Counter
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
            # Claude Opus 4.7: no sampling params (temperature/top_p/top_k silently
            # removed by Argo, but cleaner to not send them).  1M context window,
            # 128K output.  Thinking mode available via output_config if needed.
            if self.model == "claudeopus47":
                payload.pop("temperature", None)
            # Claude Opus 4.6: requires temperature + top_p (Argo rejects without them)
            # Claude Haiku 4.5, Sonnet 4.5/4.6: reject temperature + top_p
            elif self.model in ("claudesonnet45", "claudesonnet46", "claudehaiku45"):
                pass   # no top_p for these models
            else:
                # claudeopus46, claudeopus45, claudeopus41, claudesonnet4, etc.
                payload["top_p"] = 0.9
        elif self.model == "gpt55":
            # GPT-5.5: temperature must be exactly 1 (no other value accepted)
            payload["temperature"] = 1
            payload["max_completion_tokens"] = 16000
            payload["top_p"] = 0.9
        elif self.model.startswith("gpto") or self.model.startswith("gpt5"):
            # o-series and GPT-5 family: use max_completion_tokens, no temperature/top_p
            # (Argo returns HTTP 400 if either is sent.)
            payload.pop("temperature", None)
            payload["max_completion_tokens"] = 16000
        elif self.model.startswith("gpt"):
            # gpt4o, gpt41, etc.: max_completion_tokens + top_p OK
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
    # Profile-Then-Reason mode: prepend a plan-first directive to the system
    # prompt so the model emits ALL needed tool calls in one response (concurrent
    # dispatch) and prefers a single compound tool over looping a primitive.
    # Combined with the runtime fan-out guard, this prevents the failure mode
    # where the model emits 22 xray_calculate calls instead of one
    # enumerate_bragg_rings call.
    use_planning: bool = False


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
    use_planning = True,   # plan all tool calls up front; prefer compound tools
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

PRE-WORKFLOW VALIDATION — rules for heavyweight HEDM reconstruction (ff, nf, pf):
1. SKIP validate_parameter_file entirely when data_file (.MIDAS.zip or .zarr.zip) is explicitly provided.
   The tool passes --skip-validation to midas-pipeline automatically; file-discovery keys
   (RawFolder, FileStem, StartNr, EndNr) are not required in zarr mode.
   → Just call run_ff_hedm_full_workflow directly with the provided paths.

2. CALL validate_parameter_file ONLY when no data_file is given AND the user has only
   a Parameters.txt + raw frame directory. If validation fails on file-discovery keys
   and a zarr is available, skip to the zarr path (rule 1 above).

3. Do NOT validate before: midas_integrate_2d_to_1d, midas_auto_calibrate, run_gsas_refinement

4. If validation finds real geometry errors (wrong Lsd, BC, Wavelength, SpaceGroup),
   call diagnose_parameter_file and fix those before proceeding.

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
search later.

Reading the tool result:
- The tool returns JSON with "results_count", "excerpts", and "references".
- Each excerpt has fields: source, citation, page, similarity, excerpt.
- "similarity" is in [0,1]. Anything >= 0.30 is a usable match. Anything >= 0.60 is
  a strong match. Do NOT dismiss a match just because the wording differs from your
  expectation — the chunk text and citation are what matter.

How to write the answer:
- If results_count > 0 AND at least one excerpt has similarity >= 0.30:
    1. Build the answer from the excerpt text. Quote or paraphrase the chunks.
    2. Cite EVERY substantive claim inline using the citation field, formatted as
       "(FirstAuthor Year, p.PAGE)" — e.g. "(Bernier 2020, p.36)".
    3. End with a "References:" section listing each unique citation verbatim from
       the tool's "references" list.
    4. Do NOT add background facts the excerpts don't support. If the excerpts are
       narrow, the answer should be narrow.
- If results_count == 0 OR every similarity < 0.30:
    Open with: "No matching sources in the knowledge base — answering from general
    background:" then give the answer. Do NOT fabricate citations.

Never invent citations. Never paraphrase a source you didn't retrieve. If unsure
which excerpts are strong, list them all and let similarity speak for itself.

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

STEP 1: Find the data file. Call list_directory ONCE on the most specific path given.
  - If the user gives an integration/ path → list that directory directly. Do NOT also list the parent.
  - Integration outputs (*_lineout.xy, *.zarr.zip, *_lineout.bin) are in <dir>/integration/
  - Calibration outputs (*_corr.csv) are in the calibration directory
  - Always prefer *.zarr.zip over plain *.hdf or *.caked.hdf — the zarr archive is the complete output
  - ONE list_directory call is sufficient. If the user gave the path, trust it.
STEP 2: Match the file to the correct viewer — pick ONE:

| File pattern | viewer name | When to use |
|---|---|---|
| *_corr.csv | plot_calibrant_results | Calibration fit, calibration QC, lattice-vs-η |
| *.zarr.zip (integration output) | plot_caked_peaks | BEST for integration results — shows 2D heatmap + 1D profile together |
| *_lineout.xy (2-col from MIDAS integrator) | plot_caked_peaks on the *.zarr.zip | No dedicated viewer for 2-col lineout; use zarr viewer instead |
| *_lineout.xy (4-col from extract_lineouts) | plot_lineout_results | Only for extract_lineouts.py output — 4 columns (2θ, raw, bg, corrected) |
| compare calibrant vs sample lineouts | plot_lineout_comparison | Use --paramFN for ring position overlay |
| *_lineout.bin (live) | live_viewer | Real-time GPU streaming monitor |
| *_caked.hdf.zarr.zip | plot_caked_peaks | Caked data, integrated image, 2D heatmap (PREFERRED for caked data) |
| *_caked_peaks.h5 | plot_caked_peaks | Peak fitting results |
| Raw .tif/.ge/.h5 | ff_asym_qt | Raw detector image, diffraction image, ring overlays |
| Grains.csv + .zarr | interactiveFFplotting | FF-HEDM grain map, grain results |
| .mic/.map (NF) | nf_qt | NF-HEDM microstructure |

DISAMBIGUATION — when the user request is ambiguous, pick ONE using these rules:
- "calibrated image" / "calibration results" / "calibration fit" → plot_calibrant_results
- "caked image" / "caked data" / "integrated data" / "integration result" → plot_caked_peaks
- "integration results" / "show integration" / "lineout" / "1D profile" → plot_caked_peaks on *.zarr.zip (NOT plot_lineout_results — that only works with 4-col extract_lineouts.py output)
- "compare lineouts" / "calibrant vs sample" → plot_lineout_comparison
- "peak fitting results" (4-col .xy from extract_lineouts.py) → plot_lineout_results
- RULE: whenever a *.zarr.zip exists alongside a *_lineout.xy, always prefer the zarr viewer (plot_caked_peaks) — it shows more information
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

🔒 TOOL RESULTS ARE GROUND TRUTH — NEVER CAPITULATE:
When a tool result and a user's claim conflict, the TOOL IS CORRECT. Do NOT:
- Apologise for what the tool returned
- Agree with the user's count/value/filename if it contradicts the tool
- Re-run a tool just because the user challenges its output
- Say "you're right" when the tool result proves otherwise

DO say: "The tool returned [X]. I'm confident that's correct — the listing shows
[exact tool output]." Then offer to re-run the tool ONCE if the user insists,
and again report the tool result verbatim.

This applies especially to: file counts, frame numbers, lattice parameters, beam
centre coordinates, detector distances, and calibration residuals. These values come
from measurements — the instrument does not make counting errors.

The ONLY exception: if you made an arithmetic error SUMMARISING a tool result
(e.g., the tool returned 20 files and you wrote 21), then correct yourself and
state the correct value from the tool result. Do NOT apologise beyond one word.

✅ SHELL UTILITIES VIA run_command — USE THESE FREELY:
run_command is available for grep, awk, sed, find, wc, sort, uniq, diff,
head, tail, cat, ls, du, stat, and other standard utilities.
Use them whenever they are faster or more precise than a dedicated tool.

Pipes (|), semicolons (;), &&, ||, and redirections (>, >>) all work.
Each executable in the pipeline must be in the allowed list.

CORRECT patterns:
  # Count .h5 files in a directory
  TOOL_CALL: run_command
  ARGUMENTS: {"command": "find /path -name '*.h5' -type f | wc -l"}

  # Search inside a parameter file and preview
  TOOL_CALL: run_command
  ARGUMENTS: {"command": "grep -n 'Wavelength\\|LatticeConstant\\|Lsd' /path/refined_params.txt"}

  # Find the most recently modified parameter file
  TOOL_CALL: run_command
  ARGUMENTS: {"command": "find /path -name 'refined_MIDAS_params*.txt' | sort -t_ -k1 | tail -n 1"}

  # Count unique grain orientations in a CSV
  TOOL_CALL: run_command
  ARGUMENTS: {"command": "awk -F, 'NR>1 {print $3}' /path/Grains.csv | sort | uniq | wc -l"}

  # Preview a large file and filter for errors
  TOOL_CALL: run_command
  ARGUMENTS: {"command": "grep -i 'error\\|warning\\|failed' /path/autocal.log | head -n 20"}

  # Check sizes of all result files
  TOOL_CALL: run_command
  ARGUMENTS: {"command": "ls -lh /path/*.csv && du -sh /path/integration/"}

  # Compare two parameter files
  TOOL_CALL: run_command
  ARGUMENTS: {"command": "diff /path/old_params.txt /path/new_params.txt"}

  # Save grep output to a file
  TOOL_CALL: run_command
  ARGUMENTS: {"command": "grep 'Wavelength' /path/params.txt > /path/wavelength_check.txt"}

  # Multi-command inline script via bash -c (equivalent to Claude Code's Bash tool)
  TOOL_CALL: run_command
  ARGUMENTS: {"command": "bash -c 'mkdir -p /path/ceo2_att3 && cp /path/Ceria_63keV_900mm_100x100_att3_1p0s_012220.h5 /path/ceo2_att3/'"}

  # Loop over files to create output directories for each calibrant
  TOOL_CALL: run_command
  ARGUMENTS: {"command": "bash -c 'for f in /path/Ceria_*.h5; do name=$(basename $f .h5); mkdir -p /path/cal_$name; done'"}

  # Check convergence quality across all calibration runs
  TOOL_CALL: run_command
  ARGUMENTS: {"command": "bash -c 'for f in /path/*.corr.csv; do echo \"=== $f ===\"; tail -1 $f; done'"}

NOTE: rm/rmdir/unlink are NOT in the allowed list — deletion via run_command
will return "Command not allowed". Use the write_file tool or ask the user
to delete manually from the terminal.

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
4. For calibration → TOOL_CALL: midas_auto_calibrate  ARGUMENTS: {"image_file": "/path/to/file.h5"}
   (parameter is image_file, NOT data_file — always use image_file)
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

⚡ BE DECISIVE — DO NOT ASK FOR PERMISSION:
When you have enough context to make a recommendation, MAKE IT and proceed.
Do NOT end responses with "Would you like me to...?" or "Shall I...?" or
"Do you want me to...?" — these stall the workflow and frustrate operators.
Instead: state your recommendation, state why, then either execute it (if
it passes the strategy gate) or say "Type 'go' to proceed."
The operator is in charge — they will redirect you if your recommendation is wrong.

PATH HANDLING — CRITICAL:
- ALWAYS use ABSOLUTE paths in tool arguments
- When the user says "test5", they mean "<CWD>/test5" (where <CWD> is shown below)
- When the user says "current directory", they mean the CWD shown below
- Convert ALL relative paths to absolute by prepending the CWD
- If the user provides an absolute path (starts with /), use it as-is
- NEVER pass just a filename — always include the full directory path

"""

# Plan-first preamble for agents with use_planning=True (currently Analysis).
# Teaches the model the REASON → ACT pattern used by Claude Code and modern
# agentic harnesses: explore/observe first, write a structured plan, then
# execute. The plan and execution happen in the same response — no extra
# round-trips — but the plan MUST precede the first long-running tool call.
_PLAN_FIRST_PREAMBLE = """🧭 REASON FIRST. EXECUTE SECOND.

For any request involving more than one step or any choice among inputs,
follow this pattern EXACTLY — in this order, in ONE response:

┌─────────────────────────────────────────────────────────────────┐
│ SITUATION: [what you observe — files present, calibrant types,  │
│            conditions, what is already done vs. missing]        │
│                                                                 │
│ GAP: [what is needed before execution can start — e.g., no     │
│       parameter file found; calibration must precede           │
│       integration; which files are candidates and why]          │
│                                                                 │
│ PLAN:                                                           │
│   Step 1. [action] — [rationale: why this file, why this order]│
│   Step 2. [action] — [rationale]                               │
│   Step N. ...                                                   │
│                                                                 │
│ Executing step 1:                                               │
│ TOOL_CALL: tool_name                                            │
│ ARGUMENTS: {...}                                                │
└─────────────────────────────────────────────────────────────────┘

Rules:
- SITUATION and GAP must be filled from ACTUAL tool results or the
  conversation history. NEVER fill them from training-data assumptions.
- PLAN must name the specific files/calibrants chosen and WHY
  (e.g., "att3 CeO2 — mid-range attenuation avoids saturation at att0
  and underexposure at att6").
- If the choice is genuinely ambiguous (multiple equally-valid options),
  state both and ask ONE question. Do not ask permission on every step.
- Emit ALL independent tool calls concurrently in the same TOOL_CALL block.
- After each tool result, update your plan if needed, then continue.

⚠️ COMPOUND OVER PRIMITIVE — REQUIRED:
If you are about to emit the same tool more than twice with sequential
parameters (e.g., xray_calculate for hkl=(1,1,1) then (2,0,0) …), STOP.
A compound tool exists. Find it (description says "all", "enumerate",
"batch", "rings", "summary") and use it ONCE.

After you receive tool results, your DEFAULT next action is to ANSWER or
continue the plan. Only emit another TOOL_CALL block if information you
genuinely need is NOT in any prior tool result.

"""

# Tools that are long-running, irreversible, or require a file/strategy choice.
# Before any of these is dispatched, the runner requires that the model wrote
# at least _STRATEGY_MIN_WORDS words of reasoning prose in the same response.
# If not, the call is rejected and the model is asked to state its strategy
# first. This is the APEXA equivalent of Claude Code's pattern of requiring
# the model to explain before acting on any Bash/Edit call.
#
# Motor motion tools are deliberately excluded here — they have their own
# hardware safety gate and their own confirmation flag (confirm_large_move).
# Adding them here would double-gate and slow down valid motor commands.
_PLAN_REQUIRED_TOOLS: frozenset = frozenset({
    # Calibration (choice of file, calibrant, energy, Lsd)
    "midas_auto_calibrate",
    "run_ff_calibration",
    # Integration (choice of data file, param file, output format)
    "midas_integrate_2d_to_1d",
    "midas_batch_integrate",
    # Refinement (choice of data + CIF + limits)
    "run_gsas_refinement",
    # Combined pipeline (choice of backend, param file, data file, CIF)
    "run_live_analysis",
    # HEDM reconstruction workflows (long-running, choice of param file)
    "run_ff_hedm_full_workflow",
    "run_nf_hedm_reconstruction",
    "run_pf_hedm_workflow",
})

# Strategy gate: before dispatching a _PLAN_REQUIRED_TOOLS call the runner
# checks that the model wrote a plan in the REASON→ACT format. Detection
# looks for any of these structural markers rather than raw word count,
# so "Using att3." (a choice with no rationale) does NOT pass.
# A plan passes if it contains at least one of:
#   - "SITUATION:" or "GAP:" or "PLAN:" (explicit template markers)
#   - "because" / "since" / "in order to" (causal reasoning)
#   - "step 1" / "first," / "first I" (ordered sequence)
#   - "no parameter file" / "calibrat" + "before integrat" (domain sequencing)
# Word-count fallback: ≥20 words of prose regardless of markers (a complete
# sentence with reasoning will naturally reach this).
_STRATEGY_MIN_WORDS = 20   # fallback if no structural markers found
_STRATEGY_MARKERS = re.compile(
    r'(?:'
    r'SITUATION:|GAP:|PLAN:'                          # explicit template
    r'|(?:because|since|in order to|so that)\b'       # causal connective
    r'|step\s+1\b|(?:^|\.\s+|\n)first[,\s]'          # sequence marker
    r'|no\s+param(?:eter)?\s+file'                    # domain gap
    r'|calibrat\w+\s+(?:before|first|must)'           # domain sequencing
    r'|integrat\w+\s+(?:requires?|needs?)\s+calibrat' # domain dependency
    r')',
    re.I | re.MULTILINE,
)

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

    # Bare `TOOL_CALL: name` (any following args parsed separately — Format B).
    _TOOL_CALL_NAME_RE = re.compile(r'TOOL_CALL:\s*([A-Za-z_]\w*)')
    # Anthropic native tool block that leaks through Argo as plain text (Format C).
    _TOOL_USE_XML_RE = re.compile(r'<tool_use>(.*?)</tool_use>', re.DOTALL)
    # A `key: value` or `key = value` argument line (Format B body).
    _KV_LINE_RE = re.compile(r'^[-*\s]*([A-Za-z_]\w*)\s*[:=]\s*(.+?)\s*$')

    @staticmethod
    def _coerce_arg_value(s: str):
        """Coerce a bare string argument value to int/float/bool/null when it
        clearly is one; otherwise return the de-quoted string. Keeps paths as
        strings (they don't parse as numbers) but turns energy_kev: 61.332 into
        a float, matching what ARGUMENTS:{json} would have produced."""
        v = s.strip()
        if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
            return v[1:-1]                       # de-quote, keep as string
        low = v.lower()
        if low in ("true", "false"):
            return low == "true"
        if low in ("null", "none"):
            return None
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    def _parse_text_tool_calls(self, text: str) -> List[ToolCall]:
        """Extract tool calls from model text output, tolerant of format drift.

        Argo strips native tool_calls, so this text path is the only way tools
        execute. Models (notably claudeopus47) drift between three surface forms;
        we accept all of them:
          A. TOOL_CALL: name  +  ARGUMENTS: {json}        (canonical)
          B. TOOL_CALL: name  +  key: value lines         (no ARGUMENTS json)
          C. <tool_use><tool_name>..</tool_name><parameters>..</parameters></tool_use>
        """
        calls: List[ToolCall] = []
        n = 0
        consumed_spans: List[tuple] = []   # (start,end) of Format-A matches

        # ── Format A: TOOL_CALL: name + ARGUMENTS: {json} (primary) ──────────
        for match in self._TOOL_CALL_RE.finditer(text):
            name = match.group(1).strip()
            try:
                args = json.loads(match.group(2))
            except json.JSONDecodeError:
                continue
            calls.append(ToolCall(id=f"text_tc_{n}", name=name, arguments=args))
            n += 1
            consumed_spans.append((match.start(), match.end()))

        # ── Format B: TOOL_CALL: name followed by bare key:value lines ───────
        for nm in self._TOOL_CALL_NAME_RE.finditer(text):
            # Skip TOOL_CALL occurrences already handled by Format A.
            if any(s <= nm.start() < e for s, e in consumed_spans):
                continue
            name = nm.group(1).strip()
            args: Dict = {}
            for line in text[nm.end():].splitlines():
                ls = line.strip()
                if not ls:
                    if args:
                        break          # blank line ends a populated arg block
                    continue
                if ls.startswith("<") or ls.upper().startswith("TOOL_CALL:"):
                    break              # next call / XML / prose — stop
                kv = self._KV_LINE_RE.match(ls)
                if not kv:
                    break              # first non key:value line ends the block
                key = kv.group(1)
                if key.lower() in ("tool_call", "arguments"):
                    break
                args[key] = self._coerce_arg_value(kv.group(2))
            if args:
                calls.append(ToolCall(id=f"text_tc_{n}", name=name, arguments=args))
                n += 1

        # ── Format C: <tool_use> XML block ──────────────────────────────────
        for m in self._TOOL_USE_XML_RE.finditer(text):
            body = m.group(1)
            nm = re.search(r'<tool_name>\s*(.*?)\s*</tool_name>', body, re.DOTALL)
            if not nm:
                continue
            name = nm.group(1).strip()
            args = {}
            pm = re.search(r'<parameters>(.*?)</parameters>', body, re.DOTALL)
            if pm:
                pbody = pm.group(1).strip()
                try:
                    parsed = json.loads(pbody)
                    if isinstance(parsed, dict):
                        args = parsed
                except json.JSONDecodeError:
                    for em in re.finditer(r'<(\w+)>(.*?)</\1>', pbody, re.DOTALL):
                        args[em.group(1)] = self._coerce_arg_value(em.group(2).strip())
            calls.append(ToolCall(id=f"text_tc_{n}", name=name, arguments=args))
            n += 1

        return calls

    def _strip_tool_calls_from_text(self, text: str) -> str:
        """Remove tool-call blocks (all 3 formats) to get the prose part."""
        # Format A: TOOL_CALL: name + ARGUMENTS: {json}
        clean = self._TOOL_CALL_RE.sub('', text)
        # Format C: <tool_use>...</tool_use>
        clean = self._TOOL_USE_XML_RE.sub('', clean)
        # Format B: a bare `TOOL_CALL: name` and its trailing key:value lines.
        out_lines: List[str] = []
        skipping = False
        for line in clean.splitlines():
            ls = line.strip()
            if ls.upper().startswith("TOOL_CALL:"):
                skipping = True            # drop this line + following kv lines
                continue
            if skipping:
                if not ls:
                    skipping = False       # blank line ends the kv block
                    continue
                if self._KV_LINE_RE.match(ls):
                    continue               # still inside the arg block — drop
                skipping = False           # prose resumes — keep this line
            out_lines.append(line)
        clean = "\n".join(out_lines).strip()
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

    @staticmethod
    def _extract_dir_file_count(messages: List[Dict]) -> Optional[int]:
        """Scan conversation messages for the most recent list_directory result
        and return the total file count it reported, or None if not found.

        Used by _check_count_hallucination to verify that the model's stated
        file/frame count matches what the tool actually returned.
        """
        # list_directory results appear as user messages containing JSON with
        # a "total_files" key, or as the compact listing text "[Tool Result
        # for list_directory]".  Try both forms.
        for msg in reversed(messages):
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            if "[Tool Result for list_directory]" not in content and "list_directory" not in content:
                continue
            # Try JSON form first
            m = re.search(r'"total_files"\s*:\s*(\d+)', content)
            if m:
                return int(m.group(1))
            # Fall back to compact-listing summary line: "N directories, M files"
            m = re.search(r'(\d+)\s+(?:directories?,\s*)?(\d+)\s+files?', content)
            if m:
                return int(m.group(2))
        return None

    @staticmethod
    def _check_count_hallucination(text: str, messages: List[Dict]) -> Optional[str]:
        """Return a rejection message if the model asserted a specific file /
        frame count that contradicts what list_directory actually reported.

        This catches the failure mode where the model reads '920 files' from
        the tool result but writes '360 TIFF images' (or any other fabricated
        number) because its training data associates 'aero' with 360-frame
        HEDM rotation scans.

        Returns None if no contradiction is found (including if no prior
        list_directory result exists in the conversation).
        """
        actual = AgentRunner._extract_dir_file_count(messages)
        if actual is None:
            return None

        # Extract ALL integers from the model's text that appear near
        # count-like context words.  We cast a fairly wide net so we don't
        # miss paraphrases like "135+ frames", "00000 to 00359", "N images".
        count_patterns = [
            # "360 TIFF images", "920 files", "135 frames"
            re.compile(r'\b(\d+)\s*\+?\s*(?:tiff|tif|image|frame|file|scan|projection)s?\b', re.I),
            # "numbered from 00000 to 00359"
            re.compile(r'\b00000\s+to\s+0*(\d+)\b', re.I),
            # "total frames: 360"
            re.compile(r'(?:total|frame|file|image)\s+count[:\s]+(\d+)', re.I),
            # "360° with 1° steps" → 360 could come from rotation claim; too
            # broad — only flag if the number also appears in a file-count context
            # (handled by the patterns above).
        ]
        claimed_counts = set()
        for pat in count_patterns:
            for m in pat.finditer(text):
                claimed_counts.add(int(m.group(1)))

        # Allow a ±1 tolerance (sometimes the agent drops header/footer rows).
        for claimed in claimed_counts:
            if abs(claimed - actual) > 1:
                return (
                    f"⛔ COUNT HALLUCINATION DETECTED.\n\n"
                    f"You stated **{claimed}** files/frames but the `list_directory` "
                    f"tool returned **{actual}** files.\n\n"
                    "You MUST NOT invent counts, frame ranges, or angular steps from "
                    "your training data. Report ONLY what the tool told you:\n"
                    f"  • Total files: {actual}\n"
                    "  • File naming pattern: (from the listing above)\n\n"
                    "Rewrite your answer using only the tool result. "
                    "Do NOT claim to know the omega range, frame count, or "
                    "rotation geometry unless a parameter file or metadata tool "
                    "confirmed it."
                )
        return None

    @staticmethod
    def _select_history(history: List[Dict], max_msgs: int) -> List[Dict]:
        """Pick a compact slice of conversation history.

        Naive `history[-max_msgs:]` drops the original user query that
        established context. Keep that first user message AND the most recent
        (max_msgs - 1) messages so payload stays small but the agent doesn't
        forget what the session is about.
        """
        if len(history) <= max_msgs:
            return list(history)
        first_user_idx = next(
            (i for i, m in enumerate(history) if m.get("role") == "user"),
            None,
        )
        if first_user_idx is None or first_user_idx >= len(history) - (max_msgs - 1):
            return list(history[-max_msgs:])
        return [history[first_user_idx]] + list(history[-(max_msgs - 1):])

    # ── Main loop ────────────────────────────────────────────────────────────

    async def run(self, agent: APEXAAgent, query: str,
                  provider: ArgoProvider, all_tools: List[Dict],
                  history: Optional[List[Dict]] = None,
                  max_iterations: int = 10,
                  log_entry: Optional[InteractionEntry] = None,
                  on_tool_result: OnToolResultFn = None,
                  history_summary: str = "") -> str:

        tools = self._filter_tools(agent.tool_names, all_tools)

        # Build system message: strong preamble + agent-specific instructions.
        # Agents with use_planning=True get the plan-first preamble prepended,
        # which (a) tells the model to batch all needed tool calls in one
        # response and (b) explicitly forbids primitive-tool fan-out where a
        # compound tool exists.
        cwd = str(Path.cwd())
        plan_pre = _PLAN_FIRST_PREAMBLE if getattr(agent, "use_planning", False) else ""
        system_content = plan_pre + _TOOL_PREAMBLE + f"\nCurrent working directory (CWD): {cwd}\n" + agent.instructions

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
        # Compacted summary of older turns (from the orchestrator). Injected as
        # a second system message so it is never dropped by _select_history and
        # always frames the recent verbatim turns. Carries load-bearing facts
        # (energy, paths, calibration params) from beyond the recent window.
        if history_summary:
            messages.append({
                "role": "system",
                "content": (
                    "Summary of earlier conversation in this session "
                    "(older turns, compacted — treat as established context):\n"
                    + history_summary
                ),
            })
        if history:
            # Motor/viz agents need less history (repetitive commands confuse the model)
            hist_limit = 4 if agent.name in ("MotorAgent", "VisualizationAgent") else 8
            selected = self._select_history(history, hist_limit)
            messages.extend(selected)
        messages.append({"role": "user", "content": query})

        last_tool_name = None          # track repeated tool calls (consecutive)
        _last_tool_args = None         # track repeated tool arguments
        repeat_count   = 0
        # Cumulative tool-call counter across ALL iterations of this user turn.
        # Catches iterative fan-out (model emits one primitive call per turn
        # for N turns) which the per-response guard misses.  When any single
        # tool reaches the threshold, we send the same compound-tool redirect.
        _turn_tool_counts: Counter = Counter()
        # Also track unique argument sets per tool so we can distinguish
        # legitimate multi-file reads from true fan-out.
        _turn_tool_args: dict = {}     # tool_name → set of frozen arg strings
        _FANOUT_THRESHOLD = 3          # same as per-response guard
        # Tools where calling N times with N DISTINCT arguments is legitimate.
        # Fan-out only fires when the SAME arguments recur, not on distinct inputs.
        # midas_auto_calibrate is included: calibrating 7 files with 7 different
        # paths is the correct multi-file pattern (no "batch calibrate" tool exists).
        _MULTI_ARG_OK_TOOLS = {
            "read_file", "get_file_info", "run_command",
            "midas_auto_calibrate", "midas_integrate_2d_to_1d",
            "run_gsas_refinement",
            # Read-only inspection tools: calling them on N distinct paths/files
            # is normal exploration, not fan-out. Only repeated IDENTICAL args
            # trip the guard (the arg-diversity check below enforces that).
            "list_directory", "read_document", "inspect_dataset_file",
            "diagnose_parameter_file", "validate_parameter_file",
        }
        # Track which _PLAN_REQUIRED_TOOLS have been gate-rejected this turn.
        # On a second rejection of the same tool, strengthen the message to
        # emphasise that plan + TOOL_CALL must be in the SAME response.
        _plan_gate_strikes: Counter = Counter()
        # Whether the conversation history contains a plan from a prior turn.
        # Used to skip the strategy gate when the user approved an existing plan.
        _prior_plan_in_history = any(
            bool(_STRATEGY_MARKERS.search(m.get("content") or ""))
            for m in (history or [])
            if m.get("role") == "assistant"
        )
        _approval_re = re.compile(
            r'\b(yes|go|proceed|ok|sure|start|execute|run\s+it|do\s+it|'
            r'go\s+with\s+this|go\s+ahead|sounds\s+good|let\'?s\s+do\s+it)\b',
            re.I,
        )
        _query_is_approval = bool(_approval_re.search(query)) and len(query.split()) <= 10

        for _ in range(max_iterations):
            response = await provider.chat(messages, agent.temperature)

            # ── Mode 1: Native API tool_calls ──
            if response.tool_calls:
                # Cross-iteration fan-out check for native tool_calls.
                _turn_tool_counts.update(tc.name for tc in response.tool_calls)
                _worst_tool, _worst_count = _turn_tool_counts.most_common(1)[0]
                if _worst_count >= _FANOUT_THRESHOLD:
                    print(f"  \033[33m⚠ cumulative fan-out:\033[0m {_worst_count}× {_worst_tool}")
                    messages.append(self._assistant_message(response, provider.model))
                    messages.append({
                        "role": "user",
                        "content": (
                            f"⛔ CUMULATIVE FAN-OUT: you have now called `{_worst_tool}` "
                            f"{_worst_count} times across this turn. "
                            "Use the compound tool that returns all values in ONE call instead. "
                            "Check your tool list for a tool whose description says 'all', "
                            "'enumerate', 'batch', 'rings', 'summary', 'report', or 'inspect'. "
                            "Call it ONCE and then ANSWER the user."
                        ),
                    })
                    continue
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

                # ── Runtime fan-out guards ───────────────────────────────────
                # Two complementary checks:
                # (A) Per-response: ≥3 of the same tool in ONE model response
                # (B) Cumulative: ≥3 of the same tool across ALL responses this
                #     turn — catches iterative single-call-per-turn fan-out.
                # Both are per-tool-agnostic structural checks; no per-tool rules.
                _tool_counts = Counter(tc.name for tc in text_calls)
                _top_tool, _top_count = _tool_counts.most_common(1)[0]

                # (A) per-response check — arg-diversity aware
                _per_resp_args = {
                    _top_tool: set(json.dumps(tc2.arguments, sort_keys=True)
                                   for tc2 in text_calls if tc2.name == _top_tool)
                }
                _per_resp_unique = len(_per_resp_args.get(_top_tool, set()))
                _per_resp_fanout = (
                    _top_count >= 3
                    and not (_top_tool in _MULTI_ARG_OK_TOOLS and _per_resp_unique >= _top_count)
                )
                if _per_resp_fanout:
                    print(f"  \033[33m⚠ fan-out guard:\033[0m {_top_count}× {_top_tool} — rejecting batch")
                    # run_command fan-out: the fix is glob consolidation, not
                    # a compound tool — give a more specific redirect.
                    if _top_tool == "run_command":
                        _fanout_msg = (
                            f"⛔ FAN-OUT DETECTED: you emitted {_top_count} separate `run_command` "
                            "calls. Shell commands must be consolidated into ONE call using "
                            "glob patterns or a semicolon-separated sequence.\n\n"
                            "Example — deleting multiple file types:\n"
                            "  WRONG: 5 separate `rm file1.csv`, `rm file2.csv`, ...\n"
                            "  RIGHT: `rm -f /path/*.corr.csv /path/*.checkpoint.txt /path/*.png`\n\n"
                            "Example — running multiple operations:\n"
                            "  WRONG: 3 separate run_command calls\n"
                            "  RIGHT: `mkdir -p dir1 dir2 dir3 && cp file1 dir1/ && cp file2 dir2/`\n\n"
                            "Rewrite as ONE run_command call now. Also note: if the command "
                            "contains `rm`, you must list the files first and confirm with "
                            "the user before deleting (previous calls may have been blocked)."
                        )
                    else:
                        _fanout_msg = (
                            f"⛔ FAN-OUT DETECTED: you emitted {_top_count} calls to "
                            f"`{_top_tool}` in one response. This is the failure mode "
                            "this system is designed to prevent.\n\n"
                            "A compound tool almost certainly exists that returns all of "
                            "these values in ONE call. Search YOUR TOOL LIST above for a "
                            "tool whose description mentions 'all', 'enumerate', 'list', "
                            "'batch', 'rings', 'summary', 'report', or 'inspect' and use "
                            f"THAT tool ONCE instead of {_top_tool} {_top_count} times.\n\n"
                            "Examples of the right pattern:\n"
                            "  • Per-hkl d-spacings → enumerate_bragg_rings (not many xray_calculate)\n"
                            "  • Per-file inspection → diagnose_parameter_file or inspect_dataset_file\n"
                            "  • Per-grain stats → read_grains_summary\n\n"
                            "If you genuinely need primitive calls (e.g., one xray_calculate "
                            "for a single user-asked d-spacing), emit AT MOST ONE call now, "
                            "then ANSWER the user."
                        )
                    messages.append({"role": "user", "content": _fanout_msg})
                    continue   # skip dispatch; retry with consolidated command or compound tool
                # end (A)

                # (B) cumulative fan-out check is deferred until AFTER the
                # strategy gate (below). A call the strategy gate rejects is
                # never executed, so counting it as fan-out here would let two
                # guards fight: the gate asks the model to retry with a plan,
                # and the retry would inflate the cumulative counter until the
                # fan-out guard kills the turn — the call never runs. Only
                # count calls that survive every guard and actually dispatch.

                # ── Strategy gate (Claude-Code-style pre-action reasoning) ───
                # If the response contains a tool from _PLAN_REQUIRED_TOOLS
                # (long-running / irreversible / choice-dependent), require
                # that the model wrote at least _STRATEGY_MIN_WORDS words of
                # reasoning prose in the SAME response before the TOOL_CALL.
                # If not, reject and ask for a strategy statement first.
                #
                # This mirrors how Claude Code works: the model must explain
                # what it is about to do and why before executing any action
                # that is hard to undo or requires a choice among inputs.
                # The check is tool-agnostic — _PLAN_REQUIRED_TOOLS is the
                # only configuration knob, no per-tool rules.
                _plan_needed = [tc for tc in text_calls
                                if tc.name in _PLAN_REQUIRED_TOOLS]
                if _plan_needed:
                    prose_words = len(prose.split()) if prose else 0
                    has_markers = bool(_STRATEGY_MARKERS.search(prose)) if prose else False
                    # Prior-plan bypass: if the conversation history already
                    # contains a structured plan (SITUATION/GAP/PLAN markers in
                    # a prior assistant turn) AND the current query is a short
                    # approval ("yes", "go with this", "proceed", etc.), the user
                    # has approved the plan — do not require it to be re-written.
                    plan_ok = (
                        has_markers
                        or prose_words >= _STRATEGY_MIN_WORDS
                        or (_prior_plan_in_history and _query_is_approval)
                    )
                    if not plan_ok:
                        _tool_names_str = ", ".join(
                            f"`{tc.name}`" for tc in _plan_needed
                        )
                        _plan_gate_strikes.update(tc.name for tc in _plan_needed)
                        _is_retry = any(
                            _plan_gate_strikes[tc.name] > 1 for tc in _plan_needed
                        )
                        print(f"  \033[33m⚠ strategy gate:\033[0m {_tool_names_str} — no plan ({prose_words}w, markers={has_markers}, retry={_is_retry})")
                        if _is_retry:
                            _gate_msg = (
                                f"⛔ PLAN STILL MISSING (second attempt on {_tool_names_str}).\n\n"
                                "⚠️ CRITICAL: THE PLAN AND THE TOOL_CALL MUST BE IN THE SAME RESPONSE.\n"
                                "You cannot send a plan and then wait — the runner does not continue "
                                "from a plan-only response. Write the plan AND the TOOL_CALL together:\n\n"
                                "SITUATION: ...\n"
                                "GAP: ...\n"
                                "PLAN:\n"
                                "  Step 1. [exact file] — [reason]\n\n"
                                "Executing step 1:\n"
                                f"TOOL_CALL: {_plan_needed[0].name}\n"
                                "ARGUMENTS: {...}\n\n"
                                "The TOOL_CALL must appear at the END of this response, after the plan. "
                                "Do NOT call raw MIDAS executables via run_command — use the dedicated "
                                f"tool `{_plan_needed[0].name}` which handles all parameters correctly."
                            )
                        else:
                            _gate_msg = (
                                f"⛔ PLAN REQUIRED before calling {_tool_names_str}.\n\n"
                                "Write a brief strategy (1-3 sentences) AND the TOOL_CALL in the SAME response.\n\n"
                                "For a simple parameter change + proceed (e.g. 'no dark file'), one sentence is enough:\n"
                                "  'Proceeding without dark subtraction using the same calibration geometry.'\n"
                                "  TOOL_CALL: midas_auto_calibrate\n"
                                "  ARGUMENTS: {...}\n\n"
                                "For a multi-file or first-time setup, use the full structure:\n"
                                "  SITUATION: [what files/conditions exist]\n"
                                "  GAP: [what must be resolved first]\n"
                                "  PLAN: Step 1. [specific file] — [reason]\n"
                                "  Executing step 1:\n"
                                f"  TOOL_CALL: {_tool_names_str}\n\n"
                                "Rules: name the EXACT file; plan + TOOL_CALL in ONE response."
                            )
                        messages.append({"role": "user", "content": _gate_msg})
                        continue   # let the model retry with a real plan

                # (B) cumulative fan-out check — runs only after the per-response
                # guard AND the strategy gate have passed, so the counter reflects
                # calls that will actually execute (not gate-rejected retries).
                _turn_tool_counts.update(_tool_counts)
                # Track unique argument fingerprints per tool so arg-diverse
                # tools (read_file on different files) don't false-fire.
                for tc in text_calls:
                    args_key = json.dumps(tc.arguments, sort_keys=True)
                    _turn_tool_args.setdefault(tc.name, set()).add(args_key)

                _cum_top, _cum_count = _turn_tool_counts.most_common(1)[0]
                if _cum_count >= _FANOUT_THRESHOLD and _cum_top == _top_tool and _top_count < _FANOUT_THRESHOLD:
                    # For tools where diverse args are legitimate, only fire
                    # if the argument SET is smaller than the call count (i.e.
                    # same args repeated, not different files each time).
                    _unique_args = len(_turn_tool_args.get(_cum_top, set()))
                    if _cum_top in _MULTI_ARG_OK_TOOLS and _unique_args >= _cum_count:
                        pass   # diverse args — not true fan-out, let it through
                    else:
                        # Cumulative threshold crossed and NOT already caught by (A)
                        print(f"  \033[33m⚠ cumulative fan-out:\033[0m {_cum_count}× {_cum_top}")
                        if _cum_top == "run_command":
                            _fanout_redirect = (
                                f"⛔ CUMULATIVE FAN-OUT: you have called `run_command` "
                                f"{_cum_count} times across this turn. "
                                "Consolidate into ONE call using a bash -c script or pipes:\n"
                                "  bash -c 'head -20 file1.csv; echo ---; head -20 file2.csv'"
                            )
                        else:
                            _fanout_redirect = (
                                f"⛔ CUMULATIVE FAN-OUT: you have called `{_cum_top}` "
                                f"{_cum_count} times across this turn in separate responses. "
                                "Use the compound tool that returns all values in ONE call."
                            )
                        messages.append({"role": "user", "content": _fanout_redirect})
                        continue

                _once_per_response = set()
                _ONCE_TOOLS = {"run_midas_viewer"}

                # ── Pre-validate all calls (guards run sequentially, side-effect-free
                #    on the network) so we know which ones to dispatch in parallel.
                to_execute: List[ToolCall] = []
                forced_break = False
                for tc in text_calls:
                    tc_args_str = json.dumps(tc.arguments, sort_keys=True)
                    if tc.name == last_tool_name and tc_args_str == _last_tool_args:
                        repeat_count += 1
                    else:
                        last_tool_name = tc.name
                        _last_tool_args = tc_args_str
                        repeat_count = 0

                    if repeat_count >= 2:
                        messages.append({
                            "role": "user",
                            "content": (
                                f"You already called {tc.name} with the same arguments and got the result above. "
                                "Do NOT call it again. Summarise the result for the user now."
                            ),
                        })
                        forced_break = True
                        break

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

                    if tc.name == "run_command":
                        cmd_str = str(tc.arguments.get("command", "")).lower()
                        # Guard: raw MIDAS calibration executables called directly
                        # instead of through the midas_auto_calibrate tool.
                        _MIDAS_CAL_BINS = [
                            "autocalibrateZarr", "autocalibratezarr",
                            "calibrantintegratoromp", "calibrantomp",
                            "calibrantpanelshiftsomp", "fittiltbclsdsample",
                        ]
                        if any(kw in cmd_str for kw in _MIDAS_CAL_BINS):
                            err = json.dumps({
                                "error": "Do NOT call MIDAS calibration binaries directly via run_command. "
                                         "Use TOOL_CALL: midas_auto_calibrate — it handles all parameters, "
                                         "energy/wavelength conversion, and AutoCalibrateZarr.py correctly.",
                                "correct_tool": "midas_auto_calibrate",
                            })
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"[Tool Result for {tc.name}]\n{err}\n\n"
                                    "You bypassed the calibration tool. Use midas_auto_calibrate instead:\n"
                                    "TOOL_CALL: midas_auto_calibrate\n"
                                    "ARGUMENTS: {\"image_file\": \"/path/to/Ceria_att3_*.h5\"}\n"
                                    "The tool auto-detects energy, Lsd, and calibrant from the filename.\n"
                                    "Required parameter name: image_file (NOT data_file, NOT image_path)."
                                ),
                            })
                            continue

                        if any(kw in cmd_str for kw in ["gsas", "refine", "rietveld", "gsas_ii_refine"]):
                            err = json.dumps({
                                "error": "Do NOT use run_command for GSAS-II refinement. "
                                         "Use TOOL_CALL: run_gsas_refinement with data_file (.zarr.zip) and cif_files.",
                                "correct_tool": "run_gsas_refinement",
                            })
                            messages.append({
                                "role": "user",
                                "content": f"[Tool Result for {tc.name}]\n{err}\n\n"
                                           "You used the WRONG tool. Use run_gsas_refinement instead of run_command. "
                                           "First call list_directory to find the .zarr.zip and .cif files, "
                                           "then call run_gsas_refinement with those paths.",
                            })
                            continue

                        # ── Destructive command gate ─────────────────────────
                        # Detect rm/rmdir/unlink in run_command arguments.
                        # Deletion is irreversible on beamline scratch storage
                        # (no recycle bin, no undo).  Require confirmation:
                        # the model must list EXACTLY what will be deleted and
                        # ask once before any rm executes.
                        # Exception: if the command is already preceded by a
                        # confirmation marker in the prose (CONFIRMED: or
                        # user-typed "yes, delete" / "go ahead" in prior turn),
                        # allow it through.
                        _RM_RE = re.compile(r'\brm\b|\brmdir\b|\bunlink\b', re.I)
                        if _RM_RE.search(cmd_str):
                            # rm/rmdir/unlink are NOT in ALLOWED_COMMANDS in
                            # beamline_core_server.py — the tool will return
                            # "Command not allowed: rm" regardless. Block here
                            # early so the model doesn't hallucinate success
                            # after receiving that error, and redirect to the
                            # correct workflow.
                            print(f"  \033[31m⛔ destructive gate:\033[0m rm blocked — not in ALLOWED_COMMANDS")
                            messages.append({
                                "role": "user",
                                "content": (
                                    "⛔ rm/rmdir IS NOT ALLOWED via run_command.\n\n"
                                    f"Your command `{tc.arguments.get('command', '')}` will fail "
                                    "because `rm` is not in the allowed command list for safety reasons. "
                                    "Do NOT retry with rm — it will always be rejected.\n\n"
                                    "To delete files on this beamline system, the correct approach is:\n"
                                    "1. Tell the user EXACTLY which files you want to delete "
                                    "(use find or ls via run_command to list them first)\n"
                                    "2. Ask the user to delete them manually from the terminal:\n"
                                    "   `rm -f /path/*.corr.csv /path/*.checkpoint.txt`\n"
                                    "3. Or ask if there is a dedicated cleanup script for this workflow.\n\n"
                                    "Do NOT claim files were deleted. They were NOT deleted."
                                ),
                            })
                            forced_break = True
                            break

                    to_execute.append(tc)

                # ── Execute all approved calls concurrently. Tools emitted in one
                #    model response are independent by construction (the model
                #    couldn't see any of their results when it produced them), so
                #    parallel dispatch is always safe here.
                async def _run_one(tc: ToolCall):
                    print(f"  \033[36m▸\033[0m \033[1m{tc.name}\033[0m")
                    t0 = time.monotonic()
                    result = await self._execute(tc.name, tc.arguments)
                    return tc, result, int((time.monotonic() - t0) * 1000)

                exec_results: List = []
                if to_execute:
                    exec_results = await asyncio.gather(
                        *[_run_one(tc) for tc in to_execute]
                    )

                # ── Process results sequentially to preserve message ordering and
                #    maintain deterministic side effects (logging, list_directory
                #    rendering, follow-up prompts).
                for tc, result, dur in exec_results:
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
                    if tc.name == "list_directory":
                        followup = (
                            "The directory listing is displayed above. "
                            "File count and filenames are GROUND TRUTH — do not dispute them, do not re-list. "
                            "\n\n"
                            "Now do TWO things in your response:\n"
                            "1. DESCRIBE what you found (calibrant types, conditions, file counts — "
                            "derived ONLY from the filenames and tool result, no training-data assumptions).\n"
                            "2. PROPOSE a master plan: based on what you see, outline the full recommended "
                            "analysis workflow in numbered steps (e.g., create output dirs → calibrate each "
                            "file → compare residuals → integrate with best params → report). "
                            "For each step: name the specific files/tools and the reason for that choice. "
                            "End with: 'Type **go** to execute Phase 1, or tell me what to change.'\n\n"
                            "Do NOT ask 'Would you like me to...?' — propose the plan and let the user "
                            "redirect if needed. If the user's original request was a specific task "
                            "(not just 'what's here'), skip the master plan and execute that task directly."
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
                if forced_break:
                    break
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

            # Count-hallucination guard: check if the model stated a file/frame
            # count that contradicts what list_directory actually returned.
            # This catches "360 TIFF images" when the tool said "920 files".
            if text:
                count_rejection = self._check_count_hallucination(text, messages)
                if count_rejection:
                    print(f"  \033[33m⚠ count hallucination detected — rejecting\033[0m")
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content": count_rejection})
                    continue

            return text or "Analysis complete."

        # ── Forced finalize at iteration cap ─────────────────────────────────
        # Loop exhausted without the model producing a tool-call-free final
        # response. Falling through to "return last message" silently drops
        # raw tool JSON onto the user. Instead, make ONE more LLM call with
        # tools forbidden and a hard instruction to summarise what we already
        # have. This guarantees a user-facing answer even when the model is
        # mid-fan-out at the cap.
        print(f"  \033[33m⚠ iteration cap reached — forcing finalize\033[0m")
        messages.append({
            "role": "user",
            "content": (
                "⛔ TOOL BUDGET EXHAUSTED. You have reached the maximum allowed "
                "tool calls for this turn. You are now FORBIDDEN from emitting "
                "any further TOOL_CALL blocks.\n\n"
                "Write the FINAL answer for the user RIGHT NOW. "
                "CRITICAL RULES for this final answer:\n"
                "1. If any tool calls were BLOCKED by a guard (fan-out, destructive, "
                "strategy gate), say so EXPLICITLY — do NOT claim success for "
                "operations that were blocked. Say 'X was blocked and did not execute.'\n"
                "2. Report only what ACTUALLY executed based on the tool results above.\n"
                "3. If an operation is partially complete (some files deleted, some not), "
                "state exactly what was done and what remains.\n"
                "4. Use markdown: **bold** key values, bullet points for lists.\n"
                "5. Do NOT apologise; do NOT describe your process; just report facts."
            ),
        })
        try:
            final_response = await provider.chat(messages, agent.temperature)
            final_text = final_response.content or ""
            # Strip any TOOL_CALL: blocks that slipped through (the model
            # sometimes ignores the no-tools instruction on the first try).
            final_text = self._strip_tool_calls_from_text(final_text).strip()
            if final_text:
                return final_text
        except Exception as e:
            print(f"  \033[31m✗ finalize call failed: {e}\033[0m")

        # True last resort: surface the last assistant text we have
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

    # Fast-path patterns: deterministic natural-language commands that map
    # 1:1 onto a single tool call. Executing them directly skips the entire
    # orchestrator/agent loop and saves a full LLM round-trip on the most
    # common queries. Patterns must be VERY specific to avoid false matches.
    _FAST_PATHS = [
        (re.compile(r'^\s*(?:list|ls|show)\s+(?:the\s+)?files?\s+(?:in|under|inside|of)\s+(.+?)\s*\??\s*$', re.I),
         "list_directory", lambda m: {"path": m.group(1).strip().strip('"\'')}),
        (re.compile(r'^\s*(?:list|ls|show)\s+(?:the\s+)?(?:current\s+)?(?:dir(?:ectory)?|files|folder)\s*\??\s*$', re.I),
         "list_directory", lambda m: {"path": "."}),
        (re.compile(r'^\s*what\s+files\s+are\s+(?:in|under|inside)\s+(.+?)\s*\??\s*$', re.I),
         "list_directory", lambda m: {"path": m.group(1).strip().strip('"\'')}),
    ]

    def __init__(self, execute_tool_fn: ExecuteToolFn,
                 all_tools: List[Dict], context=None):
        self.runner    = AgentRunner(execute_tool_fn)
        self.all_tools = all_tools
        self.context   = context
        self.conversation_history: List[Dict] = []
        # Running compacted summary of turns older than the recent window.
        # Built incrementally by _compact_history(); injected into model
        # context by the runner so long sessions retain early facts without
        # unbounded token cost.
        self.running_summary: str = ""
        self.logger    = InteractionLogger()
        self._last_agent: Optional[APEXAAgent] = None
        self._last_turn_had_tool_error: bool = False
        self._execute  = execute_tool_fn

    # Context-window management knobs (modern summarize-older + keep-recent).
    _KEEP_RECENT: int = 8       # messages kept verbatim in model context
    _COMPACT_TRIGGER: int = 16  # compact once history grows beyond this

    def clear_history(self):
        self.conversation_history = []
        self.running_summary = ""
        self._last_agent = None
        self._last_turn_had_tool_error = False

    def export_history(self) -> List[Dict]:
        """Return the conversation history for session persistence.

        Returns a shallow copy so the caller can serialize it without racing
        against in-flight turns mutating the live list.
        """
        return list(self.conversation_history)

    def import_history(self, history: List[Dict]):
        """Restore conversation history from a saved/auto-saved session.

        Keeps only the last 12 messages — the same working-context cap the
        process loop enforces (see conversation_history truncation) so a
        resumed session feeds the model the same amount of context a live
        one would, regardless of how long the saved transcript is.
        """
        if not history:
            return
        cleaned = [
            m for m in history
            if isinstance(m, dict) and "role" in m and "content" in m
        ]
        # Keep the recent verbatim window; older turns are carried by the
        # restored running_summary (see import_summary), matching how a live
        # session would present them after compaction.
        self.conversation_history = cleaned[-self._KEEP_RECENT:]

    def export_summary(self) -> str:
        """Return the running compacted summary for session persistence."""
        return self.running_summary

    def import_summary(self, summary: str):
        """Restore the running compacted summary from a saved session."""
        self.running_summary = summary or ""

    async def _compact_history(self, provider: "ArgoProvider"):
        """Summarize-older + keep-recent context management.

        When the conversation grows beyond _COMPACT_TRIGGER, fold every message
        older than the recent window into self.running_summary via one LLM call
        and drop them from conversation_history. This replaces the old hard
        12-message truncation, so a long beamline session keeps early
        load-bearing facts (beam energy, file paths, calibration params,
        decisions) instead of silently forgetting them. Best-effort: if the
        summarization call fails, fall back to a hard trim so memory stays
        bounded and the turn still completes.
        """
        if len(self.conversation_history) <= self._COMPACT_TRIGGER:
            return
        keep = self.conversation_history[-self._KEEP_RECENT:]
        older = self.conversation_history[:-self._KEEP_RECENT]
        if not older:
            return
        new_summary = await self._summarize_messages(older, provider)
        if new_summary:
            self.running_summary = new_summary
            self.conversation_history = keep
        else:
            # Summarization unavailable — stay bounded rather than grow forever.
            self.conversation_history = self.conversation_history[-self._COMPACT_TRIGGER:]

    async def _summarize_messages(self, messages: List[Dict],
                                  provider: "ArgoProvider") -> str:
        """Fold `messages` (and any prior summary) into one concise summary.

        The prompt is tuned for beamline work: it must preserve concrete
        values, not prose — paths, numbers, units, calibrant/sample names,
        tool successes/failures, and open tasks.
        """
        convo_text = "\n".join(
            f"{m.get('role','?')}: {m.get('content','')}" for m in messages
        )
        sys_msg = {
            "role": "system",
            "content": (
                "You compact a synchrotron beamline assistant's conversation "
                "into a dense factual summary so the assistant can continue "
                "without re-reading older turns. PRESERVE every concrete fact "
                "needed to keep working: file and directory paths, numeric "
                "parameters with units (beam energy keV, wavelength Å, detector "
                "distance, beam center, lattice parameters, tilts), calibrant "
                "and sample names, which tools were run and whether they "
                "succeeded or failed (with the error), decisions made, and any "
                "open/next tasks. Drop pleasantries and restating of the "
                "obvious. Merge the PRIOR SUMMARY with the NEW MESSAGES into a "
                "single updated summary. Output plain text, no markdown "
                "headers, at most ~250 words."
            ),
        }
        user_msg = {
            "role": "user",
            "content": (
                (f"PRIOR SUMMARY:\n{self.running_summary}\n\n"
                 if self.running_summary else "")
                + f"NEW MESSAGES TO FOLD IN:\n{convo_text}\n\n"
                "Produce the updated summary."
            ),
        }
        try:
            resp = await provider.chat([sys_msg, user_msg], temperature=0.2)
            return self.runner._strip_tool_calls_from_text(
                resp.content or ""
            ).strip()
        except Exception as e:
            print(f"[compaction] summarization failed: {e}", file=sys.stderr)
            return ""

    def _score_route(self, query: str) -> tuple:
        """Return (best_domain, scores_dict) — pure scoring, no fallback."""
        q = query.lower()
        scores = {
            domain: sum(1 for kw in keywords if kw in q)
            for domain, keywords in self._KEYWORDS.items()
        }
        return max(scores, key=scores.get), scores

    # Pattern that recognises "retry with context" follow-ups — queries that
    # are giving the agent a file path, folder, or file to work with, typically
    # after a prior tool call failed (e.g., "the calibration was done in
    # /home/…/test_cali").  When the prior turn had a tool error AND this
    # pattern matches, we retain _last_agent rather than re-routing.
    _PATH_CONTEXT_RE = re.compile(
        r'(?:'
        r'/[^\s]+'                     # absolute path  /home/…
        r'|[a-zA-Z0-9_\-]+/[^\s]+'    # relative path  test_cali/…
        r'|\bthis folder\b'
        r'|\bthis directory\b'
        r'|\bthis file\b'
        r'|\buse this\b'
        r'|\bhere is\b'
        r'|\bwas done in\b'
        r'|\bfound in\b'
        r')',
        re.I,
    )

    def _route(self, query: str) -> APEXAAgent:
        best, scores = self._score_route(query)

        if scores[best] > 0:
            # Stateful-routing bias: if the prior turn ended in a tool error
            # AND the current query looks like "retry with this context" (it
            # contains a path or folder reference), keep the prior agent rather
            # than re-routing on keywords alone.  This prevents the failure
            # mode where "the calibration was done in /home/…/test_cali" routes
            # to CalibrationAgent (matching "calibration") when the user's
            # intent was actually to supply context for a pending Visualization
            # or Analysis retry.
            if (self._last_turn_had_tool_error
                    and self._last_agent is not None
                    and self._PATH_CONTEXT_RE.search(query)
                    and scores[best] <= 2):          # weak keyword signal only
                return self._last_agent

            # Break ties: analysis wins by default (most general agent, handles
            # post-calibration workflows). Exception: a conceptual question stem
            # at the START of the query routes to knowledge so the KB tool fires.
            top_score = scores[best]
            tied = [d for d, s in scores.items() if s == top_score]
            if len(tied) > 1 and "analysis" in tied:
                q_lstrip = query.lower().lstrip()
                is_conceptual_question = any(
                    q_lstrip.startswith(stem) for stem in (
                        "what is", "what's", "whats", "what are", "what does",
                        "explain", "describe", "define", "tell me about",
                        "how does", "how do", "how is",
                    )
                )
                best = "knowledge" if is_conceptual_question and "knowledge" in tied else "analysis"
            return self._ROUTES[best]      # strong keyword match → switch agent

        # No keywords matched — stay with current agent if we have one
        # This handles follow-ups like "yes", "ok", "fetch one for Ceria"
        if self._last_agent is not None:
            return self._last_agent

        return ANALYSIS_AGENT              # first query, no context → default

    async def _llm_disambiguate(self, query: str, candidates: List[str],
                                provider: ArgoProvider) -> Optional[str]:
        """Single cheap LLM call to pick a domain when keywords are ambiguous.

        Returns one of the candidate domain names, or None if the model's
        reply doesn't unambiguously match exactly one.
        """
        opts = ", ".join(candidates)
        prompt = (
            "You route synchrotron-beamline assistant queries to one specialist agent. "
            f"Pick exactly ONE domain from: {opts}. "
            "Respond with the single domain word, nothing else.\n\n"
            f"Query: {query}\n\nDomain:"
        )
        try:
            resp = await provider.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
        except Exception:
            return None
        text = (resp.content or "").strip().lower()
        matched = [c for c in candidates if c in text]
        return matched[0] if len(matched) == 1 else None

    async def _route_with_fallback(self, query: str,
                                   provider: ArgoProvider) -> APEXAAgent:
        """Run keyword routing first; only fall back to an LLM call when the
        keyword scoring is genuinely ambiguous (multi-way tie at top score 1).
        Avoids paying the extra round-trip on the common, confidently-routed
        case while still recovering from edge cases the keyword set misses.
        """
        agent = self._route(query)
        best, scores = self._score_route(query)
        if scores[best] <= 1:
            tied = [d for d, s in scores.items() if s == scores[best] and s > 0]
            if len(tied) >= 2:
                pick = await self._llm_disambiguate(query, tied, provider)
                if pick and pick in self._ROUTES:
                    return self._ROUTES[pick]
        return agent

    def _match_fast_path(self, query: str) -> Optional[tuple]:
        """Return (tool_name, args) if the query is a deterministic command."""
        for pattern, tool, build_args in self._FAST_PATHS:
            m = pattern.match(query)
            if m:
                return tool, build_args(m)
        return None

    async def _run_fast_path(self, query: str, tool_name: str,
                             args: Dict, on_tool_result: OnToolResultFn = None,
                             use_history: bool = True) -> str:
        """Execute a fast-path tool directly — no LLM call, no agent loop."""
        log_entry = self.logger.start(query, model="fast_path")
        log_entry.set_agent("FastPath")

        print(f"  \033[36m▸\033[0m \033[1m{tool_name}\033[0m \033[2m(fast-path)\033[0m")
        t0 = time.monotonic()
        result = await self._execute(tool_name, args)
        dur = int((time.monotonic() - t0) * 1000)
        ok = "error" not in result.lower()[:100]
        log_entry.add_tool_call(tool_name, args, result, ok, dur)

        if on_tool_result:
            try:
                await on_tool_result(tool_name, args, result)
            except Exception:
                pass

        if tool_name == "list_directory":
            try:
                r = json.loads(result)
                rendered = _compact_listing(r)
                print(f"\n{rendered}\n")
                summary = rendered
            except (json.JSONDecodeError, KeyError):
                summary = result
        else:
            summary = result

        log_entry.finish(summary, iterations=1, looped=False)
        self.logger.save(log_entry)

        if use_history:
            self.conversation_history.extend([
                {"role": "user",      "content": query},
                {"role": "assistant", "content": summary},
            ])
            if len(self.conversation_history) > 12:
                self.conversation_history = self.conversation_history[-12:]

        return summary

    # Patterns that mean "explain / recap / summarize what you JUST did".
    # When matched on a query and we have at least one prior assistant turn
    # in history, route to _explain_prior_turn() — a single no-tools LLM
    # call — instead of re-executing the work in a specialist agent. Without
    # this, queries like "how did you calculate that?" hit the Analysis
    # agent's keyword set on "calculate" and the entire tool chain re-fires.
    # "Explain what you just did" — recap prior computation.
    _EXPLAIN_PRIOR_PATTERNS = [
        re.compile(r"^\s*(what\s+(was|is|did\s+you\s+get|are\s+the)\s+(the\s+)?(outcome|result|answer|finding|number|value|conclusion|summary))", re.I),
        re.compile(r"^\s*(how\s+did\s+you|why\s+did\s+you|how\s+was\s+(it|that)\s+(calc|comput|deriv|obtain))", re.I),
        re.compile(r"^\s*(explain\s+(that|how|why|what\s+you\s+(did|just)))", re.I),
        re.compile(r"^\s*(walk\s+me\s+through|talk\s+me\s+through)", re.I),
        re.compile(r"^\s*(summari[sz]e|recap|tl;?dr)\s+(that|this|the\s+(result|output|answer|previous))", re.I),
        re.compile(r"^\s*(show\s+me\s+(the\s+)?(answer|result|outcome|summary)\s*(again)?)\s*\??\s*$", re.I),
    ]

    # "What should I do next?" — recommend next step from context.
    # These need a DIFFERENT handler than explain-prior: the system prompt
    # must say "recommend from what you see" not "explain what you did",
    # otherwise the model correctly says "I haven't done any analysis."
    _RECOMMEND_NEXT_PATTERNS = [
        re.compile(r"^\s*(how\s+(should|do|can)\s+(i|we|you)\s+(proceed|continue|start|begin|go\s+(?:from\s+here|ahead|next)))", re.I),
        re.compile(r"^\s*(what\s+(should|do)\s+(i|we)\s+(do|try|run|use|pick|choose|start\s+with))", re.I),
        re.compile(r"^\s*(what(?:'s|\s+is|\s+are)\s+(the\s+)?(next\s+step|best\s+(approach|way|option|start)|recommend))", re.I),
        re.compile(r"^\s*(where\s+(do|should)\s+(i|we)\s+start)", re.I),
        re.compile(r"^\s*ok[,.]?\s*(so\s+)?(how|what|where)\b", re.I),
        re.compile(r"^\s*(yes[,.]?\s*)?(proceed|go\s+ahead|continue|do\s+it|run\s+it)\s*\??\s*$", re.I),
    ]

    def _is_explain_prior(self, query: str) -> bool:
        return any(p.search(query) for p in self._EXPLAIN_PRIOR_PATTERNS)

    def _is_recommend_next(self, query: str) -> bool:
        return any(p.search(query) for p in self._RECOMMEND_NEXT_PATTERNS)

    async def _recommend_from_context(self, query: str, provider: ArgoProvider,
                                      use_history: bool) -> str:
        """Answer "how should I proceed?" from conversation context.

        Different from _explain_prior_turn: the system prompt tells the model
        to recommend the NEXT concrete action based on what it has already
        observed (directory listing, calibration status, etc.) — not to recap
        what it computed. This prevents the "I haven't done any analysis" failure.
        """
        history = self._select_history_for_explain()
        if not history:
            return ""

        sys_msg = {
            "role": "system",
            "content": (
                "You are APEXA, an expert beamline assistant. The user is asking "
                "what to do next. Answer from the CONVERSATION CONTEXT below — "
                "the directory listing, file names, and any prior results are your "
                "evidence. Do NOT call any tools in this response.\n\n"
                "Your answer must:\n"
                "1. State the SPECIFIC recommended next action (e.g., 'calibrate "
                "using Ceria att3 because...' not 'you could calibrate')\n"
                "2. Name the exact file(s) and tool you recommend\n"
                "3. Give the reason in ONE sentence (which att level, why that calibrant)\n"
                "4. End with: 'Type go to proceed' — do NOT ask 'Would you like me to?'\n\n"
                "If context is insufficient to make a specific recommendation, ask "
                "ONE clarifying question only."
            ),
        }
        messages = [sys_msg] + history + [{"role": "user", "content": query}]
        try:
            resp = await provider.chat(messages, temperature=0.3)
            text = self.runner._strip_tool_calls_from_text(resp.content or "").strip()
        except Exception as e:
            return f"(recommendation failed: {e})"
        if not text:
            return ""
        if use_history:
            self.conversation_history.extend([
                {"role": "user",      "content": query},
                {"role": "assistant", "content": text},
            ])
            if len(self.conversation_history) > 12:
                self.conversation_history = self.conversation_history[-12:]
        return text

    async def _explain_prior_turn(self, query: str, provider: ArgoProvider,
                                  use_history: bool) -> str:
        """Answer an explanation/recap follow-up from conversation history."""
        history = self._select_history_for_explain()
        if not history:
            return ""

        sys_msg = {
            "role": "system",
            "content": (
                "You are APEXA. The user is asking you to EXPLAIN or RECAP what "
                "you just did in the previous turn. You MUST answer ONLY from "
                "the conversation history below. Do NOT call any tools. Do NOT "
                "request new information. Do NOT re-execute the prior task.\n\n"
                "If the user asks how a value was calculated, describe the tool(s) "
                "you used and the inputs from the prior turn. If they ask for the "
                "outcome, restate the result concisely. Use markdown: **bold** key "
                "values; bullets for lists; ≤8 lines unless detail is requested."
            ),
        }
        messages = [sys_msg] + history + [{"role": "user", "content": query}]
        try:
            resp = await provider.chat(messages, temperature=0.2)
            text = (resp.content or "").strip()
        except Exception as e:
            return f"(could not generate explanation: {e})"

        # Strip any TOOL_CALL: blocks that leaked through despite the
        # no-tools instruction — we are NOT going to execute them here.
        text = self.runner._strip_tool_calls_from_text(text).strip()
        if not text:
            return "(no prior result to explain)"

        if use_history:
            self.conversation_history.extend([
                {"role": "user",      "content": query},
                {"role": "assistant", "content": text},
            ])
            if len(self.conversation_history) > 12:
                self.conversation_history = self.conversation_history[-12:]
        return text

    def _select_history_for_explain(self) -> List[Dict]:
        """Return the most recent 6 messages (≈3 exchanges) for context.
        Empty list if no usable history."""
        if not self.conversation_history:
            return []
        return list(self.conversation_history[-6:])

    async def process(self, query: str, provider: ArgoProvider,
                      use_history: bool = True,
                      on_tool_result: OnToolResultFn = None) -> str:
        # Recommendation short-circuit: "how should I proceed?", "what next?",
        # "yes, proceed", etc. Gives a concrete next-step recommendation from
        # context WITHOUT tool calls. Separate from explain-prior because the
        # system prompt says "recommend from what you see" not "recap what you
        # did" — prevents "I haven't done any analysis" response.
        if self._is_recommend_next(query) and self.conversation_history:
            rec = await self._recommend_from_context(
                query, provider, use_history=use_history,
            )
            if rec:
                return rec

        # Explanation short-circuit: "what was the outcome?", "how did you
        # calculate that?", "explain that", "recap", etc. Answers from
        # conversation history without re-running any tools. Skipped if
        # we have no prior turn to explain.
        if self._is_explain_prior(query) and self.conversation_history:
            explained = await self._explain_prior_turn(
                query, provider, use_history=use_history,
            )
            if explained:
                return explained
            # else fall through to normal routing

        # Fast path: deterministic NL commands skip the LLM entirely
        fast = self._match_fast_path(query)
        if fast:
            tool_name, args = fast
            return await self._run_fast_path(
                query, tool_name, args,
                on_tool_result=on_tool_result, use_history=use_history,
            )

        agent   = await self._route_with_fallback(query, provider)
        self._last_agent = agent
        history = self.conversation_history if use_history else None

        log_entry = self.logger.start(query, model=provider.model)
        log_entry.set_agent(agent.name)

        result = await self.runner.run(
            agent, query, provider, self.all_tools, history,
            log_entry=log_entry,
            on_tool_result=on_tool_result,
            history_summary=self.running_summary if use_history else "",
        )

        # Track whether this turn had any tool errors — used by _route() on
        # the NEXT query to decide whether to bias toward the same agent when
        # the user's follow-up looks like "retry with this context".
        self._last_turn_had_tool_error = any(
            not tc.success for tc in log_entry.tool_calls
        ) if log_entry.tool_calls else False

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
            # Summarize-older + keep-recent: fold overflow into running_summary
            # rather than dropping it (replaces the old hard 12-msg truncation).
            await self._compact_history(provider)

        if self.context:
            self.context.add_analysis(agent.name, result)

        return result
