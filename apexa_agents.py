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
        t0 = time.monotonic()
        response = await self._client.post(
            self.url, json=payload,
            headers={"Content-Type": "application/json"},
        )
        elapsed = time.monotonic() - t0
        if os.environ.get("APEXA_SHOW_TIMING"):
            print(f"  ⏱ {self.model} responded in {elapsed:.1f}s", flush=True)
        if response.status_code != 200:
            print(f"  Argo API error ({response.status_code}): {response.text[:500]}", file=sys.stderr)
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

After calibration report: refined BC, Lsd, tilts, and convergence quality.""",
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

Standard workflow:
  1. list_directory to find data files
  2. midas_integrate_2d_to_1d for 2D → 1D (produces .zarr.zip)
  3. run_gsas_refinement for peak fitting / lattice refinement on .zarr.zip
  4. Or run_live_analysis for combined integration + refinement in one step
  5. run_ff_hedm_full_workflow or run_nf_hedm_reconstruction
  6. Post-process: match_grains, run_ff_grain_tracking, overlay_ff_nf_results, extract_grain_centroids
  7. Export: convert_nf_to_dream3d

GSAS-II refinement workflow:
  1. If no CIF file → call fetch_cif_from_mp to download one (you have this tool)
  2. Read the CIF path from the fetch result
  3. IMMEDIATELY call run_gsas_refinement with data_file=<.zarr.zip path> and cif_files=[<CIF path>]
  4. NEVER use run_command for GSAS-II — always use run_gsas_refinement

CRITICAL: After calling a tool, read the result carefully. Do NOT call list_directory
to verify files you already know about. Use the paths from the tool results directly.

Always report: grains found, convergence quality, Rwp, output file paths.""",
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
    ],
    instructions = """You are an HEDM knowledge expert with access to scientific literature,
experimental logbooks, and crystallographic databases.

When the user asks about materials, parameters, or HEDM methodology, use your tools:
- query_hedm_knowledge for methodology, best practices, literature
- get_material_properties for crystallographic data (lattice params, space groups, d-spacings)
- get_typical_hedm_parameters for recommended parameter ranges
- estimate_parameters_from_image to estimate beam parameters from diffraction images
- list_common_calibrants for calibrant materials
- xray_calculate for ANY calculation (NEVER compute manually)
- fetch_cif_from_mp to download CIF files from Materials Project for any material

When the user asks for a CIF file, call fetch_cif_from_mp IMMEDIATELY with the formula.
The tool downloads the most stable structures and saves .cif files locally.
Report: formula, space group, crystal system, stability, and file path.

Always cite your source: paper title, logbook entry, or database name.
Prefer tool results over your own knowledge — the tools have verified data.""",
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
        "list_motors",
        "home_motor",
    ],
    instructions = """You are a motor control specialist for EPICS-based beamline instruments at APS.

Default IOC prefix is "20idMotSim". Motor names: "m1" through "m8".
The prefix parameter defaults automatically — you do NOT need to specify it.

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

After each move report: target, final RBV, and units.""",
)

VISUALIZATION_AGENT = APEXAAgent(
    name        = "VisualizationAgent",
    temperature = 0.3,
    tool_names  = [
        "list_directory",
        "read_file",
        "get_file_info",
        "run_command",
        "check_environment",
    ],
    instructions = """You are a visualization specialist for HEDM diffraction data at APS.

STEP 1: Call check_environment FIRST to get the MIDAS path and Python interpreter.
  The response contains "midas.path" (e.g. /home/user/opt/MIDAS) and "python_executable".
  Use these to build commands — NEVER hardcode paths.

STEP 2: Match the user's data files to the correct MIDAS viewer script:

| File available | Script | Relative path in MIDAS |
|---|---|---|
| Raw 2D image (.tif/.ge/.h5/.zip) | ff_asym_qt.py | gui/ff_asym_qt.py |
| *_lineout.xy | plot_lineout_results.py | gui/viewers/plot_lineout_results.py |
| *_lineout.xy (with ring overlay) | plot_lineout_comparison.py | gui/viewers/plot_lineout_comparison.py |
| *_lineout.bin (live GPU) | live_viewer.py | gui/viewers/live_viewer.py |
| *_caked.hdf.zarr.zip | plot_integrator_peaks.py | gui/viewers/plot_integrator_peaks.py |
| *_caked_peaks.h5 | plot_caked_peaks.py | gui/viewers/plot_caked_peaks.py |
| *_corr.csv (calibration) | plot_calibrant_results.py | gui/viewers/plot_calibrant_results.py |
| Grains.csv + SpotMatrix.csv + .zarr | interactiveFFplotting.py | gui/viewers/interactiveFFplotting.py |
| .mic / .map (NF microstructure) | nf_qt.py | gui/nf_qt.py |

STEP 3: Build the command using the MIDAS path from check_environment:
  <MIDAS_PYTHON> <MIDAS_PATH>/gui/viewers/<script>.py <args>

  Where MIDAS_PYTHON is found by looking for midas_env conda Python. Check these in order:
  - $CONDA_PREFIX/../midas_env/bin/python
  - ~/miniconda3/envs/midas_env/bin/python
  - ~/anaconda3/envs/midas_env/bin/python
  - python3 (fallback)

  Or use: run_command with "which python3" in the midas_env if unsure.

STEP 4: Call run_command to execute it. NEVER just print the command.

Critical flags:
- live_viewer.py: --nRBins (capital R, capital B — case sensitive), NOT --n-rbins
- viz_caking.py: DO NOT USE — requires calcMiso; use plot_integrator_peaks.py instead
- interactiveFFplotting.py: requires BOTH -resultFolder AND -dataFileName (.zarr)
- plot_caked_peaks.py: run fit_caked_peaks.py first if _caked_peaks.h5 doesn't exist
- ff_asym_qt.py: auto-detects files from CWD — just cd to data dir and launch
- Always pass --paramFN when refined_MIDAS_params*.txt is available (enables 2θ/Q axes)

After launching, report the exact command so the user can re-run it manually.""",
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

User: "Convert 61.332 keV to wavelength"
✅ CORRECT:
TOOL_CALL: xray_calculate
ARGUMENTS: {"calculation_type": "energy_to_wavelength", "energy_kev": 61.332}

User: "Show me the lineout for CeO2 integration in test1"
✅ CORRECT:
TOOL_CALL: check_environment
ARGUMENTS: {}
[get MIDAS path from response, then:]
TOOL_CALL: list_directory
ARGUMENTS: {"path": "test1/integration"}
[then after finding *_lineout.xy, build command with MIDAS path from check_environment:]
TOOL_CALL: run_command
ARGUMENTS: {"command": "<MIDAS_PYTHON> <MIDAS_PATH>/gui/viewers/plot_lineout_results.py /full/path/to/lineout.xy --paramFN /full/path/to/refined_MIDAS_params_CeO2.txt"}

User: "Plot the caked output"
✅ CORRECT (after check_environment + finding *_caked.hdf.zarr.zip):
TOOL_CALL: run_command
ARGUMENTS: {"command": "<MIDAS_PYTHON> <MIDAS_PATH>/gui/viewers/plot_integrator_peaks.py /full/path/to/file.caked.hdf.zarr.zip"}

User: "Plot calibration results in test1"
✅ CORRECT (after check_environment + finding *_corr.csv):
TOOL_CALL: run_command
ARGUMENTS: {"command": "<MIDAS_PYTHON> <MIDAS_PATH>/gui/viewers/plot_calibrant_results.py /full/path/to/file_corr.csv"}

User: "Refine the caked output with GSAS-II using the CeO2 CIF"
✅ CORRECT:
TOOL_CALL: run_gsas_refinement
ARGUMENTS: {"data_file": "/path/to/CeO2_caked.hdf.zarr.zip", "cif_files": ["/path/to/CeO2.cif"]}

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
ARGUMENTS: {"ioc_prefix": "20idMotSim", "motor_name": "m1", "target_position": 25.5}

User: "Where is motor m1?"
✅ CORRECT:
TOOL_CALL: get_motor_position
ARGUMENTS: {"ioc_prefix": "20idMotSim", "motor_name": "m1"}

❌ WRONG — NEVER do these:
- NEVER calculate d = a/√(h²+k²+l²) yourself — call xray_calculate
- NEVER say "you can use ls" or "here's how to do it in Python"
- NEVER say "Let me proceed" or "I can move it" without actually calling a tool
- NEVER describe what you WOULD do — DO IT with TOOL_CALL
- NEVER read_file to show plot data — launch the viewer with run_command
- NEVER run bare commands like "plot radial ..." — always use the full Python path

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
10. For visualization/plotting → TOOL_CALL: run_command with the full Python viewer command
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

    # ── Main loop ────────────────────────────────────────────────────────────

    async def run(self, agent: APEXAAgent, query: str,
                  provider: ArgoProvider, all_tools: List[Dict],
                  history: Optional[List[Dict]] = None,
                  max_iterations: int = 10,
                  log_entry: Optional[InteractionEntry] = None) -> str:

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
        repeat_count   = 0

        for _ in range(max_iterations):
            response = await provider.chat(messages, agent.temperature)

            # ── Mode 1: Native API tool_calls ──
            if response.tool_calls:
                messages.append(self._assistant_message(response, provider.model))
                for tc in response.tool_calls:
                    print(f"  → {tc.name}")
                    t0 = time.monotonic()
                    result = await self._execute(tc.name, tc.arguments)
                    dur = int((time.monotonic() - t0) * 1000)
                    ok = "error" not in result.lower()[:100]
                    if log_entry:
                        log_entry.add_tool_call(tc.name, tc.arguments, result, ok, dur)
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

                for tc in text_calls:
                    # Detect repeated identical tool calls (loop bug)
                    if tc.name == last_tool_name:
                        repeat_count += 1
                    else:
                        last_tool_name = tc.name
                        repeat_count = 0

                    if repeat_count >= 2:
                        # Model is looping — force it to summarise
                        messages.append({
                            "role": "user",
                            "content": (
                                f"You already called {tc.name} and got the result above. "
                                "Do NOT call it again. Summarise the result for the user now."
                            ),
                        })
                        break

                    print(f"  → {tc.name}")
                    t0 = time.monotonic()
                    result = await self._execute(tc.name, tc.arguments)
                    dur = int((time.monotonic() - t0) * 1000)
                    ok = "error" not in result.lower()[:100]
                    if log_entry:
                        log_entry.add_tool_call(tc.name, tc.arguments, result, ok, dur)
                    # Truncate very long tool results to prevent context overflow
                    if len(result) > 3000:
                        result = result[:3000] + "\n... [truncated]"
                    # Build a context-aware follow-up prompt
                    if tc.name == "list_directory":
                        followup = (
                            "You have the file listing above. "
                            "If you have all the information needed, call the analysis/calibration/refinement tool now. "
                            "Do NOT call list_directory again."
                        )
                    elif tc.name == "fetch_cif_from_mp":
                        followup = (
                            "CIF file downloaded. The file path is in the result above. "
                            "Now call run_gsas_refinement with the CIF path and the .zarr.zip data file. "
                            "Do NOT call list_directory or fetch_cif_from_mp again."
                        )
                    else:
                        followup = (
                            "Proceed with the user's request using the result above. "
                            "If the task is complete, summarize the results for the user. "
                            "Do NOT repeat the same tool call."
                        )
                    messages.append({
                        "role": "user",
                        "content": f"[Tool Result for {tc.name}]\n{result}\n\n{followup}",
                    })
                continue

            # ── No tool calls at all — return final text ──
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
        },
        "knowledge": {
            "explain", "what is", "what are", "how does", "how do",
            "tell me", "describe", "typical", "literature", "paper",
            "best practice", "recommend", "suggest", "look up",
            "material propert", "search", "parameter range",
            "cif file", "cif", "fetch cif", "download cif", "materials project",
            "crystal structure",
        },
        "visualization": {
            "plot", "visualiz", "view", "show", "display", "see",
            "lineout", "caked", "heatmap", "chart", "graph",
            "live viewer", "overlay", "pattern", "diffraction image",
            "peak plot", "grain plot", "3d grain", "spots",
            "ring", "fit result", "caking", "zarr", "lineout.xy",
        },
        "motor": {
            "motor", "move", "position", "caget", "caput", "epics",
            "ioc", "rbv", "readback", "jog", "tweak", "home motor",
            "stop motor", "velocity", "speed", "limit switch",
            "soft limit", "hls", "lls", "dmov", "pv", "channel access",
            "20idmotsim", "motorsim",
            # Common motor names so "where is m1" routes here
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
            return self._ROUTES[best]      # strong keyword match → switch agent

        # No keywords matched — stay with current agent if we have one
        # This handles follow-ups like "yes", "ok", "fetch one for Ceria"
        if self._last_agent is not None:
            return self._last_agent

        return ANALYSIS_AGENT              # first query, no context → default

    async def process(self, query: str, provider: ArgoProvider,
                      use_history: bool = True) -> str:
        agent   = self._route(query)
        self._last_agent = agent
        history = self.conversation_history if use_history else None

        # Start interaction log entry
        log_entry = self.logger.start(query, model=provider.model)
        log_entry.set_agent(agent.name)

        result = await self.runner.run(
            agent, query, provider, self.all_tools, history,
            log_entry=log_entry,
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
