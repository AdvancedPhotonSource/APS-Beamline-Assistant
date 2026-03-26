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
import re
import httpx
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Awaitable

# ── Constants ───────────────────────────────────────────────────────────────

PROD_URL = "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/"
DEV_URL  = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"

DEV_ONLY_MODELS = {
    "gpt5", "gpt5mini", "gpt5nano",
    "gemini25pro", "gemini25flash",
    "claudeopus41", "claudeopus4", "claudesonnet45",
    "claudesonnet4", "claudesonnet37",
    "gpto1", "gpto3mini", "gpto4mini",
    "gpt41", "gpt41mini", "gpt41nano",
}

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

        # Max tokens per model family
        if self.model.startswith("claude"):
            payload["max_tokens"] = 21000
            if self.model != "claudesonnet45":   # sonnet45 rejects top_p
                payload["top_p"] = 0.9
        elif self.model.startswith("gpt4o"):
            payload["max_tokens"] = 16000
            payload["top_p"]      = 0.9
        else:
            payload["max_tokens"] = 4000
            payload["top_p"]      = 0.9

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
        response = await self._client.post(
            self.url, json=payload,
            headers={"Content-Type": "application/json"},
        )
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

Responsibilities:
- Auto-calibrate detectors using standard powders (CeO2, LaB6, Si, Al2O3)
- Validate beamline parameters (wavelength, detector distance, beam center)
- Perform X-ray geometry calculations (Bragg's law, d-spacing, energy↔wavelength)

Always use midas_auto_calibrate — never attempt manual calibration.
If the user provides energy in keV, first call xray_calculate to convert to wavelength.

After calibration report: refined BC (beam center x/y), Lsd (detector distance),
tilts (tx/ty/tz), and whether convergence was achieved (stopping strain reached).""",
)

ANALYSIS_AGENT = APEXAAgent(
    name        = "AnalysisAgent",
    temperature = 0.5,
    tool_names  = [
        # Integration
        "midas_integrate_2d_to_1d",
        "midas_batch_integrate",
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

When the user asks to integrate, reconstruct, track grains, or run any workflow,
you MUST call the appropriate tool. Never describe steps in text — execute them.

Capabilities (use the matching tool for each):
- 2D → 1D integration: midas_integrate_2d_to_1d or midas_batch_integrate
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
  2. midas_integrate_2d_to_1d for 2D → 1D
  3. run_ff_hedm_full_workflow or run_nf_hedm_reconstruction
  4. Post-process: match_grains, run_ff_grain_tracking, overlay_ff_nf_results, extract_grain_centroids
  5. Export: convert_nf_to_dream3d

Always report: grains found, convergence quality, output file paths.""",
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

You control motors via EPICS Channel Access using the motor MCP tools.
The IOC prefix (e.g. "20idMotSim") and motor name (e.g. "m1") identify each motor.

Workflow rules:
1. Before any move, call get_motor_status to know current position, limits, and state.
2. Use move_motor_absolute for absolute targets; move_motor_relative for +/- steps.
3. Always wait=True unless the user explicitly asks for a non-blocking move.
4. For small exploratory moves, use tweak_motor or jog_motor.
5. If a move is rejected due to limits, call get_motor_limits to show the user the range.
6. STOP_MOTOR is always safe to call — do it immediately if the user says "stop".

Safety rules (NEVER bypass):
- Never move a motor that is at a limit switch (HLS=1 or LLS=1).
- Never set STOP=0 (arming the stop — not your responsibility).
- Confirm large moves (>50% travel range) with the user before setting confirm_large_move=True.
- Never home a motor without explicit user instruction.

Common IOC prefix: "20idMotSim" (simulation IOC for testing at 20-ID).

After each move report: start position, target, final RBV, and units.""",
)

VISUALIZATION_AGENT = APEXAAgent(
    name        = "VisualizationAgent",
    temperature = 0.3,
    tool_names  = [
        "list_directory",
        "read_file",
        "get_file_info",
        "run_command",
    ],
    instructions = """You are a visualization specialist for HEDM diffraction data at APS.

Match the user's data files to the correct MIDAS viewer script:

| File available | Script to use | Location |
|---|---|---|
| Raw 2D image (.tif/.ge/.h5/.zip) | ff_asym_qt.py | ~/Git/MIDAS/gui/ |
| *_lineout.xy | plot_lineout_results.py or plot_lineout_comparison.py | ~/Git/MIDAS/gui/viewers/ |
| *_lineout.bin (live GPU) | live_viewer.py | ~/Git/MIDAS/gui/viewers/ |
| *_caked.hdf.zarr.zip | plot_integrator_peaks.py | ~/Git/MIDAS/gui/viewers/ |
| *_caked_peaks.h5 | plot_caked_peaks.py | ~/Git/MIDAS/gui/viewers/ |
| *_corr.csv (calibration) | plot_calibrant_results.py | ~/Git/MIDAS/gui/viewers/ |
| Grains.csv + SpotMatrix.csv + .zarr | interactiveFFplotting.py (Dash browser) | ~/Git/MIDAS/gui/viewers/ |
| .mic / .map (NF microstructure) | nf_qt.py | ~/Git/MIDAS/gui/ |

Workflow:
1. Use list_directory to find output files in the result folder
2. Match file type to viewer from the table above
3. Build the command, then IMMEDIATELY call run_command to execute it
4. For ff_asym_qt.py and nf_qt.py: cd to data directory first, then launch with &
5. For all others: pass explicit file paths as arguments
6. Always pass --paramFN or --params when refined_MIDAS_params*.txt is available (enables 2θ/Q axes)

⚠️ CRITICAL: ALWAYS call run_command to launch the viewer — NEVER just print the command.

Critical flags:
- live_viewer.py: --nRBins (capital R, capital B — case sensitive), NOT --n-rbins
- viz_caking.py: DO NOT USE — requires calcMiso which is not in midas_env; use plot_integrator_peaks.py instead
- interactiveFFplotting.py: requires BOTH -resultFolder AND -dataFileName (.zarr)
- plot_caked_peaks.py: run fit_caked_peaks.py first if _caked_peaks.h5 doesn't exist
- ff_asym_qt.py: auto-detects files from CWD — just cd to data dir and launch

Always use midas_env Python:
  /Users/b324240/miniconda3/envs/midas_env/bin/python ~/Git/MIDAS/gui/viewers/<script>.py <args>

After launching, report the exact command so the user can re-run it manually if needed.""",
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
✅ CORRECT:
TOOL_CALL: list_directory
ARGUMENTS: {"path": "."}

User: "Calibrate the CeO2 image in test5"
✅ CORRECT:
TOOL_CALL: list_directory
ARGUMENTS: {"path": "test5"}

User: "Convert 61.332 keV to wavelength"
✅ CORRECT:
TOOL_CALL: xray_calculate
ARGUMENTS: {"calculation_type": "energy_to_wavelength", "energy_kev": 61.332}

User: "Show me the lineout for CeO2 integration in test1"
✅ CORRECT:
TOOL_CALL: list_directory
ARGUMENTS: {"path": "test1/integration"}
[then after finding *_lineout.xy:]
TOOL_CALL: run_command
ARGUMENTS: {"command": "/Users/b324240/miniconda3/envs/midas_env/bin/python /Users/b324240/Git/MIDAS/gui/viewers/plot_lineout_results.py /full/path/to/lineout.xy --paramFN /full/path/to/refined_MIDAS_params_CeO2.txt"}

User: "Plot the caked output"
✅ CORRECT (after finding *_caked.hdf.zarr.zip):
TOOL_CALL: run_command
ARGUMENTS: {"command": "/Users/b324240/miniconda3/envs/midas_env/bin/python /Users/b324240/Git/MIDAS/gui/viewers/viz_caking.py /full/path/to/file.caked.hdf.zarr.zip"}

❌ WRONG — NEVER do these:
- NEVER calculate d = a/√(h²+k²+l²) yourself — call xray_calculate
- NEVER say "you can use ls" or "here's how to do it in Python"
- NEVER say "Let me proceed" without actually calling a tool
- NEVER describe what you WOULD do — DO IT with TOOL_CALL
- NEVER read_file to show plot data — launch the viewer with run_command
- NEVER run bare commands like "plot radial ..." — always use the full Python path

RULES:
1. For ANY X-ray calculation → TOOL_CALL: xray_calculate
2. For file listing → TOOL_CALL: list_directory
3. For reading files → TOOL_CALL: read_file
4. For calibration → TOOL_CALL: midas_auto_calibrate
5. For integration → TOOL_CALL: midas_integrate_2d_to_1d
6. For HEDM workflows → TOOL_CALL: the appropriate workflow tool
7. For visualization/plotting → TOOL_CALL: run_command with the full Python viewer command

Only generate text WITHOUT a TOOL_CALL when:
- User says hello/greeting
- User asks a conceptual question ("what is HEDM?", "explain calibration")
- User asks what you can do

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
                  max_iterations: int = 10) -> str:

        tools = self._filter_tools(agent.tool_names, all_tools)

        # Build system message: strong preamble + agent-specific instructions
        system_content = _TOOL_PREAMBLE + agent.instructions

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
            messages.extend(history[-10:])   # last 5 exchanges
        messages.append({"role": "user", "content": query})

        for _ in range(max_iterations):
            response = await provider.chat(messages, agent.temperature)

            # ── Mode 1: Native API tool_calls ──
            if response.tool_calls:
                messages.append(self._assistant_message(response, provider.model))
                for tc in response.tool_calls:
                    print(f"  → {tc.name}")
                    result = await self._execute(tc.name, tc.arguments)
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
                    print(f"  → {tc.name}")
                    result = await self._execute(tc.name, tc.arguments)
                    messages.append({
                        "role": "user",
                        "content": f"[Tool Result for {tc.name}]\n{result}",
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
        },
        "knowledge": {
            "explain", "what is", "what are", "how does", "how do",
            "tell me", "describe", "typical", "literature", "paper",
            "best practice", "recommend", "suggest", "look up",
            "material propert", "search", "parameter range",
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
        },
    }

    def __init__(self, execute_tool_fn: ExecuteToolFn,
                 all_tools: List[Dict], context=None):
        self.runner    = AgentRunner(execute_tool_fn)
        self.all_tools = all_tools
        self.context   = context
        self.conversation_history: List[Dict] = []

    def clear_history(self):
        self.conversation_history = []

    def _route(self, query: str) -> APEXAAgent:
        q = query.lower()
        scores = {
            domain: sum(1 for kw in keywords if kw in q)
            for domain, keywords in self._KEYWORDS.items()
        }
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return ANALYSIS_AGENT   # sensible default for HEDM context
        return self._ROUTES[best]

    async def process(self, query: str, provider: ArgoProvider,
                      use_history: bool = True) -> str:
        agent   = self._route(query)
        history = self.conversation_history if use_history else None

        result = await self.runner.run(
            agent, query, provider, self.all_tools, history
        )

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
