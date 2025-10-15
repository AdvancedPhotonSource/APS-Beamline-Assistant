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

class APEXAClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.sessions = {}
        self.exit_stack = AsyncExitStack()

        # Smart context manager for session persistence
        self.context = ExperimentContext()

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

        if "_" in tool_name:
            server_name, original_tool_name = tool_name.split("_", 1)
        else:
            server_name = "midas"
            original_tool_name = tool_name

        if server_name not in self.sessions:
            return f"Error: Server '{server_name}' not connected"

        try:
            session = self.sessions[server_name]
            result = await session.call_tool(original_tool_name, arguments)
            result_text = str(result.content[0].text if result.content else "No result")

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

🔧 AVAILABLE TOOLS:
- midas_identify_crystalline_phases
  Args: {"peak_positions": [12.5, 18.2]}
- midas_run_ff_hedm_full_workflow
  Args: {"example_dir": "~/path", "n_cpus": 20}
- midas_detect_diffraction_rings
  Args: {"image_path": "/path/image.tif"}
- midas_integrate_2d_to_1d
  Args: {"image_path": "/path/image.tif", "calibration_file": "calib.txt"}
  Or: {"image_path": "/path/image.tif", "wavelength": 0.22, "detector_distance": 1000, "beam_center_x": 1024, "beam_center_y": 1024}
  With dark subtraction: {"image_path": "/path/image.tif", "calibration_file": "calib.txt", "dark_file": "/path/dark.tif"}
- filesystem_read_file
  Args: {"file_path": "/path/file"}
- filesystem_list_directory
  Args: {"path": "/path/dir"}
- executor_run_command
  Args: {"command": "ls -la"}

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
        print("Commands: analyze, models, tools, servers, ls, run, clear, help, quit")
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
                elif user_input.lower() == 'tools':
                    tools = await self.get_all_available_tools()
                    print(f"\nAvailable tools ({len(tools)}):")
                    for tool in tools:
                        print(f"  - {tool['function']['name']}: {tool['function']['description'][:80]}")
                elif user_input.lower() == 'help':
                    print("""
Beamline Assistant Commands:

  analyze <query>  - Run full diffraction analysis
  models           - Show available AI models
  model <name>     - Switch to different model
  tools            - List all available analysis tools
  servers          - Show connected MCP servers
  ls <path>        - List directory contents
  run <command>    - Execute system command (whitelisted)
  clear            - Clear conversation history
  help             - Show this help
  quit             - Exit assistant

Natural Language Queries:
  - "I have peaks at X, Y, Z degrees. What phases?"
  - "Run FF-HEDM workflow in <directory>"
  - "Analyze quality of <image.tif>"
  - "Integrate pattern from <image>"
  - "Read the Parameters.txt file there" (remembers context)

Use ↑/↓ arrow keys for command history
Use Tab for command completion
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