import asyncio
import json
import os
import re
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import httpx
from dotenv import load_dotenv

load_dotenv()

class ArgoMCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.sessions = {}
        self.exit_stack = AsyncExitStack()
        
        self.argo_chat_url = "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/"
        self.argo_embed_url = "https://apps.inside.anl.gov/argoapi/api/v1/resource/embed/"
        
        self.anl_username = os.getenv("ANL_USERNAME")
        self.selected_model = os.getenv("ARGO_MODEL", "gpt4o")
        
        self.http_client = httpx.AsyncClient(timeout=120.0)
        
        if not self.anl_username:
            raise ValueError("ANL_USERNAME must be set in environment (.env file)")

        self.available_models = {
            "OpenAI": {
                "gpt35": "GPT-3.5 Turbo (4K tokens)",
                "gpt4": "GPT-4 (8K tokens)", 
                "gpt4turbo": "GPT-4 Turbo (128K input)",
                "gpt4o": "GPT-4o (128K input, 16K output)",
                "gpt5": "GPT-5 (272K input, dev only)"
            },
            "Google": {
                "gemini25pro": "Gemini 2.5 Pro (1M tokens, dev only)",
                "gemini25flash": "Gemini 2.5 Flash (1M tokens, dev only)"
            },
            "Anthropic": {
                "claudeopus4": "Claude Opus 4 (200K input, dev only)",
                "claudesonnet4": "Claude Sonnet 4 (200K input, dev only)"
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
                print(f"Connected to {name} server with tools: {[tool.name for tool in tools]}")
                
            except Exception as e:
                print(f"Failed to connect to {name} server: {e}")
        
        if "midas" in self.sessions:
            self.session = self.sessions["midas"]

    def _prepare_argo_payload(self, messages: List[Dict[str, str]], model: str, tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "user": self.anl_username,
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        if model.startswith("gpt5"):
            payload["max_completion_tokens"] = 16000
        elif model.startswith("claude"):
            payload["max_tokens"] = 21000
        else:
            payload["max_tokens"] = 1000

        if tools:
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
            
            if 'response' in result:
                return {
                    'choices': [{
                        'message': {
                            'role': 'assistant',
                            'content': result['response']
                        }
                    }]
                }
            elif 'choices' in result:
                return result
            else:
                return {
                    'choices': [{
                        'message': {
                            'role': 'assistant', 
                            'content': str(result)
                        }
                    }]
                }
            
        except Exception as e:
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
                print(f"Error getting tools from {server_name}: {e}")
        
        return all_tools

    async def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
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
            return str(result.content[0].text if result.content else "No result")
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def _extract_peak_positions(self, query: str) -> List[float]:
        patterns = [
            r'\[([\d.,\s]+)\]',
            r'(?:peaks?\s+at\s+|positions?\s+)([\d.,\s]+)(?:\s+degrees?)?',
            r'((?:\d+\.?\d*[,\s]+)+\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                numbers_str = match.group(1)
                numbers = re.findall(r'\d+\.?\d*', numbers_str)
                try:
                    return [float(num) for num in numbers if float(num) > 0]
                except ValueError:
                    continue
        
        return []

    def _format_phase_analysis_results(self, json_result: str) -> str:
        """Format JSON phase analysis results into readable text"""
        try:
            data = json.loads(json_result)
            
            if "identified_phases" not in data:
                return json_result  # Return original if parsing fails
            
            output = []
            output.append("PHASE IDENTIFICATION RESULTS")
            output.append("=" * 50)
            
            # Analysis parameters
            params = data.get("analysis_parameters", {})
            output.append(f"Input peaks analyzed: {params.get('input_peaks', 'N/A')}")
            output.append(f"Peak positions: {params.get('peak_positions', [])}")
            output.append(f"Temperature: {params.get('temperature_celsius', 25)}°C")
            output.append("")
            
            # Identified phases
            phases = data.get("identified_phases", [])
            if phases:
                output.append("IDENTIFIED PHASES:")
                output.append("-" * 20)
                
                for i, phase in enumerate(phases, 1):
                    output.append(f"{i}. {phase.get('phase_name', 'Unknown')} ({phase.get('chemical_formula', 'N/A')})")
                    output.append(f"   Crystal System: {phase.get('crystal_system', 'N/A').title()}")
                    output.append(f"   Space Group: {phase.get('space_group', 'N/A')}")
                    
                    if phase.get('phase_fraction'):
                        fraction_pct = phase['phase_fraction'] * 100
                        output.append(f"   Phase Fraction: {fraction_pct:.1f}%")
                    
                    confidence = phase.get('confidence_score', 0)
                    output.append(f"   Confidence: {confidence:.1%}")
                    
                    # Matched peaks
                    matched = phase.get('matched_peaks', [])
                    if matched:
                        output.append("   Matched Peaks:")
                        for peak in matched:
                            obs = peak.get('observed', 0)
                            calc = peak.get('calculated', 0) 
                            hkl = peak.get('hkl', 'N/A')
                            intensity = peak.get('relative_intensity', 0)
                            output.append(f"     • {obs:.1f}° → {hkl} (calc: {calc:.2f}°, intensity: {intensity}%)")
                    output.append("")
            else:
                output.append("No phases identified with sufficient confidence.")
                output.append("")
            
            # Unmatched peaks
            unmatched = data.get("unmatched_peaks", [])
            if unmatched:
                output.append("UNMATCHED PEAKS:")
                output.append("-" * 15)
                for peak in unmatched:
                    pos = peak.get('position', 0)
                    assignment = peak.get('possible_assignment', 'Unknown')
                    output.append(f"• {pos:.1f}° - {assignment}")
                output.append("")
            
            # Summary
            summary = data.get("phase_summary", {})
            overall = data.get("overall_analysis", {})
            
            output.append("ANALYSIS SUMMARY:")
            output.append("-" * 17)
            output.append(f"Total phases found: {summary.get('total_phases_found', 0)}")
            output.append(f"Peaks matched: {summary.get('total_matched_peaks', 0)}/{params.get('input_peaks', 0)}")
            output.append(f"Match quality: {overall.get('match_quality', 'N/A').title()}")
            
            recommendation = overall.get('recommended_action', '')
            if recommendation:
                output.append(f"Recommendation: {recommendation}")
            
            return "\n".join(output)
            
        except (json.JSONDecodeError, KeyError) as e:
            return f"Error formatting results: {e}\n\nRaw output:\n{json_result}"

    async def process_diffraction_query(self, query: str, image_path: str = None, experimental_params: Dict[str, Any] = None) -> str:
        if not self.sessions:
            return "Error: Not connected to any MCP servers."
        
        available_tools = await self.get_all_available_tools()

        system_prompt = """You are Beamline Assistant, an expert AI system for synchrotron X-ray diffraction analysis at Argonne National Laboratory. You have access to analysis tools and must use them.

Available Tools:
- midas_identify_crystalline_phases: Match peaks to known crystal phases
- midas_detect_diffraction_rings: Detect rings in 2D patterns
- midas_integrate_2d_to_1d: Convert 2D to 1D patterns
- midas_analyze_diffraction_peaks: Analyze peaks in patterns
- filesystem_read_file: Read files
- executor_run_command: Execute commands

When users mention peak positions, identify the crystalline phases. Provide brief context before analysis."""

        system_prompt += f"\n\nCurrent Model: {self.selected_model} via Argo Gateway"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        if image_path:
            messages[-1]["content"] += f"\n\nImage: {image_path}"
        if experimental_params:
            messages[-1]["content"] += f"\n\nParameters: {json.dumps(experimental_params)}"

        print(f"Processing with {self.selected_model}...")

        try:
            response_data = await self.call_argo_chat_api(messages, available_tools)
            
            if 'choices' not in response_data or not response_data['choices']:
                return f"Unexpected response format: {response_data}"
                
            choice = response_data['choices'][0]
            
            if 'message' in choice:
                message = choice['message']
            else:
                return f"Unexpected choice format: {choice}"
            
            response_text = message.get('content', '').lower()
            
            # Check if AI mentioned phase identification
            if 'identify_crystalline_phases' in response_text or 'phase' in response_text:
                peak_positions = self._extract_peak_positions(query)
                if peak_positions:
                    print("AI intended to use phase identification - executing...")
                    try:
                        result = await self.execute_tool_call(
                            "midas_identify_crystalline_phases", 
                            {"peak_positions": peak_positions}
                        )
                        
                        # Format the results nicely
                        formatted_results = self._format_phase_analysis_results(result)
                        
                        return f"{message['content']}\n\n{formatted_results}"
                    except Exception as e:
                        return f"{message['content']}\n\nError in phase analysis: {e}"
            
            return message['content']
            
        except Exception as e:
            return f"Error processing query: {str(e)}"

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
        print(f"\nBeamline Assistant - AI Diffraction Analysis")
        print("=" * 60)
        print(f"Current AI Model: {self.selected_model}")
        print(f"ANL User: {self.anl_username}")
        print(f"Connected Servers: {list(self.sessions.keys())}")
        print()
        print("Commands: analyze, models, tools, servers, ls, run, help, quit")
        print()
        
        while True:
            try:
                user_input = input("Beamline> ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'models':
                    self.show_available_models()
                elif user_input.lower() == 'servers':
                    print(f"Connected: {list(self.sessions.keys())}")
                elif user_input.startswith('model '):
                    model_name = user_input[6:].strip()
                    if self._is_valid_model(model_name):
                        self.selected_model = model_name
                        print(f"Switched to: {model_name}")
                    else:
                        print(f"Invalid model: {model_name}")
                elif user_input.startswith('run '):
                    command = user_input[4:].strip()
                    if "executor" in self.sessions:
                        result = await self.execute_tool_call("executor_run_command", {"command": command})
                        print(f"\n{result}\n")
                    else:
                        print("Executor server not connected")
                elif user_input:
                    response = await self.process_diffraction_query(user_input)
                    print(f"\n{response}\n")
                    
            except KeyboardInterrupt:
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
    
    client = ArgoMCPClient()
    
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