#!/usr/bin/env python3
"""
Beamline Assistant - Fixed for macOS libedit readline
"""
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
        
        self.argo_urls = {
            "prod": {"chat": "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/"},
            "dev": {"chat": "https://apps.inside.anl.gov/argoapi/dev/api/v1/resource/chat/"}
        }
        
        self.model_env_mapping = {
            "gpt35": "prod", "gpt4": "prod", "gpt4turbo": "prod", "gpt4o": "prod",
            "gpt5": "dev", "gemini25pro": "dev", "gemini25flash": "dev", 
            "claudeopus4": "dev", "claudesonnet4": "dev"
        }
        
        self.anl_username = os.getenv("ANL_USERNAME")
        self.selected_model = os.getenv("ARGO_MODEL", "gpt4o")
        self.http_client = httpx.AsyncClient(timeout=120.0)
        
        if not self.anl_username:
            raise ValueError("ANL_USERNAME must be set in environment (.env file)")

        # Setup readline/libedit
        self.setup_completion()

    def setup_completion(self):
        """Setup tab completion specifically for libedit (macOS)"""
        try:
            import readline
            import atexit
            
            commands = ['models', 'model', 'servers', 'tools', 'help', 'quit', 'ls', 'run']
            
            def completer(text, state):
                matches = [cmd for cmd in commands if cmd.startswith(text)]
                if state < len(matches):
                    return matches[state]
                return None
            
            readline.set_completer(completer)
            
            # libedit-specific bindings
            readline.parse_and_bind("bind ^I rl_complete")
            
            # History
            history_file = os.path.expanduser("~/.beamline_history")
            readline.set_history_length(1000)
            
            try:
                readline.read_history_file(history_file)
            except FileNotFoundError:
                pass
            
            atexit.register(readline.write_history_file, history_file)
            
            print("libedit tab completion configured - try 'm' + TAB")
            
        except Exception as e:
            print(f"Tab completion setup failed: {e}")

    def _get_argo_url(self, model: str) -> str:
        env = self.model_env_mapping.get(model, "prod")
        return self.argo_urls[env]["chat"]

    async def connect_to_multiple_servers(self, server_configs: List[Dict[str, str]]):
        self.sessions = {}
        
        for config in server_configs:
            name = config["name"]
            script_path = config["script_path"]
            
            try:
                command = "python" if script_path.endswith('.py') else "node"
                server_params = StdioServerParameters(command=command, args=[script_path], env=None)
                
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                stdio, write = stdio_transport
                session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                await session.initialize()
                
                self.sessions[name] = session
                
                response = await session.list_tools()
                tools = response.tools
                print(f"Connected to {name} server with {len(tools)} tools")
                
            except Exception as e:
                print(f"Failed to connect to {name} server: {e}")

    def _prepare_argo_payload(self, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        payload = {
            "user": self.anl_username,
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4000
        }
        return payload

    async def call_argo_chat_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        try:
            url = self._get_argo_url(self.selected_model)
            payload = self._prepare_argo_payload(messages, self.selected_model)
            
            print(f"Calling Argo API with {self.selected_model}...")
            
            response = await self.http_client.post(url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            result = response.json()
            
            if 'response' in result:
                return {'choices': [{'message': {'role': 'assistant', 'content': result['response']}}]}
            elif 'choices' in result:
                return result
            else:
                return {'choices': [{'message': {'role': 'assistant', 'content': str(result)}}]}
                
        except Exception as e:
            print(f"Argo API Error: {str(e)}")
            raise

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

    def _format_phase_results(self, json_result: str) -> str:
        try:
            data = json.loads(json_result)
            if "identified_phases" not in data:
                return json_result
            
            output = ["PHASE IDENTIFICATION RESULTS", "=" * 50]
            
            params = data.get("analysis_parameters", {})
            output.append(f"Input peaks: {params.get('input_peaks', 'N/A')}")
            output.append(f"Peak positions: {params.get('peak_positions', [])}")
            output.append("")
            
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
                    output.append("")
            
            summary = data.get("phase_summary", {})
            output.append("ANALYSIS SUMMARY:")
            output.append(f"Total phases found: {summary.get('total_phases_found', 0)}")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Error formatting results: {e}"

    def _format_directory_listing(self, json_result: str) -> str:
        try:
            data = json.loads(json_result)
            output = [f"Directory: {data['directory']}", f"Items: {data['item_count']}", "-" * 50]
            
            for item in data['items']:
                icon = "DIR " if item['type'] == 'DIR' else "FILE"
                size = f" ({item['size']} bytes)" if item['size'] != '-' else ""
                output.append(f"[{icon}] {item['name']}{size}")
            
            return "\n".join(output)
        except:
            return json_result

    async def process_query(self, query: str) -> str:
        if not self.sessions:
            return "Error: Not connected to any MCP servers."
        
        # Phase identification
        peak_positions = self._extract_peak_positions(query)
        if peak_positions and len(peak_positions) >= 2:
            print(f"Detected {len(peak_positions)} peaks: {peak_positions}")
            try:
                result = await self.execute_tool_call("midas_identify_crystalline_phases", {"peak_positions": peak_positions})
                return self._format_phase_results(result)
            except Exception as e:
                return f"Error in phase analysis: {e}"
        
        # Directory operations
        if "files" in query.lower() and "directory" in query.lower():
            try:
                result = await self.execute_tool_call("filesystem_list_directory", {"path": "."})
                return self._format_directory_listing(result)
            except Exception as e:
                return f"Error listing directory: {e}"
        
        # General AI query
        try:
            messages = [
                {"role": "system", "content": "You are a helpful X-ray diffraction analysis assistant."},
                {"role": "user", "content": query}
            ]
            
            response_data = await self.call_argo_chat_api(messages)
            if 'choices' in response_data and response_data['choices']:
                return response_data['choices'][0]['message']['content']
            else:
                return f"Unexpected response format"
                
        except Exception as e:
            return f"Error processing query: {str(e)}"

    async def run_interactive_session(self):
        print("Beamline Assistant - AI Diffraction Analysis")
        print("=" * 60)
        print(f"Current AI Model: {self.selected_model}")
        print(f"ANL User: {self.anl_username}")
        print(f"Connected Servers: {list(self.sessions.keys())}")
        print()
        print("Commands: models, model <n>, servers, tools, ls, run <cmd>, help, quit")
        print("Natural language queries supported!")
        print()
        
        while True:
            try:
                user_input = input("Beamline> ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                    
                elif user_input.lower() == 'models':
                    print("Available Models: gpt4o (current), gpt4turbo, claudesonnet4")
                    
                elif user_input.lower() == 'servers':
                    print(f"Connected servers: {list(self.sessions.keys())}")
                    
                elif user_input.lower() == 'tools':
                    total = 0
                    for server_name, session in self.sessions.items():
                        try:
                            response = await session.list_tools()
                            tools = response.tools
                            print(f"{server_name}: {len(tools)} tools")
                            total += len(tools)
                        except Exception as e:
                            print(f"{server_name}: Error - {e}")
                    print(f"Total tools: {total}")
                    
                elif user_input.startswith('model '):
                    model_name = user_input[6:].strip()
                    if model_name in ['gpt4o', 'gpt4turbo', 'claudesonnet4']:
                        self.selected_model = model_name
                        print(f"Switched to: {model_name}")
                    else:
                        print(f"Invalid model: {model_name}")
                        
                elif user_input.startswith('run '):
                    command = user_input[4:].strip()
                    if "executor" in self.sessions:
                        result = await self.execute_tool_call("executor_run_command", {"command": command})
                        try:
                            data = json.loads(result)
                            if data.get('success'):
                                if data['stdout']:
                                    print(data['stdout'].strip())
                            else:
                                print(f"Command failed: {data.get('stderr', 'Unknown error')}")
                        except:
                            print(result)
                    else:
                        print("Executor server not connected")
                        
                elif user_input.startswith('ls'):
                    path = user_input[2:].strip() or "."
                    if "filesystem" in self.sessions:
                        result = await self.execute_tool_call("filesystem_list_directory", {"path": path})
                        formatted = self._format_directory_listing(result)
                        print(f"\n{formatted}\n")
                    else:
                        print("Filesystem server not connected")
                        
                elif user_input.lower() == 'help':
                    print("""
Commands:
• models       - Show available AI models
• model <n> - Switch AI model 
• servers      - Show connected servers  
• tools        - List all tools
• ls [path]    - List directory
• run <cmd>    - Execute system command
• help         - Show help
• quit         - Exit

Try: "I have peaks at 12.5, 18.2, 25.8 degrees"
Tab completion: Type 'm' and press TAB
""")
                    
                elif user_input:
                    response = await self.process_query(user_input)
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
            
        await client.run_interactive_session()
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())