import subprocess
import json
import sys
import os
import platform
import shutil
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("command-executor")

# Allowed commands for security
ALLOWED_COMMANDS = {
    'ls', 'pwd', 'cat', 'head', 'tail', 'grep', 'find',
    'python', 'python3', 'pip', 'pip3', 'uv', 'git',
    'ps', 'df', 'du', 'whoami', 'env', 'echo', 'date',
    'which', 'file', 'wc', 'sort', 'uniq'
}

def is_command_allowed(command: str) -> bool:
    cmd_parts = command.strip().split()
    if not cmd_parts:
        return False
    base_command = cmd_parts[0].lower()
    return base_command in ALLOWED_COMMANDS

@mcp.tool()
async def run_command(command: str, working_dir: str = None, timeout: int = 30) -> str:
    """Execute a system command safely."""
    try:
        if not is_command_allowed(command):
            return f"Command not allowed: {command.split()[0] if command.split() else command}"
        
        cwd = working_dir if working_dir and Path(working_dir).exists() else None
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )
        
        response = {
            "command": command,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
        
        return json.dumps(response, indent=2)
        
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"

@mcp.tool()
async def check_environment() -> str:
    """Get system environment information."""
    try:
        info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "current_directory": str(Path.cwd()),
            "home_directory": str(Path.home())
        }
        
        return json.dumps(info, indent=2)
        
    except Exception as e:
        return f"Error checking environment: {str(e)}"

@mcp.tool()
async def find_executable(program: str) -> str:
    """Find location of an executable."""
    try:
        location = shutil.which(program)
        
        if location:
            result = {
                "program": program,
                "found": True,
                "location": location
            }
        else:
            result = {
                "program": program,
                "found": False,
                "message": f"Program '{program}' not found"
            }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return f"Error finding executable: {str(e)}"

if __name__ == "__main__":
    print("Starting Command Executor FastMCP Server...", file=sys.stderr)
    mcp.run(transport='stdio')