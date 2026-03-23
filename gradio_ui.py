#!/usr/bin/env python3
"""
APEXA Gradio UI - Conversational interface for beamline analysis
Connects to existing MCP servers via argo_mcp_client
"""

import gradio as gr
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional
import sys
import json

# Import existing MCP client
from argo_mcp_client import APEXAClient


class APEXAGradioUI:
    """Gradio UI wrapper for APEXA MCP client"""

    def __init__(self, servers_config: str = "servers.config"):
        self.servers_config = servers_config
        self.client = None
        self.chat_history = []

    def _read_servers_config(self) -> List:
        """Parse servers.config into a list of {name, script_path} dicts."""
        configs = []
        config_path = Path(self.servers_config)
        if not config_path.exists():
            print(f"⚠️  {self.servers_config} not found — no MCP servers will be loaded")
            return configs
        for line in config_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                name, script_path = line.split(":", 1)
                if Path(script_path).exists():
                    configs.append({"name": name.strip(), "script_path": script_path.strip()})
                    print(f"  ✓ {name.strip()} ({script_path.strip()})")
                else:
                    print(f"  ✗ {name.strip()} — file not found: {script_path.strip()}")
        return configs

    def initialize_sync(self):
        """Initialize MCP client with servers (synchronous wrapper)"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._initialize_async())
            print("✓ APEXA initialized successfully!")
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            loop.close()

    async def _initialize_async(self):
        """Initialize MCP client and connect to MCP servers from servers.config."""
        self.client = APEXAClient()
        print(f"✓ Argo API: {self.client.environment} environment, model={self.client.selected_model}")

        print("Loading MCP servers:")
        server_configs = self._read_servers_config()

        if server_configs:
            await self.client.connect_to_multiple_servers(server_configs)
            print(f"✓ Connected to {len(self.client.sessions)} server(s): "
                  f"{list(self.client.sessions.keys())}")
        else:
            print("⚠️  No MCP servers loaded — tool calls will fail")

    async def chat(self, message: str, history: List[Tuple[str, str]]) -> str:
        """Process chat message through MCP client"""
        if not self.client:
            return "⚠️ APEXA not initialized. Please restart."

        try:
            # Process query through MCP client
            response = await self.client.run_query(
                query=message,
                use_history=True
            )

            return response

        except Exception as e:
            return f"❌ Error: {str(e)}\n\nTry rephrasing your request or check the logs."

    def upload_file(self, file) -> str:
        """Handle file upload"""
        if file is None:
            return "No file uploaded"

        # Copy to uploads directory
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)

        file_path = Path(file.name)
        dest = uploads_dir / file_path.name

        # Copy file
        import shutil
        shutil.copy(file.name, dest)

        return f"✓ Uploaded: {dest}\n\nYou can now reference it in chat: '{dest.name}'"


# Custom CSS for scientific look
CUSTOM_CSS = """
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}
.chat-message {
    font-size: 14px;
}
footer {
    display: none !important;
}
"""


def create_ui():
    """Create Gradio interface"""

    ui = APEXAGradioUI()

    with gr.Blocks(
        title="APEXA - Advanced Photon EXperiment Assistant"
    ) as demo:

        # Header
        gr.Markdown("""
        # 🔬 APEXA - Advanced Photon EXperiment Assistant
        **Your AI Scientist at the Beamline** | Argonne National Laboratory

        Natural language interface to MIDAS workflows: calibration, integration, phase analysis, and more.
        """)

        with gr.Row():
            with gr.Column(scale=3):
                # Main chat interface
                chatbot = gr.Chatbot(
                    label="APEXA Chat",
                    height=600,
                    value=[]  # Initialize with empty list
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Ask me to calibrate, integrate, or analyze your data...",
                        scale=9,
                        lines=2
                    )
                    submit = gr.Button("Send", variant="primary", scale=1)

                # Example prompts
                gr.Examples(
                    examples=[
                        "Calibrate the CeO2 image in test5 folder with stopping strain 0.003",
                        "Show me files in the current directory",
                        "What is a good calibration strain value?",
                        "Integrate the .tif file using refined parameters",
                        "Explain the FF-HEDM workflow",
                        "What materials are common calibrants for HEDM?"
                    ],
                    inputs=msg,
                    label="Example Prompts"
                )

            with gr.Column(scale=1):
                # Sidebar with file upload and quick info
                gr.Markdown("### 📁 Quick Actions")

                file_upload = gr.File(
                    label="Upload Data File",
                    file_types=[".tif", ".tiff", ".ge2", ".ge3", ".ge4", ".ge5",
                               ".h5", ".hdf5", ".txt", ".bin", ".npy"],
                    type="filepath"
                )
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False,
                    lines=3
                )

                file_upload.change(
                    ui.upload_file,
                    inputs=file_upload,
                    outputs=upload_status
                )

                gr.Markdown("### 🔧 Available Tools")
                gr.Markdown("""
                - **Calibration**: AutoCalibrateZarr.py
                - **Integration**: 2D → 1D caking
                - **Phase ID**: Peak matching
                - **Workflows**: FF/NF/PF-HEDM
                - **Knowledge Base**: Research papers & logbooks
                - **Materials DB**: Materials Project
                """)

                gr.Markdown("### 📊 Status")
                status_box = gr.Textbox(
                    label="",
                    value="✓ Connected to MCP servers\n✓ MIDAS environment ready\n✓ Knowledge base indexed",
                    interactive=False,
                    lines=4
                )

        # Initialize client on first load
        ui.initialize_sync()

        # Chat interaction
        def respond(message, history):
            # Initialize history if None
            if history is None:
                history = []

            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(ui.chat(message, history))
            loop.close()

            # Add to history in Gradio 6.x format
            if not history:
                history = []
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ]
            return "", history

        # Submit handlers
        msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        submit.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

        # Footer
        gr.Markdown("""
        ---
        **APEXA** - Powered by MCP Servers | [User Manual](USER_MANUAL.md) | [GitHub](https://github.com/AdvancedPhotonSource/APS-Beamline-Assistant)
        """)

    return demo


if __name__ == "__main__":
    print("🚀 Starting APEXA Gradio UI...")
    print("=" * 60)

    # Create and launch
    demo = create_ui()

    # Launch with custom settings
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set True for public Gradio link
        show_error=True,
        quiet=False,
        # Gradio 6.x parameters
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CUSTOM_CSS
    )
