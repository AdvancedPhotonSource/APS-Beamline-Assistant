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
from argo_mcp_client import ArgoMCPClient


class APEXAGradioUI:
    """Gradio UI wrapper for APEXA MCP client"""

    def __init__(self, servers_config: str = "servers.config"):
        self.servers_config = servers_config
        self.client = None
        self.chat_history = []

    async def initialize(self):
        """Initialize MCP client with servers"""
        self.client = ArgoMCPClient()

        # Parse servers.config
        servers = {}
        with open(self.servers_config, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if ':' in line:
                        name, path = line.split(':', 1)
                        servers[name.strip()] = path.strip()

        # Connect to servers
        for name, path in servers.items():
            try:
                await self.client.connect_to_server(name, path)
                print(f"✓ Connected to {name}")
            except Exception as e:
                print(f"✗ Failed to connect to {name}: {e}")

        return "✓ APEXA initialized successfully!"

    async def chat(self, message: str, history: List[Tuple[str, str]]) -> str:
        """Process chat message through MCP client"""
        if not self.client:
            return "⚠️ APEXA not initialized. Please restart."

        try:
            # Process query through MCP client
            response = await self.client.process_diffraction_query(
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


def create_ui():
    """Create Gradio interface"""

    ui = APEXAGradioUI()

    # Custom CSS for scientific look
    custom_css = """
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

    with gr.Blocks(
        title="APEXA - Advanced Photon EXperiment Assistant",
        theme=gr.themes.Soft(primary_hue="blue"),
        css=custom_css
    ) as demo:

        # Header
        gr.Markdown("""
        # 🔬 APEXA - Advanced Photon EXperiment Assistant
        **Your AI Scientist at the Beamline** | Argonne National Laboratory

        Natural language interface to MIDAS workflows: calibration, integration, phase analysis, and more.
        """)

        # Initialize on load
        demo.load(ui.initialize, outputs=None)

        with gr.Row():
            with gr.Column(scale=3):
                # Main chat interface
                chatbot = gr.Chatbot(
                    label="APEXA Chat",
                    height=600,
                    show_copy_button=True,
                    bubble_full_width=False,
                    avatar_images=(None, "🤖")
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

        # Chat interaction
        def respond(message, history):
            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(ui.chat(message, history))
            loop.close()

            # Add to history
            history.append((message, response))
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
        quiet=False
    )
