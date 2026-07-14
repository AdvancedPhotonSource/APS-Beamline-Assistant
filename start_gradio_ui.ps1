# APEXA Gradio UI launcher for Windows (PowerShell). Opens http://localhost:7860
Set-Location -Path $PSScriptRoot
$env:NUMEXPR_MAX_THREADS = "10"
if (Get-Command uv -ErrorAction SilentlyContinue) { uv run python gradio_ui.py @args } else { python gradio_ui.py @args }
