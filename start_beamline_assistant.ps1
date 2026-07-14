# APEXA launcher for Windows (PowerShell).
#   Run:  .\start_beamline_assistant.ps1
# If script execution is blocked:  powershell -ExecutionPolicy Bypass -File .\start_beamline_assistant.ps1
Set-Location -Path $PSScriptRoot
$env:NUMEXPR_MAX_THREADS = "10"
# Let older CMake-based native deps (e.g. midas-index) build under CMake 4.x
$env:CMAKE_POLICY_VERSION_MINIMUM = "3.5"
if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv run python launch.py @args
} else {
    Write-Host "uv not found on PATH - falling back to system Python."
    Write-Host "Install uv: https://docs.astral.sh/uv/  (then run: uv sync)"
    python launch.py @args
}
