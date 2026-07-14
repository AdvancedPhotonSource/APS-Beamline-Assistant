# APEXA Web UI + Image Viewer launcher for Windows (PowerShell). Opens http://localhost:8001
Set-Location -Path $PSScriptRoot
$env:NUMEXPR_MAX_THREADS = "10"

# The built frontend (frontend/dist) ships with the repo, so the Web UI runs
# WITHOUT Node/npm. Only rebuild if dist is missing AND npm is available.
if (-not (Test-Path "frontend/dist/index.html")) {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Write-Host "Building React frontend (one-time)..."
        Push-Location frontend; npm install --silent; npm run build; Pop-Location
    } else {
        Write-Host "[!] frontend/dist not found and npm is not installed."
        Write-Host "    Pull the repo again (dist is committed) or install Node.js LTS:"
        Write-Host "        winget install --id=OpenJS.NodeJS.LTS -e"
    }
}

if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv run python web_server.py @args
} else {
    Write-Host "uv not found on PATH - falling back to system Python."
    python web_server.py @args
}
