# APEXA Web UI + Image Viewer launcher for Windows (PowerShell). Opens http://localhost:8001
Set-Location -Path $PSScriptRoot
$env:NUMEXPR_MAX_THREADS = "10"
$dist = "frontend/dist/index.html"
if (Test-Path "frontend") {
    $stale = -not (Test-Path $dist)
    foreach ($src in @("frontend/src/components/viz/VizLauncher.tsx",
                       "frontend/src/components/viz/VizPanel.tsx")) {
        if ((Test-Path $src) -and ((-not (Test-Path $dist)) -or
            ((Get-Item $src).LastWriteTime -gt (Get-Item $dist).LastWriteTime))) { $stale = $true }
    }
    if ($stale) {
        Write-Host "Building React frontend..."
        Push-Location frontend; npm install --silent; npm run build; Pop-Location
    }
}
if (Get-Command uv -ErrorAction SilentlyContinue) { uv run python web_server.py @args } else { python web_server.py @args }
