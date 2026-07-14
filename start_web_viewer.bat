@echo off
setlocal
REM APEXA Web UI + Image Viewer launcher for Windows. Opens http://localhost:8001
cd /d "%~dp0"
set NUMEXPR_MAX_THREADS=10

REM The built frontend (frontend\dist) ships with the repo, so the Web UI runs
REM WITHOUT Node/npm. Only rebuild if dist is missing AND npm is available.
if not exist "frontend\dist\index.html" (
    where npm >nul 2>nul
    if errorlevel 1 (
        echo [!] frontend\dist not found and npm is not installed.
        echo     The Web UI needs the built frontend. Either pull the repo again
        echo     ^(dist is committed^) or install Node.js LTS and re-run:
        echo         winget install --id=OpenJS.NodeJS.LTS -e
    ) else (
        echo Building React frontend ^(one-time^)...
        pushd frontend
        call npm install --silent
        call npm run build
        popd
    )
)

where uv >nul 2>nul
if errorlevel 1 (
    echo [!] uv not found on PATH - falling back to system Python.
    python web_server.py %*
) else (
    uv run python web_server.py %*
)

if errorlevel 1 (
    echo.
    echo ============================================================
    echo APEXA Web UI exited with an error. Read the message above.
    echo Common fixes:  run "uv sync"  ^|  reopen terminal after installing uv
    echo ============================================================
    pause
)
