@echo off
REM APEXA Desktop UI Startup Script (Windows)
REM Launches the web UI in a native desktop window (pywebview) — the same React
REM frontend + FastAPI backend as the web server, no browser required.

cd /d "%~dp0"

echo Starting APEXA Desktop UI

REM Build the React frontend if the bundle is missing.
if not exist "frontend\dist\index.html" (
    echo Building React frontend...
    pushd frontend
    call npm install
    call npm run build
    popd
)

echo Opening desktop window (close the window to stop)...
uv run python apexa_desktop.py
