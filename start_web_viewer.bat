@echo off
REM APEXA Web UI + Image Viewer launcher for Windows. Opens http://localhost:8001
cd /d "%~dp0"
set NUMEXPR_MAX_THREADS=10
if exist "frontend" if not exist "frontend\dist\index.html" (
    echo Building React frontend ^(one-time^)...
    pushd frontend
    call npm install --silent
    call npm run build
    popd
)
where uv >nul 2>nul && (uv run python web_server.py %*) || (python web_server.py %*)

if errorlevel 1 (
    echo.
    echo APEXA exited with an error - read the message above.
    pause
)
