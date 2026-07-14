@echo off
REM APEXA launcher for Windows (Command Prompt).
REM Double-click, or run:  start_beamline_assistant.bat
cd /d "%~dp0"
set NUMEXPR_MAX_THREADS=10
where uv >nul 2>nul
if %ERRORLEVEL%==0 (
    uv run python launch.py %*
) else (
    echo uv not found on PATH - falling back to system Python.
    echo Install uv: https://docs.astral.sh/uv/  ^(then run: uv sync^)
    python launch.py %*
)
