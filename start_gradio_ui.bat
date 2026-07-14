@echo off
REM APEXA Gradio UI launcher for Windows. Opens http://localhost:7860
cd /d "%~dp0"
set NUMEXPR_MAX_THREADS=10
where uv >nul 2>nul && (uv run python gradio_ui.py %*) || (python gradio_ui.py %*)

if errorlevel 1 (
    echo.
    echo APEXA exited with an error - read the message above.
    pause
)
