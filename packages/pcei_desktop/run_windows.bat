@echo off
rem ==============================================================================
rem PCEI 750M Desktop Launcher for Windows
rem Double-click this file to launch PCEI Resonance Chat in Command Prompt.
rem ==============================================================================

cd /d "%~dp0"
title PCEI 750M Resonance Chat

echo ================================================================
echo   PCEI 750M Desktop — Windows Auto-Boot Terminal Launcher
echo ================================================================

rem Detect Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    where py >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [!] Error: Python was not found in your PATH.
        echo     Please install Python 3.10+ from https://www.python.org/
        echo     (Make sure to check "Add Python to PATH" during installation)
        pause
        exit /b 1
    ) else (
        set PY_CMD=py
    )
) else (
    set PY_CMD=python
)

rem Initialize local virtual environment if not present
if not exist "venv" (
    echo [*] Initializing local environment (venv)...
    %PY_CMD% -m venv venv
    call venv\Scripts\activate.bat
    echo [*] Installing runtime dependencies...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo [*] Launching PCEI Resonance Chat...
python chat.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Chat session exited with code %ERRORLEVEL%.
    pause
)
