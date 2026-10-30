@echo off
rem ==============================================================================
rem PCEI 750M Desktop Launcher for Windows
rem Double-click to launch PCEI Resonance Chat. Self-repairs its Python environment
rem and uses an NVIDIA GPU automatically when possible.
rem ==============================================================================

cd /d "%~dp0"
title PCEI 750M Resonance Chat

echo ================================================================
echo   PCEI 750M Desktop - Windows Auto-Boot Terminal Launcher
echo ================================================================

set PY_CMD=
python -c "import sys" >nul 2>nul
if not errorlevel 1 set PY_CMD=python
if "%PY_CMD%"=="" (
    py -3 -c "import sys" >nul 2>nul
    if not errorlevel 1 set PY_CMD=py -3
)
if "%PY_CMD%"=="" (
    echo [!] Python was not found. Install Python 3.10 to 3.12 from https://www.python.org/
    echo     and tick Add Python to PATH during installation.
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo [*] Creating local environment...
    %PY_CMD% -m venv venv
    if errorlevel 1 (
        echo [!] Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

echo [*] Checking dependencies and GPU support...
venv\Scripts\python.exe bootstrap.py
if errorlevel 1 (
    echo [!] Dependency setup failed. See the messages above.
    pause
    exit /b 1
)

echo [*] Launching PCEI Resonance Chat, Temp 0.5, multi-turn history ON...
venv\Scripts\python.exe chat.py --temperature 0.5 --top-p 0.9 --repetition-penalty 1.05 %*
if errorlevel 1 (
    echo.
    echo [!] Chat exited with an error.
    pause
)
