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
    python -m pip install --upgrade pip
    where nvidia-smi >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo [*] NVIDIA GPU detected! Installing CUDA-accelerated PyTorch...
        pip install torch --index-url https://download.pytorch.org/whl/cu124
    ) else (
        echo [*] Installing PyTorch...
        pip install torch
    )
    pip install tiktoken huggingface_hub
) else (
    call venv\Scripts\activate.bat
    where nvidia-smi >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
        if %ERRORLEVEL% NEQ 0 (
            echo [*] NVIDIA GPU detected but PyTorch is CPU-only. Upgrading to CUDA PyTorch...
            pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
        )
    )
)

echo [*] Launching PCEI Resonance Chat (GPU Auto-Accelerated, Temp=0.5, Multi-Turn History)...
python chat.py --temperature 0.5 --top-p 0.9 --repetition-penalty 1.05 %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Chat session exited with code %ERRORLEVEL%.
    pause
)
