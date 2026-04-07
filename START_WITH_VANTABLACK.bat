@echo off
REM Start AI_Agent_Ecosystem with VantaBlack RAG Integration
echo ===================================
echo AI Agent Ecosystem + VantaBlack RAG
echo ===================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    pause
    exit /b 1
)

REM Check VantaBlack connectivity
echo Testing VantaBlack connection...
ssh -i C:\Users\matte\.ssh\id_ed25519 lightspeed@vantablack "echo Connected" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Cannot connect to VantaBlack
    echo Make sure SSH is configured and VantaBlack is online
    pause
)

echo.
echo Starting AI Agent Ecosystem...
echo Web UI will be available at: http://localhost:8000
echo.

REM Activate venv if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate
) else (
    echo WARNING: Virtual environment not found
    echo Run: python -m venv venv
    pause
)

REM Start the API
python api/main.py

pause
