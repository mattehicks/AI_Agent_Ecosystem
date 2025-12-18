@echo off
echo Starting AI Agent Ecosystem...

REM Start the orchestrator
cd /d "X:\AI_Agent_Ecosystem"
start "Orchestrator" python orchestrator\orchestrator.py

REM Wait a moment for orchestrator to start
timeout /t 5 /nobreak > nul

REM Start the API server
start "API Server" python api\main.py

echo AI Agent Ecosystem started!
echo API available at: http://localhost:8000
echo Documentation at: http://localhost:8000/docs

pause
