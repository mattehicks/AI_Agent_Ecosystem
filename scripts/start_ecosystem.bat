@echo off
REM Start orchestrator + API; repo root is parent of scripts\
cd /d "%~dp0.."
echo Starting AI Agent Ecosystem...
echo ROOT=%CD%

start "Orchestrator" python orchestrator\orchestrator.py

timeout /t 5 /nobreak > nul

start "API Server" python api\main.py

echo AI Agent Ecosystem started!
echo API available at: http://localhost:8000
echo Documentation at: http://localhost:8000/docs

pause
