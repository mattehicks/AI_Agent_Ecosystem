#!/bin/bash
# AI Agent Ecosystem startup (orchestrator + API). Run from repo clone.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1

echo "Starting AI Agent Ecosystem on Linux..."
echo "ROOT=$ROOT"

source venv/bin/activate

echo "Starting orchestrator..."
nohup python orchestrator/orchestrator.py > logs/orchestrator.log 2>&1 &
ORCHESTRATOR_PID=$!
echo "Orchestrator started with PID: $ORCHESTRATOR_PID"

sleep 5

echo "Starting API server..."
nohup python api/main.py > logs/api.log 2>&1 &
API_PID=$!
echo "API server started with PID: $API_PID"

echo $ORCHESTRATOR_PID > /tmp/ai_ecosystem_orchestrator.pid
echo $API_PID > /tmp/ai_ecosystem_api.pid

echo ""
echo "=== AI Agent Ecosystem Started Successfully! ==="
echo "API available at: http://localhost:8000"
echo "Documentation at: http://localhost:8000/docs"
echo "Logs available in: $ROOT/logs/"
echo ""
echo "To stop: $ROOT/scripts/stop_ecosystem.sh"
echo "To monitor logs: tail -f logs/orchestrator.log logs/api.log"
