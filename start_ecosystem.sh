#!/bin/bash
# AI Agent Ecosystem Startup Script for Linux

echo "Starting AI Agent Ecosystem on Linux..."

# Change to project directory
cd /mnt/llm/AI_Agent_Ecosystem

# Activate virtual environment
source venv/bin/activate

# Start the orchestrator in background
echo "Starting orchestrator..."
nohup python orchestrator/orchestrator.py > logs/orchestrator.log 2>&1 &
ORCHESTRATOR_PID=$!
echo "Orchestrator started with PID: $ORCHESTRATOR_PID"

# Wait for orchestrator to initialize
sleep 5

# Start the API server in background
echo "Starting API server..."
nohup python api/main.py > logs/api.log 2>&1 &
API_PID=$!
echo "API server started with PID: $API_PID"

# Save PIDs for shutdown script
echo $ORCHESTRATOR_PID > /tmp/ai_ecosystem_orchestrator.pid
echo $API_PID > /tmp/ai_ecosystem_api.pid

echo ""
echo "=== AI Agent Ecosystem Started Successfully! ==="
echo "API available at: http://localhost:8000"
echo "Documentation at: http://localhost:8000/docs"
echo "Logs available in: /mnt/llm/AI_Agent_Ecosystem/logs/"
echo ""
echo "To stop the system, run: ./stop_ecosystem.sh"
echo "To monitor logs: tail -f logs/orchestrator.log logs/api.log"
