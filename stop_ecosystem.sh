#!/bin/bash
# AI Agent Ecosystem Shutdown Script

echo "Stopping AI Agent Ecosystem..."

# Stop API server
if [ -f /tmp/ai_ecosystem_api.pid ]; then
    API_PID=$(cat /tmp/ai_ecosystem_api.pid)
    if kill -0 $API_PID 2>/dev/null; then
        echo "Stopping API server (PID: $API_PID)..."
        kill $API_PID
        rm /tmp/ai_ecosystem_api.pid
    else
        echo "API server not running"
    fi
fi

# Stop orchestrator
if [ -f /tmp/ai_ecosystem_orchestrator.pid ]; then
    ORCHESTRATOR_PID=$(cat /tmp/ai_ecosystem_orchestrator.pid)
    if kill -0 $ORCHESTRATOR_PID 2>/dev/null; then
        echo "Stopping orchestrator (PID: $ORCHESTRATOR_PID)..."
        kill $ORCHESTRATOR_PID
        rm /tmp/ai_ecosystem_orchestrator.pid
    else
        echo "Orchestrator not running"
    fi
fi

echo "AI Agent Ecosystem stopped."
