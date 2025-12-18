#!/usr/bin/env python3
import asyncio
import subprocess
import sys
import time
from pathlib import Path

async def start_ecosystem():
    print("Starting AI Agent Ecosystem...")
    
    # Start orchestrator
    orchestrator_proc = subprocess.Popen([
        sys.executable, "orchestrator/orchestrator.py"
    ], cwd="X:/AI_Agent_Ecosystem")
    
    # Wait for orchestrator to initialize
    await asyncio.sleep(5)
    
    # Start API server
    api_proc = subprocess.Popen([
        sys.executable, "api/main.py"
    ], cwd="X:/AI_Agent_Ecosystem")
    
    print("✅ AI Agent Ecosystem started!")
    print("📡 API available at: http://localhost:8000")
    print("📚 Documentation at: http://localhost:8000/docs")
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(10)
    except KeyboardInterrupt:
        print("\nShutting down...")
        orchestrator_proc.terminate()
        api_proc.terminate()

if __name__ == "__main__":
    asyncio.run(start_ecosystem())
