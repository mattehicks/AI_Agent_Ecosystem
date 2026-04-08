#!/usr/bin/env python3
import asyncio
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


async def start_ecosystem():
    print("Starting AI Agent Ecosystem...")
    print(f"ROOT={ROOT}")

    orchestrator_proc = subprocess.Popen(
        [sys.executable, str(ROOT / "orchestrator" / "orchestrator.py")],
        cwd=str(ROOT),
    )

    await asyncio.sleep(5)

    api_proc = subprocess.Popen(
        [sys.executable, str(ROOT / "api" / "main.py")],
        cwd=str(ROOT),
    )

    print("AI Agent Ecosystem started!")
    print("API available at: http://localhost:8000")
    print("Documentation at: http://localhost:8000/docs")

    try:
        while True:
            await asyncio.sleep(10)
    except KeyboardInterrupt:
        print("\nShutting down...")
        orchestrator_proc.terminate()
        api_proc.terminate()


if __name__ == "__main__":
    asyncio.run(start_ecosystem())
