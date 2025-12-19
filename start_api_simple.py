#!/usr/bin/env python3
"""
Simple API startup script for AI Agent Ecosystem
Starts only the core API without GPU platform dependencies
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    # Set environment variable to disable GPU platform
    os.environ['DISABLE_GPU_PLATFORM'] = '1'
    
    print("Starting AI Agent Ecosystem API...")
    print("Access the web interface at: http://192.168.2.151:8000")
    
    try:
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nAPI server stopped.")
    except Exception as e:
        print(f"Failed to start API server: {e}")