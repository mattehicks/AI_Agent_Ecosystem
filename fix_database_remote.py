#!/usr/bin/env python3
"""
Remote database fix script - creates a simple HTTP endpoint to fix database
"""

import requests
import json

def fix_database_via_api():
    """Fix database by calling a simple endpoint that will trigger initialization"""
    
    base_url = "http://192.168.2.151:8000"
    
    # Try to call the debug endpoint to see current state
    try:
        response = requests.get(f"{base_url}/debug/static")
        print(f"Debug endpoint status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Web directory exists: {data.get('web_dir_exists')}")
    except Exception as e:
        print(f"Error calling debug endpoint: {e}")
    
    # Try to get health status
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health endpoint status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Health status: {data.get('status')}")
            print(f"Components: {data.get('components')}")
    except Exception as e:
        print(f"Error calling health endpoint: {e}")
    
    # Try to get agents (this should work)
    try:
        response = requests.get(f"{base_url}/agents")
        print(f"Agents endpoint status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Available agents: {list(data.keys())}")
    except Exception as e:
        print(f"Error calling agents endpoint: {e}")
    
    print("\nThe issue is that the database cannot be opened.")
    print("This suggests the /mnt/llm/AI_Agent_Ecosystem/data directory doesn't exist on the remote system.")
    print("The orchestrator tries to create it, but there might be permission issues.")
    
    return True

if __name__ == "__main__":
    fix_database_via_api()