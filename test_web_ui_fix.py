#!/usr/bin/env python3
"""
Test script to verify web UI fixes
"""

import requests
import json
import time

def test_api_endpoints():
    """Test all API endpoints to verify they work"""
    
    base_url = "http://192.168.2.151:8000"
    
    endpoints_to_test = [
        ("/health", "Health Check"),
        ("/agents", "Agents List"),
        ("/metrics", "System Metrics"),
        ("/debug/static", "Static Files Debug"),
        ("/api", "API Info")
    ]
    
    print("🔧 Testing AI Agent Ecosystem Web UI")
    print("=" * 50)
    
    for endpoint, description in endpoints_to_test:
        try:
            print(f"\n📡 Testing {description} ({endpoint})...")
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            
            if response.status_code == 200:
                print(f"   ✅ SUCCESS: {response.status_code}")
                
                # Show some data for key endpoints
                if endpoint == "/metrics":
                    data = response.json()
                    print(f"   📊 Database Available: {data.get('database_available', 'unknown')}")
                    print(f"   📊 Active Tasks: {data.get('active_tasks', 0)}")
                    print(f"   📊 Agent Instances: {len(data.get('agent_instances', {}))}")
                    
                elif endpoint == "/agents":
                    data = response.json()
                    agent_count = len(data)
                    print(f"   🤖 Agent Types: {agent_count}")
                    for agent_type, info in data.items():
                        instances = len(info.get('instances', []))
                        queue_size = info.get('queue_size', 0)
                        print(f"      - {agent_type}: {instances} instances, {queue_size} queued")
                        
                elif endpoint == "/health":
                    data = response.json()
                    print(f"   💚 Status: {data.get('status', 'unknown')}")
                    components = data.get('components', {})
                    for comp, status in components.items():
                        print(f"      - {comp}: {status.get('status', 'unknown')}")
                        
            else:
                print(f"   ❌ FAILED: {response.status_code}")
                print(f"   📝 Response: {response.text[:200]}...")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ CONNECTION ERROR: {e}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    print("\n" + "=" * 50)
    
    # Test web interface loading
    print("\n🌐 Testing Web Interface...")
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            print("   ✅ Web interface loads successfully")
            
            # Check for key HTML elements
            html = response.text
            if "AI Agent Ecosystem - Dashboard" in html:
                print("   ✅ Dashboard title found")
            if "script.js" in html:
                print("   ✅ JavaScript file referenced")
            if "styles.css" in html:
                print("   ✅ CSS file referenced")
            if "nav-menu" in html:
                print("   ✅ Navigation menu found")
                
        else:
            print(f"   ❌ Web interface failed to load: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Web interface error: {e}")
    
    # Test static files
    print("\n📁 Testing Static Files...")
    static_files = [
        ("/static/styles.css", "CSS"),
        ("/static/script.js", "JavaScript")
    ]
    
    for file_path, file_type in static_files:
        try:
            response = requests.get(f"{base_url}{file_path}", timeout=5)
            if response.status_code == 200:
                print(f"   ✅ {file_type} file loads successfully")
                print(f"      Size: {len(response.content)} bytes")
            else:
                print(f"   ❌ {file_type} file failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ {file_type} file error: {e}")
    
    print("\n🎯 Summary:")
    print("   The web UI should now be working properly!")
    print("   Navigate to: http://192.168.2.151:8000")
    print("   Check browser console for any JavaScript errors.")
    
    return True

if __name__ == "__main__":
    test_api_endpoints()