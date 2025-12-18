# AI Agent Ecosystem Web UI Troubleshooting Guide

## Current Status

✅ **Working Components:**
- API Server is running on http://192.168.2.151:8000
- Web interface loads correctly (HTML, CSS, JS)
- Agent management is functional
- Health check endpoint works
- Agents endpoint returns data
- Static file serving works

❌ **Issues Found:**

### 1. Database Access Error (CRITICAL)
**Problem:** `/metrics` endpoint returns 500 error with "unable to open database file"
**Cause:** Missing `/mnt/llm/AI_Agent_Ecosystem/data` directory on remote Linux system
**Impact:** Dashboard metrics don't load, affecting real-time updates

**Solution:**
```bash
# On the remote Linux system (192.168.2.151):
mkdir -p /mnt/llm/AI_Agent_Ecosystem/data
mkdir -p /mnt/llm/AI_Agent_Ecosystem/logs
chmod 755 /mnt/llm/AI_Agent_Ecosystem/data
chmod 755 /mnt/llm/AI_Agent_Ecosystem/logs
```

### 2. JavaScript API Calls Failing
**Problem:** Dashboard shows "System Offline" because metrics endpoint fails
**Cause:** The JavaScript tries to fetch from `/metrics` which returns 500
**Impact:** Real-time dashboard updates don't work

## Quick Fixes

### Fix 1: Create Missing Directories
The orchestrator expects these directories to exist:
- `/mnt/llm/AI_Agent_Ecosystem/data/` - for SQLite database
- `/mnt/llm/AI_Agent_Ecosystem/logs/` - for log files

### Fix 2: Database Initialization
The database needs these tables:
- `tasks` - for task tracking
- `agent_instances` - for agent status
- `metrics` - for performance data
- `result_cache` - for caching results

### Fix 3: Graceful Error Handling
The metrics endpoint should return default values when database is unavailable.

## Testing Steps

1. **Test API Endpoints:**
```bash
# These should work:
curl http://192.168.2.151:8000/health
curl http://192.168.2.151:8000/agents
curl http://192.168.2.151:8000/debug/static

# This currently fails:
curl http://192.168.2.151:8000/metrics
```

2. **Test Web Interface:**
- Navigate to http://192.168.2.151:8000
- Check browser console for JavaScript errors
- Verify dashboard loads with agent data

3. **Test Real-time Updates:**
- Check WebSocket connection in browser dev tools
- Verify metrics update every 5 seconds

## Implementation Priority

1. **HIGH:** Fix database directory creation
2. **HIGH:** Add graceful error handling to metrics endpoint
3. **MEDIUM:** Improve JavaScript error handling
4. **LOW:** Add retry logic for failed API calls

## Current Workarounds

While the database issue is being fixed:
1. The web interface still loads and shows the UI
2. Agent status can be viewed via `/agents` endpoint
3. Health check works via `/health` endpoint
4. Static dashboard data is visible

## Next Steps

1. Access the remote Linux system to create missing directories
2. Restart the API service to reinitialize database
3. Test all endpoints to verify fixes
4. Update orchestrator code to handle directory creation more robustly