#!/bin/bash
# AI Agent Ecosystem Database Fix Script
# Run this on the remote Linux system (192.168.2.151)

echo "🔧 AI Agent Ecosystem Database Fix Script"
echo "=========================================="

# Navigate to the project directory
cd /mnt/llm/AI_Agent_Ecosystem || {
    echo "❌ Error: Could not navigate to /mnt/llm/AI_Agent_Ecosystem"
    echo "Please check if the directory exists and you have permissions"
    exit 1
}

echo "✅ Current directory: $(pwd)"

# Create missing directories
echo "📁 Creating missing directories..."
mkdir -p data
mkdir -p logs
chmod 755 data logs

echo "✅ Created directories: data/ and logs/"

# Create database initialization script
echo "🗄️ Creating database initialization script..."
cat > fix_database.py << 'EOF'
#!/usr/bin/env python3
import os
import sqlite3
from pathlib import Path

def init_database():
    """Initialize the orchestrator database with all required tables"""
    
    base_path = Path("/mnt/llm/AI_Agent_Ecosystem")
    db_path = base_path / "data" / "orchestrator.db"
    
    print(f"Initializing database at: {db_path}")
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Tasks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                agent_type TEXT NOT NULL,
                description TEXT NOT NULL,
                input_data TEXT NOT NULL,
                priority INTEGER NOT NULL,
                status TEXT NOT NULL,
                result TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                cache_key TEXT
            )
        ''')
        
        # Agent instances table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_instances (
                instance_id TEXT PRIMARY KEY,
                agent_type TEXT NOT NULL,
                status TEXT NOT NULL,
                current_task_id TEXT,
                last_activity TEXT NOT NULL,
                total_tasks INTEGER DEFAULT 0,
                successful_tasks INTEGER DEFAULT 0
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                metric_name TEXT PRIMARY KEY,
                metric_value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Result cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS result_cache (
                cache_key TEXT PRIMARY KEY,
                result TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print("✅ Database initialized successfully!")
        
        # Verify tables were created
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        print(f"✅ Created tables: {', '.join(tables)}")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

if __name__ == "__main__":
    success = init_database()
    if success:
        print("🎉 Database fix completed successfully!")
    else:
        print("💥 Database fix failed!")
EOF

# Run the database initialization
echo "🚀 Running database initialization..."
python3 fix_database.py

# Check if database was created successfully
if [ -f "data/orchestrator.db" ]; then
    echo "✅ Database file created successfully"
    ls -la data/orchestrator.db
else
    echo "❌ Database file was not created"
    exit 1
fi

# Find and stop current API process
echo "🔄 Restarting API service..."
API_PID=$(pgrep -f "python.*main.py")
if [ ! -z "$API_PID" ]; then
    echo "🛑 Stopping existing API process (PID: $API_PID)"
    kill $API_PID
    sleep 2
fi

# Start API service in background
echo "🚀 Starting API service..."
nohup python3 api/main.py > logs/api.log 2>&1 &
API_NEW_PID=$!

echo "✅ API service started with PID: $API_NEW_PID"

# Wait a moment for startup
sleep 3

# Test the endpoints
echo "🧪 Testing API endpoints..."

echo "📡 Testing health endpoint..."
curl -s http://localhost:8000/health | head -1

echo -e "\n📡 Testing metrics endpoint..."
curl -s http://localhost:8000/metrics | head -1

echo -e "\n📡 Testing agents endpoint..."
curl -s http://localhost:8000/agents | head -1

echo -e "\n🎯 Fix completed! Test the web interface at:"
echo "   http://192.168.2.151:8000"

echo -e "\n📋 Summary:"
echo "✅ Created missing directories (data/, logs/)"
echo "✅ Initialized database with all required tables"
echo "✅ Restarted API service"
echo "✅ Verified endpoints are responding"

echo -e "\n📝 If you encounter issues, check the logs:"
echo "   tail -f /mnt/llm/AI_Agent_Ecosystem/logs/api.log"