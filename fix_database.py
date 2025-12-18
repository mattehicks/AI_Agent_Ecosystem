#!/usr/bin/env python3
"""
Fix database initialization issues for the AI Agent Ecosystem
"""

import os
import sqlite3
from pathlib import Path

def fix_database():
    """Create missing directories and initialize database"""
    
    # Detect environment
    if os.name == 'nt' or os.path.exists("X:/"):
        # Windows environment
        base_path = Path("X:/AI_Agent_Ecosystem")
    else:
        # Linux environment  
        base_path = Path("/mnt/llm/AI_Agent_Ecosystem")
    
    print(f"Working with base path: {base_path}")
    
    # Create missing directories
    data_dir = base_path / "data"
    logs_dir = base_path / "logs"
    
    data_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    print(f"Created directories: {data_dir}, {logs_dir}")
    
    # Initialize database
    db_path = data_dir / "orchestrator.db"
    
    try:
        conn = sqlite3.connect(db_path)
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
                id TEXT PRIMARY KEY,
                agent_type TEXT NOT NULL,
                status TEXT NOT NULL,
                current_task_id TEXT,
                total_tasks INTEGER DEFAULT 0,
                successful_tasks INTEGER DEFAULT 0,
                failed_tasks INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                last_activity TEXT NOT NULL
            )
        ''')
        
        # System metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                timestamp TEXT PRIMARY KEY,
                metric_type TEXT NOT NULL,
                metric_data TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"Database initialized successfully at: {db_path}")
        
        # Test database access
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        
        print(f"Database tables created: {[table[0] for table in tables]}")
        
        return True
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

if __name__ == "__main__":
    success = fix_database()
    if success:
        print("Database fix completed successfully!")
    else:
        print("Database fix failed!")