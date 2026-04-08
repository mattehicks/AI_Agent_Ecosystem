#!/usr/bin/env python3
"""
Patch for orchestrator to fix database initialization issues
"""

import os
import sqlite3
from pathlib import Path

def patch_orchestrator_database():
    """
    This function contains the fixed database initialization logic
    that should be applied to the orchestrator
    """
    
    # Detect environment
    if os.name == 'nt' or os.path.exists("X:/"):
        # Windows environment
        base_path = Path("X:/AI_Agent_Ecosystem")
    else:
        # Linux environment  
        base_path = Path("/mnt/llm/AI_Agent_Ecosystem")
    
    db_path = base_path / "data" / "orchestrator.db"
    
    # Ensure directories exist with proper error handling
    try:
        (base_path / "data").mkdir(parents=True, exist_ok=True)
        (base_path / "logs").mkdir(parents=True, exist_ok=True)
        print(f"Created directories: {base_path}/data, {base_path}/logs")
    except Exception as e:
        print(f"Error creating directories: {e}")
        return False
    
    # Initialize database with error handling
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
        
        print(f"Database initialized successfully at: {db_path}")
        return True
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

def get_system_metrics_safe():
    """
    Safe version of get_system_metrics that handles database errors gracefully
    """
    
    # Detect environment
    if os.name == 'nt' or os.path.exists("X:/"):
        base_path = Path("X:/AI_Agent_Ecosystem")
    else:
        base_path = Path("/mnt/llm/AI_Agent_Ecosystem")
    
    db_path = base_path / "data" / "orchestrator.db"
    
    # Default metrics if database is not available
    default_metrics = {
        'uptime_seconds': 0,
        'task_stats': {
            'created': 0,
            'completed': 0,
            'failed': 0,
            'pending': 0
        },
        'agent_stats': {},
        'active_tasks': 0,
        'agent_instances': {},
        'queue_sizes': {}
    }
    
    try:
        # Try to initialize database first
        if not db_path.exists():
            patch_orchestrator_database()
        
        # Try to connect and get metrics
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get basic task stats
        cursor.execute("SELECT COUNT(*) FROM tasks")
        total_tasks = cursor.fetchone()[0] if cursor.fetchone() else 0
        
        cursor.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status")
        status_counts = dict(cursor.fetchall())
        
        conn.close()
        
        # Update metrics with actual data
        default_metrics['task_stats'].update(status_counts)
        
        return default_metrics
        
    except Exception as e:
        print(f"Database error in metrics: {e}")
        # Return default metrics if database fails
        return default_metrics

if __name__ == "__main__":
    print("Testing database patch...")
    success = patch_orchestrator_database()
    if success:
        print("Database patch applied successfully!")
        metrics = get_system_metrics_safe()
        print(f"Sample metrics: {metrics}")
    else:
        print("Database patch failed!")