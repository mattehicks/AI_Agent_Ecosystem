#!/usr/bin/env python3
"""
AI Agent Ecosystem Setup Script for Linux
Initializes and configures the AI agent system on Linux with virtual environment
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentEcosystemSetup:
    """Setup and configuration manager for the AI Agent Ecosystem"""
    
    def __init__(self):
        self.base_path = Path("/mnt/llm/AI_Agent_Ecosystem")
        self.venv_path = self.base_path / "venv"
        self.python_exe = self.venv_path / "bin" / "python"
        self.pip_exe = self.venv_path / "bin" / "pip"
        
        self.required_dirs = [
            "orchestrator",
            "agents", 
            "api",
            "config",
            "logs",
            "data",
            "temp",
            "integrations"
        ]
        
        self.required_files = {
            "config/system.yaml": "System configuration",
            "orchestrator/orchestrator.py": "Main orchestrator",
            "api/main.py": "API server",
            "agents/base_agent.py": "Base agent class"
        }

    def run_setup(self):
        """Run complete setup process"""
        logger.info("Starting AI Agent Ecosystem setup on Linux...")
        
        try:
            self.check_prerequisites()
            self.create_directory_structure()
            self.initialize_database()
            self.verify_integrations()
            self.create_startup_scripts()
            self.run_health_check()
            
            logger.info("[SUCCESS] Setup completed successfully!")
            self.print_next_steps()
            
        except Exception as e:
            logger.error(f"[ERROR] Setup failed: {e}")
            sys.exit(1)

    def check_prerequisites(self):
        """Check system prerequisites"""
        logger.info("Checking prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise Exception("Python 3.8 or higher is required")
        
        # Check virtual environment
        if not self.venv_path.exists():
            raise Exception("Virtual environment not found. Please create it first with: python3 -m venv venv")
        
        if not self.python_exe.exists():
            raise Exception("Virtual environment Python not found")
        
        # Check /mnt/llm accessibility
        if not Path("/mnt/llm").exists():
            raise Exception("/mnt/llm not accessible. Please ensure the drive is mounted.")
        
        # Check existing AI infrastructure
        self.check_ai_infrastructure()
        
        logger.info("[OK] Prerequisites check passed")

    def check_ai_infrastructure(self):
        """Check existing AI infrastructure"""
        logger.info("Checking existing AI infrastructure...")
        
        infrastructure_checks = {
            "privateGPT": Path("/mnt/llm/privateGPT/privateGPT.py"),
            "GPT4All CLI": Path("/mnt/llm/gpt4all_cli/app.py"),
            "LM Studio": Path("/mnt/llm/LM Studio/LM Studio.exe"),
            "Models Directory": Path("/mnt/llm/LLM-Models"),
            "Text Vault": Path("/mnt/llm/TEXT-VAULT"),
            "AI Prompts": Path("/mnt/llm/AIPROMPTS")
        }
        
        for name, path in infrastructure_checks.items():
            if path.exists():
                logger.info(f"[OK] {name} found at {path}")
            else:
                logger.warning(f"[WARN] {name} not found at {path}")

    def create_directory_structure(self):
        """Create required directory structure"""
        logger.info("Creating directory structure...")
        
        for dir_name in self.required_dirs:
            dir_path = self.base_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"[OK] Created directory: {dir_path}")
        
        # Create subdirectories
        subdirs = [
            "logs/agents",
            "logs/integrations", 
            "data/cache",
            "temp/scripts"
        ]
        
        for subdir in subdirs:
            subdir_path = self.base_path / subdir
            subdir_path.mkdir(parents=True, exist_ok=True)

    def initialize_database(self):
        """Initialize the system database"""
        logger.info("Initializing database...")
        
        db_path = self.base_path / "data" / "orchestrator.db"
        
        # Create database and tables
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
                instance_id TEXT PRIMARY KEY,
                agent_type TEXT NOT NULL,
                status TEXT NOT NULL,
                current_task_id TEXT,
                last_activity TEXT NOT NULL,
                total_tasks INTEGER DEFAULT 0,
                successful_tasks INTEGER DEFAULT 0
            )
        ''')
        
        # System metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
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
        
        logger.info(f"[OK] Database initialized at {db_path}")

    def verify_integrations(self):
        """Verify integration with existing AI tools"""
        logger.info("Verifying integrations...")
        
        integrations = {
            "privateGPT": self.test_privateGPT,
            "GPT4All": self.test_gpt4all,
            "Models": self.test_models_access
        }
        
        for name, test_func in integrations.items():
            try:
                test_func()
                logger.info(f"[OK] {name} integration verified")
            except Exception as e:
                logger.warning(f"[WARN] {name} integration issue: {e}")

    def test_privateGPT(self):
        """Test privateGPT integration"""
        privateGPT_path = Path("/mnt/llm/privateGPT")
        if not privateGPT_path.exists():
            raise Exception("privateGPT directory not found")
        
        if not (privateGPT_path / "privateGPT.py").exists():
            raise Exception("privateGPT.py not found")
        
        # Check for source documents
        docs_path = privateGPT_path / "source_documents"
        if docs_path.exists():
            doc_count = len(list(docs_path.glob("*")))
            logger.info(f"Found {doc_count} documents in privateGPT")

    def test_gpt4all(self):
        """Test GPT4All integration"""
        gpt4all_path = Path("/mnt/llm/gpt4all_cli")
        if not gpt4all_path.exists():
            raise Exception("GPT4All CLI directory not found")
        
        if not (gpt4all_path / "app.py").exists():
            raise Exception("GPT4All app.py not found")

    def test_models_access(self):
        """Test model directory access"""
        models_path = Path("/mnt/llm/LLM-Models")
        if not models_path.exists():
            raise Exception("Models directory not found")
        
        # Count available models
        model_files = list(models_path.rglob("*.gguf")) + list(models_path.rglob("*.safetensors"))
        logger.info(f"Found {len(model_files)} model files")

    def create_startup_scripts(self):
        """Create startup scripts for the system"""
        logger.info("Creating startup scripts...")
        
        # Linux startup script
        startup_script = f"""#!/bin/bash
# AI Agent Ecosystem Startup Script for Linux

echo "Starting AI Agent Ecosystem on Linux..."

# Change to project directory
cd {self.base_path}

# Activate virtual environment
source venv/bin/activate

# Start the orchestrator in background
echo "Starting orchestrator..."
nohup python orchestrator/orchestrator.py > logs/orchestrator.log 2>&1 &
ORCHESTRATOR_PID=$!
echo "Orchestrator started with PID: $ORCHESTRATOR_PID"

# Wait for orchestrator to initialize
sleep 5

# Start the API server in background
echo "Starting API server..."
nohup python api/main.py > logs/api.log 2>&1 &
API_PID=$!
echo "API server started with PID: $API_PID"

# Save PIDs for shutdown script
echo $ORCHESTRATOR_PID > /tmp/ai_ecosystem_orchestrator.pid
echo $API_PID > /tmp/ai_ecosystem_api.pid

echo ""
echo "=== AI Agent Ecosystem Started Successfully! ==="
echo "API available at: http://localhost:8000"
echo "Documentation at: http://localhost:8000/docs"
echo "Logs available in: {self.base_path}/logs/"
echo ""
echo "To stop the system, run: ./stop_ecosystem.sh"
echo "To monitor logs: tail -f logs/orchestrator.log logs/api.log"
"""
        
        with open(self.base_path / "start_ecosystem.sh", "w") as f:
            f.write(startup_script)
        
        # Make executable
        os.chmod(self.base_path / "start_ecosystem.sh", 0o755)
        
        # Create shutdown script
        shutdown_script = f"""#!/bin/bash
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
"""
        
        with open(self.base_path / "stop_ecosystem.sh", "w") as f:
            f.write(shutdown_script)
        
        # Make executable
        os.chmod(self.base_path / "stop_ecosystem.sh", 0o755)
        
        logger.info("[OK] Startup scripts created")

    def run_health_check(self):
        """Run system health check"""
        logger.info("Running health check...")
        
        # Check file structure
        for file_path, description in self.required_files.items():
            full_path = self.base_path / file_path
            if full_path.exists():
                logger.info(f"[OK] {description}: {full_path}")
            else:
                logger.warning(f"[WARN] {description} missing: {full_path}")
        
        # Check database
        db_path = self.base_path / "data" / "orchestrator.db"
        if db_path.exists():
            logger.info(f"[OK] Database: {db_path}")
        else:
            logger.warning(f"[WARN] Database missing: {db_path}")
        
        # Check virtual environment
        if self.python_exe.exists():
            logger.info(f"[OK] Virtual environment: {self.venv_path}")
        else:
            logger.warning(f"[WARN] Virtual environment issue: {self.venv_path}")
        
        logger.info("[OK] Health check completed")

    def print_next_steps(self):
        """Print next steps for the user"""
        print("\n" + "="*60)
        print("*** AI AGENT ECOSYSTEM SETUP COMPLETE! ***")
        print("="*60)
        print("\nNEXT STEPS:")
        print("\n1. Start the system:")
        print(f"   • SSH: ssh lightspeed@192.168.2.151")
        print(f"   • Navigate: cd {self.base_path}")
        print(f"   • Start: ./start_ecosystem.sh")
        
        print("\n2. Access the API:")
        print("   • API Endpoint: http://192.168.2.151:8000")
        print("   • Documentation: http://192.168.2.151:8000/docs")
        print("   • Health Check: http://192.168.2.151:8000/health")
        
        print("\n3. Monitor the system:")
        print("   • View logs: tail -f logs/orchestrator.log logs/api.log")
        print("   • Check status: curl http://192.168.2.151:8000/health")
        print("   • Stop system: ./stop_ecosystem.sh")
        
        print("\n4. Test the system:")
        print("   • Document analysis: curl -X POST 'http://192.168.2.151:8000/analyze-document?document_path=/mnt/llm/TEXT-VAULT/your_doc.txt&analysis_type=summary'")
        print("   • System metrics: curl http://192.168.2.151:8000/metrics")
        
        print("\nKey directories:")
        print(f"   • System logs: {self.base_path}/logs/")
        print(f"   • Configuration: {self.base_path}/config/")
        print(f"   • Database: {self.base_path}/data/")
        print(f"   • Virtual env: {self.base_path}/venv/")
        
        print("\nConfiguration:")
        print(f"   • Edit {self.base_path}/config/system.yaml for customization")
        print("   • Adjust agent settings, model preferences, etc.")
        
        print("\n" + "="*60)

def main():
    """Main setup function"""
    setup = AgentEcosystemSetup()
    setup.run_setup()

if __name__ == "__main__":
    main()