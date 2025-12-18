#!/usr/bin/env python3
"""
AI Agent Ecosystem Setup Script
Initializes and configures the AI agent system
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
        self.base_path = Path("X:/AI_Agent_Ecosystem")
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
        logger.info("Starting AI Agent Ecosystem setup...")
        
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
        
        # Check required Python packages
        required_packages = [
            'asyncio', 'fastapi', 'uvicorn', 'requests', 
            'sqlite3', 'pathlib', 'json', 'logging'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.warning(f"Missing packages: {missing_packages}")
            logger.info("Installing missing packages...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "fastapi", "uvicorn", "requests", "pydantic"
                ])
            except subprocess.CalledProcessError as e:
                raise Exception(f"Failed to install packages: {e}")
        
        # Check X: drive accessibility
        if not Path("X:/").exists():
            raise Exception("X: drive not accessible. Please ensure the remote system is mounted.")
        
        # Check existing AI infrastructure
        self.check_ai_infrastructure()
        
        logger.info("[OK] Prerequisites check passed")

    def check_ai_infrastructure(self):
        """Check existing AI infrastructure"""
        logger.info("Checking existing AI infrastructure...")
        
        infrastructure_checks = {
            "privateGPT": Path("X:/privateGPT/privateGPT.py"),
            "GPT4All CLI": Path("X:/gpt4all_cli/app.py"),
            "LM Studio": Path("X:/LM Studio/LM Studio.exe"),
            "Models Directory": Path("X:/LLM-Models"),
            "Text Vault": Path("X:/TEXT-VAULT"),
            "AI Prompts": Path("X:/AIPROMPTS")
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
        privateGPT_path = Path("X:/privateGPT")
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
        gpt4all_path = Path("X:/gpt4all_cli")
        if not gpt4all_path.exists():
            raise Exception("GPT4All CLI directory not found")
        
        if not (gpt4all_path / "app.py").exists():
            raise Exception("GPT4All app.py not found")

    def test_models_access(self):
        """Test model directory access"""
        models_path = Path("X:/LLM-Models")
        if not models_path.exists():
            raise Exception("Models directory not found")
        
        # Count available models
        model_files = list(models_path.rglob("*.gguf")) + list(models_path.rglob("*.safetensors"))
        logger.info(f"Found {len(model_files)} model files")

    def create_startup_scripts(self):
        """Create startup scripts for the system"""
        logger.info("Creating startup scripts...")
        
        # Windows batch script
        batch_script = """@echo off
echo Starting AI Agent Ecosystem...

REM Start the orchestrator
cd /d "X:\\AI_Agent_Ecosystem"
start "Orchestrator" python orchestrator\\orchestrator.py

REM Wait a moment for orchestrator to start
timeout /t 5 /nobreak > nul

REM Start the API server
start "API Server" python api\\main.py

echo AI Agent Ecosystem started!
echo API available at: http://localhost:8000
echo Documentation at: http://localhost:8000/docs

pause
"""
        
        with open(self.base_path / "start_ecosystem.bat", "w", encoding='utf-8') as f:
            f.write(batch_script)
        
        # Python startup script
        python_script = """#!/usr/bin/env python3
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
        print("\\nShutting down...")
        orchestrator_proc.terminate()
        api_proc.terminate()

if __name__ == "__main__":
    asyncio.run(start_ecosystem())
"""
        
        with open(self.base_path / "start_ecosystem.py", "w", encoding='utf-8') as f:
            f.write(python_script)
        
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
        
        logger.info("[OK] Health check completed")

    def print_next_steps(self):
        """Print next steps for the user"""
        print("\n" + "="*60)
        print("*** AI AGENT ECOSYSTEM SETUP COMPLETE! ***")
        print("="*60)
        print("\nNEXT STEPS:")
        print("\n1. Start the system:")
        print(f"   • Windows: Run X:\\AI_Agent_Ecosystem\\start_ecosystem.bat")
        print(f"   • Python:  python X:\\AI_Agent_Ecosystem\\start_ecosystem.py")
        
        print("\n2. Access the API:")
        print("   • API Endpoint: http://localhost:8000")
        print("   • Documentation: http://localhost:8000/docs")
        print("   • Interactive API: http://localhost:8000/redoc")
        
        print("\n3. Test the system:")
        print("   • Create a simple document analysis task")
        print("   • Monitor agent performance via API")
        print("   • Check logs in X:\\AI_Agent_Ecosystem\\logs\\")
        
        print("\n4. Integration examples:")
        print("   • Document analysis: POST /analyze-document")
        print("   • Code generation: POST /generate-code") 
        print("   • Research tasks: POST /research")
        
        print("\nKey directories:")
        print(f"   • System logs: {self.base_path}/logs/")
        print(f"   • Configuration: {self.base_path}/config/")
        print(f"   • Database: {self.base_path}/data/")
        
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