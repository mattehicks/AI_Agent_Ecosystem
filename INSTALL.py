#!/usr/bin/env python3
"""
AI Agent Ecosystem - One-Click Installer
=========================================

This script automatically sets up the AI Agent Ecosystem with all dependencies
and configuration needed to run the system.

Usage:
    python INSTALL.py

Requirements:
    - Python 3.8 or higher
    - Internet connection (for dependency installation)
    - Windows or Linux operating system
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import shutil
from pathlib import Path
import json
import time

class AIAgentInstaller:
    def __init__(self):
        self.system_os = platform.system().lower()
        self.python_version = sys.version_info
        self.install_dir = Path.cwd()
        self.venv_dir = self.install_dir / "venv"
        
    def print_banner(self):
        print("=" * 70)
        print("🤖 AI Agent Ecosystem - One-Click Installer")
        print("=" * 70)
        print(f"📍 Installation Directory: {self.install_dir}")
        print(f"🐍 Python Version: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        print(f"💻 Operating System: {platform.system()} {platform.release()}")
        print("=" * 70)
        
    def check_prerequisites(self):
        print("\n🔍 Checking Prerequisites...")
        
        # Check Python version
        if self.python_version < (3, 8):
            print("❌ ERROR: Python 3.8 or higher is required!")
            print(f"   Current version: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
            print("   Please upgrade Python and try again.")
            sys.exit(1)
        print(f"✅ Python {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro} - OK")
        
        # Check pip
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "--version"], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✅ pip - OK")
        except subprocess.CalledProcessError:
            print("❌ ERROR: pip is not available!")
            print("   Please install pip and try again.")
            sys.exit(1)
            
        # Check internet connection
        try:
            urllib.request.urlopen('https://pypi.org', timeout=5)
            print("✅ Internet Connection - OK")
        except:
            print("⚠️  WARNING: Internet connection may be limited")
            print("   Some features may not work properly")
            
    def create_virtual_environment(self):
        print("\n🏗️  Creating Virtual Environment...")
        
        if self.venv_dir.exists():
            print("📁 Removing existing virtual environment...")
            shutil.rmtree(self.venv_dir)
            
        try:
            subprocess.check_call([sys.executable, "-m", "venv", str(self.venv_dir)])
            print("✅ Virtual environment created successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR: Failed to create virtual environment: {e}")
            sys.exit(1)
            
    def get_venv_python(self):
        """Get the path to the Python executable in the virtual environment"""
        if self.system_os == "windows":
            return self.venv_dir / "Scripts" / "python.exe"
        else:
            return self.venv_dir / "bin" / "python"
            
    def get_venv_pip(self):
        """Get the path to the pip executable in the virtual environment"""
        if self.system_os == "windows":
            return self.venv_dir / "Scripts" / "pip.exe"
        else:
            return self.venv_dir / "bin" / "pip"
            
    def install_dependencies(self):
        print("\n📦 Installing Dependencies...")
        
        venv_pip = self.get_venv_pip()
        
        # Upgrade pip first
        try:
            subprocess.check_call([str(venv_pip), "install", "--upgrade", "pip"])
            print("✅ pip upgraded successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  WARNING: Failed to upgrade pip: {e}")
            
        # Install requirements
        requirements_file = self.install_dir / "requirements.txt"
        if requirements_file.exists():
            print("📋 Installing from requirements.txt...")
            try:
                subprocess.check_call([
                    str(venv_pip), "install", "-r", str(requirements_file)
                ])
                print("✅ All dependencies installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ ERROR: Failed to install dependencies: {e}")
                print("   You may need to install some packages manually")
        else:
            print("⚠️  WARNING: requirements.txt not found, installing core packages...")
            core_packages = [
                "fastapi", "uvicorn[standard]", "pydantic", "requests", 
                "aiofiles", "pyyaml", "rich", "psutil"
            ]
            for package in core_packages:
                try:
                    subprocess.check_call([str(venv_pip), "install", package])
                    print(f"✅ {package} installed")
                except subprocess.CalledProcessError:
                    print(f"⚠️  Failed to install {package}")
                    
    def create_configuration(self):
        print("\n⚙️  Creating Configuration...")
        
        # Create config directory
        config_dir = self.install_dir / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Create basic system configuration
        system_config = {
            "system": {
                "max_concurrent_tasks": 5,
                "task_timeout": 300,
                "cache_ttl": 3600,
                "log_level": "INFO"
            },
            "agents": {
                "document_analyzer": {
                    "max_instances": 2,
                    "timeout": 120
                },
                "code_generator": {
                    "max_instances": 1,
                    "timeout": 180
                },
                "research_assistant": {
                    "max_instances": 2,
                    "timeout": 240
                }
            },
            "api": {
                "host": "localhost",
                "port": 8000,
                "reload": False
            }
        }
        
        import yaml
        with open(config_dir / "system.yaml", "w") as f:
            yaml.dump(system_config, f, default_flow_style=False)
        print("✅ System configuration created")
        
        # Create environment file
        env_content = f"""# AI Agent Ecosystem Environment Configuration
PYTHONPATH={self.install_dir}
AGENT_LOG_LEVEL=INFO
AGENT_CONFIG_PATH={config_dir / 'system.yaml'}
AGENT_DATA_PATH={self.install_dir / 'data'}
"""
        
        with open(self.install_dir / ".env", "w") as f:
            f.write(env_content)
        print("✅ Environment configuration created")
        
    def create_directories(self):
        print("\n📁 Creating Directory Structure...")
        
        directories = [
            "logs", "data", "temp", "cache",
            "logs/agents", "logs/api", "data/cache"
        ]
        
        for dir_name in directories:
            dir_path = self.install_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {dir_name}/")
            
    def create_startup_scripts(self):
        print("\n🚀 Creating Startup Scripts...")
        
        venv_python = self.get_venv_python()
        
        # Windows batch script
        if self.system_os == "windows":
            batch_content = f"""@echo off
echo.
echo ==========================================
echo 🤖 AI Agent Ecosystem Starting...
echo ==========================================
echo.

cd /d "{self.install_dir}"

echo 📡 Starting API Server...
start "AI Agent API" "{venv_python}" -m uvicorn api.main:app --host localhost --port 8000

echo.
echo ✅ AI Agent Ecosystem Started!
echo.
echo 📍 API Endpoint: http://localhost:8000
echo 📚 Documentation: http://localhost:8000/docs
echo 🔧 Admin Panel: http://localhost:8000/admin
echo.
echo Press any key to open the web interface...
pause > nul

start http://localhost:8000

echo.
echo To stop the system, close this window or press Ctrl+C
pause
"""
            with open(self.install_dir / "START.bat", "w") as f:
                f.write(batch_content)
            print("✅ Windows startup script created (START.bat)")
            
        # Cross-platform Python script
        python_startup = f"""#!/usr/bin/env python3
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def main():
    print("=" * 50)
    print("🤖 AI Agent Ecosystem")
    print("=" * 50)
    
    # Change to installation directory
    install_dir = Path(__file__).parent
    import os
    os.chdir(install_dir)
    
    # Start the API server
    venv_python = "{venv_python}"
    
    print("📡 Starting API Server...")
    try:
        # Start the server
        process = subprocess.Popen([
            str(venv_python), "-m", "uvicorn", 
            "api.main:app", "--host", "localhost", "--port", "8000"
        ])
        
        print("✅ API Server started!")
        print()
        print("📍 API Endpoint: http://localhost:8000")
        print("📚 Documentation: http://localhost:8000/docs")
        print("🔧 Admin Panel: http://localhost:8000/admin")
        print()
        
        # Wait a moment then open browser
        time.sleep(3)
        try:
            webbrowser.open("http://localhost:8000")
        except:
            pass
            
        print("Press Ctrl+C to stop the system...")
        
        # Keep running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\\n🛑 Shutting down...")
            process.terminate()
            process.wait()
            print("✅ System stopped")
            
    except Exception as e:
        print(f"❌ ERROR: Failed to start system: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
        
        with open(self.install_dir / "START.py", "w") as f:
            f.write(python_startup)
        print("✅ Python startup script created (START.py)")
        
        # Make executable on Unix systems
        if self.system_os != "windows":
            os.chmod(self.install_dir / "START.py", 0o755)
            
    def create_quick_start_guide(self):
        print("\n📖 Creating Quick Start Guide...")
        
        guide_content = f"""# 🤖 AI Agent Ecosystem - Quick Start Guide

## 🚀 Getting Started (You're almost done!)

### Step 1: Start the System
Choose one of these options:

**Windows Users:**
- Double-click `START.bat`
- Or run: `START.bat` from command prompt

**All Platforms:**
- Run: `python START.py`
- Or: `{self.get_venv_python()} START.py`

### Step 2: Access the Interface
The system will automatically open your browser to:
- **Main Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Interactive API**: http://localhost:8000/redoc

### Step 3: Test the System
Try these API endpoints:
- `GET /health` - Check system status
- `GET /agents` - List available agents
- `POST /tasks` - Create a new task

## 📁 Directory Structure
```
AI_Agent_Ecosystem/
├── START.bat           # Windows startup script
├── START.py           # Cross-platform startup script
├── INSTALL.py         # This installer (you just ran it!)
├── requirements.txt   # Python dependencies
├── config/           # Configuration files
├── agents/           # AI agent implementations
├── api/             # REST API server
├── web/             # Web interface
├── logs/            # System logs
├── data/            # Application data
└── venv/            # Python virtual environment
```

## 🔧 Configuration
- Edit `config/system.yaml` to customize settings
- Modify `.env` for environment variables
- Check `logs/` directory for troubleshooting

## 🆘 Troubleshooting

**System won't start?**
1. Check Python version: `python --version` (need 3.8+)
2. Try: `{self.get_venv_python()} -m pip install --upgrade pip`
3. Check logs in `logs/` directory

**Port 8000 already in use?**
1. Edit `config/system.yaml`
2. Change the port number under `api.port`
3. Restart the system

**Need help?**
1. Check the logs: `logs/system.log`
2. Visit the documentation: http://localhost:8000/docs
3. Review the README.md file

## 🎉 Success!
Your AI Agent Ecosystem is ready to use!

---
Installation completed on: {time.strftime('%Y-%m-%d %H:%M:%S')}
Installation directory: {self.install_dir}
Python version: {sys.version}
"""
        
        with open(self.install_dir / "QUICK_START.md", "w") as f:
            f.write(guide_content)
        print("✅ Quick start guide created")
        
    def run_installation(self):
        """Run the complete installation process"""
        try:
            self.print_banner()
            self.check_prerequisites()
            self.create_virtual_environment()
            self.install_dependencies()
            self.create_configuration()
            self.create_directories()
            self.create_startup_scripts()
            self.create_quick_start_guide()
            
            print("\n" + "=" * 70)
            print("🎉 INSTALLATION COMPLETE!")
            print("=" * 70)
            print()
            print("🚀 To start the AI Agent Ecosystem:")
            if self.system_os == "windows":
                print("   • Double-click START.bat")
                print("   • Or run: START.bat")
            print("   • Or run: python START.py")
            print()
            print("📖 Read QUICK_START.md for detailed instructions")
            print("🌐 System will be available at: http://localhost:8000")
            print()
            print("=" * 70)
            
        except KeyboardInterrupt:
            print("\n\n❌ Installation cancelled by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n❌ Installation failed: {e}")
            print("Please check the error message above and try again")
            sys.exit(1)

def main():
    installer = AIAgentInstaller()
    installer.run_installation()

if __name__ == "__main__":
    main()