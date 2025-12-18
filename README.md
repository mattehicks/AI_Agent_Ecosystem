# AI Agent Ecosystem

A comprehensive AI agent system that leverages your existing LLM infrastructure to provide intelligent automation and analysis capabilities.

## Overview

The AI Agent Ecosystem is designed to work with your existing AI tools:
- **privateGPT** - Document ingestion & RAG system
- **LM Studio** - Model management & inference server  
- **GPT4All** - Local model runtime with CLI
- **AnythingLLM** - Web-based LLM interface
- **Text Generation WebUI** - Advanced generation interface

## Features

### Intelligent Agents
- **Document Analyzer**: Summarization, entity extraction, classification, sentiment analysis
- **Code Generator**: Multi-language code generation, testing, documentation, refactoring
- **Research Assistant**: Comprehensive research, fact-checking, trend analysis
- **Data Processor**: Data cleaning, transformation, analysis, visualization
- **Task Coordinator**: Complex workflow orchestration and automation

### System Capabilities
- **Unified API**: Single interface to all AI services
- **Task Queue System**: Asynchronous task processing with priorities
- **Performance Monitoring**: Real-time metrics and agent analytics
- **Caching System**: Intelligent result caching for improved performance
- **Error Handling**: Robust retry mechanisms and fallback strategies

### Privacy & Security
- **Local Processing**: All AI processing stays on your infrastructure
- **No External Calls**: No data sent to external services
- **Secure Access**: API authentication and IP restrictions

## Installation

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **OS**: Windows 10+, Linux, or macOS
- **Network**: Internet connection for initial setup

### Quick Installation
1. **Download** the latest release from GitHub
2. **Extract** the ZIP file to your desired location
3. **Run** the installer:
   ```bash
   python INSTALL.py
   ```
4. **Start** the system:
   ```bash
   # Windows
   START.bat
   
   # Cross-platform
   python START.py
   ```
5. **Access** the system at http://localhost:8000

### Manual Installation
```bash
# Clone the repository
git clone https://github.com/mattehicks/AI_Agent_Ecosystem.git
cd AI_Agent_Ecosystem

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run setup
python setup.py

# Start the system
python start_ecosystem.py
```

## Quick Start

### API Access
- **API Endpoint**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Interactive API**: http://localhost:8000/redoc

## Basic Usage

### Document Analysis
```python
import requests

# Analyze a document
response = requests.post("http://localhost:8000/analyze-document", 
    params={
        "document_path": "X:/TEXT-VAULT/my_document.txt",
        "analysis_type": "summary"
    }
)

task_id = response.json()["task_id"]

# Check task status
status = requests.get(f"http://localhost:8000/tasks/{task_id}")
print(status.json())
```

### Code Generation
```python
# Generate Python code
response = requests.post("http://localhost:8000/generate-code",
    params={
        "requirements": "Create a function to calculate fibonacci numbers",
        "language": "python"
    }
)

task_id = response.json()["task_id"]
```

## Agent Capabilities

### Document Analyzer
- Summary generation (multi-level)
- Entity extraction (people, organizations, locations, dates, amounts)
- Document classification (technical, business, legal, academic)
- Keyword extraction and sentiment analysis
- Structure analysis and document comparison

### Code Generator
- Multi-language support (Python, JavaScript, Java, C++, Go, Rust, SQL, HTML, CSS)
- Code generation, refactoring, review, debugging, optimization
- Test generation and documentation creation
- Best practices implementation with error handling and type hints

### Research Assistant
- Query-based, comparative, and comprehensive research
- Multi-source analysis and synthesis
- Integration with documents, knowledge base, and privateGPT
- Structured report generation

### Data Processor
- File format support (CSV, JSON, TXT, LOG)
- Data cleaning, transformation, aggregation, filtering, validation
- Statistical analysis and visualization recommendations

### Task Coordinator
- Complex workflow orchestration
- Sequential, parallel, conditional, pipeline, and workflow execution
- Pre-built workflows for common tasks
- Dependency management and error handling

## Configuration

The system uses YAML configuration files in the `config/` directory. Key settings include:

- **System settings**: Task limits, timeouts, cache TTL
- **Agent settings**: Instance limits, memory allocation, timeouts
- **Model preferences**: Primary and fallback model selection
- **Integration paths**: Paths to existing AI tools and resources

## Monitoring

### Health and Metrics
```python
# System health status
health = requests.get("http://localhost:8000/health")

# Performance metrics
metrics = requests.get("http://localhost:8000/metrics")
```

### Logging
- System logs: `logs/system.log`
- Agent logs: `logs/agents/`
- API logs: `logs/api.log`

## Documentation

For detailed information, see:
- **DEPLOYMENT.md** - Complete deployment guide
- **RELEASE_NOTES.md** - Feature documentation and changelog
- **API Documentation** - Available at http://localhost:8000/docs

## Architecture

The system follows a microservices architecture with:
- **Agent Orchestrator**: Manages task queues, agent pools, and result storage
- **Specialized Agents**: Document Analyzer, Code Generator, Research Assistant, Data Processor, Task Coordinator
- **REST API**: Unified interface for all operations
- **Web Interface**: Browser-based dashboard and controls

## Contributing

The system is extensible. To add new agents:
1. Create a new agent class inheriting from `BaseAgent`
2. Implement the `process_task` method
3. Add the agent type to the orchestrator
4. Update configuration files

## License

This project is designed for private use with your existing AI infrastructure.

## Support

For issues and questions:
1. Check the logs in the `logs/` directory
2. Review the configuration in `config/system.yaml`
3. Use the API documentation at `/docs`
4. Monitor system health via `/health` endpoint