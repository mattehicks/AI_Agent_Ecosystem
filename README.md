# AI Agent Ecosystem

A comprehensive AI agent management platform with specialized research categories, document processing pipeline, and integrated LLM backends.

## 🌟 Key Features

### 🎨 Research Categories Interface
- **Unique Differentiator**: Beautiful gradient-based category containers
- **Three Specialized Workspaces**:
  - 📝 Text Generation - Content creation, summaries, reports
  - 📊 Technical Analysis - Data analysis, comparisons, insights  
  - 📄 Document Processing - Batch processing, organization, synthesis

### 📁 Document Processing Pipeline
- **Multi-format Support**: PDF, DOCX, TXT, MD, JSON, CSV, HTML
- **Batch Processing**: Extract text, summarize, analyze, categorize
- **File Management**: Upload, organize, retrieve, delete
- **Storage Analytics**: Usage statistics and optimization

### 🤖 LLM Integration
- **Ollama Backend**: Local LLM inference with privacy
- **Model Management**: Import and manage models from `/mnt/llm/LLM-Models`
- **Multiple Models**: Support for Llama, Dolphin, CodeLlama, and more
- **Real-time Generation**: Streaming and batch text generation

### 🔧 System Architecture
- **FastAPI Backend**: High-performance async API
- **Agent Orchestrator**: Task management and coordination
- **WebSocket Support**: Real-time updates and monitoring
- **Modular Design**: Pluggable components and backends

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Ollama installed
- Access to model files

### Documentation

- **[docs/setup.md](docs/setup.md)** — VantaBlack + split UI (local dashboard, remote API), networking, Git  
- **[docs/integration.md](docs/integration.md)** — Integration and architecture  
- **[docs/README.md](docs/README.md)** — Doc index  
- **[docs/archive/](docs/archive/)** — Older troubleshooting notes  

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AI_Agent_Ecosystem
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Ollama server**:
   ```bash
   ollama serve --host 0.0.0.0
   ```

5. **Start the API**:
   ```bash
   python api/main.py
   ```

6. **Access the web interface**:
   Open `http://localhost:8000` in your browser

## 📖 Usage

### Research Categories
1. Navigate to the **Research** section
2. Choose from three specialized categories:
   - **Text Generation**: For content creation and writing assistance
   - **Technical Analysis**: For data analysis and insights
   - **Document Processing**: For batch document workflows

### Document Upload
- Drag and drop files into any workspace
- Supported formats: PDF, DOCX, TXT, MD, JSON, CSV, HTML
- Files are automatically processed and analyzed

### Model Management
- Discover available models: `GET /models/discover`
- Import GGUF models: `POST /models/import`
- List Ollama models: `GET /models/ollama`

## 🏗️ Architecture

```
AI_Agent_Ecosystem/
├── api/                 # FastAPI backend
├── agents/              # AI agent implementations
├── orchestrator/        # Task coordination
├── llm_backends/        # Ollama and other LLM adapters
├── features/            # Feature modules (file_management, document_analysis, …)
├── web/                 # Dashboard (static HTML/JS/CSS)
├── monitoring/          # Status tracking, realtime monitor
├── scripts/             # start/stop helpers, fix DB, remote API starter
├── tests/               # Ad-hoc / integration test scripts
├── docs/                # Setup, integration, archived notes
├── config/              # YAML/JSON settings
└── logs/                # Runtime logs (gitignored)
```

## 🔌 API Endpoints

### Core Endpoints
- `GET /health` - System health check
- `POST /tasks` - Create new tasks
- `GET /tasks/{task_id}` - Get task status

### File Management
- `POST /files/upload` - Upload files
- `GET /files` - List uploaded files
- `POST /files/batch-process` - Batch process files

### Text Generation
- `POST /generate-text` - Generate text with LLM
- `POST /analyze` - Analyze content
- `POST /process-batch` - Process batch requests

### Model Management
- `GET /models/discover` - Discover available models
- `POST /models/import` - Import model to Ollama
- `GET /models/ollama` - List Ollama models

## 🎯 Unique Differentiators

1. **Research Category Interface**: Beautiful, intuitive workspace organization
2. **Local LLM Focus**: Privacy-first with Ollama integration
3. **Document Pipeline**: Comprehensive file processing capabilities
4. **Model Flexibility**: Easy import and management of local models
5. **Real-time Updates**: WebSocket-based live system monitoring

## 🛠️ Development

### Project Structure
- **Modular Architecture**: Each feature is a separate module
- **Async Design**: Built on FastAPI and asyncio
- **Type Safety**: Full type hints throughout
- **Error Handling**: Comprehensive error management
- **Logging**: Structured logging for debugging

### Adding New Features
1. Create feature module in `features/`
2. Add API routes in `api/`
3. Update orchestrator if needed
4. Add frontend components in `web/`

## 📊 System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB for models and data
- **GPU**: Optional but recommended for faster inference
- **Network**: For model downloads and updates

## 🔒 Security

- **Local Processing**: All LLM inference runs locally
- **File Isolation**: User files are properly sandboxed
- **Input Validation**: All inputs are validated and sanitized
- **Error Handling**: Secure error messages without data leakage

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue on GitHub  
- See **[docs/setup.md](docs/setup.md)** and the [docs/](docs/) folder  
- With the API running, open **http://localhost:8000/docs** (Swagger) or **/redoc**

---

**Built with ❤️ for the AI community**