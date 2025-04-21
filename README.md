# Code Agent Architecture

## Overview
This project implements a sophisticated, production-ready intelligent agent architecture designed for complex code understanding, analysis, modification, and validation tasks. The system employs a modular design, integrating various AI techniques and external tools into an extensible framework.

## Key Features
- **Advanced LLM Integration**: Leverages Large Language Models (e.g., GPT-4, Ollama, vLLM) for core reasoning, analysis, and code generation tasks.
- **Reflection & Planning**: Incorporates `Reflector` (for query clarity assessment, keyword extraction, decomposition) and `Planner` (for dynamic step generation) agents to guide execution.
- **Vector Database Integration**: Utilizes Qdrant via `VectorStoreManager` for efficient semantic search of code snippets and storage of interaction history.
- **Docker-based Code Execution Sandbox**: Executes and validates generated code safely within isolated Docker containers using `DockerSandboxRunner`.
- **Extensible Tool Integration**: Designed to easily incorporate external tools:
    - Currently integrated (simulated or basic): Web Search, GitHub Code Search, Stack Overflow Search.
    - Example potential tools: File System interaction (Apply Fix), etc.
- **Modular Architecture**: Core logic is being refactored into focused components like `PlanExecutor` for better maintainability and testability.
- **Caching**: Redis integration for caching LLM embeddings to improve performance and reduce costs.
- **Asynchronous Operations**: Built with `asyncio` for efficient handling of I/O-bound tasks (LLM calls, tool interactions).
- **Configuration Management**: Uses a `Config` class and potentially external settings (e.g., via `python-dotenv`) for flexible setup.

## Architecture Components
- **`CodeAgent`**: The central orchestrator managing the workflow from query intake to final response.
- **LLM Providers (`OpenAIProvider`, `OllamaProvider`, etc.)**: Abstract interactions with different LLM APIs.
- **`Reflector`**: Analyzes the input query for clarity, keywords, and decomposition.
- **`Planner`**: Generates a sequence of steps (plan) to address the query based on reflection outputs.
- **`PlanExecutor`**: Executes the generated plan step-by-step, invoking necessary tools and agent methods.
- **`VectorStoreManager`**: Handles interactions with the Qdrant vector database for storing and retrieving code and interactions.
- **Search Providers (`GitHubSearchProvider`, `StackOverflowSearchProvider`)**: Interfaces for searching external code repositories (currently basic implementations).
- **Sandbox Runner (`DockerSandboxRunner`)**: Executes code securely in an isolated environment.
- **Configuration (`Config`, `settings`)**: Manages API keys, model names, and service endpoints.
- **(Potential Future Components)**: Tool Manager, Output Formatter, etc.

## Workflow
1.  **Query Input**: User provides a query.
2.  **Reflection**: `Reflector` assesses clarity, extracts keywords, and decomposes the query.
3.  **Planning**: `Planner` creates an execution plan (e.g., search code, analyze, generate fix, validate).
4.  **Execution**: `PlanExecutor` iterates through the plan:
    - Executes search steps using `VectorStoreManager` or external providers.
    - Calls `CodeAgent.analyze_code` to interpret results.
    - Calls `CodeAgent._generate_code_fix` if a fix is needed.
    - Calls `CodeAgent.sandbox_runner.run_code` to validate the fix.
    - (Simulates) calls `edit_file` tool if applying the fix.
5.  **Response Generation**: Results (analysis, code snippets, fix details, validation status) are compiled.
6.  **Post-Processing**: Interaction details are saved to the vector store.

## Installation

### Prerequisites
- Python 3.9+
- Docker Engine (if using `DockerSandboxRunner`)
- Access to a Qdrant instance
- Access to a Redis instance (for caching)
- API keys for any desired LLM providers (e.g., OpenAI, OpenRouter)

### Setup
```bash
# 1. Clone the repository
git clone <repository-url>
cd code-agent

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# 3. Install dependencies
pip install -r requirements.txt
# Ensure you have the 'docker' library installed if using the sandbox:
# pip install docker

# 4. Configure environment variables
# Create a .env file (copy from .env.example if provided)
# and add your API keys and service endpoints:
# OPENAI_API_KEY=sk-...
# QDRANT_URL=http://localhost:6333
# REDIS_HOST=localhost
# REDIS_PORT=6379
# OLLAMA_BASE_URL=http://localhost:11434 # If using Ollama
# ... other settings ...
```

## Running the Demo
Ensure your external services (Qdrant, Redis, Docker daemon) are running.

```bash
python demo.py
```
This script initializes the `CodeAgent`, stores sample code in Qdrant, and runs several test queries through the agent's workflow, logging the output.

## Testing
Tests are implemented using `pytest`. Ensure development dependencies are installed.

```bash
# Install test dependencies (if separated in requirements-dev.txt)
# pip install -r requirements-dev.txt

# Run tests
pytest
```

## Future Enhancements (Tracking in `improvements-tasks.json`)
- Implement actual tool calls for external searches and file editing (replace simulations).
- Add more comprehensive tests, especially for `PlanExecutor` and integration scenarios.
- Further refactor `CodeAgent` into more specialized components.
- Enhance error handling and reporting.
- Implement more sophisticated RAG strategies.
- Develop a proper API interface (e.g., using FastAPI).

## Project Structure

```
.
├── app/                 # FastAPI application
│   ├── __init__.py
│   ├── main.py          # Main FastAPI application
│   ├── config.py        # Configuration with Pydantic
│   ├── schemas.py       # Request/Response models
│   ├── api/             # API endpoints
│   │   └── agent.py     # Agent API routes
│   └── services/        # Business logic
│       └── code_agent.py # Agent service wrapper
├── agent/               # Core agent implementation
│   ├── __init__.py
│   ├── agent.py         # Main agent implementation
│   ├── logger.py        # Logging with MeiliSearch
│   ├── memory.py        # Short and long-term memory
│   ├── orchestrator.py  # Tool orchestration
│   ├── planner.py       # Planning and replanning
│   ├── reflection.py    # Self-reflection
│   └── subgoals.py      # Subgoal management
├── models/              # ML models
│   ├── __init__.py
│   ├── embeddings.py    # Vector embeddings
│   └── llm.py           # LLM integration
├── tools/               # Agent tools
│   ├── __init__.py
│   ├── code_executor.py # Docker code execution
│   ├── doc_analysis.py  # Document analysis
│   ├── image_analysis.py# OCR image analysis
│   ├── knowledge_graph.py # Neo4j integration
│   ├── rag_retrieval.py # RAG implementation
│   ├── vector_search.py # Qdrant vector search
│   └── web_browser.py   # Web search
├── tests/               # Unit tests
├── logs/                # Application logs
├── data/                # Persistent data
├── Dockerfile           # Container definition
├── docker-compose.yml   # Service orchestration
├── prometheus.yml       # Prometheus configuration
├── requirements.txt     # Python dependencies
├── run.sh               # Run script
└── README.md            # Documentation
```

## Features

### FastAPI Integration

The system exposes its capabilities through a RESTful API powered by FastAPI, providing:

- High-performance async endpoints
- Automatic OpenAPI documentation
- Request validation with Pydantic
- Proper error handling and logging
- Authentication (can be enabled as needed)

### Model Context Protocol (MCP)

Support for the Model Context Protocol allows for more advanced interactions with language models:

- Context persistence between requests
- Enhanced context management
- Tool integrations
- Chain-of-thought preservation

### Production Readiness

The system includes several features that make it production-ready:

- Comprehensive logging
- Performance monitoring with Prometheus/Grafana
- Graceful error handling
- Health checks for Kubernetes integration
- Rate limiting and security features
- Docker containerization

### Scalability

The architecture is designed for scalability:

- Stateless API design
- Separate data persistence layers
- Asynchronous processing
- Worker pool management
- Service-oriented architecture 