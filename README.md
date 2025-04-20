# Advanced Agent Architecture / Gelişmiş Agent Mimarisi

[English](#english) | [Türkçe](#turkish)

---

<a name="english"></a>
## English

### Overview
This project implements a production-ready intelligent agent architecture with advanced capabilities. The system uses a modular approach to combine various AI technologies into a coherent, extensible framework for solving complex tasks.

### Key Features
- **Docker-based Code Execution**: Safe execution of code in isolated containers
- **Vector Database Integration**: Using Qdrant for semantic search and document retrieval
- **Centralized Logging**: MeiliSearch integration for fast, searchable logs
- **Security Directives**: Safety measures embedded in all LLM prompts
- **Large Text Processing**: Chunking strategies for handling large documents
- **Advanced Chain-of-Thought**: Hidden reflection mechanism with self-criticism
- **Multiple Tool Integration**: OCR, web search, knowledge graphs, and more
- **FastAPI REST Interface**: Expose agent functionality through an API
- **Model Context Protocol (MCP)**: Support for MCP for enhanced model interactions

### Architecture
The system is built on a modular architecture with the following components:

1. **Agent**: Core orchestrator that manages the entire workflow
2. **Memory**: Both short-term (session) and long-term storage
3. **Planner**: Creates and updates execution plans
4. **Subgoals**: Breaks down complex tasks into manageable steps
5. **Orchestrator**: Selects and executes appropriate tools
6. **Reflection**: Evaluates results and triggers replanning if needed
7. **Tools**: Specialized modules for specific tasks:
   - Code Execution (Docker)
   - Vector Search (Qdrant)
   - Document Analysis
   - Image Analysis (OCR with Tesseract)
   - Knowledge Graph (Neo4j)
   - RAG Retrieval
   - Web Search (Google Custom Search)
8. **FastAPI Interface**: RESTful API for accessing agent functionality
9. **Monitoring**: Prometheus and Grafana for observability

### Installation

#### Prerequisites
- Python 3.8+
- Docker and Docker Compose
- Tesseract OCR
- External services: MeiliSearch, Qdrant, Neo4j

#### Using Docker (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd code-agent

# Create .env file from template
cp .env.example .env
# Edit .env file with your API keys and settings

# Start all services
docker-compose up -d
```

#### Manual Installation
```bash
# Clone the repository
git clone <repository-url>
cd code-agent

# Install Python dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
export GOOGLE_API_KEY="your-google-api-key"
export GOOGLE_CSE_ID="your-custom-search-id"

# Run the application
# Development mode
./run.sh

# Production mode
./run.sh --prod --workers 4
```

### API Usage

#### Standard Agent Endpoint

```bash
curl -X POST http://localhost:8000/api/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query": "I installed a new library and I am getting a TypeError. Please help me fix it."}'
```

#### Model Context Protocol (MCP) Endpoint

```bash
curl -X POST http://localhost:8000/api/agent/mcp \
  -H "Content-Type: application/json" \
  -d '{"message": "I have an issue with my OCR implementation.", "context": {"previous_solutions": ["Updated pytesseract"]}}'
```

### Python Client Example

```python
import requests
import json

def run_agent_query(query):
    response = requests.post(
        "http://localhost:8000/api/agent/run",
        json={"query": query}
    )
    return response.json()

result = run_agent_query("I'm getting an ImportError with my newly installed package.")
print(result["final_answer"])
```

### Testing
```bash
python -m unittest discover -s tests
```

### Monitoring

The system includes Prometheus and Grafana for monitoring:

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default login: admin/admin)

---

<a name="turkish"></a>
## Türkçe

### Genel Bakış
Bu proje, gelişmiş yeteneklere sahip üretime hazır bir akıllı ajan mimarisi sunar. Sistem, çeşitli yapay zeka teknolojilerini karmaşık görevleri çözmek için tutarlı ve genişletilebilir bir çerçevede bir araya getirmek için modüler bir yaklaşım kullanır.

### Temel Özellikler
- **Docker Tabanlı Kod Yürütme**: İzole edilmiş konteynerler içinde güvenli kod çalıştırma
- **Vektör Veritabanı Entegrasyonu**: Anlamsal arama ve belge erişimi için Qdrant kullanımı
- **Merkezi Loglama**: Hızlı, aranabilir loglar için MeiliSearch entegrasyonu
- **Güvenlik Direktifleri**: Tüm LLM prompt'larına gömülü güvenlik önlemleri
- **Büyük Metin İşleme**: Büyük belgeleri işlemek için chunklama stratejileri
- **Gelişmiş Chain-of-Thought**: Öz-eleştirili gizli düşünce mekanizması
- **Çoklu Araç Entegrasyonu**: OCR, web arama, bilgi grafikleri ve daha fazlası
- **FastAPI REST Arayüzü**: Agent işlevselliğini API üzerinden sunma
- **Model Context Protocol (MCP)**: Gelişmiş model etkileşimi için MCP desteği

### Mimari
Sistem, aşağıdaki bileşenleri içeren modüler bir mimari üzerine inşa edilmiştir:

1. **Agent**: Tüm iş akışını yöneten ana orkestratör
2. **Memory**: Hem kısa vadeli (oturum) hem de uzun vadeli depolama
3. **Planner**: Yürütme planları oluşturur ve günceller
4. **Subgoals**: Karmaşık görevleri yönetilebilir adımlara böler
5. **Orchestrator**: Uygun araçları seçer ve yürütür
6. **Reflection**: Sonuçları değerlendirir ve gerekirse yeniden planlama tetikler
7. **Tools**: Belirli görevler için özelleştirilmiş modüller:
   - Kod Yürütme (Docker)
   - Vektör Arama (Qdrant)
   - Belge Analizi
   - Görüntü Analizi (Tesseract ile OCR)
   - Bilgi Grafiği (Neo4j)
   - RAG Erişimi
   - Web Arama (Google Custom Search)
8. **FastAPI Arayüzü**: Agent işlevselliğine erişim için RESTful API
9. **İzleme**: Gözlemlenebilirlik için Prometheus ve Grafana

### Kurulum

#### Ön Koşullar
- Python 3.8+
- Docker ve Docker Compose
- Tesseract OCR
- Harici servisler: MeiliSearch, Qdrant, Neo4j

#### Docker Kullanarak (Önerilen)
```bash
# Depoyu klonlayın
git clone <repository-url>
cd code-agent

# Şablon dosyasından .env oluşturun
cp .env.example .env
# .env dosyasını API anahtarlarınız ve ayarlarınızla düzenleyin

# Tüm servisleri başlatın
docker-compose up -d
```

#### Manuel Kurulum
```bash
# Depoyu klonlayın
git clone <repository-url>
cd code-agent

# Python bağımlılıklarını yükleyin
pip install -r requirements.txt

# Ortam değişkenlerini ayarlayın
export OPENAI_API_KEY="your-api-key"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
export GOOGLE_API_KEY="your-google-api-key"
export GOOGLE_CSE_ID="your-custom-search-id"

# Uygulamayı çalıştırın
# Geliştirme modu
./run.sh

# Üretim modu
./run.sh --prod --workers 4
```

### API Kullanımı

#### Standart Agent Endpoint'i

```bash
curl -X POST http://localhost:8000/api/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query": "Yeni bir kütüphane yükledim ve TypeError alıyorum. Lütfen düzeltmeme yardımcı ol."}'
```

#### Model Context Protocol (MCP) Endpoint'i

```bash
curl -X POST http://localhost:8000/api/agent/mcp \
  -H "Content-Type: application/json" \
  -d '{"message": "OCR uygulamam ile ilgili bir sorunum var.", "context": {"previous_solutions": ["pytesseract güncellendi"]}}'
```

### Python İstemci Örneği

```python
import requests
import json

def agent_sorgusunu_calistir(sorgu):
    yanit = requests.post(
        "http://localhost:8000/api/agent/run",
        json={"query": sorgu}
    )
    return yanit.json()

sonuc = agent_sorgusunu_calistir("Yeni yüklediğim paketle ilgili bir ImportError alıyorum.")
print(sonuc["final_answer"])
```

### Test
```bash
python -m unittest discover -s tests
```

### İzleme

Sistem, izleme için Prometheus ve Grafana içerir:

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (varsayılan giriş: admin/admin)

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