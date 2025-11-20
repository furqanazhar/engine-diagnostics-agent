# Yamaha F115 Engine Diagnostic Agent

AI-powered outboard engine diagnostic system trained on every workshop manual and 50 years of real-world marine fault data, with a precision RAG architecture to deliver brand-specific, expert-level insights like a seasoned marine engineer.

## Overview

An intelligent conversational AI assistant specialized exclusively for diagnosing and troubleshooting Yamaha F115 outboard marine engines. The system combines semantic search over a comprehensive fault knowledge base with advanced language models to provide accurate, context-aware diagnostic assistance.

## Key Features

### 🧠 Intelligent Diagnosis
- **Semantic Search**: Uses vector embeddings to match user symptoms with 31 documented fault patterns
- **Dual Knowledge Base**: 
  - **Fault Knowledge Base**: 31 documented fault patterns with detailed solutions
  - **Service Manual**: Complete Yamaha F115 service manual with 536 chunks covering specifications, procedures, and technical data
- **Smart Matching**: Finds relevant faults and service manual content even when symptoms are described differently

### 💬 Conversational AI
- **Context Awareness**: Maintains conversation history (last 20 messages) for natural multi-turn interactions
- **Follow-up Understanding**: Understands references to previous topics without requiring repetition
- **Natural Language**: Conversational interface that feels like talking to an expert technician

### 🔧 Comprehensive Support
- **Symptoms Analysis**: Identifies potential causes based on engine symptoms
- **Diagnostic Procedures**: Step-by-step troubleshooting guidance
- **Repair Solutions**: Reported fixes and solutions from real-world cases
- **Parts Information**: Required parts and components for repairs
- **Preventive Maintenance**: Maintenance notes and preventive measures
- **Tool Recommendations**: Diagnostic tools and equipment needed
- **Technical Specifications**: Torque values, clearances, dimensions, and tolerances
- **Service Procedures**: Assembly/disassembly procedures from official service manual
- **Tool Part Numbers**: Special tool part numbers (e.g., YB-35956, 90890-06762)

### 🛠️ Smart Tool Usage
- **Prioritized Search**: Always checks fault knowledge base first for diagnostic questions
- **Service Manual Integration**: Automatically searches service manual for specifications, procedures, and technical data
- **Dual Tool Strategy**: Uses both fault knowledge base and service manual when appropriate
- **Engineering Fallback**: Uses engineering knowledge when specific information isn't found
- **Emergency Escalation**: Escalates to human assistance for emergencies or explicit support requests

### 🚀 API Ready
- **RESTful API**: FastAPI-based service with `/health` and `/query` endpoints
- **Structured Responses**: JSON responses with assistance requirement flags
- **Easy Integration**: Ready for frontend integration and deployment

## Technical Architecture

### Core Components

- **LLM**: GPT-4o for natural language understanding and generation
- **Vector Database**: ChromaDB with OpenAI embeddings for semantic search
- **Agent Framework**: LangChain with ReAct (Reasoning + Acting) pattern
- **API Framework**: FastAPI for REST endpoints
- **Memory System**: In-memory conversation history (last 20 messages)

### Knowledge Base

#### Fault Knowledge Base
- **31 Fault Records**: Comprehensive coverage of common Yamaha F115 issues
- **Structured Data**: Each fault includes:
  - Symptoms
  - Reported fixes
  - Parts used
  - Preventive notes
  - Diagnostic procedures
  - Tags and metadata
- **Semantic Search**: Vector embeddings enable natural language query matching

#### Service Manual Knowledge Base
- **536 Chunks**: Complete Yamaha F115 service manual processed and chunked
- **Rich Metadata**: Each chunk includes:
  - Section name
  - Content type (text, procedure, table, tool_list, safety)
  - Source information
  - Chunk indexing for reference
- **Comprehensive Coverage**: 
  - Torque specifications
  - Tool part numbers
  - Assembly/disassembly procedures
  - Technical specifications (dimensions, clearances, tolerances)
  - Maintenance procedures
  - Inspection procedures
  - Special tools and equipment
- **Semantic Search**: Vector embeddings for natural language query matching

## Project Structure

```
engine-diagnostics-agent/
├── src/
│   ├── main.py                    # FastAPI application
│   └── engine_diagnostic_agent.py # Core agent implementation
├── data/
│   ├── f115_fault_knowledge_base.jsonl  # Source fault data
│   └── ...                       # Reference documents
├── ops/
│   ├── fault_knowledge/
│   │   ├── build_f115_embeddings.py      # Fault embedding generation
│   │   ├── validate_fault_records.py     # Fault validation script
│   │   └── check_collection_count.py     # Collection verification
│   └── service_manual/
│       ├── build_service_manual_embeddings.py  # Service manual embedding generation
│       ├── validate_service_manual_search.py   # Service manual validation
│       ├── test_service_manual_queries.py      # 100 test questions
│       ├── process_tables_with_llm.py          # Table processing script
│       ├── EMBEDDING_STRATEGY.md                # Embedding strategy document
│       └── F115AET_68V_28197_ZA_C1_english_only.md  # Processed service manual
├── chroma_db/                    # ChromaDB vector database
│   ├── f115_faults collection    # Fault knowledge base
│   └── service_manual collection # Service manual knowledge base
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd engine-diagnostics-agent
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Build the knowledge base embeddings**
   
   **Fault Knowledge Base:**
   ```bash
   cd ops/fault_knowledge
   python build_f115_embeddings.py
   ```
   
   **Service Manual:**
   ```bash
   cd ops/service_manual
   python build_service_manual_embeddings.py
   ```

6. **Verify the setup**
   ```bash
   # Verify fault knowledge base
   cd ops/fault_knowledge
   python check_collection_count.py
   
   # Verify service manual (optional)
   cd ../service_manual
   python validate_service_manual_search.py
   ```

## Usage

### Starting the API Server

```bash
cd src
python main.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Service is running. Diagnostic agent: available"
}
```

#### Query Endpoint
```http
POST /query
Content-Type: application/json

{
  "query": "My engine is hard to start and surges",
  "limit": 10
}
```

**Response:**
```json
{
  "query": "My engine is hard to start and surges",
  "response": "Based on your symptoms, this could indicate a VST filter issue...",
  "is_assistance_required": false
}
```

### Example Queries

**Fault Diagnosis:**
- "My engine is hard to start and surges"
- "What causes overheating at idle?"
- "How do I diagnose a fuel pump problem?"
- "What parts do I need for that?" (in context of previous conversation)

**Service Manual Queries:**
- "What is the flywheel nut torque specification?"
- "What is the part number for the pressure tester?"
- "How do I remove the flywheel magnet assembly?"
- "What is the valve clearance specification for intake valves?"
- "What tools do I need for engine diagnostics?"

## Development

### Building Embeddings

**Fault Knowledge Base:**

To rebuild the fault knowledge base embeddings:

```bash
cd ops/fault_knowledge
python build_f115_embeddings.py
```

This script:
1. Loads fault records from JSONL file
2. Formats each record using GPT-4o for readability
3. Generates embeddings using OpenAI
4. Stores in ChromaDB collection `f115_faults` with metadata

**Service Manual:**

To build the service manual embeddings:

```bash
cd ops/service_manual
python build_service_manual_embeddings.py
```

This script:
1. Reads the English-only service manual markdown file
2. Identifies sections by markdown headers
3. Chunks content intelligently (target: ~1000 chars, overlap: ~200 chars)
4. Generates embeddings using OpenAI
5. Stores in ChromaDB collection `service_manual` with rich metadata

### Validating Records

**Fault Knowledge Base:**

To validate fault records using semantic search:

```bash
cd ops/fault_knowledge
python validate_fault_records.py
```

**Service Manual:**

To validate service manual search with 100 test questions:

```bash
cd ops/service_manual
python validate_service_manual_search.py
```

This script:
- Runs 100 test questions covering all service manual categories
- Validates similarity scores and keyword matching
- Generates detailed validation report
- Saves results to `validation_results.json`

### Checking Collections

**Fault Knowledge Base:**

To verify the fault ChromaDB collection:

```bash
cd ops/fault_knowledge
python check_collection_count.py
```

**Service Manual:**

View all 100 test questions:

```bash
cd ops/service_manual
python test_service_manual_queries.py
```

## Configuration

### Agent Configuration

The agent can be configured in `src/engine_diagnostic_agent.py`:

- `max_memory_size`: Number of messages to keep in memory (default: 20)
- `chromadb_dir`: Path to ChromaDB directory
- `faults_collection`: ChromaDB collection name for faults (default: "f115_faults")
- `service_manual_collection`: ChromaDB collection name for service manual (default: "service_manual")
- `model`: LLM model (default: "gpt-4o")
- `temperature`: LLM temperature (default: 0.3)

### API Configuration

The API server can be configured in `src/main.py`:

- `host`: Server host (default: "0.0.0.0")
- `port`: Server port (default: 8000)

## Features in Detail

### ReAct Agent Pattern

The agent uses the ReAct (Reasoning + Acting) pattern:
1. **Reason**: Analyzes the question and determines what tool(s) to use
2. **Act**: Executes the appropriate tool(s):
   - `search_faults`: For diagnostic questions about symptoms and faults
   - `search_service_manual`: For specifications, procedures, torque values, and tool part numbers
   - `get_help`: For emergencies or explicit support requests
3. **Observe**: Processes the tool results
4. **Respond**: Provides the final answer based on observations

### Short-Term Memory

- Maintains last 20 messages in conversation history
- Enables natural follow-up questions
- Provides context for ambiguous queries
- Automatically manages memory size

### Semantic Search

**Fault Knowledge Base:**
- Uses OpenAI embeddings for vector similarity search
- Finds relevant faults even with different wording
- Returns top 3 most similar fault records
- Includes similarity scores for ranking

**Service Manual:**
- Uses OpenAI embeddings for vector similarity search
- Finds relevant service manual content by semantic meaning
- Returns top 3 most similar chunks
- Includes section information and content type metadata
- Covers specifications, procedures, tools, and technical data

## Testing

### Test Questions

**Fault Knowledge Base:**
- Test questions are embedded in `validate_fault_records.py` script

**Service Manual:**
- **100 Test Questions** available in `ops/service_manual/test_service_manual_queries.py`
- Organized into 8 categories:
  - Torque Specifications (20 questions)
  - Tool Part Numbers (15 questions)
  - Assembly/Disassembly Procedures (15 questions)
  - Technical Specifications (15 questions)
  - Maintenance Procedures (10 questions)
  - Inspection Procedures (10 questions)
  - Special Tools and Equipment (10 questions)
  - Dimensions and Clearances (5 questions)

### Validation Scripts

**Fault Knowledge Base:**
- `validate_fault_records.py`: Semantic search validation with predefined test cases
- `check_collection_count.py`: Collection verification

**Service Manual:**
- `validate_service_manual_search.py`: Runs all 100 test questions and generates validation report
- `test_service_manual_queries.py`: Contains test questions and utility functions

### Running Validation

**Service Manual Validation:**
```bash
cd ops/service_manual
python validate_service_manual_search.py
```

This will:
- Test all 100 questions against the service manual collection
- Validate similarity scores and keyword matching
- Generate summary report by category
- Save detailed results to `validation_results.json`

## Troubleshooting

### Common Issues

1. **ChromaDB not found**
   - Ensure embeddings have been built:
     - Fault knowledge: `python ops/fault_knowledge/build_f115_embeddings.py`
     - Service manual: `python ops/service_manual/build_service_manual_embeddings.py`
   - Check `chroma_db` directory exists

2. **Service Manual Collection Missing**
   - Run `python ops/service_manual/build_service_manual_embeddings.py`
   - Verify collection exists: Check ChromaDB directory

3. **OpenAI API errors**
   - Verify `OPENAI_API_KEY` is set in `.env`
   - Check API key validity and quota

4. **Memory issues**
   - Reduce `max_memory_size` if needed
   - Monitor conversation history size

5. **Service Manual Search Returns No Results**
   - Verify service manual embeddings were built successfully
   - Check that `F115AET_68V_28197_ZA_C1_english_only.md` exists
   - Run validation script to identify issues

## Roadmap

### Next Steps

1. ✅ **Service Manual Tool Integration**: ✅ Complete - Service manual tool integrated with 536 chunks
2. **UI for Interaction**: Develop a user interface for seamless interaction with the diagnostic agent
3. **Multi-Engine Support**: Extend support to additional engine models
4. **Enhanced Analytics**: Add usage analytics and performance metrics
5. **Table Processing**: Process HTML tables in service manual through LLM for better formatting

## Contributing

This is a specialized diagnostic system for Yamaha F115 engines. For contributions:

1. Follow the existing code structure
2. Maintain the ReAct agent pattern
3. Update documentation for new features
4. Test thoroughly before submitting

## License

[Specify your license here]

## Contact

[Add contact information]

---

**Status**: ✅ Fully Operational - Ready for testing and deployment

*This agent is designed to assist marine engine technicians and boat owners in diagnosing Yamaha F115 outboard engine issues efficiently and accurately.*
