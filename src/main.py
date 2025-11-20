import logging
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from engine_diagnostic_agent import EngineDiagnosticAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Outboard Engine Diagnostic Agent")

# Initialize the diagnostic agent
# Get project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMADB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

try:
    diagnostic_agent = EngineDiagnosticAgent(
        chromadb_dir=CHROMADB_DIR,
        faults_collection="f115_faults",
        service_manual_collection="service_manual",
    )
    logger.info("✅ Engine Diagnostic Agent initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Engine Diagnostic Agent: {e}")
    diagnostic_agent = None


class QueryRequest(BaseModel):
    query: str
    limit: Optional[int] = 10


class HealthResponse(BaseModel):
    status: str
    message: str


class QueryResponse(BaseModel):
    query: str
    response: str
    is_assistance_required: bool


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    agent_status = "available" if diagnostic_agent else "unavailable"
    return HealthResponse(
        status="healthy",
        message=f"Service is running. Diagnostic agent: {agent_status}",
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query endpoint for outboard engine diagnostics"""
    if not diagnostic_agent:
        return QueryResponse(
            query=request.query,
            response="Diagnostic agent is not available. Please check server logs.",
            is_assistance_required=True,
        )

    try:
        # Process the query using the ReAct agent
        result = diagnostic_agent.process_message(request.query)

        return QueryResponse(
            query=request.query,
            response=result.get("msg", "No response generated"),
            is_assistance_required=result.get("is_assistance_required", False),
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return QueryResponse(
            query=request.query,
            response=f"Error processing your query: {str(e)}",
            is_assistance_required=True,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)