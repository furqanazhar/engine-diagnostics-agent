import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("build_f115_embeddings")

# Get the project root directory (go up two levels from ops/fault_knowledge/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

F115_JSONL_PATH = os.path.join(PROJECT_ROOT, "data", "f115_fault_knowledge_base.jsonl")
OUTPUT_TEXT_FILE = os.path.join(SCRIPT_DIR, "f115_fault_knowledge_base_formatted.txt")
CHROMADB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "f115_faults"
DEFAULT_MODEL = "f115a"
MODEL_VALUE = "f115"  # Model number for metadata
LLM_MODEL = "gpt-4o"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@dataclass
class FaultRecord:
    """Container for a single fault entry."""

    record_id: str
    fault: str
    raw: Dict[str, str]
    formatted_text: Optional[str] = None


def validate_environment() -> bool:
    """Ensure required environment variables are available."""

    missing = [var for var in ["OPENAI_API_KEY"] if not os.getenv(var)]
    if missing:
        logger.error("Missing required env vars: %s", ", ".join(missing))
        return False
    return True


def load_faults(file_path: str) -> List[FaultRecord]:
    """Read JSONL file containing F115 fault entries (handles multi-line JSON)."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Knowledge base not found: {file_path}")

    records: List[FaultRecord] = []
    
    # Read entire file content
    with open(file_path, "r", encoding="utf-8") as fh:
        content = fh.read()
    
    # Split by closing brace followed by opening brace (separates JSON objects)
    # This handles both single-line and multi-line JSON objects
    json_objects = []
    current_obj = ""
    brace_count = 0
    
    for char in content:
        current_obj += char
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                # Complete JSON object found
                json_objects.append(current_obj.strip())
                current_obj = ""
    
    # Parse each JSON object
    for obj_str in json_objects:
        if not obj_str.strip():
            continue
        try:
            data = json.loads(obj_str)
            records.append(
                FaultRecord(
                    record_id=data.get("id", ""),
                    fault=data.get("fault", "Unknown fault"),
                    raw=data,
                )
            )
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON object: %s", e)
            continue

    logger.info("Loaded %d fault records from %s", len(records), file_path)
    return records


def llm_format_fault(client: OpenAI, record: FaultRecord) -> str:
    """Use GPT-4o to create a more readable narrative for a fault entry."""

    prompt = (
        "You are assisting a marine outboard diagnostic agent.\n"
        "Reformat the supplied JSON fault object into a readable technical brief.\n"
        "Do not drop any facts; keep terminology unchanged. Use bullet sections "
        "for Symptoms, Reported Fix, Parts Used, Preventive Notes, and Diagnostic Procedure.\n"
        "Respond with plain text only.\n\n"
        f"Fault data:\n{json.dumps(record.raw, ensure_ascii=False, indent=2)}"
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a technical documentation assistant for marine engine diagnostics."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content.strip()
    logger.debug("Formatted fault %s via LLM", record.record_id)
    return content


def write_record_to_file(record: FaultRecord, idx: int, output_path: str, is_first: bool = False) -> None:
    """Write a single formatted record to the text file immediately."""
    
    mode = "w" if is_first else "a"
    with open(output_path, mode, encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"FAULT #{idx}: {record.record_id}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Fault: {record.fault}\n")
        f.write(f"Model: {DEFAULT_MODEL}\n")
        f.write(f"Tags: {record.raw.get('tags', '')}\n")
        f.write("\n")
        f.write(record.formatted_text or "")
        f.write("\n\n")
    
    logger.info("Wrote fault %s (#%d) to text file", record.record_id, idx)


def enrich_records_with_llm_and_store(
    records: List[FaultRecord], 
    output_path: str,
    chromadb_dir: str,
    collection_name: str
) -> List[Document]:
    """Annotate each record with LLM, write immediately, then create embeddings and store in ChromaDB."""

    client = OpenAI()
    documents: List[Document] = []
    
    # Initialize ChromaDB (will be populated incrementally)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    os.makedirs(chromadb_dir, exist_ok=True)
    
    # Check if collection exists, if so delete it to start fresh
    try:
        existing_db = Chroma(
            persist_directory=chromadb_dir,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        # Delete collection if it exists
        existing_db.delete_collection()
        logger.info("Deleted existing collection: %s", collection_name)
    except Exception:
        # Collection doesn't exist, that's fine
        pass
    
    for idx, record in enumerate(records, 1):
        try:
            # Format with LLM
            logger.info("Processing record %d/%d: %s", idx, len(records), record.record_id)
            record.formatted_text = llm_format_fault(client, record)
            
            # Write immediately to text file
            write_record_to_file(record, idx, output_path, is_first=(idx == 1))
            
            # Create Document for ChromaDB
            doc = Document(
                page_content=record.formatted_text,
                metadata={
                    "id": record.record_id,
                    "fault": record.fault,
                    "model": MODEL_VALUE,
                    "tags": record.raw.get("tags", ""),
                },
            )
            documents.append(doc)
            
            # Add to ChromaDB immediately
            if idx == 1:
                # Create new collection with first document
                vectordb = Chroma.from_documents(
                    documents=[doc],
                    embedding=embeddings,
                    persist_directory=chromadb_dir,
                    collection_name=collection_name,
                )
            else:
                # Add to existing collection
                vectordb = Chroma(
                    persist_directory=chromadb_dir,
                    collection_name=collection_name,
                    embedding_function=embeddings
                )
                vectordb.add_documents([doc])
            
            logger.info("Added record %s to ChromaDB", record.record_id)
            
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to format record %s: %s", record.record_id, exc)
            # Fallback to concatenated raw fields if LLM call fails
            record.formatted_text = "\n".join(
                f"{key.title().replace('_', ' ')}: {value}"
                for key, value in record.raw.items()
                if isinstance(value, str)
            )
            # Still write the fallback content
            write_record_to_file(record, idx, output_path, is_first=(idx == 1))
            
            # Create Document with fallback content
            doc = Document(
                page_content=record.formatted_text,
                metadata={
                    "id": record.record_id,
                    "fault": record.fault,
                    "model": MODEL_VALUE,
                    "tags": record.raw.get("tags", ""),
                },
            )
            documents.append(doc)
            
            # Add to ChromaDB
            if idx == 1:
                vectordb = Chroma.from_documents(
                    documents=[doc],
                    embedding=embeddings,
                    persist_directory=chromadb_dir,
                    collection_name=collection_name,
                )
            else:
                vectordb = Chroma(
                    persist_directory=chromadb_dir,
                    collection_name=collection_name,
                    embedding_function=embeddings
                )
                vectordb.add_documents([doc])
    
    logger.info("Processed %d records and stored in ChromaDB", len(documents))
    return documents




def main() -> None:
    logger.info("***** Starting F115 Fault Formatting and Embedding Pipeline *****")

    if not validate_environment():
        logger.error("Environment validation failed. Exiting.")
        return

    try:
        # Load fault records
        records = load_faults(F115_JSONL_PATH)
        
        # Process each record: format with LLM, write to file immediately, then create embeddings and store in ChromaDB
        documents = enrich_records_with_llm_and_store(
            records, 
            OUTPUT_TEXT_FILE,
            CHROMADB_DIR,
            COLLECTION_NAME
        )
        
        logger.info("***** Pipeline completed successfully! *****")
        logger.info("Formatted data written to: %s", OUTPUT_TEXT_FILE)
        logger.info("Embeddings stored in ChromaDB collection: %s", COLLECTION_NAME)
        logger.info("ChromaDB directory: %s", CHROMADB_DIR)
        logger.info("All records have model='%s' in metadata", MODEL_VALUE)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()

