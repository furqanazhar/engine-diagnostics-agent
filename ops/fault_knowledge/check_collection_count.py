import logging
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("check_collection_count")

# Get the project root directory (go up two levels from ops/fault_knowledge/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

CHROMADB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "f115_faults"


def validate_environment() -> bool:
    """Ensure required environment variables are available."""
    missing = [var for var in ["OPENAI_API_KEY"] if not os.getenv(var)]
    if missing:
        logger.error("Missing required env vars: %s", ", ".join(missing))
        return False
    return True


def check_collection() -> None:
    """Check the ChromaDB collection count and list all fault IDs."""
    
    if not os.path.exists(CHROMADB_DIR):
        raise FileNotFoundError(f"ChromaDB directory not found: {CHROMADB_DIR}")

    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        vectordb = Chroma(
            persist_directory=CHROMADB_DIR,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
        logger.info("Loaded ChromaDB collection: %s", COLLECTION_NAME)
        
        # Get collection count
        collection = vectordb._collection
        count = collection.count()
        
        print("\n" + "=" * 80)
        print("CHROMADB COLLECTION VALIDATION")
        print("=" * 80)
        print(f"Collection Name: {COLLECTION_NAME}")
        print(f"Total Records: {count}")
        print("=" * 80)
        
        # Get all documents to list fault IDs
        if count > 0:
            # Get all documents (using a large limit)
            all_docs = collection.get(limit=count)
            
            if all_docs and 'metadatas' in all_docs:
                fault_ids = []
                for metadata in all_docs['metadatas']:
                    if metadata and 'id' in metadata:
                        fault_ids.append(metadata['id'])
                
                fault_ids.sort()  # Sort for easier reading
                
                print(f"\nFault IDs in collection ({len(fault_ids)}):")
                print("-" * 80)
                for idx, fault_id in enumerate(fault_ids, 1):
                    print(f"{idx:2d}. {fault_id}")
                
                print("\n" + "=" * 80)
                print(f"Summary: {count} total record(s) found")
                print("=" * 80)
            else:
                print("\n⚠️  Warning: Could not retrieve metadata from collection")
        else:
            print("\n⚠️  Collection is empty!")
            
    except Exception as e:
        logger.error("Failed to check ChromaDB collection: %s", e)
        raise


def main() -> None:
    """Main function."""
    logger.info("***** Checking ChromaDB Collection Count *****")

    if not validate_environment():
        logger.error("Environment validation failed. Exiting.")
        return

    try:
        check_collection()
    except Exception as exc:
        logger.exception("Collection check failed: %s", exc)
        raise


if __name__ == "__main__":
    main()

