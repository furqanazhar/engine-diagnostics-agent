import logging
import os
import sys
from typing import List, Optional

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
logger = logging.getLogger("validate_fault_records")

# Get the project root directory (go up two levels from ops/fault_knowledge/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

CHROMADB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "f115_faults"
DEFAULT_TOP_K = 5


def validate_environment() -> bool:
    """Ensure required environment variables are available."""

    missing = [var for var in ["OPENAI_API_KEY"] if not os.getenv(var)]
    if missing:
        logger.error("Missing required env vars: %s", ", ".join(missing))
        return False
    return True


def load_chromadb() -> Chroma:
    """Load the ChromaDB collection."""

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
        return vectordb
    except Exception as e:
        logger.error("Failed to load ChromaDB collection: %s", e)
        raise


def semantic_search(
    vectordb: Chroma, query: str, top_k: int = DEFAULT_TOP_K, filter_dict: Optional[dict] = None
) -> List[dict]:
    """Perform semantic similarity search in ChromaDB."""

    logger.info("Searching for: '%s' (top_k=%d)", query, top_k)

    try:
        # Perform similarity search
        if filter_dict:
            results = vectordb.similarity_search_with_score(
                query, k=top_k, filter=filter_dict
            )
        else:
            results = vectordb.similarity_search_with_score(query, k=top_k)

        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score),
                }
            )

        logger.info("Found %d results", len(formatted_results))
        return formatted_results

    except Exception as e:
        logger.error("Search failed: %s", e)
        raise


def display_results(results: List[dict], query: str) -> None:
    """Display search results in a readable format."""

    if not results:
        print("\n❌ No matching records found.")
        return

    print("\n" + "=" * 80)
    print(f"SEMANTIC SEARCH RESULTS FOR: '{query}'")
    print("=" * 80)
    print(f"Found {len(results)} similar record(s)\n")

    for idx, result in enumerate(results, 1):
        metadata = result["metadata"]
        score = result["similarity_score"]
        content = result["content"]

        # Calculate similarity percentage (lower score = more similar in ChromaDB)
        # ChromaDB uses distance, so we convert to similarity
        similarity_pct = max(0, (1 - score) * 100)

        print("-" * 80)
        print(f"RESULT #{idx} (Similarity: {similarity_pct:.1f}%)")
        print("-" * 80)
        print(f"Fault ID: {metadata.get('id', 'N/A')}")
        print(f"Fault: {metadata.get('fault', 'N/A')}")
        print(f"Model: {metadata.get('model', 'N/A')}")
        print(f"Tags: {metadata.get('tags', 'N/A')}")
        print(f"Distance Score: {score:.4f}")
        print("\nContent Preview:")
        print("-" * 80)
        # Show first 500 characters of content
        preview = content[:500] + "..." if len(content) > 500 else content
        print(preview)
        print("\n")


def validate_new_record(
    vectordb: Chroma,
    symptoms: str,
    fault_description: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
) -> None:
    """Validate a new fault record by finding similar existing records."""

    # Combine symptoms and fault description for search
    query_parts = []
    if fault_description:
        query_parts.append(fault_description)
    if symptoms:
        query_parts.append(f"Symptoms: {symptoms}")

    query = " ".join(query_parts) if query_parts else symptoms

    if not query:
        print("❌ Error: Please provide at least symptoms or fault description.")
        return

    print(f"\n🔍 Validating new fault record...")
    print(f"Query: {query}\n")

    # Perform semantic search
    results = semantic_search(vectordb, query, top_k=top_k)

    # Display results
    display_results(results, query)

    # Provide validation feedback
    if results:
        top_score = results[0]["similarity_score"]
        similarity_pct = max(0, (1 - top_score) * 100)

        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        if similarity_pct > 80:
            print(
                f"⚠️  WARNING: Very similar record found ({similarity_pct:.1f}% similarity)"
            )
            print("   This fault may already exist in the knowledge base.")
            print(f"   Most similar: {results[0]['metadata'].get('id', 'N/A')}")
        elif similarity_pct > 60:
            print(
                f"ℹ️  INFO: Similar record found ({similarity_pct:.1f}% similarity)"
            )
            print("   Review the similar records to ensure this is a unique fault.")
            print(f"   Most similar: {results[0]['metadata'].get('id', 'N/A')}")
        else:
            print(f"✅ No highly similar records found (top match: {similarity_pct:.1f}%)")
            print("   This appears to be a new or unique fault.")
    else:
        print("\n✅ No similar records found. This appears to be a new fault.")


def run_test_examples(vectordb: Chroma) -> None:
    """Run predefined test examples based on different fault numbers."""

    # Test examples based on different fault scenarios from the knowledge base
    test_examples = [
        {
            "name": "Test 1: VST Filter Issue (f115_fault_01)",
            "query": "hard starting, surging, stalls after throttle, low rail pressure, WOT power loss",
        },
        {
            "name": "Test 2: Fuel Dilution (f115_fault_02)",
            "query": "rising oil level, fuel smell in oil, foamy milky oil, rough cold idle",
        },
        {
            "name": "Test 3: IAC Failure (f115_fault_03)",
            "query": "stalling at idle, idle hunting, blown 10A fuse, erratic idle RPM",
        },
        {
            "name": "Test 4: Overheating Issue (f115_fault_06)",
            "query": "weak or steaming tell-tale, overheat alarm, RPM limiting, temperature spike after high RPM",
        },
        {
            "name": "Test 5: Trim/Tilt Problem (f115_fault_12)",
            "query": "trim moves slowly, laboured sound from motor, stops under load, difference in speed up vs down",
        },
        {
            "name": "Test 6: Ignition Coil Issue (f115_fault_15)",
            "query": "misfire at mid-range or WOT, one plug consistently wet, engine shakes at idle",
        },
        {
            "name": "Test 7: Starter Relay Failure (f115_fault_18)",
            "query": "single click but no crank, intermittent crank, starter engages weakly, relay hot to touch",
        },
        {
            "name": "Test 8: Anti-Siphon Valve (f115_fault_23)",
            "query": "primer bulb collapses, engine dies after running at speed, RPM slowly drops under load",
        },
        {
            "name": "Test 9: HP Pump Weak (f115_fault_26)",
            "query": "loss of top-end power, RPM stuck below normal, surging at mid-range, fuel rail pressure drops",
        },
        {
            "name": "Test 10: Rectifier Plug Corrosion (f115_fault_30)",
            "query": "charging voltage fluctuates, intermittent low voltage warnings, rectifier plug hot, tachometer bounce",
        },
    ]

    print("\n" + "=" * 80)
    print("F115 FAULT RECORD VALIDATION - TEST EXAMPLES")
    print("=" * 80)
    print(f"Running {len(test_examples)} test examples...\n")

    for idx, test in enumerate(test_examples, 1):
        print("\n" + "=" * 80)
        print(f"{test['name']}")
        print("=" * 80)
        
        try:
            # Perform semantic search
            results = semantic_search(vectordb, test["query"], top_k=3)
            
            # Display results
            if results:
                top_result = results[0]
                metadata = top_result["metadata"]
                score = top_result["similarity_score"]
                similarity_pct = max(0, (1 - score) * 100)
                
                print(f"\n✅ Top Match: {metadata.get('id', 'N/A')} - {metadata.get('fault', 'N/A')}")
                print(f"   Similarity: {similarity_pct:.1f}% (Distance: {score:.4f})")
                
                if len(results) > 1:
                    print(f"\n   Other matches:")
                    for i, result in enumerate(results[1:], 2):
                        other_meta = result["metadata"]
                        other_score = result["similarity_score"]
                        other_sim = max(0, (1 - other_score) * 100)
                        print(f"   {i}. {other_meta.get('id', 'N/A')} - {other_meta.get('fault', 'N/A')} ({other_sim:.1f}%)")
            else:
                print("\n❌ No matches found")
                
        except Exception as e:
            logger.error("Error in test %d: %s", idx, e)
            print(f"\n❌ Error: {e}")

    print("\n" + "=" * 80)
    print("TEST EXAMPLES COMPLETED")
    print("=" * 80)


def main() -> None:
    """Main function."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Validate fault records using semantic similarity search"
    )
    parser.add_argument(
        "--query", "-q", type=str, help="Search query (symptoms, fault description, etc.)"
    )
    parser.add_argument(
        "--symptoms",
        "-s",
        type=str,
        help="Symptoms of the fault (for validation mode)",
    )
    parser.add_argument(
        "--fault",
        "-f",
        type=str,
        help="Fault description (for validation mode)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode (requires user input)",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Run predefined test examples (default behavior)",
    )

    args = parser.parse_args()

    logger.info("***** Starting Fault Record Validation Tool *****")

    if not validate_environment():
        logger.error("Environment validation failed. Exiting.")
        sys.exit(1)

    try:
        # Load ChromaDB
        vectordb = load_chromadb()

        # Get collection count
        try:
            collection = vectordb._collection
            count = collection.count()
            logger.info("ChromaDB collection contains %d records", count)
        except Exception:
            logger.warning("Could not get collection count")

        # Run in appropriate mode
        if args.interactive:
            # Interactive mode (requires user input)
            print("\n" + "=" * 80)
            print("F115 FAULT RECORD VALIDATION TOOL - INTERACTIVE MODE")
            print("=" * 80)
            print("Enter queries to find similar fault records.")
            print("Type 'quit' or 'exit' to stop.\n")
            
            while True:
                try:
                    query = input("Enter search query (symptoms, fault description, etc.): ").strip()
                    
                    if query.lower() in ["quit", "exit", "q"]:
                        print("\n👋 Goodbye!")
                        break
                    
                    if not query:
                        print("❌ Please enter a query.")
                        continue
                    
                    results = semantic_search(vectordb, query, top_k=args.top_k)
                    display_results(results, query)
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except Exception as e:
                    logger.error("Error: %s", e)
                    print(f"❌ Error: {e}")
        elif args.symptoms or args.fault:
            # Validation mode
            validate_new_record(vectordb, args.symptoms or "", args.fault, args.top_k)
        elif args.query:
            # Simple search mode
            results = semantic_search(vectordb, args.query, top_k=args.top_k)
            display_results(results, args.query)
        else:
            # Default to running test examples
            run_test_examples(vectordb)

    except Exception as exc:
        logger.exception("Validation tool failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

