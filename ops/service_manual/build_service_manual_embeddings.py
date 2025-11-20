"""
Script to process, chunk, embed, and store service manual in ChromaDB.

This script:
1. Reads the English-only markdown file
2. Chunks content intelligently by sections with overlap
3. Generates embeddings using OpenAI
4. Stores in ChromaDB with rich metadata
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAS_TEXT_SPLITTER = True
except ImportError:
    HAS_TEXT_SPLITTER = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("build_service_manual_embeddings")

# Load environment variables
load_dotenv()

# Get project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# File paths
INPUT_FILE = os.path.join(SCRIPT_DIR, "F115AET_68V_28197_ZA_C1_english_only.md")
CHROMADB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "service_manual"

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_VALUE = "f115"
SOURCE_NAME = "F115AET_68V_28197_ZA_C1"

# Chunking parameters (tokens approximate to characters with ~4 chars per token)
CHUNK_SIZE = 1000  # Target chunk size in characters (~250 tokens)
CHUNK_OVERLAP = 200  # Overlap in characters (~50 tokens)
MAX_CHUNK_SIZE = 2000  # Max chunk size in characters (~500 tokens)


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: ~4 chars per token)."""
    return len(text) // 4


def simple_text_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Simple text splitter that splits by paragraphs with overlap.
    Fallback when langchain_text_splitters is not available.
    """
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph would exceed chunk size, save current chunk
        if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap (last N characters of previous chunk)
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    
    # Add remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def identify_sections(content: str) -> List[Tuple[int, str, int]]:
    """
    Identify sections in markdown by headers.
    
    Returns:
        List of tuples: (line_number, section_name, header_level)
    """
    sections = []
    lines = content.split('\n')
    
    # Pattern to match markdown headers (# Header, ## Header, etc.)
    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
    
    for idx, line in enumerate(lines, 1):
        match = header_pattern.match(line.strip())
        if match:
            level = len(match.group(1))
            section_name = match.group(2).strip()
            sections.append((idx, section_name, level))
    
    logger.info(f"Identified {len(sections)} sections in the document")
    return sections


def extract_section_content(content: str, start_line: int, end_line: Optional[int] = None) -> str:
    """Extract content between two line numbers."""
    lines = content.split('\n')
    if end_line is None:
        end_line = len(lines)
    return '\n'.join(lines[start_line - 1:end_line - 1])


def detect_content_type(text: str) -> str:
    """Detect content type based on text patterns."""
    text_lower = text.lower()
    
    if re.search(r'^\s*\d+\.\s+', text, re.MULTILINE):
        return "procedure"
    elif re.search(r'p/n\.|part number|tool.*yb-|model.*:', text_lower):
        return "tool_list"
    elif re.search(r'<table>|specification|torque|mm\s*\(|in\)', text_lower):
        return "table"
    elif re.search(r'warning|caution|note:', text_lower, re.IGNORECASE):
        return "safety"
    else:
        return "text"


def has_tables(text: str) -> bool:
    """Check if text contains tables."""
    return bool(re.search(r'<table>|specification|torque.*nm', text, re.IGNORECASE))


def chunk_content_by_sections(
    content: str,
    sections: List[Tuple[int, str, int]]
) -> List[Dict[str, any]]:
    """
    Chunk content intelligently by sections with overlap.
    
    Returns:
        List of chunk dictionaries with content and metadata
    """
    chunks = []
    lines = content.split('\n')
    
    # Process each section
    for i, (start_line, section_name, level) in enumerate(sections):
        # Determine end line (next section or end of file)
        if i + 1 < len(sections):
            end_line = sections[i + 1][0]
        else:
            end_line = len(lines) + 1
        
        # Extract section content
        section_content = extract_section_content(content, start_line, end_line)
        
        # Skip empty sections
        if not section_content.strip():
            continue
        
        # Estimate tokens
        estimated_tokens = estimate_tokens(section_content)
        
        # If section is small enough, create single chunk
        if estimated_tokens <= 250:  # ~1000 chars
            chunks.append({
                "content": section_content,
                "section": section_name,
                "start_line": start_line,
                "end_line": end_line,
                "chunk_index": 0,
                "total_chunks": 1,
            })
        else:
            # Split large section into smaller chunks
            if HAS_TEXT_SPLITTER:
                # Use RecursiveCharacterTextSplitter for intelligent splitting
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                section_chunks = text_splitter.split_text(section_content)
            else:
                # Simple fallback: split by paragraphs with overlap
                section_chunks = simple_text_split(section_content, CHUNK_SIZE, CHUNK_OVERLAP)
            
            for chunk_idx, chunk_text in enumerate(section_chunks):
                chunks.append({
                    "content": chunk_text,
                    "section": section_name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(section_chunks),
                })
    
    logger.info(f"Created {len(chunks)} chunks from {len(sections)} sections")
    return chunks


def create_documents_from_chunks(
    chunks: List[Dict[str, any]],
    source: str
) -> List[Document]:
    """
    Convert chunks to LangChain Document objects with rich metadata.
    """
    documents = []
    
    for idx, chunk in enumerate(chunks):
        content = chunk["content"]
        content_type = detect_content_type(content)
        has_table = has_tables(content)
        
        # Create unique chunk ID
        chunk_id = f"{source}_chunk_{idx:04d}"
        
        # Create document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "chunk_id": chunk_id,
                "section": chunk["section"],
                "content_type": content_type,
                "language": "en",
                "source": source,
                "chunk_index": chunk["chunk_index"],
                "total_chunks_in_section": chunk["total_chunks"],
                "has_tables": has_table,
                "has_images": False,  # Images were removed in filtering
                "model": MODEL_VALUE,
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
            }
        )
        documents.append(doc)
    
    logger.info(f"Created {len(documents)} Document objects")
    return documents


def store_in_chromadb(
    documents: List[Document],
    chromadb_dir: str,
    collection_name: str,
    batch_size: int = 50
) -> None:
    """
    Store documents in ChromaDB incrementally in batches.
    """
    if not documents:
        logger.warning("No documents to store")
        return
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    os.makedirs(chromadb_dir, exist_ok=True)
    
    # Check if collection exists, delete if it does
    try:
        existing_db = Chroma(
            persist_directory=chromadb_dir,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        existing_db.delete_collection()
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        # Collection doesn't exist, that's fine
        pass
    
    # Process in batches
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(documents))
        batch = documents[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} documents)")
        
        if batch_idx == 0:
            # Create new collection with first batch
            vectordb = Chroma.from_documents(
                documents=batch,
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
            vectordb.add_documents(batch)
        
        logger.info(f"Stored batch {batch_idx + 1}/{total_batches} in ChromaDB")
    
    logger.info(f"Successfully stored {len(documents)} documents in ChromaDB collection: {collection_name}")


def main() -> None:
    """Main function to process service manual and create embeddings."""
    logger.info("=" * 80)
    logger.info("SERVICE MANUAL EMBEDDING PIPELINE")
    logger.info("=" * 80)
    
    # Validate environment
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return
    
    # Log text splitter availability
    if not HAS_TEXT_SPLITTER:
        logger.warning("langchain_text_splitters not available, using simple text splitter")
    
    # Read the markdown file
    logger.info(f"Reading markdown file: {INPUT_FILE}")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Read {len(content)} characters from file")
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return
    
    # Identify sections
    logger.info("Identifying sections...")
    sections = identify_sections(content)
    
    if not sections:
        logger.warning("No sections found. Creating single chunk from entire content.")
        # Create a single chunk from entire content
        chunks = [{
            "content": content,
            "section": "ENTIRE_DOCUMENT",
            "start_line": 1,
            "end_line": len(content.split('\n')),
            "chunk_index": 0,
            "total_chunks": 1,
        }]
    else:
        # Chunk content by sections
        logger.info("Chunking content by sections...")
        chunks = chunk_content_by_sections(content, sections)
    
    # Create documents with metadata
    logger.info("Creating Document objects with metadata...")
    documents = create_documents_from_chunks(chunks, SOURCE_NAME)
    
    # Store in ChromaDB
    logger.info(f"Storing {len(documents)} documents in ChromaDB...")
    store_in_chromadb(
        documents,
        CHROMADB_DIR,
        COLLECTION_NAME,
        batch_size=50
    )
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total sections identified: {len(sections)}")
    logger.info(f"Total chunks created: {len(chunks)}")
    logger.info(f"Total documents stored: {len(documents)}")
    logger.info(f"Collection name: {COLLECTION_NAME}")
    logger.info(f"ChromaDB directory: {CHROMADB_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

