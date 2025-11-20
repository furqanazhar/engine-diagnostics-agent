# Service Manual Embedding Strategy

## Overview
This document outlines the best approach for creating embeddings from a 700-page service manual. Note: English language filtering is handled separately by `filter_english_markdown.py`.

## Recommended Approach

### 1. **Chunking Strategy**

#### Recommended: **Section-Based Chunking with Overlap**

For service manuals, semantic coherence is critical. Recommended approach:

```
1. **Page-Level Chunks** (Primary)
   - Each page = 1 chunk
   - Preserves page context
   - Easy to reference back to source
   - Size: ~500-2000 tokens per page (manageable)

2. **Section-Based Chunks** (For Large Sections)
   - If a section spans multiple pages, create section chunks
   - Split by major headings (e.g., "SPECIAL TOOLS", "SAFETY WHILE WORKING")
   - Overlap: 100-200 tokens between chunks

3. **Table-Specific Chunks** (Optional Enhancement)
   - Extract tables separately with context
   - Include surrounding text (2-3 paragraphs before/after)
   - Helps with tool/part number queries
```

**Chunk Size Guidelines:**
- **Optimal**: 500-1000 tokens per chunk
- **Max**: 2000 tokens (OpenAI embedding limit is ~8000, but smaller is better)
- **Overlap**: 100-200 tokens for context preservation

### 3. **Processing Strategy**

#### Incremental Batch Processing (Recommended)

```python
# Pseudo-code structure
1. Extract all pages from PDF (or use pre-filtered English markdown)
2. Process in batches of 50-100 pages
3. Create chunks for each page/section
4. Generate embeddings in batches (OpenAI allows batch API)
5. Store incrementally in ChromaDB
6. Save progress checkpoints (resume if interrupted)
```

**Benefits:**
- Can resume if interrupted
- Memory efficient
- Progress tracking
- Error recovery

### 4. **Metadata Strategy**

Store rich metadata for better filtering and retrieval:

```python
metadata = {
    "page_number": 18,
    "section": "IDENTIFICATION",  # Extract from headers
    "content_type": "text|table|procedure|tool_list",
    "language": "en",
    "source": "F115AET_68V_28197_ZA_C1",
    "chunk_index": 0,  # If page is split into multiple chunks
    "total_chunks_in_page": 1,
    "has_tables": True,
    "has_images": True,
    "model": "f115"
}
```

### 5. **Collection Organization**

#### Option A: Separate Collection (Recommended)
- **Collection Name**: `f115_service_manual`
- **Advantages**: 
  - Clean separation from fault knowledge
  - Can query independently or together
  - Easier to manage and update
- **Usage**: Agent can search both collections

#### Option B: Same Collection with Metadata Filtering
- **Collection Name**: `f115_knowledge_base`
- **Metadata**: Add `source_type: "fault" | "service_manual"`
- **Advantages**: Single query can search both
- **Disadvantages**: Larger collection, more complex filtering

### 6. **Implementation Steps**

#### Phase 1: Extraction & Filtering (Already Complete)
1. Extract all pages from PDF (using `extract_pdf_pages.py`)
2. Filter English content (using `filter_english_markdown.py`)
3. Result: English-only markdown file ready for processing

#### Phase 2: Chunking
1. Process each page from the filtered markdown
2. Identify sections (by headers)
3. Create chunks:
   - Small pages (<1000 tokens): Single chunk
   - Large pages (>1000 tokens): Split by sections
   - Tables: Separate chunks with context
4. Add overlap between chunks

#### Phase 3: Embedding Generation
1. Batch chunks (50-100 at a time)
2. Generate embeddings using OpenAI
3. Store in ChromaDB incrementally
4. Track progress (page numbers processed)

#### Phase 4: Validation
1. Verify all pages are included
2. Test semantic search on sample queries
3. Check chunk quality and coherence
4. Validate metadata accuracy

### 7. **Performance Optimizations**

#### Batch Embedding API
- Use OpenAI's batch embedding endpoint for large volumes
- More cost-effective for 700 pages
- Faster processing

#### Parallel Processing
- Process multiple pages in parallel
- Embedding generation can be batched
- ChromaDB writes can be batched

#### Caching
- Cache extracted pages (avoid re-extraction)
- Resume from last processed page

### 8. **Cost Estimation**

For filtered English content (approximately 350 pages):
- **Pages**: ~350 pages
- **Chunks**: ~400-500 chunks (some pages split)
- **Embedding Cost**: ~$0.10-0.20 per 1M tokens
- **Estimated Cost**: $5-15 total (depending on chunk size)

### 9. **Recommended Script Structure**

```
ops/service_manual/
├── extract_pdf_pages.py              # Extract all pages
├── filter_english_markdown.py        # Filter English content (✅ Complete)
├── chunk_service_manual.py           # Create chunks
├── build_service_manual_embeddings.py  # Generate embeddings & store
└── validate_service_manual.py        # Validate collection
```

### 10. **Key Considerations**

✅ **DO:**
- Use section-based chunking for better semantic coherence
- Store rich metadata for filtering
- Process incrementally with checkpoints
- Validate chunk quality

❌ **DON'T:**
- Don't create chunks that span multiple unrelated sections
- Don't skip metadata (makes retrieval harder)
- Don't process all pages at once (memory issues)
- Don't ignore tables (they contain important info)
- Don't forget to handle edge cases (empty pages, image-only pages)

## Example Chunk Structure

```python
{
    "page_content": "SPECIAL TOOLS\n\nUsing the correct special tools...\n\nMEASURING\n1 Pressure tester\nP/N. YB-35956...",
    "metadata": {
        "page_number": 26,
        "section": "SPECIAL TOOLS",
        "content_type": "tool_list",
        "language": "en",
        "source": "F115AET_68V_28197_ZA_C1",
        "chunk_index": 0,
        "total_chunks_in_page": 1,
        "has_tables": True,
        "has_images": True,
        "model": "f115"
    }
}
```

## Next Steps

1. ✅ Complete: `filter_english_markdown.py` - English content filtering
2. Create `chunk_service_manual.py` for intelligent chunking
3. Create `build_service_manual_embeddings.py` following the pattern from `build_f115_embeddings.py`
4. Test with a small subset first
5. Scale to full filtered content

