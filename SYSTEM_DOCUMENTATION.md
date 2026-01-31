# Board Minutes RAG System - Technical Documentation

## Overview

This is a Retrieval-Augmented Generation (RAG) system designed to answer questions about board meeting minutes using AI. The system combines vector similarity search, LLM-powered date extraction, metadata filtering, and neural reranking to provide accurate, date-aware responses.

## System Architecture

### Core Components

1. **Document Processing** ([python_files/rag_system.py](python_files/rag_system.py))
   - Loads board meeting minutes from text files
   - Extracts date metadata from JSON headers and filenames
   - Splits documents into chunks for vector storage
   - Creates and manages FAISS vector store

2. **Query Processing** ([python_files/rag_system.py](python_files/rag_system.py))
   - Extracts date filters from natural language queries using LLM
   - Retrieves candidate documents via vector similarity search
   - Filters documents by date metadata
   - Reranks results using CrossEncoder
   - Generates final answer using GPT-4o-mini

3. **User Interface** ([python_files/rag_app.py](python_files/rag_app.py))
   - Streamlit-based chat interface
   - Maintains conversation history
   - Displays source documents with citations

## Data Flow

```
User Query → Date Extraction (LLM) → Vector Search (FAISS) 
    ↓
Date Filtering → Reranking (CrossEncoder) → Context Assembly 
    ↓
Answer Generation (GPT-4o-mini) → User Response + Sources
```

## Key Features Implemented

### 1. Date Metadata Extraction

**Location**: `extract_date_from_document()` function

**Purpose**: Extracts dates from documents and adds them to metadata for filtering.

**Implementation Details**:
- Parses JSON metadata at the beginning of each document
- Looks for ISO 8601 format dates (`2025-11-17T00:00:00Z`)
- Extracts date part using string split: `date_str.split('T')[0]`
- Fallback: Extracts date from filename if JSON parsing fails
- Stores date in `YYYY-MM-DD` format in document metadata

**Example Document Structure**:
```
{
  "name": "Regular Meeting",
  "description": "",
  "unique": "DBSTPS786050",
  "date": "2025-11-17T00:00:00Z"
}

---

ST. LOUIS COUNTY LIBRARY DISTRICT
MEETING OF THE BOARD
...
```

### 2. LLM-Powered Date Filter Extraction

**Location**: `extract_date_filter_from_query()` function

**Purpose**: Analyzes user queries to identify date constraints using natural language understanding.

**Capabilities**:
- **Specific dates**: "December 15, 2025" → `{"type": "single", "date": "2025-12-15"}`
- **Date ranges**: "December 2025" → `{"type": "range", "start_date": "2025-12-01", "end_date": "2025-12-31"}`
- **Years**: "in 2024" → `{"type": "range", "start_date": "2024-01-01", "end_date": "2024-12-31"}`
- **Relative dates**: "last month", "recent", "this year" (converts relative to absolute dates)
- **No dates**: Generic queries return `{"type": "none"}`

**Error Handling**:
- JSON parsing failures default to `{"type": "none"}`
- Continues execution even if date extraction fails
- Logs errors for debugging

### 3. Metadata-Based Document Filtering

**Location**: `query_documents()` function (inline filtering logic)

**Why Post-Retrieval Filtering**:
- FAISS community version does not support metadata filtering during similarity search
- Solution: Retrieve broader set (k=200), then filter by date

**Filtering Logic**:
```python
# For single dates
if doc_date == target_date:
    filtered_docs.append(doc)

# For date ranges  
if start_date <= doc_date <= end_date:
    filtered_docs.append(doc)
```

**Performance Considerations**:
- Retrieves 200 documents initially to ensure sufficient coverage after filtering
- Filtering is fast (simple datetime comparisons)
- Only documents with date metadata are considered

### 4. Two-Stage Retrieval: Vector Search + Reranking

**Stage 1 - Vector Search (FAISS)**:
- Uses OpenAI embeddings for semantic similarity
- Fast but sometimes imprecise for nuanced queries
- Returns 200 candidates (or fewer after date filtering)

**Stage 2 - Neural Reranking (CrossEncoder)**:
- Model: `BAAI/bge-reranker-large`
- Scores each document's relevance to the query
- More accurate than vector similarity alone
- Returns top 5 most relevant documents

**Why This Approach**:
- Vector search is fast for initial retrieval
- Reranking improves precision for final results
- Better than either method alone

## File Structure

```
board_docs/
├── rag_system.py          # Core RAG logic and date filtering
├── rag_app.py             # Streamlit user interface
├── rebuild_index.py       # Script to rebuild vector store with date metadata
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not in repo)
├── minutes/               # Board meeting documents (*.txt)
│   ├── 2012-04-16.txt
│   ├── 2025-12-15.txt
│   └── ...
├── faiss_index/           # FAISS vector store
│   └── index.faiss
└── SYSTEM_DOCUMENTATION.md  # This file
```

## Function Reference

### Core Functions

#### `extract_date_from_document(doc)`
Extracts date from document and adds to metadata.
- **Input**: LangChain Document object
- **Output**: Modified Document with `date` metadata field
- **Side Effects**: Prints extraction status messages

#### `create_vectorstore()`
Creates new FAISS vector store from documents in `minutes/` directory.
- **Input**: None (reads from `MINUTES_DIR`)
- **Output**: FAISS vectorstore object
- **Side Effects**: Saves index to disk at `INDEX_PATH`

#### `load_vectorstore()`
Loads existing FAISS vector store from disk.
- **Input**: None
- **Output**: FAISS vectorstore object
- **Note**: Uses `allow_dangerous_deserialization=True` for pickle loading

#### `extract_date_filter_from_query(question)`
Uses LLM to extract date/date range from natural language query.
- **Input**: User's question string
- **Output**: Dictionary with `type`, `date`, `start_date`, `end_date` fields
- **LLM Used**: GPT-4o-mini with structured JSON output

#### `query_documents(question, verbose=False)`
Main query function that orchestrates the entire retrieval pipeline.
- **Input**: 
  - `question` (str): User's question
  - `verbose` (bool): Enable debug output
- **Output**: Tuple of (answer_string, list_of_documents)
- **Process**:
  1. Extract date filter from question
  2. Vector search for 200 candidates
  3. Filter by date if applicable
  4. Rerank with CrossEncoder
  5. Generate answer with GPT-4o-mini

## Changes Made During Development

### Issue 1: Date Metadata Not Extracted
**Problem**: Original code didn't extract dates from documents during indexing.

**Solution**: 
- Created `extract_date_from_document()` function
- Integrated into `create_vectorstore()` pipeline
- Handles both JSON metadata and filename extraction

### Issue 2: Multi-line JSON Not Parsed
**Problem**: Regex `r'^\{[^}]+\}'` couldn't match multi-line JSON blocks.

**Root Cause**: `[^}]` stops at first `}` character, which appears in formatting.

**Solution**: Changed regex to `r'^\{.*?\}'` with `re.DOTALL` flag for non-greedy multi-line matching.

### Issue 3: ISO Date Format Mismatch
**Problem**: Dates stored as `2025-11-17T00:00:00Z` but filtering expected `YYYY-MM-DD`.

**Solution**: 
- Used string split: `date_str.split('T')[0]` 
- Simpler and more reliable than `datetime.fromisoformat()`
- Normalizes all dates to `YYYY-MM-DD` format

### Issue 4: FAISS Doesn't Support Metadata Filtering
**Problem**: Attempted to use `filter` parameter in `similarity_search()` but FAISS community version doesn't support it.

**Solution**:
- Retrieve broader set (k=200) first
- Apply date filtering post-retrieval
- Still efficient due to fast datetime comparisons

### Issue 5: Duplicate Code
**Problem**: `filter_docs_by_date()` function was unused; logic duplicated in `query_documents()`.

**Solution**: Removed the 30-line duplicate function.

### Issue 6: Excessive Debug Output
**Problem**: Print statements cluttered production logs.

**Solution**: Added `verbose` parameter to `query_documents()`:
- Default `False` for clean production output
- Set `True` for debugging

### Issue 7: Poor Error Messages
**Problem**: Generic "no documents found" message wasn't helpful.

**Solution**: 
- Show specific date that had no results
- Mention available date range (2012-2025)
- Different messages for filtered vs unfiltered queries

## Usage

### Running the Streamlit App

```bash
streamlit run rag_app.py
```

### Direct Query (Testing)

```python
from rag_system import query_documents

# With date filtering
response, docs = query_documents("What happened in December 2025?", verbose=True)

# Without date filtering  
response, docs = query_documents("What policies were discussed?")
```

### Rebuilding Vector Store

Run this after adding new documents or to add date metadata:

```bash
python rebuild_index.py
```

This will:
1. Delete existing `faiss_index/` directory
2. Reload all documents from `minutes/`
3. Extract date metadata
4. Create new vector store with updated metadata

## Configuration

### Environment Variables (.env)

```
OPENAI_API_KEY=your_api_key_here
```

### Key Parameters

**In `rag_system.py`**:
- `MINUTES_DIR = "minutes"` - Directory containing documents
- `INDEX_PATH = "faiss_index"` - Vector store location
- `chunk_size=1000` - Characters per chunk
- `chunk_overlap=200` - Overlap between chunks
- `k=200` - Initial retrieval count
- `model="gpt-4o-mini"` - OpenAI model for answers
- `temperature=0.2` - LLM creativity (lower = more factual)

**In `query_documents()`**:
- Top 5 documents returned after reranking
- CrossEncoder model: `BAAI/bge-reranker-large`

## Example Queries

### Date-Specific Queries
- "What happened in the December 15, 2025 meeting?"
- "Summarize all meetings in 2024"
- "What was discussed in November 2025?"
- "Tell me about meetings in the first quarter of 2025"

### General Queries
- "What policies were reviewed?"
- "Who attended board meetings?"
- "What grants did the library receive?"
- "What strategic initiatives were discussed?"

## Performance Characteristics

- **Initial Load**: ~2-5 seconds (loads vector store and reranker model)
- **Query Time**: 
  - Date extraction: ~1-2 seconds (LLM call)
  - Vector search: <1 second
  - Date filtering: <0.1 seconds
  - Reranking: ~1-2 seconds
  - Answer generation: ~2-5 seconds (LLM call)
- **Total per query**: ~5-10 seconds

## Dependencies

Key libraries:
- `langchain-community` - Document loaders and vector stores
- `langchain-openai` - OpenAI embeddings and chat models
- `sentence-transformers` - CrossEncoder reranking
- `faiss-cpu` - Vector similarity search
- `streamlit` - Web interface
- `python-dotenv` - Environment variable management

## Future Enhancement Opportunities

1. **Caching**: Cache date filter extraction results for repeated queries
2. **Batch Processing**: Process multiple documents in parallel during indexing
3. **Advanced Filtering**: Add filters for meeting type, topics, attendees
4. **Hybrid Search**: Combine vector search with keyword search
5. **Query Expansion**: Automatically expand queries with related terms
6. **Conversation Memory**: Maintain context across multiple questions
7. **Export Features**: Allow users to export answers and sources

## Troubleshooting

### No Documents Found
- Check that vector store contains date metadata: Run `debug_metadata.py`
- Verify date format in queries matches extraction capabilities
- Try verbose mode: `query_documents(question, verbose=True)`

### Slow Performance
- Reduce `k` parameter in similarity_search (but may reduce accuracy)
- Consider using GPU for CrossEncoder reranking
- Cache embeddings if querying same documents repeatedly

### Date Extraction Errors
- Check LLM API key is valid
- Verify date extraction prompt is receiving today's date
- Look for JSON parsing errors in logs

## Additional Scripts

### `debug_metadata.py`
Diagnostic script to verify date metadata exists in vector store.
- Samples 20 documents
- Shows metadata for each
- Counts documents with/without date field

### `test_date_filter.py`
Tests date filtering with various query types.
- Specific date queries
- Date range queries  
- No-date queries

## Notes for Future Development

1. **Metadata Persistence**: Date metadata is stored in FAISS index. Rebuilding the index is required if documents change or metadata extraction logic is updated.

2. **FAISS Limitations**: The community version doesn't support pre-filtering by metadata. If upgrading to enterprise FAISS or switching to a different vector store (e.g., Pinecone, Weaviate), implement filtering during search for better performance.

3. **Date Extraction Accuracy**: The LLM-based date extraction is flexible but adds latency and cost. For production at scale, consider:
   - Caching date extractions
   - Using regex patterns for common date formats
   - Hybrid approach: Regex first, LLM fallback

4. **Chunk Size Trade-offs**: Current 1000-character chunks balance context and precision. Smaller chunks = more precise retrieval but less context. Larger chunks = more context but may dilute relevance scores.

5. **Error Recovery**: The system gracefully degrades when date extraction fails (defaults to no filtering). This ensures the system always returns some results rather than failing completely.

---

**Last Updated**: January 19, 2026  
**System Version**: 1.0  
**Author**: AI Assistant (GitHub Copilot)
