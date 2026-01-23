# SLCL Board Docs AI Assistant
## Description:
- Uses RAG to retrieve relevant parts of SLCL meeting minutes and uses AI to answer a query based on the documents.

## Quick Start:
- run `streamlit rag_system.py to open an interactive web UI (recommended for general use)

## Usage:
- run `python rag_system.py to run the single query at the bottom of the file (best used for testing)
- run `python rebuild_index.py to rebuild FAISS and BM25 indexes (only necessary if new documents are added)
- run `node scrapeDocs.js to fetch more recent board docs. If new documents are fetched, rebuild indexes.

## About rag_system.py
- rag_system.py is the backbone of the application. 