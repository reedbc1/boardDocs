"""
Rebuild the ChromaDB index with date metadata.
Run this script to rebuild your vector store with date information.
"""

import os
import shutil
import sys
import time
from rag_system import create_vectorstore, INDEX_PATH, BM25_INDEX_PATH

def rebuild_index():
    """Delete existing index and create a new one with date metadata."""
    
    # Remove BM25 index if exists
    if os.path.exists(BM25_INDEX_PATH):
        print(f"Removing existing BM25 index at {BM25_INDEX_PATH}...")
        os.remove(BM25_INDEX_PATH)
        print("BM25 index removed.")
    
    # Try to remove ChromaDB directory
    if os.path.exists(INDEX_PATH):
        print(f"Removing existing ChromaDB index at {INDEX_PATH}...")
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                shutil.rmtree(INDEX_PATH)
                print("ChromaDB index removed.")
                break
            except PermissionError as e:
                if attempt < max_attempts - 1:
                    print(f"ChromaDB is locked, waiting 2 seconds... (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(2)
                else:
                    print(f"\\nError: Cannot delete {INDEX_PATH} - the database is in use.")
                    print("Please close any applications using the database and try again.")
                    print("(Close Python processes, Streamlit apps, or terminals accessing the DB)")
                    sys.exit(1)
    
    print("\\nCreating new vector store with date metadata and BM25 index...")
    vectorstore = create_vectorstore()
    print("\\nIndex rebuilt successfully!")
    print(f"ChromaDB vector store saved to {INDEX_PATH}")
    print(f"BM25 index saved to {BM25_INDEX_PATH}")

if __name__ == "__main__":
    rebuild_index()
