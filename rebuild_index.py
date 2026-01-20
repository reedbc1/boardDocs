"""
Rebuild the FAISS index with date metadata.
Run this script to rebuild your vector store with date information.
"""

import os
import shutil
from rag_system import create_vectorstore, INDEX_PATH

def rebuild_index():
    """Delete existing index and create a new one with date metadata."""
    
    if os.path.exists(INDEX_PATH):
        print(f"Removing existing index at {INDEX_PATH}...")
        shutil.rmtree(INDEX_PATH)
        print("Existing index removed.")
    
    print("\nCreating new vector store with date metadata...")
    vectorstore = create_vectorstore()
    print("\nIndex rebuilt successfully!")
    print(f"Vector store saved to {INDEX_PATH}")

if __name__ == "__main__":
    rebuild_index()
