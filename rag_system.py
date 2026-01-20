import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from sentence_transformers import CrossEncoder

load_dotenv()

# Configuration
MINUTES_DIR = "minutes"
INDEX_PATH = "faiss_index"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
reranker = CrossEncoder("BAAI/bge-reranker-large")

def extract_date_from_document(doc):
    """Extract date from document metadata/content and add to metadata."""
    try:
        # Try to parse JSON metadata at the beginning of the document
        content = doc.page_content
        # Match multi-line JSON object at the start (using .*? for non-greedy match)
        json_match = re.search(r'^\{.*?\}', content, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            metadata = json.loads(json_str)
            if 'date' in metadata:
                date_str = metadata['date']
                # Parse ISO format date (e.g., "2025-11-17T00:00:00Z")
                # Extract just the date part before 'T'
                date_only = date_str.split('T')[0]
                doc.metadata['date'] = date_only
                print(f"Extracted date {date_only} from {doc.metadata.get('source', 'unknown')}")
                return doc
        
        # Fallback: try to extract from filename if available
        source = doc.metadata.get('source', '')
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', source)
        if date_match:
            doc.metadata['date'] = date_match.group(1)
            print(f"Extracted date {date_match.group(1)} from filename {source}")
            return doc
            
    except Exception as e:
        print(f"Warning: Could not extract date from document {doc.metadata.get('source', 'unknown')}: {e}")
    
    return doc

def create_vectorstore():
    """Load documents, split into chunks, and create FAISS vector store."""
    print("Loading documents...")
    loader = DirectoryLoader(MINUTES_DIR, glob="*.txt", loader_cls=lambda path: TextLoader(path, encoding='utf-8'))
    documents = loader.load()

    print(f"Loaded {len(documents)} documents.")
    
    # Extract and add date metadata to documents
    print("Extracting date metadata...")
    documents = [extract_date_from_document(doc) for doc in documents]

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print(f"Split into {len(chunks)} chunks.")

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save the vector store
    vectorstore.save_local(INDEX_PATH)
    print(f"Vector store saved to {INDEX_PATH}")

    return vectorstore

def load_vectorstore():
    """Load existing FAISS vector store."""
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def create_llm():
    """Create OpenAI ChatGPT LLM."""
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.2
    )
    return llm

# Initialize or load vector store
if os.path.exists(INDEX_PATH):
    print("Loading existing vector store...")
    vectorstore = load_vectorstore()
else:
    print("Creating new vector store...")
    vectorstore = create_vectorstore()

llm = None

def get_llm():
    global llm
    if llm is None:
        llm = create_llm()
    return llm

def extract_date_filter_from_query(question):
    """Use LLM to extract date or date range from the user's question."""
    llm = get_llm()
    
    prompt = f"""Analyze the following question and extract any date or date range mentioned.

Question: {question}

If the question mentions:
- A specific date, return it in YYYY-MM-DD format
- A date range, return both start and end dates in YYYY-MM-DD format
- A month/year, return the start and end of that month
- A year, return the start and end of that year
- Relative dates ("last month", "this year", "recent"), convert to actual dates (today is {datetime.now().strftime('%Y-%m-%d')})
- No date mentioned, return "NONE"

Respond ONLY with a JSON object in one of these formats:
1. For a specific date: {{"type": "single", "date": "YYYY-MM-DD"}}
2. For a date range: {{"type": "range", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}}
3. For no date: {{"type": "none"}}

Examples:
- "What happened in December 2025?" → {{"type": "range", "start_date": "2025-12-01", "end_date": "2025-12-31"}}
- "Summarize the meeting on December 15, 2025" → {{"type": "single", "date": "2025-12-15"}}
- "What decisions were made?" → {{"type": "none"}}
- "Tell me about 2024" → {{"type": "range", "start_date": "2024-01-01", "end_date": "2024-12-31"}}

"""
    
    try:
        response = llm.invoke(prompt)
        result = json.loads(response.content.strip())
        return result
    except Exception as e:
        print(f"Error extracting date filter: {e}")
        return {"type": "none"}

def query_documents(question, verbose=False):
    """Query the RAG system with manual reranking and date filtering.
    
    Args:
        question: The user's question
        verbose: If True, print debug information
    
    Returns:
        tuple: (answer_text, list_of_source_documents)
    """
    
    # Extract date filter from question
    if verbose:
        print("Extracting date filter from query...")
    date_filter = extract_date_filter_from_query(question)
    if verbose:
        print(f"Date filter: {date_filter}")

    # Retrieve more documents initially (we'll filter after)
    # FAISS doesn't support metadata filtering during search
    candidate_docs = vectorstore.similarity_search(question, k=200)
    if verbose:
        print(f"Retrieved {len(candidate_docs)} candidate documents from vectorstore")
    
    # Apply date filtering if applicable
    if date_filter['type'] != 'none':
        if verbose:
            print(f"Filtering documents by date...")
        filtered_docs = []
        
        for doc in candidate_docs:
            doc_date_str = doc.metadata.get('date')
            if not doc_date_str:
                continue
                
            try:
                doc_date = datetime.strptime(doc_date_str, '%Y-%m-%d')
                
                if date_filter['type'] == 'single':
                    target_date = datetime.strptime(date_filter['date'], '%Y-%m-%d')
                    if doc_date == target_date:
                        filtered_docs.append(doc)
                        
                elif date_filter['type'] == 'range':
                    start_date = datetime.strptime(date_filter['start_date'], '%Y-%m-%d')
                    end_date = datetime.strptime(date_filter['end_date'], '%Y-%m-%d')
                    if start_date <= doc_date <= end_date:
                        filtered_docs.append(doc)
                        
            except Exception as e:
                print(f"Error parsing date for document: {e}")
                continue
        
        candidate_docs = filtered_docs
        if verbose:
            print(f"After date filtering: {len(candidate_docs)} documents")
    
    # If no documents match the date filter, inform the user
    if len(candidate_docs) == 0:
        if date_filter['type'] != 'none':
            date_info = date_filter.get('date') or f"{date_filter.get('start_date')} to {date_filter.get('end_date')}"
            return f"No documents found for the specified date: {date_info}. The available document dates range from 2012 to 2025.", []
        return "No relevant documents found for your query.", []

    # Rerank
    pairs = [(question, doc.page_content) for doc in candidate_docs]
    scores = reranker.predict(pairs)

    reranked_docs = sorted(
        zip(candidate_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    docs = [doc for doc, score in reranked_docs[:20]]

    context = "\n\n".join(
        [f"Source: {doc.metadata['source']}\n{doc.page_content}" for doc in docs]
    )

    prompt = f"""You are an AI assistant helping analyze board meeting minutes. Use the following context from board meeting documents to answer the question accurately and comprehensively.


Context from board meetings:
{context}

Question: {question}

Please provide a detailed answer based on the context above. If the context doesn't contain enough information to fully answer the question, say so and provide what information is available.
"""

    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content, docs

if __name__ == "__main__":
    # Test query
    question = "Summarize the year 2014."
    response, docs = query_documents(question, verbose=True)
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {response}")
    print(f"\nSources: {[doc.metadata['source'] for doc in docs]}")
    print(f"\nSources: {[doc.metadata['source'] for doc in docs]}")
