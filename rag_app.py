import streamlit as st
from rag_system import query_documents

st.set_page_config(page_title="Board Minutes AI Assistant", page_icon="📋", layout="centered")

st.title("SLCL Board Minutes AI Assistant")
st.markdown("Ask questions about SLCL board meeting minutes and get AI-powered answers with source citations.")

container = st.container()
    
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with container.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with container.expander("📄 View Source Documents"):
                for doc in message["sources"]:
                    st.markdown(f"**{doc.metadata['source']}**")
                    st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    st.divider()

# React to user input
if prompt := st.chat_input("Ask a question about the board meetings..."):
    # Add user message to history first
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get response from RAG system
    with container.spinner("🤔 Analyzing board meeting documents..."):
        try:
            response, docs = query_documents(prompt)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": docs
            })
        except Exception as e:
            error_msg = f"Error: {str(e)}\n\n💡 Make sure your OPENAI_API_KEY is set in the .env file"
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sources": []
            })
    
    # Rerun to display the new messages
    st.rerun()

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This AI assistant uses:
    - **OpenAI GPT-4o-mini** for intelligent responses
    - **Hybrid Search** combining:
      - **Semantic search** (OpenAI embeddings + ChromaDB)
      - **Keyword search** (BM25)
    - **Cross-encoder reranking** for best results
    - **RAG** (Retrieval Augmented Generation)
    
    Ask questions like:
    - Describe circulation trends from 2015 through 2025.
    - Any mentions of the Weber Road Branch?
    - How have new federal and state laws impacted the library?
    - What is the timeline for the Samuel C. Sachs branch reconstruction?
    """)
    
    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
