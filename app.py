import streamlit as st
from utils import initialize_agent, query_pinecone, ChatMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Tech-Docs GPT",
    page_icon="🤖",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.header("📚 About")
    st.markdown("""
    This chatbot uses advanced RAG (Retrieval Augmented Generation) techniques to provide accurate answers based on the available context related to tech Documents.
    """)

    st.header("Question Bank 📚")
    st.markdown("[Question Bank for reference questions](https://www.google.com)")


    st.header("🛠️ Features")
    st.markdown("""
    - RAG Fusion for better document retrieval
    - Multi-query generation for diverse search
    - Smart context handling and response generation
    - Debug mode for transparency
    """)
    
    st.header("🔍 Retrieval Strategies")
    st.markdown("""
    - **Simple**: Direct document retrieval
    - **Multi-query**: Generates diverse search queries
    - **RAG Fusion**: Combines results from multiple queries
    - **Decomposition**: Breaks down complex queries
    - **Step-back**: Takes a broader perspective
    - **HyDE**: Uses hypothetical documents
    - **LLM-only**: Direct LLM response
    """)
    
    st.header("📋 Instructions")
    st.markdown("""
    1. Select a retrieval strategy from the dropdown
    2. Ask questions about different TI devices like AWR2544, AM263P, MMWAVE radar sensors, etc
    3. The system will:
        - Generate multiple search queries
        - Retrieve relevant documents
        - Provide a contextual answer
    """)

# Main content
st.title("🤖 Tech-Docs GPT")
st.markdown("Ask questions about the different TI devices!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_orchestrator" not in st.session_state:
    st.session_state.rag_orchestrator = None

if "current_strategy" not in st.session_state:
    st.session_state.current_strategy = "simple"

# Get API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Dropdown for Retrieval Strategy
retrieval_strategy = st.selectbox(
    "Select Retrieval Strategy",
    ["simple", "multi_query", "rag_fusion", "decomposition", "step_back", "hyde", "llm_only"],
    index=["simple", "multi_query", "rag_fusion", "decomposition", "step_back", "hyde", "llm_only"].index(st.session_state.current_strategy)
)

# Initialize or reinitialize if strategy changes
if (st.session_state.rag_orchestrator is None or 
    st.session_state.current_strategy != retrieval_strategy):
    
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        st.error("Please set your PINECONE_API_KEY and OPENAI_API_KEY in the .env file")
    else:
        st.session_state.current_strategy = retrieval_strategy
        orchestrator = initialize_agent(OPENAI_API_KEY, PINECONE_API_KEY, retrieval_strategy)
        if orchestrator:
            st.session_state.rag_orchestrator = orchestrator
            #if not st.session_state.messages:  # Only setup data if it's first initialization
            #    setup_index_with_data(orchestrator)
            #    st.success("RAG system initialized successfully!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message.role):
        st.write(message.content)

# Chat input
if prompt := st.chat_input("Ask a question about the data..."):
    if not st.session_state.rag_orchestrator:
        st.error("Please make sure the RAG system is properly initialized")
    else:
        # Add user message to chat
        st.session_state.messages.append(ChatMessage(role="user", content=prompt))
        with st.chat_message("user"):
            st.write(prompt)

        # Get response using RAG
        try:
            with st.spinner("Thinking..."):
                response = query_pinecone(st.session_state.rag_orchestrator, prompt, retrieval_strategy)
            
            # Add assistant message to chat
            st.session_state.messages.append(ChatMessage(role="assistant", content=response))
            with st.chat_message("assistant"):
                st.write(response)
        
        except Exception as e:
            st.error(f"Error getting response: {str(e)}")
