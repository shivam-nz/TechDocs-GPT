from utils import *

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# --- Example 1: RAG Fusion with Debugging Enabled ---
print("\n\n" + "--- Running with RAG strategy ---")
config = {
    "llm_model": "gpt-4o-mini",
    "retrieval_strategy": "decomposition",
    "post_retrieval_processing": "semantic_re_ranking+contextual_compression",
    "prompt_strategy": "permissive_context",
    "index_name": "test-index",
    "namespace": "example-namespace",
    "top_k": 10,
    "reranker_top_n": 5,  # This is for the re-ranker, if used
    "debug": True  # <-- Enable debugging
}

orchestrator_RAG = RAGOrchestrator(config, OPENAI_API_KEY, PINECONE_API_KEY)
question = "What are the key features of the iphone?"
result_RAG = orchestrator_RAG.invoke(question)
print("\n--- FINAL OUTPUT ---")
pprint(result_RAG['answer'])
