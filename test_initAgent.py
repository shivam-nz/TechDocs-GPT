from utils import *

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


#orchestrator_RAG = RAGOrchestrator(config, OPENAI_API_KEY, PINECONE_API_KEY)
orchestrator_RAG = initialize_agent(OPENAI_API_KEY, PINECONE_API_KEY)


question = "What are the key features of the iphone?"
#result_RAG = orchestrator_RAG.invoke(question)
#print("\n--- FINAL OUTPUT ---")
#pprint(result_RAG['answer'])

answer = query_pinecone(orchestrator_RAG,question)
print(answer)
