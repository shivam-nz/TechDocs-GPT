import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "test-index"
NAMESPACE = "example-namespace"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")