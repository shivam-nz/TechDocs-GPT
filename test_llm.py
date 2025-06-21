from langchain_openai import ChatOpenAI
import os

OPENAI_KEY = os.getenv("OPENAI_API_KEY") 

llm = ChatOpenAI(model="gpt-4", api_key=OPENAI_KEY)

print(llm.invoke("What is 1+1?"))
