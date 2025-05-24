from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('MISTRALAI_API_KEY')
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

vector_stores = Chroma.from_documents(
    embedding=embeddings,
    documents=documents,
    collection_name='langchain'
)

retriever = vector_stores.as_retriever(search_kwargs={'k':2})

query='tell me about langchain'

result =retriever.invoke(query)
# print(result)

for i, res in enumerate(result):
  print("Result...", i+1)
  print("Content...", res.page_content)