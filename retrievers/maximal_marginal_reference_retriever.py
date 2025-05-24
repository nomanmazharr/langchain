# gives us unique data for our query like if there is two rows same and we have run a query similar to them we will get only one from them other one will be different
# using faiss to get familiar with it too
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('MISTRALAI_API_KEY')
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]
embeddings = OpenAIEmbeddings()

vector_stores = FAISS.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name='langchain'
)

mmr_retriever = vector_stores.as_retriever(
    search_type="mmr",                   # <-- This enables MMR
    search_kwargs={"k": 3, "lambda_mult": 0.4}
)
query='what is langchain?'
result = mmr_retriever.invoke(query)
# print(result)
for i, res in enumerate(result):
  print("Result...", i+1)
  print("Content...", res.page_content)