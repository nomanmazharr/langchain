from langchain_huggingface import HuggingFaceEmbeddings
import os

os.environ['HF_HOME'] = 'F:/huggingface_cache'

llm = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

result = llm.embed_query('Hey! how you doing')

print(result)