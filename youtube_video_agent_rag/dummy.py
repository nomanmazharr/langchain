from dotenv import load_dotenv
import os
from langchain_voyageai import VoyageAIEmbeddings
# import voyageai

load_dotenv()
key = os.getenv('VOYAGE_API_KEY')
# vo = voyageai.Client(api_key=key)
embeddings = VoyageAIEmbeddings(api_key=key, model="voyage-3")
# This will automatically use the environment variable VOYAGE_API_KEY.
# Alternatively, you can use vo = voyageai.Client(api_key="<your secret key>")

texts = ["Sample text 1", "hey how you doing", 'tell me about you']

# result = embeddings.embed_query(texts)
result = embeddings.embed_documents(texts)
print(result)