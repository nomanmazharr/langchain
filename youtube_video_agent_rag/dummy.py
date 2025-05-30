from dotenv import load_dotenv
import os
import voyageai

load_dotenv()
key = os.getenv('VOYAGE_API_KEY')
vo = voyageai.Client(api_key=key)
# This will automatically use the environment variable VOYAGE_API_KEY.
# Alternatively, you can use vo = voyageai.Client(api_key="<your secret key>")

texts = ["Sample text 1"]

result = vo.embed(texts, model="voyage-3", input_type="document")
print(result.embeddings)
# print(result.embeddings[1])