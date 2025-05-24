# from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("MISTRALAI_API_KEY")

embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

doc1 = Document(
    page_content="""Babar Azam, a stylish right-hander with impeccable timing, Babar leads Pakistan with grace and consistency.
                    Ranked among the world's top batters, he's the backbone of the modern Pakistani batting lineup.""",
    metadata={"team": "Peshawar Zalmi"}
)

doc2 = Document(
    page_content="""Shaheen Afridi, a fiery left-arm pacer, Shaheen swings the new ball with deadly precision.
    His early breakthroughs often set the tone for Pakistan's bowling dominance.""",
    metadata= {"team": "Lahore Qalandars"}
)

doc3 = Document(
    page_content="""Rizwan, Reliable behind the stumps and dynamic with the bat, Rizwan brings unmatched energy to the team.
He's known for his work ethic and clutch performances in high-pressure games.""",
    metadata= {"team": "Multan Sultans"}
)

doc4 = Document(
    page_content=""" Shadab Khan, an agile all-rounder, Shadab dazzles with his leg-spin and sharp fielding.
He's a game-changer in white-ball cricket with his explosive lower-order batting.""",
    metadata= {"team": "Islamabad United"}
)

docs = [doc1, doc2, doc3, doc4]

# print(docs)
# In the same way we can inititate any vector store database
vector_store = Chroma(
    collection_name="cricket",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
vector_store.add_documents(documents=docs)

print(vector_store.get(include=['embeddings','documents', 'metadatas']))