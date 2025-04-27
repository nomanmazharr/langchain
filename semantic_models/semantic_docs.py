import torch
from langchain_huggingface import HuggingFaceEmbeddings
import os
from sklearn.metrics.pairwise import cosine_similarity


os.environ['HF_HOME'] = 'C:/Users/voltic/huggingface_cache'

# Use a sentence transformer model from Hugging Face
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-mpnet-base-v2',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

docs = [
    "The Eiffel Tower is one of the most famous landmarks in Paris, France. It attracts millions of tourists every year.",
    "In 1889, the Eiffel Tower was completed as part of the 1889 World's Fair in Paris. It was initially criticized for its design.",
    "The Louvre Museum in Paris is home to thousands of works of art, including the Mona Lisa and the Venus de Milo.",
    "The Statue of Liberty, located in New York Harbor, was a gift from France to the United States in 1886.",
    "The Great Wall of China is a series of fortifications that stretch across northern China, built to protect against invasions.",
    "Tokyo, the capital of Japan, is known for its modern architecture, technology, and bustling urban life.",
    "London is home to famous landmarks such as the Big Ben, Tower Bridge, and Buckingham Palace.",
    "The Amazon rainforest is the largest tropical rainforest in the world, covering much of South America.",
    "Mount Everest, located in the Himalayas, is the highest mountain in the world, attracting climbers from across the globe.",
    "The Great Barrier Reef, located off the coast of Queensland, Australia, is the worlds largest coral reef system."
]
query = 'history about mount everest'

# Generate embeddings
doc_embeddings = embeddings.embed_documents(docs)
query_embeddings = embeddings.embed_query(query)

scores = cosine_similarity(doc_embeddings, [query_embeddings])

# Format the results to be more readable
print(f"Query: {query}\n")
print("Similarity scores for each document:")
for i, (doc, score) in enumerate(zip(docs, scores)):
    print(f"Document {i+1}: {score[0]:.4f} - {doc}")

# Find the most relevant document
# print(scores)
most_relevant_idx = scores.argmax()
print(f"\nMost relevant document: {docs[most_relevant_idx]}")
print(f"Similarity score: {scores[most_relevant_idx][0]:.4f}")